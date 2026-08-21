# 真实图像检测结果接入 World Model（离线回放）

本文说明外部 YOLO 检测结果如何进入 `wm_kit.WorldModel`，当前交付到哪一步，
以及还需要哪些真实硬件参数才能把“离线录制检测结果”切换为“真实相机实时接入”。

## 1. 外部检测代码审查结论

外部参考仓库：<https://github.com/jiangyuyue111/sg2002_yolo_inference>

审查范围：`sg2002_tpu/decode.py`、`pipeline/inference.py`、
`pipeline/preprocessor.py`、`pipeline/image_source.py`、`main.py`、
`run.py`、`mock_camera.py`、`pc_tools/preprocess_images.py`。

结论：

| 项目 | 结论 |
|------|------|
| bbox 格式 | **xyxy**。`pipeline/inference.py` 的 `Detection` 直接存 `x1,y1,x2,y2`；`sg2002_tpu/decode.py` 的底层 `Detection` 存 `cx,cy,w,h`，但 `box` 属性输出 xyxy |
| bbox 尺寸 | `sg2002_tpu/decode_nms` 输出的是**模型输入尺寸（640x640）**坐标；`pipeline/inference.py` 的 `_decode_py`/`_decode_c` 也直接透传模型输入尺寸坐标，**没有回映到原始图像**。若使用 `pc_tools/preprocess_images.py` 的 letterbox，需要手动执行 `(x - pad_left) / scale` |
| class/confidence 类型 | `label: str`（或 class_id + label），`confidence: float`（阈值过滤后 0..1） |
| frame ID | 没有独立 frame ID；`main.py` 用调用方传入的 `frame_id` 参数打印，不进入 Detection |
| timestamp | 摄像头二进制协议头里有 `timestamp_ms`（u64，毫秒），但 `Detection` 本身不带 timestamp；`main.py` 用本地 `time.time()` 计算耗时 |
| 可复用 | `sg2002_xywh_to_xyxy` 的格式换算；letterbox 还原公式；摄像头帧头协议（24B header + magic + timestamp_ms）；`mock_camera.py` 的帧打包格式 |
| 只能参考 | TPU 引擎、C NMS、板端预处理等运行时不能直接复制进 `wm_kit`；检测结果进入 `wm_kit` 前必须先完成 letterbox 还原，并显式补充 frame_id 与 timestamp |

本仓库新增的适配工具（只做格式，不复制板端运行时）：

- `world_model.adapters.sg2002_xywh_to_xyxy(cx, cy, w, h)`
- `world_model.adapters.letterbox_bbox_to_original(bbox_xyxy, scale, pad_left, pad_top)`

## 2. 检测输入契约

定义在 `world_model/raw.py`：

```python
@dataclass
class RawDetection:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]   # xyxy，原始图像坐标
    frame_id: str
    timestamp: float
    camera_id: str = "overhead"

@dataclass
class DetectionFrame:
    frame_id: str
    timestamp: float
    image_width: int
    image_height: int
    detections: list[RawDetection]
```

规则：

- bbox 必须是 `(x1, y1, x2, y2)`，对应**原始图像**像素坐标系（左上角原点，u 向右，v 向下）
- 必须 `x1 < x2` 且 `y1 < y2`，整框必须在 `[0, image_width] x [0, image_height]` 内
- confidence 必须是有限值且在 `[0, 1]`
- timestamp 必须是有限值；相对时间与 Unix 时间不能混用（见第 6 节）
- class_name 保留检测器原始类名（如 `tennis_ball`、`sports ball`、`bucket`），别名映射在 `WorldModel` 关联阶段完成
- 以上任何一条不满足直接 `ValueError`，**不 clip、不重排、不静默修正**

## 3. 像素到地面坐标转换

实现在 `world_model/providers/base.py::CameraDetectorProvider.pixel_to_ground(u, v)`。

算法：

```text
u = (x1 + x2) / 2        # 检测框底边中点
v = y2

像素点
  -> 相机归一化射线 (xn, yn, 1)
  -> 绕相机 x 轴旋转 pitch_rad
  -> 与平面 y = ground_plane_height_m 求交
  -> 机器人/世界坐标 (x, z)
```

坐标系定义：

| 轴 | 方向 |
|----|------|
| x | 向右 |
| y | 向上 |
| z | 向前（WorldModel FOV 判断以 z 为前向） |
| u | 图像向右 |
| v | 图像向下 |
| pitch_rad | **正值表示相机光轴向下俯**（overhead 相机看桌面的方向） |

单位：米。

异常处理：射线与平面平行、交点在相机后方、或投影结果非有限值时，`pixel_to_ground` 抛 `ValueError`；`stream()` 逐条检测捕获并打印 warning 后跳过该条检测，不整体崩溃。

## 4. 相机标定参数

JSON 字段与 `world_model/calibration.py::CameraCalibration` 一一对应：

```json
{
  "name": "overhead_camera",
  "image_width": 640,
  "image_height": 640,
  "fx": 500.0,
  "fy": 500.0,
  "cx": 320.0,
  "cy": 320.0,
  "camera_height_m": 1.5,
  "pitch_rad": 0.5235987755982988,
  "camera_x_m": 0.0,
  "camera_z_m": 0.0,
  "ground_plane_height_m": 0.0,
  "source": "test",
  "is_real_calibration": false
}
```

示例文件：`configs/overhead_camera.example.json`。

**重要：该示例明确标为测试用途（`source: "test"`, `is_real_calibration: false`），
不是真实相机标定结果。** 真实联调前需要队友补充并替换：

- fx / fy / cx / cy
- 图像分辨率
- 相机高度
- 相机俯仰角（pitch，正方向见上）
- 相机相对机器人坐标 `camera_x_m` / `camera_z_m`
- 地面或桌面平面高度 `ground_plane_height_m`

## 5. 离线回放运行方式

```bash
python run_camera_replay.py \
  --detections scenes/tennis_detection_replay.jsonl \
  --calibration configs/overhead_camera.example.json
```

输出包含：

- 每帧 `frame_id`
- 原始 bbox / confidence
- 转换后的 x / z
- 更新后的 `obj_id` / 对象状态 / `last_seen`
- `scene_observations`

回放数据 `scenes/tennis_detection_replay.jsonl` 覆盖：

- 球连续出现
- 单帧漏检
- 球短时遮挡（两帧，对象不删除）
- 球大位移移动（保持原 obj_id）
- 桶保持静止
- 一个低置信度误检（不 CONFIRMED，随后缓慢衰减）

## 6. 时间戳语义

- JSON/JSONL 帧里的 `timestamp` 是**场景相对秒**，不是 Unix 时间。
- `time_origin` 是相对时刻 0.0 对应的 Unix 时间。
- JSON 顶层可写 `{"time_origin": 123, "frames": [...]}`。
- JSONL 第一行可写 `{"time_origin": 123}` 作为 meta 行；也可通过 `--time-origin` 传入。
- `scene_observations` 输出 ISO8601 UTC（由 `time_origin + last_seen` 计算），
  避免把相对秒直接当 1970 纪元格式化。

## 7. 当前结论与限制

- 当前验证基于**录制/合成的检测帧回放**，使用的是**测试用途标定参数**，
  不是真实图像和真实相机标定。
- 尚未完成真实相机接入：`CameraDetectorProvider.stream()` 目前只支持
  JSON/JSONL 回放；实时相机与板端 TPU 是独立 adapter，未在本轮接入。
- 尚未获得真实 fx/fy/cx/cy、相机外参、桌面高度等参数。
- 物体定位依赖“检测框底边中点在地面/桌面平面上”的假设；悬空物体会有系统误差。
- `FovConfig` 仍使用默认水平 FOV 70°/4m 经验值，真实标定完成后应替换。

## 8. 如何接入 chaofeng 的 Agent/Judge

1. 在 chaofeng 侧构造 `CameraDetectorProvider`（标定文件 + 回放文件或未来实时相机 adapter）。
2. 用 `provider.time_origin` 构造 `WorldModel(time_origin=...)`，保证 `scene_observations` 时间戳正确。
3. 对每帧 `(ts, pose, dets)` 调用 `wm.update(dets, pose, now=ts)`。
4. Agent 动作前调用 `before = wm.snapshot()`，动作后调用 `after = wm.snapshot()`。
5. 调用 `WorldModelDiffProvider().judge(req, before, after, ctx=JudgeContext(target_id=...))`。
6. 外层循环用 try/except 包住单帧 update，provider 异常不应导致 WorldModel 状态丢失。
