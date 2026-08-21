# wm_kit — World Model + 视觉判定内核骨架

嘲风项目 · 杨铮 · 2026-07-31

设计说明与讨论结论见 **[DESIGN.md](DESIGN.md)**。

## 跑起来

```bash
python3 run_demo.py                  # 端到端：mock感知 -> World Model -> 契约 -> Judge evidence
python3 run_camera_replay.py \
    --detections scenes/tennis_detection_replay.jsonl \
    --calibration configs/overhead_camera.example.json \
    --log-file logs/camera_replay.log
python3 -m pytest tests/ -q          # 107 passed
python3 tools/false_verdict_probe.py # 判定可靠性探针：16 PASS / 0 FAIL / 1 已知边界
```

零外部依赖（仅 pytest 用于测试）。外部 YOLO 检测接入与离线回放说明见
**[docs/CAMERA_REPLAY.md](docs/CAMERA_REPLAY.md)**。

## 目录

```
wm_kit/
├── DESIGN.md                     方案总结、边界、待确认清单
├── docs/CAMERA_REPLAY.md         外部检测接入/回放/标定参数说明
├── run_camera_replay.py          离线回放演示
├── configs/
│   └── overhead_camera.example.json  测试用途标定（非真实标定）
├── world_model/
│   ├── types.py                  Detection / TrackedObject / RobotPose / 状态机
│   ├── raw.py                    RawDetection / DetectionFrame 检测输入契约
│   ├── calibration.py            CameraCalibration 标定参数 + 统一可见性模型
│   ├── size_policy.py            尺寸证据策略（detector/instance/bbox/default）
│   ├── aliases.py                #5 别名表
│   ├── association.py            #3 关联：代价矩阵 + 门控 + 贪心
│   ├── decay.py                  #4 时效：统一可见性模型 + 双模衰减
│   ├── core.py                   WorldModel 主体（update/get_scene/get_object/snapshot）
│   ├── adapters.py               导出 scene_observations 契约 + YOLO bbox 适配
│   └── providers/
│       ├── base.py               provider 抽象 + CameraDetectorProvider
│       └── mock.py               读 JSON 场景序列，零硬件
├── judge/
│   ├── evidence.py               填 evidence 字段：身份锁定 + 前置条件 + reason code
│   └── providers.py              WorldModelDiff(已实现) / RewardClassifier(待填) / YoloOverlap(备选)
├── scenes/
│   ├── demo_scene.json           覆盖四类验收场景的 mock 观测序列
│   └── tennis_detection_replay.jsonl  外部YOLO检测帧回放（球/桶/漏检/遮挡/误检）
├── tools/false_verdict_probe.py  判定可靠性探针：对抗场景，查虚假判定
├── logs/                         各次运行的输出记录
└── tests/
    ├── test_world_model.py       20 条：信念维护的不变式
    ├── test_judge.py             29 条：每条对应一个曾经的虚假判定
    ├── test_camera_provider.py   24 条：相机provider/像素投影/输入契约/回放
    └── test_strict_safety.py     34 条：严格失败关闭/尺寸证据/边界/对抗不变量
```

判定内核的设计与修复记录见 **[DESIGN.md 第十节](DESIGN.md)**，
判定可靠性复核报告见 **[2026-08-11_判定可靠性review.md](2026-08-11_判定可靠性review.md)**。

## 我的任务

两块：**World Model**（维护世界状态信念，回答"现在场景里有什么、分别在哪"）和**视觉判定内核**（回答"这件事成没成"，填 `judge_service` 里的 `RewardClassifierProvider` 与 `evidence` 字段）。

## 换数据源不改核心

```
MockProvider -> 曹志伟 SO101 双摄 -> 真机 RGBD
```
实现 `PerceptionProvider.stream()` 即可，`WorldModel` 一行不动。
