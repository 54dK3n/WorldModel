"""感知源 provider 抽象。

遵循叶志阳那条原则：**契约不改，在外围加 provider。**

    MockProvider（读 JSON 场景序列）   <- 零硬件依赖
          ↓ 换
    CameraDetectorProvider（离线 JSON/JSONL 检测帧回放；实时相机留 adapter）
          ↓ 换
    真机 RGBD

换 provider 不动 WorldModel 一行代码。
"""
from __future__ import annotations

import json
import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..calibration import CameraCalibration, load_camera_calibration
from ..raw import DetectionFrame, RawDetection
from ..types import Detection, RobotPose

logger = logging.getLogger(__name__)


class PerceptionProvider(ABC):
    """一次 step 返回：(时间戳, 机器人位姿, 该帧检测列表)"""

    @abstractmethod
    def stream(self) -> Iterator[Tuple[float, RobotPose, List[Detection]]]:
        ...

    def close(self) -> None:
        pass


class CameraDetectorProvider(PerceptionProvider):
    """相机 + 检测器 provider。

    #1 检测：外部 YOLO 输出（bbox/class/confidence），先转成 RawDetection。
    #2 定位：检测框底边中点 -> 相机射线 -> 与地平面求交 -> (x, z)。
             全程不需要深度值，透明/反光物体照样能定位。
             前提：物体接地；需标定相机高度与俯仰角。

    当前交付：
      - pixel_to_ground()：像素坐标 -> 世界坐标 x/z
      - stream()：可复现的离线 JSON / JSONL 检测帧回放
    实时相机与板端 TPU 留成独立 adapter，不让基础测试依赖真实硬件。
    """

    def __init__(
        self,
        camera_height_m: float | None = None,
        pitch_rad: float | None = None,
        fx: float | None = None,
        fy: float | None = None,
        cx: float | None = None,
        cy: float | None = None,
        image_width: int | None = None,
        image_height: int | None = None,
        camera_x_m: float = 0.0,
        camera_z_m: float = 0.0,
        ground_plane_height_m: float = 0.0,
        source: str = "camera",
        calibration: CameraCalibration | None = None,
        calibration_path: str | Path | None = None,
        replay_path: str | Path | None = None,
        time_origin: float = 0.0,
    ):
        """两种用法：

        1) 显式传参（旧接口兼容）：
           CameraDetectorProvider(1.5, 0.52, 500, 500, 320, 320,
                                  image_width=640, image_height=640)
        2) 读标定文件：
           CameraDetectorProvider(calibration_path="configs/overhead_camera.example.json",
                                  replay_path="scenes/tennis_detection_replay.jsonl")
        """
        if calibration_path is not None:
            if calibration is not None:
                raise ValueError("calibration 与 calibration_path 只能传一个")
            calibration = load_camera_calibration(calibration_path)

        if calibration is not None:
            camera_height_m = calibration.camera_height_m
            pitch_rad = calibration.pitch_rad
            fx, fy, cx, cy = calibration.fx, calibration.fy, calibration.cx, calibration.cy
            image_width, image_height = calibration.image_width, calibration.image_height
            camera_x_m = calibration.camera_x_m
            camera_z_m = calibration.camera_z_m
            ground_plane_height_m = calibration.ground_plane_height_m
            source = calibration.name
        else:
            missing = [
                name for name, value in (
                    ("camera_height_m", camera_height_m),
                    ("pitch_rad", pitch_rad),
                    ("fx", fx),
                    ("fy", fy),
                    ("cx", cx),
                    ("cy", cy),
                ) if value is None
            ]
            if missing:
                raise ValueError(
                    f"缺少相机参数 {missing}；请传 calibration 或 calibration_path"
                )
            if image_width is None:
                image_width = int(round(float(cx) * 2.0)) if cx is not None else 640
            if image_height is None:
                image_height = int(round(float(cy) * 2.0)) if cy is not None else 640

        self.calibration = CameraCalibration(
            image_width=int(image_width),
            image_height=int(image_height),
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            camera_height_m=float(camera_height_m),
            pitch_rad=float(pitch_rad),
            camera_x_m=float(camera_x_m),
            camera_z_m=float(camera_z_m),
            ground_plane_height_m=float(ground_plane_height_m),
            source="test" if calibration is None else calibration.source,
            is_real_calibration=calibration.is_real_calibration if calibration is not None else False,
            name=source if calibration is None else calibration.name,
        )
        self.source = source
        self.replay_path = Path(replay_path) if replay_path is not None else None
        self.time_origin = float(time_origin)

    # ------------------------------------------------------------------ 像素 -> 地面

    def pixel_to_ground(self, u: float, v: float) -> Tuple[float, float]:
        """像素坐标 -> 世界/机器人坐标 (x, z)。

        算法：
          1. 像素 -> 相机归一化射线 (xn, yn, 1)
          2. 绕相机 x 轴旋转 pitch_rad（正方向：光轴向下俯）
          3. 射线与平面 y = ground_plane_height_m 求交
          4. 返回交点的 x/z

        坐标系：
          x 向右，y 向上，z 向前；相机中心 (camera_x_m, camera_height_m, camera_z_m)。
        异常：
          射线与平面平行、交点在相机后方、或产生非有限值时抛 ValueError。
        """
        c = self.calibration
        if not math.isfinite(float(u)) or not math.isfinite(float(v)):
            raise ValueError(f"像素坐标必须是有限数值，收到 u={u!r}, v={v!r}")

        xn = (float(u) - c.cx) / c.fx
        yn = (float(v) - c.cy) / c.fy
        sin_p = math.sin(c.pitch_rad)
        cos_p = math.cos(c.pitch_rad)

        # 相机坐标 x 向右、y 向下、z 向前；世界坐标 y 向上。
        # 绕 x 轴旋转后：
        #   y_down' = yn*cos_p + sin_p
        #   z'      = -yn*sin_p + cos_p
        # 世界 y 向上 = -y_down'
        d_x = xn
        d_y = -(yn * cos_p + sin_p)
        d_z = -yn * sin_p + cos_p

        denom = d_y
        if abs(denom) < 1e-12:
            raise ValueError(
                f"像素 ({u:.2f}, {v:.2f}) 的射线与地面平面平行（d_y={d_y:.6e}），"
                "无法求交"
            )

        t = (c.ground_plane_height_m - c.camera_height_m) / denom
        if t <= 0.0:
            raise ValueError(
                f"像素 ({u:.2f}, {v:.2f}) 的射线与地面交点在相机后方（t={t:.3e}）"
            )

        x = c.camera_x_m + t * d_x
        z = c.camera_z_m + t * d_z
        if not (math.isfinite(x) and math.isfinite(z)):
            raise ValueError(
                f"像素 ({u:.2f}, {v:.2f}) 投影结果非有限值：x={x!r}, z={z!r}"
            )
        return x, z

    # ------------------------------------------------------------------ 回放

    def stream(self) -> Iterator[Tuple[float, RobotPose, List[Detection]]]:
        """离线 JSON/JSONL 检测帧回放。

        JSON 顶层结构：
            {"time_origin": 0.0, "frames": [{...}, ...]}
        或直接是 frames 数组。
        JSONL 每行一个 frame 对象；如果第一行是
            {"time_origin": 0.0}
        则作为相对时钟原点。

        每帧字段：
            frame_id: str
            timestamp: float（相对秒，time_origin 对应的内部时钟）
            image_width / image_height: 原始图像分辨率
            robot_pose: {x, z, yaw_rad, pose_uncertainty_cm}
            detections: [{class_name, confidence, bbox: [x1, y1, x2, y2], camera_id}]
        """
        if self.replay_path is None:
            raise NotImplementedError(
                "CameraDetectorProvider.stream() 目前只支持离线 JSON/JSONL 回放；"
                "实时相机与板端 TPU 是独立 adapter，尚未接入"
            )

        frames, self.time_origin = self._load_replay_frames(
            self.replay_path, default_time_origin=self.time_origin
        )

        for raw_frame in frames:
            try:
                yield self._frame_to_output(raw_frame)
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning(
                    "跳过无法解析的检测帧：%s（%s）",
                    raw_frame.get("frame_id", "<unknown>") if isinstance(raw_frame, dict) else raw_frame,
                    exc,
                )

    @staticmethod
    def _load_replay_frames(
        path: Path, default_time_origin: float = 0.0
    ) -> Tuple[List[Dict[str, Any]], float]:
        """读取 JSON 或 JSONL 回放文件。返回 (frames, time_origin)。"""
        text = path.read_text(encoding="utf-8")

        if path.suffix.lower() == ".jsonl":
            parsed = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]
            time_origin = default_time_origin
            if parsed and "frame_id" not in parsed[0] and "time_origin" in parsed[0]:
                time_origin = float(parsed[0]["time_origin"])
                parsed = parsed[1:]
            return parsed, time_origin

        data = json.loads(text)
        if isinstance(data, list):
            return data, default_time_origin
        if isinstance(data, dict):
            time_origin = float(data.get("time_origin", default_time_origin))
            frames = data.get("frames", [])
            return frames, time_origin
        raise ValueError(f"回放文件必须是 JSON 对象/数组或 JSONL，收到 {type(data).__name__}")

    def _frame_to_output(
        self, raw_frame: Dict[str, Any]
    ) -> Tuple[float, RobotPose, List[Detection]]:
        detections_raw = [
            RawDetection(
                class_name=str(d["class_name"]),
                confidence=float(d["confidence"]),
                bbox=tuple(float(v) for v in d["bbox"]),
                frame_id=str(raw_frame["frame_id"]),
                timestamp=float(raw_frame["timestamp"]),
                camera_id=str(d.get("camera_id", "overhead")),
            )
            for d in raw_frame.get("detections", [])
        ]
        frame = DetectionFrame(
            frame_id=str(raw_frame["frame_id"]),
            timestamp=float(raw_frame["timestamp"]),
            image_width=int(raw_frame["image_width"]),
            image_height=int(raw_frame["image_height"]),
            detections=detections_raw,
        )

        ts = frame.timestamp
        p = raw_frame.get("robot_pose", {})
        pose = RobotPose(
            x=float(p.get("x", 0.0)),
            z=float(p.get("z", 0.0)),
            yaw_rad=float(p.get("yaw_rad", 0.0)),
            pose_uncertainty_cm=float(p.get("pose_uncertainty_cm", 0.0)),
        )

        detections: List[Detection] = []
        for raw in frame.to_raw_detections():
            x1, y1, x2, y2 = raw.bbox
            u = (x1 + x2) / 2.0
            v = y2
            try:
                x, z = self.pixel_to_ground(u, v)
            except ValueError as exc:
                logger.warning(
                    "frame %s 检测 %s bbox=%s 跳过：%s",
                    frame.frame_id, raw.class_name, raw.bbox, exc,
                )
                continue
            detections.append(
                Detection(
                    class_name=raw.class_name,
                    x=x,
                    z=z,
                    confidence=raw.confidence,
                    bbox=raw.bbox,
                    frame_id=frame.frame_id,
                    source=self.source,
                    timestamp=ts,
                )
            )
        return ts, pose, detections
