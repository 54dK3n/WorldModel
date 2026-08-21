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
from ..raw import RawDetection, validate_detection_for_frame
from ..size_policy import SizePolicy
from ..types import (
    COORDINATE_FRAME_WORLD,
    Detection,
    FrameQuality,
    RobotPose,
)

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
    #2 定位：检测框底边中点 -> 相机射线 -> 与地平面求交 -> (x, z) -> RobotPose 转世界坐标。
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
        camera_x_m: float | None = None,
        camera_z_m: float | None = None,
        ground_plane_height_m: float | None = None,
        source: str = "camera",
        calibration: CameraCalibration | None = None,
        calibration_path: str | Path | None = None,
        replay_path: str | Path | None = None,
        time_origin: float | None = None,
        relative_time_only: bool = False,
        size_policy: SizePolicy | None = None,
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
            # 标定与显式内参/外参不能混传，避免冲突参数静默覆盖
            explicit = {
                name: value
                for name, value in (
                    ("camera_height_m", camera_height_m),
                    ("pitch_rad", pitch_rad),
                    ("fx", fx),
                    ("fy", fy),
                    ("cx", cx),
                    ("cy", cy),
                    ("image_width", image_width),
                    ("image_height", image_height),
                    ("camera_x_m", camera_x_m),
                    ("camera_z_m", camera_z_m),
                    ("ground_plane_height_m", ground_plane_height_m),
                ) if value is not None
            }
            if explicit:
                raise ValueError(
                    "传入 calibration 时不能再显式传内参/外参，"
                    f"冲突字段：{sorted(explicit)}"
                )
            cal = calibration
            self.source = source
        else:
            missing = [
                name for name, value in (
                    ("camera_height_m", camera_height_m),
                    ("pitch_rad", pitch_rad),
                    ("fx", fx),
                    ("fy", fy),
                    ("cx", cx),
                    ("cy", cy),
                    ("image_width", image_width),
                    ("image_height", image_height),
                ) if value is None
            ]
            if missing:
                raise ValueError(
                    f"缺少相机参数 {missing}；请传 calibration 或 calibration_path"
                )
            if image_width is None or image_height is None:
                raise ValueError("image_width/image_height 必须显式提供，无法可靠推导")
            cal = CameraCalibration(
                image_width=int(image_width),
                image_height=int(image_height),
                fx=float(fx),
                fy=float(fy),
                cx=float(cx),
                cy=float(cy),
                camera_height_m=float(camera_height_m),
                pitch_rad=float(pitch_rad),
                camera_x_m=float(camera_x_m if camera_x_m is not None else 0.0),
                camera_z_m=float(camera_z_m if camera_z_m is not None else 0.0),
                ground_plane_height_m=float(ground_plane_height_m if ground_plane_height_m is not None else 0.0),
                source="test",
                is_real_calibration=False,
                name=source,
            )
            self.source = source

        self.calibration = cal
        self.calibration_trusted = cal.is_real_calibration
        if not self.calibration_trusted:
            logger.warning(
                "测试用途相机标定 name=%s source=%s is_real_calibration=false；"
                "严格 Judge 不会据其返回成功。",
                cal.name,
                cal.source,
            )

        self.replay_path = Path(replay_path) if replay_path is not None else None
        self.time_origin = time_origin
        self.relative_time_only = relative_time_only
        self.size_policy = size_policy or SizePolicy()

        self.frames_loaded = 0
        self.frames_yielded = 0
        self.frames_skipped = 0
        self.detections_skipped = 0

    # ------------------------------------------------------------------ 像素 -> 地面

    def pixel_to_ground(
        self, u: float, v: float, pose: RobotPose | None = None
    ) -> Tuple[float, float]:
        """像素坐标 -> 世界坐标 (x, z)。

        pose 为 None 时等价于机器人位于世界原点（即返回机器人局部坐标）。
        """
        x_local, z_local = self.calibration.project_pixel_to_ground(u, v)
        if pose is None:
            return x_local, z_local
        return pose.to_world(x_local, z_local)

    # ------------------------------------------------------------------ 回放

    def stream(self) -> Iterator[Tuple[float, RobotPose, List[Detection]]]:
        """离线 JSON/JSONL 检测帧回放。

        JSON 顶层结构：
            {"time_origin": 1786417200.0, "frames": [{...}, ...]}
        或直接是 frames 数组（此时必须通过构造参数提供 time_origin，
        或显式 relative_time_only=True）。
        JSONL 每行一个 frame 对象；第一行可为
            {"time_origin": 1786417200.0}
        作为相对时钟原点。

        每帧字段：
            frame_id: str
            timestamp: float（相对秒，time_origin 对应的内部时钟）
            image_width / image_height: 原始图像分辨率
            robot_pose: {x, z, yaw_rad, pose_uncertainty_cm}
            detections: [{class_name, confidence, bbox: [x1, y1, x2, y2],
                          radius_cm?, size_source?, size_trusted?, camera_id?}]
        """
        if self.replay_path is None:
            raise NotImplementedError(
                "CameraDetectorProvider.stream() 目前只支持离线 JSON/JSONL 回放；"
                "实时相机与板端 TPU 是独立 adapter，尚未接入"
            )

        frames, loaded_time_origin = self._load_replay_frames(
            self.replay_path, default_time_origin=self.time_origin
        )
        if loaded_time_origin is None and not self.relative_time_only:
            raise ValueError(
                "缺少 time_origin：严格回放必须提供 time_origin（JSON/JSONL meta 或构造参数），"
                "或显式 relative_time_only=True"
            )
        self.time_origin = loaded_time_origin

        self.frames_loaded = len(frames)
        self.frames_yielded = 0
        self.frames_skipped = 0
        self.detections_skipped = 0
        logger.info(
            "开始回放 path=%s frames=%d time_origin=%s calibration=%s calibration_trusted=%s",
            self.replay_path,
            self.frames_loaded,
            self.time_origin,
            self.calibration.name,
            self.calibration_trusted,
        )

        for raw_frame in frames:
            try:
                output = self._frame_to_output(raw_frame)
            except (ValueError, KeyError, TypeError) as exc:
                self.frames_skipped += 1
                logger.warning(
                    "跳过整帧 frame_id=%s 原因=%s",
                    raw_frame.get("frame_id", "<unknown>") if isinstance(raw_frame, dict) else raw_frame,
                    exc,
                )
                continue
            self.frames_yielded += 1
            logger.debug(
                "回放帧 frame_id=%s timestamp=%.3f detections=%d",
                raw_frame.get("frame_id", "<unknown>"),
                output[0],
                len(output[2]),
            )
            yield output

        logger.info(
            "回放结束 loaded=%d yielded=%d skipped_frames=%d skipped_detections=%d",
            self.frames_loaded,
            self.frames_yielded,
            self.frames_skipped,
            self.detections_skipped,
        )

    @staticmethod
    def _load_replay_frames(
        path: Path, default_time_origin: float | None = None
    ) -> Tuple[List[Dict[str, Any]], float | None]:
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
            if "frames" not in data:
                raise ValueError("JSON 对象缺少 frames 字段")
            time_origin = float(data["time_origin"]) if "time_origin" in data else default_time_origin
            frames = data.get("frames", [])
            if not isinstance(frames, list):
                raise ValueError("frames 必须是列表")
            return frames, time_origin
        raise ValueError(f"回放文件必须是 JSON 对象/数组或 JSONL，收到 {type(data).__name__}")

    def _frame_to_output(
        self, raw_frame: Dict[str, Any]
    ) -> Tuple[float, RobotPose, List[Detection]]:
        """帧级校验 + 检测级校验。

        帧级错误（frame_id/timestamp/分辨率/robot_pose）抛 ValueError 跳过整帧；
        单条检测错误只跳过该检测，并计入 detections_skipped。
        """
        if not isinstance(raw_frame, dict):
            raise ValueError(f"frame 必须是对象，收到 {type(raw_frame).__name__}")

        frame_id = str(raw_frame.get("frame_id", ""))
        if not frame_id.strip():
            raise ValueError("frame_id 必须是非空字符串")
        frame_id = frame_id.strip()

        timestamp = raw_frame.get("timestamp")
        try:
            timestamp = float(timestamp)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"timestamp 必须是有限数值，收到 {timestamp!r}") from exc
        if not math.isfinite(timestamp):
            raise ValueError(f"timestamp 必须是有限数值，收到 {timestamp!r}")

        try:
            image_width = int(raw_frame["image_width"])
            image_height = int(raw_frame["image_height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("image_width/image_height 必须是正整数") from exc
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"图像尺寸必须是正整数，收到 {image_width}x{image_height}")
        c = self.calibration
        if (image_width, image_height) != (c.image_width, c.image_height):
            raise ValueError(
                "回放分辨率与标定不一致："
                f"frame={image_width}x{image_height}, "
                f"calibration={c.image_width}x{c.image_height}"
            )

        p = raw_frame.get("robot_pose", {})
        if not isinstance(p, dict):
            raise ValueError("robot_pose 必须是对象")
        try:
            pose = RobotPose(
                x=float(p.get("x", 0.0)),
                z=float(p.get("z", 0.0)),
                yaw_rad=float(p.get("yaw_rad", 0.0)),
                pose_uncertainty_cm=float(p.get("pose_uncertainty_cm", 0.0)),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"robot_pose 字段必须是有限数值：{p!r}") from exc
        for label, value in (
            ("pose.x", pose.x), ("pose.z", pose.z),
            ("pose.yaw_rad", pose.yaw_rad), ("pose.pose_uncertainty_cm", pose.pose_uncertainty_cm),
        ):
            if not math.isfinite(value):
                raise ValueError(f"robot_pose.{label} 必须是有限数值，收到 {value!r}")

        detections: List[Detection] = []
        frame_detections_skipped = 0
        raw_detections = raw_frame.get("detections", [])
        if not isinstance(raw_detections, list):
            raise ValueError(f"detections 必须是列表，收到 {type(raw_detections).__name__}")

        for det_idx, det_record in enumerate(raw_detections):
            if not isinstance(det_record, dict):
                self._skip_detection(frame_id, det_idx, "<non-object>", "检测项必须是对象")
                frame_detections_skipped += 1
                continue
            class_name = str(det_record.get("class_name", ""))
            try:
                raw = RawDetection(
                    class_name=class_name,
                    confidence=float(det_record.get("confidence", float("nan"))),
                    bbox=tuple(float(v) for v in det_record.get("bbox", [])),
                    frame_id=frame_id,
                    timestamp=timestamp,
                    camera_id=str(det_record.get("camera_id", "overhead")),
                )
                validate_detection_for_frame(raw, frame_id, timestamp, image_width, image_height)
                x1, y1, x2, y2 = raw.bbox
                u = (x1 + x2) / 2.0
                v = y2
                x_local, z_local = self.calibration.project_pixel_to_ground(u, v)
                x, z = pose.to_world(x_local, z_local)
                size_evidence = self._resolve_size(det_record, raw, x_local, z_local)
            except (TypeError, ValueError) as exc:
                self._skip_detection(frame_id, det_idx, class_name or "<unknown>", str(exc))
                frame_detections_skipped += 1
                continue

            detections.append(
                Detection(
                    class_name=raw.class_name,
                    x=x,
                    z=z,
                    confidence=raw.confidence,
                    radius_cm=size_evidence.radius_cm,
                    size_source=size_evidence.size_source,
                    size_trusted=size_evidence.size_trusted,
                    bbox=raw.bbox,
                    frame_id=frame_id,
                    source=self.source,
                    timestamp=timestamp,
                    frame_quality=FrameQuality(
                        frame_id=frame_id,
                        degraded=frame_detections_skipped > 0,
                        frames_skipped=0,
                        detections_skipped=frame_detections_skipped,
                        calibration_trusted=self.calibration_trusted,
                        coordinate_frame=COORDINATE_FRAME_WORLD,
                    ),
                )
            )

        # 帧内检测被跳过时，本帧所有检测都要带着 degraded 标记
        if frame_detections_skipped > 0:
            for det in detections:
                if det.frame_quality is not None:
                    det.frame_quality.degraded = True
                    det.frame_quality.detections_skipped = frame_detections_skipped

        return timestamp, pose, detections

    def _skip_detection(self, frame_id: str, det_idx: int, class_name: str, reason: str) -> None:
        self.detections_skipped += 1
        logger.warning(
            "跳过检测 frame_id=%s index=%d class=%s 原因=%s",
            frame_id, det_idx, class_name, reason,
        )

    def _resolve_size(self, det_record: Dict[str, Any], raw: RawDetection,
                      x_local: float, z_local: float):
        """尺寸来源策略：显式物理尺寸 > 实例配置 > bbox 启发式 > 未知。"""
        try:
            explicit = self.size_policy.from_detection_record(det_record)
            if explicit is not None:
                return explicit
        except ValueError as exc:
            raise ValueError(f"显式尺寸无效：{exc}") from exc

        from_config = self.size_policy.from_instance_config(raw.class_name)
        if from_config is not None:
            return from_config

        ground_range = math.hypot(
            x_local - self.calibration.camera_x_m,
            z_local - self.calibration.camera_z_m,
        )
        return self.size_policy.from_bbox_heuristic(
            raw.bbox,
            ground_range_m=ground_range,
            fx=self.calibration.fx,
            camera_height_m=self.calibration.camera_height_m,
            ground_plane_height_m=self.calibration.ground_plane_height_m,
        )
