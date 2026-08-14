"""Mock 感知源：读一份 JSON 场景序列。

零硬件依赖，用来开发和单测 #3 关联 与 #4 时效 —— 这两块才是 World Model
真正的内容，它们都不需要相机。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Tuple

from ..types import Detection, RobotPose
from .base import PerceptionProvider


class MockProvider(PerceptionProvider):
    def __init__(self, scene_path: str | Path):
        with open(scene_path, "r", encoding="utf-8") as f:
            self.scene = json.load(f)
        self.frames = self.scene["frames"]
        # 场景相对时刻 0.0 对应的 Unix 时间。回放数据的 t 是相对秒，
        # 不声明原点就没法导出正确的 ISO 时间戳。
        self.time_origin: float = float(self.scene.get("time_origin", 0.0))

    def stream(self) -> Iterator[Tuple[float, RobotPose, List[Detection]]]:
        for frame in self.frames:
            ts = float(frame["t"])
            p = frame.get("robot_pose", {})
            pose = RobotPose(
                x=p.get("x", 0.0),
                z=p.get("z", 0.0),
                yaw_rad=p.get("yaw_rad", 0.0),
                pose_uncertainty_cm=p.get("pose_uncertainty_cm", 0.0),
            )
            dets = [
                Detection(
                    class_name=d["class_name"],
                    x=d["x"],
                    z=d["z"],
                    confidence=d.get("confidence", 0.9),
                    radius_cm=d.get("radius_cm", 5.0),
                    bbox=tuple(d["bbox"]) if d.get("bbox") else None,
                    frame_id=frame.get("frame_id"),
                    source=d.get("source", "mock"),
                    timestamp=ts,
                )
                for d in frame.get("detections", [])
            ]
            yield ts, pose, dets
