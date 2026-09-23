"""Mock 感知源：读一份 JSON 场景序列。

Mock 与 Camera 回放必须走同一个 SizePolicy：外部 JSON 不能声明 size_trusted，
可信尺寸只能来自本地 ObjectSizeRegistry。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, List, Tuple

from ..aliases import AliasTable
from ..size_policy import ObjectSizeRegistry, SizePolicy
from .base import PerceptionProvider
from ..types import (
    COORDINATE_FRAME_WORLD,
    Detection,
    FrameQuality,
    RobotPose,
)

logger = logging.getLogger(__name__)


class MockProvider(PerceptionProvider):
    def __init__(
        self,
        scene_path: str | Path,
        object_sizes_path: str | Path | None = None,
        aliases: AliasTable | None = None,
    ):
        with open(scene_path, "r", encoding="utf-8") as f:
            self.scene = json.load(f)
        self.frames = self.scene["frames"]
        origin = self.scene.get("time_origin")
        self.time_origin: float | None = float(origin) if origin is not None else None
        self.aliases = aliases or AliasTable()
        self.object_size_registry = (
            ObjectSizeRegistry.from_json(object_sizes_path)
            if object_sizes_path is not None
            else ObjectSizeRegistry()
        )
        self.size_policy = SizePolicy(
            registry=self.object_size_registry,
            aliases=self.aliases,
        )
        self.detections_skipped = 0

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
            dets: List[Detection] = []
            for det_idx, d in enumerate(frame.get("detections", [])):
                if not isinstance(d, dict):
                    self.detections_skipped += 1
                    logger.warning(
                        "跳过检测 frame_id=%s index=%d class=<non-object> 原因=检测项必须是对象",
                        frame.get("frame_id"), det_idx,
                    )
                    continue
                if "size_trusted" in d:
                    self.detections_skipped += 1
                    logger.warning(
                        "跳过检测 frame_id=%s index=%d class=%s 原因=external size_trusted is forbidden",
                        frame.get("frame_id"), det_idx, d.get("class_name", "<unknown>"),
                    )
                    continue
                canonical = self.aliases.canonical(str(d.get("class_name", "")))
                local = (
                    self.object_size_registry.lookup_instance(str(d["instance_id"]))
                    if d.get("instance_id") is not None
                    else None
                )
                if local is None:
                    local = self.object_size_registry.lookup_class(canonical)
                if local is not None:
                    radius_cm = local.radius_cm
                    size_source = local.size_source
                    size_trusted = local.size_trusted
                    radius_semantics = local.radius_semantics
                elif "radius_cm" in d:
                    radius_cm = float(d["radius_cm"])
                    size_source = "external_claim"
                    size_trusted = False
                    radius_semantics = "unknown"
                else:
                    radius_cm = -1.0
                    size_source = "unknown"
                    size_trusted = False
                    radius_semantics = "unknown"
                dets.append(
                    Detection(
                        class_name=d["class_name"],
                        x=d["x"],
                        z=d["z"],
                        confidence=d.get("confidence", 0.9),
                        radius_cm=radius_cm,
                        size_source=size_source,
                        size_trusted=size_trusted,
                        radius_semantics=radius_semantics,
                        canonical_name=canonical,
                        bbox=tuple(d["bbox"]) if d.get("bbox") else None,
                        frame_id=frame.get("frame_id"),
                        source=d.get("source", "mock"),
                        timestamp=ts,
                        frame_quality=FrameQuality(
                            frame_id=frame.get("frame_id"),
                            degraded=False,
                            frames_skipped=0,
                            detections_skipped=0,
                            calibration_trusted=True,
                            coordinate_frame=COORDINATE_FRAME_WORLD,
                        ),
                    )
                )
            yield ts, pose, dets
