"""尺寸证据策略。

CameraDetectorProvider 不得静默给所有检测 radius_cm=5.0。尺寸来源必须显式：
  - detector:           检测器输出中显式提供了物理尺寸
  - instance_config:    可信对象实例配置（例如规则书/队友提供的球 3.3cm、桶 15cm）
  - bbox_heuristic:     根据标定与 bbox 估算（不可信，严格 Judge 不得据此成功）
  - default/unknown:    缺失或类别默认值（不可信）
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from .types import (
    SIZE_SOURCE_BBOX_HEURISTIC,
    SIZE_SOURCE_DEFAULT,
    SIZE_SOURCE_DETECTOR,
    SIZE_SOURCE_INSTANCE_CONFIG,
    SIZE_SOURCE_UNKNOWN,
)


@dataclass
class SizeEvidence:
    radius_cm: float
    size_source: str
    size_trusted: bool


DEFAULT_INSTANCE_SIZES_CM = {
    "sports ball": 3.3,
    "tennis ball": 3.3,
    "tennis_ball": 3.3,
    "ball": 3.3,
    "basket": 15.0,
    "bucket": 15.0,
}


class SizePolicy:
    """按来源解析物理尺寸。未知/启发式尺寸绝不标记为可信。"""

    def __init__(self, instance_sizes_cm: Optional[Mapping[str, float]] = None):
        self.instance_sizes_cm = dict(instance_sizes_cm or DEFAULT_INSTANCE_SIZES_CM)

    def from_detection_record(self, record: Mapping) -> Optional[SizeEvidence]:
        """检测记录里显式给了 radius_cm 时使用。

        只有带明确可信来源的显式物理尺寸才可信；只给 radius_cm 而没有来源
        时按 detector 处理（检测器显式提供的物理尺寸），保持可信。
        """
        if "radius_cm" not in record:
            return None
        radius_cm = float(record["radius_cm"])
        if not math.isfinite(radius_cm) or radius_cm <= 0.0:
            raise ValueError(f"radius_cm 必须是有限正数，收到 {radius_cm!r}")
        source = str(record.get("size_source", SIZE_SOURCE_DETECTOR))
        trusted = bool(record.get("size_trusted", source in (SIZE_SOURCE_DETECTOR, SIZE_SOURCE_INSTANCE_CONFIG)))
        if source in (SIZE_SOURCE_DETECTOR, SIZE_SOURCE_INSTANCE_CONFIG):
            return SizeEvidence(radius_cm=radius_cm, size_source=source, size_trusted=trusted)
        return SizeEvidence(radius_cm=radius_cm, size_source=source, size_trusted=False)

    def from_instance_config(self, class_name: str) -> Optional[SizeEvidence]:
        """可信对象实例配置：显式维护的类别物理尺寸表。"""
        key = str(class_name).strip()
        if key in self.instance_sizes_cm:
            return SizeEvidence(
                radius_cm=float(self.instance_sizes_cm[key]),
                size_source=SIZE_SOURCE_INSTANCE_CONFIG,
                size_trusted=True,
            )
        return None

    def from_bbox_heuristic(
        self,
        bbox: Tuple[float, float, float, float],
        ground_range_m: float,
        fx: float,
        camera_height_m: float,
        ground_plane_height_m: float,
    ) -> SizeEvidence:
        """根据标定和 bbox 估算物理尺寸。永远不可信。"""
        x1, y1, x2, y2 = bbox
        width_px = max(float(x2) - float(x1), 1e-6)
        height_px = max(float(y2) - float(y1), 1e-6)
        diameter_px = min(width_px, height_px)
        # 物体离相机的近似直线距离；像素直径 -> 物理直径 -> 半径
        distance_m = math.hypot(ground_range_m, camera_height_m - ground_plane_height_m)
        diameter_m = diameter_px * distance_m / max(float(fx), 1e-6)
        radius_cm = max(diameter_m / 2.0 * 100.0, 0.1)
        return SizeEvidence(
            radius_cm=radius_cm,
            size_source=SIZE_SOURCE_BBOX_HEURISTIC,
            size_trusted=False,
        )

    def unknown(self) -> SizeEvidence:
        return SizeEvidence(radius_cm=-1.0, size_source=SIZE_SOURCE_UNKNOWN, size_trusted=False)

    def default(self) -> SizeEvidence:
        return SizeEvidence(radius_cm=5.0, size_source=SIZE_SOURCE_DEFAULT, size_trusted=False)
