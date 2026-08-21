"""检测输入契约：外部检测器原始输出，未做任何坐标/类别转换。

这里的字段只做**合法性检查**，不做静默修正（不 clip、不重排 bbox、不改名）。
bbox 契约：
  - 格式：xyxy（x1, y1, x2, y2）
  - 坐标：原始图像像素坐标系，左上角为原点，u 向右，v 向下
  - x1 < x2 且 y1 < y2，且整框必须落在图像范围内
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


def _require_finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{label} 必须是有限数值，收到 {value!r}")
    return value


@dataclass
class RawDetection:
    """单条原始检测结果，字段对应外部 YOLO 输出。

    class_name 保留检测器原始类名（如 "sports ball" / "tennis_ball"），
    别名映射是 WorldModel 关联阶段的事，不在这里做。
    """

    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]   # xyxy，原始图像像素坐标
    frame_id: str
    timestamp: float
    camera_id: str = "overhead"

    def __post_init__(self) -> None:
        if not isinstance(self.class_name, str) or not self.class_name.strip():
            raise ValueError(f"class_name 必须是非空字符串，收到 {self.class_name!r}")
        self.class_name = self.class_name.strip()

        self.confidence = _require_finite(float(self.confidence), "confidence")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence 必须在 [0, 1]，收到 {self.confidence!r}（不允许静默截断）"
            )

        if not isinstance(self.bbox, (tuple, list)) or len(self.bbox) != 4:
            raise ValueError(f"bbox 必须是 (x1, y1, x2, y2) 四元组，收到 {self.bbox!r}")
        x1, y1, x2, y2 = (float(v) for v in self.bbox)
        for name, v in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)):
            _require_finite(v, f"bbox.{name}")
        if x2 <= x1:
            raise ValueError(f"bbox 倒置或零面积：x2={x2!r} 必须大于 x1={x1!r}")
        if y2 <= y1:
            raise ValueError(f"bbox 倒置或零面积：y2={y2!r} 必须大于 y1={y1!r}")
        if x1 < 0 or y1 < 0:
            raise ValueError(
                f"bbox 不能出现负坐标（原始图像左上角为原点）：({x1}, {y1}, {x2}, {y2})"
            )
        self.bbox = (x1, y1, x2, y2)

        self.timestamp = _require_finite(float(self.timestamp), "timestamp")
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise ValueError(f"frame_id 必须是非空字符串，收到 {self.frame_id!r}")
        self.frame_id = self.frame_id.strip()
        if not isinstance(self.camera_id, str) or not self.camera_id.strip():
            raise ValueError(f"camera_id 必须是非空字符串，收到 {self.camera_id!r}")
        self.camera_id = self.camera_id.strip()


@dataclass
class DetectionFrame:
    """一帧图像里所有原始检测。

    image_width / image_height 是**原始图像**分辨率，用来做 bbox 越界检查。
    """

    frame_id: str
    timestamp: float
    image_width: int
    image_height: int
    detections: List[RawDetection]

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise ValueError(f"frame_id 必须是非空字符串，收到 {self.frame_id!r}")
        self.frame_id = self.frame_id.strip()
        self.timestamp = _require_finite(float(self.timestamp), "timestamp")

        if not isinstance(self.image_width, int) or self.image_width <= 0:
            raise ValueError(f"image_width 必须是正整数，收到 {self.image_width!r}")
        if not isinstance(self.image_height, int) or self.image_height <= 0:
            raise ValueError(f"image_height 必须是正整数，收到 {self.image_height!r}")

        if not isinstance(self.detections, list):
            raise ValueError(f"detections 必须是列表，收到 {self.detections!r}")

        for det in self.detections:
            if not isinstance(det, RawDetection):
                raise ValueError(f"detections 每项必须是 RawDetection，收到 {det!r}")
            if det.frame_id != self.frame_id:
                raise ValueError(
                    f"DetectionFrame.frame_id={self.frame_id!r} 与 RawDetection.frame_id="
                    f"{det.frame_id!r} 不一致"
                )
            if det.timestamp != self.timestamp:
                raise ValueError(
                    f"DetectionFrame.timestamp={self.timestamp!r} 与 RawDetection.timestamp="
                    f"{det.timestamp!r} 不一致"
                )
            x1, y1, x2, y2 = det.bbox
            if x2 > float(self.image_width):
                raise ValueError(
                    f"bbox 越界：x2={x2!r} > image_width={self.image_width}（({x1}, {y1}, {x2}, {y2})）"
                )
            if y2 > float(self.image_height):
                raise ValueError(
                    f"bbox 越界：y2={y2!r} > image_height={self.image_height}（({x1}, {y1}, {x2}, {y2})）"
                )

    def to_raw_detections(self) -> List[RawDetection]:
        """拆成 provider 可直接消费的原始检测列表。"""
        return list(self.detections)
