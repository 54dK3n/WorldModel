"""相机标定参数。

坐标系定义（与 WorldModel 保持一致）：
  - 世界/机器人坐标：x 向右，y 向上，z 向前（FOV 判断用 z 为前向）
  - 像素坐标：u 向右，v 向下，左上角为原点
  - pitch_rad 正方向：相机光轴向下俯（摄像头往桌面看时为正）
  - 相机中心位于 (camera_x_m, camera_height_m, camera_z_m)
  - 地面/桌面平面为 y = ground_plane_height_m
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class CameraCalibration:
    """针孔相机 + 俯仰角 + 地面平面高度。

    is_real_calibration=False 表示测试用途的占位参数，不是真实标定结果。
    """

    image_width: int = 640
    image_height: int = 640
    fx: float = 500.0
    fy: float = 500.0
    cx: float = 320.0
    cy: float = 320.0
    camera_height_m: float = 1.5
    pitch_rad: float = 0.5235987755982988   # 30°，测试用途，非真实标定
    camera_x_m: float = 0.0
    camera_z_m: float = 0.0
    ground_plane_height_m: float = 0.0
    name: str = "overhead_camera"
    source: str = "test"                    # 标定来源；test = 测试占位
    is_real_calibration: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.image_width, int) or self.image_width <= 0:
            raise ValueError(f"image_width 必须是正整数，收到 {self.image_width!r}")
        if not isinstance(self.image_height, int) or self.image_height <= 0:
            raise ValueError(f"image_height 必须是正整数，收到 {self.image_height!r}")

        for label in ("fx", "fy"):
            v = float(getattr(self, label))
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{label} 必须是有限正数，收到 {v!r}")
            setattr(self, label, v)

        for label in ("cx", "cy"):
            v = float(getattr(self, label))
            if not math.isfinite(v):
                raise ValueError(f"{label} 必须是有限数值，收到 {v!r}")
            setattr(self, label, v)

        for label in ("camera_height_m", "pitch_rad", "camera_x_m", "camera_z_m", "ground_plane_height_m"):
            v = float(getattr(self, label))
            if not math.isfinite(v):
                raise ValueError(f"{label} 必须是有限数值，收到 {v!r}")
            setattr(self, label, v)

        if self.camera_height_m <= self.ground_plane_height_m:
            raise ValueError(
                "camera_height_m 必须大于 ground_plane_height_m（相机在平面之上）"
            )

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "camera_height_m": self.camera_height_m,
            "pitch_rad": self.pitch_rad,
            "camera_x_m": self.camera_x_m,
            "camera_z_m": self.camera_z_m,
            "ground_plane_height_m": self.ground_plane_height_m,
            "source": self.source,
            "is_real_calibration": self.is_real_calibration,
        }


def load_camera_calibration(path: str | Path) -> CameraCalibration:
    """从 JSON 文件读取标定。字段名与 CameraCalibration 一致。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"标定文件必须是 JSON 对象，收到 {type(data).__name__}")
    return CameraCalibration(**data)
