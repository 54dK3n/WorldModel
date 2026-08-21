"""相机标定参数与统一相机可见性模型。

坐标系定义：
  - 世界坐标：x 向右，y 向上，z 向前（WorldModel 统一使用世界坐标）
  - 机器人局部坐标：x 向右，y 向上，z 向前；相机中心
    (camera_x_m, camera_height_m, camera_z_m) 是相机相对机器人坐标
  - 像素坐标：u 向右，v 向下，左上角为原点
  - pitch_rad 正方向：相机光轴向下俯（overhead 相机看桌面时为正）
  - 地面/桌面平面为 y = ground_plane_height_m

有效投影范围（地面点相对相机地面投影点的水平距离）：
  min_ground_range_m <= range <= max_ground_range_m
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from .types import RobotPose


def _finite(v: float, label: str) -> float:
    if not math.isfinite(v):
        raise ValueError(f"{label} 必须是有限数值，收到 {v!r}")
    return v


@dataclass
class CameraCalibration:
    """针孔相机 + 俯仰角 + 地面平面高度 + 有效投影范围。

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
    min_ground_range_m: float = 0.15
    max_ground_range_m: float = 4.0
    name: str = "overhead_camera"           # 相机名称，区别于检测 source 与标定 source
    source: str = "test"                    # 标定来源；test = 测试占位
    is_real_calibration: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.image_width, int) or self.image_width <= 0:
            raise ValueError(f"image_width 必须是正整数，收到 {self.image_width!r}")
        if not isinstance(self.image_height, int) or self.image_height <= 0:
            raise ValueError(f"image_height 必须是正整数，收到 {self.image_height!r}")

        for label in ("fx", "fy"):
            v = _finite(float(getattr(self, label)), label)
            if v <= 0:
                raise ValueError(f"{label} 必须是有限正数，收到 {v!r}")
            setattr(self, label, v)

        for label in ("cx", "cy"):
            setattr(self, label, _finite(float(getattr(self, label)), label))

        for label in ("camera_height_m", "pitch_rad", "camera_x_m", "camera_z_m",
                      "ground_plane_height_m", "min_ground_range_m", "max_ground_range_m"):
            setattr(self, label, _finite(float(getattr(self, label)), label))

        if self.camera_height_m <= self.ground_plane_height_m:
            raise ValueError(
                "camera_height_m 必须大于 ground_plane_height_m（相机在平面之上）"
            )
        if self.min_ground_range_m < 0:
            raise ValueError(f"min_ground_range_m 必须 >= 0，收到 {self.min_ground_range_m!r}")
        if self.max_ground_range_m <= self.min_ground_range_m:
            raise ValueError(
                "max_ground_range_m 必须大于 min_ground_range_m，"
                f"收到 {self.max_ground_range_m!r} <= {self.min_ground_range_m!r}"
            )

    # ------------------------------------------------------------- 投影几何

    @property
    def height_above_ground_m(self) -> float:
        return self.camera_height_m - self.ground_plane_height_m

    def project_pixel_to_ground(self, u: float, v: float) -> Tuple[float, float]:
        """像素 -> 机器人局部地面坐标 (x, z)。不做 RobotPose 变换。"""
        if not math.isfinite(float(u)) or not math.isfinite(float(v)):
            raise ValueError(f"像素坐标必须是有限数值，收到 u={u!r}, v={v!r}")

        xn = (float(u) - self.cx) / self.fx
        yn = (float(v) - self.cy) / self.fy
        sin_p = math.sin(self.pitch_rad)
        cos_p = math.cos(self.pitch_rad)

        # 相机坐标 x 向右、y 向下、z 向前；世界 y 向上。
        # 绕 x 轴旋转后：
        #   y_down' = yn*cos_p + sin_p
        #   z'      = -yn*sin_p + cos_p
        # 世界 y 向上 = -y_down'
        d_x = xn
        d_y = -(yn * cos_p + sin_p)
        d_z = -yn * sin_p + cos_p

        if abs(d_y) < 1e-12:
            raise ValueError(
                f"像素 ({u:.2f}, {v:.2f}) 位于可投影地面区域之外：射线与地面平行"
            )
        t = (self.ground_plane_height_m - self.camera_height_m) / d_y
        if t <= 0.0:
            raise ValueError(
                f"像素 ({u:.2f}, {v:.2f}) 位于可投影地面区域之外："
                "射线与地面无正向交点（地平线以上或相机后方）"
            )

        x = self.camera_x_m + t * d_x
        z = self.camera_z_m + t * d_z
        if not (math.isfinite(x) and math.isfinite(z)):
            raise ValueError(
                f"像素 ({u:.2f}, {v:.2f}) 投影结果非有限值：x={x!r}, z={z!r}"
            )

        ground_range = math.hypot(x - self.camera_x_m, z - self.camera_z_m)
        if ground_range < self.min_ground_range_m:
            raise ValueError(
                f"像素 ({u:.2f}, {v:.2f}) 的交点小于最小量程："
                f"{ground_range:.3f}m < {self.min_ground_range_m}m"
            )
        if ground_range > self.max_ground_range_m:
            raise ValueError(
                f"像素 ({u:.2f}, {v:.2f}) 的交点超过有效量程："
                f"{ground_range:.3f}m > {self.max_ground_range_m}m"
            )
        return x, z

    def project_pixel_to_world(
        self, u: float, v: float, pose: RobotPose
    ) -> Tuple[float, float]:
        """像素 -> 世界坐标 (x, z)。"""
        x_local, z_local = self.project_pixel_to_ground(u, v)
        return pose.to_world(x_local, z_local)

    def ground_point_to_pixel(self, x_local: float, z_local: float) -> Tuple[float, float]:
        """机器人局部地面坐标 -> 像素坐标（project_pixel_to_ground 的逆）。"""
        x_local = _finite(float(x_local), "x_local")
        z_local = _finite(float(z_local), "z_local")

        dx = x_local - self.camera_x_m
        dz = z_local - self.camera_z_m
        h = self.height_above_ground_m
        sin_p = math.sin(self.pitch_rad)
        cos_p = math.cos(self.pitch_rad)

        # 反推归一化射线：
        #   dx = t * xn
        #   dz = t * d_z
        #   t  = h / (yn*cos_p + sin_p)
        q = -dz / h
        denom = sin_p - q * cos_p
        if abs(denom) < 1e-12:
            raise ValueError("地面点位于投影盲区，无法映射到像素")
        yn = (cos_p + q * sin_p) / denom

        denom_t = yn * cos_p + sin_p
        if abs(denom_t) < 1e-12:
            raise ValueError("地面点位于相机后方，无法映射到像素")
        t = h / denom_t
        if t <= 0:
            raise ValueError("地面点位于相机后方，无法映射到像素")

        xn = dx / t
        u = self.cx + self.fx * xn
        v = self.cy + self.fy * yn
        if not (math.isfinite(u) and math.isfinite(v)):
            raise ValueError(f"像素投影非有限值：u={u!r}, v={v!r}")
        return u, v

    def is_ground_point_visible(
        self, x_world: float, z_world: float, pose: RobotPose
    ) -> bool:
        """统一可见性模型：世界点 -> 机器人局部 -> 像素 -> 图像范围内。"""
        if not math.isfinite(float(x_world)) or not math.isfinite(float(z_world)):
            return False
        x_local, z_local = pose.to_local(x_world, z_world)
        ground_range = math.hypot(x_local - self.camera_x_m, z_local - self.camera_z_m)
        if ground_range < self.min_ground_range_m or ground_range > self.max_ground_range_m:
            return False
        try:
            u, v = self.ground_point_to_pixel(x_local, z_local)
        except ValueError:
            return False
        if u < 0.0 or u > float(self.image_width) or v < 0.0 or v > float(self.image_height):
            return False
        return True

    def is_visible(self, x_world: float, z_world: float, pose: RobotPose) -> bool:
        """统一可见性模型接口别名。"""
        return self.is_ground_point_visible(x_world, z_world, pose)

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
            "min_ground_range_m": self.min_ground_range_m,
            "max_ground_range_m": self.max_ground_range_m,
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
