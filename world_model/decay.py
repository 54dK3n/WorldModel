"""子问题 #4：时效与置信度衰减。

核心区分（本模块存在的唯一理由）：

    "没看见"有两种，处理方式相反。

    不在视野里        -> 只是转过头了，不构成"东西不在了"的证据，confidence 几乎不掉
    在视野内但没检测到 -> 这才是证据，说明可能被拿走了，confidence 快速衰减

两者混在一起处理，机器人转个身整个世界就忘光了。

"某坐标是否在当前视野内"只需要位姿 + 相机 FOV，不需要任何模型。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

from .calibration import CameraCalibration
from .types import ObjectState, RobotPose, TrackedObject


@dataclass
class FovConfig:
    """固定角度/距离的相机视野。仅作为未标定时的兜底可见性模型。

    相机投影产生的对象不应继续使用本模型；应传入 CameraCalibration 作为
    WorldModel(visibility=...) 或 apply_decay(..., fov_or_visibility=cal)。
    """

    horizontal_fov_deg: float = 70.0
    max_range_m: float = 4.0
    min_range_m: float = 0.15


@dataclass
class DecayConfig:
    """半衰期（秒）。confidence *= 0.5 ** (dt / half_life)"""

    # 在视野内却没检测到 —— 快
    half_life_in_fov_missed_s: float = 1.5
    # 不在视野内 —— 慢（几乎只是缓慢遗忘）
    half_life_out_of_fov_s: float = 60.0

    # 按类别调整：静止物慢衰减，滚动物快衰减
    # 乘在半衰期上，>1 = 更耐久
    class_half_life_scale: Dict[str, float] = field(
        default_factory=lambda: {
            "basket": 4.0,
            "table": 8.0,
            "ball": 1.0,       # 会滚，衰减最快
            "bottle": 2.0,
        }
    )

    stale_threshold: float = 0.50   # 低于此值 -> STALE
    lost_threshold: float = 0.15    # 低于此值 -> LOST，移出快照
    confirm_hits: int = 3           # 命中几次后 TENTATIVE -> CONFIRMED


def _norm_angle(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def is_in_fov(x: float, z: float, pose: RobotPose, fov: FovConfig) -> bool:
    """固定角度/距离的兜底 FOV 判断。相机投影对象请使用统一可见性模型。"""
    dx, dz = x - pose.x, z - pose.z
    dist = math.hypot(dx, dz)
    if dist > fov.max_range_m or dist < fov.min_range_m:
        return False
    bearing = math.atan2(dx, dz)              # z 为前向
    rel = abs(_norm_angle(bearing - pose.yaw_rad))
    return rel <= math.radians(fov.horizontal_fov_deg / 2.0)


def resolve_visibility(fov_or_visibility) -> object:
    """把旧 FovConfig 或新 CameraCalibration 统一为可见性模型。

    可见性模型只需要实现 is_visible(x_world, z_world, pose) -> bool。
    """
    if isinstance(fov_or_visibility, CameraCalibration):
        return fov_or_visibility
    if hasattr(fov_or_visibility, "is_visible"):
        return fov_or_visibility
    return FovVisibility(fov_or_visibility if isinstance(fov_or_visibility, FovConfig) else FovConfig())


class FovVisibility:
    """FovConfig 的适配器。"""

    def __init__(self, fov: FovConfig):
        self.fov = fov

    def is_visible(self, x_world: float, z_world: float, pose: RobotPose) -> bool:
        return is_in_fov(x_world, z_world, pose, self.fov)


def half_life_for(obj: TrackedObject, in_fov: bool, cfg: DecayConfig) -> float:
    base = cfg.half_life_in_fov_missed_s if in_fov else cfg.half_life_out_of_fov_s
    return base * cfg.class_half_life_scale.get(obj.name, 1.0)


def apply_decay(
    obj: TrackedObject,
    now: float,
    pose: RobotPose,
    fov_or_visibility: object,
    cfg: DecayConfig,
) -> TrackedObject:
    """对一个"这一帧没被匹配上"的对象做衰减。**不删除对象。**

    fov_or_visibility 可以是 FovConfig（旧接口）或 CameraCalibration。
    可见性未知时按不在视野内处理，避免把长期保留当默认。
    """
    dt = max(0.0, now - obj.last_updated)
    if dt == 0.0:
        return obj

    visibility = resolve_visibility(fov_or_visibility)
    try:
        in_fov = bool(visibility.is_visible(obj.x, obj.z, pose))
    except Exception:
        # 可见性未知时按“在视野内但没检测到”处理（快衰减），
        # 不允许以未知为借口长期保留对象。
        in_fov = True
    if in_fov:
        obj.miss_count += 1   # 只在视野内的漏检才算 miss

    hl = half_life_for(obj, in_fov, cfg)
    obj.confidence *= 0.5 ** (dt / max(hl, 1e-6))
    obj.last_updated = now

    # 位姿不确定度随时间累积（里程计漂移）；位姿被修正时上层可回写
    obj.pose_uncertainty_cm = max(obj.pose_uncertainty_cm, pose.pose_uncertainty_cm)

    return _update_state(obj, cfg)


def on_hit(obj: TrackedObject, cfg: DecayConfig) -> TrackedObject:
    obj.hit_count += 1
    return _update_state(obj, cfg)


def _update_state(obj: TrackedObject, cfg: DecayConfig) -> TrackedObject:
    if obj.confidence < cfg.lost_threshold:
        obj.state = ObjectState.LOST
    elif obj.confidence < cfg.stale_threshold:
        obj.state = ObjectState.STALE
    elif obj.hit_count >= cfg.confirm_hits:
        obj.state = ObjectState.CONFIRMED
    else:
        obj.state = ObjectState.TENTATIVE
    return obj
