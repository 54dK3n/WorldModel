"""数据类型定义。

分两层：
  Detection     — 单帧观测，provider 输出，无状态
  TrackedObject — 跨帧维护的对象信念，有状态（这才是 World Model 的"Model"）

坐标约定（统一为世界坐标）：
  Detection.x/z 和 TrackedObject.x/z 都是世界坐标（米）。
  机器人本体/相机局部坐标只存在于 provider 内部，进入 WorldModel 前必须完成
  RobotPose.to_world() 转换。yaw 旋转方向：yaw=+pi/2 表示机器人左转 90 度，
  即世界朝向从 +z 转向 +x。

对外契约（scene_observations）由 adapters.py 从 TrackedObject 导出，
内部结构与对外契约解耦，契约变动只改 adapter。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

# 尺寸证据来源
SIZE_SOURCE_LOCAL_REGISTRY = "local_registry"  # 本地人工测量/规格/规则配置（唯一可信来源）
SIZE_SOURCE_EXTERNAL_CLAIM = "external_claim"    # 检测器/回放/公开数据声称的尺寸（不可信）
SIZE_SOURCE_BBOX_HEURISTIC = "bbox_heuristic"    # 根据标定与 bbox 估算（不可信）
SIZE_SOURCE_DEFAULT = "default"               # 类别/代码默认值（不可信）
SIZE_SOURCE_UNKNOWN = "unknown"               # 尺寸未知或缺失

# 兼容旧字段（生产输入不得自行声明可信）
SIZE_SOURCE_DETECTOR = "detector"
SIZE_SOURCE_INSTANCE_CONFIG = "instance_config"

# 半径语义
RADIUS_SEMANTICS_OUTER = "outer_radius"
RADIUS_SEMANTICS_INNER = "inner_radius"
RADIUS_SEMANTICS_UNKNOWN = "unknown"

# 坐标帧
COORDINATE_FRAME_WORLD = "world"
COORDINATE_FRAME_ROBOT = "robot"


class ObjectState(str, Enum):
    """对象生命周期状态机。

    TENTATIVE --hit_count>=N--> CONFIRMED --conf<stale--> STALE --conf<lost--> LOST
    LOST 的对象移出 get_scene() 快照，但保留在历史里（不物理删除）。
    """

    TENTATIVE = "tentative"   # 刚出现，还没confirm，可能是误检
    CONFIRMED = "confirmed"   # 稳定可信
    STALE = "stale"           # 一段时间没更新，置信度已衰减
    LOST = "lost"             # 基本可以认为不在了


@dataclass
class FrameQuality:
    """一帧观测的质量标记。严格 Judge 只在 frame quality 完整时允许成功。"""

    frame_id: Optional[str] = None
    degraded: bool = False
    frames_skipped: int = 0
    detections_skipped: int = 0
    calibration_trusted: bool = False
    coordinate_frame: str = COORDINATE_FRAME_WORLD

    def is_complete(self) -> bool:
        return (
            not self.degraded
            and self.frames_skipped == 0
            and self.detections_skipped == 0
            and self.calibration_trusted is True
            and self.coordinate_frame == COORDINATE_FRAME_WORLD
        )


@dataclass
class RobotPose:
    """机器人位姿。**由底盘/仿真器提供，World Model 只消费不计算。**

    pose_uncertainty_cm 是里程计累计误差估计；观测入库时会把当时的
    不确定度一起记下来（见 TrackedObject.pose_uncertainty_cm）。
    """

    x: float = 0.0
    z: float = 0.0
    yaw_rad: float = 0.0          # 朝向，用于 FOV 判断
    pose_uncertainty_cm: float = 0.0

    def to_world(self, x_local: float, z_local: float) -> Tuple[float, float]:
        """机器人局部坐标 -> 世界坐标。

        yaw 旋转方向：yaw=+pi/2 表示机器人从 +z 轴左转到 +x 轴。
        """
        cos = math.cos(self.yaw_rad)
        sin = math.sin(self.yaw_rad)
        x = self.x + x_local * cos + z_local * sin
        z = self.z + z_local * cos - x_local * sin
        return x, z

    def to_local(self, x_world: float, z_world: float) -> Tuple[float, float]:
        """世界坐标 -> 机器人局部坐标（to_world 的逆变换）。"""
        dx = x_world - self.x
        dz = z_world - self.z
        cos = math.cos(self.yaw_rad)
        sin = math.sin(self.yaw_rad)
        x_local = dx * cos - dz * sin
        z_local = dx * sin + dz * cos
        return x_local, z_local


@dataclass
class Detection:
    """单帧检测结果。provider 的输出格式。

    x/z 是世界坐标（米）。bbox 是原始图像像素 xyxy，进 evidence。
    """

    class_name: str               # 检测器原始类名，如 "sports ball"
    x: float                      # 世界坐标（米），地平面求交 + RobotPose 转换得出
    z: float
    confidence: float
    radius_cm: float = 5.0
    size_source: str = SIZE_SOURCE_DEFAULT
    size_trusted: bool = False
    radius_semantics: str = RADIUS_SEMANTICS_UNKNOWN
    canonical_name: Optional[str] = None
    bbox: Optional[Tuple[float, float, float, float]] = None   # 像素框 xyxy，进 evidence
    frame_id: Optional[str] = None                     # 帧引用，进 evidence
    source: str = "mock"
    timestamp: float = 0.0
    frame_quality: Optional[FrameQuality] = None


@dataclass
class TrackedObject:
    """跨帧维护的对象。内部状态，比对外契约丰富。"""

    obj_id: str                   # 稳定 id，全生命周期不变
    name: str                     # 规范名（别名表映射后）
    aliases: List[str] = field(default_factory=list)

    x: float = 0.0                # 世界坐标（米）
    z: float = 0.0                # 世界坐标（米）
    radius_cm: float = 5.0
    size_source: str = SIZE_SOURCE_DEFAULT
    size_trusted: bool = False
    radius_semantics: str = RADIUS_SEMANTICS_UNKNOWN
    confidence: float = 0.0

    first_seen: float = 0.0
    last_seen: float = 0.0        # 最后一次被检测到
    last_updated: float = 0.0     # 最后一次状态变更（含衰减）

    hit_count: int = 0
    miss_count: int = 0           # 仅统计"在视野内但没检测到"

    state: ObjectState = ObjectState.TENTATIVE
    pose_uncertainty_cm: float = 0.0

    source: str = "mock"
    last_bbox: Optional[Tuple[float, float, float, float]] = None
    last_frame_id: Optional[str] = None
    last_frame_quality: Optional[FrameQuality] = None

    def age(self, now: float) -> float:
        """距上次被真实看到过了多久（秒）。

        **刻意不做 max(0.0, ...) 夹取。** 返回负数说明 now 早于 last_seen，
        即调用方时钟与观测时钟不一致。夹成 0 会让判定层的陈旧度护栏静默失效
        —— 一份很旧的快照会被当成"刚刚看到"（见 judge 的 clock_skew 检查）。
        需要非负值的地方（如衰减 dt）自己夹。
        """
        return now - self.last_seen
