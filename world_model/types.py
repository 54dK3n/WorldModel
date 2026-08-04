"""数据类型定义。

分两层：
  Detection     — 单帧观测，provider 输出，无状态
  TrackedObject — 跨帧维护的对象信念，有状态（这才是 World Model 的"Model"）

对外契约（scene_observations）由 adapters.py 从 TrackedObject 导出，
内部结构与对外契约解耦，契约变动只改 adapter。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


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
class RobotPose:
    """机器人位姿。**由底盘/仿真器提供，World Model 只消费不计算。**

    pose_uncertainty_cm 是里程计累计误差估计；观测入库时会把当时的
    不确定度一起记下来（见 TrackedObject.pose_uncertainty_cm）。
    """

    x: float = 0.0
    z: float = 0.0
    yaw_rad: float = 0.0          # 朝向，用于 FOV 判断
    pose_uncertainty_cm: float = 0.0
    # TODO(问宋红): 若坐标系原点为机器人本体，本类可退化为常量，#2b 整块砍掉


@dataclass
class Detection:
    """单帧检测结果。provider 的输出格式。"""

    class_name: str               # 检测器原始类名，如 "sports ball"
    x: float                      # 世界坐标（米），地平面求交得出
    z: float
    confidence: float
    radius_cm: float = 5.0
    bbox: Optional[Tuple[int, int, int, int]] = None   # 像素框 xyxy，进 evidence
    frame_id: Optional[str] = None                     # 帧引用，进 evidence
    source: str = "mock"
    timestamp: float = 0.0


@dataclass
class TrackedObject:
    """跨帧维护的对象。内部状态，比对外契约丰富。"""

    obj_id: str                   # 稳定 id，全生命周期不变
    name: str                     # 规范名（别名表映射后）
    aliases: List[str] = field(default_factory=list)

    x: float = 0.0
    z: float = 0.0
    radius_cm: float = 5.0
    confidence: float = 0.0

    first_seen: float = 0.0
    last_seen: float = 0.0        # 最后一次被检测到
    last_updated: float = 0.0     # 最后一次状态变更（含衰减）

    hit_count: int = 0
    miss_count: int = 0           # 仅统计"在视野内但没检测到"

    state: ObjectState = ObjectState.TENTATIVE
    pose_uncertainty_cm: float = 0.0

    source: str = "mock"
    last_bbox: Optional[Tuple[int, int, int, int]] = None
    last_frame_id: Optional[str] = None

    def age(self, now: float) -> float:
        """距上次被真实看到过了多久（秒）。"""
        return max(0.0, now - self.last_seen)
