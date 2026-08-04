"""感知源 provider 抽象。

遵循叶志阳那条原则：**契约不改，在外围加 provider。**

    MockProvider（读 JSON 场景序列）   <- 今天就能开工，零硬件依赖
          ↓ 换
    曹志伟桌面臂 overhead/wrist 相机
          ↓ 换
    真机 RGBD

换 provider 不动 WorldModel 一行代码。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple

from ..types import Detection, RobotPose


class PerceptionProvider(ABC):
    """一次 step 返回：(时间戳, 机器人位姿, 该帧检测列表)"""

    @abstractmethod
    def stream(self) -> Iterator[Tuple[float, RobotPose, List[Detection]]]:
        ...

    def close(self) -> None:
        pass


# ----------------------------------------------------------------- 待实现桩

class CameraDetectorProvider(PerceptionProvider):
    """真实相机 + 检测器。子问题 #1 调包 + #2 地平面求交。

    #1 检测：YOLO / GroundingDINO 现成模型，不自己训。
    #2 定位：检测框底边中点 -> 相机射线 -> 与地平面求交 -> (x, z)。
             全程不需要深度值，透明/反光物体照样能定位。
             前提：物体接地；需标定相机高度与俯仰角。

    TODO: 数据源锁定后接入（曹志伟 SO101 双摄 / 蒋玉月 SG2002 YUYV 帧）。
    """

    def __init__(self, camera_height_m: float, pitch_rad: float, fx: float, fy: float,
                 cx: float, cy: float):
        self.camera_height_m = camera_height_m
        self.pitch_rad = pitch_rad
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy

    def pixel_to_ground(self, u: float, v: float) -> Tuple[float, float]:
        """像素 -> 地平面交点 (x, z)。待标定参数确定后实现。"""
        raise NotImplementedError("等相机标定参数")

    def stream(self):
        raise NotImplementedError("等数据源落点确认")
