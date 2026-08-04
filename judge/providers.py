"""判定内核 provider。

外壳（judge_service + HTTP + JudgeRequest/JudgeResponse）已冻结，只填内核。

三条路线：
  WorldModelDiffProvider   本骨架已实现，零依赖，今天就能跑通闭环
  RewardClassifierProvider 主线，LeRobot 自带，与 SmolVLA 同生态  <- 待填
  YoloOverlapProvider      备选，检测框重叠判定
  （VLM 只做早期原型，不进主线）
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from world_model.types import TrackedObject

from .evidence import build_containment_evidence, build_grasp_evidence


# ---------------------------------------------------------------- 冻结契约镜像
# 真实定义在 judge_service 里，此处仅为本地开发镜像，**不要改字段**。

@dataclass
class JudgeRequest:
    task: str                                   # 如 "put_ball_in_basket"
    target: Optional[str] = None
    container: Optional[str] = None
    gripper_closed: bool = False
    now: float = 0.0
    images: List[str] = field(default_factory=list)   # 帧路径/引用，给视觉 provider 用


@dataclass
class JudgeResponse:
    success: bool
    detail: str
    evidence: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------- 基类

class JudgeProvider(ABC):
    name: str = "base"

    @abstractmethod
    def judge(
        self,
        req: JudgeRequest,
        before: Sequence[TrackedObject],
        after: Sequence[TrackedObject],
    ) -> JudgeResponse:
        ...


# ------------------------------------------------------------------- 已实现

class WorldModelDiffProvider(JudgeProvider):
    """基于 World Model 前后快照差分判定。无模型依赖，可立即上线。"""

    name = "world_model_diff"

    def judge(self, req, before, after) -> JudgeResponse:
        if req.task in ("put_in", "put_ball_in_basket", "place"):
            ev = build_containment_evidence(
                before, after, req.target, req.container, req.now
            )
            rel = ev.get("relation")
            ok = bool(rel and rel["satisfied"] and not ev["warnings"])
            detail = (
                f"{req.target} 距 {req.container} {rel['distance_cm']}cm，阈值 {rel['threshold_cm']}cm"
                if rel else "缺少目标或容器观测"
            )
            return JudgeResponse(success=ok, detail=detail, evidence=ev)

        if req.task in ("grasp", "pick"):
            ev = build_grasp_evidence(after, req.target, req.gripper_closed, req.now)
            return JudgeResponse(
                success=ev["satisfied"],
                detail="夹爪+视觉双条件" + ("通过" if ev["satisfied"] else "未通过"),
                evidence=ev,
            )

        return JudgeResponse(False, f"未支持的任务类型: {req.task}", {})


# --------------------------------------------------------------------- 待填

class RewardClassifierProvider(JudgeProvider):
    """主线路线：LeRobot 自带的 RewardClassifier。

    与 SmolVLA 同生态，训练数据可直接用曹志伟的 so101_ball_pick_v1
    带 episode 标注的数据集。

    TODO:
      1. 确认数据源落点（曹志伟双摄 episode / 蒋玉月 SG2002 帧）
      2. 训练集切分与标注口径（success/failure 帧级还是 episode 级）
      3. 推理延迟预算 —— 判定在闭环里，不能拖慢重试节奏
      4. evidence 需回填：分类概率、所用帧引用、检测框
    """

    name = "reward_classifier"

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cpu"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model = None

    def load(self) -> None:
        raise NotImplementedError("等数据源与 checkpoint 确定")

    def judge(self, req, before, after) -> JudgeResponse:
        raise NotImplementedError("等数据源与 checkpoint 确定")


class YoloOverlapProvider(JudgeProvider):
    """备选路线：检测框重叠判定。比 WorldModelDiff 更直接但更脆。"""

    name = "yolo_overlap"

    def judge(self, req, before, after) -> JudgeResponse:
        raise NotImplementedError("备选路线，主线跑通后再评估")


PROVIDERS = {
    p.name: p
    for p in (WorldModelDiffProvider, RewardClassifierProvider, YoloOverlapProvider)
}


def get_provider(name: str = "world_model_diff", **kwargs) -> JudgeProvider:
    if name not in PROVIDERS:
        raise KeyError(f"未知 provider: {name}，可选 {list(PROVIDERS)}")
    return PROVIDERS[name](**kwargs)
