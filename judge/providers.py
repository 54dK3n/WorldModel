"""判定内核 provider。

外壳（judge_service + HTTP + JudgeRequest/JudgeResponse）已冻结，只填内核。

三条路线：
  WorldModelDiffProvider   本骨架已实现，零依赖，今天就能跑通闭环
  RewardClassifierProvider 主线，LeRobot 自带，与 SmolVLA 同生态  <- 待填
  YoloOverlapProvider      备选，检测框重叠判定
  （VLM 只做早期原型，不进主线）
"""
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from world_model.types import TrackedObject

from .evidence import (EvidencePolicy, build_containment_evidence,
                       build_grasp_evidence)


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


# ------------------------------------------------------- 契约外的补充输入
# 判定要做对，需要几样冻结契约里没有的东西。**不动 JudgeRequest**，
# 而是走这个旁路对象传进来 —— 等外壳那边同意扩契约，再并回 JudgeRequest。
#
# 需要外壳补充的字段（复核结论，待与外壳 owner 确认）：
#   target_id / container_id  判定主语的稳定 id，缺了只能靠名字猜（同名多实例即歧义）
#   gripper_pose              夹爪位置，缺了"视觉确认"就退化成"球还在世界上"
#   now 语义                  当前是 float 默认 0.0，无法区分"没传"和"场景零时刻"

@dataclass
class JudgeContext:
    target_id: Optional[str] = None
    container_id: Optional[str] = None
    gripper_pose: Optional[Tuple[float, float]] = None   # 末端 (x, z)，世界系
    policy: Optional[EvidencePolicy] = None


# ---------------------------------------------------------------------- 基类

class JudgeProvider(ABC):
    name: str = "base"

    @abstractmethod
    def judge(
        self,
        req: JudgeRequest,
        before: Sequence[TrackedObject],
        after: Sequence[TrackedObject],
        ctx: Optional[JudgeContext] = None,
    ) -> JudgeResponse:
        ...


CONTAINMENT_TASKS = ("put_in", "put_ball_in_basket", "place")
GRASP_TASKS = ("grasp", "pick")


# ------------------------------------------------------------------- 已实现

class WorldModelDiffProvider(JudgeProvider):
    """基于 World Model 前后快照差分判定。无模型依赖，可立即上线。

    success 只有一个来源：evidence["verdict"]["success"]。
    原实现里 success 走 `satisfied and not warnings`，而 relation.satisfied
    只看距离，同一份响应里能出现两个互相矛盾的结论（下游读哪个字段得到哪个
    答案）。现在 relation.satisfied 明确只表示"几何成立"，判定结论只看
    verdict.success，失败原因在 verdict.reasons 里，detail 里也带上。
    """

    name = "world_model_diff"

    def __init__(self, policy: Optional[EvidencePolicy] = None):
        self.policy = policy or EvidencePolicy()

    def judge(self, req, before, after, ctx: Optional[JudgeContext] = None) -> JudgeResponse:
        ctx = ctx or JudgeContext()
        policy = ctx.policy or self.policy

        if req.task in CONTAINMENT_TASKS:
            ev = build_containment_evidence(
                before, after, req.target, req.container, req.now,
                policy=policy,
                target_id=ctx.target_id,
                container_id=ctx.container_id,
                gripper_closed=req.gripper_closed,
            )
            return JudgeResponse(
                success=ev["verdict"]["success"],
                detail=self._detail(ev, f"{req.target} -> {req.container}"),
                evidence=ev,
            )

        if req.task in GRASP_TASKS:
            ev = build_grasp_evidence(
                after, req.target, req.gripper_closed, req.now,
                gripper_pose=ctx.gripper_pose,
                policy=policy,
                target_id=ctx.target_id,
            )
            return JudgeResponse(
                success=ev["verdict"]["success"],
                detail=self._detail(ev, f"抓取 {req.target}"),
                evidence=ev,
            )

        return JudgeResponse(False, f"未支持的任务类型: {req.task}", {})

    @staticmethod
    def _detail(ev: Dict, subject: str) -> str:
        """detail 必须解释结论 —— 成功说依据，失败说原因，不能只报个距离。"""
        rel = ev.get("relation")
        geo = (f"距离 {rel['distance_cm']}cm / 阈值 {rel['threshold_cm']}cm"
               if rel else None)
        if ev["verdict"]["success"]:
            return f"{subject} 判定成立" + (f"（{geo}）" if geo else "")
        head = f"{subject} 判定不成立"
        if geo:
            head += f"（{geo}）"
        if ev.get("warnings"):
            head += "：" + "；".join(ev["warnings"])
        return head


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

    def judge(self, req, before, after, ctx=None) -> JudgeResponse:
        # 不抛异常：外壳是已冻结的 HTTP 服务，抛出去就是 500 而不是"判定不通过"。
        return _not_implemented(self.name, "等数据源与 checkpoint 确定")


class YoloOverlapProvider(JudgeProvider):
    """备选路线：检测框重叠判定。比 WorldModelDiff 更直接但更脆。"""

    name = "yolo_overlap"

    def judge(self, req, before, after, ctx=None) -> JudgeResponse:
        return _not_implemented(self.name, "备选路线，主线跑通后再评估")


def _not_implemented(name: str, why: str) -> JudgeResponse:
    return JudgeResponse(
        success=False,
        detail=f"provider {name} 尚未实现：{why}",
        evidence={
            "verdict_basis": name,
            "verdict": {"success": False, "reasons": ["provider_not_implemented"], "caveats": []},
            "warnings": [f"provider {name} 尚未实现：{why}"],
        },
    )


PROVIDERS = {
    p.name: p
    for p in (WorldModelDiffProvider, RewardClassifierProvider, YoloOverlapProvider)
}


def get_provider(name: str = "world_model_diff", **kwargs) -> JudgeProvider:
    """按名取 provider。

    kwargs 只透传给认得的构造参数 —— 原来无差别透传，
    给 world_model_diff 传个 checkpoint_path 会直接 TypeError。
    """
    if name not in PROVIDERS:
        raise KeyError(f"未知 provider: {name}，可选 {list(PROVIDERS)}")
    cls = PROVIDERS[name]
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    ignored = sorted(set(kwargs) - accepted)
    if ignored:
        # 静默吞掉参数比报错更坑，这里明确告知
        raise TypeError(f"provider {name} 不接受参数 {ignored}，它只认 {sorted(accepted)}")
    return cls(**kwargs)
