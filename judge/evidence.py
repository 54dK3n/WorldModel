"""填 Judge 契约里那个恒为 {} 的 evidence 字段。

判定不能靠技能自报成功，得靠"我看见球现在在桶里"。
所以 evidence 装的是**动作前后两份世界快照的差分**。

JudgeResponse = {success, detail, evidence} —— 这个契约已冻结，只填 evidence。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from world_model.adapters import to_debug_dict
from world_model.types import TrackedObject


def _find(objs: Sequence[TrackedObject], name: str) -> Optional[TrackedObject]:
    cands = [o for o in objs if o.name == name]
    return max(cands, key=lambda o: o.confidence) if cands else None


def _dist_cm(a: TrackedObject, b: TrackedObject) -> float:
    return math.hypot(a.x - b.x, a.z - b.z) * 100.0


def build_containment_evidence(
    before: Sequence[TrackedObject],
    after: Sequence[TrackedObject],
    target_name: str,
    container_name: str,
    now: float,
    threshold_cm: Optional[float] = None,
) -> Dict:
    """判定"target 是否在 container 里"，并给出完整证据链。

    阈值默认取容器半径；也可由调用方覆盖。
    TODO(问宋红): radius_cm 是外接圆半径还是包围盒半宽 —— 直接影响这个阈值。
    """
    t_before = _find(before, target_name)
    t_after = _find(after, target_name)
    c_after = _find(after, container_name)

    warnings: List[str] = []
    evidence: Dict = {
        "verdict_basis": "world_model_diff",
        "target": {
            "name": target_name,
            "before": to_debug_dict(t_before) if t_before else None,
            "after": to_debug_dict(t_after) if t_after else None,
        },
        "container": {
            "name": container_name,
            "after": to_debug_dict(c_after) if c_after else None,
        },
        "relation": None,
        "observations": [],
        "staleness_s": None,
        "warnings": warnings,
    }

    if t_after is None:
        warnings.append(f"目标 {target_name} 在动作后快照中不存在")
        return evidence
    if c_after is None:
        warnings.append(f"容器 {container_name} 在动作后快照中不存在")
        return evidence

    thr = threshold_cm if threshold_cm is not None else c_after.radius_cm
    d = _dist_cm(t_after, c_after)

    evidence["relation"] = {
        "type": "inside",
        "distance_cm": round(d, 2),
        "threshold_cm": round(thr, 2),
        "satisfied": d <= thr,
    }

    # 陈旧度：判定依据有多新。快照太旧不该当作视觉确认。
    staleness = max(t_after.age(now), c_after.age(now))
    evidence["staleness_s"] = round(staleness, 3)
    if staleness > 1.0:
        warnings.append(f"证据陈旧 {staleness:.2f}s，视觉确认不可靠")

    # 位姿不确定度可能吃掉判定余量
    unc = max(t_after.pose_uncertainty_cm, c_after.pose_uncertainty_cm)
    if abs(d - thr) < unc:
        warnings.append(f"距离 {d:.1f}cm 与阈值 {thr:.1f}cm 之差小于位姿不确定度 {unc:.1f}cm")

    for o in (t_after, c_after):
        evidence["observations"].append(
            {
                "name": o.name,
                "frame_id": o.last_frame_id,
                "bbox": list(o.last_bbox) if o.last_bbox else None,
                "confidence": round(o.confidence, 4),
                "source": o.source,
            }
        )

    return evidence


def build_grasp_evidence(
    after: Sequence[TrackedObject],
    target_name: str,
    gripper_closed: bool,
    now: float,
) -> Dict:
    """抓取确认需**双条件**：夹爪反馈 + 视觉确认。

    文档明确要求：不能因为发了抓取命令就断言抓住了。
    """
    t = _find(after, target_name)
    visual_ok = t is not None and t.confidence >= 0.5 and t.age(now) < 1.0
    return {
        "verdict_basis": "gripper_feedback + visual_confirmation",
        "gripper_closed": gripper_closed,
        "visual_confirmed": visual_ok,
        "satisfied": bool(gripper_closed and visual_ok),
        "target": to_debug_dict(t) if t else None,
        "warnings": [] if (gripper_closed and visual_ok) else ["双条件未同时满足，不判定为抓取成功"],
    }
