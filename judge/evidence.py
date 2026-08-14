"""填 Judge 契约里那个恒为 {} 的 evidence 字段。

判定不能靠技能自报成功，得靠"我看见球现在在桶里"。
所以 evidence 装的是**动作前后两份世界快照的差分**。

JudgeResponse = {success, detail, evidence} —— 这个契约已冻结，只填 evidence。

--------------------------------------------------------------------------
2026-08-11 判定可靠性复核后的重写。原实现在下面四件事上会给出虚假判定：

  1. 判定对象按 name 取 max(confidence) —— 干扰物、断轨残留、单帧误检
     都可能被抽中，判定结论跟着置信度竞赛走，而不是跟着物理走。
     现在：**用 before 锁定 obj_id，在 after 里按同一个 id 找**，
     找不到同 id 就明确报 identity_broken 并判失败，不静默换一个物体。
  2. satisfied 只看 after —— 动作前就已满足（空动作、球本来就在桶里）
     照样报成功。现在：**要求 before 不满足且 after 满足**。
  3. 护栏是事后 warning，且两条判定路径口径不一致（抓取有置信度下限、
     容器判定一个都没有；状态机从没被查过）。现在：**统一前置条件
     check_quality()，输出机器可读的 reason code，不满足直接判失败**。
  4. 时钟：now 早于 last_seen 时旧 age() 夹成 0，陈旧度护栏静默失效。
     现在：age() 如实返回负值，这里显式检出 clock_skew。

仍未解决（需契约层决策，见 DESIGN.md 待确认清单）：
  契约只有 x/z 没有 y，"球被举在桶正上方"与"球掉进桶里"俯视投影相同。
  本模块的对策是 (a) 夹爪仍闭合时判 still_grasped，(b) 每次 containment
  判定都带 no_height_evidence 提示。真正解决要么加 y，要么加接触判定。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from world_model.adapters import to_debug_dict
from world_model.types import ObjectState, TrackedObject


# ------------------------------------------------------------------ 原因码
# 机器可读，下游按码分支；warnings 里的中文只给人看。

class Reason:
    MISSING_TARGET = "missing_target"
    MISSING_CONTAINER = "missing_container"
    AMBIGUOUS_TARGET = "ambiguous_target"
    AMBIGUOUS_CONTAINER = "ambiguous_container"
    IDENTITY_BROKEN = "identity_broken"
    LOW_CONFIDENCE = "low_confidence"
    UNCONFIRMED_TRACK = "unconfirmed_track"
    STALE_EVIDENCE = "stale_evidence"
    CLOCK_SKEW = "clock_skew"
    GEOMETRY_NOT_SATISFIED = "geometry_not_satisfied"
    NO_STATE_CHANGE = "no_state_change"
    UNCERTAINTY_OVERLAP = "uncertainty_overlap"
    STILL_GRASPED = "still_grasped"
    GRIPPER_OPEN = "gripper_open"
    NO_GRIPPER_POSE = "no_gripper_pose"
    OUT_OF_REACH = "out_of_reach"


# 非阻断提示：判定照给，但下游应知道这条判定的能力边界
class Caveat:
    NO_HEIGHT_EVIDENCE = "no_height_evidence"
    NO_BEFORE_BASELINE = "no_before_baseline"
    HEURISTIC_CONFIDENCE = "heuristic_confidence"


@dataclass
class EvidencePolicy:
    """判定前置条件。两条判定路径（容器 / 抓取）共用同一套口径。"""

    min_confidence: float = 0.50        # 低于此值的信念不构成"视觉确认"
    require_confirmed: bool = True      # TENTATIVE（未达 confirm_hits）不作为证据
    max_staleness_s: float = 1.0        # 证据最多允许多旧
    require_state_change: bool = True   # 必须 before 不满足 -> after 满足
    fully_inside: bool = True           # 阈值扣掉目标半径，要求目标整体进入投影
    grasp_reach_m: float = 0.25         # 目标与夹爪的最大距离
    grasp_grace_s: float = 1.5          # 抓在手里会自遮挡，视觉确认给的宽限窗口


# ------------------------------------------------------------------ 内部工具

def _by_id(objs: Sequence[TrackedObject], obj_id: str) -> Optional[TrackedObject]:
    for o in objs:
        if o.obj_id == obj_id:
            return o
    return None


def _by_name(objs: Sequence[TrackedObject], name: str) -> List[TrackedObject]:
    return sorted([o for o in objs if o.name == name],
                  key=lambda o: o.confidence, reverse=True)


def _dist_cm(a: TrackedObject, b: TrackedObject) -> float:
    return math.hypot(a.x - b.x, a.z - b.z) * 100.0


def resolve_pair(
    before: Sequence[TrackedObject],
    after: Sequence[TrackedObject],
    name: str,
    obj_id: Optional[str],
    role: str,
) -> Tuple[Optional[TrackedObject], Optional[TrackedObject], List[str]]:
    """在 before / after 里定位**同一个**物体。

    这是本次重写的核心：判定的主语必须是一个确定的物体，不能是"叫这个名字的
    东西里置信度最高的那个"。优先用调用方给的 obj_id；没给就用 before 锁定，
    再拿它的 obj_id 去 after 找。after 里没有同 id —— 说明轨迹断了或物体被
    换掉了 —— 如实报 identity_broken，不拿另一个同名物体顶替。
    """
    reasons: List[str] = []
    ambiguous = (Reason.AMBIGUOUS_TARGET if role == "target"
                 else Reason.AMBIGUOUS_CONTAINER)

    if obj_id:
        b = _by_id(before, obj_id)
        a = _by_id(after, obj_id)
        if a is None:
            reasons.append(Reason.IDENTITY_BROKEN)
        return b, a, reasons

    b_cands = _by_name(before, name)
    a_cands = _by_name(after, name)
    if len(b_cands) > 1 or len(a_cands) > 1:
        # 同名多实例：不知道说的是哪一个，只能猜 —— 判定层不猜。
        reasons.append(ambiguous)

    b = b_cands[0] if b_cands else None
    if b is not None:
        a = _by_id(after, b.obj_id)
        if a is None:
            a = a_cands[0] if a_cands else None
            reasons.append(Reason.IDENTITY_BROKEN)
    else:
        a = a_cands[0] if a_cands else None
    return b, a, reasons


def check_quality(
    obj: Optional[TrackedObject],
    now: float,
    policy: EvidencePolicy,
    missing_reason: str,
) -> List[str]:
    """一个对象够不够格充当"视觉确认"。两条判定路径共用。"""
    if obj is None:
        return [missing_reason]

    reasons: List[str] = []
    age = obj.age(now)                       # 可能为负 —— 那是时钟问题
    if age < 0:
        reasons.append(Reason.CLOCK_SKEW)
    elif age > policy.max_staleness_s:
        reasons.append(Reason.STALE_EVIDENCE)
    if obj.confidence < policy.min_confidence:
        reasons.append(Reason.LOW_CONFIDENCE)
    if policy.require_confirmed and obj.state != ObjectState.CONFIRMED:
        reasons.append(Reason.UNCONFIRMED_TRACK)
    return reasons


def _threshold_cm(container: TrackedObject, target: TrackedObject,
                  policy: EvidencePolicy, override: Optional[float]) -> float:
    """判定阈值。

    fully_inside=True 时扣掉目标半径，判据变成"目标整体落在容器投影内"，
    而不是"目标中心刚进圆"（后者会把架在桶沿上的球算成进桶）。
    TODO(问宋红): radius_cm 是外接圆半径还是包围盒半宽 —— 直接影响这个阈值。
    """
    if override is not None:
        return override
    thr = container.radius_cm
    if policy.fully_inside:
        thr = max(thr - target.radius_cm, 0.0)
    return thr


# -------------------------------------------------------------- 容器判定

def build_containment_evidence(
    before: Sequence[TrackedObject],
    after: Sequence[TrackedObject],
    target_name: str,
    container_name: str,
    now: float,
    threshold_cm: Optional[float] = None,
    policy: Optional[EvidencePolicy] = None,
    target_id: Optional[str] = None,
    container_id: Optional[str] = None,
    gripper_closed: bool = False,
) -> Dict:
    """判定"target 是否**被放进**了 container"，并给出完整证据链。

    注意判据是"被放进"而不是"在里面"：动作前就已经在里面的，不算这次动作成功。
    """
    policy = policy or EvidencePolicy()
    reasons: List[str] = []
    caveats: List[str] = [Caveat.NO_HEIGHT_EVIDENCE, Caveat.HEURISTIC_CONFIDENCE]
    warnings: List[str] = []

    t_before, t_after, t_reasons = resolve_pair(before, after, target_name, target_id, "target")
    c_before, c_after, c_reasons = resolve_pair(before, after, container_name, container_id, "container")
    reasons += t_reasons + c_reasons

    evidence: Dict = {
        "verdict_basis": "world_model_diff",
        "verdict": {"success": False, "reasons": reasons, "caveats": caveats},
        "target": {
            "name": target_name,
            "obj_id": t_after.obj_id if t_after else (t_before.obj_id if t_before else None),
            "before": to_debug_dict(t_before) if t_before else None,
            "after": to_debug_dict(t_after) if t_after else None,
        },
        "container": {
            "name": container_name,
            "obj_id": c_after.obj_id if c_after else None,
            "before": to_debug_dict(c_before) if c_before else None,
            "after": to_debug_dict(c_after) if c_after else None,
        },
        "relation": None,
        "observations": [],
        "staleness_s": None,
        "warnings": warnings,
    }

    reasons += check_quality(t_after, now, policy, Reason.MISSING_TARGET)
    reasons += check_quality(c_after, now, policy, Reason.MISSING_CONTAINER)

    if t_after is None or c_after is None:
        _fill_warnings(evidence, target_name, container_name)
        return evidence

    # ---- 几何 ----
    thr = _threshold_cm(c_after, t_after, policy, threshold_cm)
    d_after = _dist_cm(t_after, c_after)
    satisfied_after = d_after <= thr

    d_before: Optional[float] = None
    satisfied_before: Optional[bool] = None
    if t_before is not None and c_before is not None:
        d_before = _dist_cm(t_before, c_before)
        satisfied_before = d_before <= _threshold_cm(c_before, t_before, policy, threshold_cm)
    else:
        caveats.append(Caveat.NO_BEFORE_BASELINE)

    evidence["relation"] = {
        "type": "inside",
        "distance_cm": round(d_after, 2),
        "threshold_cm": round(thr, 2),
        "satisfied": satisfied_after,          # 纯几何，不含护栏
        "before_distance_cm": round(d_before, 2) if d_before is not None else None,
        "before_satisfied": satisfied_before,
    }

    if not satisfied_after:
        reasons.append(Reason.GEOMETRY_NOT_SATISFIED)
    # 动作前就已经满足 -> 这次动作没造成变化，不能算这次成功
    if policy.require_state_change and satisfied_before is True and satisfied_after:
        reasons.append(Reason.NO_STATE_CHANGE)
    # 夹爪还闭着 -> 球多半还在手里悬在桶上方，俯视投影分不出来
    if gripper_closed:
        reasons.append(Reason.STILL_GRASPED)

    # ---- 陈旧度 ----
    staleness = max(t_after.age(now), c_after.age(now))
    evidence["staleness_s"] = round(staleness, 3)

    # ---- 位姿不确定度可能吃掉判定余量 ----
    unc = max(t_after.pose_uncertainty_cm, c_after.pose_uncertainty_cm)
    if abs(d_after - thr) < unc:
        reasons.append(Reason.UNCERTAINTY_OVERLAP)

    for o in (t_after, c_after):
        evidence["observations"].append({
            "name": o.name,
            "obj_id": o.obj_id,
            "frame_id": o.last_frame_id,
            "bbox": list(o.last_bbox) if o.last_bbox else None,
            "confidence": round(o.confidence, 4),
            "state": o.state.value,
            "source": o.source,
        })

    _fill_warnings(evidence, target_name, container_name)
    evidence["verdict"]["success"] = not reasons
    return evidence


_REASON_TEXT = {
    Reason.MISSING_TARGET: "目标 {t} 在动作后快照中不存在",
    Reason.MISSING_CONTAINER: "容器 {c} 在动作后快照中不存在",
    Reason.AMBIGUOUS_TARGET: "场景中存在多个名为 {t} 的对象，无法确定判定主语（请传 target_id）",
    Reason.AMBIGUOUS_CONTAINER: "场景中存在多个名为 {c} 的对象，无法确定判定主语（请传 container_id）",
    Reason.IDENTITY_BROKEN: "动作前后不是同一条轨迹（断轨或对象被替换），无法确认是同一个物体",
    Reason.LOW_CONFIDENCE: "证据置信度低于下限，不构成视觉确认",
    Reason.UNCONFIRMED_TRACK: "轨迹尚未 CONFIRMED（可能是单帧误检），不作为判定证据",
    Reason.STALE_EVIDENCE: "证据过旧，视觉确认不可靠",
    Reason.CLOCK_SKEW: "now 早于观测时刻，调用方时钟与观测时钟不一致，判定作废",
    Reason.GEOMETRY_NOT_SATISFIED: "几何关系不成立",
    Reason.NO_STATE_CHANGE: "动作前该关系就已成立，本次动作未造成状态变化",
    Reason.UNCERTAINTY_OVERLAP: "距离与阈值之差小于位姿不确定度，判定处在噪声里",
    Reason.STILL_GRASPED: "夹爪仍闭合，目标可能还悬在容器上方（契约无 y 轴，投影无法区分）",
    Reason.GRIPPER_OPEN: "夹爪未闭合",
    Reason.NO_GRIPPER_POSE: "未提供夹爪位姿，无法确认目标在手里",
    Reason.OUT_OF_REACH: "目标不在夹爪可及范围内",
}


def _dedupe(items: List[str]) -> None:
    """原地去重保序。目标和容器可能命中同一个原因码（都陈旧），下游不该看到重复。"""
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    items[:] = out


def _fill_warnings(evidence: Dict, target_name: str = "", container_name: str = "") -> None:
    """把 reason code 翻成人话。warnings 只给人看，不参与判定。"""
    _dedupe(evidence["verdict"]["reasons"])
    _dedupe(evidence["verdict"]["caveats"])
    evidence["warnings"][:] = [
        _REASON_TEXT.get(r, r).format(t=target_name, c=container_name)
        for r in evidence["verdict"]["reasons"]
    ]


# -------------------------------------------------------------- 抓取判定

def build_grasp_evidence(
    after: Sequence[TrackedObject],
    target_name: str,
    gripper_closed: bool,
    now: float,
    gripper_pose: Optional[Tuple[float, float]] = None,
    policy: Optional[EvidencePolicy] = None,
    target_id: Optional[str] = None,
) -> Dict:
    """抓取确认需**双条件**：夹爪反馈 + 视觉确认。

    文档明确要求：不能因为发了抓取命令就断言抓住了。

    重写要点：原来的"视觉确认"只检查"这个球存在、够新、conf>=0.5"，
    球在三米外照样算确认 —— 双条件退化成一个自报条件。现在视觉那条要求
    **目标出现在夹爪够得着的位置**，并对抓取姿态下的自遮挡给宽限窗口
    （抓在手里恰恰最容易被夹爪挡住，硬卡 age<1s 会稳定误杀真成功）。
    """
    policy = policy or EvidencePolicy()
    reasons: List[str] = []
    caveats: List[str] = [Caveat.HEURISTIC_CONFIDENCE]

    cands = _by_name(after, target_name)
    t = _by_id(after, target_id) if target_id else (cands[0] if cands else None)
    if target_id is None and len(cands) > 1:
        reasons.append(Reason.AMBIGUOUS_TARGET)

    if not gripper_closed:
        reasons.append(Reason.GRIPPER_OPEN)

    # 视觉确认：够新（含自遮挡宽限）+ 够可信 + 在夹爪够得着的地方
    stale_policy = EvidencePolicy(
        min_confidence=policy.min_confidence,
        require_confirmed=policy.require_confirmed,
        max_staleness_s=policy.grasp_grace_s,
    )
    reasons += check_quality(t, now, stale_policy, Reason.MISSING_TARGET)

    reach_cm: Optional[float] = None
    if t is not None:
        if gripper_pose is None:
            reasons.append(Reason.NO_GRIPPER_POSE)
        else:
            reach_cm = math.hypot(t.x - gripper_pose[0], t.z - gripper_pose[1]) * 100.0
            if reach_cm > policy.grasp_reach_m * 100.0:
                reasons.append(Reason.OUT_OF_REACH)

    visual_ok = t is not None and not [
        r for r in reasons
        if r in (Reason.MISSING_TARGET, Reason.LOW_CONFIDENCE, Reason.UNCONFIRMED_TRACK,
                 Reason.STALE_EVIDENCE, Reason.CLOCK_SKEW, Reason.NO_GRIPPER_POSE,
                 Reason.OUT_OF_REACH)
    ]
    satisfied = bool(gripper_closed and visual_ok and not reasons)

    evidence = {
        "verdict_basis": "gripper_feedback + visual_confirmation",
        "verdict": {"success": satisfied, "reasons": reasons, "caveats": caveats},
        "gripper_closed": gripper_closed,
        "gripper_pose": list(gripper_pose) if gripper_pose else None,
        "visual_confirmed": visual_ok,
        "reach_cm": round(reach_cm, 2) if reach_cm is not None else None,
        "reach_limit_cm": round(policy.grasp_reach_m * 100.0, 2),
        "staleness_s": round(t.age(now), 3) if t else None,
        "satisfied": satisfied,          # 兼容旧字段，与 verdict.success 同值
        "target": to_debug_dict(t) if t else None,
        "warnings": [],
    }
    _fill_warnings(evidence, target_name, "")
    return evidence
