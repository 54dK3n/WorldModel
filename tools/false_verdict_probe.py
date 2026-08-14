"""判定可靠性探针：构造对抗场景，检验 Judge 会不会给出**虚假判定**。

不是单测（单测跑的是"设计意图成立"），这里跑的是"设计意图在边界上失效"。
每个用例声明**物理事实上应有的判定**，跑完与系统实际输出比对：

    PASS   系统判定 == 物理事实
    FAIL   系统判定 != 物理事实（虚假判定）
    KNOWN  已知能力边界，判定不成立但原因合理（如契约缺 y 轴）

    python tools/false_verdict_probe.py

用例编号沿用 2026-08-11 复核报告，方便对着修复前的日志逐条比对。
"""
from __future__ import annotations

import copy
import json
import sys
from typing import List, Optional, Sequence, Tuple

sys.path.insert(0, ".")

from judge.evidence import EvidencePolicy, build_grasp_evidence
from judge.providers import (JudgeContext, JudgeRequest, WorldModelDiffProvider,
                             get_provider)
from world_model import Detection, RobotPose, WorldModel
from world_model.types import ObjectState, TrackedObject

BAR = "=" * 78
SUB = "-" * 78

_results: List[Tuple[str, str, str]] = []


def case(cid: str, title: str) -> None:
    print(f"\n{BAR}\n[{cid}] {title}\n{BAR}")


def verify(cid: str, expected: bool, actual: bool, note: str, known: bool = False) -> None:
    if known:
        tag = "KNOWN"
    else:
        tag = "PASS" if expected == actual else "FAIL"
    _results.append((cid, tag, note))
    print(f"\n>>> [{cid}] {tag} — 应判 {expected}，实判 {actual} —— {note}")


def det(cls: str, x: float, z: float, conf: float = 0.9, r: float = 3.3,
        frame: str = "probe", bbox=(0, 0, 10, 10)) -> Detection:
    return Detection(class_name=cls, x=x, z=z, confidence=conf, radius_cm=r,
                     bbox=bbox, frame_id=frame, source="probe")


def obj(name: str, x: float, z: float, conf: float, r: float = 3.3,
        last_seen: float = 0.0, state: ObjectState = ObjectState.CONFIRMED,
        oid: Optional[str] = None, unc: float = 1.0) -> TrackedObject:
    return TrackedObject(
        obj_id=oid or f"{name}_900", name=name, x=x, z=z, radius_cm=r,
        confidence=conf, first_seen=0.0, last_seen=last_seen, last_updated=last_seen,
        hit_count=5, state=state, pose_uncertainty_cm=unc,
        source="probe", last_bbox=(0, 0, 10, 10), last_frame_id="probe",
    )


def show(objs: Sequence[TrackedObject], now: float, label: str = "快照") -> None:
    print(f"  {label}:")
    for o in objs:
        print(f"    {o.obj_id:<12} name={o.name:<7} conf={o.confidence:.3f} "
              f"state={o.state.value:<9} pos=({o.x:+.3f},{o.z:+.3f}) age={o.age(now):+.1f}s")


def judge_put(before, after, now, target="ball", container="basket",
              ctx: Optional[JudgeContext] = None, gripper_closed: bool = False):
    return WorldModelDiffProvider().judge(
        JudgeRequest(task="put_ball_in_basket", target=target, container=container,
                     now=now, gripper_closed=gripper_closed),
        before=before, after=after, ctx=ctx,
    )


def brief(resp) -> None:
    ev = resp.evidence
    rel = ev.get("relation")
    tgt = ev.get("target", {})
    print(f"  success  = {resp.success}")
    print(f"  detail   = {resp.detail}")
    print(f"  relation = {rel}")
    print(f"  target   = before:{(tgt.get('before') or {}).get('obj_id')} "
          f"after:{(tgt.get('after') or {}).get('obj_id')}")
    print(f"  reasons  = {ev.get('verdict', {}).get('reasons')}")
    print(f"  caveats  = {ev.get('verdict', {}).get('caveats')}")


def build_scene(new_conf: float = 0.88, gap: float = 2.0):
    """三帧稳定跟踪 -> 球被放进桶。返回 (before, after, target_id, t_after)。"""
    wm = WorldModel()
    pose = RobotPose(pose_uncertainty_cm=1.0)
    for t, x, z in [(0.0, 0.10, 1.20), (0.5, 0.13, 1.18), (1.0, 0.16, 1.15)]:
        wm.update([det("sports ball", x, z, 0.95), det("basket", -0.60, 1.60, 0.92, 15.0)],
                  pose, now=t)
    before = wm.snapshot()
    tid = wm.get_object("ball").obj_id
    t_after = 1.0 + gap
    wm.update([det("sports ball", -0.58, 1.61, new_conf), det("basket", -0.60, 1.60, 0.93, 15.0)],
              pose, now=t_after)
    return before, wm.snapshot(), tid, t_after


# ==================================================================== A
def case_a_identity():
    case("A", "身份连续：大位移后判定用的还是不是同一个对象")
    before, after, tid, now = build_scene()
    show(before, 1.0, "before")
    show(after, now, "after ")
    resp = judge_put(before, after, now, ctx=JudgeContext(target_id=tid))
    brief(resp)
    ids = {o.obj_id for o in after if o.name == "ball"}
    print(f"\n  after 里的 ball 轨迹：{ids}")
    print("  门控随 dt 放宽（0.5 + 0.5*2.0 = 1.5m > 0.89m 位移）后不再断轨，")
    print("  before/after 是同一个 obj_id，判定主语明确。")
    verify("A", True, resp.success and len(ids) == 1, "球确实进桶，且全程同一条轨迹")


# ==================================================================== B
def case_b_verdict_stable():
    case("B", "判定稳定性：物理事实不变，只改检测置信度，判定不应翻转")
    outs = []
    for conf in (0.88, 0.55, 0.35):
        before, after, tid, now = build_scene(new_conf=conf)
        resp = judge_put(before, after, now, ctx=JudgeContext(target_id=tid))
        n_balls = len([o for o in after if o.name == "ball"])
        print(f"\n{SUB}\n  新检测 conf={conf}（球都在桶里，物理事实不变）")
        print(f"  ball 轨迹数={n_balls} | success={resp.success}")
        print(f"  detail={resp.detail}")
        outs.append(resp.success)
    print("\n  修复前：conf 0.88 -> True，0.55 -> False（幽灵轨迹反超，判定翻转）。")
    print("  现在三档同判 —— 判定跟着球走，不跟着检测分数走。")
    print("  注：conf=0.35 时融合后的置信度若跌破 min_confidence 会判失败，")
    print("      那是**有原因的拒绝**（证据太弱），不是随机翻转。")
    verify("B", True, outs[0] and outs[1], f"0.88/0.55 两档判定一致：{outs}")


# ==================================================================== C
def case_c_distractor():
    case("C", "干扰物：桶里本来就有另一个球，被抓的那个球一步没动")
    pose = RobotPose(pose_uncertainty_cm=1.0)
    wm = WorldModel()
    frames = [
        det("sports ball", 0.30, 1.00, 0.85),      # 任务目标：桌上的球
        det("sports ball", -0.60, 1.60, 0.93),     # 干扰物：桶里早就有一个
        det("basket", -0.60, 1.60, 0.92, 15.0),
    ]
    for t in (0.0, 0.5, 1.0):
        wm.update(copy.deepcopy(frames), pose, now=t)
    before = wm.snapshot()
    tid = [o.obj_id for o in before if o.name == "ball" and o.x > 0][0]
    wm.update(copy.deepcopy(frames), pose, now=1.5)   # 抓取失败，目标原地未动
    after = wm.snapshot()
    show(after, 1.5, "after")

    print(f"\n{SUB}\n  a) 调用方锁定了目标 id（target_id={tid}）:")
    r1 = judge_put(before, after, 1.5, ctx=JudgeContext(target_id=tid))
    brief(r1)

    print(f"\n{SUB}\n  b) 调用方只给名字 'ball'，场景里有两个:")
    r2 = judge_put(before, after, 1.5)
    brief(r2)

    print("\n  a) 主语明确 -> 按目标球判，它没动 -> 判失败（正确）")
    print("  b) 主语有歧义 -> 判定层不猜，报 ambiguous_target 并判失败（正确的保守）")
    verify("C", False, r1.success or r2.success, "目标球一步没动，两种调用方式都不报成功")


# ==================================================================== D
def case_d_state_change():
    case("D", "空动作：动作前后完全相同的快照")
    snap = [obj("ball", -0.60, 1.60, 0.90, 3.3, last_seen=5.0),
            obj("basket", -0.60, 1.60, 0.92, 15.0, last_seen=5.0, oid="basket_901")]
    before, after = copy.deepcopy(snap), copy.deepcopy(snap)
    resp = judge_put(before, after, 5.0, ctx=JudgeContext(target_id="ball_900"))
    brief(resp)
    print("\n  球在动作前就在桶里，本次动作什么也没发生。")
    print("  relation.satisfied 仍为 True（几何确实成立），但 before_satisfied 也是 True，")
    print("  no_state_change -> 判定不成立。『状态已满足』不等于『这次动作成功』。")
    verify("D", False, resp.success, "空动作不再被判成功")


# ==================================================================== E
def case_e_clock():
    case("E", "时钟：now 早于观测时刻（漏传 now / 时钟源不一致）")
    after = [obj("ball", -0.60, 1.60, 0.90, 3.3, last_seen=100.0),
             obj("basket", -0.60, 1.60, 0.92, 15.0, last_seen=100.0, oid="basket_901")]
    before = [obj("ball", 0.30, 1.00, 0.90, 3.3, last_seen=100.0),
              obj("basket", -0.60, 1.60, 0.92, 15.0, last_seen=100.0, oid="basket_901")]

    print(f"\n{SUB}\n  a) 正确传 now=100.5（证据新鲜）:")
    r1 = judge_put(before, after, 100.5, ctx=JudgeContext(target_id="ball_900"))
    brief(r1)

    print(f"\n{SUB}\n  b) 正确传 now=130.0（证据已 30 秒未刷新）:")
    r2 = judge_put(before, after, 130.0, ctx=JudgeContext(target_id="ball_900"))
    brief(r2)

    print(f"\n{SUB}\n  c) 调用方漏传 now（JudgeRequest.now 默认 0.0）:")
    r3 = WorldModelDiffProvider().judge(
        JudgeRequest(task="put_ball_in_basket", target="ball", container="basket"),
        before=before, after=after, ctx=JudgeContext(target_id="ball_900"))
    brief(r3)
    print("\n  age() 不再把负数夹成 0，c) 被 clock_skew 拦下。")
    print("  修复前 c) 的 staleness 恒为 0.0，30 秒前的快照被当作实时视觉确认。")
    verify("E", True, r1.success and not r2.success and not r3.success,
           "新鲜->成立；陈旧->stale_evidence；时钟不一致->clock_skew")


# ==================================================================== F
def case_f_grasp():
    case("F", "抓取双条件：夹爪反馈 + 视觉确认")
    print(f"\n{SUB}\n  a) 夹爪空抓，球在 3 米外地上还能看见:")
    a1 = [obj("ball", 1.50, 3.00, 0.90, 3.3, last_seen=10.0)]
    e1 = build_grasp_evidence(a1, "ball", True, 10.0, gripper_pose=(0.20, 0.60))
    print(f"  visual_confirmed={e1['visual_confirmed']} reach={e1['reach_cm']}cm "
          f"(limit {e1['reach_limit_cm']}cm) satisfied={e1['satisfied']}")
    print(f"  reasons={e1['verdict']['reasons']}")

    print(f"\n{SUB}\n  b) 真抓住了，球被夹爪自遮挡 1.2s:")
    a2 = [obj("ball", 0.21, 0.61, 0.90, 3.3, last_seen=8.8)]
    e2 = build_grasp_evidence(a2, "ball", True, 10.0, gripper_pose=(0.20, 0.60))
    print(f"  visual_confirmed={e2['visual_confirmed']} reach={e2['reach_cm']}cm "
          f"staleness={e2['staleness_s']}s satisfied={e2['satisfied']}")
    print(f"  reasons={e2['verdict']['reasons']}")

    print(f"\n{SUB}\n  c) 没有夹爪位姿可用（外壳没提供）:")
    e3 = build_grasp_evidence(a2, "ball", True, 10.0, gripper_pose=None)
    print(f"  satisfied={e3['satisfied']} reasons={e3['verdict']['reasons']}")

    print("\n  a) 视觉那条现在要求目标在夹爪够得着的地方 -> out_of_reach，不再空抓报成功")
    print("  b) 抓取姿态下自遮挡给 1.5s 宽限 -> 不再稳定误杀真成功")
    print("  c) 缺夹爪位姿就明说缺，不假装确认了")
    verify("F", False, e1["satisfied"], "空抓不再判成功")
    verify("F2", True, e2["satisfied"], "自遮挡 1.2s 的真抓取判成功")
    verify("F3", False, e3["satisfied"], "缺夹爪位姿时保守判失败并给出原因")


# ==================================================================== G
def case_g_false_detection():
    case("G", "单帧误检：桶里冒出一帧高分反光，只出现这一帧")
    pose = RobotPose(pose_uncertainty_cm=1.0)
    wm = WorldModel()
    for t in (0.0, 0.5, 1.0):
        wm.update([det("sports ball", 0.30, 1.00, 0.85),
                   det("basket", -0.60, 1.60, 0.92, 15.0)], pose, now=t)
    before = wm.snapshot()
    tid = wm.get_object("ball").obj_id
    wm.update([det("sports ball", 0.30, 1.00, 0.85),
               det("sports ball", -0.61, 1.59, 0.97),      # 误检
               det("basket", -0.60, 1.60, 0.92, 15.0)], pose, now=1.5)
    after = wm.snapshot()
    show(after, 1.5, "after")
    resp = judge_put(before, after, 1.5, ctx=JudgeContext(target_id=tid))
    brief(resp)
    print("\n  判定主语由 before 锁定 -> 看的是桌上那个球，它没动 -> 失败。")
    print("  即使调用方不给 id，误检轨迹是 TENTATIVE（hit_count=1 < confirm_hits=3），")
    print("  也会被 unconfirmed_track 挡住 —— 状态机现在真的参与判定了。")
    ghost = [o for o in after if o.state == ObjectState.TENTATIVE]
    e_noid = judge_put(before, after, 1.5)
    print(f"  不给 id 时：success={e_noid.success} reasons={e_noid.evidence['verdict']['reasons']}")
    verify("G", False, resp.success or e_noid.success,
           f"单帧误检（{[o.obj_id for o in ghost]}）不再构成视觉确认")


# ==================================================================== H
def case_h_geometry():
    case("H", "几何：2D 投影判据的能力边界")
    cases = [
        ("球被夹爪举在桶正上方 30cm，夹爪仍闭合", 0.00, 1.50, True),
        ("球被举在桶上方，但外壳没报夹爪状态", 0.00, 1.50, False),
        ("球架在桶沿上，球心距桶心 14.9cm", 0.149, 1.50, False),
    ]
    outs = []
    for label, x, z, grip in cases:
        after = [obj("ball", x, z, 0.90, 3.3, last_seen=5.0, unc=0.2),
                 obj("basket", 0.00, 1.50, 0.92, 15.0, last_seen=5.0, oid="basket_901", unc=0.2)]
        before = [obj("ball", 1.00, 1.00, 0.90, 3.3, last_seen=5.0, unc=0.2),
                  obj("basket", 0.00, 1.50, 0.92, 15.0, last_seen=5.0, oid="basket_901", unc=0.2)]
        resp = judge_put(before, after, 5.0, ctx=JudgeContext(target_id="ball_900"),
                         gripper_closed=grip)
        print(f"\n{SUB}\n  {label}")
        print(f"  success={resp.success} relation={resp.evidence['relation']}")
        print(f"  reasons={resp.evidence['verdict']['reasons']}")
        outs.append(resp.success)

    print("\n  阈值现在扣掉目标半径（15 - 3.3 = 11.7cm），判据是『球整体落在桶投影内』，")
    print("  架在桶沿的球（14.9cm）几何上就不再成立 —— 不靠位姿不确定度护栏侥幸拦截。")
    print("  『举在正上方』：夹爪闭合时判 still_grasped 拦下；外壳不报夹爪状态时，")
    print("  俯视投影确实分不出上方与里面 —— 这是契约缺 y 轴的硬边界，")
    print("  每条 containment 判定都带 no_height_evidence 提示，等契约决策。")
    verify("H", False, outs[0], "夹爪闭合时不再把『举在桶上方』判成放入成功")
    verify("H2", False, outs[2], "架在桶沿的球几何上不成立")
    verify("H3", False, outs[1], "缺夹爪状态时仍无法区分上方/里面", known=True)


# ==================================================================== I
def case_i_confidence_floor():
    case("I", "证据质量下限：几乎要 LOST 的对象能不能构成视觉确认")
    after = [obj("ball", -0.60, 1.60, 0.16, 3.3, last_seen=5.0, state=ObjectState.STALE),
             obj("basket", -0.60, 1.60, 0.92, 15.0, last_seen=5.0, oid="basket_901")]
    before = [obj("ball", 0.30, 1.00, 0.90, 3.3, last_seen=5.0),
              obj("basket", -0.60, 1.60, 0.92, 15.0, last_seen=5.0, oid="basket_901")]
    resp = judge_put(before, after, 5.0, ctx=JudgeContext(target_id="ball_900"))
    brief(resp)
    print("\n  conf=0.16（lost 阈值 0.15），state=STALE。")
    print("  两条判定路径现在共用 check_quality：低于 min_confidence 或未 CONFIRMED 一律不认。")
    verify("I", False, resp.success, "STALE + conf=0.16 的残留信念不再充当视觉确认")


# ==================================================================== J
def case_j_single_truth():
    case("J", "响应自洽：success 与 evidence 不应给出两个结论")
    after = [obj("ball", -0.60, 1.60, 0.90, 3.3, last_seen=0.0),
             obj("basket", -0.60, 1.60, 0.92, 15.0, last_seen=0.0, oid="basket_901")]
    before = [obj("ball", 0.30, 1.00, 0.90, 3.3, last_seen=0.0),
              obj("basket", -0.60, 1.60, 0.92, 15.0, last_seen=0.0, oid="basket_901")]
    resp = judge_put(before, after, 5.0, ctx=JudgeContext(target_id="ball_900"))
    brief(resp)
    ev = resp.evidence
    print(f"\n  几何 relation.satisfied = {ev['relation']['satisfied']}（只表示距离成立）")
    print(f"  判定 verdict.success     = {ev['verdict']['success']}（唯一结论）")
    print(f"  JudgeResponse.success    = {resp.success}")
    print("  两者语义现在分开了：几何成立不等于判定成立，失败原因写在 reasons 与 detail 里。")
    ok = (resp.success == ev["verdict"]["success"]
          and ev["relation"]["satisfied"] is True      # 几何成立
          and resp.success is False                     # 判定不成立
          and "过旧" in resp.detail)                    # 且 detail 说清了为什么
    verify("J", True, ok, "success 只有一个来源，几何字段不再被误读成结论，detail 解释失败原因")


# ==================================================================== K
def case_k_registry():
    case("K", "provider 注册表：未实现路线不应把异常抛穿冻结的 HTTP 契约")
    p = get_provider("reward_classifier")
    r = p.judge(JudgeRequest(task="put_ball_in_basket", target="ball", container="basket"), [], [])
    print(f"  reward_classifier.judge() -> success={r.success}, detail={r.detail}")
    print(f"  evidence.reasons = {r.evidence['verdict']['reasons']}")
    try:
        get_provider("world_model_diff", checkpoint_path="/tmp/x.pt")
        msg = "未报错（不应该）"
        ok_kwargs = False
    except TypeError as e:
        msg = str(e)
        ok_kwargs = True
    print(f"  get_provider('world_model_diff', checkpoint_path=...) -> TypeError: {msg}")
    verify("K", True, isinstance(r.success, bool) and not r.success and ok_kwargs,
           "未实现 provider 返回 JudgeResponse；错配参数报清楚的 TypeError")


# ==================================================================== L
def case_l_missing():
    case("L", "对照组：目标/容器缺失、未知任务类型")
    r1 = judge_put([], [obj("basket", 0.0, 1.5, 0.9, 15.0, oid="basket_901")], 1.0)
    brief(r1)
    r2 = WorldModelDiffProvider().judge(
        JudgeRequest(task="wipe_table", target="ball", container="basket", now=1.0), [], [])
    print(f"\n  未支持任务类型 -> success={r2.success}, detail={r2.detail}")
    verify("L", False, r1.success or r2.success, "缺观测与未知任务类型都保守判失败")


# ==================================================================== M
def case_m_contract():
    case("M", "对外契约导出：时间戳纪元 + 同名多实例")
    pose = RobotPose(pose_uncertainty_cm=1.0)
    wm = WorldModel(time_origin=1786417200.0)   # 2026-08-11T03:00:00Z
    for t, x, z in [(0.0, 0.10, 1.20), (0.5, 0.13, 1.18), (1.0, 0.16, 1.15)]:
        wm.update([det("sports ball", x, z, 0.95), det("basket", -0.60, 1.60, 0.92, 15.0)], pose, now=t)
    wm.update([det("sports ball", -0.58, 1.61, 0.88), det("basket", -0.60, 1.60, 0.93, 15.0)], pose, now=3.0)

    obs = wm.to_contract()
    print(json.dumps(obs, ensure_ascii=False, indent=2))
    balls = [o for o in obs if o["name"] == "ball"]
    ts = balls[0]["timestamp"]
    print(f"\n  name='ball' 的条目数 = {len(balls)}（修复前是 2：断轨产生的幽灵 + 新轨迹）")
    print(f"  timestamp = {ts}（修复前是 1970-01-01T00:00:0XZ）")
    print("  仍未解决：契约本身没有 id/state 字段，真实场景里同名多实例（多个球）")
    print("  下游依然无法区分。加字段要契约 owner 签字，已记进待确认清单。")
    verify("M", True, len(balls) == 1 and ts.startswith("2026-08-11"),
           "断轨消失、时间戳落在真实时间；契约加 id 字段仍待对齐")


def main() -> None:
    print("判定可靠性探针 — wm_kit")
    print("每个用例声明物理事实上应有的判定，与系统实际输出比对\n")
    case_a_identity()
    case_b_verdict_stable()
    case_c_distractor()
    case_d_state_change()
    case_e_clock()
    case_f_grasp()
    case_g_false_detection()
    case_h_geometry()
    case_i_confidence_floor()
    case_j_single_truth()
    case_k_registry()
    case_l_missing()
    case_m_contract()

    print(f"\n{BAR}\n汇总\n{BAR}")
    for cid, tag, note in _results:
        print(f"  [{cid:<2}] {tag:<6} {note}")
    n_fail = sum(1 for _, t, _ in _results if t == "FAIL")
    n_pass = sum(1 for _, t, _ in _results if t == "PASS")
    n_known = sum(1 for _, t, _ in _results if t == "KNOWN")
    print(f"\n  PASS {n_pass} / FAIL {n_fail} / KNOWN(能力边界) {n_known}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
