"""判定内核回归单测。

每条都对应 2026-08-11 判定可靠性复核里的一个虚假判定用例。
这些场景当时全都能稳定复现，修完就是现成的验收条件 —— 别让它们回来。

    A/B  身份连续：断轨导致判定在两条同名轨迹间二选一
    C    干扰物：桶里本来就有另一个球
    D    空动作：动作前后快照相同也报成功
    E    时钟：now 早于观测时刻时陈旧度护栏静默失效
    F    抓取：空抓 + 远处看得见球 = 成功
    G    误检：单帧 TENTATIVE 轨迹充当视觉确认
    I    证据质量：STALE + conf=0.16 也算视觉确认
    J    响应自洽：success 与 relation.satisfied 互相矛盾
"""
import copy

import pytest

from judge.evidence import (Caveat, EvidencePolicy, Reason,
                            build_containment_evidence, build_grasp_evidence)
from judge.providers import (JudgeContext, JudgeRequest, WorldModelDiffProvider,
                             get_provider)
from world_model import Detection, RobotPose, WorldModel
from world_model.types import ObjectState, TrackedObject


def det(cls="sports ball", x=0.0, z=1.0, conf=0.95, r=3.3):
    return Detection(class_name=cls, x=x, z=z, confidence=conf, radius_cm=r,
                     bbox=(0, 0, 10, 10), frame_id="f", source="test")


def obj(name, x, z, conf, r=3.3, last_seen=0.0, state=ObjectState.CONFIRMED,
        oid=None, unc=1.0):
    return TrackedObject(
        obj_id=oid or f"{name}_900", name=name, x=x, z=z, radius_cm=r,
        confidence=conf, first_seen=0.0, last_seen=last_seen, last_updated=last_seen,
        hit_count=5, state=state, pose_uncertainty_cm=unc,
        source="test", last_bbox=(0, 0, 10, 10), last_frame_id="f",
    )


def put_scene(new_conf=0.88, gap=2.0):
    """三帧稳定跟踪 -> 球被放进桶（位移 0.89m，超过基础门控 0.5m）。"""
    wm = WorldModel()
    pose = RobotPose(pose_uncertainty_cm=1.0)
    for t, x, z in [(0.0, 0.10, 1.20), (0.5, 0.13, 1.18), (1.0, 0.16, 1.15)]:
        wm.update([det(x=x, z=z), det(cls="basket", x=-0.60, z=1.60, conf=0.92, r=15.0)],
                  pose, now=t)
    before = wm.snapshot()
    tid = wm.get_object("ball").obj_id
    now = 1.0 + gap
    wm.update([det(x=-0.58, z=1.61, conf=new_conf),
               det(cls="basket", x=-0.60, z=1.60, conf=0.93, r=15.0)], pose, now=now)
    return before, wm.snapshot(), tid, now


def judge(before, after, now, ctx=None, gripper_closed=False,
          task="put_ball_in_basket"):
    return WorldModelDiffProvider().judge(
        JudgeRequest(task=task, target="ball", container="basket",
                     now=now, gripper_closed=gripper_closed),
        before=before, after=after, ctx=ctx,
    )


# --------------------------------------------------- A/B 身份连续与判定稳定

def test_large_displacement_keeps_track_identity():
    """dt 自适应门控：2 秒 0.89m 的合法位移不应断轨。"""
    _, after, tid, _ = put_scene()
    balls = [o for o in after if o.name == "ball"]
    assert len(balls) == 1, "断轨会产生幽灵轨迹，判定层就得在两条轨迹间二选一"
    assert balls[0].obj_id == tid, "obj_id 必须保持不变"


def test_put_in_basket_succeeds():
    before, after, tid, now = put_scene()
    resp = judge(before, after, now, ctx=JudgeContext(target_id=tid))
    assert resp.success is True
    assert resp.evidence["relation"]["before_satisfied"] is False
    assert resp.evidence["relation"]["satisfied"] is True


@pytest.mark.parametrize("conf", [0.88, 0.70, 0.55])
def test_verdict_does_not_flip_with_detection_score(conf):
    """物理事实不变，只改检测置信度，判定不能翻转。

    修复前：0.88 -> True，0.55 -> False（衰减中的幽灵轨迹反超新轨迹）。
    """
    before, after, tid, now = put_scene(new_conf=conf)
    resp = judge(before, after, now, ctx=JudgeContext(target_id=tid))
    assert resp.success is True, f"conf={conf} 时判定翻转了"


def test_position_not_left_halfway_after_big_move():
    """固定 EMA 会把位置留在起终点之间，直接把 containment 拖成假阴性。"""
    _, after, tid, _ = put_scene()
    ball = [o for o in after if o.obj_id == tid][0]
    assert abs(ball.x - (-0.58)) < 0.05 and abs(ball.z - 1.61) < 0.05


def test_identity_broken_is_reported_not_papered_over():
    """after 里没有同 id 对象时，不能拿另一个同名物体顶替。"""
    before = [obj("ball", 0.3, 1.0, 0.9), obj("basket", -0.6, 1.6, 0.92, 15.0, oid="basket_901")]
    after = [obj("ball", -0.6, 1.6, 0.9, oid="ball_777"),   # 另一条轨迹
             obj("basket", -0.6, 1.6, 0.92, 15.0, oid="basket_901")]
    ev = build_containment_evidence(before, after, "ball", "basket", now=0.5)
    assert Reason.IDENTITY_BROKEN in ev["verdict"]["reasons"]
    assert ev["verdict"]["success"] is False


# ------------------------------------------------------------------- C 干扰物

def test_distractor_in_container_does_not_count():
    """桶里本来就有另一个球，目标球一步没动 -> 不能报成功。"""
    pose = RobotPose(pose_uncertainty_cm=1.0)
    wm = WorldModel()
    frame = [det(x=0.30, z=1.00, conf=0.85),                     # 目标：桌上
             det(x=-0.60, z=1.60, conf=0.93),                    # 干扰物：桶里
             det(cls="basket", x=-0.60, z=1.60, conf=0.92, r=15.0)]
    for t in (0.0, 0.5, 1.0):
        wm.update(copy.deepcopy(frame), pose, now=t)
    before = wm.snapshot()
    tid = [o.obj_id for o in before if o.name == "ball" and o.x > 0][0]
    wm.update(copy.deepcopy(frame), pose, now=1.5)
    after = wm.snapshot()

    assert judge(before, after, 1.5, ctx=JudgeContext(target_id=tid)).success is False


def test_ambiguous_target_is_refused_not_guessed():
    """同名多实例且调用方没给 id -> 报歧义并判失败，不按置信度猜。"""
    before = [obj("ball", 0.3, 1.0, 0.85, oid="ball_1"),
              obj("ball", -0.6, 1.6, 0.93, oid="ball_2"),
              obj("basket", -0.6, 1.6, 0.92, 15.0, oid="basket_9")]
    ev = build_containment_evidence(before, copy.deepcopy(before), "ball", "basket", now=0.5)
    assert Reason.AMBIGUOUS_TARGET in ev["verdict"]["reasons"]
    assert ev["verdict"]["success"] is False


# --------------------------------------------------------------- D 状态变化

def test_no_state_change_is_not_success():
    """动作前就在桶里 -> 本次动作没造成变化，不算成功。"""
    snap = [obj("ball", -0.6, 1.6, 0.9, last_seen=5.0),
            obj("basket", -0.6, 1.6, 0.92, 15.0, last_seen=5.0, oid="basket_901")]
    resp = judge(copy.deepcopy(snap), copy.deepcopy(snap), 5.0,
                 ctx=JudgeContext(target_id="ball_900"))
    assert resp.success is False
    assert Reason.NO_STATE_CHANGE in resp.evidence["verdict"]["reasons"]
    # 几何关系本身仍然成立 —— 这两件事必须分开表达
    assert resp.evidence["relation"]["satisfied"] is True


# ------------------------------------------------------------------- E 时钟

def _fresh_pair(last_seen=100.0):
    before = [obj("ball", 0.3, 1.0, 0.9, last_seen=last_seen),
              obj("basket", -0.6, 1.6, 0.92, 15.0, last_seen=last_seen, oid="basket_901")]
    after = [obj("ball", -0.6, 1.6, 0.9, last_seen=last_seen),
             obj("basket", -0.6, 1.6, 0.92, 15.0, last_seen=last_seen, oid="basket_901")]
    return before, after


def test_fresh_evidence_passes():
    before, after = _fresh_pair()
    assert judge(before, after, 100.5, ctx=JudgeContext(target_id="ball_900")).success is True


def test_stale_evidence_rejected():
    before, after = _fresh_pair()
    resp = judge(before, after, 130.0, ctx=JudgeContext(target_id="ball_900"))
    assert resp.success is False
    assert Reason.STALE_EVIDENCE in resp.evidence["verdict"]["reasons"]


def test_clock_skew_rejected():
    """调用方漏传 now（默认 0.0）或时钟不一致时，护栏不能静默失效。"""
    before, after = _fresh_pair()
    resp = WorldModelDiffProvider().judge(
        JudgeRequest(task="put_ball_in_basket", target="ball", container="basket"),
        before=before, after=after, ctx=JudgeContext(target_id="ball_900"))
    assert resp.success is False
    assert Reason.CLOCK_SKEW in resp.evidence["verdict"]["reasons"]


def test_age_is_not_clamped():
    """age() 夹成 0 正是陈旧度护栏失效的根源。"""
    assert obj("ball", 0, 1, 0.9, last_seen=100.0).age(0.0) == -100.0


# ------------------------------------------------------------------- F 抓取

def test_empty_grasp_with_visible_ball_is_not_success():
    """空抓 + 三米外看得见球 != 抓到了。"""
    after = [obj("ball", 1.5, 3.0, 0.9, last_seen=10.0)]
    ev = build_grasp_evidence(after, "ball", True, 10.0, gripper_pose=(0.2, 0.6))
    assert ev["satisfied"] is False
    assert Reason.OUT_OF_REACH in ev["verdict"]["reasons"]


def test_grasp_survives_self_occlusion():
    """抓在手里最容易被夹爪自遮挡，硬卡 age<1s 会稳定误杀真成功。"""
    after = [obj("ball", 0.21, 0.61, 0.9, last_seen=8.8)]
    ev = build_grasp_evidence(after, "ball", True, 10.0, gripper_pose=(0.2, 0.6))
    assert ev["satisfied"] is True
    assert ev["staleness_s"] == pytest.approx(1.2)


def test_grasp_without_gripper_pose_is_conservative():
    after = [obj("ball", 0.21, 0.61, 0.9, last_seen=10.0)]
    ev = build_grasp_evidence(after, "ball", True, 10.0, gripper_pose=None)
    assert ev["satisfied"] is False
    assert Reason.NO_GRIPPER_POSE in ev["verdict"]["reasons"]


def test_grasp_requires_gripper_feedback():
    after = [obj("ball", 0.21, 0.61, 0.9, last_seen=10.0)]
    ev = build_grasp_evidence(after, "ball", False, 10.0, gripper_pose=(0.2, 0.6))
    assert ev["satisfied"] is False
    assert Reason.GRIPPER_OPEN in ev["verdict"]["reasons"]


# ------------------------------------------------------------- G/I 证据质量

def test_single_frame_false_detection_is_not_evidence():
    """一帧未确认的误检不能充当视觉确认。"""
    pose = RobotPose(pose_uncertainty_cm=1.0)
    wm = WorldModel()
    for t in (0.0, 0.5, 1.0):
        wm.update([det(x=0.30, z=1.00, conf=0.85),
                   det(cls="basket", x=-0.60, z=1.60, conf=0.92, r=15.0)], pose, now=t)
    before = wm.snapshot()
    wm.update([det(x=0.30, z=1.00, conf=0.85),
               det(x=-0.61, z=1.59, conf=0.97),          # 桶里一帧高分误检
               det(cls="basket", x=-0.60, z=1.60, conf=0.92, r=15.0)], pose, now=1.5)
    after = wm.snapshot()
    ghost = [o for o in after if o.state == ObjectState.TENTATIVE]
    assert ghost, "这一帧确实产生了未确认轨迹"
    assert judge(before, after, 1.5).success is False


def test_low_confidence_belief_is_not_visual_confirmation():
    before = [obj("ball", 0.3, 1.0, 0.9, last_seen=5.0),
              obj("basket", -0.6, 1.6, 0.92, 15.0, last_seen=5.0, oid="basket_901")]
    after = [obj("ball", -0.6, 1.6, 0.16, last_seen=5.0, state=ObjectState.STALE),
             obj("basket", -0.6, 1.6, 0.92, 15.0, last_seen=5.0, oid="basket_901")]
    resp = judge(before, after, 5.0, ctx=JudgeContext(target_id="ball_900"))
    assert resp.success is False
    assert Reason.LOW_CONFIDENCE in resp.evidence["verdict"]["reasons"]


# --------------------------------------------------------------- 几何与边界

def test_ball_on_rim_is_not_inside():
    """阈值扣掉目标半径：球心 14.9cm、桶半径 15cm、球半径 3.3cm -> 不算进桶。"""
    before = [obj("ball", 1.0, 1.0, 0.9, last_seen=5.0, unc=0.2),
              obj("basket", 0.0, 1.5, 0.92, 15.0, last_seen=5.0, oid="basket_901", unc=0.2)]
    after = [obj("ball", 0.149, 1.5, 0.9, last_seen=5.0, unc=0.2),
             obj("basket", 0.0, 1.5, 0.92, 15.0, last_seen=5.0, oid="basket_901", unc=0.2)]
    resp = judge(before, after, 5.0, ctx=JudgeContext(target_id="ball_900"))
    assert resp.success is False
    assert resp.evidence["relation"]["threshold_cm"] == pytest.approx(11.7)


def test_still_grasped_blocks_placement_verdict():
    """契约无 y 轴，举在桶上方与掉进桶里投影相同；夹爪仍闭合时不能判放入成功。"""
    before = [obj("ball", 1.0, 1.0, 0.9, last_seen=5.0),
              obj("basket", 0.0, 1.5, 0.92, 15.0, last_seen=5.0, oid="basket_901")]
    after = [obj("ball", 0.0, 1.5, 0.9, last_seen=5.0),
             obj("basket", 0.0, 1.5, 0.92, 15.0, last_seen=5.0, oid="basket_901")]
    resp = judge(before, after, 5.0, ctx=JudgeContext(target_id="ball_900"),
                 gripper_closed=True)
    assert resp.success is False
    assert Reason.STILL_GRASPED in resp.evidence["verdict"]["reasons"]


def test_height_limitation_is_always_disclosed():
    before = [obj("ball", 1.0, 1.0, 0.9, last_seen=5.0),
              obj("basket", 0.0, 1.5, 0.92, 15.0, last_seen=5.0, oid="basket_901")]
    after = [obj("ball", 0.0, 1.5, 0.9, last_seen=5.0),
             obj("basket", 0.0, 1.5, 0.92, 15.0, last_seen=5.0, oid="basket_901")]
    ev = build_containment_evidence(before, after, "ball", "basket", 5.0,
                                    target_id="ball_900")
    assert Caveat.NO_HEIGHT_EVIDENCE in ev["verdict"]["caveats"]


# ----------------------------------------------------------------- J 自洽性

def test_response_has_single_source_of_truth():
    before = [obj("ball", 0.3, 1.0, 0.9, last_seen=0.0),
              obj("basket", -0.6, 1.6, 0.92, 15.0, last_seen=0.0, oid="basket_901")]
    after = [obj("ball", -0.6, 1.6, 0.9, last_seen=0.0),
             obj("basket", -0.6, 1.6, 0.92, 15.0, last_seen=0.0, oid="basket_901")]
    resp = judge(before, after, 5.0, ctx=JudgeContext(target_id="ball_900"))
    assert resp.success == resp.evidence["verdict"]["success"] is False
    assert resp.evidence["relation"]["satisfied"] is True   # 几何成立，但判定不成立
    assert "过旧" in resp.detail, "detail 必须解释失败原因，不能只报距离"


def test_reasons_are_deduped():
    before, after = _fresh_pair()
    resp = judge(before, after, 130.0, ctx=JudgeContext(target_id="ball_900"))
    reasons = resp.evidence["verdict"]["reasons"]
    assert len(reasons) == len(set(reasons))


# ------------------------------------------------------------- K provider

def test_unimplemented_provider_returns_response_not_exception():
    """外壳是冻结的 HTTP 服务，抛异常就是 500 而不是『判定不通过』。"""
    resp = get_provider("reward_classifier").judge(
        JudgeRequest(task="grasp", target="ball", now=1.0), [], [])
    assert resp.success is False
    assert "provider_not_implemented" in resp.evidence["verdict"]["reasons"]


def test_get_provider_rejects_foreign_kwargs():
    with pytest.raises(TypeError):
        get_provider("world_model_diff", checkpoint_path="/tmp/x.pt")


def test_unknown_task_is_refused():
    resp = judge([], [], 1.0, task="wipe_table")
    assert resp.success is False


def test_policy_is_overridable():
    """判定口径可配 —— 真机接入后阈值要按标定结果调。"""
    before = [obj("ball", 0.3, 1.0, 0.9, last_seen=0.0),
              obj("basket", -0.6, 1.6, 0.92, 15.0, last_seen=0.0, oid="basket_901")]
    after = [obj("ball", -0.6, 1.6, 0.4, last_seen=0.0),      # conf 低于默认下限
             obj("basket", -0.6, 1.6, 0.92, 15.0, last_seen=0.0, oid="basket_901")]
    ctx = JudgeContext(target_id="ball_900",
                       policy=EvidencePolicy(min_confidence=0.3))
    assert judge(before, after, 0.2, ctx=ctx).success is True
    assert judge(before, after, 0.2,
                 ctx=JudgeContext(target_id="ball_900")).success is False
