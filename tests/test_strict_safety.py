"""严格失败关闭与边界测试。

这些测试验证：证据不足时 Judge 绝不虚假返回 success=True。
任何缺失、未知、歧义、降级、不可信标定/尺寸都必须以机器可读 reason 失败。
"""
from __future__ import annotations

import math

import pytest

from judge.evidence import EvidencePolicy, Reason, build_containment_evidence
from judge.providers import JudgeContext, JudgeRequest, WorldModelDiffProvider
from world_model import Detection, FrameQuality, RobotPose, WorldModel
from world_model.calibration import CameraCalibration
from world_model.providers import CameraDetectorProvider
from world_model.types import (
    COORDINATE_FRAME_WORLD,
    SIZE_SOURCE_BBOX_HEURISTIC,
    SIZE_SOURCE_DEFAULT,
    SIZE_SOURCE_INSTANCE_CONFIG,
    SIZE_SOURCE_UNKNOWN,
    ObjectState,
    TrackedObject,
)

COMPLETE_FQ = FrameQuality(frame_id="f", degraded=False, calibration_trusted=True)


def make_trusted_track(name, x, z, r, oid, last_seen=0.0, conf=0.9,
                       state=ObjectState.CONFIRMED, unc=0.0, hit_count=5):
    return TrackedObject(
        obj_id=oid, name=name, x=x, z=z, radius_cm=r,
        size_source=SIZE_SOURCE_INSTANCE_CONFIG, size_trusted=True,
        confidence=conf, first_seen=0.0, last_seen=last_seen, last_updated=last_seen,
        hit_count=hit_count, state=state, pose_uncertainty_cm=unc,
        source="test", last_bbox=(0, 0, 10, 10), last_frame_id="f",
        last_frame_quality=COMPLETE_FQ,
    )


def strict_judge(before, after, now=0.5, *, target_id="ball_001",
                 calibration_trusted=True, frame_quality=COMPLETE_FQ,
                 coordinate_frame=COORDINATE_FRAME_WORLD, gripper_closed=False):
    return WorldModelDiffProvider().judge(
        JudgeRequest(task="put_ball_in_basket", target="ball", container="basket",
                     now=now, gripper_closed=gripper_closed),
        before=before, after=after,
        ctx=JudgeContext(
            target_id=target_id,
            calibration_trusted=calibration_trusted,
            frame_quality=frame_quality,
            coordinate_frame=coordinate_frame,
        ),
    )


def baseline_pair(ball_after=(0.1169, 0.0), ball_before=(1.0, 0.0)):
    before = [
        make_trusted_track("ball", ball_before[0], ball_before[1], 3.3, "ball_001"),
        make_trusted_track("basket", 0.0, 0.0, 15.0, "basket_001"),
    ]
    after = [
        make_trusted_track("ball", ball_after[0], ball_after[1], 3.3, "ball_001"),
        make_trusted_track("basket", 0.0, 0.0, 15.0, "basket_001"),
    ]
    return before, after


# ------------------------------------------------------------------ 尺寸边界

def test_trusted_size_geometry_11_69_succeeds():
    before, after = baseline_pair((0.1169, 0.0))
    resp = strict_judge(before, after)
    assert resp.success is True
    assert resp.evidence["verdict"]["reasons"] == []
    assert resp.evidence["relation"]["before_satisfied"] is False
    assert resp.evidence["relation"]["satisfied"] is True


def test_trusted_size_geometry_11_71_fails():
    before, after = baseline_pair((0.1171, 0.0))
    resp = strict_judge(before, after)
    assert resp.success is False
    assert Reason.GEOMETRY_NOT_SATISFIED in resp.evidence["verdict"]["reasons"]


def test_missing_size_evidence_fails():
    before, after = baseline_pair()
    for o in before + after:
        o.radius_cm = -1.0
        o.size_source = SIZE_SOURCE_UNKNOWN
        o.size_trusted = False
    resp = strict_judge(before, after)
    assert resp.success is False
    assert Reason.MISSING_SIZE_EVIDENCE in resp.evidence["verdict"]["reasons"]


def test_bbox_heuristic_size_evidence_fails():
    before, after = baseline_pair()
    for o in before + after:
        o.size_source = SIZE_SOURCE_BBOX_HEURISTIC
        o.size_trusted = False
    resp = strict_judge(before, after)
    assert resp.success is False
    assert Reason.UNTRUSTED_SIZE_EVIDENCE in resp.evidence["verdict"]["reasons"]


def test_default_5cm_size_evidence_fails():
    before, after = baseline_pair()
    for o in before + after:
        o.radius_cm = 5.0
        o.size_source = SIZE_SOURCE_DEFAULT
        o.size_trusted = False
    resp = strict_judge(before, after)
    assert resp.success is False
    assert Reason.UNTRUSTED_SIZE_EVIDENCE in resp.evidence["verdict"]["reasons"]


# ------------------------------------------------------------------ 单帧误检

def test_single_frame_false_detection_does_not_enter_contract_or_judge():
    wm = WorldModel(time_origin=1786417200.0)
    pose = RobotPose(pose_uncertainty_cm=1.0)
    for t in (0.0, 0.5, 1.0):
        wm.update([
            Detection("basket", 0.0, 0.0, 0.92, radius_cm=15.0,
                      size_source=SIZE_SOURCE_INSTANCE_CONFIG, size_trusted=True,
                      frame_quality=COMPLETE_FQ),
        ], pose, now=t)
    before = wm.snapshot()
    wm.update([
        Detection("basket", 0.0, 0.0, 0.92, radius_cm=15.0,
                  size_source=SIZE_SOURCE_INSTANCE_CONFIG, size_trusted=True,
                  frame_quality=COMPLETE_FQ),
        Detection("sports ball", 0.0, 0.0, 0.99, radius_cm=3.3,
                  size_source=SIZE_SOURCE_INSTANCE_CONFIG, size_trusted=True,
                  frame_quality=COMPLETE_FQ),
    ], pose, now=1.5)
    after = wm.snapshot()

    ghost = [o for o in after if o.name == "ball"]
    assert len(ghost) == 1 and ghost[0].state == ObjectState.TENTATIVE
    contract_balls = [o for o in wm.to_contract(now=1.5) if o["name"] == "ball"]
    assert contract_balls == [], "TENTATIVE 单帧误检不得进入默认正式契约"

    resp = strict_judge(before, after, now=1.5, target_id=ghost[0].obj_id)
    assert resp.success is False
    assert Reason.UNCONFIRMED_TRACK in resp.evidence["verdict"]["reasons"]


def test_single_frame_false_detection_decays_away_without_ambiguity():
    wm = WorldModel(visibility=make_cal())
    pose = RobotPose()
    for t in (0.0, 0.5, 1.0):
        wm.update([Detection("basket", -0.5, 1.6, 0.92, radius_cm=15.0,
                             size_source=SIZE_SOURCE_INSTANCE_CONFIG, size_trusted=True)],
                  pose, now=t)
    wm.update([Detection("basket", -0.5, 1.6, 0.92, radius_cm=15.0,
                         size_source=SIZE_SOURCE_INSTANCE_CONFIG, size_trusted=True),
               Detection("sports ball", -0.5, 1.6, 0.99, radius_cm=3.3,
                         size_source=SIZE_SOURCE_INSTANCE_CONFIG, size_trusted=True)],
              pose, now=1.5)
    assert wm.get_object("ball") is not None
    for t in (2.0, 2.5, 3.0, 4.0, 6.0, 10.0):
        wm.update([Detection("basket", -0.5, 1.6, 0.92, radius_cm=15.0,
                             size_source=SIZE_SOURCE_INSTANCE_CONFIG, size_trusted=True)],
                  pose, now=t)
    assert wm.get_object("ball") is None, "单帧误检不得长期作为同名对象残留"


# ------------------------------------------------------------------ FOV 与量程

def make_cal(**overrides):
    params = dict(image_width=640, image_height=640, fx=500.0, fy=500.0,
                  cx=320.0, cy=320.0, camera_height_m=1.5,
                  pitch_rad=math.radians(30.0), camera_x_m=0.0, camera_z_m=0.0,
                  ground_plane_height_m=0.0, min_ground_range_m=0.15,
                  max_ground_range_m=4.0, source="test", is_real_calibration=False)
    params.update(overrides)
    return CameraCalibration(**params)


def test_projection_range_guard_and_visibility():
    cal = make_cal()

    # 图像中心投影有效
    x, z = cal.project_pixel_to_ground(320, 320)
    assert math.isfinite(x) and math.isfinite(z)

    # 左右边缘内 1 像素投影方向正确
    xl, _ = cal.project_pixel_to_ground(1, 420)
    xr, _ = cal.project_pixel_to_ground(639, 420)
    assert xl < 0 < xr

    # 边缘外 1 像素的投影不用于可见性判断（可见性模型会排除）
    for u, v in [(-1, 420), (641, 420)]:
        xs, zs = cal.project_pixel_to_ground(u, v)
        assert cal.is_ground_point_visible(xs, zs, RobotPose()) is False

    # 3.99m 可见，4.01m 不可见
    u1, v1 = cal.ground_point_to_pixel(0.0, 3.99)
    u2, v2 = cal.ground_point_to_pixel(0.0, 4.01)
    assert cal.is_ground_point_visible(0.0, 3.99, RobotPose()) is True
    assert cal.is_ground_point_visible(0.0, 4.01, RobotPose()) is False

    # 地平线附近：v=32 超出量程，v=31 在地平线以上
    with pytest.raises(ValueError, match="超过有效量程"):
        cal.project_pixel_to_ground(320, 32)
    with pytest.raises(ValueError, match="可投影地面区域之外"):
        cal.project_pixel_to_ground(320, 31)

    for u, v in [(float("nan"), 100), (320, float("inf"))]:
        with pytest.raises(ValueError, match="有限"):
            cal.project_pixel_to_ground(u, v)


# ------------------------------------------------------------------ 坐标变换

def test_yaw_rotation_direction_is_explicit():
    # 当前约定：yaw=+pi/2 表示机器人从 +z 轴左转到 +x 轴。
    pose = RobotPose(0.0, 0.0, math.pi / 2)
    x, z = pose.to_world(0.0, 1.0)  # 机器人正前方
    assert (x, z) == pytest.approx((1.0, 0.0))
    pose = RobotPose(0.0, 0.0, -math.pi / 2)
    x, z = pose.to_world(0.0, 1.0)
    assert (x, z) == pytest.approx((-1.0, 0.0))


def test_robot_translation_does_not_change_static_world_point():
    for robot_x, robot_z in [(0.0, 0.0), (0.4, 0.0), (0.8, 0.0)]:
        pose = RobotPose(robot_x, 0.0, robot_z)
        x_local, z_local = pose.to_local(0.3, 1.2)
        x_world, z_world = pose.to_world(x_local, z_local)
        assert (x_world, z_world) == pytest.approx((0.3, 1.2))

def test_coordinate_transform_consistency_and_stable_track():
    cal = make_cal()
    world_x, world_z = 0.3, 1.2
    poses = [
        RobotPose(0.0, 0.0, 0.0),
        RobotPose(0.4, 0.0, 0.0),
        RobotPose(0.8, 0.0, 0.0),
        RobotPose(0.0, 0.0, math.pi / 2),
        RobotPose(0.0, 0.0, -math.pi / 2),
    ]
    wm = WorldModel(visibility=cal)
    oid = None
    for i, pose in enumerate(poses):
        x_local, z_local = pose.to_local(world_x, world_z)
        u, v = cal.ground_point_to_pixel(x_local, z_local)
        x_back, z_back = cal.project_pixel_to_ground(u, v)
        x_world, z_world = pose.to_world(x_back, z_back)
        assert abs(x_world - world_x) < 1e-6
        assert abs(z_world - world_z) < 1e-6
        det = Detection("basket", x_world, z_world, 0.92, radius_cm=15.0,
                        size_source=SIZE_SOURCE_INSTANCE_CONFIG, size_trusted=True,
                        frame_quality=COMPLETE_FQ)
        wm.update([det], pose, now=float(i))
        if oid is None:
            oid = wm.get_object("basket").obj_id
        else:
            assert wm.get_object("basket").obj_id == oid

    assert len(wm.get_scene()) == 1
    assert abs(wm.get_object("basket").x - world_x) < 1e-6
    assert abs(wm.get_object("basket").z - world_z) < 1e-6


def test_robot_motion_does_not_create_false_judge_change():
    before = [make_trusted_track("ball", 0.1169, 0.0, 3.3, "ball_001"),
              make_trusted_track("basket", 0.0, 0.0, 15.0, "basket_001")]
    after = [make_trusted_track("ball", 0.1169, 0.0, 3.3, "ball_001"),
             make_trusted_track("basket", 0.0, 0.0, 15.0, "basket_001")]
    resp = strict_judge(before, after, now=0.5)
    assert resp.success is False
    assert Reason.NO_STATE_CHANGE in resp.evidence["verdict"]["reasons"]


# ------------------------------------------------------------------ 部分坏检测

def test_partial_bad_detections_keep_good_ones_and_mark_degraded(tmp_path, caplog):
    import json
    frame_id = "f0"
    good_ball_bbox = [308, 543, 332, 567]
    good_bucket_bbox = [163, 367, 243, 437]
    frame_data = {
        "frame_id": frame_id,
        "timestamp": 0.0,
        "image_width": 640,
        "image_height": 640,
        "robot_pose": {"x": 0.0, "z": 0.0, "yaw_rad": 0.0, "pose_uncertainty_cm": 1.0},
        "detections": [
            {"class_name": "tennis_ball", "confidence": 0.94, "bbox": good_ball_bbox,
             "radius_cm": 3.3, "size_source": "instance_config", "size_trusted": True},
            {"class_name": "tennis_ball", "confidence": 0.9,
             "bbox": [0, 0, 640.4, 20]},
            {"class_name": "tennis_ball", "confidence": float("nan"), "bbox": [10, 10, 30, 30]},
            {"class_name": "bucket", "confidence": 0.92, "bbox": good_bucket_bbox,
             "radius_cm": 15.0, "size_source": "instance_config", "size_trusted": True},
        ],
    }
    path = tmp_path / "partial.jsonl"
    path.write_text(json.dumps({"time_origin": 1786417200.0}) + "\n" +
                    json.dumps(frame_data) + "\n", encoding="utf-8")
    cal = make_cal(is_real_calibration=True, source="real")
    p = CameraDetectorProvider(calibration=cal, replay_path=path)
    with caplog.at_level("WARNING"):
        outputs = list(p.stream())
    assert len(outputs) == 1
    ts, pose, dets = outputs[0]
    assert ts == 0.0
    assert len(dets) == 2, "两个合法检测必须保留"
    assert {d.class_name for d in dets} == {"tennis_ball", "bucket"}
    assert p.detections_skipped == 2
    assert p.frames_skipped == 0
    assert all(d.frame_quality.degraded is True for d in dets)
    assert all(d.frame_quality.detections_skipped == 2 for d in dets)
    assert "index=1" in caplog.text and "index=2" in caplog.text


# ------------------------------------------------------------------ 时间与标定

def test_strict_replay_missing_time_origin_fails(tmp_path):
    import json
    path = tmp_path / "no_origin.jsonl"
    path.write_text(json.dumps({"frame_id": "f0", "timestamp": 0.0,
                                "image_width": 640, "image_height": 640,
                                "robot_pose": {}, "detections": []}) + "\n",
                    encoding="utf-8")
    p = CameraDetectorProvider(calibration=make_cal(), replay_path=path)
    with pytest.raises(ValueError, match="缺少 time_origin"):
        list(p.stream())


def test_replay_with_time_origin_no_1970(tmp_path):
    import json
    path = tmp_path / "ok.jsonl"
    path.write_text(json.dumps({"time_origin": 1786417200.0}) + "\n" +
                    json.dumps({"frame_id": "f0", "timestamp": 0.0,
                                "image_width": 640, "image_height": 640,
                                "robot_pose": {}, "detections": []}) + "\n",
                    encoding="utf-8")
    p = CameraDetectorProvider(calibration=make_cal(is_real_calibration=True, source="real"),
                               replay_path=path)
    outputs = list(p.stream())
    assert len(outputs) == 1
    assert p.time_origin == 1786417200.0


def test_test_calibration_blocks_strict_success():
    before, after = baseline_pair()
    resp = strict_judge(before, after, calibration_trusted=False)
    assert resp.success is False
    assert Reason.UNTRUSTED_CALIBRATION in resp.evidence["verdict"]["reasons"]


def test_real_calibration_allows_success_when_all_else_ok():
    before, after = baseline_pair()
    resp = strict_judge(before, after, calibration_trusted=True)
    assert resp.success is True


def test_clock_skew_blocks_success():
    before, after = baseline_pair()
    resp = strict_judge(before, after, now=-0.1)
    assert resp.success is False
    assert Reason.CLOCK_SKEW in resp.evidence["verdict"]["reasons"]


# ------------------------------------------------------------------ 虚假成功不变量

def _track(name, x, z, r, oid, **kw):
    return make_trusted_track(name, x, z, r, oid, **kw)


@pytest.mark.parametrize("scenario", [
    "single_frame_ghost_in_basket",
    "ambiguous_two_balls",
    "already_in_basket_before",
    "identity_broken",
    "unknown_coordinate_frame",
    "unknown_radius",
    "untrusted_calibration",
    "stale_evidence",
    "clock_skew",
    "tentative_target",
    "stale_target",
    "target_out_of_range",
    "degraded_frame",
    "gripper_still_closed",
    "robot_motion_only",
    "bbox_heuristic_size_only",
])
def test_false_success_invariants(scenario):
    fq_degraded = FrameQuality(frame_id="f", degraded=True, detections_skipped=1,
                               calibration_trusted=True)
    base_before = [
        _track("ball", 1.0, 0.0, 3.3, "ball_001"),
        _track("basket", 0.0, 0.0, 15.0, "basket_001"),
    ]
    base_after = [
        _track("ball", 0.1169, 0.0, 3.3, "ball_001"),
        _track("basket", 0.0, 0.0, 15.0, "basket_001"),
    ]
    before = base_before
    after = base_after
    kw = {}
    if scenario == "single_frame_ghost_in_basket":
        before = [_track("basket", 0.0, 0.0, 15.0, "basket_001")]
        after = [_track("ball", 0.0, 0.0, 3.3, "ball_001", state=ObjectState.TENTATIVE, hit_count=1),
                 _track("basket", 0.0, 0.0, 15.0, "basket_001")]
        kw["target_id"] = "ball_001"
    elif scenario == "ambiguous_two_balls":
        before = base_before + [_track("ball", 2.0, 0.0, 3.3, "ball_002")]
        after = base_after + [_track("ball", 2.0, 0.0, 3.3, "ball_002")]
        kw["target_id"] = None
    elif scenario == "already_in_basket_before":
        before = base_after
        after = base_after
    elif scenario == "identity_broken":
        after = [_track("ball", 0.1169, 0.0, 3.3, "ball_999"),
                 _track("basket", 0.0, 0.0, 15.0, "basket_001")]
    elif scenario == "unknown_coordinate_frame":
        kw["coordinate_frame"] = None
    elif scenario == "unknown_radius":
        before = [_track("ball", 1.0, 0.0, -1.0, "ball_001"),
                  _track("basket", 0.0, 0.0, 15.0, "basket_001")]
        after = [_track("ball", 0.1169, 0.0, -1.0, "ball_001"),
                 _track("basket", 0.0, 0.0, 15.0, "basket_001")]
        for o in before + after:
            if o.obj_id.startswith("ball"):
                o.size_source = SIZE_SOURCE_UNKNOWN
                o.size_trusted = False
    elif scenario == "untrusted_calibration":
        kw["calibration_trusted"] = False
    elif scenario == "stale_evidence":
        before = [_track("ball", 1.0, 0.0, 3.3, "ball_001", last_seen=0.0),
                  _track("basket", 0.0, 0.0, 15.0, "basket_001", last_seen=0.0)]
        after = [_track("ball", 0.1169, 0.0, 3.3, "ball_001", last_seen=0.0),
                 _track("basket", 0.0, 0.0, 15.0, "basket_001", last_seen=0.0)]
        kw["now"] = 30.0
    elif scenario == "clock_skew":
        before = [_track("ball", 1.0, 0.0, 3.3, "ball_001", last_seen=10.0),
                  _track("basket", 0.0, 0.0, 15.0, "basket_001", last_seen=10.0)]
        after = [_track("ball", 0.1169, 0.0, 3.3, "ball_001", last_seen=10.0),
                 _track("basket", 0.0, 0.0, 15.0, "basket_001", last_seen=10.0)]
        kw["now"] = 5.0
    elif scenario == "tentative_target":
        after = [_track("ball", 0.1169, 0.0, 3.3, "ball_001", state=ObjectState.TENTATIVE, hit_count=1),
                 _track("basket", 0.0, 0.0, 15.0, "basket_001")]
    elif scenario == "stale_target":
        after = [_track("ball", 0.1169, 0.0, 3.3, "ball_001", state=ObjectState.STALE, conf=0.4),
                 _track("basket", 0.0, 0.0, 15.0, "basket_001")]
    elif scenario == "target_out_of_range":
        after = [_track("ball", 0.1169, 0.0, 3.3, "ball_001"),
                 _track("basket", 5.0, 0.0, 15.0, "basket_001")]
        before = [_track("ball", 1.0, 0.0, 3.3, "ball_001"),
                  _track("basket", 5.0, 0.0, 15.0, "basket_001")]
    elif scenario == "degraded_frame":
        kw["frame_quality"] = fq_degraded
    elif scenario == "gripper_still_closed":
        kw["gripper_closed"] = True
    elif scenario == "robot_motion_only":
        before = base_before
        after = base_before
    elif scenario == "bbox_heuristic_size_only":
        before = [_track("ball", 1.0, 0.0, 3.3, "ball_001"),
                  _track("basket", 0.0, 0.0, 15.0, "basket_001")]
        after = [_track("ball", 0.1169, 0.0, 3.3, "ball_001"),
                 _track("basket", 0.0, 0.0, 15.0, "basket_001")]
        for o in before + after:
            o.size_source = SIZE_SOURCE_BBOX_HEURISTIC
            o.size_trusted = False

    resp = strict_judge(
        kw.pop("before", before),
        kw.pop("after", after),
        now=kw.pop("now", 0.5),
        target_id=kw.pop("target_id", "ball_001"),
        calibration_trusted=kw.pop("calibration_trusted", True),
        frame_quality=kw.pop("frame_quality", COMPLETE_FQ),
        coordinate_frame=kw.pop("coordinate_frame", COORDINATE_FRAME_WORLD),
        gripper_closed=kw.pop("gripper_closed", False),
    )
    assert resp.success is False, f"场景 {scenario} 不应成功，但返回了 True"
    assert resp.evidence["verdict"]["success"] is False
