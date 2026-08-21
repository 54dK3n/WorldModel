"""CameraDetectorProvider / RawDetection / 像素到地面坐标 的补充测试。"""
from __future__ import annotations

import json
import math

import pytest

from world_model import (
    Detection,
    DetectionFrame,
    RawDetection,
    RobotPose,
    WorldModel,
    bbox_bottom_center,
)
from world_model.adapters import (
    letterbox_bbox_to_original,
    sg2002_xywh_to_xyxy,
    to_debug_dict,
)
from world_model.aliases import AliasTable
from world_model.association import AssociationConfig, associate
from world_model.providers import CameraDetectorProvider
from world_model.types import ObjectState, TrackedObject


def make_provider():
    """测试用途标定：640x640, fx=fy=500, cx=cy=320, 相机高 1.5m, 俯仰 30°。"""
    return CameraDetectorProvider(
        camera_height_m=1.5,
        pitch_rad=math.radians(30.0),
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=320.0,
        image_width=640,
        image_height=640,
        source="test_camera",
    )


def make_replay_provider(tmp_path, frames, **kwargs):
    """带测试标定的回放 provider。"""
    kwargs.setdefault("time_origin", 1786417200.0)
    return CameraDetectorProvider(
        calibration=make_provider().calibration,
        replay_path=write_jsonl(tmp_path, frames, **kwargs),
        source="test_camera",
    )


def write_jsonl(tmp_path, frames, time_origin=1786417200.0):
    path = tmp_path / "replay.jsonl"
    lines = []
    if time_origin is not None:
        lines.append(json.dumps({"time_origin": time_origin}))
    lines += [json.dumps(f, separators=(",", ":")) for f in frames]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def frame(frame_id, t, detections, w=640, h=640):
    return {
        "frame_id": frame_id,
        "timestamp": t,
        "image_width": w,
        "image_height": h,
        "robot_pose": {"x": 0.0, "z": 0.0, "yaw_rad": 0.0, "pose_uncertainty_cm": 1.0},
        "detections": detections,
    }


def ball_bbox(cx=320, y2=567, half_w=12, h=24):
    return [cx - half_w, y2 - h, cx + half_w, y2]


def bucket_bbox():
    return [163, 367, 243, 437]


# ------------------------------------------------------------------ 1. 底边中点

def test_bbox_bottom_center_is_correct():
    assert bbox_bottom_center((10, 20, 30, 80)) == (20.0, 80.0)


# ------------------------------------------------------------------ 2/3/4. 像素投影方向与无效射线

def test_pixel_below_center_projects_straight_ahead():
    p = make_provider()
    x, z = p.pixel_to_ground(320.0, 420.0)   # 图像中心下方
    assert x == pytest.approx(0.0, abs=1e-9)
    assert z > 0.0


def test_left_and_right_pixels_project_correctly():
    p = make_provider()
    x_left, z_left = p.pixel_to_ground(220.0, 420.0)
    x_right, z_right = p.pixel_to_ground(420.0, 420.0)
    assert x_left < 0.0 < x_right
    assert z_left > 0.0 and z_right > 0.0


@pytest.mark.parametrize("u,v", [(float("nan"), 100.0), (320.0, float("inf")), (320.0, 0.0)])
def test_invalid_rays_raise_not_nan(u, v):
    p = make_provider()
    with pytest.raises(ValueError):
        p.pixel_to_ground(u, v)


# ------------------------------------------------------------------ 5. 坏 bbox 被拒绝

@pytest.mark.parametrize(
    "kwargs",
    [
        {"class_name": "ball", "confidence": float("nan"), "bbox": (1, 1, 10, 10), "frame_id": "f", "timestamp": 0.0},
        {"class_name": "ball", "confidence": -0.1, "bbox": (1, 1, 10, 10), "frame_id": "f", "timestamp": 0.0},
        {"class_name": "ball", "confidence": 1.5, "bbox": (1, 1, 10, 10), "frame_id": "f", "timestamp": 0.0},
        {"class_name": "ball", "confidence": 0.9, "bbox": (10, 1, 1, 10), "frame_id": "f", "timestamp": 0.0},
        {"class_name": "ball", "confidence": 0.9, "bbox": (1, 10, 10, 1), "frame_id": "f", "timestamp": 0.0},
        {"class_name": "ball", "confidence": 0.9, "bbox": (-1, 1, 10, 10), "frame_id": "f", "timestamp": 0.0},
        {"class_name": "ball", "confidence": 0.9, "bbox": (1, 1, 10, 10), "frame_id": "f", "timestamp": float("nan")},
    ],
)
def test_bad_raw_detection_rejected(kwargs):
    with pytest.raises(ValueError):
        RawDetection(**kwargs)


def test_out_of_bounds_bbox_rejected_by_frame():
    with pytest.raises(ValueError):
        DetectionFrame(
            frame_id="f",
            timestamp=0.0,
            image_width=9,
            image_height=9,
            detections=[RawDetection("ball", 0.9, (1, 1, 10, 10), "f", 0.0)],
        )


def test_replay_resolution_must_match_calibration(tmp_path, caplog):
    p = make_replay_provider(
        tmp_path,
        [frame("wrong-size", 0.0, [], w=1280, h=720)],
    )
    with caplog.at_level("WARNING"):
        assert list(p.stream()) == []
    assert p.frames_loaded == 1
    assert p.frames_yielded == 0
    assert p.frames_skipped == 1
    assert "回放分辨率与标定不一致" in caplog.text


# ------------------------------------------------------------------ 6. 别名映射

def test_external_class_name_maps_through_alias():
    assert AliasTable().canonical("tennis_ball") == "ball"
    wm = WorldModel()
    wm.update([Detection(class_name="tennis_ball", x=0.0, z=1.0, confidence=0.9)],
              RobotPose(), now=0.0)
    assert wm.get_object("ball").name == "ball"
    assert wm.get_object("球") is not None


# ------------------------------------------------------------------ 7/8/9/10/11. WorldModel 行为

def test_continuous_frames_keep_stable_obj_id(tmp_path):
    frames = [
        frame("f0", 0.0, [
            {"class_name": "tennis_ball", "confidence": 0.94, "bbox": ball_bbox(320)},
            {"class_name": "bucket", "confidence": 0.92, "bbox": bucket_bbox()},
        ]),
        frame("f1", 0.5, [
            {"class_name": "tennis_ball", "confidence": 0.94, "bbox": ball_bbox(332)},
            {"class_name": "bucket", "confidence": 0.92, "bbox": bucket_bbox()},
        ]),
        frame("f2", 1.0, [
            {"class_name": "tennis_ball", "confidence": 0.94, "bbox": ball_bbox(345)},
            {"class_name": "bucket", "confidence": 0.92, "bbox": bucket_bbox()},
        ]),
    ]
    p = make_replay_provider(tmp_path, frames)

    wm = WorldModel()
    ids = {}
    for ts, pose, dets in p.stream():
        wm.update(dets, pose, now=ts)
        ball = wm.get_object("ball")
        ids[ts] = ball.obj_id
    assert len(set(ids.values())) == 1
    assert wm.get_object("basket") is not None


def test_short_occlusion_does_not_delete_object(tmp_path):
    frames = [
        frame("f0", 0.0, [
            {"class_name": "tennis_ball", "confidence": 0.94, "bbox": ball_bbox(320)},
            {"class_name": "bucket", "confidence": 0.92, "bbox": bucket_bbox()},
        ]),
        frame("f1", 0.5, [
            {"class_name": "tennis_ball", "confidence": 0.94, "bbox": ball_bbox(332)},
            {"class_name": "bucket", "confidence": 0.92, "bbox": bucket_bbox()},
        ]),
        frame("f2", 1.0, [
            {"class_name": "bucket", "confidence": 0.92, "bbox": bucket_bbox()},
        ]),
    ]
    p = make_replay_provider(tmp_path, frames)

    wm = WorldModel()
    for ts, pose, dets in p.stream():
        wm.update(dets, pose, now=ts)
    assert wm.get_object("ball") is not None, "短时遮挡不应删除对象"
    assert wm.get_object("ball").last_seen == 0.5


def test_out_of_fov_slow_decay_vs_in_fov_fast_decay():
    dt = 2.0
    wm_in = WorldModel()
    wm_in.update([Detection("tennis_ball", 0.0, 1.5, 0.95)], RobotPose(), now=0.0)
    wm_in.update([], RobotPose(), now=dt)
    conf_missed = wm_in.get_object("ball").confidence

    wm_out = WorldModel()
    wm_out.update([Detection("tennis_ball", 0.0, 1.5, 0.95)], RobotPose(), now=0.0)
    wm_out.update([], RobotPose(yaw_rad=math.pi), now=dt)
    conf_turned = wm_out.get_object("ball").confidence

    assert conf_turned > conf_missed
    assert conf_turned > 0.9 * 0.95


def test_large_displacement_keeps_obj_id():
    wm = WorldModel()
    wm.update([Detection("tennis_ball", 0.0, 1.0, 0.94)], RobotPose(), now=0.0)
    oid = wm.get_object("ball").obj_id
    # 2 秒移动 0.8m，超过基础门控 0.5m，但 dt 自适应门控应允许关联
    wm.update([Detection("tennis_ball", 0.8, 1.0, 0.94)], RobotPose(), now=2.0)
    assert len(wm.get_scene()) == 1
    assert wm.get_scene()[0].obj_id == oid


def test_stale_same_class_track_does_not_steal_nearby_detection():
    """门控只决定能不能配，不参与代价缩放；否则旧轨迹 gate 大反而抢新轨迹的检测。"""
    now = 10.0
    fresh = TrackedObject(
        obj_id="ball_001", name="ball", x=0.0, z=1.0, confidence=0.9,
        first_seen=now, last_seen=now, last_updated=now, hit_count=3,
        state=ObjectState.CONFIRMED,
    )
    stale = TrackedObject(
        obj_id="ball_002", name="ball", x=-0.9, z=0.9, confidence=0.3,
        first_seen=5.0, last_seen=5.0, last_updated=5.0, hit_count=1,
        state=ObjectState.STALE,
    )
    det = Detection("tennis_ball", x=0.4, z=1.0, confidence=0.9)
    result = associate([fresh, stale], [det], AliasTable(), AssociationConfig(), now=now)
    assert result.matches == [(0, 0)], "检测应配给更近的新鲜轨迹，而不是被旧轨迹抢走"


# ------------------------------------------------------------------ 12. 证据字段

def test_frame_id_bbox_timestamp_enter_evidence(tmp_path):
    frames = [
        frame("f0", 0.0, [
            {"class_name": "tennis_ball", "confidence": 0.94, "bbox": ball_bbox(320)},
        ]),
        frame("f1", 1.0, [
            {"class_name": "tennis_ball", "confidence": 0.94, "bbox": ball_bbox(345)},
        ]),
    ]
    p = make_replay_provider(tmp_path, frames)

    wm = WorldModel()
    for ts, pose, dets in p.stream():
        wm.update(dets, pose, now=ts)

    ball = wm.get_object("ball")
    assert ball.last_frame_id == "f1"
    assert tuple(ball.last_bbox) == tuple(ball_bbox(345))
    assert ball.last_seen == 1.0

    debug = to_debug_dict(ball)
    assert debug["last_frame_id"] == "f1"
    assert debug["last_bbox"] == ball_bbox(345)
    assert debug["timestamp"] == 1.0


# ------------------------------------------------------------------ 13. 相对时间与 Unix 时间

def test_relative_time_and_unix_time_are_not_mixed(tmp_path):
    trusted_ball = {
        "class_name": "tennis_ball", "confidence": 0.94,
        "bbox": ball_bbox(320), "radius_cm": 3.3,
        "size_source": "instance_config", "size_trusted": True,
    }
    frames = [
        frame("f7", 7.0, [trusted_ball]),
        frame("f8", 8.0, [trusted_ball]),
        frame("f9", 9.0, [trusted_ball]),
    ]
    p = make_replay_provider(tmp_path, frames, time_origin=1786417200.0)

    wm = WorldModel(time_origin=1786417200.0)
    for ts, pose, dets in p.stream():
        assert ts in (7.0, 8.0, 9.0), "provider 内部时钟必须是相对秒，不是 Unix 时间"
        assert dets[0].timestamp == ts
        wm.update(dets, pose, now=ts)

    obs = wm.to_contract(now=9.0)
    assert len(obs) == 1
    assert obs[0]["timestamp"] == "2026-08-11T03:00:09Z"


# ------------------------------------------------------------------ 14. 外部 YOLO 适配工具

def test_sg2002_xywh_to_xyxy_and_letterbox_restore():
    assert sg2002_xywh_to_xyxy(10, 20, 4, 6) == (8.0, 17.0, 12.0, 23.0)
    restored = letterbox_bbox_to_original((100, 100, 200, 200), scale=0.5, pad_left=10, pad_top=20)
    assert restored == (180.0, 160.0, 380.0, 360.0)
