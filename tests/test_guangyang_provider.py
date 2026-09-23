"""Guangyang platform provider tests.

These tests do not require the browser simulator.  They lock the adapter
contract used by the P2 spatial-memory integration.
"""
from __future__ import annotations

import math

import pytest

from world_model import WorldModel
from world_model.adapters import to_scene_observations
from world_model.decay import DecayConfig, FovConfig
from world_model.providers.guangyang import (
    GuangyangNoiseConfig,
    GuangyangProvider,
    observation_to_detection,
    odometry_to_pose,
)
from world_model.types import ObjectState, RobotPose


def test_fov_defaults_match_guangyang_camera():
    fov = FovConfig()
    assert fov.horizontal_fov_deg == pytest.approx(75.2)
    assert fov.max_range_m == pytest.approx(8.0)


def test_decay_has_guangyang_class_scales():
    scales = DecayConfig().class_half_life_scale
    assert scales["target"] == pytest.approx(1.0)
    assert scales["distractor"] == pytest.approx(1.5)
    assert scales["obstacle"] == pytest.approx(8.0)
    assert scales["storage-zone"] == pytest.approx(8.0)


def test_odometry_to_pose_maps_forward_right_and_heading():
    pose = odometry_to_pose({
        "forwardCm": 100.0,
        "rightCm": 50.0,
        "headingDeg": 90.0,
        "distanceCm": 111.8,
    })
    assert pose.x == pytest.approx(0.5)
    assert pose.z == pytest.approx(1.0)
    assert pose.yaw_rad == pytest.approx(-math.pi / 2.0)


def test_odometry_noise_is_reproducible_with_seed():
    odometry = {"forwardCm": 200.0, "rightCm": 0.0, "headingDeg": 0.0, "distanceCm": 200.0}
    cfg = GuangyangNoiseConfig(distance_relative_std=0.02, heading_deg_std=1.0, seed=7)
    first = odometry_to_pose(odometry, cfg)
    second = odometry_to_pose(odometry, cfg, rng=__import__("random").Random(7))
    assert first.x == pytest.approx(second.x)
    assert first.z == pytest.approx(second.z)
    assert first.yaw_rad == pytest.approx(second.yaw_rad)
    assert first.pose_uncertainty_cm == pytest.approx(4.0)


def test_observation_polar_conversion_uses_pose_yaw_plus_bearing():
    pose = RobotPose(x=1.0, z=2.0, yaw_rad=math.radians(90.0))
    det = observation_to_detection({
        "category": "target",
        "distanceCm": 200.0,
        "bearingDeg": 0.0,
        "confidence": 0.91,
    }, pose, timestamp=3.0)
    assert det.class_name == "target"
    assert det.x == pytest.approx(3.0)
    assert det.z == pytest.approx(2.0)
    assert det.radius_cm == pytest.approx(22.0)
    assert det.timestamp == pytest.approx(3.0)


def test_observation_direction_fallback_is_coarse_only():
    det = observation_to_detection(
        {"category": "obstacle", "distanceCm": 100.0, "direction": "左", "confidence": 0.8},
        RobotPose(),
    )
    assert det.x < 0.0
    assert det.z > 0.0


def test_provider_stream_and_world_model_confirm_tracks():
    frames = [
        {
            "timestamp": frame_index * 0.5,
            "odometry": {"forwardCm": 0.0, "rightCm": 0.0, "headingDeg": 0.0, "distanceCm": 0.0},
            "observations": [{
                "category": "target",
                "distanceCm": 100.0,
                "bearingDeg": 0.0,
                "confidence": 0.94,
                "frameId": f"f{frame_index}",
            }],
        }
        for frame_index in range(3)
    ]
    wm = WorldModel()
    for timestamp, pose, detections in GuangyangProvider(frames).stream():
        wm.update(detections, pose, now=timestamp)
    target = wm.get_object("target")
    assert target is not None
    assert target.state == ObjectState.CONFIRMED
    observations = to_scene_observations([target], time_origin=None)
    assert set(observations[0]) == {
        "name", "aliases", "x", "z", "radius_cm", "source", "timestamp", "confidence",
    }


def test_provider_keeps_two_same_class_objects_separate():
    frames = [
        {
            "timestamp": 0.0,
            "odometry": {"forwardCm": 0.0, "rightCm": 0.0, "headingDeg": 0.0, "distanceCm": 0.0},
            "observations": [
                {"category": "target", "distanceCm": 100.0, "bearingDeg": -20.0, "confidence": 0.9},
                {"category": "target", "distanceCm": 100.0, "bearingDeg": 20.0, "confidence": 0.9},
            ],
        },
        {
            "timestamp": 0.5,
            "odometry": {"forwardCm": 0.0, "rightCm": 0.0, "headingDeg": 0.0, "distanceCm": 0.0},
            "observations": [
                {"category": "target", "distanceCm": 100.0, "bearingDeg": -20.0, "confidence": 0.9},
                {"category": "target", "distanceCm": 100.0, "bearingDeg": 20.0, "confidence": 0.9},
            ],
        },
    ]
    wm = WorldModel()
    for timestamp, pose, detections in GuangyangProvider(frames).stream():
        wm.update(detections, pose, now=timestamp)
    active = [obj for obj in wm.get_scene() if obj.name == "target"]
    assert len(active) == 2
    assert len({obj.obj_id for obj in active}) == 2


def test_missing_target_decays_below_stale_threshold_within_three_seconds():
    wm = WorldModel()
    for timestamp in (0.0, 0.5, 1.0):
        det = observation_to_detection({
            "category": "target",
            "distanceCm": 100.0,
            "bearingDeg": 0.0,
            "confidence": 0.95,
            "frameId": f"hit-{timestamp}",
        }, RobotPose(), timestamp=timestamp)
        wm.update([det], RobotPose(), now=timestamp)
    target = wm.get_object("target")
    assert target is not None
    wm.update([], RobotPose(), now=4.0)
    target = wm.get_object("target")
    assert target is not None
    assert target.confidence < DecayConfig().stale_threshold


def test_static_config_fuses_hits_by_equal_weight_mean():
    """广阳岛静态配置：确认命中按等权平均，不受命中间隔 dt 影响（dt=4.5s 时指数滤波≈只用最后一次）。"""
    from world_model.core import WorldModel
    from world_model.providers.guangyang import guangyang_static_association_config
    from world_model.types import Detection, RobotPose

    wm = WorldModel(assoc_cfg=guangyang_static_association_config())
    points = [(1.00, 2.00), (1.10, 2.06), (0.96, 2.10)]
    for i, (x, z) in enumerate(points):
        det = Detection(class_name="target", x=x, z=z, confidence=0.9)
        wm.update([det], RobotPose(x=0.0, z=-0.2 * i), now=4.5 * i)  # 三个相距 20cm 的位姿
    obj = wm.get_object("target")
    assert obj.hit_count == 3
    assert abs(obj.x - sum(p[0] for p in points) / 3) < 1e-9
    assert abs(obj.z - sum(p[1] for p in points) / 3) < 1e-9


def test_default_config_keeps_time_constant_smoothing():
    from world_model.association import AssociationConfig
    from world_model.core import WorldModel
    from world_model.types import Detection, RobotPose

    assert AssociationConfig().static_equal_weight is False
    wm = WorldModel()
    wm.update([Detection(class_name="target", x=0.0, z=0.0, confidence=0.9)], RobotPose(), now=0.0)
    wm.update([Detection(class_name="target", x=0.1, z=0.0, confidence=0.9)], RobotPose(), now=4.5)
    assert wm.get_object("target").x > 0.0999


def test_static_config_ignores_repeat_observation_from_same_pose():
    """与已有命中位姿相距 <15cm 的观测：不增加命中数、不参与平均（map-02 v27：同一位姿观测两次被判 CONFIRMED）。"""
    from world_model.core import WorldModel
    from world_model.providers.guangyang import guangyang_static_association_config
    from world_model.types import Detection, ObjectState, RobotPose

    wm = WorldModel(assoc_cfg=guangyang_static_association_config())
    wm.update([Detection(class_name="target", x=1.00, z=2.00, confidence=0.9)], RobotPose(x=0.0, z=0.0), now=0.0)
    # 同一位姿（差 5cm）再看一次，读数略不同：忽略
    wm.update([Detection(class_name="target", x=1.20, z=2.00, confidence=0.9)], RobotPose(x=0.05, z=0.0), now=4.5)
    obj = wm.get_object("target")
    assert obj.hit_count == 1
    assert abs(obj.x - 1.00) < 1e-9
    assert obj.last_seen == 4.5
    # 换到 20cm 外的位姿：计入，等权平均
    wm.update([Detection(class_name="target", x=1.10, z=2.10, confidence=0.9)], RobotPose(x=0.0, z=-0.20), now=9.0)
    obj = wm.get_object("target")
    assert obj.hit_count == 2
    assert abs(obj.x - 1.05) < 1e-9 and abs(obj.z - 2.05) < 1e-9
    assert obj.state != ObjectState.CONFIRMED  # 两个位置不够确认


def test_default_config_counts_repeat_pose_hits():
    from world_model.association import AssociationConfig
    from world_model.core import WorldModel
    from world_model.types import Detection, RobotPose

    assert AssociationConfig().min_hit_pose_gap_m == 0.0
    wm = WorldModel()
    for i in range(2):
        wm.update([Detection(class_name="target", x=1.0, z=2.0, confidence=0.9)], RobotPose(), now=0.5 * i)
    assert wm.get_object("target").hit_count == 2
