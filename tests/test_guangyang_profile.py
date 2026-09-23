"""Generic defaults stay at the migration baseline; GY opts in as a whole."""
import copy
import math

import pytest

from world_model import Detection, ObjectState, RobotPose, WorldModel
from world_model.association import AssociationConfig
from world_model.decay import DecayConfig, FovConfig
from world_model.providers import guangyang_static_world_model


LEGACY_SCALES = {"basket": 4.0, "table": 8.0, "ball": 1.0, "bottle": 2.0}
GY_SCALES = dict(LEGACY_SCALES, target=1.0, distractor=1.5, obstacle=8.0,
                 **{"storage-zone": 8.0, "cleanup-zone": 8.0})


def target(x=0.0, z=0.6):
    return Detection("target", x=x, z=z, confidence=0.9)


def test_generic_defaults_match_migration_baseline():
    wm = WorldModel()
    assert vars(wm.fov_cfg) == {
        "horizontal_fov_deg": 70.0, "max_range_m": 4.0, "min_range_m": 0.15,
    }
    assert wm.fov_cfg == FovConfig()
    assert vars(wm.decay_cfg) == {
        "half_life_in_fov_missed_s": 1.5, "half_life_out_of_fov_s": 60.0,
        "class_half_life_scale": LEGACY_SCALES, "stale_threshold": 0.5,
        "lost_threshold": 0.15, "confirm_hits": 3,
    }
    assert wm.decay_cfg == DecayConfig()
    assert wm.assoc_cfg == AssociationConfig()
    assert wm.assoc_cfg.static_equal_weight is False
    assert wm.assoc_cfg.min_hit_pose_gap_m == 0.0
    assert wm.include_lost_in_get_object is False
    assert wm.position_smoothing == 0.6 and wm.nominal_dt_s == 0.5


@pytest.mark.parametrize("max_range", [8.0, 0.9])
def test_gy_factory_sets_complete_old_vendor_profile(max_range):
    wm = (guangyang_static_world_model() if max_range == 8.0
          else guangyang_static_world_model(max_range_m=max_range))
    assert vars(wm.fov_cfg) == {
        "horizontal_fov_deg": 75.2, "max_range_m": max_range,
        "min_range_m": 0.15,
    }
    assert wm.visibility is wm.fov_cfg
    assert vars(wm.decay_cfg) == {
        "half_life_in_fov_missed_s": 1.5, "half_life_out_of_fov_s": 60.0,
        "class_half_life_scale": GY_SCALES, "stale_threshold": 0.5,
        "lost_threshold": 0.15, "confirm_hits": 3,
    }
    assert wm.assoc_cfg.gate_distance_m == 0.30
    assert wm.assoc_cfg.max_gate_distance_m == 0.30
    assert wm.assoc_cfg.max_speed_mps == 0.0
    assert wm.assoc_cfg.static_equal_weight is True
    assert wm.assoc_cfg.min_hit_pose_gap_m == 0.15
    assert wm.include_lost_in_get_object is True
    assert wm.position_smoothing == 0.6 and wm.nominal_dt_s == 0.5
    assert wm.time_origin is None


def test_factory_configs_are_independent_and_do_not_change_generic_defaults():
    first, second = guangyang_static_world_model(), guangyang_static_world_model()
    first.assoc_cfg.min_hit_pose_gap_m = 9.0
    first.fov_cfg.horizontal_fov_deg = 180.0
    first.decay_cfg.class_half_life_scale["ball"] = 90.0
    assert second.assoc_cfg.min_hit_pose_gap_m == 0.15
    assert second.fov_cfg.horizontal_fov_deg == 75.2
    assert second.decay_cfg.class_half_life_scale == GY_SCALES
    assert WorldModel().decay_cfg.class_half_life_scale == LEGACY_SCALES
    assert FovConfig().horizontal_fov_deg == 70.0
    assert AssociationConfig().min_hit_pose_gap_m == 0.0


def test_static_mean_and_repeat_pose_behavior_require_explicit_opt_in():
    generic, gy = WorldModel(), guangyang_static_world_model()
    for wm in (generic, gy):
        wm.update([target(x=0.0)], RobotPose(x=0.0), now=0.0)
        wm.update([target(x=0.2)], RobotPose(x=0.0), now=0.5)
    assert generic.get_object("target").hit_count == 2
    assert generic.get_object("target").x == pytest.approx(0.12)
    assert gy.get_object("target").hit_count == 1
    assert gy.get_object("target").x == 0.0
    gy.update([target(x=0.2)], RobotPose(x=0.2), now=1.0)
    gy.update([target(x=0.1)], RobotPose(x=0.4), now=1.5)
    assert gy.get_object("target").hit_count == 3
    assert gy.get_object("target").x == pytest.approx(0.1)
    assert gy.get_object("target").state == ObjectState.CONFIRMED


@pytest.mark.parametrize("factory,default_lost", [
    (WorldModel, False), (guangyang_static_world_model, True),
])
def test_archive_access_is_explicit_and_does_not_change_snapshots(factory, default_lost):
    wm = factory()
    wm.update([target()], RobotPose(), now=0.0)
    obj = wm.get_object("target")
    obj_id = obj.obj_id
    assert wm.get_archived(obj_id) is None
    wm.update([], RobotPose(), now=20.0)
    assert obj.state == ObjectState.LOST
    assert wm.get_object(obj_id) is (obj if default_lost else None)
    assert wm.get_object("target") is (obj if default_lost else None)
    assert wm.get_object(obj_id, include_lost=False) is None
    assert wm.get_object("target", include_lost=False) is None
    assert wm.get_object(obj_id, include_lost=True) is obj
    assert wm.get_object("target", include_lost=True) is obj
    assert wm.get_archived(obj_id) is obj
    assert wm.get_archived("target") is None  # Archive API is exact-ID only.
    assert wm.get_archived("missing") is None
    assert wm.get_scene() == wm.snapshot() == wm.to_contract(now=20.0) == []
    history = copy.deepcopy(obj)
    wm.update([target()], RobotPose(x=0.2), now=21.0)
    active = wm.get_object("target")
    assert active.obj_id != obj_id and active.state == ObjectState.TENTATIVE
    assert wm.get_object("target", include_lost=True) is active
    assert wm.get_object(obj_id, include_lost=True) is obj
    assert wm.get_archived(obj_id) == history


def test_gy_fov_and_class_decay_are_explicit_behavior_not_generic_defaults():
    # 36 degrees lies outside generic 35-degree half-FOV and inside GY 37.6.
    x, z = math.sin(math.radians(36)), math.cos(math.radians(36))
    generic, gy = WorldModel(), guangyang_static_world_model()
    for wm in (generic, gy):
        wm.update([target(x=x, z=z)], RobotPose(), now=0.0)
        wm.update([], RobotPose(), now=1.5)
    assert generic.get_object("target").confidence == pytest.approx(0.9 * 0.5 ** (1.5 / 60))
    assert gy.get_object("target").confidence == pytest.approx(0.45)
    for wm in (WorldModel(), guangyang_static_world_model()):
        wm.update([Detection("obstacle", x=0.0, z=0.6, confidence=0.9)],
                  RobotPose(), now=0.0)
        wm.update([], RobotPose(), now=1.5)
        expected = 0.9 * 0.5 ** (1 / 8) if wm.include_lost_in_get_object else 0.45
        assert wm.get_object("obstacle").confidence == pytest.approx(expected)


def test_program_range_override_keeps_outside_window_decay_semantics():
    full = guangyang_static_world_model()
    short = guangyang_static_world_model(max_range_m=0.9)
    for wm in (full, short):
        wm.update([target(z=1.0)], RobotPose(), now=0.0)
        wm.update([], RobotPose(), now=1.5)
    assert full.get_object("target").confidence == pytest.approx(0.45)
    assert short.get_object("target").confidence == pytest.approx(0.9 * 0.5 ** (1.5 / 60))


@pytest.mark.parametrize("factory", [WorldModel, guangyang_static_world_model])
def test_action_removal_is_never_automatic_and_keeps_original_history(factory):
    wm = factory()
    for i in range(3):
        wm.update([target()], RobotPose(x=i * 0.2), now=i * 0.5)
    obj = wm.get_object("target")
    assert obj.state == ObjectState.CONFIRMED
    assert obj.confidence > 0.0 and wm.get_archived(obj.obj_id) is None
    before = copy.deepcopy(obj.__dict__)
    assert wm.mark_removed(obj.obj_id, now=2.0)
    assert wm.get_archived(obj.obj_id) is obj
    assert obj.state == ObjectState.LOST and obj.confidence == 0.0
    for name, value in before.items():
        if name not in {"state", "confidence", "last_updated"}:
            assert getattr(obj, name) == value
    after = copy.deepcopy(obj.__dict__)
    assert wm.mark_removed(obj.obj_id, now=3.0)
    assert obj.__dict__ == after
    assert wm.get_scene() == []
