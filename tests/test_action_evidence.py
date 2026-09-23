"""Explicit action evidence is separate from vision fusion and decay."""
import copy
import math

import pytest

from world_model import Detection, FrameQuality, RobotPose, WorldModel
from world_model.association import AssociationConfig
from world_model.types import ObjectState


def confirmed_pair():
    wm = WorldModel(assoc_cfg=AssociationConfig(min_hit_pose_gap_m=0.15))
    for i in range(3):
        trusted = dict(size_trusted=True, radius_semantics="outer_radius",
                       frame_quality=FrameQuality(frame_id=str(i),
                                                  calibration_trusted=True))
        wm.update([
            Detection(class_name="target", x=0.1, z=1.2, confidence=0.9, **trusted),
            Detection(class_name="target", x=2.1, z=1.2, confidence=0.9, **trusted),
        ], RobotPose(x=i * 0.2), now=float(i))
    first, second = sorted(wm.get_scene(), key=lambda obj: obj.x)
    assert first.state == second.state == ObjectState.CONFIRMED
    return wm, first.obj_id, second.obj_id


def test_verified_removal_archives_without_changing_observation_history():
    wm, selected_id, other_id = confirmed_pair()
    selected_before = copy.deepcopy(wm.get_object(selected_id).__dict__)
    other_before = copy.deepcopy(wm.get_object(other_id).__dict__)
    hit_poses_before = copy.deepcopy(wm._hit_poses)
    pose_before = copy.deepcopy(wm.pose)
    assert len(wm.to_contract(now=3.0)) == 2

    assert wm.mark_removed(selected_id, now=3.0) is True

    archived = wm.get_archived(selected_id)
    assert archived.state == ObjectState.LOST
    assert archived.confidence == 0.0
    assert archived.last_updated == wm.last_update_time == 3.0
    for field, value in selected_before.items():
        if field not in {"state", "confidence", "last_updated"}:
            assert getattr(archived, field) == value, field
    assert wm.get_object(other_id).__dict__ == other_before
    assert wm._hit_poses == hit_poses_before
    assert wm.pose == pose_before
    assert selected_id not in {obj.obj_id for obj in wm.snapshot()}
    assert selected_id not in {obj.obj_id for obj in wm.get_scene()}
    # Existing external contract has no ID; prove only the other location remains.
    assert [obj["x"] for obj in wm.to_contract(now=3.0)] == [2.1]
    assert wm._lost[selected_id] is archived


def test_repeated_action_evidence_is_idempotent():
    wm, selected_id, _ = confirmed_pair()
    assert wm.mark_removed(selected_id, now=3.0)
    before = copy.deepcopy((wm._objects, wm._lost, wm._hit_poses,
                            wm.last_update_time))
    assert wm.mark_removed(selected_id, now=9.0) is True
    assert (wm._objects, wm._lost, wm._hit_poses, wm.last_update_time) == before


@pytest.mark.parametrize("identifier", ["target", "目标物", "missing_999"])
def test_unknown_or_category_is_not_resolved(identifier):
    wm, _, _ = confirmed_pair()
    before = copy.deepcopy((wm._objects, wm._lost, wm._hit_poses,
                            wm.last_update_time))
    assert wm.mark_removed(identifier, now=3.0) is False
    assert (wm._objects, wm._lost, wm._hit_poses, wm.last_update_time) == before


@pytest.mark.parametrize("now", [math.nan, math.inf, -math.inf])
def test_nonfinite_action_time_fails_without_mutating_any_track(now):
    wm, selected_id, _ = confirmed_pair()
    before = copy.deepcopy((wm._objects, wm._lost, wm._hit_poses,
                            wm.last_update_time))
    with pytest.raises(ValueError, match="finite"):
        wm.mark_removed(selected_id, now=now)
    assert (wm._objects, wm._lost, wm._hit_poses, wm.last_update_time) == before


def test_later_observation_does_not_revive_removed_identity():
    wm, selected_id, _ = confirmed_pair()
    wm.mark_removed(selected_id, now=3.0)
    archived_before = copy.deepcopy(wm.get_archived(selected_id).__dict__)
    wm.update([Detection(class_name="target", x=0.1, z=1.2, confidence=0.9)],
              RobotPose(x=0.6), now=4.0)
    new_track = min(wm.get_scene(), key=lambda obj: obj.x)
    assert new_track.obj_id != selected_id
    assert new_track.hit_count == 1
    assert new_track.state == ObjectState.TENTATIVE
    assert wm.get_archived(selected_id).__dict__ == archived_before


def test_action_evidence_can_zero_an_already_decayed_archive():
    wm, selected_id, _ = confirmed_pair()
    obj = wm.get_object(selected_id)
    # Reproduce the existing archive state, independent of decay parameters.
    obj.state, obj.confidence = ObjectState.LOST, 0.01
    wm._archive_lost()
    seen, position = obj.last_seen, (obj.x, obj.z)
    assert wm.mark_removed(selected_id, now=3.0)
    assert obj.confidence == 0.0 and obj.state == ObjectState.LOST
    assert obj.last_seen == seen and (obj.x, obj.z) == position


def test_other_track_next_fusion_is_identical_to_control():
    wm, selected_id, other_id = confirmed_pair()
    control = copy.deepcopy(wm)
    wm.mark_removed(selected_id, now=3.0)
    observations = [Detection(class_name="target", x=2.2, z=1.2,
                              confidence=0.87)]
    for model in (wm, control):
        model.update(observations, RobotPose(x=0.6), now=4.0)
    assert wm.get_object(other_id) == control.get_object(other_id)


@pytest.mark.parametrize("holding,expected", [(None, False), ("混淆物", False),
                                              ("目标物", True)])
def test_public_holding_evidence_caller_boundary(holding, expected):
    """Abstract the public holding result; do not infer success from grab()."""
    wm, selected_id, _ = confirmed_pair()
    # The production caller owns this check and locks the selected exact ID.
    # WM has no robot/truth dependency and must not manufacture action evidence.
    if holding == "目标物":
        wm.mark_removed(selected_id, now=3.0)
    assert (wm.get_object(selected_id, include_lost=True).state == ObjectState.LOST) is expected
