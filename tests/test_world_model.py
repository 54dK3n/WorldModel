"""单测覆盖文档要求的四类场景。

    1. 正常跟踪：id 稳定
    2. 遮挡：不能删除对象，只能降置信度 + 记 last_seen
    3. 视野外：慢衰减
    4. 视野内漏检：快衰减
"""
import math

import pytest

from world_model import Detection, RobotPose, WorldModel
from world_model.aliases import AliasTable
from world_model.association import AssociationConfig, associate
from world_model.decay import FovConfig, is_in_fov
from world_model.types import ObjectState


def det(cls="sports ball", x=0.0, z=1.0, conf=0.95):
    return Detection(class_name=cls, x=x, z=z, confidence=conf, radius_cm=3.3)


# ------------------------------------------------------------------ 场景 1

def test_id_stable_across_frames():
    wm = WorldModel()
    wm.update([det(x=0.10, z=1.20)], RobotPose(), now=0.0)
    first_id = wm.get_scene()[0].obj_id
    for i, x in enumerate([0.12, 0.14, 0.16], start=1):
        wm.update([det(x=x, z=1.18)], RobotPose(), now=i * 0.5)
    assert len(wm.get_scene()) == 1
    assert wm.get_scene()[0].obj_id == first_id


def test_confirmed_after_enough_hits():
    wm = WorldModel()
    for i in range(4):
        wm.update([det(x=0.1, z=1.2)], RobotPose(), now=i * 0.3)
    assert wm.get_object("ball").state == ObjectState.CONFIRMED


def test_alias_resolution():
    wm = WorldModel()
    wm.update([det(cls="sports ball")], RobotPose(), now=0.0)
    assert wm.get_object("球") is not None
    assert wm.get_object("网球") is not None
    assert wm.get_object("ball").name == "ball"


# ------------------------------------------------------------------ 场景 2

def test_occlusion_does_not_delete_object():
    """遮挡时对象必须还在，只是置信度下降。"""
    wm = WorldModel()
    wm.update([det(x=0.1, z=1.2)], RobotPose(), now=0.0)
    conf0 = wm.get_object("ball").confidence

    wm.update([], RobotPose(), now=0.5)   # 视野内漏检
    obj = wm.get_object("ball")

    assert obj is not None, "遮挡时不允许删除对象"
    assert obj.confidence < conf0
    assert obj.last_seen == 0.0, "last_seen 应停在最后一次真实看到的时刻"


# --------------------------------------------------------- 场景 3 / 4 对照

def test_out_of_fov_decays_slower_than_missed_in_fov():
    """本项目最关键的一条不变式：两种'没看见'处理方式相反。"""
    dt = 2.0

    wm_in = WorldModel()
    wm_in.update([det(x=0.0, z=1.5)], RobotPose(yaw_rad=0.0), now=0.0)
    wm_in.update([], RobotPose(yaw_rad=0.0), now=dt)          # 一直看着，没检测到
    conf_missed = wm_in.get_object("ball").confidence

    wm_out = WorldModel()
    wm_out.update([det(x=0.0, z=1.5)], RobotPose(yaw_rad=0.0), now=0.0)
    wm_out.update([], RobotPose(yaw_rad=math.pi), now=dt)     # 转身，球出视野
    conf_turned = wm_out.get_object("ball").confidence

    assert conf_turned > conf_missed
    assert conf_turned > 0.9 * 0.95, "转个身不该把世界忘光"


def test_miss_count_only_increments_in_fov():
    wm = WorldModel()
    wm.update([det(x=0.0, z=1.5)], RobotPose(yaw_rad=0.0), now=0.0)
    wm.update([], RobotPose(yaw_rad=math.pi), now=1.0)
    assert wm.get_object("ball").miss_count == 0


def test_object_becomes_lost_and_leaves_snapshot():
    wm = WorldModel()
    wm.update([det(x=0.0, z=1.5)], RobotPose(), now=0.0)
    wm.update([], RobotPose(), now=20.0)   # 视野内长时间漏检
    assert wm.get_object("ball") is None
    assert "ball_001" in wm._lost, "LOST 对象应归档而非物理销毁"


# ------------------------------------------------------------------ FOV

@pytest.mark.parametrize(
    "x,z,yaw,expected",
    [
        (0.0, 1.0, 0.0, True),      # 正前方
        (0.0, -1.0, 0.0, False),    # 正后方
        (0.0, 1.0, math.pi, False), # 转身后
        (0.0, 10.0, 0.0, False),    # 超出量程
        (0.0, 0.05, 0.0, False),    # 过近
    ],
)
def test_is_in_fov(x, z, yaw, expected):
    assert is_in_fov(x, z, RobotPose(yaw_rad=yaw), FovConfig()) is expected


# ------------------------------------------------------------------ 关联

def test_gating_rejects_far_detection():
    wm = WorldModel()
    wm.update([det(x=0.0, z=1.0)], RobotPose(), now=0.0)
    wm.update([det(x=3.0, z=1.0)], RobotPose(), now=0.2)   # 超出门控距离
    assert len(wm.get_scene()) == 2, "超门控的检测应新建对象而非错配"


def test_class_mismatch_not_associated():
    tracks = list(WorldModel().__dict__ and [])   # 占位，见下方直接构造
    wm = WorldModel()
    wm.update([det(cls="sports ball", x=0.0, z=1.0)], RobotPose(), now=0.0)
    wm.update([det(cls="basket", x=0.02, z=1.0)], RobotPose(), now=0.2)
    names = sorted(o.name for o in wm.get_scene())
    assert names == ["ball", "basket"]


def test_greedy_assignment_prefers_nearest():
    wm = WorldModel()
    wm.update(
        [det(x=0.0, z=1.0), det(x=0.4, z=1.0)],
        RobotPose(), now=0.0,
    )
    ids = {round(o.x, 2): o.obj_id for o in wm.get_scene()}
    wm.update(
        [det(x=0.42, z=1.0), det(x=0.03, z=1.0)],   # 顺序打乱
        RobotPose(), now=0.2,
    )
    new_ids = {round(o.x, 1): o.obj_id for o in wm.get_scene()}
    assert len(wm.get_scene()) == 2
    assert set(ids.values()) == set(new_ids.values()), "不应因检测顺序变化而换 id"


# ------------------------------------------------------------------ 契约

def test_scene_observations_schema():
    from world_model import to_scene_observations

    wm = WorldModel()
    wm.update([det(x=0.1, z=1.2)], RobotPose(), now=0.0)
    obs = to_scene_observations(wm.get_scene())[0]
    assert set(obs) == {
        "name", "aliases", "x", "z", "radius_cm",
        "source", "timestamp", "confidence",
    }
