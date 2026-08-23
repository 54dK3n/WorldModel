"""SizePolicy / ObjectSizeRegistry 的信任边界测试。"""
from __future__ import annotations

import json
import math

import pytest

from world_model.aliases import AliasTable
from world_model.size_policy import (
    ObjectSizeRegistry,
    ProjectionContext,
    SizeEvidence,
    SizePolicy,
)
from world_model.types import (
    RADIUS_SEMANTICS_INNER,
    RADIUS_SEMANTICS_OUTER,
    SIZE_SOURCE_BBOX_HEURISTIC,
    SIZE_SOURCE_EXTERNAL_CLAIM,
    SIZE_SOURCE_LOCAL_REGISTRY,
)


def write_registry(tmp_path, data):
    path = tmp_path / "object_sizes.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def make_registry(tmp_path):
    return ObjectSizeRegistry.from_json(write_registry(tmp_path, {
        "version": 1,
        "unit": "cm",
        "objects": {
            "ball": {"radius_cm": 3.3, "radius_semantics": "outer_radius",
                     "source": "manual_measurement", "reviewed": True},
            "basket": {"radius_cm": 15.0, "radius_semantics": "inner_radius",
                       "source": "manual_measurement", "reviewed": True},
        },
    }))


def test_registry_from_json_alias_path(tmp_path):
    registry = make_registry(tmp_path)
    policy = SizePolicy(registry=registry, aliases=AliasTable())
    for raw in ["tennis_ball", "sports ball", "tennis ball", "ball", "网球", "球", "小球"]:
        ev = policy.resolve(raw, {}, (10, 10, 20, 20), ProjectionContext(1.0, 1.0, 500, 500))
        assert ev.canonical_name == "ball"
        assert ev.radius_cm == pytest.approx(3.3)
        assert ev.radius_semantics == RADIUS_SEMANTICS_OUTER
        assert ev.size_source == SIZE_SOURCE_LOCAL_REGISTRY
        assert ev.size_trusted is True

    for raw in ["basket", "bucket", "篮子", "桶", "框"]:
        ev = policy.resolve(raw, {}, (10, 10, 20, 20), ProjectionContext(1.0, 1.0, 500, 500))
        assert ev.canonical_name == "basket"
        assert ev.radius_cm == pytest.approx(15.0)
        assert ev.radius_semantics == RADIUS_SEMANTICS_INNER
        assert ev.size_source == SIZE_SOURCE_LOCAL_REGISTRY
        assert ev.size_trusted is True


def test_registry_rejects_bad_configs(tmp_path):
    base = {
        "version": 1,
        "unit": "cm",
        "objects": {
            "ball": {"radius_cm": 3.3, "radius_semantics": "outer_radius",
                     "source": "manual_measurement", "reviewed": True},
        },
    }
    bad_cases = []
    b = json.loads(json.dumps(base)); b["unit"] = "m"; bad_cases.append(b)
    b = json.loads(json.dumps(base)); b["objects"]["ball"]["radius_cm"] = -1; bad_cases.append(b)
    b = json.loads(json.dumps(base)); b["objects"]["ball"]["radius_semantics"] = "outer"; bad_cases.append(b)
    b = json.loads(json.dumps(base)); b["objects"]["ball"]["source"] = "unknown_source"; bad_cases.append(b)
    b = json.loads(json.dumps(base)); b["objects"]["ball"]["reviewed"] = "true"; bad_cases.append(b)
    b = json.loads(json.dumps(base)); b["objects"]["ball"]["reviewed"] = False; bad_cases.append(b)
    b = json.loads(json.dumps(base)); b["objects"]["ball"]["unknown_field"] = 1; bad_cases.append(b)
    for bad in bad_cases:
        with pytest.raises(ValueError):
            ObjectSizeRegistry.from_json(write_registry(tmp_path, bad))


def test_external_size_trusted_rejected():
    policy = SizePolicy(registry=ObjectSizeRegistry(), aliases=AliasTable())
    with pytest.raises(ValueError, match="external size_trusted is forbidden"):
        policy.from_detection_record({"size_trusted": True}, canonical_name="ball")
    with pytest.raises(ValueError, match="external size_trusted is forbidden"):
        policy.from_detection_record({"size_trusted": "false"}, canonical_name="ball")


def test_bare_radius_is_external_claim_not_trusted():
    policy = SizePolicy(registry=ObjectSizeRegistry(), aliases=AliasTable())
    ev = policy.resolve(
        "unknown_object",
        {"radius_cm": 150},
        (10, 10, 20, 20),
        ProjectionContext(1.0, 1.0, 500, 500),
    )
    assert ev.size_source == SIZE_SOURCE_EXTERNAL_CLAIM
    assert ev.size_trusted is False
    assert ev.radius_semantics == "unknown"


def test_local_registry_wins_over_external_claim(tmp_path):
    registry = make_registry(tmp_path)
    policy = SizePolicy(registry=registry, aliases=AliasTable())
    ev = policy.resolve(
        "bucket",
        {"radius_cm": 150},
        (10, 10, 20, 20),
        ProjectionContext(1.0, 1.0, 500, 500),
    )
    assert ev.radius_cm == pytest.approx(15.0)
    assert ev.size_source == SIZE_SOURCE_LOCAL_REGISTRY
    assert ev.size_trusted is True


def test_bbox_heuristic_uses_optical_depth_and_is_untrusted():
    policy = SizePolicy(registry=ObjectSizeRegistry(), aliases=AliasTable())
    ev = policy.from_bbox_heuristic(
        (0, 0, 100, 80),
        ProjectionContext(ground_range_m=1.0, optical_axis_depth_m=2.0, fx=500, fy=400),
    )
    assert ev.size_source == SIZE_SOURCE_BBOX_HEURISTIC
    assert ev.size_trusted is False
    # 水平: 100px * 2m / 500 = 0.4m 直径; 垂直: 80px * 2m / 400 = 0.4m 直径 -> 半径 20cm
    assert ev.radius_cm == pytest.approx(20.0, rel=1e-6)


def test_missing_registry_never_trusts_unknown_class():
    policy = SizePolicy(registry=ObjectSizeRegistry(), aliases=AliasTable())
    ev = policy.resolve("unknown_object", {}, (10, 10, 20, 20),
                        ProjectionContext(1.0, 1.0, 500, 500))
    assert ev.size_trusted is False
