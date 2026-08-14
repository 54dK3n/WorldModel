"""对外契约适配层。

内部 TrackedObject 结构与对外契约解耦：
契约怎么定（宋红的 scene_observations 二维平面版 / 任务规划文档的 base_link 三维版），
改的都只是本文件，核心逻辑不动。

待确认（见 DESIGN.md 第六节）：
  - radius_cm 是外接圆半径还是包围盒半宽
  - 坐标系原点是世界固定点还是机器人本体
  - 多观测冲突时以谁为准
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Sequence

from .types import TrackedObject


def _iso(ts: float, time_origin: float = 0.0) -> str:
    """内部时刻 -> ISO8601。

    内部时间是"场景相对秒"（回放/mock 从 0 开始），不是 Unix 时间。
    time_origin 是相对时刻 0.0 对应的 Unix 时间；不传就默认内部时间已经是
    Unix 时间。原来这里直接把相对秒当纪元格式化，导出的时间戳全是
    1970-01-01T00:00:0Xz，下游拿它算陈旧度会得到 56 年。
    """
    return datetime.fromtimestamp(time_origin + ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def to_scene_observation(obj: TrackedObject, time_origin: float = 0.0) -> Dict:
    """宋红的 scene_observations 契约。字段严格对齐，不多不少。

    待确认：契约里没有 id 也没有 state，同名多实例（干扰物、断轨残留）
    下游无法区分。加字段要对方签字，先记在 DESIGN.md 待确认清单里。
    """
    return {
        "name": obj.name,
        "aliases": obj.aliases,
        "x": round(obj.x, 4),
        "z": round(obj.z, 4),
        "radius_cm": round(obj.radius_cm, 2),
        "source": obj.source,
        "timestamp": _iso(obj.last_seen, time_origin),
        "confidence": round(obj.confidence, 4),
    }


def to_scene_observations(objs: Sequence[TrackedObject], time_origin: float = 0.0) -> List[Dict]:
    return [to_scene_observation(o, time_origin) for o in objs]


def to_base_link_pose(obj: TrackedObject, y: float = 0.0, time_origin: float = 0.0) -> Dict:
    """任务规划文档那套三维桌面版契约。备用适配器。"""
    return {
        "id": obj.obj_id,
        "class": obj.name,
        "pose": {"frame": "base_link", "x": obj.x, "y": y, "z": obj.z},
        "state": obj.state.value,
        "confidence": obj.confidence,
        "last_seen": _iso(obj.last_seen, time_origin),
    }


def to_debug_dict(obj: TrackedObject) -> Dict:
    """内部全量字段，仅用于调试和 evidence，不对外承诺。"""
    return {
        "obj_id": obj.obj_id,
        "name": obj.name,
        "x": obj.x,
        "z": obj.z,
        "radius_cm": obj.radius_cm,
        "confidence": obj.confidence,
        "state": obj.state.value,
        "first_seen": obj.first_seen,
        "last_seen": obj.last_seen,
        "hit_count": obj.hit_count,
        "miss_count": obj.miss_count,
        "pose_uncertainty_cm": obj.pose_uncertainty_cm,
        "last_bbox": list(obj.last_bbox) if obj.last_bbox else None,
        "last_frame_id": obj.last_frame_id,
        "source": obj.source,
    }
