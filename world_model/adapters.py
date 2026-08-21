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
from typing import Dict, List, Sequence, Tuple

from .raw import RawDetection
from .types import Detection, TrackedObject


def _iso(ts: float, time_origin: float = 0.0) -> str:
    """内部时刻 -> ISO8601。

    内部时间是"场景相对秒"（回放/mock 从 0 开始），不是 Unix 时间。
    time_origin 是相对时刻 0.0 对应的 Unix 时间；不传就默认内部时间已经是
    Unix 时间。原来这里直接把相对秒当纪元格式化，导出的时间戳全是
    1970-01-01T00:00:0Xz，下游拿它算陈旧度会得到 56 年。
    """
    return datetime.fromtimestamp(time_origin + ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def bbox_bottom_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """检测框底边中点：u = (x1 + x2) / 2, v = y2。"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


def raw_detection_to_detection(
    raw: RawDetection,
    x: float,
    z: float,
    source: str = "camera",
) -> Detection:
    """外部原始检测 -> wm_kit.Detection，保留全部证据字段。"""
    return Detection(
        class_name=raw.class_name,
        x=float(x),
        z=float(z),
        confidence=raw.confidence,
        bbox=raw.bbox,
        frame_id=raw.frame_id,
        source=source,
        timestamp=raw.timestamp,
    )


def sg2002_xywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """SG2002 的 sg2002_tpu.decode_nms 输出 cx/cy/w/h（模型输入尺寸），
    这里只做格式转换：cx,cy,w,h -> xyxy。坐标还原（letterbox）需另行处理。"""
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def letterbox_bbox_to_original(
    bbox_xyxy: Tuple[float, float, float, float],
    scale: float,
    pad_left: float,
    pad_top: float,
) -> Tuple[float, float, float, float]:
    """letterbox 输入尺寸的 xyxy -> 原始图像 xyxy。

    参考外部仓库 pc_tools/preprocess_images.py：
      new_w = min(target_size, round(scale * orig_w))
      canvas.paste(resized, (pad_left, pad_top))
    还原公式：x_orig = (x_letterbox - pad_left) / scale
    """
    x1, y1, x2, y2 = bbox_xyxy
    return (
        (x1 - pad_left) / scale,
        (y1 - pad_top) / scale,
        (x2 - pad_left) / scale,
        (y2 - pad_top) / scale,
    )



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
        "timestamp": obj.last_seen,
        "hit_count": obj.hit_count,
        "miss_count": obj.miss_count,
        "pose_uncertainty_cm": obj.pose_uncertainty_cm,
        "last_bbox": list(obj.last_bbox) if obj.last_bbox else None,
        "last_frame_id": obj.last_frame_id,
        "source": obj.source,
    }
