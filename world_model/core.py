"""World Model 主体。

    update(detections, pose, now)  写入：关联 -> 融合 -> 衰减 -> 建/删
    get_scene()                    只读查询：当前场景快照
    get_object(name)               只读查询：单个对象
    snapshot()                     深拷贝快照，给 Judge 做前后差分

只读查询只返回事实，不做任何"能不能执行"的判断 —— 那是编排层的事。
"""
from __future__ import annotations

import copy
import itertools
import logging
import math
import time
from typing import Dict, List, Optional, Sequence

from .aliases import AliasTable
from .association import AssociationConfig, associate
from .decay import DecayConfig, FovConfig, apply_decay, on_hit
from .types import (
    COORDINATE_FRAME_WORLD,
    RADIUS_SEMANTICS_UNKNOWN,
    Detection,
    FrameQuality,
    ObjectState,
    RobotPose,
    TrackedObject,
)

logger = logging.getLogger(__name__)


class WorldModel:
    def __init__(
        self,
        aliases: Optional[AliasTable] = None,
        assoc_cfg: Optional[AssociationConfig] = None,
        decay_cfg: Optional[DecayConfig] = None,
        fov_cfg: Optional[FovConfig] = None,
        visibility: object | None = None,   # 统一可见性模型：CameraCalibration 或 FovConfig
        position_smoothing: float = 0.6,   # 名义帧间隔下的新观测权重，1.0 = 完全信新观测
        nominal_dt_s: float = 0.5,         # position_smoothing 对应的名义帧间隔
        time_origin: Optional[float] = None,  # 内部时刻 0.0 对应的 Unix 时间；None 表示相对时间
    ):
        self.aliases = aliases or AliasTable()
        self.assoc_cfg = assoc_cfg or AssociationConfig()
        self.decay_cfg = decay_cfg or DecayConfig()
        self.fov_cfg = fov_cfg or FovConfig()
        self.visibility = visibility if visibility is not None else self.fov_cfg
        self.position_smoothing = position_smoothing
        self.nominal_dt_s = nominal_dt_s
        self.time_origin = time_origin

        self._objects: Dict[str, TrackedObject] = {}
        self._lost: Dict[str, TrackedObject] = {}
        self._id_counter = itertools.count(1)
        self.pose = RobotPose()
        self.last_update_time: Optional[float] = None

    # ------------------------------------------------------------------ 写入

    def update(
        self,
        detections: Sequence[Detection],
        pose: Optional[RobotPose] = None,
        now: Optional[float] = None,
    ) -> None:
        now = now if now is not None else time.time()
        if pose is not None:
            self.pose = pose
        self.last_update_time = now

        tracks = list(self._objects.values())
        result = associate(tracks, detections, self.aliases, self.assoc_cfg, now=now)

        for t_idx, d_idx in result.matches:
            self._fuse(tracks[t_idx], detections[d_idx], now)

        for t_idx in result.unmatched_tracks:
            apply_decay(tracks[t_idx], now, self.pose, self.visibility, self.decay_cfg)

        for d_idx in result.unmatched_detections:
            self._spawn(detections[d_idx], now)

        # LOST 的移出活跃表，但不物理销毁（文档要求：遮挡不能删除物体）
        self._archive_lost()

    def _smoothing_for(self, dt: float) -> float:
        """把固定权重换成时间常数固定的指数滤波。

        帧间隔 == nominal_dt_s 时退化为 position_smoothing（与原行为逐位一致），
        间隔越长越信新观测。固定权重在长间隔 + 大位移下会把位置留在起终点之间 ——
        球明明进了桶，融合后的坐标却停在半路，containment 判定直接假阴性。
        """
        a0 = self.position_smoothing
        if dt <= 0.0 or a0 >= 1.0:
            return a0
        return min(1.0, 1.0 - (1.0 - a0) ** (dt / self.nominal_dt_s))

    def _fuse(self, obj: TrackedObject, det: Detection, now: float) -> None:
        a = self._smoothing_for(max(0.0, now - obj.last_seen))
        obj.x = a * det.x + (1 - a) * obj.x
        obj.z = a * det.z + (1 - a) * obj.z
        self._fuse_size(obj, det, a)
        # 置信度向观测靠拢，取较高者防止单帧抖动把信念打没
        obj.confidence = max(obj.confidence * 0.3 + det.confidence * 0.7, det.confidence * 0.9)
        obj.confidence = min(obj.confidence, 0.99)
        obj.last_seen = now
        obj.last_updated = now
        obj.source = det.source
        obj.last_bbox = det.bbox
        obj.last_frame_id = det.frame_id
        if det.frame_quality is not None:
            obj.last_frame_quality = det.frame_quality
        obj.pose_uncertainty_cm = self.pose.pose_uncertainty_cm
        on_hit(obj, self.decay_cfg)

    def _fuse_size(self, obj: TrackedObject, det: Detection, a: float) -> None:
        """尺寸证据融合：本地可信尺寸优先，不可信尺寸不能通过多帧融合变可信。"""
        if det.size_trusted and not obj.size_trusted:
            self._set_size(obj, det)
        elif det.size_trusted and obj.size_trusted:
            # 本地 registry 的尺寸不随 bbox 平滑；仅当两者都可信时做保守平均
            obj.radius_cm = a * det.radius_cm + (1 - a) * obj.radius_cm
            obj.size_source = det.size_source
            obj.size_trusted = True
            obj.radius_semantics = det.radius_semantics
        elif not det.size_trusted and obj.size_trusted:
            # 外部声明 / bbox heuristic 不得覆盖本地可信尺寸
            logger.warning(
                "尺寸来源冲突 obj_id=%s：忽略不可信观测 source=%s radius=%.2f，"
                "保留本地可信尺寸 %.2f",
                obj.obj_id, det.size_source, det.radius_cm, obj.radius_cm,
            )
        else:
            obj.radius_cm = a * det.radius_cm + (1 - a) * obj.radius_cm
            obj.size_source = det.size_source
            obj.size_trusted = False
            obj.radius_semantics = det.radius_semantics

    @staticmethod
    def _set_size(obj: TrackedObject, det: Detection) -> None:
        obj.radius_cm = det.radius_cm
        obj.size_source = det.size_source
        obj.size_trusted = det.size_trusted
        obj.radius_semantics = det.radius_semantics

    def _spawn(self, det: Detection, now: float) -> None:
        name = self.aliases.canonical(det.class_name)
        obj_id = f"{name}_{next(self._id_counter):03d}"
        self._objects[obj_id] = TrackedObject(
            obj_id=obj_id,
            name=name,
            aliases=self.aliases.aliases_of(name),
            x=det.x,
            z=det.z,
            radius_cm=det.radius_cm,
            confidence=det.confidence,
            size_source=det.size_source,
            size_trusted=det.size_trusted,
            radius_semantics=det.radius_semantics,
            first_seen=now,
            last_seen=now,
            last_updated=now,
            hit_count=1,
            state=ObjectState.TENTATIVE,
            pose_uncertainty_cm=self.pose.pose_uncertainty_cm,
            source=det.source,
            last_bbox=det.bbox,
            last_frame_id=det.frame_id,
            last_frame_quality=det.frame_quality,
        )

    def _archive_lost(self) -> None:
        for oid in [k for k, v in self._objects.items() if v.state == ObjectState.LOST]:
            self._lost[oid] = self._objects.pop(oid)

    # ------------------------------------------------------------------ 只读

    def get_scene(self, min_confidence: float = 0.0) -> List[TrackedObject]:
        return [o for o in self._objects.values() if o.confidence >= min_confidence]

    def get_object(self, name_or_id: str) -> Optional[TrackedObject]:
        """按 obj_id 精确查，或按规范名/别名查置信度最高的一个。"""
        if name_or_id in self._objects:
            return self._objects[name_or_id]
        canonical = self.aliases.canonical(name_or_id)
        candidates = [o for o in self._objects.values() if o.name == canonical]
        return max(candidates, key=lambda o: o.confidence) if candidates else None

    def snapshot(self) -> List[TrackedObject]:
        """深拷贝，给 Judge 做动作前后差分。"""
        return copy.deepcopy(list(self._objects.values()))

    def to_contract(
        self,
        min_confidence: float = 0.5,
        max_staleness_s: float = 1.0,
        now: Optional[float] = None,
        require_size_trusted: bool = True,
    ) -> List[Dict]:
        """导出正式 scene_observations。

        默认契约过滤：仅 CONFIRMED、置信度达标、未过期、尺寸可信、语义正确、
        帧质量完整、标定可信、坐标为 world、坐标有限。
        调试请直接使用 get_scene() / snapshot()，不要把 TENTATIVE 误检当正式输出。
        """
        from .adapters import to_scene_observations
        now = now if now is not None else (self.last_update_time if self.last_update_time is not None else time.time())
        objs = []
        for o in self._objects.values():
            reasons = self._contract_exclude_reasons(
                o, now, min_confidence, max_staleness_s, require_size_trusted
            )
            if reasons:
                logger.debug(
                    "exclude obj_id=%s reasons=%s",
                    o.obj_id, ",".join(reasons),
                )
                continue
            objs.append(o)
        return to_scene_observations(objs, time_origin=self.time_origin)

    def _contract_exclude_reasons(
        self,
        o: TrackedObject,
        now: float,
        min_confidence: float,
        max_staleness_s: float,
        require_size_trusted: bool,
    ) -> List[str]:
        reasons: List[str] = []
        if o.state != ObjectState.CONFIRMED:
            reasons.append("unconfirmed_track")
        if o.confidence < min_confidence:
            reasons.append("low_confidence")
        age = o.age(now)
        if age < 0.0:
            reasons.append("clock_skew")
        elif age > max_staleness_s:
            reasons.append("stale_evidence")
        if require_size_trusted and not o.size_trusted:
            reasons.append("untrusted_size_evidence")
        if o.radius_semantics not in ("outer_radius", "inner_radius"):
            reasons.append("invalid_size_semantics")
        fq = o.last_frame_quality
        if fq is None:
            reasons.append("missing_frame_quality")
        else:
            if fq.degraded:
                reasons.append("degraded_frame")
            if fq.frames_skipped > 0:
                reasons.append("frames_skipped")
            if fq.detections_skipped > 0:
                reasons.append("detections_skipped")
            if fq.calibration_trusted is not True:
                reasons.append("untrusted_calibration")
            if fq.coordinate_frame != COORDINATE_FRAME_WORLD:
                reasons.append("unknown_coordinate_frame")
        if not (math.isfinite(o.x) and math.isfinite(o.z)):
            reasons.append("non_finite_coordinates")
        return reasons
