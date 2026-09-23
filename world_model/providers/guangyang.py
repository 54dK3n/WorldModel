"""Guangyang platform PerceptionProvider.

本文件把广阳岛公开的 ``robot.observe()`` / ``robot.odometry()`` 数据转换成
WorldModel 的 ``Detection`` / ``RobotPose``。它不读取地图真值，只消费公开字段。

平台坐标约定（已由平台探针实测冻结）：

* ``odometry.headingDeg``：左转为正；
* ``odometry.rightCm``：右为正；
* ``observation.bearingDeg``：右为正。

WorldModel 内部帧约定：

* ``x`` 向右，``z`` 向前；
* ``RobotPose.yaw_rad`` 的正方向定义为**右转**（从 +z 转向 +x）。

因此平台航向到内部航向只做一处转换：

    yaw_rad = -radians(heading_deg)

检测极坐标到世界坐标：

    x = pose.x + distance_m * sin(pose.yaw_rad + bearing_rad)
    z = pose.z + distance_m * cos(pose.yaw_rad + bearing_rad)

平台报告的 ``forwardCm`` / ``rightCm`` 是相对起点车体坐标系的位移，因此
内部帧直接映射为 x=right、z=forward。
"""
from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from ..association import AssociationConfig
from ..types import (
    COORDINATE_FRAME_WORLD,
    RADIUS_SEMANTICS_OUTER,
    SIZE_SOURCE_LOCAL_REGISTRY,
    Detection,
    FrameQuality,
    RobotPose,
)
from .base import PerceptionProvider


CENTIMETERS_PER_METER = 100.0
DEFAULT_SOURCE = "guangyang"

logger = logging.getLogger(__name__)

# Platform physical widths documented in the integration plan.  Radius is half
# of the documented width, in centimetres.
GUANGYANG_CLASS_WIDTH_M: Dict[str, float] = {
    "target": 0.44,
    "distractor": 0.44,
    "obstacle": 0.46,
    "storage-zone": 0.38,
    "cleanup-zone": 0.38,
}

GUANGYANG_CLASS_RADIUS_CM: Dict[str, float] = {
    category: width_m * CENTIMETERS_PER_METER / 2.0
    for category, width_m in GUANGYANG_CLASS_WIDTH_M.items()
}

# 静态场景门控：广阳岛任务物在被抓取前不动。
# 取值依据：10 套布局同类物体最小间距为 0.65 m，门控上限必须小于该间距的一半
# （0.65 / 2 = 0.325 m）。这里取 0.30 m，严格满足 <= 0.30 m 的门限。
# 该配置只供广阳岛静态场景使用；mock 动态场景继续使用 AssociationConfig 默认值。
GUANGYANG_MIN_SAME_CLASS_GAP_M = 0.65
GUANGYANG_STATIC_ASSOCIATION_CONFIG = AssociationConfig(
    gate_distance_m=0.30,
    max_speed_mps=0.0,
    max_gate_distance_m=0.30,
    static_equal_weight=True,
    min_hit_pose_gap_m=0.15,
)


def guangyang_static_association_config() -> AssociationConfig:
    """返回广阳岛静态场景门控配置的独立副本。"""
    return AssociationConfig(
        gate_distance_m=GUANGYANG_STATIC_ASSOCIATION_CONFIG.gate_distance_m,
        max_speed_mps=GUANGYANG_STATIC_ASSOCIATION_CONFIG.max_speed_mps,
        max_gate_distance_m=GUANGYANG_STATIC_ASSOCIATION_CONFIG.max_gate_distance_m,
        class_mismatch_penalty=GUANGYANG_STATIC_ASSOCIATION_CONFIG.class_mismatch_penalty,
        allow_cross_class=GUANGYANG_STATIC_ASSOCIATION_CONFIG.allow_cross_class,
        appearance_weight=GUANGYANG_STATIC_ASSOCIATION_CONFIG.appearance_weight,
        static_equal_weight=GUANGYANG_STATIC_ASSOCIATION_CONFIG.static_equal_weight,
        min_hit_pose_gap_m=GUANGYANG_STATIC_ASSOCIATION_CONFIG.min_hit_pose_gap_m,
    )


def warn_if_gate_exceeds_half_min_gap(
    min_same_class_gap_m: float,
    cfg: AssociationConfig,
) -> bool:
    """启动期检查：门控上限必须小于场景内同类物体最小间距的一半。

    返回 True 表示配置安全，False 表示已打印警告。本函数不调用真值接口，
    只接受调用方从任务/场景配置中传入的已知最小间距。
    """
    threshold = 2.0 * float(cfg.max_gate_distance_m)
    if float(min_same_class_gap_m) < threshold:
        logger.warning(
            "关联门控上限 %.3f m 不满足静态场景要求：已知同类最小间距 %.3f m "
            "必须 >= 2 * max_gate_distance_m = %.3f m；存在错并风险",
            cfg.max_gate_distance_m,
            min_same_class_gap_m,
            threshold,
        )
        return False
    return True

_CATEGORY_ALIASES = {
    "target": "target",
    "目标物": "target",
    "红球": "target",
    "distractor": "distractor",
    "混淆物": "distractor",
    "干扰物": "distractor",
    "obstacle": "obstacle",
    "障碍物": "obstacle",
    "障碍": "obstacle",
    "storage-zone": "storage-zone",
    "storage_zone": "storage-zone",
    "存放点": "storage-zone",
    "存放区": "storage-zone",
    "cleanup-zone": "cleanup-zone",
    "cleanup_zone": "cleanup-zone",
    "清理点": "cleanup-zone",
    "清理区": "cleanup-zone",
}


def normalize_guangyang_category(value: Any) -> Optional[str]:
    """Return the canonical Guangyang category, or None for unsupported input."""
    if value is None:
        return None
    raw = str(value).strip()
    if raw == "":
        return None
    lowered = raw.lower().replace("_", "-")
    return _CATEGORY_ALIASES.get(raw) or _CATEGORY_ALIASES.get(lowered)


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be a finite number, got {}".format(label, value)) from exc
    if not math.isfinite(number):
        raise ValueError("{} must be a finite number, got {}".format(label, value))
    return number


@dataclass
class GuangyangNoiseConfig:
    """Configurable odometry noise for robustness tests.

    ``distance_relative_std`` applies a multiplicative distance error on each
    odometry read.  ``heading_deg_std`` adds Gaussian heading error on each read.
    A fixed seed makes a test run reproducible.
    """

    distance_relative_std: float = 0.0
    heading_deg_std: float = 0.0
    seed: Optional[int] = None


@dataclass
class GuangyangFrame:
    """One raw frame from a platform JSONL recording."""

    timestamp: float
    odometry: Mapping[str, Any]
    observations: Sequence[Mapping[str, Any]] = field(default_factory=list)
    pose: Optional[RobotPose] = None


def odometry_to_pose(
    odometry: Mapping[str, Any],
    noise: Optional[GuangyangNoiseConfig] = None,
    rng: Optional[random.Random] = None,
) -> RobotPose:
    """Convert ``robot.odometry()`` payload to a WorldModel ``RobotPose``.

    The platform returns centimetres and degrees.  The adapter maps its
    initial-frame ``rightCm`` / ``forwardCm`` axes to WorldModel x / z.
    """
    if not isinstance(odometry, Mapping):
        raise TypeError("odometry payload must be a mapping")
    forward_cm = _finite_float(odometry.get("forwardCm", 0.0), "odometry.forwardCm")
    right_cm = _finite_float(odometry.get("rightCm", 0.0), "odometry.rightCm")
    heading_deg = _finite_float(odometry.get("headingDeg", 0.0), "odometry.headingDeg")
    distance_cm = _finite_float(odometry.get("distanceCm", math.hypot(forward_cm, right_cm)),
                                "odometry.distanceCm")

    random_source = rng
    if noise is not None and (noise.distance_relative_std or noise.heading_deg_std):
        if random_source is None:
            random_source = random.Random(noise.seed)
    scale = 1.0
    if noise is not None and noise.distance_relative_std:
        scale = max(0.0, 1.0 + random_source.gauss(0.0, noise.distance_relative_std))
    # 平台 headingDeg 左转为正；内部 yaw_rad 右转为正。
    # 这是整个项目里唯一一次航向符号转换，不允许在其它层再次取反。
    heading_rad = -math.radians(heading_deg)
    if noise is not None and noise.heading_deg_std:
        heading_rad += math.radians(random_source.gauss(0.0, noise.heading_deg_std))

    pose = RobotPose(
        x=right_cm / CENTIMETERS_PER_METER * scale,
        z=forward_cm / CENTIMETERS_PER_METER * scale,
        yaw_rad=heading_rad,
    )
    if noise is not None and noise.distance_relative_std:
        pose.pose_uncertainty_cm = distance_cm * abs(noise.distance_relative_std)
    return pose


def observation_to_detection(
    observation: Mapping[str, Any],
    pose: RobotPose,
    timestamp: float = 0.0,
    source: str = DEFAULT_SOURCE,
    default_confidence: float = 0.0,
    class_name: Optional[str] = None,
    distance_scale: float = 1.0,
) -> Detection:
    """Convert one public ``robot.observe()`` item to a ``Detection``.

    Required platform fields are ``distanceCm`` and, after the P1 platform
    change, ``bearingDeg``.  A missing ``bearingDeg`` is accepted only when the
    three-way ``direction`` field provides a coarse fallback; callers that need
    the P1 precision should supply ``bearingDeg``.
    """
    if not isinstance(observation, Mapping):
        raise TypeError("observation must be a mapping")
    if not isinstance(pose, RobotPose):
        raise TypeError("pose must be a RobotPose")

    category = normalize_guangyang_category(class_name)
    if category is None:
        category = normalize_guangyang_category(observation.get("category"))
    if category is None:
        category = normalize_guangyang_category(observation.get("name"))
    if category is None:
        category = normalize_guangyang_category(observation.get("label"))
    if category is None:
        raise ValueError("unsupported Guangyang observation category: {}".format(observation))

    distance_cm = _finite_float(observation.get("distanceCm"), "observation.distanceCm")
    if distance_cm < 0.0:
        raise ValueError("observation.distanceCm must be non-negative")
    scale = _finite_float(distance_scale, "distance_scale")
    if scale <= 0.0:
        raise ValueError("distance_scale must be positive")
    distance_m = distance_cm * scale / CENTIMETERS_PER_METER

    bearing_value = observation.get("bearingDeg")
    if bearing_value is None or bearing_value == "":
        direction = str(observation.get("direction") or "中间").strip()
        bearing_deg = {"左": -30.0, "中间": 0.0, "右": 30.0}.get(direction)
        if bearing_deg is None:
            raise ValueError("observation.bearingDeg is required when direction is unknown")
    else:
        bearing_deg = _finite_float(bearing_value, "observation.bearingDeg")

    bearing_rad = math.radians(bearing_deg)
    x = pose.x + distance_m * math.sin(pose.yaw_rad + bearing_rad)
    z = pose.z + distance_m * math.cos(pose.yaw_rad + bearing_rad)
    confidence = _finite_float(observation.get("confidence", default_confidence),
                               "observation.confidence")
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("observation.confidence must be in [0, 1]")

    # 平台公开 observe() 没有 frameId 字段，但 provider 仍是可信公开观测源。
    # 为了让 to_contract() 能拿到完整帧证据，这里始终生成 FrameQuality；
    # 没有 frameId 时用时间戳构造确定性帧引用。
    frame_id_value = observation.get("frameId", observation.get("frame_id"))
    frame_id = None if frame_id_value is None else str(frame_id_value).strip() or None
    if frame_id is None:
        frame_id = "guangyang:{:.6f}".format(float(timestamp))
    frame_quality = FrameQuality(
        frame_id=frame_id,
        calibration_trusted=True,
        coordinate_frame=COORDINATE_FRAME_WORLD,
    )

    return Detection(
        class_name=category,
        x=x,
        z=z,
        confidence=confidence,
        radius_cm=GUANGYANG_CLASS_RADIUS_CM.get(category, 5.0),
        size_source=SIZE_SOURCE_LOCAL_REGISTRY,
        size_trusted=True,
        radius_semantics=RADIUS_SEMANTICS_OUTER,
        bbox=None,
        frame_id=frame_id,
        source=source,
        timestamp=float(timestamp),
        frame_quality=frame_quality,
    )


@dataclass(frozen=True)
class GuangyangArtifactEntry:
    """平台地图纹理伪检测的固定世界位置条目。"""

    category: str
    x: float
    z: float
    radius_m: float = 0.10


class GuangyangArtifactFilter:
    """provider 层地图纹理伪检测过滤器。

    条目由测试侧离线标定脚本从平台原始观测中生成，只记录固定伪检测的
    类别和世界位置；运行时不读取真值，也不参与 planning/selection 决策。
    被过滤数量通过 ``filtered_counts`` 统计并写入验收报告。
    """

    _CELL_M = 0.25

    def __init__(self, entries: Sequence[GuangyangArtifactEntry] = (),
                 allowed_categories: Optional[Sequence[str]] = None):
        self.entries = list(entries)
        self.allowed_categories = None if allowed_categories is None else set(allowed_categories)
        self.filtered_counts: Dict[str, int] = {}
        self._grid: Dict[Tuple[int, int], List[GuangyangArtifactEntry]] = {}
        for entry in self.entries:
            key = (int(math.floor(entry.x / self._CELL_M)), int(math.floor(entry.z / self._CELL_M)))
            self._grid.setdefault(key, []).append(entry)

    @classmethod
    def from_path(cls, path: str, allowed_categories: Optional[Sequence[str]] = None) -> "GuangyangArtifactFilter":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        entries = []
        for item in payload.get("entries", []):
            entries.append(
                GuangyangArtifactEntry(
                    category=str(item["category"]),
                    x=float(item["x"]),
                    z=float(item["z"]),
                    radius_m=float(item.get("radius_m", 0.10)),
                )
            )
        return cls(entries, allowed_categories=allowed_categories)

    def _nearby_entries(self, x: float, z: float) -> Iterable[GuangyangArtifactEntry]:
        cell_x = int(math.floor(x / self._CELL_M))
        cell_z = int(math.floor(z / self._CELL_M))
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for entry in self._grid.get((cell_x + dx, cell_z + dz), ()):
                    yield entry

    def matches(self, category: str, x: float, z: float) -> bool:
        for entry in self._nearby_entries(x, z):
            if entry.category != category:
                continue
            if math.hypot(entry.x - x, entry.z - z) <= entry.radius_m:
                return True
        return False

    def filter_detections(self, detections: Sequence[Detection]) -> List[Detection]:
        kept: List[Detection] = []
        for detection in detections:
            if self.allowed_categories is not None and detection.class_name not in self.allowed_categories:
                self.filtered_counts[detection.class_name] = self.filtered_counts.get(detection.class_name, 0) + 1
                continue
            if self.matches(detection.class_name, detection.x, detection.z):
                self.filtered_counts[detection.class_name] = self.filtered_counts.get(detection.class_name, 0) + 1
            else:
                kept.append(detection)
        return kept

    @property
    def filtered_total(self) -> int:
        return sum(self.filtered_counts.values())


def _assert_frame_convention() -> None:
    """provider 启动自检：平台航向/bearing 符号必须与内部帧一致。

    这组输入来自平台探针实测约定，不是真值布局。任何一处符号处理回归，
    provider 初始化时立即抛异常，避免静默左右镜像。
    """
    cases = (
        ({"forwardCm": 0, "rightCm": 0, "headingDeg": 0, "distanceCm": 0},
         {"category": "target", "distanceCm": 100, "bearingDeg": 0.0, "confidence": 0.9}, (0.0, 1.0)),
        ({"forwardCm": 0, "rightCm": 0, "headingDeg": 0, "distanceCm": 0},
         {"category": "target", "distanceCm": 100, "bearingDeg": 90.0, "confidence": 0.9}, (1.0, 0.0)),
        ({"forwardCm": 0, "rightCm": 0, "headingDeg": 90, "distanceCm": 0},
         {"category": "target", "distanceCm": 100, "bearingDeg": 0.0, "confidence": 0.9}, (-1.0, 0.0)),
        ({"forwardCm": 0, "rightCm": 0, "headingDeg": -90, "distanceCm": 0},
         {"category": "target", "distanceCm": 100, "bearingDeg": 0.0, "confidence": 0.9}, (1.0, 0.0)),
        ({"forwardCm": 0, "rightCm": 0, "headingDeg": 180, "distanceCm": 0},
         {"category": "target", "distanceCm": 100, "bearingDeg": 0.0, "confidence": 0.9}, (0.0, -1.0)),
    )
    failures = []
    for odometry, observation, expected in cases:
        pose = odometry_to_pose(odometry)
        det = observation_to_detection(observation, pose, timestamp=0.0)
        if abs(det.x - expected[0]) > 1e-6 or abs(det.z - expected[1]) > 1e-6:
            failures.append(
                "heading={} bearing={} -> ({:.3f},{:.3f}) 期望 ({:.3f},{:.3f})".format(
                    odometry.get("headingDeg"), observation.get("bearingDeg"),
                    det.x, det.z, expected[0], expected[1],
                )
            )
    if failures:
        raise ValueError("GuangyangProvider 坐标约定自检失败：\n" + "\n".join(failures))


class GuangyangProvider(PerceptionProvider):
    """PerceptionProvider over platform public observation frames.

    Parameters
    ----------
    frames:
        Optional iterable of raw frames.  Each item may be a ``GuangyangFrame``
        or a mapping shaped like ``{"timestamp": ..., "odometry": {...},
        "observations": [...]}``.
    noise:
        Optional ``GuangyangNoiseConfig`` or mapping used to perturb odometry.
    """

    def __init__(
        self,
        frames: Optional[Iterable[Any]] = None,
        noise: Optional[Any] = None,
        source: str = DEFAULT_SOURCE,
        known_min_same_class_gap_m: Optional[float] = None,
        association_config: Optional[AssociationConfig] = None,
    ):
        _assert_frame_convention()
        if known_min_same_class_gap_m is not None:
            warn_if_gate_exceeds_half_min_gap(
                known_min_same_class_gap_m,
                association_config or GUANGYANG_STATIC_ASSOCIATION_CONFIG,
            )
        self.source = source
        if isinstance(noise, GuangyangNoiseConfig) or noise is None:
            self.noise = noise
        elif isinstance(noise, Mapping):
            self.noise = GuangyangNoiseConfig(
                distance_relative_std=float(noise.get("distance_relative_std", 0.0)),
                heading_deg_std=float(noise.get("heading_deg_std", 0.0)),
                seed=None if noise.get("seed") is None else int(noise["seed"]),
            )
        else:
            raise TypeError("noise must be a GuangyangNoiseConfig, mapping, or None")
        self.frames: List[Any] = list(frames or [])
        self._rng = random.Random(self.noise.seed) if self.noise is not None else None

    def _coerce_frame(self, frame: Any) -> GuangyangFrame:
        if isinstance(frame, GuangyangFrame):
            return frame
        if not isinstance(frame, Mapping):
            raise TypeError("frame must be a GuangyangFrame or mapping")
        timestamp = _finite_float(frame.get("timestamp", frame.get("t", 0.0)), "frame.timestamp")
        odometry = frame.get("odometry")
        if odometry is None:
            odometry = frame.get("robot_odometry", {})
        if not isinstance(odometry, Mapping):
            raise TypeError("frame.odometry must be a mapping")
        observations = frame.get("observations", frame.get("detections", []))
        if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
            raise TypeError("frame.observations must be a sequence")
        explicit_pose = frame.get("pose")
        pose = explicit_pose if isinstance(explicit_pose, RobotPose) else None
        return GuangyangFrame(float(timestamp), odometry, observations, pose)

    def odometry(self, payload: Optional[Mapping[str, Any]] = None,
                 timestamp: Optional[float] = None) -> RobotPose:
        """Convert an odometry payload; with no payload, use the current frame."""
        if payload is None:
            if not self.frames:
                raise ValueError("no odometry payload and no frame is pending")
            frame = self._coerce_frame(self.frames[0])
            payload = frame.odometry
        return odometry_to_pose(payload, self.noise, self._rng)

    def observe(self, observation: Mapping[str, Any],
                pose: Optional[RobotPose] = None,
                timestamp: float = 0.0) -> Detection:
        """Convert one observation.  A pose may be supplied explicitly."""
        if pose is None:
            if not self.frames:
                raise ValueError("observe() needs an explicit pose when no frame is pending")
            frame = self._coerce_frame(self.frames[0])
            pose = frame.pose or odometry_to_pose(frame.odometry, self.noise, self._rng)
        return observation_to_detection(
            observation, pose, timestamp=timestamp, source=self.source,
        )

    def stream(self) -> Iterator[Tuple[float, RobotPose, List[Detection]]]:
        for raw_frame in self.frames:
            frame = self._coerce_frame(raw_frame)
            pose = frame.pose or odometry_to_pose(frame.odometry, self.noise, self._rng)
            detections: List[Detection] = []
            for observation in frame.observations:
                detections.append(
                    observation_to_detection(
                        observation, pose, timestamp=frame.timestamp, source=self.source,
                    )
                )
            yield frame.timestamp, pose, detections


__all__ = [
    "CENTIMETERS_PER_METER",
    "DEFAULT_SOURCE",
    "GUANGYANG_CLASS_WIDTH_M",
    "GUANGYANG_CLASS_RADIUS_CM",
    "GuangyangNoiseConfig",
    "GuangyangFrame",
    "GuangyangProvider",
    "normalize_guangyang_category",
    "odometry_to_pose",
    "observation_to_detection",
    "GUANGYANG_MIN_SAME_CLASS_GAP_M",
    "GUANGYANG_STATIC_ASSOCIATION_CONFIG",
    "guangyang_static_association_config",
    "warn_if_gate_exceeds_half_min_gap",
    "GuangyangArtifactEntry",
    "GuangyangArtifactFilter",
]
