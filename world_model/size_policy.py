"""尺寸证据策略与本地对象尺寸注册表。

信任边界：
  - 相机/YOLO/公开数据集/回放文件只能提供原始检测、bbox、类别和检测置信度，
    不允许自行声明物理尺寸可信。
  - 本地对象尺寸配置（人工测量、产品规格、竞赛规则、已审核实例配置）
    是唯一可以产生可信物理尺寸的来源。
  - CONFIRMED 只表示对象存在性经多帧确认，不等于物理尺寸可信。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from .aliases import AliasTable
from .types import (
    RADIUS_SEMANTICS_INNER,
    RADIUS_SEMANTICS_OUTER,
    RADIUS_SEMANTICS_UNKNOWN,
    SIZE_SOURCE_BBOX_HEURISTIC,
    SIZE_SOURCE_EXTERNAL_CLAIM,
    SIZE_SOURCE_LOCAL_REGISTRY,
    SIZE_SOURCE_UNKNOWN,
)

ALLOWED_SIZE_SOURCES = {
    "manual_measurement",
    "official_spec",
    "competition_rule",
    "reviewed_instance_config",
}

DEFAULT_MAX_RADIUS_CM = 200.0


@dataclass
class SizeEvidence:
    radius_cm: float
    size_source: str
    size_trusted: bool
    canonical_name: Optional[str] = None
    radius_semantics: str = RADIUS_SEMANTICS_UNKNOWN
    conflict_note: Optional[str] = None


@dataclass
class ObjectSizeEntry:
    canonical_name: str
    radius_cm: float
    radius_semantics: str
    source: str
    reviewed: bool
    instance_id: Optional[str] = None


class ObjectSizeRegistry:
    """本地可信尺寸注册表。只保存 canonical name，不保存全部别名。"""

    def __init__(self) -> None:
        self._class_entries: Dict[str, ObjectSizeEntry] = {}
        self._instance_entries: Dict[str, ObjectSizeEntry] = {}

    @classmethod
    def from_json(cls, path: str | Path) -> "ObjectSizeRegistry":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"对象尺寸配置必须是 JSON 对象，收到 {type(data).__name__}")
        registry = cls()
        registry._load(data)
        return registry

    def _load(self, data: Mapping[str, Any]) -> None:
        if data.get("version") != 1:
            raise ValueError("object_sizes 配置 version 必须为 1")
        unknown_top = set(data) - {"version", "unit", "objects", "instances"}
        if unknown_top:
            raise ValueError(f"object_sizes 配置含未知字段 {sorted(unknown_top)}")
        unit = data.get("unit")
        if unit != "cm":
            raise ValueError("object_sizes 配置 unit 必须为 cm")

        objects = data.get("objects", {})
        if not isinstance(objects, dict):
            raise ValueError("objects 必须是对象")
        for canonical_name, raw in objects.items():
            entry = self._parse_entry(canonical_name, raw)
            self._class_entries[entry.canonical_name] = entry

        instances = data.get("instances", {})
        if not isinstance(instances, dict):
            raise ValueError("instances 必须是对象")
        for instance_id, raw in instances.items():
            entry = self._parse_entry(str(instance_id), raw, instance_id=str(instance_id))
            if entry.canonical_name not in self._class_entries:
                # 实例级配置应引用类别级配置；没有类别级时也允许实例自身成为唯一来源
                pass
            self._instance_entries[str(instance_id)] = entry

    @staticmethod
    def _parse_entry(name: str, raw: Any, instance_id: Optional[str] = None) -> ObjectSizeEntry:
        if not isinstance(raw, dict):
            raise ValueError(f"对象尺寸条目 {name!r} 必须是对象")
        unknown = set(raw) - {
            "canonical_name", "radius_cm", "radius_semantics", "source",
            "reviewed", "measured_by", "instance_id",
        }
        if unknown:
            raise ValueError(f"对象尺寸条目 {name!r} 含未知字段 {sorted(unknown)}")
        canonical_name = raw.get("canonical_name", name if instance_id is None else name)
        if instance_id is None:
            canonical_name = name
        else:
            canonical_name = raw.get("canonical_name", instance_id)
        if not isinstance(canonical_name, str) or not canonical_name.strip():
            raise ValueError(f"对象尺寸条目 {name!r} 缺少 canonical_name")

        radius_cm = raw.get("radius_cm")
        try:
            radius_cm = float(radius_cm)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"对象尺寸条目 {name!r} radius_cm 必须是有限正数") from exc
        if not math.isfinite(radius_cm) or radius_cm <= 0:
            raise ValueError(f"对象尺寸条目 {name!r} radius_cm 必须是有限正数")
        if radius_cm > DEFAULT_MAX_RADIUS_CM:
            raise ValueError(
                f"对象尺寸条目 {name!r} radius_cm={radius_cm} 超过合理上限 {DEFAULT_MAX_RADIUS_CM}"
            )

        radius_semantics = raw.get("radius_semantics")
        if radius_semantics not in (RADIUS_SEMANTICS_OUTER, RADIUS_SEMANTICS_INNER):
            raise ValueError(
                f"对象尺寸条目 {name!r} radius_semantics 必须是 "
                f"{RADIUS_SEMANTICS_OUTER!r} 或 {RADIUS_SEMANTICS_INNER!r}"
            )

        source = raw.get("source")
        if source not in ALLOWED_SIZE_SOURCES:
            raise ValueError(
                f"对象尺寸条目 {name!r} source={source!r} 不在允许列表 "
                f"{sorted(ALLOWED_SIZE_SOURCES)}"
            )

        reviewed = raw.get("reviewed")
        if type(reviewed) is not bool:
            raise ValueError(
                f"对象尺寸条目 {name!r} reviewed 必须是 JSON boolean，"
                f"收到 {reviewed!r}（字符串 'true'/'false' 非法）"
            )
        if reviewed is not True:
            raise ValueError(f"对象尺寸条目 {name!r} reviewed 必须为 true")

        return ObjectSizeEntry(
            canonical_name=str(canonical_name),
            radius_cm=radius_cm,
            radius_semantics=str(radius_semantics),
            source=str(source),
            reviewed=reviewed,
            instance_id=instance_id,
        )

    def lookup_class(self, canonical_name: str) -> Optional[SizeEvidence]:
        entry = self._class_entries.get(str(canonical_name))
        if entry is None:
            return None
        return SizeEvidence(
            radius_cm=entry.radius_cm,
            size_source=SIZE_SOURCE_LOCAL_REGISTRY,
            size_trusted=True,
            canonical_name=entry.canonical_name,
            radius_semantics=entry.radius_semantics,
        )

    def lookup_instance(self, instance_id: str) -> Optional[SizeEvidence]:
        entry = self._instance_entries.get(str(instance_id))
        if entry is None:
            return None
        return SizeEvidence(
            radius_cm=entry.radius_cm,
            size_source=SIZE_SOURCE_LOCAL_REGISTRY,
            size_trusted=True,
            canonical_name=entry.canonical_name,
            radius_semantics=entry.radius_semantics,
        )

    def has_class(self, canonical_name: str) -> bool:
        return str(canonical_name) in self._class_entries


@dataclass
class ProjectionContext:
    """bbox heuristic 所需的投影几何上下文。"""

    ground_range_m: float
    optical_axis_depth_m: float
    fx: float
    fy: float


class SizePolicy:
    """统一尺寸解析。

    解析顺序：
      1. AliasTable 将原始类别转为 canonical name
      2. 如果存在可信 instance_id，查询实例尺寸
      3. 否则查询本地 canonical class 尺寸
      4. 外部记录中的 radius_cm 只作为 external_claim，不可信
      5. 没有本地配置时使用 bbox heuristic（永远不可信）
    """

    def __init__(
        self,
        registry: ObjectSizeRegistry | None = None,
        aliases: AliasTable | None = None,
        max_radius_cm: float = DEFAULT_MAX_RADIUS_CM,
    ):
        self.registry = registry or ObjectSizeRegistry()
        self.aliases = aliases or AliasTable()
        self.max_radius_cm = max_radius_cm

    def canonicalize(self, class_name: str) -> str:
        return self.aliases.canonical(str(class_name))

    def resolve(
        self,
        class_name: str,
        record: Mapping[str, Any],
        bbox: Tuple[float, float, float, float],
        projection_context: ProjectionContext,
        instance_id: Optional[str] = None,
    ) -> SizeEvidence:
        """返回最终 SizeEvidence。"""
        canonical_name = self.canonicalize(class_name)

        # 实例级本地配置优先
        if instance_id is not None:
            inst = self.registry.lookup_instance(instance_id)
            if inst is not None:
                return inst

        # 类别级本地配置（唯一类别级可信来源）
        local = self.registry.lookup_class(canonical_name)
        if local is not None:
            return local

        # 外部 radius_cm 只作为不可信声明
        external = self.from_detection_record(record, canonical_name=canonical_name)
        if external is not None:
            return external

        # 最后使用 bbox heuristic，永远不可信
        return self.from_bbox_heuristic(bbox, projection_context, canonical_name=canonical_name)

    def from_detection_record(
        self,
        record: Mapping[str, Any],
        canonical_name: Optional[str] = None,
    ) -> Optional[SizeEvidence]:
        """解析外部检测记录中的 radius_cm。

        外部输入不得包含 size_trusted，出现即抛 ValueError。
        """
        if "size_trusted" in record:
            raise ValueError("external size_trusted is forbidden")

        if "size_source" in record:
            # 外部 size_source 只能作为冲突说明，不授予信任
            pass

        if "radius_cm" not in record:
            return None
        try:
            radius_cm = float(record["radius_cm"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"radius_cm 必须是有限正数，收到 {record['radius_cm']!r}") from exc
        if not math.isfinite(radius_cm) or radius_cm <= 0:
            raise ValueError(f"radius_cm 必须是有限正数，收到 {radius_cm!r}")
        if radius_cm > self.max_radius_cm:
            raise ValueError(f"radius_cm={radius_cm} 超过合理上限 {self.max_radius_cm}")
        return SizeEvidence(
            radius_cm=radius_cm,
            size_source=SIZE_SOURCE_EXTERNAL_CLAIM,
            size_trusted=False,
            canonical_name=canonical_name,
            radius_semantics=RADIUS_SEMANTICS_UNKNOWN,
            conflict_note=(
                f"外部声称 {radius_cm}cm；本地配置优先，不可信"
                if self.registry.has_class(canonical_name or "") else None
            ),
        )

    def from_bbox_heuristic(
        self,
        bbox: Tuple[float, float, float, float],
        ctx: ProjectionContext,
        canonical_name: Optional[str] = None,
    ) -> SizeEvidence:
        """bbox 启发式诊断估计。永远不可信。

        水平方向使用 fx，垂直方向使用 fy；深度使用光轴深度，不使用斜距。
        """
        x1, y1, x2, y2 = bbox
        width_px = max(float(x2) - float(x1), 1e-6)
        height_px = max(float(y2) - float(y1), 1e-6)
        depth = max(float(ctx.optical_axis_depth_m), 1e-6)
        diameter_x_m = width_px * depth / max(float(ctx.fx), 1e-6)
        diameter_y_m = height_px * depth / max(float(ctx.fy), 1e-6)
        diameter_m = min(diameter_x_m, diameter_y_m)
        radius_cm = max(diameter_m / 2.0 * 100.0, 0.1)
        radius_cm = min(radius_cm, self.max_radius_cm)
        return SizeEvidence(
            radius_cm=radius_cm,
            size_source=SIZE_SOURCE_BBOX_HEURISTIC,
            size_trusted=False,
            canonical_name=canonical_name,
            radius_semantics=RADIUS_SEMANTICS_UNKNOWN,
        )

    def unknown(self, canonical_name: Optional[str] = None) -> SizeEvidence:
        return SizeEvidence(
            radius_cm=-1.0,
            size_source=SIZE_SOURCE_UNKNOWN,
            size_trusted=False,
            canonical_name=canonical_name,
            radius_semantics=RADIUS_SEMANTICS_UNKNOWN,
        )
