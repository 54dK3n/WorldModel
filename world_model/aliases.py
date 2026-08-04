"""子问题 #5：别名表。

planner 说"球"，检测器输出 "sports ball"，语音可能说 "网球" —— 得对上。
一张 JSON 就够，不需要模型。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

# 规范名 -> 所有可接受的写法（含检测器类名、中文、口语）
DEFAULT_ALIASES: Dict[str, List[str]] = {
    "ball": ["sports ball", "tennis ball", "球", "网球", "小球"],
    "basket": ["basket", "bucket", "篮子", "桶", "框"],
    "table": ["dining table", "desk", "桌子", "台面"],
    "bottle": ["bottle", "coke_bottle", "瓶子", "可乐瓶"],
    "left_arm": ["left_arm", "左臂"],
    "right_arm": ["right_arm", "右臂"],
}


class AliasTable:
    def __init__(self, mapping: Dict[str, List[str]] | None = None):
        self.canonical_to_aliases = dict(mapping or DEFAULT_ALIASES)
        self._reverse: Dict[str, str] = {}
        for canonical, aliases in self.canonical_to_aliases.items():
            self._reverse[canonical.lower()] = canonical
            for a in aliases:
                self._reverse[a.lower()] = canonical

    @classmethod
    def from_json(cls, path: str | Path) -> "AliasTable":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def canonical(self, raw_name: str) -> str:
        """检测器类名 / 中文口语 -> 规范名。查不到就原样返回并保留。"""
        return self._reverse.get(raw_name.strip().lower(), raw_name.strip())

    def aliases_of(self, canonical_name: str) -> List[str]:
        return list(self.canonical_to_aliases.get(canonical_name, []))

    def same_class(self, a: str, b: str) -> bool:
        """关联时判断类别是否一致。"""
        return self.canonical(a) == self.canonical(b)
