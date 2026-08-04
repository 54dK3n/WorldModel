"""子问题 #3：关联（Data Association）。

本质是**指派问题**，不是置信区间问题。
当前 N 个已知对象，这一帧 M 个检测，要决定谁配谁：

    构造 N×M 代价矩阵：代价 = 归一化距离 + 类别不一致罚项 (+ 外观相似度，暂不上)
    门控：距离超阈值直接判为不可配（inf）
    求解：贪心（物体少、移动慢，SORT 级别足够；需要时可换匈牙利算法）

    配不上的新检测 -> 新建对象
    配不上的老对象 -> 交给 decay.py 走衰减

参考谱系：SORT < ByteTrack < DeepSORT（复杂度递增）。
ReID appearance embedding 是 DeepSORT 的做法，本期不上，留了接口位。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from .aliases import AliasTable
from .types import Detection, TrackedObject

INF = float("inf")


@dataclass
class AssociationConfig:
    # 门控：超过此距离一律不配
    # 已知局限（run_demo.py 复现）：固定门控在长帧间隔下会误判。
    # demo 中球在 2s 内移动 0.75m > 0.5m 门控，被新建成 ball_003 而非关联到 ball_001。
    # 正确做法是门控随 dt 缩放：gate = base + v_max * dt，或引入恒速运动预测。
    # TODO(P1): 帧率稳定后改为 dt 自适应门控 + 简单运动模型
    gate_distance_m: float = 0.5
    class_mismatch_penalty: float = 1.0
    allow_cross_class: bool = False   # False = 类别不一致直接门控掉
    appearance_weight: float = 0.0    # 预留 ReID 权重，本期为 0


@dataclass
class AssociationResult:
    matches: List[Tuple[int, int]]        # (track_idx, detection_idx)
    unmatched_tracks: List[int]
    unmatched_detections: List[int]


def euclidean(ax: float, az: float, bx: float, bz: float) -> float:
    return math.hypot(ax - bx, az - bz)


def build_cost_matrix(
    tracks: Sequence[TrackedObject],
    detections: Sequence[Detection],
    aliases: AliasTable,
    cfg: AssociationConfig,
    appearance_fn: Optional[Callable[[TrackedObject, Detection], float]] = None,
) -> List[List[float]]:
    """N×M 代价矩阵。inf 表示被门控掉、不可配。"""
    matrix: List[List[float]] = []
    for t in tracks:
        row: List[float] = []
        for d in detections:
            dist = euclidean(t.x, t.z, d.x, d.z)

            # 门控 1：距离
            if dist > cfg.gate_distance_m:
                row.append(INF)
                continue

            same = aliases.same_class(t.name, d.class_name)
            # 门控 2：类别
            if not same and not cfg.allow_cross_class:
                row.append(INF)
                continue

            cost = dist / max(cfg.gate_distance_m, 1e-6)
            if not same:
                cost += cfg.class_mismatch_penalty
            if appearance_fn is not None and cfg.appearance_weight > 0:
                cost += cfg.appearance_weight * appearance_fn(t, d)
            row.append(cost)
        matrix.append(row)
    return matrix


def greedy_assign(matrix: List[List[float]]) -> AssociationResult:
    """把所有可行 (i, j) 按代价升序排，逐个占坑。

    小规模下与匈牙利算法结果几乎一致；要换最优解把这个函数替成
    scipy.optimize.linear_sum_assignment 即可，接口不变。
    """
    n = len(matrix)
    m = len(matrix[0]) if n else 0

    pairs = [
        (matrix[i][j], i, j)
        for i in range(n)
        for j in range(m)
        if matrix[i][j] != INF
    ]
    pairs.sort(key=lambda p: p[0])

    used_t, used_d = set(), set()
    matches: List[Tuple[int, int]] = []
    for _cost, i, j in pairs:
        if i in used_t or j in used_d:
            continue
        used_t.add(i)
        used_d.add(j)
        matches.append((i, j))

    return AssociationResult(
        matches=matches,
        unmatched_tracks=[i for i in range(n) if i not in used_t],
        unmatched_detections=[j for j in range(m) if j not in used_d],
    )


def associate(
    tracks: Sequence[TrackedObject],
    detections: Sequence[Detection],
    aliases: AliasTable,
    cfg: AssociationConfig | None = None,
) -> AssociationResult:
    cfg = cfg or AssociationConfig()
    if not tracks:
        return AssociationResult([], [], list(range(len(detections))))
    if not detections:
        return AssociationResult([], list(range(len(tracks))), [])
    return greedy_assign(build_cost_matrix(tracks, detections, aliases, cfg))
