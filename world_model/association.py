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
    """门控随 dt 自适应：gate = base + v_max * dt（有上限）。

    历史：这里原本是固定 0.5m 门控。固定门控在长帧间隔下会断轨 ——
    demo 里球在 2s 内移动 0.75m 超过门控，被新建成 ball_003 而不是关联到
    ball_001。后果不只是"id 不好看"：断轨产生的幽灵轨迹会一路传导到判定层，
    让 Judge 在两条同名轨迹之间按置信度二选一，判定结论随检测分数翻转。
    详见 2026-08-11 判定可靠性复核用例 A / B。
    """

    gate_distance_m: float = 0.5        # 基础门控（dt=0 时）
    max_speed_mps: float = 0.5          # 目标运动速度上限，用来按 dt 放大门控
    max_gate_distance_m: float = 2.0    # 门控上限，避免长时间不更新后门控大到乱配
    class_mismatch_penalty: float = 1.0
    allow_cross_class: bool = False   # False = 类别不一致直接门控掉
    appearance_weight: float = 0.0    # 预留 ReID 权重，本期为 0


def gate_for(track: TrackedObject, now: Optional[float], cfg: AssociationConfig) -> float:
    """该轨迹这一帧的门控距离。距上次真实观测越久，允许的位移越大。"""
    dt = max(0.0, now - track.last_seen) if now is not None else 0.0
    return min(cfg.gate_distance_m + cfg.max_speed_mps * dt, cfg.max_gate_distance_m)


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
    now: Optional[float] = None,
) -> List[List[float]]:
    """N×M 代价矩阵。inf 表示被门控掉、不可配。

    门控按每条轨迹各自的 dt 计算 —— 刚看到的轨迹门控紧，很久没更新的放宽。
    """
    matrix: List[List[float]] = []
    for t in tracks:
        row: List[float] = []
        gate = gate_for(t, now, cfg)
        for d in detections:
            dist = euclidean(t.x, t.z, d.x, d.z)

            # 门控 1：距离（随 dt 放宽）
            if dist > gate:
                row.append(INF)
                continue

            same = aliases.same_class(t.name, d.class_name)
            # 门控 2：类别
            if not same and not cfg.allow_cross_class:
                row.append(INF)
                continue

            cost = dist / max(gate, 1e-6)
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
    now: Optional[float] = None,
) -> AssociationResult:
    cfg = cfg or AssociationConfig()
    if not tracks:
        return AssociationResult([], [], list(range(len(detections))))
    if not detections:
        return AssociationResult([], list(range(len(tracks))), [])
    return greedy_assign(build_cost_matrix(tracks, detections, aliases, cfg, now=now))
