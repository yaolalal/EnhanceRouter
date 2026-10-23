"""Deterministic validation-selected exact top-k policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from math import ceil
from typing import Sequence

import numpy as np


DEFAULT_GRID = tuple(i / 100.0 for i in range(5, 101, 5))


@dataclass(frozen=True)
class TopKCurvePoint:
    k_fraction: float
    quota: int
    correct: int
    accuracy: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TopKSelection:
    k_fraction: float
    quota: int
    correct: int
    accuracy: float
    best_accuracy: float
    curve: tuple[TopKCurvePoint, ...]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["curve"] = [point.to_dict() for point in self.curve]
        return value


def _validated(node_ids: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if ids.size != values.size:
        raise ValueError("node_ids and scores must have equal length")
    if ids.size == 0:
        raise ValueError("population must be non-empty")
    if np.unique(ids).size != ids.size:
        raise ValueError("node_ids must be unique")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    return ids, values


def rank_exact(node_ids: np.ndarray, scores: np.ndarray, k_fraction: float) -> np.ndarray:
    """Return a mask selecting exactly ceil(k*N), score-descending then ID-ascending."""

    ids, values = _validated(node_ids, scores)
    k = float(k_fraction)
    if not 0.0 < k <= 1.0:
        raise ValueError("k_fraction must be in (0, 1]")
    # Interpret user/config decimals exactly. Binary floats such as 0.55 can
    # otherwise turn an integral quota (110/200) into 111 after ``ceil``.
    quota = min(ids.size, int(ceil(Fraction(str(k)) * ids.size)))
    order = np.lexsort((ids, -values))
    mask = np.zeros(ids.size, dtype=bool)
    mask[order[:quota]] = True
    return mask


def apply_frozen_k(node_ids: np.ndarray, scores: np.ndarray, k_fraction: float) -> np.ndarray:
    return rank_exact(node_ids, scores, k_fraction)


def select_validation_k(
    node_ids: np.ndarray,
    scores: np.ndarray,
    correct: np.ndarray,
    grid: Sequence[float] = DEFAULT_GRID,
    min_count: int = 10,
    tolerance: float = 0.01,
) -> TopKSelection:
    """Select largest valid k whose accuracy is within tolerance of the best."""

    ids, values = _validated(node_ids, scores)
    labels = np.asarray(correct, dtype=bool).reshape(-1)
    if labels.size != ids.size:
        raise ValueError("correct must align with node_ids")
    if ids.size < min_count:
        raise ValueError(f"population must contain at least {min_count} nodes")
    points: list[TopKCurvePoint] = []
    for fraction in sorted({float(item) for item in grid}):
        mask = rank_exact(ids, values, fraction)
        quota = int(mask.sum())
        if quota < min_count:
            continue
        n_correct = int(labels[mask].sum())
        points.append(TopKCurvePoint(fraction, quota, n_correct, n_correct / quota))
    if not points:
        raise ValueError("grid has no candidate satisfying min_count")
    best = max(point.accuracy for point in points)
    eligible = [point for point in points if point.accuracy >= best - float(tolerance) - 1e-15]
    chosen = max(eligible, key=lambda point: point.k_fraction)
    return TopKSelection(
        k_fraction=chosen.k_fraction,
        quota=chosen.quota,
        correct=chosen.correct,
        accuracy=chosen.accuracy,
        best_accuracy=best,
        curve=tuple(points),
    )
