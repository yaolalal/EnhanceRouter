from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import torch


@dataclass(frozen=True)
class PairBatch:
    left: torch.Tensor
    right: torch.Tensor
    direction: torch.Tensor
    weight: torch.Tensor
    delta_advantage: torch.Tensor

    def __len__(self) -> int:
        return int(self.left.numel())


def _node_key(row: dict) -> tuple[str, int]:
    return str(row["dataset"]), int(row["node_id"])


def _environment_key(row: dict) -> tuple[str, int, str]:
    return str(row["dataset"]), int(row["node_id"]), str(row["environment_id"])


def node_environment_weights(rows: list[dict]) -> torch.Tensor:
    if not rows:
        return torch.empty(0, dtype=torch.float32)
    counts: dict[tuple[str, int], int] = {}
    for row in rows:
        key = _node_key(row)
        counts[key] = counts.get(key, 0) + 1
    value = torch.tensor([1.0 / counts[_node_key(row)] for row in rows], dtype=torch.float64)
    return (value / value.mean()).to(torch.float32)


def _allowed(left: dict, right: dict) -> bool:
    left_channel, right_channel = str(left["channel"]), str(right["channel"])
    if left_channel == "clean" or right_channel == "clean":
        return left_channel != right_channel
    if {left_channel, right_channel} == {"graph", "text"}:
        return abs(float(left["severity"]) - float(right["severity"])) <= 1e-12
    return False


def build_counterfactual_pairs(
    rows: list[dict], *, epsilon: float = 0.1, kappa: float = 1.0
) -> PairBatch:
    epsilon, kappa = float(epsilon), float(kappa)
    if epsilon < 0 or kappa <= 0:
        raise ValueError("epsilon must be non-negative and kappa must be positive")
    seen: set[tuple[str, int, str]] = set()
    grouped: dict[tuple[str, int], list[int]] = {}
    for index, row in enumerate(rows):
        key = _environment_key(row)
        if key in seen:
            raise ValueError(f"duplicate node/environment row: {key}")
        seen.add(key)
        advantage = float(row["advantage"])
        if not math.isfinite(advantage):
            raise ValueError("counterfactual advantage must be finite")
        grouped.setdefault(_node_key(row), []).append(index)

    left_ids, right_ids, directions, weights, deltas = [], [], [], [], []
    for key in sorted(grouped):
        for left, right in combinations(sorted(grouped[key]), 2):
            if not _allowed(rows[left], rows[right]):
                continue
            delta = float(rows[right]["advantage"]) - float(rows[left]["advantage"])
            if abs(delta) < epsilon:
                continue
            left_ids.append(left)
            right_ids.append(right)
            directions.append(1.0 if delta > 0 else -1.0)
            weights.append(min(abs(delta) / kappa, 1.0))
            deltas.append(delta)
    balanced = torch.tensor(weights, dtype=torch.float64)
    if balanced.numel():
        node_groups: dict[tuple[str, int], list[int]] = {}
        for pair_index, left_index in enumerate(left_ids):
            node_groups.setdefault(_node_key(rows[left_index]), []).append(pair_index)
        for indices in node_groups.values():
            index = torch.tensor(indices, dtype=torch.long)
            balanced[index] /= balanced[index].sum()
        dataset_groups: dict[str, list[int]] = {}
        for pair_index, left_index in enumerate(left_ids):
            dataset_groups.setdefault(str(rows[left_index]["dataset"]), []).append(pair_index)
        for indices in dataset_groups.values():
            index = torch.tensor(indices, dtype=torch.long)
            balanced[index] /= balanced[index].sum()
        balanced /= balanced.mean()
    return PairBatch(
        left=torch.tensor(left_ids, dtype=torch.long),
        right=torch.tensor(right_ids, dtype=torch.long),
        direction=torch.tensor(directions, dtype=torch.float32),
        weight=balanced.to(torch.float32),
        delta_advantage=torch.tensor(deltas, dtype=torch.float32),
    )
