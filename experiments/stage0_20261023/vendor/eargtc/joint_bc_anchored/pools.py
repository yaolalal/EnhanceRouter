from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    selected_count: int
    coverage: float
    empirical_precision: float
    wilson_lcb: float


@dataclass(frozen=True)
class SupervisionPools:
    gold: torch.Tensor
    hard_agreement: torch.Tensor
    hard_conflict_gnn: torch.Tensor
    hard_conflict_llm: torch.Tensor
    soft_agreement: torch.Tensor
    soft_conflict_gnn: torch.Tensor
    soft_conflict_llm: torch.Tensor
    abstain: torch.Tensor


def _wilson_lcb(successes: int, count: int, z: float = 1.96) -> float:
    if count <= 0:
        return 0.0
    p = successes / count
    denominator = 1.0 + z * z / count
    center = p + z * z / (2.0 * count)
    radius = z * math.sqrt(p * (1.0 - p) / count + z * z / (4.0 * count * count))
    return (center - radius) / denominator


def select_largest_coverage_threshold(
    trust: torch.Tensor,
    correct: torch.Tensor,
    node_ids: torch.Tensor,
    target_precision: float,
) -> ThresholdResult | None:
    q = torch.as_tensor(trust, dtype=torch.float64).view(-1)
    y = torch.as_tensor(correct, dtype=torch.bool).view(-1)
    ids = torch.as_tensor(node_ids, dtype=torch.long).view(-1)
    if not (q.numel() == y.numel() == ids.numel()) or q.numel() == 0:
        raise ValueError("threshold inputs must be equal and non-empty")
    target = float(target_precision)
    if not 0.0 <= target <= 1.0:
        raise ValueError("target_precision must be in [0,1]")
    if not bool(torch.isfinite(q).all()):
        raise ValueError("trust contains non-finite values")
    # Sort once and evaluate only complete trust tie-groups.  This is exactly
    # equivalent to repeated q >= threshold masks, without O(N^2) behavior on
    # large official validation splits.
    order = sorted(range(q.numel()), key=lambda index: (-float(q[index]), int(ids[index])))
    trials = []
    count = 0
    successes = 0
    position = 0
    while position < len(order):
        threshold = float(q[order[position]])
        end = position
        while end < len(order) and float(q[order[end]]) == threshold:
            successes += int(y[order[end]])
            count += 1
            end += 1
        precision = successes / count
        if precision >= target:
            trials.append(
                ThresholdResult(
                    threshold=threshold,
                    selected_count=count,
                    coverage=count / q.numel(),
                    empirical_precision=precision,
                    wilson_lcb=_wilson_lcb(successes, count),
                )
            )
        position = end
    if not trials:
        return None
    return min(
        trials,
        key=lambda row: (-row.selected_count, -row.empirical_precision, -row.threshold),
    )


def build_tiered_pools(
    *,
    node_ids: torch.Tensor,
    agreement: torch.Tensor,
    choose_gnn: torch.Tensor,
    selected_trust: torch.Tensor,
    gold_ids: torch.Tensor,
    query_ids: torch.Tensor,
    tau_h: float | None,
    tau_l: float | None,
) -> SupervisionPools:
    ids = torch.as_tensor(node_ids, dtype=torch.long).view(-1)
    same = torch.as_tensor(agreement, dtype=torch.bool).view(-1)
    route_g = torch.as_tensor(choose_gnn, dtype=torch.bool).view(-1)
    q = torch.as_tensor(selected_trust, dtype=torch.float64).view(-1)
    if len({ids.numel(), same.numel(), route_g.numel(), q.numel()}) != 1:
        raise ValueError("pool feature inputs must have equal row counts")
    query = torch.isin(ids, torch.as_tensor(query_ids, dtype=torch.long).view(-1))
    high = torch.zeros_like(query)
    if tau_h is not None:
        high = query & (q >= float(tau_h))
    medium = torch.zeros_like(query)
    if tau_l is not None:
        medium = query & (q >= float(tau_l)) & ~high
        if tau_h is not None:
            medium &= q < float(tau_h)
    low = query & ~high & ~medium

    def take(mask: torch.Tensor) -> torch.Tensor:
        return ids[mask].clone()

    return SupervisionPools(
        gold=torch.as_tensor(gold_ids, dtype=torch.long).view(-1).clone(),
        hard_agreement=take(high & same),
        hard_conflict_gnn=take(high & ~same & route_g),
        hard_conflict_llm=take(high & ~same & ~route_g),
        soft_agreement=take(medium & same),
        soft_conflict_gnn=take(medium & ~same & route_g),
        soft_conflict_llm=take(medium & ~same & ~route_g),
        abstain=take(low),
    )


def select_top_fraction(
    node_ids: torch.Tensor,
    selected_trust: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    ids = torch.as_tensor(node_ids, dtype=torch.long).view(-1)
    q = torch.as_tensor(selected_trust, dtype=torch.float64).view(-1)
    if ids.numel() != q.numel() or ids.numel() == 0:
        raise ValueError("Top-K inputs must be equal and non-empty")
    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("fraction must be in [0,1]")
    count = math.ceil(float(fraction) * ids.numel())
    order = sorted(range(ids.numel()), key=lambda i: (-float(q[i]), int(ids[i])))
    return ids[torch.tensor(order[:count], dtype=torch.long)]


def select_random_matched(
    node_ids: torch.Tensor,
    selected_count: int,
    *,
    seed: int,
) -> torch.Tensor:
    ids = torch.as_tensor(node_ids, dtype=torch.long).view(-1)
    count = int(selected_count)
    if count < 0 or count > ids.numel():
        raise ValueError("selected_count is outside the candidate pool")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return ids[torch.randperm(ids.numel(), generator=generator)[:count]]
