from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    selected_count: int
    coverage: float
    empirical_precision: float


@dataclass(frozen=True)
class SupervisionPools:
    gold: torch.Tensor
    hard: torch.Tensor
    agreement_kd: torch.Tensor
    conflict_kd: torch.Tensor
    abstain: torch.Tensor


def select_precision_threshold(
    *,
    trust: torch.Tensor,
    correct: torch.Tensor,
    node_ids: torch.Tensor,
    target_precision: float,
) -> ThresholdResult | None:
    q = torch.as_tensor(trust, dtype=torch.float64).view(-1)
    y = torch.as_tensor(correct, dtype=torch.bool).view(-1)
    ids = torch.as_tensor(node_ids, dtype=torch.long).view(-1)
    if q.numel() == 0 or not (q.numel() == y.numel() == ids.numel()):
        raise ValueError("threshold inputs must be equal and non-empty")
    if not bool(torch.isfinite(q).all()):
        raise ValueError("trust contains non-finite values")
    target = float(target_precision)
    if not 0.0 <= target <= 1.0:
        raise ValueError("target precision must be in [0,1]")
    order = sorted(range(q.numel()), key=lambda i: (-float(q[i]), int(ids[i])))
    candidates = []
    successes = 0
    count = 0
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
            candidates.append(ThresholdResult(
                threshold=threshold,
                selected_count=count,
                coverage=count / q.numel(),
                empirical_precision=precision,
            ))
        position = end
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (-row.selected_count, -row.empirical_precision, -row.threshold),
    )


def build_supervision_pools(
    *,
    node_ids: torch.Tensor,
    agreement: torch.Tensor,
    selected_trust: torch.Tensor,
    gold_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    holdout_ids: torch.Tensor,
    tau_high: float,
    tau_medium: float,
) -> SupervisionPools:
    ids = torch.as_tensor(node_ids, dtype=torch.long).view(-1)
    same = torch.as_tensor(agreement, dtype=torch.bool).view(-1)
    trust = torch.as_tensor(selected_trust, dtype=torch.float64).view(-1)
    if not (ids.numel() == same.numel() == trust.numel()):
        raise ValueError("pool features must have equal length")
    if float(tau_high) < float(tau_medium):
        raise ValueError("high threshold must not be below medium threshold")
    candidates = torch.as_tensor(candidate_ids, dtype=torch.long).view(-1)
    holdout = torch.as_tensor(holdout_ids, dtype=torch.long).view(-1)
    eligible = torch.isin(ids, candidates) & ~torch.isin(ids, holdout)
    high = eligible & (trust >= float(tau_high))
    medium = eligible & (trust >= float(tau_medium)) & ~high
    low = eligible & ~high & ~medium
    return SupervisionPools(
        gold=torch.as_tensor(gold_ids, dtype=torch.long).view(-1).clone(),
        hard=ids[high].clone(),
        agreement_kd=ids[(high | medium) & same].clone(),
        conflict_kd=ids[(high | medium) & ~same].clone(),
        abstain=ids[low].clone(),
    )

