"""Deterministic top-percent selection and cross-repair supervision pools."""

from __future__ import annotations

import math
from typing import Any

import torch


PERCENTS = tuple(range(5, 101, 5))


def _aligned_inputs(scores: torch.Tensor, node_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(scores, dtype=torch.float64).detach().view(-1)
    ids = torch.as_tensor(node_ids, dtype=torch.long).detach().view(-1)
    if values.numel() != ids.numel():
        raise ValueError("scores and node IDs must have equal length")
    if ids.numel() != torch.unique(ids).numel():
        raise ValueError("node IDs must be unique")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("scores must be finite")
    return values, ids


def _rank(scores: torch.Tensor, node_ids: torch.Tensor) -> list[int]:
    return sorted(
        range(node_ids.numel()), key=lambda index: (-float(scores[index]), int(node_ids[index]))
    )


def apply_topk(scores: torch.Tensor, node_ids: torch.Tensor, percent: float | None) -> torch.Tensor:
    """Select ``ceil(percent * N / 100)`` IDs with deterministic tie-breaking."""
    values, ids = _aligned_inputs(scores, node_ids)
    if percent is None or ids.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    percentage = float(percent)
    if not 0.0 < percentage <= 100.0:
        raise ValueError("selected percent must be in (0, 100]")
    count = math.ceil(percentage * ids.numel() / 100.0)
    chosen = _rank(values, ids)[:count]
    return ids[torch.tensor(chosen, dtype=torch.long)].clone()


def select_topk_policy(
    scores: torch.Tensor,
    correct: torch.Tensor,
    node_ids: torch.Tensor,
    min_count: int = 10,
    tolerance: float = 0.01,
) -> dict[str, Any]:
    """Calibrate a top-percent rule using only threshold-half correctness."""
    values, ids = _aligned_inputs(scores, node_ids)
    labels = torch.as_tensor(correct, dtype=torch.bool).detach().view(-1)
    if labels.numel() != ids.numel():
        raise ValueError("correctness and node IDs must have equal length")
    if int(min_count) <= 0:
        raise ValueError("min_count must be positive")
    if float(tolerance) < 0.0:
        raise ValueError("tolerance must be non-negative")
    if ids.numel() < int(min_count):
        return {
            "selected_percent": None,
            "candidates": [],
            "chosen_accuracy": None,
            "chosen_count": 0,
        }

    ordered = _rank(values, ids)
    candidates: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for percent in PERCENTS:
        count = math.ceil(percent * ids.numel() / 100.0)
        accepted = count >= int(min_count)
        accuracy = (
            sum(int(labels[index]) for index in ordered[:count]) / count
            if accepted else None
        )
        row = {
            "percent": percent,
            "count": count,
            "accuracy": accuracy,
            "accepted": accepted,
        }
        candidates.append(row)
        if accepted:
            eligible.append(row)
    best = max(float(row["accuracy"]) for row in eligible)
    chosen = max(
        (row for row in eligible if best - float(row["accuracy"]) <= float(tolerance) + 1e-12),
        key=lambda row: int(row["percent"]),
    )
    return {
        "selected_percent": int(chosen["percent"]),
        "candidates": candidates,
        "chosen_accuracy": float(chosen["accuracy"]),
        "chosen_count": int(chosen["count"]),
    }


def _validate_test_subset(test_ids: torch.Tensor, candidate_ids: torch.Tensor) -> None:
    if candidate_ids.numel() != torch.unique(candidate_ids).numel() or not bool(
        torch.isin(candidate_ids, test_ids).all()
    ):
        raise ValueError("candidate IDs must be a unique subset of test IDs")


def build_pools(
    test_ids: torch.Tensor,
    agreement: torch.Tensor,
    choose_g: torch.Tensor,
    scores: torch.Tensor,
    policies: dict[str, dict[str, Any]],
    candidate_ids: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Create full-node-ID cross-teaching and frozen-expert retention pools."""
    test = torch.as_tensor(test_ids, dtype=torch.long).detach().view(-1)
    if test.numel() != torch.unique(test).numel():
        raise ValueError("test IDs must be unique")
    candidate = test.clone() if candidate_ids is None else torch.as_tensor(
        candidate_ids, dtype=torch.long
    ).detach().view(-1)
    _validate_test_subset(test, candidate)
    same = torch.as_tensor(agreement, dtype=torch.bool).detach().view(-1)
    route_g = torch.as_tensor(choose_g, dtype=torch.bool).detach().view(-1)
    trust = torch.as_tensor(scores, dtype=torch.float64).detach().view(-1)
    if not (same.numel() == route_g.numel() == trust.numel()):
        raise ValueError("full agreement, route, and score arrays must have equal length")
    if test.numel() and (int(test.min()) < 0 or int(test.max()) >= trust.numel()):
        raise ValueError("test IDs are outside full feature arrays")
    if not bool(torch.isfinite(trust).all()):
        raise ValueError("scores must be finite")
    required = {"conflict_gnn", "conflict_llm", "agreement"}
    if set(policies) != required:
        raise ValueError("policies must cover conflict_gnn, conflict_llm, and agreement")

    eligible = candidate
    agreement_ids = eligible[same[eligible]]
    conflict_gnn_ids = eligible[~same[eligible] & route_g[eligible]]
    conflict_llm_ids = eligible[~same[eligible] & ~route_g[eligible]]

    def selected(ids: torch.Tensor, policy_name: str) -> torch.Tensor:
        policy = policies[policy_name]
        if "selected_percent" not in policy:
            raise ValueError(f"{policy_name} policy lacks selected_percent")
        return apply_topk(trust[ids], ids, policy["selected_percent"])

    selected_gnn = selected(conflict_gnn_ids, "conflict_gnn")
    selected_llm = selected(conflict_llm_ids, "conflict_llm")
    selected_agreement = selected(agreement_ids, "agreement")
    keep_g = torch.unique(torch.cat((selected_gnn, selected_agreement)), sorted=True)
    keep_l = torch.unique(torch.cat((selected_llm, selected_agreement)), sorted=True)
    selected_any = torch.unique(
        torch.cat((selected_gnn, selected_llm, selected_agreement)), sorted=True
    )
    return {
        "teach_g": selected_llm,
        "teach_l": selected_gnn,
        "keep_g": keep_g,
        "keep_l": keep_l,
        "agreement": selected_agreement,
        "unselected": test[~torch.isin(test, selected_any)].clone(),
    }
