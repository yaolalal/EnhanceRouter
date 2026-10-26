from __future__ import annotations

import math

import torch


def binary_auroc(score: torch.Tensor, target: torch.Tensor) -> float | None:
    score = torch.as_tensor(score, dtype=torch.float64).view(-1)
    target = torch.as_tensor(target, dtype=torch.bool).view(-1)
    positives = int(target.sum().item())
    negatives = int((~target).sum().item())
    if positives == 0 or negatives == 0:
        return None
    order = torch.argsort(score)
    ranks = torch.empty_like(score)
    ranks[order] = torch.arange(1, score.numel() + 1, dtype=torch.float64)
    unique, inverse, counts = torch.unique(score, sorted=True, return_inverse=True, return_counts=True)
    if bool((counts > 1).any().item()):
        sums = torch.zeros(unique.numel(), dtype=torch.float64)
        sums.index_add_(0, inverse, ranks)
        ranks = (sums / counts.to(torch.float64))[inverse]
    rank_sum = float(ranks[target].sum().item())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def probability_metrics(probability: torch.Tensor, target: torch.Tensor, bins: int = 10) -> dict:
    p = torch.as_tensor(probability, dtype=torch.float64).view(-1).clamp(1e-7, 1.0 - 1e-7)
    y = torch.as_tensor(target, dtype=torch.float64).view(-1)
    if p.numel() == 0:
        return {"count": 0, "auroc": None, "brier": None, "nll": None, "ece": None}
    nll = -(y * p.log() + (1.0 - y) * (1.0 - p).log()).mean()
    ece = 0.0
    for i in range(int(bins)):
        low, high = i / bins, (i + 1) / bins
        mask = (p >= low) & ((p < high) if i + 1 < bins else (p <= high))
        if bool(mask.any().item()):
            ece += float(mask.float().mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return {
        "count": int(p.numel()),
        "auroc": binary_auroc(p, y.bool()),
        "brier": float((p - y).square().mean().item()),
        "nll": float(nll.item()),
        "ece": float(ece),
    }


def routing_metrics(
    ids: torch.Tensor,
    labels: torch.Tensor,
    gnn_pred: torch.Tensor,
    llm_pred: torch.Tensor,
    preference: torch.Tensor,
    threshold: float,
    selected_correctness: torch.Tensor | None = None,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    ids = ids.long()
    y = labels[ids]
    g = gnn_pred[ids]
    l = llm_pred[ids]
    score = preference[ids]
    choose_g = (g == l) | (score >= float(threshold))
    selected = torch.where(choose_g, g, l)
    correct = selected == y
    g_correct, l_correct = g == y, l == y
    conflict = g != l
    oracle_correct = g_correct | l_correct
    metrics = {
        "count": int(ids.numel()),
        "conflict_count": int(conflict.sum().item()),
        "decisive_conflict_count": int((conflict & (g_correct != l_correct)).sum().item()),
        "gnn_accuracy": float(g_correct.float().mean().item()),
        "llm_accuracy": float(l_correct.float().mean().item()),
        "best_single_accuracy": max(
            float(g_correct.float().mean().item()), float(l_correct.float().mean().item())
        ),
        "router_accuracy": float(correct.float().mean().item()),
        "conflict_accuracy": (
            float(correct[conflict].float().mean().item()) if bool(conflict.any().item()) else None
        ),
        "oracle_accuracy": float(oracle_correct.float().mean().item()),
        "threshold": float(threshold),
        "gnn_route_rate_on_conflict": (
            float(choose_g[conflict].float().mean().item()) if bool(conflict.any().item()) else None
        ),
    }
    metrics["gain_over_best_single"] = metrics["router_accuracy"] - metrics["best_single_accuracy"]
    metrics["oracle_gap"] = metrics["oracle_accuracy"] - metrics["router_accuracy"]
    metrics["beats_best_single"] = metrics["gain_over_best_single"] > 0.0
    decisive = conflict & (g_correct != l_correct)
    metrics["preference_auroc"] = binary_auroc(score[decisive], g_correct[decisive])
    if selected_correctness is not None:
        metrics["selected_correctness"] = probability_metrics(
            selected_correctness[ids], correct
        )
    return metrics, choose_g, correct
