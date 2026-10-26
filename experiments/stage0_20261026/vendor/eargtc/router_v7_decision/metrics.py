from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def _arrays(
    scores: torch.Tensor | np.ndarray, targets: torch.Tensor | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    score = np.asarray(torch.as_tensor(scores, dtype=torch.float64).view(-1).cpu())
    target = np.asarray(torch.as_tensor(targets, dtype=torch.long).view(-1).cpu())
    if score.size == 0:
        raise ValueError("decisive metric input is empty")
    if score.size != target.size:
        raise ValueError("scores and decisive targets do not align")
    if not np.isfinite(score).all():
        raise ValueError("scores contain non-finite values")
    if not set(np.unique(target).tolist()) <= {0, 1}:
        raise ValueError("decisive targets must be binary")
    return score, target


def decisive_metrics(
    scores: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    score, target = _arrays(scores, targets)
    route_gnn = score >= float(threshold)
    gnn = target == 1
    llm = target == 0
    correct = route_gnn == gnn
    gnn_recall = float(route_gnn[gnn].mean()) if gnn.any() else float("nan")
    llm_recall = float((~route_gnn[llm]).mean()) if llm.any() else float("nan")
    if np.unique(target).size == 2:
        auroc = float(roc_auc_score(target, score))
        average_precision = float(average_precision_score(target, score))
    else:
        auroc = float("nan")
        average_precision = float("nan")
    clipped = np.clip(score, 1e-7, 1 - 1e-7)
    nll = float(
        -np.mean(target * np.log(clipped) + (1 - target) * np.log1p(-clipped))
    )
    return {
        "threshold": float(threshold),
        "gnn_only_recall": gnn_recall,
        "llm_only_recall": llm_recall,
        "decisive_accuracy": float(correct.mean()),
        "auroc": auroc,
        "average_precision": average_precision,
        "nll": nll,
        "counts": {
            "gnn_only": int(gnn.sum()),
            "llm_only": int(llm.sum()),
            "route_gnn": int(route_gnn.sum()),
            "route_llm": int((~route_gnn).sum()),
            "correct": int(correct.sum()),
            "total": int(target.size),
        },
    }


def stratified_bootstrap_intervals(
    scores: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    threshold: float,
    *,
    repetitions: int = 2000,
    seed: int = 42,
) -> dict[str, list[float]]:
    score, target = _arrays(scores, targets)
    groups = [np.flatnonzero(target == value) for value in (0, 1)]
    rng = np.random.default_rng(int(seed))
    keys = ("gnn_only_recall", "llm_only_recall", "decisive_accuracy")
    draws = {key: [] for key in keys}
    for _ in range(int(repetitions)):
        sampled = np.concatenate(
            [
                rng.choice(group, size=group.size, replace=True)
                for group in groups
                if group.size
            ]
        )
        result = decisive_metrics(score[sampled], target[sampled], threshold)
        for key in keys:
            draws[key].append(result[key])
    return {
        key: [
            float(np.nanquantile(values, 0.025)),
            float(np.nanquantile(values, 0.975)),
        ]
        for key, values in draws.items()
    }


__all__ = ["decisive_metrics", "stratified_bootstrap_intervals"]
