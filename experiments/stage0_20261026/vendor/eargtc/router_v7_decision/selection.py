from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import torch

from .metrics import decisive_metrics
from .models import CandidateSpec, DomainData, fit_candidate


@dataclass(frozen=True)
class ThresholdRecord:
    objective: str
    threshold: float
    decisive_accuracy: float
    gnn_only_recall: float
    llm_only_recall: float
    harmonic_recall: float


@dataclass(frozen=True)
class SelectionRecord:
    selected: CandidateSpec
    ranking: list[dict]


def _threshold_candidates(scores: np.ndarray) -> list[float]:
    unique = np.unique(scores)
    boundaries = [
        float(np.nextafter(unique[0], -np.inf)),
        float(np.nextafter(unique[-1], np.inf)),
    ]
    midpoints = ((unique[:-1] + unique[1:]) / 2.0).tolist()
    return sorted(set(boundaries + [float(v) for v in midpoints]))


def _record(objective: str, threshold: float, result: dict) -> ThresholdRecord:
    rg = float(result["gnn_only_recall"])
    rl = float(result["llm_only_recall"])
    harmonic = 0.0 if not (isfinite(rg) and isfinite(rl)) or rg + rl == 0 else 2 * rg * rl / (rg + rl)
    return ThresholdRecord(
        objective=objective,
        threshold=float(threshold),
        decisive_accuracy=float(result["decisive_accuracy"]),
        gnn_only_recall=rg,
        llm_only_recall=rl,
        harmonic_recall=float(harmonic),
    )
def _threshold_records(
    scores: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    objective: str,
) -> list[ThresholdRecord]:
    score = np.asarray(torch.as_tensor(scores, dtype=torch.float64).view(-1).cpu())
    target = np.asarray(torch.as_tensor(targets, dtype=torch.long).view(-1).cpu())
    if score.size == 0:
        raise ValueError("threshold selection input is empty")
    if score.size != target.size:
        raise ValueError("scores and targets do not align")
    if not np.isfinite(score).all() or not set(np.unique(target).tolist()) <= {0, 1}:
        raise ValueError("threshold selection requires finite scores and binary targets")
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    sorted_target = target[order]
    prefix_gnn = np.concatenate([[0], np.cumsum(sorted_target == 1)])
    prefix_llm = np.concatenate([[0], np.cumsum(sorted_target == 0)])
    total_gnn = int(prefix_gnn[-1])
    total_llm = int(prefix_llm[-1])
    records = []
    for threshold in _threshold_candidates(score):
        position = int(np.searchsorted(sorted_score, threshold, side="left"))
        true_gnn = total_gnn - int(prefix_gnn[position])
        true_llm = int(prefix_llm[position])
        gnn_recall = true_gnn / total_gnn if total_gnn else float("nan")
        llm_recall = true_llm / total_llm if total_llm else float("nan")
        records.append(
            _record(
                objective,
                threshold,
                {
                    "gnn_only_recall": gnn_recall,
                    "llm_only_recall": llm_recall,
                    "decisive_accuracy": (true_gnn + true_llm) / score.size,
                },
            )
        )
    return records




def choose_accuracy_threshold(
    scores: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
) -> ThresholdRecord:
    records = _threshold_records(scores, targets, "accuracy")
    return max(
        records,
        key=lambda value: (
            value.decisive_accuracy,
            min(value.gnn_only_recall, value.llm_only_recall),
            -round(abs(value.threshold - 0.5), 6),
            value.threshold,
        ),
    )


def choose_balanced_threshold(
    scores: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
) -> ThresholdRecord:
    records = _threshold_records(scores, targets, "harmonic_recall")
    return max(
        records,
        key=lambda value: (
            value.harmonic_recall,
            value.decisive_accuracy,
            -abs(value.threshold - 0.5),
            value.threshold,
        ),
    )


def select_source_lodo(
    candidates: list[CandidateSpec],
    source_domains: list[DomainData],
    *,
    seed: int = 42,
) -> SelectionRecord:
    if len(source_domains) < 2 or not candidates:
        raise ValueError("LODO selection needs candidates and at least two domains")
    ranking = []
    complexity = {
        "logistic": 1,
        "residual_logistic": 2,
        "rbf": 3,
        "extra_trees": 4,
        "mlp": 5,
    }
    for candidate in candidates:
        folds = []
        for index, held_out in enumerate(source_domains):
            train = source_domains[:index] + source_domains[index + 1 :]
            head = fit_candidate(candidate, train, seed=seed + index)
            scores = head.predict_proba(held_out.x)
            result = decisive_metrics(scores, held_out.y, 0.5)
            recalls = [
                value
                for value in (
                    result["gnn_only_recall"],
                    result["llm_only_recall"],
                )
                if np.isfinite(value)
            ]
            folds.append(
                {
                    "domain": held_out.name,
                    "decisive_accuracy": result["decisive_accuracy"],
                    "minimum_recall": min(recalls) if recalls else 0.0,
                    "nll": result["nll"],
                }
            )
        ranking.append(
            {
                "name": candidate.name,
                "spec": candidate,
                "macro_accuracy": float(np.mean([f["decisive_accuracy"] for f in folds])),
                "macro_minimum_recall": float(np.mean([f["minimum_recall"] for f in folds])),
                "macro_nll": float(np.mean([f["nll"] for f in folds])),
                "complexity": complexity[candidate.family],
                "folds": folds,
            }
        )
    best_accuracy = max(row["macro_accuracy"] for row in ranking)
    eligible = [row for row in ranking if row["macro_accuracy"] >= best_accuracy - 0.005]
    winner = max(
        eligible,
        key=lambda row: (
            row["macro_minimum_recall"],
            -row["macro_nll"],
            -row["complexity"],
            "".join(chr(255 - ord(c)) for c in row["name"]),
        ),
    )
    serializable = [
        {key: value for key, value in row.items() if key != "spec"} for row in ranking
    ]
    return SelectionRecord(selected=winner["spec"], ranking=serializable)


__all__ = [
    "SelectionRecord",
    "ThresholdRecord",
    "choose_accuracy_threshold",
    "choose_balanced_threshold",
    "select_source_lodo",
]
