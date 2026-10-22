from __future__ import annotations

from dataclasses import dataclass

import torch

from eargtc.router_bc.evaluation import select_preference_threshold


@dataclass(frozen=True)
class CheckpointCandidate:
    name: str
    score: torch.Tensor
    view: str
    complexity: int


def select_b_checkpoint(
    candidates: list[CheckpointCandidate],
    labels: torch.Tensor,
    gnn_pred: torch.Tensor,
    llm_pred: torch.Tensor,
) -> dict:
    if not candidates:
        raise ValueError("At least one validation checkpoint is required")
    y = torch.as_tensor(labels).view(-1)
    g = torch.as_tensor(gnn_pred).view(-1)
    l = torch.as_tensor(llm_pred).view(-1)
    if not (y.numel() == g.numel() == l.numel()):
        raise ValueError("Validation labels and predictions must have equal length")
    conflict = g != l
    decisive = conflict & ((g == y) != (l == y))
    trials = []
    for candidate in candidates:
        score = torch.as_tensor(candidate.score, dtype=torch.float64).view(-1)
        if score.numel() != y.numel():
            raise ValueError("Checkpoint score length differs from validation length")
        threshold = select_preference_threshold(
            score, g == y, l == y, conflict, default_threshold=0.5
        )
        choose_g = (~conflict) | (score >= float(threshold["threshold"]))
        selected = torch.where(choose_g, g, l)
        correct = int((selected == y).sum().item())
        decisive_correct = int(((score[decisive] >= float(threshold["threshold"])) == (g[decisive] == y[decisive])).sum().item())
        trials.append(
            {
                "name": candidate.name,
                "view": candidate.view,
                "complexity": int(candidate.complexity),
                "threshold": threshold,
                "correct": correct,
                "accuracy": correct / y.numel(),
                "decisive_correct": decisive_correct,
            }
        )
    best = min(
        trials,
        key=lambda row: (
            -row["correct"],
            -row["decisive_correct"],
            row["complexity"],
            0 if row["view"] == "raw" else 1,
            row["name"],
        ),
    )
    return {**best, "selection": "official_validation_checkpoint_and_threshold", "trials": trials}

