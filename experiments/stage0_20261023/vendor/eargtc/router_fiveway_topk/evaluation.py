"""Count-safe evaluation helpers for the five-Router comparison."""

from __future__ import annotations

import torch


def population_metrics(accepted: torch.Tensor, correct: torch.Tensor) -> dict:
    mask = torch.as_tensor(accepted, dtype=torch.bool).view(-1)
    target = torch.as_tensor(correct, dtype=torch.bool).view(-1)
    if mask.numel() != target.numel() or mask.numel() == 0:
        raise ValueError("accepted and correct must be aligned and non-empty")
    accepted_count = int(mask.sum())
    if accepted_count == 0:
        raise ValueError("top-k policy must not reject all")
    correct_count = int(target[mask].sum())
    total = int(mask.numel())
    return {
        "total": total,
        "accepted": accepted_count,
        "rejected": total - accepted_count,
        "correct": correct_count,
        "incorrect": accepted_count - correct_count,
        "coverage": accepted_count / total,
        "accuracy": correct_count / accepted_count,
    }


__all__ = ["population_metrics"]
