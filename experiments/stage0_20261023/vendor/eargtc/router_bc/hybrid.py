from __future__ import annotations

import torch

from .evaluation import select_preference_threshold


def blend_preference_logits(
    linear_probability: torch.Tensor,
    nonlinear_probability: torch.Tensor,
    residual_fraction: float,
) -> torch.Tensor:
    """Interpolate on the log-odds path from linear to nonlinear evidence."""
    linear = torch.as_tensor(linear_probability, dtype=torch.float64).clamp(1e-6, 1.0 - 1e-6)
    nonlinear = torch.as_tensor(nonlinear_probability, dtype=torch.float64).clamp(1e-6, 1.0 - 1e-6)
    fraction = float(residual_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("residual_fraction must be in [0,1]")
    linear_logit = torch.logit(linear)
    nonlinear_logit = torch.logit(nonlinear)
    return torch.sigmoid(linear_logit + fraction * (nonlinear_logit - linear_logit)).to(torch.float32)


def select_early_stopping_point(
    linear_probability: torch.Tensor,
    nonlinear_probability: torch.Tensor,
    labels: torch.Tensor,
    gnn_pred: torch.Tensor,
    llm_pred: torch.Tensor,
    *,
    lambdas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict:
    labels = torch.as_tensor(labels).view(-1)
    gnn_pred = torch.as_tensor(gnn_pred).view(-1)
    llm_pred = torch.as_tensor(llm_pred).view(-1)
    g_correct = gnn_pred == labels
    l_correct = llm_pred == labels
    conflict = gnn_pred != llm_pred
    trials = []
    for residual_fraction in lambdas:
        score = blend_preference_logits(
            linear_probability, nonlinear_probability, residual_fraction
        )
        threshold = select_preference_threshold(
            score,
            g_correct,
            l_correct,
            conflict,
            default_threshold=0.5,
        )
        choose_g = (~conflict) | (score >= float(threshold["threshold"]))
        selected = torch.where(choose_g, gnn_pred, llm_pred)
        accuracy = float((selected == labels).float().mean().item())
        trials.append(
            {
                "lambda": float(residual_fraction),
                "threshold": threshold,
                "router_accuracy": accuracy,
            }
        )
    # A tie stops earlier on the residual path, which is the lower-capacity model.
    best = min(trials, key=lambda row: (-row["router_accuracy"], row["lambda"]))
    return {**best, "selection": "official_validation_early_stopping", "trials": trials}
