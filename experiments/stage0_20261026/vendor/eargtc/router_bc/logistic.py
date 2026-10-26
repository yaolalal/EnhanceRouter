from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch


@dataclass(frozen=True)
class LogisticModel:
    coef: torch.Tensor
    intercept: float
    adaptation_bias: float = 0.0

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=torch.float64) @ self.coef.to(dtype=torch.float64) + float(self.intercept) + float(self.adaptation_bias)

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(x)).to(torch.float32)


def domain_balanced_weights(domains: Sequence[str]) -> torch.Tensor:
    if not domains:
        return torch.empty(0, dtype=torch.float64)
    counts: dict[str, int] = {}
    for domain in domains:
        counts[str(domain)] = counts.get(str(domain), 0) + 1
    raw = torch.tensor([1.0 / counts[str(domain)] for domain in domains], dtype=torch.float64)
    return raw / raw.mean()


def fit_logistic(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    l2: float = 1.0,
    max_iter: int = 100,
    tolerance: float = 1e-9,
) -> LogisticModel:
    x64 = torch.as_tensor(x, dtype=torch.float64)
    y64 = torch.as_tensor(y, dtype=torch.float64).view(-1)
    if x64.ndim != 2 or x64.size(0) != y64.numel() or y64.numel() == 0:
        raise ValueError("fit_logistic requires non-empty X[N,D] and y[N]")
    if bool(((y64 != 0) & (y64 != 1)).any().item()):
        raise ValueError("Logistic targets must be binary")
    weights = torch.ones_like(y64) if sample_weight is None else torch.as_tensor(sample_weight, dtype=torch.float64).view(-1)
    if weights.numel() != y64.numel() or bool((weights <= 0).any().item()):
        raise ValueError("sample_weight must contain one positive value per row")
    weights = weights / weights.sum()
    design = torch.cat([x64, torch.ones((x64.size(0), 1), dtype=torch.float64)], dim=1)
    params = torch.zeros(design.size(1), dtype=torch.float64)
    prior = torch.zeros(design.size(1), dtype=torch.float64)
    prior[:-1] = max(float(l2), 0.0)

    for _ in range(max(int(max_iter), 1)):
        logits = design @ params
        prob = torch.sigmoid(logits).clamp(1e-9, 1.0 - 1e-9)
        gradient = design.T @ (weights * (prob - y64)) + prior * params
        curvature = weights * prob * (1.0 - prob)
        hessian = design.T @ (curvature[:, None] * design) + torch.diag(prior + 1e-8)
        try:
            step = torch.linalg.solve(hessian, gradient)
        except RuntimeError:
            step = torch.linalg.pinv(hessian) @ gradient
        params_next = params - step
        if float(step.abs().max().item()) < float(tolerance):
            params = params_next
            break
        params = params_next
    return LogisticModel(coef=params[:-1].detach().clone(), intercept=float(params[-1].item()))


def adapt_scalar_intercept(
    model: LogisticModel,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    prior_precision: float = 1.0,
    max_iter: int = 100,
) -> LogisticModel:
    x64 = torch.as_tensor(x, dtype=torch.float64)
    y64 = torch.as_tensor(y, dtype=torch.float64).view(-1)
    if y64.numel() == 0:
        return replace(model, adaptation_bias=0.0)
    weights = torch.ones_like(y64) if sample_weight is None else torch.as_tensor(sample_weight, dtype=torch.float64).view(-1)
    weights = weights / weights.sum().clamp_min(1e-12)
    base = x64 @ model.coef.to(torch.float64) + float(model.intercept)
    delta = torch.tensor(0.0, dtype=torch.float64)
    precision = max(float(prior_precision), 0.0)
    for _ in range(max(int(max_iter), 1)):
        p = torch.sigmoid(base + delta).clamp(1e-9, 1.0 - 1e-9)
        gradient = (weights * (p - y64)).sum() + precision * delta
        hessian = (weights * p * (1.0 - p)).sum() + precision + 1e-8
        step = gradient / hessian
        delta_next = delta - step
        if abs(float(step.item())) < 1e-10:
            delta = delta_next
            break
        delta = delta_next
    return LogisticModel(
        coef=model.coef.detach().clone(),
        intercept=float(model.intercept),
        adaptation_bias=float(delta.item()),
    )

