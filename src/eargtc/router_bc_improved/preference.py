from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

def configure_router_threads(num_threads: int = 1) -> int:
    """Set the small-matrix Router solver thread count and return the old value."""
    requested = int(num_threads)
    if requested <= 0:
        raise ValueError("num_threads must be positive")
    previous = torch.get_num_threads()
    torch.set_num_threads(requested)
    return previous


@dataclass(frozen=True)
class PreferenceModel:
    coef: torch.Tensor
    intercept: float

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(x, dtype=torch.float64)
        return value @ self.coef.to(torch.float64) + float(self.intercept)

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(x)).to(torch.float32)


def pairwise_domain_weights(domains: Sequence[str]) -> torch.Tensor:
    if not domains:
        return torch.empty(0, dtype=torch.float64)
    counts: dict[str, int] = {}
    for domain in domains:
        key = str(domain)
        counts[key] = counts.get(key, 0) + 1
    weight = torch.tensor(
        [1.0 / counts[str(domain)] for domain in domains], dtype=torch.float64
    )
    return weight / weight.mean()


def _domain_pair_differences(
    x: torch.Tensor, y: torch.Tensor, domains: Sequence[str]
) -> list[torch.Tensor]:
    result = []
    for domain in sorted(set(str(value) for value in domains)):
        mask = torch.tensor([str(value) == domain for value in domains], dtype=torch.bool)
        positive = x[mask & (y == 1)]
        negative = x[mask & (y == 0)]
        if positive.numel() and negative.numel():
            result.append((positive[:, None, :] - negative[None, :, :]).reshape(-1, x.size(1)))
    return result


def fit_pairwise_preference(
    x: torch.Tensor,
    y: torch.Tensor,
    domains: Sequence[str],
    *,
    l2: float = 1.0,
    rank_weight: float = 0.0,
    max_iter: int = 200,
) -> PreferenceModel:
    value = torch.as_tensor(x, dtype=torch.float64)
    target = torch.as_tensor(y, dtype=torch.float64).view(-1)
    if value.ndim != 2 or value.size(0) != target.numel() or target.numel() == 0:
        raise ValueError("Expected non-empty X[N,D] and y[N]")
    if len(domains) != target.numel():
        raise ValueError("domains must contain one name per row")
    if bool(((target != 0) & (target != 1)).any().item()):
        raise ValueError("Preference targets must be binary")
    if float(l2) < 0 or float(rank_weight) < 0:
        raise ValueError("l2 and rank_weight must be non-negative")

    row_weight = pairwise_domain_weights(domains)
    row_weight = row_weight / row_weight.sum()
    pair_differences = _domain_pair_differences(value, target, domains)
    design = torch.cat(
        [value, torch.ones((value.size(0), 1), dtype=torch.float64)], dim=1
    )
    params = torch.zeros(design.size(1), dtype=torch.float64)
    prior = torch.zeros_like(params)
    prior[:-1] = float(l2)
    for _ in range(max(int(max_iter), 1)):
        probability = torch.sigmoid(design @ params).clamp(1e-9, 1.0 - 1e-9)
        gradient = design.T @ (row_weight * (probability - target)) + prior * params
        curvature = row_weight * probability * (1.0 - probability)
        hessian = design.T @ (curvature[:, None] * design) + torch.diag(prior + 1e-8)
        if pair_differences and float(rank_weight) > 0.0:
            pair_scale = float(rank_weight) / len(pair_differences)
            for difference in pair_differences:
                losing_probability = torch.sigmoid(-(difference @ params[:-1]))
                gradient[:-1] -= pair_scale * (
                    losing_probability[:, None] * difference
                ).mean(dim=0)
                pair_curvature = losing_probability * (1.0 - losing_probability)
                hessian[:-1, :-1] += pair_scale * (
                    difference.T @ (pair_curvature[:, None] * difference)
                ) / difference.size(0)
        try:
            step = torch.linalg.solve(hessian, gradient)
        except RuntimeError:
            step = torch.linalg.pinv(hessian) @ gradient
        params_next = params - step
        params = params_next
        if float(step.abs().max().item()) < 1e-9:
            break
    return PreferenceModel(params[:-1].detach().clone(), float(params[-1].item()))

