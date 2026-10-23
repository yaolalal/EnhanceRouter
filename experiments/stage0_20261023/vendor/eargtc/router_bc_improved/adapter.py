from __future__ import annotations

from dataclasses import dataclass

import torch

from .preference import PreferenceModel


@dataclass(frozen=True)
class DomainBasis:
    vectors: torch.Tensor

    def __post_init__(self):
        value = torch.as_tensor(self.vectors, dtype=torch.float64)
        if value.ndim != 2:
            raise ValueError("DomainBasis vectors must be [D,K]")
        if value.size(1) > 2:
            raise ValueError("Target adapter rank is capped at two")
        object.__setattr__(self, "vectors", value.detach().clone())

    @property
    def rank(self) -> int:
        return int(self.vectors.size(1))


@dataclass(frozen=True)
class AdaptedPreference:
    base: PreferenceModel
    basis: DomainBasis
    coordinates: torch.Tensor
    bias: float

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(x, dtype=torch.float64)
        residual = value @ self.basis.vectors @ self.coordinates
        return self.base.logits(value) + residual + float(self.bias)

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(x)).to(torch.float32)


def adapt_low_rank(
    base: PreferenceModel,
    basis: DomainBasis,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    precision: float = 1.0,
    max_iter: int = 100,
) -> AdaptedPreference:
    value = torch.as_tensor(x, dtype=torch.float64)
    target = torch.as_tensor(y, dtype=torch.float64).view(-1)
    if value.ndim != 2 or value.size(0) != target.numel():
        raise ValueError("Target adapter expects X[N,D] and y[N]")
    if value.size(1) != base.coef.numel() or basis.vectors.size(0) != value.size(1):
        raise ValueError("Adapter feature dimensions do not match the base model")
    if float(precision) < 0:
        raise ValueError("precision must be non-negative")
    if target.numel() == 0:
        return AdaptedPreference(
            base, basis, torch.zeros(basis.rank, dtype=torch.float64), 0.0
        )
    design = torch.cat(
        [value @ basis.vectors, torch.ones((value.size(0), 1), dtype=torch.float64)], dim=1
    )
    offset = base.logits(value).detach()
    params = torch.zeros(design.size(1), dtype=torch.float64)
    prior = torch.full_like(params, float(precision))
    for _ in range(max(int(max_iter), 1)):
        probability = torch.sigmoid(offset + design @ params).clamp(1e-9, 1.0 - 1e-9)
        gradient = design.T @ ((probability - target) / target.numel()) + prior * params
        curvature = probability * (1.0 - probability) / target.numel()
        hessian = design.T @ (curvature[:, None] * design) + torch.diag(prior + 1e-8)
        try:
            step = torch.linalg.solve(hessian, gradient)
        except RuntimeError:
            step = torch.linalg.pinv(hessian) @ gradient
        params_next = params - step
        params = params_next
        if float(step.abs().max().item()) < 1e-10:
            break
    return AdaptedPreference(base, basis, params[:-1].detach().clone(), float(params[-1].item()))


def build_domain_basis(
    global_coef: torch.Tensor,
    domain_coefs: list[torch.Tensor],
    rank: int,
) -> DomainBasis:
    requested = int(rank)
    if requested < 0 or requested > 2:
        raise ValueError("rank must be in {0,1,2}")
    center = torch.as_tensor(global_coef, dtype=torch.float64).view(-1)
    if requested == 0 or not domain_coefs:
        return DomainBasis(torch.empty((center.numel(), 0), dtype=torch.float64))
    deviations = torch.stack(
        [torch.as_tensor(coef, dtype=torch.float64).view(-1) - center for coef in domain_coefs]
    )
    _, _, vh = torch.linalg.svd(deviations, full_matrices=False)
    available = min(requested, vh.size(0))
    vectors = vh[:available].T.contiguous()
    for index in range(vectors.size(1)):
        pivot = int(vectors[:, index].abs().argmax().item())
        if float(vectors[pivot, index]) < 0:
            vectors[:, index] *= -1.0
    if available < requested:
        vectors = torch.cat(
            [vectors, torch.zeros((center.numel(), requested - available), dtype=torch.float64)], dim=1
        )
    return DomainBasis(vectors)

