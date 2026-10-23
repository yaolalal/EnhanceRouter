from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossWeights:
    hard: float = 0.5
    agreement_kd: float = 0.2
    conflict_kd: float = 0.1
    residual: float = 0.01


def _residual_penalty(gamma: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    if gamma.numel():
        return (gamma.float().square() * residual.float().square().sum(dim=-1)).mean()
    return residual.sum() * 0.0 + gamma.sum() * 0.0


def selected_expert_teacher(
    gnn_logits: torch.Tensor,
    llm_logits: torch.Tensor,
    choose_gnn: torch.Tensor,
) -> torch.Tensor:
    g = torch.softmax(torch.as_tensor(gnn_logits).float(), dim=-1)
    l = torch.softmax(torch.as_tensor(llm_logits).float(), dim=-1)
    route = torch.as_tensor(choose_gnn, dtype=torch.bool, device=g.device).view(-1)
    if g.shape != l.shape or route.numel() != g.size(0):
        raise ValueError("expert teacher shapes do not match")
    return torch.where(route[:, None], g, l)


def _ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target) if logits.size(0) else logits.sum() * 0.0


def _kd(logits: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    if not logits.size(0):
        return logits.sum() * 0.0
    probability = torch.as_tensor(teacher, device=logits.device, dtype=torch.float32)
    probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return F.kl_div(F.log_softmax(logits.float(), dim=-1), probability, reduction="batchmean")


def joint_pool_losses(
    *,
    gold_logits: torch.Tensor,
    gold_targets: torch.Tensor,
    hard_logits: torch.Tensor,
    hard_targets: torch.Tensor,
    agreement_logits: torch.Tensor,
    agreement_teacher: torch.Tensor,
    conflict_logits: torch.Tensor,
    conflict_teacher: torch.Tensor,
    gamma: torch.Tensor,
    residual: torch.Tensor,
    weights: LossWeights = LossWeights(),
) -> dict[str, torch.Tensor]:
    gold = _ce(gold_logits, gold_targets.long())
    hard = _ce(hard_logits, hard_targets.long())
    agreement_kd = _kd(agreement_logits, agreement_teacher)
    conflict_kd = _kd(conflict_logits, conflict_teacher)
    residual_loss = _residual_penalty(gamma, residual)
    total = (
        gold
        + weights.hard * hard
        + weights.agreement_kd * agreement_kd
        + weights.conflict_kd * conflict_kd
        + weights.residual * residual_loss
    )
    return {
        "gold": gold,
        "hard": hard,
        "agreement_kd": agreement_kd,
        "conflict_kd": conflict_kd,
        "residual": residual_loss,
        "total": total,
    }


def single_pool_loss(
    pool: str,
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: torch.Tensor,
    residual: torch.Tensor,
    residual_fraction: float,
    weights: LossWeights = LossWeights(),
) -> torch.Tensor:
    """One independently averaged pool term plus its exact residual share.

    Weighting the per-pool residual mean by its fraction of the concatenated
    sampled rows makes the sum of these terms identical to ``joint_pool_losses``.
    The terms can therefore be forwarded and backwarded one at a time without
    retaining every supervision-pool graph on the GPU.
    """
    if not 0.0 <= float(residual_fraction) <= 1.0:
        raise ValueError("residual_fraction must be in [0,1]")
    if pool == "gold":
        supervised = _ce(logits, target.long())
    elif pool == "hard":
        supervised = weights.hard * _ce(logits, target.long())
    elif pool == "agreement":
        supervised = weights.agreement_kd * _kd(logits, target)
    elif pool == "conflict":
        supervised = weights.conflict_kd * _kd(logits, target)
    else:
        raise ValueError(f"unknown supervision pool: {pool}")
    return supervised + weights.residual * float(residual_fraction) * _residual_penalty(gamma, residual)
