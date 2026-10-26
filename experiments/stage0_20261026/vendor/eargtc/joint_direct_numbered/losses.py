from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossWeights:
    gold: float = 1.0
    hard: float = 1.0
    agreement_kd: float = 0.1
    conflict_kd: float = 0.05


def selected_expert_teacher(
    gnn_logits: torch.Tensor,
    llm_logits: torch.Tensor,
    choose_gnn: torch.Tensor,
) -> torch.Tensor:
    gnn = torch.softmax(torch.as_tensor(gnn_logits).float(), dim=-1)
    llm = torch.softmax(torch.as_tensor(llm_logits).float(), dim=-1)
    choose = torch.as_tensor(choose_gnn, dtype=torch.bool, device=gnn.device).view(-1)
    if gnn.shape != llm.shape or choose.numel() != gnn.size(0):
        raise ValueError("expert teacher inputs have inconsistent shapes")
    return torch.where(choose[:, None], gnn, llm)


def agreement_teacher(
    gnn_logits: torch.Tensor, llm_logits: torch.Tensor
) -> torch.Tensor:
    gnn = torch.softmax(torch.as_tensor(gnn_logits).float(), dim=-1)
    llm = torch.softmax(torch.as_tensor(llm_logits).float(), dim=-1)
    if gnn.shape != llm.shape:
        raise ValueError("agreement teacher inputs have inconsistent shapes")
    return 0.5 * (gnn + llm)


def _ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (
        F.cross_entropy(logits.float(), target.long())
        if logits.size(0) else logits.sum() * 0.0
    )


def _kd(logits: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    if not logits.size(0):
        return logits.sum() * 0.0
    probability = torch.as_tensor(teacher, device=logits.device, dtype=torch.float32)
    probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return F.kl_div(
        F.log_softmax(logits.float(), dim=-1), probability, reduction="batchmean"
    )


def direct_joint_losses(
    *,
    gold_logits: torch.Tensor,
    gold_targets: torch.Tensor,
    hard_logits: torch.Tensor,
    hard_targets: torch.Tensor,
    agreement_logits: torch.Tensor,
    agreement_teacher: torch.Tensor,
    conflict_logits: torch.Tensor,
    conflict_teacher: torch.Tensor,
    weights: LossWeights = LossWeights(),
) -> dict[str, torch.Tensor]:
    gold = _ce(gold_logits, gold_targets)
    hard = _ce(hard_logits, hard_targets)
    agreement = _kd(agreement_logits, agreement_teacher)
    conflict = _kd(conflict_logits, conflict_teacher)
    total = (
        weights.gold * gold
        + weights.hard * hard
        + weights.agreement_kd * agreement
        + weights.conflict_kd * conflict
    )
    return {
        "gold_ce": gold,
        "hard_ce": hard,
        "agreement_kd": agreement,
        "conflict_kd": conflict,
        "total": total,
    }


def single_pool_loss(
    pool: str,
    logits: torch.Tensor,
    target: torch.Tensor,
    weights: LossWeights = LossWeights(),
) -> torch.Tensor:
    if pool == "gold":
        return weights.gold * _ce(logits, target)
    if pool == "hard":
        return weights.hard * _ce(logits, target)
    if pool == "agreement":
        return weights.agreement_kd * _kd(logits, target)
    if pool == "conflict":
        return weights.conflict_kd * _kd(logits, target)
    raise ValueError(f"unknown loss pool: {pool}")

