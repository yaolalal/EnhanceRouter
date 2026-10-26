from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import RouterV7Output
from .pairs import PairBatch


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if value.numel() == 0:
        return value.sum() * 0.0
    total = weight.sum()
    if not bool(torch.isfinite(total)) or float(total.detach()) <= 0:
        raise ValueError("loss weights must have positive finite mass")
    return (value * weight).sum() / total


def selected_trust_target(
    preference_logit: torch.Tensor,
    gnn_correct: torch.Tensor,
    llm_correct: torch.Tensor,
) -> torch.Tensor:
    """Detached probability that the current soft-routed label is correct."""
    score = torch.as_tensor(preference_logit).view(-1)
    g = torch.as_tensor(gnn_correct, dtype=score.dtype, device=score.device).view(-1)
    l = torch.as_tensor(llm_correct, dtype=score.dtype, device=score.device).view(-1)
    if not (score.numel() == g.numel() == l.numel()):
        raise ValueError("selected trust inputs must have equal length")
    with torch.no_grad():
        choose_g = torch.sigmoid(score.detach())
        return choose_g * g + (1.0 - choose_g) * l


def preference_point_loss(
    preference_logit: torch.Tensor,
    gnn_correct: torch.Tensor,
    llm_correct: torch.Tensor,
    row_weight: torch.Tensor,
) -> torch.Tensor:
    score = torch.as_tensor(preference_logit).view(-1)
    g = torch.as_tensor(gnn_correct, dtype=torch.bool, device=score.device).view(-1)
    l = torch.as_tensor(llm_correct, dtype=torch.bool, device=score.device).view(-1)
    weight = torch.as_tensor(row_weight, dtype=score.dtype, device=score.device).view(-1)
    if not (score.numel() == g.numel() == l.numel() == weight.numel()):
        raise ValueError("preference loss inputs must have equal length")
    decisive = g ^ l
    if not bool(decisive.any()):
        return score.sum() * 0.0
    point = F.binary_cross_entropy_with_logits(
        score[decisive], g[decisive].to(score.dtype), reduction="none"
    )
    return _weighted_mean(point, weight[decisive])


def shift_ranking_loss(
    preference_logit: torch.Tensor, pairs: PairBatch, *, temperature: float = 1.0
) -> torch.Tensor:
    score = torch.as_tensor(preference_logit).view(-1)
    if float(temperature) <= 0:
        raise ValueError("shift temperature must be positive")
    if len(pairs) == 0:
        return score.sum() * 0.0
    left = pairs.left.to(score.device)
    right = pairs.right.to(score.device)
    if int(torch.cat([left, right]).max()) >= score.numel():
        raise ValueError("pair index exceeds preference scores")
    direction = pairs.direction.to(score.device, score.dtype)
    weight = pairs.weight.to(score.device, score.dtype)
    change = (score[right] - score[left]) / float(temperature)
    return _weighted_mean(F.softplus(-direction * change), weight)


def shift_regression_loss(
    preference_logit: torch.Tensor,
    pairs: PairBatch,
    *,
    temperature: float = 1.0,
    delta_clip: float = 4.0,
) -> torch.Tensor:
    """Match same-node score changes to bounded true-advantage changes."""
    score = torch.as_tensor(preference_logit).view(-1)
    if float(temperature) <= 0 or float(delta_clip) <= 0:
        raise ValueError("shift temperature and delta_clip must be positive")
    if len(pairs) == 0:
        return score.sum() * 0.0
    left = pairs.left.to(score.device)
    right = pairs.right.to(score.device)
    if int(torch.cat([left, right]).max()) >= score.numel():
        raise ValueError("pair index exceeds preference scores")
    weight = pairs.weight.to(score.device, score.dtype)
    change = (score[right] - score[left]) / float(temperature)
    target = pairs.delta_advantage.to(score.device, score.dtype).clamp(
        -float(delta_clip), float(delta_clip)
    )
    return _weighted_mean(F.smooth_l1_loss(change, target, reduction="none"), weight)


def router_v7_loss(
    output: RouterV7Output,
    *,
    gnn_correct: torch.Tensor,
    llm_correct: torch.Tensor,
    trust_target: torch.Tensor | None,
    row_weight: torch.Tensor,
    pairs: PairBatch,
    clean_mask: torch.Tensor,
    shift_weight: float = 1.0,
    clean_weight: float = 0.5,
    shift_objective: str = "huber",
    delta_clip: float = 4.0,
    trust_weight: float = 1.0,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    score = output.preference_logit.view(-1)
    weight = torch.as_tensor(row_weight, dtype=score.dtype, device=score.device).view(-1)
    clean = torch.as_tensor(clean_mask, dtype=torch.bool, device=score.device).view(-1)
    target = (
        selected_trust_target(score, gnn_correct, llm_correct)
        if trust_target is None
        else torch.as_tensor(trust_target, dtype=score.dtype, device=score.device).view(-1)
    )
    if not (score.numel() == weight.numel() == clean.numel() == target.numel()):
        raise ValueError("Router-v7 loss row inputs must have equal length")
    point = preference_point_loss(score, gnn_correct, llm_correct, weight)
    if shift_objective == "huber":
        shift = shift_regression_loss(
            score, pairs, temperature=temperature, delta_clip=delta_clip
        )
    elif shift_objective == "ranking":
        shift = shift_ranking_loss(score, pairs, temperature=temperature)
    else:
        raise ValueError(
            f"unsupported shift_objective={shift_objective!r}"
        )
    if bool(clean.any()):
        clean_loss = preference_point_loss(
            score[clean],
            torch.as_tensor(gnn_correct, device=score.device).view(-1)[clean],
            torch.as_tensor(llm_correct, device=score.device).view(-1)[clean],
            weight[clean],
        )
    else:
        clean_loss = score.sum() * 0.0
    trust_value = F.binary_cross_entropy_with_logits(
        output.trust_logit.view(-1), target, reduction="none"
    )
    trust = _weighted_mean(trust_value, weight)
    total = point + float(shift_weight) * shift + float(clean_weight) * clean_loss + float(trust_weight) * trust
    values = {"total": total, "point": point, "shift": shift, "clean": clean_loss, "trust": trust}
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise FloatingPointError("Router-v7 loss became non-finite")
    return values
