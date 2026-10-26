"""Feature views for the two disjoint Scheme-C reliability tasks.

The builders deliberately remove the binary agreement shortcut.  Conflict
features describe the label actually selected by the frozen Scheme-B router;
agreement features never receive Scheme-B route information.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


_CONFLICT_DROP = frozenset({"agreement", "ctx_agreement_correct"})
_AGREEMENT_DROP = frozenset(
    {
        "agreement",
        "ctx_conflict_any_correct",
        "ctx_conflict_gnn_win",
        "ctx_conflict_rate",
    }
)
_EXPERT_SUMMARIES = ("entropy", "margin", "top1")


def _validate_evidence(
    evidence: torch.Tensor, feature_names: Sequence[str]
) -> tuple[torch.Tensor, list[str]]:
    matrix = torch.as_tensor(evidence, dtype=torch.float32)
    names = list(feature_names)
    if matrix.ndim != 2:
        raise ValueError(f"evidence must be 2-D, got shape {tuple(matrix.shape)}")
    if matrix.shape[1] != len(names):
        raise ValueError(
            f"feature-name count {len(names)} does not match width {matrix.shape[1]}"
        )
    if len(set(names)) != len(names):
        raise ValueError("feature names must be unique")
    if not torch.isfinite(matrix).all():
        raise ValueError("evidence contains non-finite values")
    return matrix, names


def _filtered_view(
    evidence: torch.Tensor, names: list[str], drop: frozenset[str]
) -> tuple[torch.Tensor, list[str]]:
    keep = [index for index, name in enumerate(names) if name not in drop]
    return evidence[:, keep], [names[index] for index in keep]


def agreement_features(
    evidence: torch.Tensor, feature_names: Sequence[str]
) -> tuple[torch.Tensor, list[str]]:
    """Return agreement-only evidence, excluding conflict and route context."""

    matrix, names = _validate_evidence(evidence, feature_names)
    return _filtered_view(matrix, names, _AGREEMENT_DROP)


def conflict_features(
    evidence: torch.Tensor,
    feature_names: Sequence[str],
    *,
    choose_g: torch.Tensor,
    b_score: torch.Tensor,
    b_threshold: float,
) -> tuple[torch.Tensor, list[str]]:
    """Return evidence for whether frozen Scheme B's selected label is correct.

    ``b_logit_margin`` is signed relative to the frozen threshold.  The
    selected/unselected expert summaries make the representation route-aware
    without exposing a label-dependent agreement flag.
    """

    matrix, names = _validate_evidence(evidence, feature_names)
    route = torch.as_tensor(choose_g, dtype=torch.bool).flatten()
    score = torch.as_tensor(b_score, dtype=torch.float32).flatten()
    if route.numel() != matrix.shape[0] or score.numel() != matrix.shape[0]:
        raise ValueError("choose_g and b_score must align with evidence rows")
    if not torch.isfinite(score).all():
        raise ValueError("b_score contains non-finite values")
    if not 0.0 < float(b_threshold) < 1.0:
        raise ValueError("b_threshold must be strictly between zero and one")

    indices = {name: index for index, name in enumerate(names)}
    required = {
        f"{expert}_{summary}" for expert in ("g", "l") for summary in _EXPERT_SUMMARIES
    }
    missing = sorted(required.difference(indices))
    if missing:
        raise ValueError(f"missing expert summary features: {missing}")

    base, base_names = _filtered_view(matrix, names, _CONFLICT_DROP)
    route_float = route.to(matrix.dtype)
    eps = torch.finfo(score.dtype).eps
    clipped = score.clamp(eps, 1.0 - eps)
    score_logit = torch.logit(clipped)
    threshold_logit = math.log(float(b_threshold) / (1.0 - float(b_threshold)))

    additions: list[torch.Tensor] = [route_float, score_logit - threshold_logit]
    addition_names = ["b_choose_g", "b_logit_margin"]
    for selected in (True, False):
        prefix = "selected" if selected else "unselected"
        choose_first = route if selected else ~route
        for summary in _EXPERT_SUMMARIES:
            g_value = matrix[:, indices[f"g_{summary}"]]
            l_value = matrix[:, indices[f"l_{summary}"]]
            additions.append(torch.where(choose_first, g_value, l_value))
            addition_names.append(f"{prefix}_{summary}")

    return torch.column_stack([base, *additions]), base_names + addition_names


__all__ = ["agreement_features", "conflict_features"]
