from __future__ import annotations

from collections.abc import Mapping

import torch

from eargtc.router_bc.features import invariant_evidence


LOGIT_FEATURE_NAMES = (
    "g_entropy",
    "g_margin",
    "g_top1",
    "g_top2",
    "g_top3",
    "l_entropy",
    "l_margin",
    "l_top1",
    "l_top2",
    "l_top3",
    "p_g_at_l",
    "p_l_at_g",
    "confidence_gap_signed",
    "entropy_gap_signed",
    "margin_gap_signed",
    "js_divergence",
    "bhattacharyya_overlap",
    "log_degree",
    "neighbor_gnn_agreement",
    "neighbor_gnn_entropy",
)


def build_logit_evidence(
    gnn_prob: torch.Tensor,
    llm_prob: torch.Tensor,
    structural: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, list[str]]:
    """Build the approved local 20-column probability/structure evidence.

    The representation intentionally excludes duplicated confidence aliases,
    prediction agreement, and every support/environment context feature.
    """

    evidence, names = invariant_evidence(gnn_prob, llm_prob, structural)
    index = {name: position for position, name in enumerate(names)}
    missing = [name for name in LOGIT_FEATURE_NAMES if name not in index]
    if missing:
        raise ValueError(f"missing approved logit evidence columns: {missing}")
    selected = evidence[:, [index[name] for name in LOGIT_FEATURE_NAMES]]
    if selected.shape[1] != len(LOGIT_FEATURE_NAMES):
        raise AssertionError("approved logit evidence width changed")
    if not torch.isfinite(selected).all():
        raise ValueError("logit evidence contains non-finite values")
    return selected.contiguous(), list(LOGIT_FEATURE_NAMES)


__all__ = ["LOGIT_FEATURE_NAMES", "build_logit_evidence"]

