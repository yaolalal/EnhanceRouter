from __future__ import annotations

import math
from collections.abc import Mapping

import torch


BASE_EVIDENCE_NAMES = [
    "g_entropy",
    "l_entropy",
    "g_confidence",
    "l_confidence",
    "g_margin",
    "l_margin",
    "g_top1",
    "g_top2",
    "g_top3",
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
    "agreement",
]

STRUCTURAL_EVIDENCE_NAMES = [
    "log_degree",
    "neighbor_gnn_agreement",
    "neighbor_gnn_entropy",
]


def _normalized_probabilities(prob: torch.Tensor) -> torch.Tensor:
    if prob.ndim != 2 or prob.size(1) < 2:
        raise ValueError(f"Expected [N,C] probabilities with C>=2, got {tuple(prob.shape)}")
    out = prob.to(dtype=torch.float64)
    if bool((out < 0).any().item()):
        raise ValueError("Probabilities must be non-negative")
    return out / out.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _topk_padded(prob: torch.Tensor, k: int = 3) -> torch.Tensor:
    top = prob.sort(dim=1, descending=True).values[:, : min(k, prob.size(1))]
    if top.size(1) < k:
        top = torch.cat([top, torch.zeros((top.size(0), k - top.size(1)), dtype=top.dtype)], dim=1)
    return top


def invariant_evidence(
    gnn_prob: torch.Tensor,
    llm_prob: torch.Tensor,
    structural: Mapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, list[str]]:
    """Return evidence unchanged by a joint permutation of class indices."""
    g = _normalized_probabilities(gnn_prob)
    l = _normalized_probabilities(llm_prob)
    if g.shape != l.shape:
        raise ValueError(f"Expert probability shapes differ: {tuple(g.shape)} vs {tuple(l.shape)}")

    n, c = g.shape
    denom = math.log(c)
    g_entropy = -(g * g.clamp_min(1e-12).log()).sum(dim=1) / denom
    l_entropy = -(l * l.clamp_min(1e-12).log()).sum(dim=1) / denom
    gt = _topk_padded(g)
    lt = _topk_padded(l)
    g_pred = g.argmax(dim=1)
    l_pred = l.argmax(dim=1)
    rows = torch.arange(n, dtype=torch.long)
    p_g_at_l = g[rows, l_pred]
    p_l_at_g = l[rows, g_pred]
    midpoint = 0.5 * (g + l)
    js = 0.5 * (
        (g * (g.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(dim=1)
        + (l * (l.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(dim=1)
    ) / math.log(2.0)
    bc = torch.sqrt(g * l).sum(dim=1)

    columns = [
        g_entropy,
        l_entropy,
        gt[:, 0],
        lt[:, 0],
        gt[:, 0] - gt[:, 1],
        lt[:, 0] - lt[:, 1],
        gt[:, 0],
        gt[:, 1],
        gt[:, 2],
        lt[:, 0],
        lt[:, 1],
        lt[:, 2],
        p_g_at_l,
        p_l_at_g,
        gt[:, 0] - lt[:, 0],
        g_entropy - l_entropy,
        (gt[:, 0] - gt[:, 1]) - (lt[:, 0] - lt[:, 1]),
        js,
        bc,
        (g_pred == l_pred).to(torch.float64),
    ]
    names = list(BASE_EVIDENCE_NAMES)
    if structural is not None:
        for name in STRUCTURAL_EVIDENCE_NAMES:
            if name not in structural:
                raise ValueError(f"Missing structural evidence: {name}")
            value = torch.as_tensor(structural[name], dtype=torch.float64).view(-1)
            if value.numel() != n:
                raise ValueError(f"Structural evidence {name} has {value.numel()} rows, expected {n}")
            columns.append(value)
            names.append(name)
    return torch.stack(columns, dim=1).to(torch.float32), names

