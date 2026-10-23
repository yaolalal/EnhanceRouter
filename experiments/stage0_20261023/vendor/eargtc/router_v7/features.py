from __future__ import annotations

import math
from collections.abc import Mapping

import torch


PER_EXPERT_NAMES = [
    "entropy",
    "margin",
    "top1",
    "top2",
    "top3",
]
SIGNED_NAMES = ["top1_gap_signed", "entropy_gap_signed", "margin_gap_signed"]
SOFT_GEOMETRY_NAMES = [
    "p_g_at_l",
    "p_l_at_g",
    "js_divergence",
    "bhattacharyya_overlap",
]
STRUCTURAL_NAMES = [
    "log_degree",
    "neighbor_gnn_agreement",
    "neighbor_gnn_entropy",
]
CONTEXT_NAMES = [
    "ctx_gnn_accuracy_mean",
    "ctx_gnn_accuracy_variance",
    "ctx_llm_accuracy_mean",
    "ctx_llm_accuracy_variance",
]


def _probabilities(value: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64)
    if tensor.ndim != 2 or tensor.size(1) < 2:
        raise ValueError(f"{name} probabilities must have shape [N,C] with C>=2")
    if not bool(torch.isfinite(tensor).all()) or bool((tensor < 0).any()):
        raise ValueError(f"{name} probabilities must be finite and non-negative")
    total = tensor.sum(dim=1, keepdim=True)
    if bool((total <= 0).any()):
        raise ValueError(f"{name} probability rows must have positive mass")
    return tensor / total


def _summary(probability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    classes = probability.size(1)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=1) / math.log(classes)
    top = probability.sort(dim=1, descending=True).values[:, : min(3, classes)]
    if top.size(1) < 3:
        top = torch.cat(
            [top, torch.zeros((top.size(0), 3 - top.size(1)), dtype=top.dtype)],
            dim=1,
        )
    margin = top[:, 0] - top[:, 1]
    return entropy, margin, top


def local_evidence(
    gnn_prob: torch.Tensor,
    llm_prob: torch.Tensor,
    structural: Mapping[str, torch.Tensor],
    *,
    include_soft_geometry: bool = True,
) -> tuple[torch.Tensor, list[str]]:
    g = _probabilities(gnn_prob, "GNN")
    l = _probabilities(llm_prob, "LLM")
    if g.shape != l.shape:
        raise ValueError("expert probability shape mismatch")
    g_entropy, g_margin, g_top = _summary(g)
    l_entropy, l_margin, l_top = _summary(l)
    columns = [g_entropy, g_margin, *g_top.unbind(1), l_entropy, l_margin, *l_top.unbind(1)]
    names = [f"g_{name}" for name in PER_EXPERT_NAMES] + [
        f"l_{name}" for name in PER_EXPERT_NAMES
    ]
    columns.extend([g_top[:, 0] - l_top[:, 0], g_entropy - l_entropy, g_margin - l_margin])
    names.extend(SIGNED_NAMES)
    if include_soft_geometry:
        rows = torch.arange(g.size(0), dtype=torch.long)
        g_pred, l_pred = g.argmax(1), l.argmax(1)
        midpoint = 0.5 * (g + l)
        js = 0.5 * (
            (g * (g.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(1)
            + (l * (l.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(1)
        ) / math.log(2.0)
        columns.extend([g[rows, l_pred], l[rows, g_pred], js, torch.sqrt(g * l).sum(1)])
        names.extend(SOFT_GEOMETRY_NAMES)
    for name in STRUCTURAL_NAMES:
        if name not in structural:
            raise ValueError(f"missing structural feature: {name}")
        value = torch.as_tensor(structural[name], dtype=torch.float64).view(-1)
        if value.numel() != g.size(0) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"invalid structural feature: {name}")
        columns.append(value)
        names.append(name)
    return torch.stack(columns, dim=1).to(torch.float32), names


def _posterior(success: torch.Tensor, count: int, alpha: float, beta: float) -> tuple[float, float]:
    a = float(success.sum().item()) + float(alpha)
    b = float(count - int(success.sum().item())) + float(beta)
    total = a + b
    return a / total, (a * b) / (total * total * (total + 1.0))


def beta_accuracy_context(
    gnn_correct: torch.Tensor,
    llm_correct: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> tuple[torch.Tensor, list[str]]:
    g = torch.as_tensor(gnn_correct, dtype=torch.bool).view(-1)
    l = torch.as_tensor(llm_correct, dtype=torch.bool).view(-1)
    if g.numel() != l.numel():
        raise ValueError("context correctness arrays must have equal length")
    if float(alpha) <= 0 or float(beta) <= 0:
        raise ValueError("Beta context priors must be positive")
    g_mean, g_variance = _posterior(g, g.numel(), alpha, beta)
    l_mean, l_variance = _posterior(l, l.numel(), alpha, beta)
    return (
        torch.tensor([g_mean, g_variance, l_mean, l_variance], dtype=torch.float32),
        list(CONTEXT_NAMES),
    )


def leave_one_out_accuracy_context(
    gnn_correct: torch.Tensor,
    llm_correct: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> tuple[torch.Tensor, list[str]]:
    g = torch.as_tensor(gnn_correct, dtype=torch.bool).view(-1)
    l = torch.as_tensor(llm_correct, dtype=torch.bool).view(-1)
    if g.numel() != l.numel():
        raise ValueError("context correctness arrays must have equal length")
    values = []
    for index in range(g.numel()):
        keep = torch.ones(g.numel(), dtype=torch.bool)
        keep[index] = False
        values.append(beta_accuracy_context(g[keep], l[keep], alpha=alpha, beta=beta)[0])
    result = torch.stack(values) if values else torch.empty((0, 4), dtype=torch.float32)
    return result, list(CONTEXT_NAMES)
