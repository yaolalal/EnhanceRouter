"""Independent residual repairs around two frozen expert predictors."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualHead(nn.Module):
    """One direction-specific, independently normalized residual MLP."""

    def __init__(
        self, gnn_dim: int, llm_dim: int, hidden_dim: int, num_classes: int
    ) -> None:
        super().__init__()
        self.gnn_norm = nn.LayerNorm(gnn_dim)
        self.llm_norm = nn.LayerNorm(llm_dim)
        self.hidden = nn.Linear(gnn_dim + llm_dim, hidden_dim)
        self.activation = nn.GELU()
        self.output = nn.Linear(hidden_dim, num_classes)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, graph: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        features = torch.cat((self.gnn_norm(graph), self.llm_norm(text)), dim=-1)
        return self.output(self.activation(self.hidden(features)))


class DualResidualRepair(nn.Module):
    """Learn one residual correction for each frozen expert's logits.

    The two heads have no trainable parameters in common. Frozen feature and
    base-logit tensors are detached at this boundary, so this module cannot
    update either upstream expert or its feature cache.
    """

    def __init__(
        self,
        gnn_dim: int,
        llm_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.gnn_dim = int(gnn_dim)
        self.llm_dim = int(llm_dim)
        self.num_classes = int(num_classes)
        width = int(hidden_dim)
        if min(self.gnn_dim, self.llm_dim, self.num_classes, width) <= 0:
            raise ValueError("feature, class, and hidden dimensions must be positive")
        self.gnn_head = _ResidualHead(self.gnn_dim, self.llm_dim, width, self.num_classes)
        self.llm_head = _ResidualHead(self.gnn_dim, self.llm_dim, width, self.num_classes)

    def _features(self, g: torch.Tensor, l: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.gnn_head.hidden.weight.device
        dtype = self.gnn_head.hidden.weight.dtype
        graph = torch.as_tensor(g, device=device, dtype=dtype).detach()
        text = torch.as_tensor(l, device=device, dtype=dtype).detach()
        if graph.ndim != 2 or text.ndim != 2:
            raise ValueError("GNN and LLM features must be rank-two tensors")
        if graph.size(0) != text.size(0):
            raise ValueError("GNN and LLM feature batches must match")
        if graph.size(1) != self.gnn_dim or text.size(1) != self.llm_dim:
            raise ValueError("feature widths do not match model dimensions")
        return graph, text

    def _repair(
        self,
        head: _ResidualHead,
        graph: torch.Tensor,
        text: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        base = torch.as_tensor(base_logits, device=graph.device).detach()
        if base.ndim != 2 or base.shape != (graph.size(0), self.num_classes):
            raise ValueError("base logits must match the feature batch and class count")
        return base + head(graph, text).to(dtype=base.dtype)

    def repair_g(
        self, g: torch.Tensor, l: torch.Tensor, base_g: torch.Tensor
    ) -> torch.Tensor:
        graph, text = self._features(g, l)
        return self._repair(self.gnn_head, graph, text, base_g)

    def repair_l(
        self, g: torch.Tensor, l: torch.Tensor, base_l: torch.Tensor
    ) -> torch.Tensor:
        graph, text = self._features(g, l)
        return self._repair(self.llm_head, graph, text, base_l)

    def forward(
        self,
        g: torch.Tensor,
        l: torch.Tensor,
        base_g: torch.Tensor,
        base_l: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        graph, text = self._features(g, l)
        return (
            self._repair(self.gnn_head, graph, text, base_g),
            self._repair(self.llm_head, graph, text, base_l),
        )

def pool_loss(logits: torch.Tensor, targets: torch.Tensor, kind: str) -> torch.Tensor:
    """Return an independently averaged CE or frozen-base retention loss."""
    value = torch.as_tensor(logits)
    if value.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if kind not in {"ce", "keep"}:
        raise ValueError(f"unknown pool loss kind: {kind}")
    if value.size(0) == 0:
        return value.sum() * 0.0
    target = torch.as_tensor(targets, device=value.device).detach()
    if kind == "ce":
        if target.numel() != value.size(0):
            raise ValueError("CE targets must contain one class per row")
        return F.cross_entropy(value.float(), target.view(-1).long())
    if target.shape != value.shape:
        raise ValueError("retention targets must match repaired logits")
    teacher = target.float()
    if not bool(torch.isfinite(teacher).all()) or bool((teacher < 0).any()):
        raise ValueError("retention targets must be finite non-negative probabilities")
    if not bool(torch.allclose(
        teacher.sum(dim=-1), torch.ones(teacher.size(0), device=teacher.device),
        rtol=1e-4, atol=1e-5,
    )):
        raise ValueError("retention targets must be normalized probabilities")
    return F.kl_div(F.log_softmax(value.float(), dim=-1), teacher, reduction="batchmean")
