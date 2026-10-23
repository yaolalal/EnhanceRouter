"""Independent residual head used by Joint-v6 Repair."""
from __future__ import annotations

import math
import os

import torch
from torch import nn


def _require_slurm() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Joint-v6 Repair numerical operations require a real Slurm job")


class _Projection(nn.Sequential):
    def __init__(self, input_dim: int, width: int):
        super().__init__(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )


class Head(nn.Module):
    """Predict a raw residual from detached graph and text representations."""

    def __init__(self, gnn_dim, llm_dim, classes, width, dropout):
        super().__init__()
        _require_slurm()
        dimensions = (gnn_dim, llm_dim, classes, width)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in dimensions):
            raise ValueError("head dimensions must be integers")
        if gnn_dim < 1 or llm_dim < 1 or classes < 2 or width < 1:
            raise ValueError("invalid head dimensions")
        if (not isinstance(dropout, (int, float)) or isinstance(dropout, bool)
                or not math.isfinite(float(dropout)) or not 0 <= float(dropout) < 1):
            raise ValueError("invalid dropout")

        self.gnn_dim = gnn_dim
        self.llm_dim = llm_dim
        self.classes = classes
        self.width = width
        self.dropout = float(dropout)
        self.graph_projection = _Projection(gnn_dim, width)
        self.text_projection = _Projection(llm_dim, width)
        self.fusion = nn.Sequential(
            nn.Linear(4 * width, width),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(width, classes),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, graph, text):
        _require_slurm()
        if (not torch.is_tensor(graph) or not torch.is_tensor(text)
                or graph.ndim != 2 or text.ndim != 2):
            raise ValueError("head inputs must be rank-two tensors")
        if (graph.shape[0] != text.shape[0] or graph.shape[1] != self.gnn_dim
                or text.shape[1] != self.llm_dim):
            raise ValueError("head input shapes do not match the constructor")
        if graph.device != text.device:
            raise ValueError("head inputs must share a device")
        if not graph.is_floating_point() or not text.is_floating_point():
            raise ValueError("head inputs must be floating tensors")
        if not bool(torch.isfinite(graph).all()) or not bool(torch.isfinite(text).all()):
            raise ValueError("nonfinite head input")

        graph_hidden = self.graph_projection(graph.detach().float())
        text_hidden = self.text_projection(text.detach().float())
        fused = torch.cat(
            (
                graph_hidden,
                text_hidden,
                graph_hidden * text_hidden,
                (graph_hidden - text_hidden).abs(),
            ),
            dim=1,
        )
        return self.fusion(fused)
