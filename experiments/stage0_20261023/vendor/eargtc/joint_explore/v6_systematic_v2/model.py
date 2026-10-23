"""Switchable, independent two-branch residual heads."""

from __future__ import annotations

import os

import torch
import torch.nn as nn


ARCHITECTURES = (
    "raw_concat",
    "projected_concat",
    "projected_interaction",
    "projected_sum",
    "projected_gate",
    "raw_depth",
    "projected_concat_wide",
)


def _slurm_only() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Joint-v6 systematic numerical work must run under Slurm")


class Head(nn.Module):
    """One head; output is a zero-initialized class-logit delta."""

    def __init__(
        self,
        graph_dim: int,
        text_dim: int,
        num_classes: int,
        architecture: str,
        width: int,
        dropout: float,
    ) -> None:
        _slurm_only()
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"unknown architecture: {architecture}")
        if min(graph_dim, text_dim, num_classes, width) <= 0:
            raise ValueError("model dimensions must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.graph_dim = int(graph_dim)
        self.text_dim = int(text_dim)
        self.num_classes = int(num_classes)
        self.architecture = architecture
        self.width = int(width)
        self.dropout = nn.Dropout(float(dropout))

        if architecture in {"raw_concat", "raw_depth"}:
            self.graph_norm = nn.LayerNorm(self.graph_dim)
            self.text_norm = nn.LayerNorm(self.text_dim)
            self.hidden = nn.Linear(self.graph_dim + self.text_dim, self.width)
            self.activation = nn.GELU()
            self.depth = (
                nn.Sequential(nn.Linear(self.width, self.width), nn.GELU())
                if architecture == "raw_depth"
                else nn.Identity()
            )
            output_width = self.width
        else:
            self.graph_projection = nn.Sequential(
                nn.LayerNorm(self.graph_dim),
                nn.Linear(self.graph_dim, self.width),
                nn.GELU(),
                nn.LayerNorm(self.width),
            )
            self.text_projection = nn.Sequential(
                nn.LayerNorm(self.text_dim),
                nn.Linear(self.text_dim, self.width),
                nn.GELU(),
                nn.LayerNorm(self.width),
            )
            if architecture == "projected_interaction":
                input_width, output_width = 4 * self.width, self.width
            elif architecture == "projected_concat_wide":
                input_width, output_width = 2 * self.width, 2 * self.width
            elif architecture == "projected_concat":
                input_width, output_width = 2 * self.width, self.width
            else:
                input_width, output_width = self.width, self.width
            if architecture == "projected_gate":
                self.gate = nn.Linear(2 * self.width, self.width)
            self.hidden = nn.Linear(input_width, output_width)
            self.activation = nn.GELU()
            self.depth = nn.Identity()
        self.output = nn.Linear(output_width, self.num_classes)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _inputs(self, graph: torch.Tensor, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.output.weight.device
        dtype = self.output.weight.dtype
        graph = torch.as_tensor(graph, device=device, dtype=dtype).detach()
        text = torch.as_tensor(text, device=device, dtype=dtype).detach()
        if graph.ndim != 2 or text.ndim != 2 or graph.shape[0] != text.shape[0]:
            raise ValueError("feature batches must be aligned rank-two tensors")
        if graph.shape[1] != self.graph_dim or text.shape[1] != self.text_dim:
            raise ValueError("feature widths differ from the model")
        return graph, text

    def forward(self, graph: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        graph, text = self._inputs(graph, text)
        if self.architecture in {"raw_concat", "raw_depth"}:
            fused = torch.cat((self.graph_norm(graph), self.text_norm(text)), dim=1)
        else:
            g = self.graph_projection(graph)
            l = self.text_projection(text)
            if self.architecture in {"projected_concat", "projected_concat_wide"}:
                fused = torch.cat((g, l), dim=1)
            elif self.architecture == "projected_interaction":
                fused = torch.cat((g, l, g * l, (g - l).abs()), dim=1)
            elif self.architecture == "projected_sum":
                fused = g + l
            else:
                gate = torch.sigmoid(self.gate(torch.cat((g, l), dim=1)))
                fused = gate * g + (1.0 - gate) * l
        value = self.activation(self.hidden(fused))
        value = self.depth(value)
        return self.output(self.dropout(value))


class DualResidualModel(nn.Module):
    """Two parameter-independent heads around frozen branch logits."""

    def __init__(
        self,
        graph_dim: int,
        text_dim: int,
        num_classes: int,
        architecture: str,
        width: int,
        dropout: float,
    ) -> None:
        _slurm_only()
        super().__init__()
        self.g = Head(graph_dim, text_dim, num_classes, architecture, width, dropout)
        self.l = Head(graph_dim, text_dim, num_classes, architecture, width, dropout)
        self.num_classes = int(num_classes)

    def delta(self, graph: torch.Tensor, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.g(graph, text), self.l(graph, text)

    def logits(
        self,
        branch: str,
        graph: torch.Tensor,
        text: torch.Tensor,
        base_logits: torch.Tensor,
        *,
        residual: bool,
    ) -> torch.Tensor:
        if branch not in {"g", "l"}:
            raise ValueError("branch must be g or l")
        delta = getattr(self, branch)(graph, text)
        base = torch.as_tensor(base_logits, device=delta.device, dtype=delta.dtype).detach()
        if base.shape != delta.shape:
            raise ValueError("base logits must match the head output")
        return base + delta if residual else delta

