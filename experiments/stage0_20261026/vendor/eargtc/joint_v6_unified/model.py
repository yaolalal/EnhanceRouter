"""Balanced classifiers over frozen graph, text, and expert inputs."""
from __future__ import annotations

import math
import os

import torch
from torch import nn


FAMILIES = ("direct", "residual", "attention")


def _require_slurm() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Joint-v6 numerical operations require a real Slurm job")


class _Projection(nn.Sequential):
    def __init__(self, input_dim: int, width: int):
        super().__init__(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )


class Model(nn.Module):
    """Equal-width direct, residual, and attention model families."""

    def __init__(self, gnn_dim, llm_dim, num_classes, family, width, dropout):
        super().__init__()
        _require_slurm()
        dimensions = (gnn_dim, llm_dim, num_classes, width)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in dimensions):
            raise ValueError("model dimensions must be integers")
        if gnn_dim < 1 or llm_dim < 1 or num_classes < 2 or width < 1:
            raise ValueError("invalid model dimensions")
        if family not in FAMILIES:
            raise ValueError(f"unknown model family: {family}")
        if not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("invalid dropout")

        self.gnn_dim = gnn_dim
        self.llm_dim = llm_dim
        self.num_classes = num_classes
        self.family = family
        self.width = width
        self.dropout = float(dropout)
        self.graph_projection = _Projection(gnn_dim, width)
        self.text_projection = _Projection(llm_dim, width)

        if family == "attention":
            heads = 4 if width % 4 == 0 else 2 if width % 2 == 0 else 1
            self.cls = nn.Parameter(torch.zeros(1, 1, width))
            self.modality = nn.Parameter(torch.zeros(1, 3, width))
            layer = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=2 * width,
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.classifier = nn.Linear(width, num_classes)
        else:
            self.fusion = nn.Sequential(
                nn.Linear(4 * width, width),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(width, width),
                nn.GELU(),
                nn.Dropout(self.dropout),
            )
            if family == "direct":
                self.classifier = nn.Linear(width, num_classes)
            else:
                self.graph_residual = nn.Linear(width, num_classes)
                self.text_residual = nn.Linear(width, num_classes)
                self.gate = nn.Linear(width, 2)
                for layer in (self.graph_residual, self.text_residual, self.gate):
                    nn.init.zeros_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def _check_inputs(self, g, l, logp_g, logp_l) -> None:
        values = (g, l, logp_g, logp_l)
        if any(not torch.is_tensor(value) or value.ndim != 2 for value in values):
            raise ValueError("model inputs must be rank-two tensors")
        rows = g.shape[0]
        if (
            l.shape[0] != rows
            or logp_g.shape[0] != rows
            or logp_l.shape[0] != rows
            or g.shape[1] != self.gnn_dim
            or l.shape[1] != self.llm_dim
            or logp_g.shape[1] != self.num_classes
            or logp_l.shape[1] != self.num_classes
        ):
            raise ValueError("model input shapes do not match the constructor")
        if any(not bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("nonfinite model input")
        for value in (logp_g, logp_l):
            normalizer = value.detach().float().logsumexp(1)
            if not torch.allclose(normalizer, torch.zeros_like(normalizer), atol=1e-4, rtol=1e-4):
                raise ValueError("expert inputs must be normalized log probabilities")

    @staticmethod
    def _fuse(graph, text):
        return torch.cat((graph, text, graph * text, (graph - text).abs()), dim=1)

    def forward(self, g, l, logp_g, logp_l):
        _require_slurm()
        self._check_inputs(g, l, logp_g, logp_l)
        g = g.detach().float()
        l = l.detach().float()
        logp_g = logp_g.detach().float()
        logp_l = logp_l.detach().float()
        graph = self.graph_projection(g)
        text = self.text_projection(l)

        if self.family == "attention":
            cls = self.cls.expand(g.shape[0], -1, -1)
            tokens = torch.cat((cls, graph[:, None], text[:, None]), dim=1) + self.modality
            logits = self.classifier(self.encoder(tokens)[:, 0])
            return {"log_probs": logits.log_softmax(1)}

        hidden = self.fusion(self._fuse(graph, text))
        if self.family == "direct":
            return {"log_probs": self.classifier(hidden).log_softmax(1)}

        graph_branch = (logp_g + self.graph_residual(hidden)).log_softmax(1)
        text_branch = (logp_l + self.text_residual(hidden)).log_softmax(1)
        branch_log_probs = torch.stack((graph_branch, text_branch), dim=1)
        log_gate = self.gate(hidden).log_softmax(1)
        log_probs = torch.logsumexp(log_gate[:, :, None] + branch_log_probs, dim=1)
        return {"log_probs": log_probs, "branch_log_probs": branch_log_probs}
