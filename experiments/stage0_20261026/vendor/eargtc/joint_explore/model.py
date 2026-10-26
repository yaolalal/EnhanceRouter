"""Neural joint classifiers over frozen expert features and probabilities."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


ARCHITECTURES = (
    "balanced_residual",
    "balanced_direct",
    "gated_experts",
    "attention_direct",
)


class _Projection(nn.Sequential):
    def __init__(self, input_dim: int, width: int):
        super().__init__(nn.LayerNorm(input_dim), nn.Linear(input_dim, width), nn.GELU(), nn.LayerNorm(width))


class JointClassifier(nn.Module):
    def __init__(self, gnn_dim, llm_dim, num_classes, architecture, width, dropout):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"unknown architecture: {architecture}")
        if width < 1 or num_classes < 2 or not 0 <= dropout < 1:
            raise ValueError("invalid model dimensions or dropout")
        if architecture == "attention_direct" and width % 4:
            raise ValueError("attention width must be divisible by four")
        self.architecture = architecture
        self.gnn_dim, self.llm_dim = int(gnn_dim), int(llm_dim)
        self.num_classes, self.width, self.dropout = int(num_classes), int(width), float(dropout)
        self.g_projection = _Projection(gnn_dim, width)
        self.l_projection = _Projection(llm_dim, width)
        if architecture == "attention_direct":
            self.prior_projection = _Projection(3 * num_classes, width)
            self.cls = nn.Parameter(torch.zeros(1, 1, width))
            layer = nn.TransformerEncoderLayer(width, 4, 2 * width, dropout, batch_first=True, norm_first=False)
            self.encoder = nn.TransformerEncoder(layer, 2)
            self.output = nn.Linear(width, num_classes)
        else:
            fused_dim = 4 * width + 3 * num_classes
            self.fusion = nn.Sequential(nn.Linear(fused_dim, width), nn.GELU(), nn.Dropout(dropout),
                                        nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout))
            self.output = nn.Linear(width, num_classes)
            if architecture == "balanced_residual":
                nn.init.zeros_(self.output.weight)
                nn.init.zeros_(self.output.bias)
            if architecture == "gated_experts":
                self.gate = nn.Linear(width, 3)
                nn.init.zeros_(self.gate.weight)
                with torch.no_grad():
                    self.gate.bias.copy_(torch.tensor([0.0, 0.0, -2.0]))

    def _views(self, g, l):
        return self.g_projection(g.detach().float()), self.l_projection(l.detach().float())

    def _fused(self, ug, ul, pg, pl, pb):
        return torch.cat((ug, ul, ug * ul, (ug - ul).abs(), pg, pl, pb), dim=1)

    def forward(self, g, l, pg, pl, pb):
        pg, pl, pb = pg.detach().float(), pl.detach().float(), pb.detach().float()
        ug, ul = self._views(g, l)
        if self.architecture == "attention_direct":
            prior = self.prior_projection(torch.cat((pg, pl, pb), dim=1))
            cls = self.cls.expand(len(g), -1, -1)
            return self.output(self.encoder(torch.cat((cls, ug[:, None], ul[:, None], prior[:, None]), dim=1))[:, 0])
        hidden = self.fusion(self._fused(ug, ul, pg, pl, pb))
        direct = self.output(hidden)
        if self.architecture == "balanced_residual":
            return pb.clamp_min(1e-12).log() + direct
        if self.architecture == "gated_experts":
            weights = self.gate(hidden).softmax(1)
            probability = weights[:, :1] * pg + weights[:, 1:2] * pl + weights[:, 2:] * direct.softmax(1)
            return probability.clamp_min(1e-12).log()
        return direct

    def alignment_loss(self, g, l):
        ug, ul = self._views(g, l)
        if len(ug) == 0:
            return sum((parameter.sum() * 0 for parameter in self.parameters()))
        ug, ul = F.normalize(ug, dim=1), F.normalize(ul, dim=1)
        logits = ug @ ul.T / .2
        target = torch.arange(len(ug), device=logits.device)
        return .5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
