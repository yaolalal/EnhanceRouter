from __future__ import annotations

from typing import Sequence

import torch

from .model import DirectNumberedJointClassifier as _DirectNumberedJointClassifier
from .model import DirectOutput


class DirectNumberedJointClassifier(_DirectNumberedJointClassifier):
    """Exact direct-numbered model with layout-preserving CPU saved tensors."""

    def forward(
        self,
        token_ids: torch.Tensor,
        texts: Sequence[str],
        gnn_embeddings: torch.Tensor,
    ) -> DirectOutput:
        if self.training and torch.cuda.is_available():
            with torch.autograd.graph.save_on_cpu(pin_memory=False):
                return super().forward(token_ids, texts, gnn_embeddings)
        return super().forward(token_ids, texts, gnn_embeddings)
