from __future__ import annotations

from typing import Sequence

import torch

from .layout_preserving_offload import pinned_layout_preserving_saved_tensors
from .model import DirectNumberedJointClassifier as _DirectNumberedJointClassifier
from .model import DirectOutput


class DirectNumberedJointClassifier(_DirectNumberedJointClassifier):
    """Exact direct-numbered model with stride-preserving pinned CPU tensors."""

    def forward(
        self,
        token_ids: torch.Tensor,
        texts: Sequence[str],
        gnn_embeddings: torch.Tensor,
    ) -> DirectOutput:
        if self.training and torch.cuda.is_available():
            with pinned_layout_preserving_saved_tensors():
                return super().forward(token_ids, texts, gnn_embeddings)
        return super().forward(token_ids, texts, gnn_embeddings)
