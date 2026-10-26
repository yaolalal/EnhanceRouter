from __future__ import annotations

"""Legacy hashed-text baseline utilities.

The current EAR-GTC pipeline does not import this module. Raw node text is fed
directly to the frozen LLM joint scorer; this file is kept only for comparing
early ablation prototypes.
"""

import hashlib
import re
from typing import Iterable, List

import torch
import torch.nn as nn


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _hash_token(token: str, dim: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % dim


def hash_texts(texts: Iterable[str], dim: int, device: torch.device) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    for text in texts:
        vec = torch.zeros(dim, dtype=torch.float32)
        tokens = TOKEN_RE.findall((text or "").lower())
        if not tokens:
            rows.append(vec)
            continue
        for tok in tokens:
            vec[_hash_token(tok, dim)] += 1.0
        vec = vec / vec.norm(p=2).clamp_min(1.0)
        rows.append(vec)
    return torch.stack(rows, dim=0).to(device)


class OutputTokenClassifier(nn.Module):
    """Legacy hashed-text classifier, not used by the current pipeline."""

    def __init__(self, text_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        return self.net(text_features)
