from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class Codebook:
    centers: torch.Tensor
    token_prefix: str = "STR"

    @property
    def size(self) -> int:
        return int(self.centers.size(0))

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(features.float(), self.centers.float())
        return distances.argmin(dim=1)

    def tokens(self, ids: torch.Tensor) -> List[str]:
        return [f"<{self.token_prefix}_{int(i)}>" for i in ids.detach().cpu().tolist()]

    def state_dict(self):
        return {"centers": self.centers.detach().cpu(), "token_prefix": self.token_prefix}

    @classmethod
    def load(cls, state):
        return cls(centers=state["centers"], token_prefix=state.get("token_prefix", "STR"))


def fit_codebook(features: torch.Tensor, size: int, iters: int = 25, seed: int = 42) -> Codebook:
    if size <= 0:
        raise ValueError("codebook size must be positive")
    n = features.size(0)
    k = min(size, n)
    device = features.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    perm = torch.randperm(n, generator=generator, device=device)[:k]
    centers = features[perm].float().clone()

    for _ in range(max(iters, 1)):
        distances = torch.cdist(features.float(), centers)
        assign = distances.argmin(dim=1)
        new_centers = centers.clone()
        for idx in range(k):
            mask = assign == idx
            if mask.any():
                new_centers[idx] = features[mask].float().mean(dim=0)
        if torch.allclose(new_centers, centers, atol=1e-5):
            centers = new_centers
            break
        centers = new_centers
    return Codebook(centers=centers)
