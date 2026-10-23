from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RouterV7Output:
    preference_logit: torch.Tensor
    trust_logit: torch.Tensor


class RouterV7(nn.Module):
    def __init__(
        self,
        *,
        local_dim: int,
        context_dim: int = 4,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if int(local_dim) <= 0 or int(context_dim) < 0 or int(hidden_dim) <= 0:
            raise ValueError("Router dimensions must be positive (context may be zero)")
        self.local_dim = int(local_dim)
        self.context_dim = int(context_dim)
        self.local_encoder = nn.Sequential(
            nn.Linear(self.local_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
        )
        self.local_preference = nn.Linear(int(hidden_dim), 1)
        self.context_preference = (
            nn.Linear(self.context_dim, 1, bias=False) if self.context_dim else None
        )
        self.trust_head = nn.Sequential(
            nn.Linear(int(hidden_dim) + self.context_dim + 1, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self, local_features: torch.Tensor, context_features: torch.Tensor | None = None
    ) -> RouterV7Output:
        local = torch.as_tensor(local_features, dtype=torch.float32)
        if local.ndim != 2 or local.size(1) != self.local_dim:
            raise ValueError(f"local features must have shape [N,{self.local_dim}]")
        if self.context_dim:
            if context_features is None:
                raise ValueError("context features are required")
            context = torch.as_tensor(
                context_features, dtype=torch.float32, device=local.device
            )
            if context.ndim != 2 or context.shape != (local.size(0), self.context_dim):
                raise ValueError(
                    f"context features must have shape [N,{self.context_dim}]"
                )
        else:
            context = local.new_empty((local.size(0), 0))
        hidden = self.local_encoder(local)
        preference = self.local_preference(hidden).squeeze(-1)
        if self.context_preference is not None:
            preference = preference + self.context_preference(context).squeeze(-1)
        route_probability = torch.sigmoid(preference).unsqueeze(1)
        trust_input = torch.cat([hidden, context, route_probability], dim=1)
        trust = self.trust_head(trust_input).squeeze(-1)
        return RouterV7Output(preference, trust)


def route_and_trust(
    output: RouterV7Output, *, threshold: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("route threshold must be in [0, 1]")
    return torch.sigmoid(output.preference_logit) >= threshold, torch.sigmoid(output.trust_logit)
