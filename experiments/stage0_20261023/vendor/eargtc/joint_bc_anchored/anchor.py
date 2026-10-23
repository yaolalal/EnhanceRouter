from __future__ import annotations

import torch


def expert_log_prob(logits: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(logits)
    if value.ndim != 2:
        raise ValueError("expert logits must have shape [N,C]")
    return torch.log_softmax(value.float(), dim=-1)


def build_scheme_b_anchor(
    gnn_logits: torch.Tensor,
    llm_logits: torch.Tensor,
    agreement: torch.Tensor,
    choose_gnn: torch.Tensor,
) -> torch.Tensor:
    gnn = expert_log_prob(gnn_logits)
    llm = expert_log_prob(llm_logits)
    if gnn.shape != llm.shape:
        raise ValueError("expert logits must have identical shapes")
    same = torch.as_tensor(agreement, dtype=torch.bool, device=gnn.device).view(-1)
    route_g = torch.as_tensor(choose_gnn, dtype=torch.bool, device=gnn.device).view(-1)
    if same.numel() != gnn.size(0) or route_g.numel() != gnn.size(0):
        raise ValueError("route vectors must have one value per node")
    routed = torch.where(route_g[:, None], gnn, llm)
    return torch.where(same[:, None], 0.5 * (gnn + llm), routed)
