from __future__ import annotations

import torch


def advantage_direction_record(
    base_advantage: torch.Tensor,
    shifted_advantage: torch.Tensor,
    base_score: torch.Tensor,
    shifted_score: torch.Tensor,
    *,
    epsilon: float = 0.1,
) -> dict[str, int]:
    """Count material same-node score changes with the correct advantage direction."""
    base_a = torch.as_tensor(base_advantage, dtype=torch.float32).view(-1)
    shifted_a = torch.as_tensor(shifted_advantage, dtype=torch.float32).view(-1)
    base_s = torch.as_tensor(base_score, dtype=torch.float32).view(-1)
    shifted_s = torch.as_tensor(shifted_score, dtype=torch.float32).view(-1)
    if len({base_a.numel(), shifted_a.numel(), base_s.numel(), shifted_s.numel()}) != 1:
        raise ValueError("advantage direction inputs must have equal length")
    delta_a = shifted_a - base_a
    material = delta_a.abs() >= float(epsilon)
    correct = torch.sign(shifted_s[material] - base_s[material]) == torch.sign(delta_a[material])
    return {"direction_correct": int(correct.sum()), "direction_total": int(material.sum())}
