from __future__ import annotations

import torch


def percentile_features(x: torch.Tensor) -> torch.Tensor:
    """Column-wise tie-aware empirical percentiles mapped to [-1, 1]."""
    value = torch.as_tensor(x, dtype=torch.float64)
    if value.ndim != 2:
        raise ValueError("percentile_features expects X[N,D]")
    if value.size(0) <= 1:
        return torch.zeros_like(value, dtype=torch.float32)
    columns = []
    denominator = float(value.size(0) - 1)
    for index in range(value.size(1)):
        column = value[:, index].contiguous()
        ordered = torch.sort(column).values.contiguous()
        left = torch.searchsorted(ordered, column, right=False).to(torch.float64)
        right = torch.searchsorted(ordered, column, right=True).to(torch.float64) - 1.0
        midrank = 0.5 * (left + right)
        columns.append(2.0 * midrank / denominator - 1.0)
    return torch.stack(columns, dim=1).to(torch.float32)
