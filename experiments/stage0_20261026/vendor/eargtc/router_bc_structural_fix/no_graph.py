from __future__ import annotations

from collections.abc import Mapping

import torch

from eargtc.router_bc.features import invariant_evidence

REMOVED_FEATURES = (
    "neighbor_gnn_agreement",
    "neighbor_gnn_entropy",
)


def no_graph_invariant_evidence(
    gnn_prob: torch.Tensor,
    llm_prob: torch.Tensor,
    structural: Mapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, list[str]]:
    if structural is None or "log_degree" not in structural:
        raise ValueError("no-graph evidence requires log_degree")
    base, names = invariant_evidence(gnn_prob, llm_prob, structural=None)
    degree = torch.as_tensor(
        structural["log_degree"], dtype=base.dtype
    ).view(-1, 1)
    if degree.size(0) != base.size(0):
        raise ValueError("log_degree row count does not match expert probabilities")
    output_names = names + ["log_degree"]
    if any(name in output_names for name in REMOVED_FEATURES):
        raise AssertionError("no-graph adapter retained a removed neighbor feature")
    return torch.cat([base, degree], dim=1), output_names
