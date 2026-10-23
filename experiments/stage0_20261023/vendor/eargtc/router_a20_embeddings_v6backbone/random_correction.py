from __future__ import annotations

import hashlib

import torch

from eargtc.router_bc.data_impl import SupportDomain, TargetArtifacts
from eargtc.router_v7_decision.feature_bank import DatasetFeatureBank

from .data import combined_support, combined_target


def _tensor_sha256(value: torch.Tensor) -> str:
    array = torch.as_tensor(value).detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def canonical_column_signs(weight: torch.Tensor) -> torch.Tensor:
    matrix = torch.as_tensor(weight, dtype=torch.float32)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError("projection weight must be a nonempty matrix")
    pivots = matrix.abs().argmax(dim=0)
    columns = torch.arange(matrix.shape[1])
    values = matrix[pivots, columns]
    return torch.where(values < 0, -torch.ones_like(values), torch.ones_like(values))


def canonical_random_embeddings(
    banks: dict[str, DatasetFeatureBank],
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict]:
    """Restore one shared random basis after dataset-local LLM sign correction."""
    if not banks:
        raise ValueError("random correction requires feature banks")
    aligned = {}
    canonical_g_weights = {}
    canonical_l_weights = {}
    flips = {}
    for name, bank in banks.items():
        projection = bank.projections["random"]
        zg, zl = bank.projected["random"]
        g_weight = torch.as_tensor(projection.g_weight, dtype=torch.float32)
        l_weight = torch.as_tensor(projection.l_weight, dtype=torch.float32)
        l_sign = canonical_column_signs(l_weight)
        canonical_g_weights[name] = g_weight
        canonical_l_weights[name] = l_weight * l_sign.view(1, -1)
        aligned[name] = (
            torch.as_tensor(zg, dtype=torch.float32).contiguous(),
            (torch.as_tensor(zl, dtype=torch.float32) * l_sign.view(1, -1)).contiguous(),
        )
        flips[name] = int((l_sign < 0).sum())
    first = next(iter(banks))
    for name in banks:
        if not torch.equal(canonical_g_weights[first], canonical_g_weights[name]):
            raise ValueError("random GNN bases are not shared after canonicalization")
        if not torch.equal(canonical_l_weights[first], canonical_l_weights[name]):
            raise ValueError("random LLM bases are not shared after canonicalization")
    return aligned, {
        "family": "random",
        "alignment": "independent_shared_basis_canonical_column_signs",
        "dimension": int(canonical_l_weights[first].shape[1]),
        "gnn_basis_sha256": _tensor_sha256(canonical_g_weights[first]),
        "llm_basis_sha256": _tensor_sha256(canonical_l_weights[first]),
        "llm_sign_flips_to_canonical": flips,
    }


def build_corrected_random_domains(
    a20_supports: dict[str, SupportDomain],
    a20_targets: dict[str, TargetArtifacts],
    banks: dict[str, DatasetFeatureBank],
    *,
    target_name: str,
    dim: int,
) -> tuple[dict[str, SupportDomain], TargetArtifacts, dict]:
    expected = tuple(a20_supports)
    if tuple(a20_targets) != expected or tuple(banks) != expected:
        raise ValueError("A20 domains and banks must share dataset order")
    aligned, metadata = canonical_random_embeddings(banks)
    supports = {}
    for name in expected:
        zg, zl = aligned[name]
        supports[name] = combined_support(
            a20_supports[name], banks[name].support_node_ids, zg, zl, dim=dim
        )
    zg, zl = aligned[target_name]
    target = combined_target(a20_targets[target_name], zg, zl, dim=dim)
    metadata.update(
        {
            "target_name": target_name,
            "source_names": [name for name in expected if name != target_name],
            "aligned_basis_dimension": 64,
            "output_dimension": int(dim),
            "truncation": "canonicalize_64_then_prefix",
        }
    )
    return supports, target, metadata


__all__ = [
    "build_corrected_random_domains",
    "canonical_column_signs",
    "canonical_random_embeddings",
]
