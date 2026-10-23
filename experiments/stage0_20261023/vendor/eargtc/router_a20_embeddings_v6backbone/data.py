from __future__ import annotations

import torch

from eargtc.router_bc.data_impl import SupportDomain, TargetArtifacts
from eargtc.router_v7_decision.feature_bank import DatasetFeatureBank
from eargtc.router_v7_decision.features import LOGIT_FEATURE_NAMES

from .alignment import alignment_rotations


def embedding_feature_names(dim: int) -> list[str]:
    width = int(dim)
    if width not in {16, 32, 64}:
        raise ValueError("embedding dimension must be one of 16, 32, 64")
    return [f"z_g_{index:02d}" for index in range(width)] + [
        f"z_l_{index:02d}" for index in range(width)
    ]


def _combined_x(
    a20: torch.Tensor,
    node_ids: torch.Tensor,
    z_g: torch.Tensor,
    z_l: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    width = int(dim)
    ids = torch.as_tensor(node_ids, dtype=torch.long).view(-1)
    base = torch.as_tensor(a20, dtype=torch.float32)
    g = torch.as_tensor(z_g, dtype=torch.float32)
    l = torch.as_tensor(z_l, dtype=torch.float32)
    if base.ndim != 2 or base.shape != (ids.numel(), 20):
        raise ValueError("A20 rows must align and have width 20")
    if g.shape != l.shape or g.ndim != 2 or g.shape[1] < width:
        raise ValueError("aligned projected embeddings do not support the requested dimension")
    if bool(((ids < 0) | (ids >= g.shape[0])).any()):
        raise ValueError("embedding node IDs are out of range")
    result = torch.cat([base, g[ids, :width], l[ids, :width]], dim=1).contiguous()
    if result.shape[1] != 20 + 2 * width or not bool(torch.isfinite(result).all()):
        raise ValueError("combined evidence is invalid")
    return result


def combined_support(
    support: SupportDomain,
    support_node_ids: torch.Tensor,
    z_g: torch.Tensor,
    z_l: torch.Tensor,
    *,
    dim: int,
) -> SupportDomain:
    names = list(LOGIT_FEATURE_NAMES) + embedding_feature_names(dim)
    return SupportDomain(
        name=support.name,
        rows=support.rows,
        x=_combined_x(support.x, support_node_ids, z_g, z_l, dim=dim),
        feature_names=names,
        gnn_correct=support.gnn_correct,
        llm_correct=support.llm_correct,
        disagreement=support.disagreement,
    )


def combined_target(
    target: TargetArtifacts,
    z_g: torch.Tensor,
    z_l: torch.Tensor,
    *,
    dim: int,
) -> TargetArtifacts:
    node_ids = torch.arange(target.labels.numel(), dtype=torch.long)
    names = list(LOGIT_FEATURE_NAMES) + embedding_feature_names(dim)
    return TargetArtifacts(
        name=target.name,
        labels=target.labels,
        gnn_prob=target.gnn_prob,
        llm_prob=target.llm_prob,
        gnn_pred=target.gnn_pred,
        llm_pred=target.llm_pred,
        train_ids=target.train_ids,
        val_ids=target.val_ids,
        test_ids=target.test_ids,
        x=_combined_x(target.x, node_ids, z_g, z_l, dim=dim),
        feature_names=names,
    )


def aligned_projected_embeddings(
    banks: dict[str, DatasetFeatureBank],
    *,
    target_name: str,
    family: str,
    dim: int,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict]:
    """Align one precomputed projection using only non-target datasets as reference."""
    if target_name not in banks:
        raise ValueError("target dataset is missing from feature banks")
    source_names = tuple(name for name in banks if name != target_name)
    width = int(dim)
    weights = {}
    for name, bank in banks.items():
        if family not in bank.projected or family not in bank.projections:
            raise ValueError(f"{name} is missing projection family {family}")
        zg, zl = bank.projected[family]
        weight = torch.as_tensor(bank.projections[family].l_weight)
        if min(zg.shape[1], zl.shape[1], weight.shape[1]) < 64:
            raise ValueError(f"{name}/{family} does not support the frozen 64-d basis")
        weights[name] = weight[:, :64]
    rotations, metadata = alignment_rotations(
        weights, source_names=source_names, family=family
    )
    aligned = {}
    for name, bank in banks.items():
        zg, zl = bank.projected[family]
        rotation = rotations[name]
        aligned[name] = (
            (torch.as_tensor(zg)[:, :64] @ rotation)[:, :width].float().contiguous(),
            (torch.as_tensor(zl)[:, :64] @ rotation)[:, :width].float().contiguous(),
        )
    metadata = {
        **metadata,
        "target_name": target_name,
        "aligned_basis_dimension": 64,
        "output_dimension": width,
        "truncation": "align_64_then_prefix",
    }
    return aligned, metadata


def build_combined_domains(
    a20_supports: dict[str, SupportDomain],
    a20_targets: dict[str, TargetArtifacts],
    banks: dict[str, DatasetFeatureBank],
    *,
    target_name: str,
    family: str,
    dim: int,
) -> tuple[dict[str, SupportDomain], TargetArtifacts, dict]:
    expected = tuple(a20_supports)
    if tuple(a20_targets) != expected or tuple(banks) != expected:
        raise ValueError("A20 domains and embedding banks must share dataset order")
    aligned, metadata = aligned_projected_embeddings(
        banks, target_name=target_name, family=family, dim=dim
    )
    supports = {}
    for name in expected:
        zg, zl = aligned[name]
        supports[name] = combined_support(
            a20_supports[name], banks[name].support_node_ids, zg, zl, dim=dim
        )
    target_zg, target_zl = aligned[target_name]
    target = combined_target(
        a20_targets[target_name], target_zg, target_zl, dim=dim
    )
    return supports, target, metadata


__all__ = [
    "aligned_projected_embeddings",
    "build_combined_domains",
    "combined_support",
    "combined_target",
    "embedding_feature_names",
]
