from __future__ import annotations

from pathlib import Path

import torch

from eargtc.router_bc.data_impl import SupportDomain, TargetArtifacts
from eargtc.router_bc.rank import percentile_features
from eargtc.router_v7_decision.data import (
    BaseDatasetBundle,
    load_base_dataset_bundle,
)
from eargtc.router_v7_decision.features import LOGIT_FEATURE_NAMES


def validate_a20_feature_names(names: list[str] | tuple[str, ...]) -> None:
    if tuple(names) != LOGIT_FEATURE_NAMES:
        raise ValueError("A20 evidence must contain exactly the approved 20 columns")
    forbidden = {"agreement", "g_confidence", "l_confidence"}
    if forbidden.intersection(names) or any(name.startswith("ctx_") for name in names):
        raise ValueError("A20 evidence contains a forbidden feature")


def a20_support_from_bundle(bundle: BaseDatasetBundle) -> SupportDomain:
    validate_a20_feature_names(list(LOGIT_FEATURE_NAMES))
    x = torch.as_tensor(bundle.support_logit_evidence, dtype=torch.float32).cpu()
    if tuple(x.shape) != (int(bundle.support_labels.numel()), 20):
        raise ValueError("A20 support evidence does not align or has wrong width")
    if not bool(torch.isfinite(x).all().item()):
        raise ValueError("A20 support evidence contains non-finite values")
    labels = torch.as_tensor(bundle.support_labels, dtype=torch.long).view(-1)
    gnn_pred = torch.as_tensor(bundle.support_gnn_pred, dtype=torch.long).view(-1)
    llm_pred = torch.as_tensor(bundle.support_llm_pred, dtype=torch.long).view(-1)
    node_ids = torch.as_tensor(bundle.support_node_ids, dtype=torch.long).view(-1)
    if not (labels.numel() == gnn_pred.numel() == llm_pred.numel() == node_ids.numel()):
        raise ValueError("A20 support labels, predictions, and IDs do not align")
    return SupportDomain(
        name=bundle.dataset,
        rows=[{"node_id": int(node_id)} for node_id in node_ids.tolist()],
        x=x.contiguous(),
        feature_names=list(LOGIT_FEATURE_NAMES),
        gnn_correct=(gnn_pred == labels),
        llm_correct=(llm_pred == labels),
        disagreement=(gnn_pred != llm_pred),
    )


def a20_target_from_bundle(bundle: BaseDatasetBundle) -> TargetArtifacts:
    validate_a20_feature_names(list(LOGIT_FEATURE_NAMES))
    x = torch.as_tensor(bundle.logit_evidence, dtype=torch.float32).cpu()
    if tuple(x.shape) != (int(bundle.labels.numel()), 20):
        raise ValueError("A20 query evidence does not align or has wrong width")
    if not bool(torch.isfinite(x).all().item()):
        raise ValueError("A20 query evidence contains non-finite values")
    return TargetArtifacts(
        name=bundle.dataset,
        labels=torch.as_tensor(bundle.labels, dtype=torch.long).view(-1).cpu(),
        gnn_prob=torch.as_tensor(bundle.gnn_prob).cpu(),
        llm_prob=torch.as_tensor(bundle.llm_prob).cpu(),
        gnn_pred=torch.as_tensor(bundle.gnn_pred, dtype=torch.long).view(-1).cpu(),
        llm_pred=torch.as_tensor(bundle.llm_pred, dtype=torch.long).view(-1).cpu(),
        train_ids=torch.as_tensor(bundle.train_ids, dtype=torch.long).view(-1).cpu(),
        val_ids=torch.as_tensor(bundle.val_ids, dtype=torch.long).view(-1).cpu(),
        test_ids=torch.as_tensor(bundle.test_ids, dtype=torch.long).view(-1).cpu(),
        x=x.contiguous(),
        feature_names=list(LOGIT_FEATURE_NAMES),
    )


def rank_a20_support(domain: SupportDomain) -> SupportDomain:
    validate_a20_feature_names(domain.feature_names)
    return SupportDomain(
        name=domain.name,
        rows=domain.rows,
        x=percentile_features(domain.x),
        feature_names=list(LOGIT_FEATURE_NAMES),
        gnn_correct=domain.gnn_correct,
        llm_correct=domain.llm_correct,
        disagreement=domain.disagreement,
    )


def rank_a20_query(x: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(x)
    if value.ndim != 2 or value.shape[1] != 20:
        raise ValueError("A20 query rank view requires X[N,20]")
    return percentile_features(value)


def load_a20_domain(
    dataset: str,
    *,
    upstream_root: str | Path,
    dataset_root: str | Path,
    gnn_embedding_root: str | Path,
) -> tuple[SupportDomain, TargetArtifacts]:
    bundle = load_base_dataset_bundle(
        dataset,
        upstream_root=upstream_root,
        dataset_root=dataset_root,
        gnn_embedding_root=gnn_embedding_root,
    )
    return a20_support_from_bundle(bundle), a20_target_from_bundle(bundle)


__all__ = [
    "a20_support_from_bundle",
    "a20_target_from_bundle",
    "load_a20_domain",
    "rank_a20_query",
    "rank_a20_support",
    "validate_a20_feature_names",
]
