from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from eargtc.router_bc.data_impl import build_support_domain, load_target_artifacts

from .features import LOGIT_FEATURE_NAMES


@dataclass(frozen=True)
class DecisivePartition:
    node_ids: torch.Tensor
    target: torch.Tensor
    gnn_only: torch.Tensor
    llm_only: torch.Tensor

    def __post_init__(self) -> None:
        count = int(self.node_ids.numel())
        if any(int(value.numel()) != count for value in (self.target, self.gnn_only, self.llm_only)):
            raise ValueError("decisive partition tensors must align")
        if bool((self.gnn_only & self.llm_only).any().item()):
            raise ValueError("decisive classes must be disjoint")
        if not torch.equal(self.target.bool(), self.gnn_only):
            raise ValueError("target=1 must mean GNN-only")


def decisive_partition(
    labels: torch.Tensor,
    gnn_pred: torch.Tensor,
    llm_pred: torch.Tensor,
) -> DecisivePartition:
    """Return only rows on which exactly one frozen expert is correct."""

    y = torch.as_tensor(labels, dtype=torch.long).view(-1)
    g = torch.as_tensor(gnn_pred, dtype=torch.long).view(-1)
    l = torch.as_tensor(llm_pred, dtype=torch.long).view(-1)
    if y.numel() != g.numel() or y.numel() != l.numel():
        raise ValueError("labels and expert predictions must align")
    g_correct = g == y
    l_correct = l == y
    decisive = g_correct ^ l_correct
    ids = decisive.nonzero(as_tuple=False).view(-1)
    gnn_only = g_correct[ids]
    llm_only = l_correct[ids]
    return DecisivePartition(
        node_ids=ids,
        target=gnn_only.to(torch.float32),
        gnn_only=gnn_only,
        llm_only=llm_only,
    )


def _validated_ids(ids: torch.Tensor, *, name: str, num_nodes: int) -> torch.Tensor:
    value = torch.as_tensor(ids, dtype=torch.long).view(-1)
    if torch.unique(value).numel() != value.numel():
        raise ValueError(f"{name} contains duplicate node IDs")
    if bool(((value < 0) | (value >= int(num_nodes))).any().item()):
        raise ValueError(f"{name} contains out-of-range node IDs")
    return value


def validate_split_ids(
    *,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    test_ids: torch.Tensor,
    num_nodes: int,
) -> None:
    """Fail closed unless train/validation/test form an exact node partition."""

    train = _validated_ids(train_ids, name="train_ids", num_nodes=num_nodes)
    val = _validated_ids(val_ids, name="val_ids", num_nodes=num_nodes)
    test = _validated_ids(test_ids, name="test_ids", num_nodes=num_nodes)
    owner = torch.zeros(int(num_nodes), dtype=torch.int8)
    for name, ids in (("train", train), ("validation", val), ("test", test)):
        if bool((owner[ids] != 0).any().item()):
            raise ValueError(f"split overlap detected while adding {name} IDs")
        owner[ids] = 1
    if not bool((owner == 1).all().item()):
        missing = int((owner == 0).sum().item())
        raise ValueError(f"split IDs do not cover all nodes; missing={missing}")


@dataclass(frozen=True)
class BaseDatasetBundle:
    dataset: str
    labels: torch.Tensor
    gnn_prob: torch.Tensor
    llm_prob: torch.Tensor
    gnn_pred: torch.Tensor
    llm_pred: torch.Tensor
    logit_evidence: torch.Tensor
    gnn_embeddings: torch.Tensor
    train_ids: torch.Tensor
    val_ids: torch.Tensor
    test_ids: torch.Tensor
    support_node_ids: torch.Tensor
    support_labels: torch.Tensor
    support_gnn_pred: torch.Tensor
    support_llm_pred: torch.Tensor
    support_logit_evidence: torch.Tensor


def _select_features(
    x: torch.Tensor,
    names: list[str],
    *,
    wanted: tuple[str, ...],
    source: str,
) -> torch.Tensor:
    if len(names) != int(x.shape[1]):
        raise ValueError(f"{source} feature names do not align with columns")
    positions = {name: index for index, name in enumerate(names)}
    missing = [name for name in wanted if name not in positions]
    if missing:
        raise ValueError(f"{source} is missing approved features: {missing}")
    selected = x[:, [positions[name] for name in wanted]].float().cpu().contiguous()
    if not bool(torch.isfinite(selected).all().item()):
        raise ValueError(f"{source} approved features contain non-finite values")
    return selected


def load_base_dataset_bundle(
    dataset: str,
    *,
    upstream_root: str | Path,
    dataset_root: str | Path,
    gnn_embedding_root: str | Path,
) -> BaseDatasetBundle:
    """Load and strictly align frozen experts, corrected OOF support, and SSL embeddings."""

    upstream_root = Path(upstream_root)
    pipeline_dir = upstream_root / dataset / "pipeline"
    oof_path = pipeline_dir / "router" / "oof" / "router_train.jsonl"
    support = build_support_domain(dataset, oof_path)
    target = load_target_artifacts(dataset, pipeline_dir, dataset_root, support)

    validate_split_ids(
        train_ids=target.train_ids,
        val_ids=target.val_ids,
        test_ids=target.test_ids,
        num_nodes=target.labels.numel(),
    )
    support_node_ids = torch.tensor(
        [int(row["node_id"]) for row in support.rows], dtype=torch.long
    )
    if not torch.equal(support_node_ids, target.train_ids):
        raise ValueError("corrected OOF rows must exactly match official train IDs in order")

    support_labels = torch.tensor(
        [int(row["y_true"]) for row in support.rows], dtype=torch.long
    )
    support_gnn_pred = torch.tensor(
        [int(row["gnn_pred"]) for row in support.rows], dtype=torch.long
    )
    support_llm_pred = torch.tensor(
        [int(row["llm_pred"]) for row in support.rows], dtype=torch.long
    )
    if not torch.equal(support_labels, target.labels[support_node_ids]):
        raise ValueError("corrected OOF labels do not align with graph labels")

    embedding_path = (
        Path(gnn_embedding_root) / dataset / "encoder_and_embeddings.pt"
    )
    embedding_artifact = torch.load(
        embedding_path, map_location="cpu", weights_only=False
    )
    if "embeddings" not in embedding_artifact:
        raise ValueError(f"missing embeddings tensor in {embedding_path}")
    gnn_embeddings = torch.as_tensor(
        embedding_artifact["embeddings"], dtype=torch.float32
    ).cpu().contiguous()
    expected_shape = (target.labels.numel(), 128)
    if tuple(gnn_embeddings.shape) != expected_shape:
        raise ValueError(
            f"unexpected SSL embedding shape for {dataset}: "
            f"{tuple(gnn_embeddings.shape)} != {expected_shape}"
        )
    if not bool(torch.isfinite(gnn_embeddings).all().item()):
        raise ValueError(f"SSL embeddings contain non-finite values for {dataset}")

    return BaseDatasetBundle(
        dataset=dataset,
        labels=target.labels,
        gnn_prob=target.gnn_prob,
        llm_prob=target.llm_prob,
        gnn_pred=target.gnn_pred,
        llm_pred=target.llm_pred,
        logit_evidence=_select_features(
            target.x,
            target.feature_names,
            wanted=LOGIT_FEATURE_NAMES,
            source=f"{dataset} final evidence",
        ),
        gnn_embeddings=gnn_embeddings,
        train_ids=target.train_ids,
        val_ids=target.val_ids,
        test_ids=target.test_ids,
        support_node_ids=support_node_ids,
        support_labels=support_labels,
        support_gnn_pred=support_gnn_pred,
        support_llm_pred=support_llm_pred,
        support_logit_evidence=_select_features(
            support.x,
            support.feature_names,
            wanted=LOGIT_FEATURE_NAMES,
            source=f"{dataset} corrected OOF evidence",
        ),
    )


__all__ = [
    "BaseDatasetBundle",
    "DecisivePartition",
    "decisive_partition",
    "load_base_dataset_bundle",
    "validate_split_ids",
]
