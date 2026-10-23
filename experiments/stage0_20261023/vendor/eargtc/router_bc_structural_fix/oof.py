from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from eargtc.models.gnn import MLPHead
from eargtc.router import (
    _gnn_class_weights,
    build_router_feature_rows,
)
from eargtc.train import train_classifier


@dataclass(frozen=True)
class FoldOOFResult:
    fold_id: int
    holdout_ids: list[int]
    rows: list[dict]
    full_gnn_pred: torch.Tensor
    holdout_gnn_logits: torch.Tensor


def build_rows_from_fold_full_logits(
    *,
    fold_id: int,
    holdout_ids: list[int],
    data,
    texts: list[str],
    full_gnn_logits: torch.Tensor,
    llm_logits: torch.Tensor,
) -> FoldOOFResult:
    holdout_ids = [int(node_id) for node_id in holdout_ids]
    if full_gnn_logits.ndim != 2:
        raise ValueError("fold GNN logits must have shape [num_nodes, num_classes]")
    if int(full_gnn_logits.size(0)) != int(data.num_nodes):
        raise ValueError("fold GNN logits must cover the full graph")
    if tuple(full_gnn_logits.shape) != tuple(llm_logits.shape):
        raise ValueError("fold GNN and frozen LLM logits must have identical shapes")
    if len(set(holdout_ids)) != len(holdout_ids):
        raise ValueError(f"fold {fold_id} contains duplicate holdout node IDs")

    rows = build_router_feature_rows(
        node_ids=holdout_ids,
        data=data,
        texts=texts,
        gnn_logits=full_gnn_logits.detach(),
        llm_logits=llm_logits.detach(),
    )
    holdout_idx = torch.tensor(
        holdout_ids, dtype=torch.long, device=full_gnn_logits.device
    )
    return FoldOOFResult(
        fold_id=int(fold_id),
        holdout_ids=holdout_ids,
        rows=rows,
        full_gnn_pred=full_gnn_logits.detach().argmax(dim=1).to("cpu", torch.long),
        holdout_gnn_logits=full_gnn_logits.detach()[holdout_idx].float().cpu(),
    )


def _train_fold_logits(
    *,
    data,
    embeddings: torch.Tensor,
    fold_train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    num_classes: int,
    head_config: dict,
    **_,
) -> torch.Tensor:
    device = embeddings.device
    head = MLPHead(
        input_dim=int(embeddings.size(1)),
        hidden_dim=int(head_config["head_hidden_dim"]),
        output_dim=int(num_classes),
        dropout=float(head_config["dropout"]),
    ).to(device)
    class_weights = _gnn_class_weights(
        data.y,
        fold_train_idx,
        int(num_classes),
        str(head_config["class_weight"]),
    )
    logits = train_classifier(
        head,
        embeddings,
        data.y,
        fold_train_idx,
        val_idx,
        epochs=int(head_config["epochs"]),
        lr=float(head_config["lr"]),
        weight_decay=float(head_config["weight_decay"]),
        class_weights=class_weights,
        patience=head_config.get("patience"),
        select_by=str(head_config["select_by"]),
    )
    return logits


def build_fold_full_graph_oof_rows(
    *,
    data,
    texts: list[str],
    embeddings: torch.Tensor,
    llm_logits: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    num_classes: int,
    fold_ids: list[list[int]],
    head_config: dict,
    seed: int,
    fold_logits_builder: Callable[..., torch.Tensor] | None = None,
) -> list[FoldOOFResult]:
    builder = fold_logits_builder or _train_fold_logits
    device = embeddings.device
    train_ids = [int(node_id) for node_id in train_idx.detach().cpu().tolist()]
    train_set = set(train_ids)
    torch.manual_seed(int(seed))
    results = []
    for fold_id, raw_holdout_ids in enumerate(fold_ids):
        holdout_ids = [int(node_id) for node_id in raw_holdout_ids]
        if not set(holdout_ids).issubset(train_set):
            raise ValueError(f"fold {fold_id} contains nodes outside train_idx")
        holdout_set = set(holdout_ids)
        fold_train_idx = torch.tensor(
            [node_id for node_id in train_ids if node_id not in holdout_set],
            dtype=torch.long,
            device=device,
        )
        fold_logits = builder(
            fold_id=int(fold_id),
            data=data,
            texts=texts,
            embeddings=embeddings,
            llm_logits=llm_logits,
            fold_train_idx=fold_train_idx,
            val_idx=val_idx,
            num_classes=int(num_classes),
            head_config=head_config,
            seed=int(seed),
        )
        results.append(
            build_rows_from_fold_full_logits(
                fold_id=fold_id,
                holdout_ids=holdout_ids,
                data=data,
                texts=texts,
                full_gnn_logits=fold_logits,
                llm_logits=llm_logits,
            )
        )
    return results


def combine_fold_oof_rows(
    results: list[FoldOOFResult],
    *,
    train_idx: torch.Tensor,
    labels: torch.Tensor,
) -> list[dict]:
    train_ids = [int(node_id) for node_id in train_idx.detach().cpu().tolist()]
    by_node: dict[int, dict] = {}
    for result in results:
        if len(result.rows) != len(result.holdout_ids):
            raise ValueError(f"fold {result.fold_id} row/holdout count mismatch")
        for row in result.rows:
            node_id = int(row["node_id"])
            if node_id in by_node:
                raise ValueError(f"support node {node_id} appears in multiple folds")
            output = dict(row)
            output["fold"] = int(result.fold_id)
            output["y_true"] = int(labels[node_id].item())
            output["gnn_correct"] = int(
                int(output["gnn_pred"]) == int(output["y_true"])
            )
            output["llm_correct"] = int(
                int(output["llm_pred"]) == int(output["y_true"])
            )
            by_node[node_id] = output
    if set(by_node) != set(train_ids):
        missing = sorted(set(train_ids) - set(by_node))
        extra = sorted(set(by_node) - set(train_ids))
        raise ValueError(f"OOF support coverage mismatch: missing={missing}, extra={extra}")
    return [by_node[node_id] for node_id in train_ids]
