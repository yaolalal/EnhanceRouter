from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from eargtc.data import _install_pyg_optional_extension_stubs, _to_undirected

from .context import beta_context, leave_one_out_contexts
from .features import invariant_evidence


@dataclass
class SupportDomain:
    name: str
    rows: list[dict]
    x: torch.Tensor
    feature_names: list[str]
    gnn_correct: torch.Tensor
    llm_correct: torch.Tensor
    disagreement: torch.Tensor


@dataclass
class TargetArtifacts:
    name: str
    labels: torch.Tensor
    gnn_prob: torch.Tensor
    llm_prob: torch.Tensor
    gnn_pred: torch.Tensor
    llm_pred: torch.Tensor
    train_ids: torch.Tensor
    val_ids: torch.Tensor
    test_ids: torch.Tensor
    x: torch.Tensor
    feature_names: list[str]


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _row_structural(rows: list[dict]) -> dict[str, torch.Tensor]:
    return {
        name: torch.tensor([float(row["features"][name]) for row in rows], dtype=torch.float32)
        for name in ("log_degree", "neighbor_gnn_agreement", "neighbor_gnn_entropy")
    }


def build_support_domain(name: str, oof_path: str | Path) -> SupportDomain:
    rows = read_jsonl(oof_path)
    g = torch.tensor([row["gnn_prob"] for row in rows], dtype=torch.float32)
    l = torch.tensor([row["llm_prob"] for row in rows], dtype=torch.float32)
    evidence, evidence_names = invariant_evidence(g, l, _row_structural(rows))
    context, context_names = leave_one_out_contexts(rows)
    x = torch.cat([evidence, context], dim=1)
    g_correct = torch.tensor([bool(row["gnn_correct"]) for row in rows], dtype=torch.bool)
    l_correct = torch.tensor([bool(row["llm_correct"]) for row in rows], dtype=torch.bool)
    disagreement = torch.tensor(
        [int(row["gnn_pred"]) != int(row["llm_pred"]) for row in rows], dtype=torch.bool
    )
    return SupportDomain(
        name=str(name),
        rows=rows,
        x=x,
        feature_names=evidence_names + context_names,
        gnn_correct=g_correct,
        llm_correct=l_correct,
        disagreement=disagreement,
    )


def _structural_evidence(
    edge_index: torch.Tensor, gnn_pred: torch.Tensor, num_nodes: int, num_classes: int
) -> dict[str, torch.Tensor]:
    edge_index = _to_undirected(edge_index.cpu())
    src, dst = edge_index[0].long(), edge_index[1].long()
    degree = torch.bincount(src, minlength=num_nodes).float()
    same_count = torch.zeros(num_nodes, dtype=torch.float32)
    same_count.index_add_(0, src, (gnn_pred[src] == gnn_pred[dst]).float())
    neighbor_agreement = torch.where(
        degree > 0, same_count / degree.clamp_min(1.0), torch.zeros_like(degree)
    )
    class_counts = torch.zeros((num_nodes, num_classes), dtype=torch.float32)
    class_counts.index_put_(
        (src, gnn_pred[dst]), torch.ones_like(src, dtype=torch.float32), accumulate=True
    )
    q = class_counts / degree.clamp_min(1.0).unsqueeze(1)
    neighbor_entropy = -(q * q.clamp_min(1e-12).log()).sum(dim=1) / math.log(num_classes)
    neighbor_entropy = torch.where(
        degree > 0, neighbor_entropy, torch.ones_like(neighbor_entropy)
    )
    return {
        "log_degree": torch.log1p(degree),
        "neighbor_gnn_agreement": neighbor_agreement,
        "neighbor_gnn_entropy": neighbor_entropy,
    }


def load_target_artifacts(
    name: str,
    pipeline_dir: str | Path,
    dataset_root: str | Path,
    support: SupportDomain,
) -> TargetArtifacts:
    pipeline_dir = Path(pipeline_dir)
    prediction = torch.load(
        pipeline_dir / "predictions.pt", map_location="cpu", weights_only=False
    )
    gnn_prob = torch.softmax(prediction["gnn_logits"].float(), dim=1)
    llm_prob = torch.softmax(prediction["llm_logits"].float(), dim=1)
    if gnn_prob.shape != llm_prob.shape:
        raise ValueError(f"Final expert shapes differ for {name}")
    _install_pyg_optional_extension_stubs()
    graph = torch.load(
        Path(dataset_root) / "datasets" / f"{name}.pt",
        map_location="cpu",
        weights_only=False,
    )
    labels = graph.y.view(-1).long().cpu()
    if labels.numel() != gnn_prob.size(0):
        raise ValueError(f"Label/logit node count mismatch for {name}")
    split = read_json(pipeline_dir / "splits" / f"{name}_5shot_seed42.json")
    train_ids = torch.tensor(split["train_ids"], dtype=torch.long)
    val_ids = torch.tensor(split["val_ids"], dtype=torch.long)
    excluded = torch.zeros(labels.numel(), dtype=torch.bool)
    excluded[train_ids] = True
    excluded[val_ids] = True
    test_ids = (~excluded).nonzero().view(-1)

    gnn_pred = gnn_prob.argmax(dim=1)
    structural = _structural_evidence(
        graph.edge_index, gnn_pred, labels.numel(), gnn_prob.size(1)
    )
    evidence, evidence_names = invariant_evidence(gnn_prob, llm_prob, structural)
    context, context_names = beta_context(support.rows)
    x = torch.cat(
        [evidence, context.view(1, -1).expand(evidence.size(0), -1)], dim=1
    )
    names = evidence_names + context_names
    if names != support.feature_names:
        raise ValueError(f"Support/final feature mismatch for {name}")
    return TargetArtifacts(
        name=name,
        labels=labels,
        gnn_prob=gnn_prob,
        llm_prob=llm_prob,
        gnn_pred=gnn_pred,
        llm_pred=llm_prob.argmax(dim=1),
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        x=x,
        feature_names=names,
    )
