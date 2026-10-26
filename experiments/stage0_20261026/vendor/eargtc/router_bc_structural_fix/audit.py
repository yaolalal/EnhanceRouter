from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from eargtc.data import _to_undirected

from .oof import FoldOOFResult
from .provenance import sha256_file, verify_prediction_identity


def _structural_from_predictions(
    edge_index: torch.Tensor,
    predictions: torch.Tensor,
    *,
    num_nodes: int,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_index = _to_undirected(edge_index.detach().cpu())
    predictions = predictions.detach().cpu().long()
    src, dst = edge_index[0].long(), edge_index[1].long()
    degree = torch.bincount(src, minlength=int(num_nodes)).float()
    same_count = torch.zeros(int(num_nodes), dtype=torch.float32)
    same_count.index_add_(0, src, (predictions[src] == predictions[dst]).float())
    agreement = torch.where(
        degree > 0,
        same_count / degree.clamp_min(1.0),
        torch.zeros_like(degree),
    )
    counts = torch.zeros((int(num_nodes), int(num_classes)), dtype=torch.float32)
    counts.index_put_(
        (src, predictions[dst]),
        torch.ones_like(src, dtype=torch.float32),
        accumulate=True,
    )
    q = counts / degree.clamp_min(1.0).unsqueeze(1)
    entropy = -(q * q.clamp_min(1e-12).log()).sum(dim=1) / math.log(
        int(num_classes)
    )
    entropy = torch.where(degree > 0, entropy, torch.ones_like(entropy))
    return degree, agreement, entropy


def audit_fold_oof_result(
    *,
    result: FoldOOFResult,
    data,
    num_classes: int,
    tolerance: float = 1e-6,
) -> dict:
    if result.full_gnn_pred.shape != (int(data.num_nodes),):
        raise AssertionError("structural audit: saved fold predictions do not cover graph")
    if len(result.rows) != len(result.holdout_ids):
        raise AssertionError("structural audit: row/holdout count mismatch")
    degree, agreement, entropy = _structural_from_predictions(
        data.edge_index,
        result.full_gnn_pred,
        num_nodes=int(data.num_nodes),
        num_classes=int(num_classes),
    )
    by_node = {int(row["node_id"]): row for row in result.rows}
    if set(by_node) != set(result.holdout_ids):
        raise AssertionError("structural audit: holdout node coverage mismatch")
    if int(result.holdout_gnn_logits.size(0)) != len(result.holdout_ids):
        raise AssertionError("structural audit: holdout logits count mismatch")

    max_agreement_error = 0.0
    max_entropy_error = 0.0
    max_degree_error = 0.0
    max_probability_error = 0.0
    for offset, node_id in enumerate(result.holdout_ids):
        row = by_node[int(node_id)]
        features = row["features"]
        max_agreement_error = max(
            max_agreement_error,
            abs(
                float(features["neighbor_gnn_agreement"])
                - float(agreement[node_id].item())
            ),
        )
        max_entropy_error = max(
            max_entropy_error,
            abs(
                float(features["neighbor_gnn_entropy"])
                - float(entropy[node_id].item())
            ),
        )
        max_degree_error = max(
            max_degree_error,
            abs(float(features["log_degree"]) - math.log1p(float(degree[node_id]))),
        )
        probability = torch.softmax(result.holdout_gnn_logits[offset].float(), dim=0)
        saved_probability = torch.tensor(row["gnn_prob"], dtype=torch.float32)
        max_probability_error = max(
            max_probability_error,
            float((probability - saved_probability).abs().max().item()),
        )
        if int(probability.argmax()) != int(result.full_gnn_pred[node_id]):
            raise AssertionError("structural audit: holdout/full prediction mismatch")
        if int(row["gnn_pred"]) != int(result.full_gnn_pred[node_id]):
            raise AssertionError("structural audit: saved row prediction mismatch")

    permutation = torch.arange(int(num_classes)).roll(1)
    _, permuted_agreement, permuted_entropy = _structural_from_predictions(
        data.edge_index,
        permutation[result.full_gnn_pred],
        num_nodes=int(data.num_nodes),
        num_classes=int(num_classes),
    )
    permutation_invariant = bool(
        torch.allclose(agreement, permuted_agreement, atol=tolerance, rtol=0.0)
        and torch.allclose(entropy, permuted_entropy, atol=tolerance, rtol=0.0)
    )
    checks = {
        "neighbor_agreement": max_agreement_error <= tolerance,
        "neighbor_entropy": max_entropy_error <= tolerance,
        "log_degree": max_degree_error <= tolerance,
        "gnn_probability": max_probability_error <= tolerance,
        "class_permutation_invariant": permutation_invariant,
    }
    if not all(checks.values()):
        raise AssertionError(
            "structural audit failed: "
            f"checks={checks}, tolerance={tolerance}, "
            f"max_agreement_error={max_agreement_error}, "
            f"max_entropy_error={max_entropy_error}, "
            f"max_degree_error={max_degree_error}, "
            f"max_probability_error={max_probability_error}"
        )
    return {
        "tolerance": float(tolerance),
        "fold_id": int(result.fold_id),
        "checked_rows": len(result.rows),
        "max_neighbor_agreement_error": max_agreement_error,
        "max_neighbor_entropy_error": max_entropy_error,
        "max_log_degree_error": max_degree_error,
        "max_gnn_probability_error": max_probability_error,
        "class_permutation_invariant": permutation_invariant,
        "checks": checks,
        "all_checks_pass": True,
    }



def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit_saved_corrected_dataset(
    *,
    dataset: str,
    dataset_dir: str | Path,
    old_pipeline: str | Path,
    data,
    tolerance: float = 1e-6,
) -> dict:
    dataset_dir = Path(dataset_dir)
    old_pipeline = Path(old_pipeline)
    pipeline = dataset_dir / "pipeline"
    oof_dir = pipeline / "router" / "oof"
    split_name = f"{dataset}_5shot_seed42.json"
    old_split = old_pipeline / "splits" / split_name
    new_split = pipeline / "splits" / split_name
    split_identical = sha256_file(old_split) == sha256_file(new_split)
    config_identical = sha256_file(old_pipeline / "config.json") == sha256_file(
        pipeline / "config.json"
    )
    prediction_identity = verify_prediction_identity(
        old_pipeline / "predictions.pt", pipeline / "predictions.pt"
    )
    split = json.loads(new_split.read_text(encoding="utf-8"))
    train_ids = [int(node_id) for node_id in split["train_ids"]]
    combined_rows = _read_jsonl(oof_dir / "router_train.jsonl")
    combined_ids = [int(row["node_id"]) for row in combined_rows]
    support_order_and_coverage = (
        combined_ids == train_ids
        and len(combined_ids) == len(set(combined_ids))
    )
    if not support_order_and_coverage:
        raise AssertionError("saved structural audit: support order/coverage mismatch")
    num_classes = int(
        torch.load(
            pipeline / "predictions.pt", map_location="cpu", weights_only=False
        )["gnn_logits"].size(1)
    )

    fold_ids = sorted({int(row["fold"]) for row in combined_rows})
    fold_reports = []
    max_zero_difference = 0.0
    combined_by_node = {int(row["node_id"]): row for row in combined_rows}
    for fold_id in fold_ids:
        fold_rows = _read_jsonl(oof_dir / f"fold_{fold_id}_predictions.jsonl")
        holdout_ids = [int(row["node_id"]) for row in fold_rows]
        expected_ids = {
            int(row["node_id"])
            for row in combined_rows
            if int(row["fold"]) == fold_id
        }
        if set(holdout_ids) != expected_ids or len(holdout_ids) != len(expected_ids):
            raise AssertionError("saved structural audit: fold row coverage mismatch")
        for row in fold_rows:
            if row != combined_by_node[int(row["node_id"])]:
                raise AssertionError("saved structural audit: fold/combined row mismatch")
        result = FoldOOFResult(
            fold_id=fold_id,
            holdout_ids=holdout_ids,
            rows=fold_rows,
            full_gnn_pred=torch.load(
                oof_dir / f"fold_{fold_id}_full_gnn_pred.pt",
                map_location="cpu",
                weights_only=False,
            ).long(),
            holdout_gnn_logits=torch.load(
                oof_dir / f"fold_{fold_id}_holdout_gnn_logits.pt",
                map_location="cpu",
                weights_only=False,
            ).float(),
        )
        fold_reports.append(
            audit_fold_oof_result(
                result=result,
                data=data,
                num_classes=num_classes,
                tolerance=tolerance,
            )
        )
        zero_prediction = torch.zeros(int(data.num_nodes), dtype=torch.long)
        zero_prediction[torch.tensor(holdout_ids, dtype=torch.long)] = (
            result.holdout_gnn_logits.argmax(dim=1)
        )
        _, zero_agreement, zero_entropy = _structural_from_predictions(
            data.edge_index,
            zero_prediction,
            num_nodes=int(data.num_nodes),
            num_classes=num_classes,
        )
        for row in fold_rows:
            node_id = int(row["node_id"])
            max_zero_difference = max(
                max_zero_difference,
                abs(
                    float(row["features"]["neighbor_gnn_agreement"])
                    - float(zero_agreement[node_id])
                ),
                abs(
                    float(row["features"]["neighbor_gnn_entropy"])
                    - float(zero_entropy[node_id])
                ),
            )

    old_rows = _read_jsonl(old_pipeline / "router" / "oof" / "router_train.jsonl")
    old_by_node = {int(row["node_id"]): row for row in old_rows}
    if set(old_by_node) != set(combined_by_node):
        raise AssertionError("saved structural audit: old/new support node mismatch")
    log_degree_identical = True
    llm_evidence_identical = True
    llm_feature_names = {
        "llm_conf",
        "llm_entropy",
        "llm_margin",
    }
    for node_id, new_row in combined_by_node.items():
        old_row = old_by_node[node_id]
        log_degree_identical &= (
            abs(
                float(new_row["features"]["log_degree"])
                - float(old_row["features"]["log_degree"])
            )
            <= tolerance
        )
        llm_evidence_identical &= int(new_row["llm_pred"]) == int(
            old_row["llm_pred"]
        )
        llm_evidence_identical &= torch.allclose(
            torch.tensor(new_row["llm_prob"]),
            torch.tensor(old_row["llm_prob"]),
            atol=tolerance,
            rtol=0.0,
        )
        names = llm_feature_names | {
            name
            for name in new_row["features"]
            if name.startswith("llm_logit_")
        }
        for name in names:
            llm_evidence_identical &= (
                abs(
                    float(new_row["features"][name])
                    - float(old_row["features"][name])
                )
                <= tolerance
            )
    zero_background_exact = max_zero_difference <= tolerance
    checks = {
        "split_identical": split_identical,
        "config_identical": config_identical,
        "prediction_identity": prediction_identity["all_checks_pass"],
        "support_order_and_coverage": support_order_and_coverage,
        "fold_audits": all(row["all_checks_pass"] for row in fold_reports),
        "log_degree_identical": bool(log_degree_identical),
        "llm_evidence_identical": bool(llm_evidence_identical),
        "pollution_not_reconstructible": not zero_background_exact,
    }
    if not all(checks.values()):
        raise AssertionError(f"saved structural audit failed: {checks}")
    return {
        "dataset": str(dataset),
        "support_count": len(combined_rows),
        "fold_count": len(fold_reports),
        "prediction_identity": prediction_identity,
        "split_identical": split_identical,
        "config_identical": config_identical,
        "support_order_and_coverage": support_order_and_coverage,
        "folds": fold_reports,
        "log_degree_identical": bool(log_degree_identical),
        "llm_evidence_identical": bool(llm_evidence_identical),
        "zero_background_exact_reconstruction": zero_background_exact,
        "max_zero_background_structural_difference": max_zero_difference,
        "checks": checks,
        "all_checks_pass": True,
    }
