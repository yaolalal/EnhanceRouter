from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from eargtc.router_bc.data_impl import build_support_domain, load_target_artifacts


def join_evidence(
    rows: Sequence[Mapping],
    feature_names: Sequence[str],
    matrix: np.ndarray,
    node_ids: np.ndarray,
) -> dict[int, dict[str, float]]:
    values = np.asarray(matrix, dtype=float)
    ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
    names = [str(name) for name in feature_names]
    if values.ndim != 2 or values.shape != (ids.size, len(names)):
        raise ValueError("evidence_shape_mismatch")
    if np.unique(ids).size != ids.size:
        raise ValueError("duplicate_evidence_node_id")
    lookup = {
        int(node_id): {
            name: float(values[index, column])
            for column, name in enumerate(names)
        }
        for index, node_id in enumerate(ids)
    }
    requested = [int(row["node_id"]) for row in rows]
    missing = sorted(set(requested).difference(lookup))
    if missing:
        raise ValueError(f"missing_evidence_node_id:{missing[0]}")
    return {node_id: lookup[node_id] for node_id in requested}


def reconstruct_test_evidence(
    dataset: str,
    pipeline_dir: str | Path,
    dataset_root: str | Path,
    decisions: Sequence[Mapping],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    pipeline = Path(pipeline_dir)
    support = build_support_domain(
        dataset, pipeline / "router" / "oof" / "router_train.jsonl"
    )
    target = load_target_artifacts(dataset, pipeline, dataset_root, support)
    test_ids = target.test_ids.detach().cpu().numpy().astype(np.int64)
    matrix = target.x[target.test_ids].detach().cpu().numpy().astype(float)
    decision_by_id = {int(row["node_id"]): row for row in decisions}
    if len(decision_by_id) != len(decisions):
        raise ValueError("duplicate_decision_node_id")
    if set(decision_by_id) != set(test_ids.tolist()):
        raise ValueError("decision_test_node_id_mismatch")
    for node_id in test_ids:
        row = decision_by_id[int(node_id)]
        if int(row["gnn_pred"]) != int(target.gnn_pred[int(node_id)].item()):
            raise ValueError(f"frozen_gnn_prediction_mismatch:{node_id}")
        if int(row["llm_pred"]) != int(target.llm_pred[int(node_id)].item()):
            raise ValueError(f"frozen_llm_prediction_mismatch:{node_id}")
        if int(row["true_label"]) != int(target.labels[int(node_id)].item()):
            raise ValueError(f"frozen_label_mismatch:{node_id}")
    return list(target.feature_names), matrix, test_ids

