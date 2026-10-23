#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eargtc.router_v7_decision.data import decisive_partition, load_base_dataset_bundle
from eargtc.router_v7_decision.metrics import (
    decisive_metrics,
    stratified_bootstrap_intervals,
)
from eargtc.router_v7_decision.cache import atomic_torch_save

DATASETS = ["cora", "citeseer", "pubmed", "arxiv", "ogbn-products"]
FAMILIES = ["A", "B", "C"]
FORBIDDEN_SELECTION_KEYS = {
    "test_metrics",
    "test_node_ids",
    "test_scores",
    "test_targets",
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_selection_has_no_test_information(payload: object) -> None:
    if isinstance(payload, dict):
        overlap = FORBIDDEN_SELECTION_KEYS.intersection(payload)
        if overlap:
            raise ValueError(
                "selection contains forbidden test information: "
                + ", ".join(sorted(overlap))
            )
        for value in payload.values():
            _assert_selection_has_no_test_information(value)
    elif isinstance(payload, list):
        for value in payload:
            _assert_selection_has_no_test_information(value)


def _verify_frozen(run_root: Path) -> dict:
    manifests = {}
    for family in FAMILIES:
        path = run_root / "selection" / family / "FROZEN.json"
        payload = _json(path)
        if payload.get("status") != "frozen_validation_selection":
            raise ValueError(f"selection is not frozen: {path}")
        for _, endpoint in payload["endpoints"].items():
            for kind in ("selection", "head"):
                artifact = Path(endpoint[f"{kind}_path"])
                if _hash(artifact) != endpoint[f"{kind}_sha256"]:
                    raise ValueError(f"frozen {kind} artifact mutated: {artifact}")
                if kind == "selection":
                    _assert_selection_has_no_test_information(_json(artifact))
            if "feature_bank_path" in endpoint:
                artifact = Path(endpoint["feature_bank_path"])
                if _hash(artifact) != endpoint["feature_bank_sha256"]:
                    raise ValueError(f"frozen feature bank mutated: {artifact}")
        if "feature_manifest" in payload:
            feature_manifest = Path(payload["feature_manifest"]["path"])
            if _hash(feature_manifest) != payload["feature_manifest"]["sha256"]:
                raise ValueError(
                    f"frozen feature manifest mutated: {feature_manifest}"
                )
        for relative, digest in payload.get("code_hashes", {}).items():
            artifact = ROOT / relative
            if _hash(artifact) != digest:
                raise ValueError(f"code changed after {family} freeze: {artifact}")
        manifests[family] = payload
    return manifests


def _baseline_rows(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[int(row["node_id"])] = row
    return rows


def _route_metrics(routes: np.ndarray, targets: np.ndarray, scores: np.ndarray) -> dict:
    gnn = targets == 1
    llm = targets == 0
    correct = routes == gnn
    result = {
        "gnn_only_recall": float(routes[gnn].mean()),
        "llm_only_recall": float((~routes[llm]).mean()),
        "decisive_accuracy": float(correct.mean()),
        "auroc": float(roc_auc_score(targets, scores)),
        "average_precision": float(average_precision_score(targets, scores)),
        "counts": {
            "gnn_only": int(gnn.sum()),
            "llm_only": int(llm.sum()),
            "route_gnn": int(routes.sum()),
            "route_llm": int((~routes).sum()),
            "correct": int(correct.sum()),
            "total": int(targets.size),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "runs/router_v7_decision_embeddings_20260827_v1",
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=ROOT / "runs/router_v5_structural_oof_fixed_20260729_v1",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=ROOT.parent / "GNN-as-Judge"
    )
    parser.add_argument(
        "--gnn-embedding-root",
        type=Path,
        default=ROOT / "runs/gcn_ssl_v2_selected_20260623_v1",
    )
    parser.add_argument(
        "--v6-root",
        type=Path,
        default=ROOT / "runs/router_bc_improved_structural_oof_fixed_20260729_v1",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    completion = run_root / "test" / "LOCKED_EVALUATION_COMPLETE.json"
    if completion.exists():
        raise RuntimeError(
            f"locked test evaluation already completed; refusing a second run: {completion}"
        )

    frozen = _verify_frozen(run_root)
    bundles = {
        dataset: load_base_dataset_bundle(
            dataset,
            upstream_root=args.upstream_root,
            dataset_root=args.dataset_root,
            gnn_embedding_root=args.gnn_embedding_root,
        )
        for dataset in DATASETS
    }
    banks = {
        dataset: torch.load(
            run_root / "features" / dataset / "feature_bank.pt",
            map_location="cpu",
            weights_only=False,
        )
        for dataset in DATASETS
    }
    results = {}
    prediction_payloads = {}
    for dataset in DATASETS:
        bundle = bundles[dataset]
        local = decisive_partition(
            bundle.labels[bundle.test_ids],
            bundle.gnn_pred[bundle.test_ids],
            bundle.llm_pred[bundle.test_ids],
        )
        node_ids = bundle.test_ids[local.node_ids]
        targets = local.target.long()
        results[dataset] = {}
        prediction_payloads[dataset] = {
            "node_ids": node_ids,
            "targets": targets,
        }
        for family in FAMILIES:
            selection = _json(
                run_root / "selection" / family / dataset / "selection.json"
            )
            with Path(selection["head_path"]).open("rb") as handle:
                head = pickle.load(handle)
            key = selection["feature_key"]
            if family == "A":
                x = bundle.logit_evidence[node_ids]
            else:
                x = banks[dataset].full_feature(key, node_ids)
            scores = head.predict_proba(x)
            primary_threshold = float(selection["accuracy_threshold"]["threshold"])
            balanced_threshold = float(selection["balanced_threshold"]["threshold"])
            primary = decisive_metrics(scores, targets, primary_threshold)
            primary["bootstrap_95"] = stratified_bootstrap_intervals(
                scores,
                targets,
                primary_threshold,
                repetitions=args.bootstrap_repetitions,
                seed=42,
            )
            balanced = decisive_metrics(scores, targets, balanced_threshold)
            results[dataset][family] = {
                "selected_name": selection["selected_name"],
                "feature_key": key,
                "validation": selection["validation_metrics_at_accuracy_threshold"],
                "accuracy_operating_point": primary,
                "balanced_operating_point": balanced,
            }
            prediction_payloads[dataset][family] = {
                "scores": torch.tensor(scores, dtype=torch.float32),
                "route_gnn": torch.tensor(
                    scores >= primary_threshold, dtype=torch.bool
                ),
            }

        baseline_path = args.v6_root / dataset / "decisions.jsonl"
        baseline = _baseline_rows(baseline_path)
        missing = [int(node_id) for node_id in node_ids if int(node_id) not in baseline]
        if missing:
            raise ValueError(f"v6 decisions missing locked test IDs: {missing[:10]}")
        rows = [baseline[int(node_id)] for node_id in node_ids]
        routes = np.asarray([row["route"] == "gnn" for row in rows], dtype=bool)
        preference = np.asarray(
            [float(row["preference_score"]) for row in rows], dtype=np.float64
        )
        baseline_metrics = _route_metrics(
            routes, targets.numpy(), preference
        )
        baseline_metrics["artifact_hashes"] = {"decisions_sha256": _hash(baseline_path)}
        results[dataset]["v6_scheme_b"] = baseline_metrics
        prediction_payloads[dataset]["v6_scheme_b"] = {
            "scores": torch.tensor(preference, dtype=torch.float32),
            "route_gnn": torch.tensor(routes, dtype=torch.bool),
        }

    test_root = run_root / "test"
    test_root.mkdir(parents=True, exist_ok=True)
    predictions_path = test_root / "predictions.pt"
    atomic_torch_save(prediction_payloads, predictions_path)
    summary_path = test_root / "metrics.json"
    temporary_summary = summary_path.with_name(
        f".{summary_path.name}.tmp.{os.getpid()}"
    )
    temporary_summary.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_summary, summary_path)
    complete = {
        "status": "complete",
        "families": FAMILIES,
        "datasets": DATASETS,
        "metrics_path": str(summary_path),
        "metrics_sha256": _hash(summary_path),
        "predictions_path": str(predictions_path),
        "predictions_sha256": _hash(predictions_path),
        "evaluator_code_path": str(Path(__file__).resolve()),
        "evaluator_code_sha256": _hash(Path(__file__).resolve()),
        "frozen_manifest_hashes": {
            family: _hash(run_root / "selection" / family / "FROZEN.json")
            for family in FAMILIES
        },
    }
    completion.parent.mkdir(parents=True, exist_ok=True)
    temporary = completion.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(complete, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, completion)
    print(f"Locked evaluation complete: {completion}")


if __name__ == "__main__":
    main()
