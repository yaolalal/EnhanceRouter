from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS = ("cora", "citeseer", "pubmed", "arxiv", "ogbn-products")
DEFAULT_METHODS = ("A", "B", "C", "v6_scheme_b")
PAIRED_CONTRASTS = {
    "B_minus_A": ("B", "A"),
    "C_minus_A": ("C", "A"),
    "C_minus_B": ("C", "B"),
    "A_minus_v6_scheme_b": ("A", "v6_scheme_b"),
    "B_minus_v6_scheme_b": ("B", "v6_scheme_b"),
    "C_minus_v6_scheme_b": ("C", "v6_scheme_b"),
}
FORBIDDEN_SELECTION_KEYS = {
    "test_metrics",
    "test_node_ids",
    "test_scores",
    "test_targets",
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _exact_keys(payload: dict, expected: Iterable[str], label: str) -> None:
    expected_set = set(expected)
    actual = set(payload)
    if actual != expected_set:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected_set - actual)}, "
            f"extra={sorted(actual - expected_set)}"
        )


def _verify_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    if _hash(path) != expected:
        raise ValueError(f"{label} hash mismatch: {path}")


def _reject_test_information(payload: object) -> None:
    if isinstance(payload, dict):
        overlap = FORBIDDEN_SELECTION_KEYS.intersection(payload)
        if overlap:
            raise ValueError(
                "selection contains forbidden test information: "
                + ", ".join(sorted(overlap))
            )
        for value in payload.values():
            _reject_test_information(value)
    elif isinstance(payload, list):
        for value in payload:
            _reject_test_information(value)


def _primary_metrics(record: dict, method: str) -> dict:
    if method == "v6_scheme_b":
        return record
    return record["accuracy_operating_point"]


def _route_metrics(routes: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    gnn_mask = targets == 1
    llm_mask = targets == 0
    correct = routes == targets.bool()
    return {
        "gnn_only_recall": int(routes[gnn_mask].sum().item())
        / int(gnn_mask.sum().item()),
        "llm_only_recall": int((~routes[llm_mask]).sum().item())
        / int(llm_mask.sum().item()),
        "decisive_accuracy": int(correct.sum().item()) / int(targets.numel()),
    }


def _check_close(observed: float, expected: float, label: str) -> None:
    if not np.isclose(float(observed), float(expected), atol=1e-8, rtol=1e-7):
        raise ValueError(f"{label} mismatch: {observed} != {expected}")


def audit_final_run(
    run_root: str | Path,
    hidden_root: str | Path,
    *,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    methods: tuple[str, ...] = DEFAULT_METHODS,
) -> dict:
    run_root = Path(run_root).resolve()
    hidden_root = Path(hidden_root).resolve()
    datasets = tuple(datasets)
    methods = tuple(methods)
    if not datasets or methods != DEFAULT_METHODS:
        if not datasets or set(methods) != set(DEFAULT_METHODS):
            raise ValueError("audit requires nonempty datasets and A/B/C/v6 methods")

    hidden_audit_path = hidden_root / "AUDIT.json"
    hidden_audit = _json(hidden_audit_path)
    if hidden_audit.get("status") != "pass":
        raise ValueError("hidden audit did not pass")
    hidden_records = hidden_audit.get("datasets", {})
    _exact_keys(hidden_records, datasets, "hidden audit dataset")
    for dataset, record in hidden_records.items():
        if record.get("status") != "pass" or int(record.get("rows", 0)) <= 0:
            raise ValueError(f"hidden audit endpoint failed: {dataset}")
        if int(record.get("width", 0)) != 4096:
            raise ValueError(f"hidden width mismatch: {dataset}")
        if not bool(record.get("parity_argmax_match")):
            raise ValueError(f"hidden argmax parity failed: {dataset}")
        if float(record.get("parity_max_probability_error", float("inf"))) > 0.05:
            raise ValueError(f"hidden probability parity failed: {dataset}")
        if not Path(record["output"]).is_file():
            raise FileNotFoundError(f"missing merged hidden output: {dataset}")

    feature_manifest_path = run_root / "features/MANIFEST.json"
    feature_manifest = _json(feature_manifest_path)
    if feature_manifest.get("status") != "complete":
        raise ValueError("feature manifest is not complete")
    feature_records = feature_manifest.get("datasets", {})
    _exact_keys(feature_records, datasets, "feature manifest dataset")
    for dataset, record in feature_records.items():
        if int(record.get("test_overlap", -1)) != 0:
            raise ValueError(f"projection/test overlap detected: {dataset}")
        _verify_hash(Path(record["path"]), record["sha256"], "feature bank")

    frozen_hashes = {}
    for family in ("A", "B", "C"):
        frozen_path = run_root / f"selection/{family}/FROZEN.json"
        frozen = _json(frozen_path)
        if frozen.get("status") != "frozen_validation_selection":
            raise ValueError(f"selection is not frozen: {family}")
        endpoints = frozen.get("endpoints", {})
        _exact_keys(endpoints, datasets, f"{family} frozen endpoint")
        for dataset, endpoint in endpoints.items():
            for kind in ("selection", "head"):
                artifact = Path(endpoint[f"{kind}_path"])
                _verify_hash(
                    artifact,
                    endpoint[f"{kind}_sha256"],
                    f"frozen {kind}",
                )
                if kind == "selection":
                    _reject_test_information(_json(artifact))
            if "feature_bank_path" in endpoint:
                _verify_hash(
                    Path(endpoint["feature_bank_path"]),
                    endpoint["feature_bank_sha256"],
                    "frozen feature bank",
                )
        if "feature_manifest" in frozen:
            _verify_hash(
                Path(frozen["feature_manifest"]["path"]),
                frozen["feature_manifest"]["sha256"],
                "frozen feature manifest",
            )
        for relative, digest in frozen.get("code_hashes", {}).items():
            _verify_hash(PROJECT_ROOT / relative, digest, "frozen code")
        frozen_hashes[family] = _hash(frozen_path)

    completion_path = run_root / "test/LOCKED_EVALUATION_COMPLETE.json"
    completion = _json(completion_path)
    if completion.get("status") != "complete":
        raise ValueError("locked evaluation is not complete")
    metrics_path = Path(completion["metrics_path"])
    predictions_path = Path(completion["predictions_path"])
    _verify_hash(metrics_path, completion["metrics_sha256"], "locked metrics")
    _verify_hash(
        predictions_path,
        completion["predictions_sha256"],
        "locked predictions",
    )
    _verify_hash(
        Path(completion["evaluator_code_path"]),
        completion["evaluator_code_sha256"],
        "locked evaluator",
    )
    if completion.get("frozen_manifest_hashes") != frozen_hashes:
        raise ValueError("locked evaluation frozen-manifest hashes differ")

    metrics = _json(metrics_path)
    predictions = torch.load(
        predictions_path, map_location="cpu", weights_only=False
    )
    _exact_keys(metrics, datasets, "locked metric dataset")
    _exact_keys(predictions, datasets, "locked prediction dataset")
    dataset_audit = {}
    locked_points = {}
    for dataset in datasets:
        _exact_keys(metrics[dataset], methods, f"{dataset} metric method")
        prediction = predictions[dataset]
        node_ids = torch.as_tensor(prediction["node_ids"], dtype=torch.long).view(-1)
        targets = torch.as_tensor(prediction["targets"], dtype=torch.long).view(-1)
        total = int(targets.numel())
        if total == 0 or node_ids.numel() != total:
            raise ValueError(f"empty or misaligned decisive IDs: {dataset}")
        if torch.unique(node_ids).numel() != total:
            raise ValueError(f"duplicate decisive IDs: {dataset}")
        if set(targets.tolist()) != {0, 1}:
            raise ValueError(f"both decisive classes are required: {dataset}")
        gnn_count = int((targets == 1).sum().item())
        llm_count = total - gnn_count
        locked_points[dataset] = {}
        for method in methods:
            method_prediction = prediction[method]
            scores = torch.as_tensor(method_prediction["scores"]).view(-1)
            routes = torch.as_tensor(
                method_prediction["route_gnn"], dtype=torch.bool
            ).view(-1)
            if scores.numel() != total or routes.numel() != total:
                raise ValueError(
                    f"prediction rows do not align: {dataset}/{method}"
                )
            if not bool(torch.isfinite(scores).all().item()):
                raise ValueError(f"non-finite scores: {dataset}/{method}")
            correct = routes == targets.bool()
            observed = _route_metrics(routes, targets)
            primary = _primary_metrics(metrics[dataset][method], method)
            for key, value in observed.items():
                _check_close(value, primary[key], f"{dataset}/{method}/{key}")
            locked_points[dataset][method] = observed
            expected_counts = {
                "gnn_only": gnn_count,
                "llm_only": llm_count,
                "route_gnn": int(routes.sum().item()),
                "route_llm": int((~routes).sum().item()),
                "correct": int(correct.sum().item()),
                "total": total,
            }
            if primary["counts"] != expected_counts:
                raise ValueError(f"metric counts mismatch: {dataset}/{method}")
        dataset_audit[dataset] = {
            "decisive_rows": total,
            "gnn_only": gnn_count,
            "llm_only": llm_count,
        }

    summary_path = run_root / "summary.json"
    summary = _json(summary_path)
    rows = summary.get("rows", [])
    expected_pairs = {(dataset, method) for dataset in datasets for method in methods}
    actual_pairs = {(row["dataset"], row["method"]) for row in rows}
    if len(rows) != len(expected_pairs) or actual_pairs != expected_pairs:
        raise ValueError("summary rows do not exactly cover dataset/method pairs")
    for row in rows:
        primary = _primary_metrics(metrics[row["dataset"]][row["method"]], row["method"])
        for key in ("gnn_only_recall", "llm_only_recall", "decisive_accuracy"):
            _check_close(row[key], primary[key], f"summary/{row['dataset']}/{row['method']}/{key}")
        if row["method"] != "v6_scheme_b":
            intervals = row.get("bootstrap_95", {})
            for key in (
                "gnn_only_recall",
                "llm_only_recall",
                "decisive_accuracy",
            ):
                interval = np.asarray(intervals.get(key, []), dtype=np.float64)
                if (
                    interval.shape != (2,)
                    or not np.isfinite(interval).all()
                    or interval[0] > interval[1]
                ):
                    raise ValueError(f"invalid single-method bootstrap: {row['dataset']}/{row['method']}/{key}")

    macro = summary.get("macro", {})
    _exact_keys(macro, methods, "macro method")
    for method in methods:
        for key in ("gnn_only_recall", "llm_only_recall", "decisive_accuracy"):
            expected = float(
                np.mean([locked_points[dataset][method][key] for dataset in datasets])
            )
            _check_close(macro[method][key], expected, f"macro/{method}/{key}")

    paired = summary.get("paired_differences", {})
    _exact_keys(paired, datasets, "paired dataset")
    for dataset in datasets:
        _exact_keys(paired[dataset], PAIRED_CONTRASTS, "paired contrast")
        for contrast, (first, second) in PAIRED_CONTRASTS.items():
            record = paired[dataset][contrast]
            for key in (
                "gnn_only_recall",
                "llm_only_recall",
                "decisive_accuracy",
            ):
                value = record[key]
                expected = (
                    locked_points[dataset][first][key]
                    - locked_points[dataset][second][key]
                )
                _check_close(
                    value["difference"],
                    expected,
                    f"paired/{dataset}/{contrast}/{key}",
                )
                interval = np.asarray(value["ci95"], dtype=np.float64)
                if (
                    interval.shape != (2,)
                    or not np.isfinite(interval).all()
                    or interval[0] > interval[1]
                ):
                    raise ValueError(
                        f"invalid paired interval: {dataset}/{contrast}/{key}"
                    )

    required_reports = [run_root / "summary.csv", run_root / "REPORT.md"]
    for path in required_reports:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty final report artifact: {path}")
    return {
        "status": "pass",
        "datasets": dataset_audit,
        "summary_rows": len(rows),
        "artifact_hashes": {
            "final_auditor": _hash(Path(__file__).resolve()),
            "hidden_audit": _hash(hidden_audit_path),
            "feature_manifest": _hash(feature_manifest_path),
            "locked_completion": _hash(completion_path),
            "locked_metrics": _hash(metrics_path),
            "locked_predictions": _hash(predictions_path),
            "summary_json": _hash(summary_path),
            "summary_csv": _hash(run_root / "summary.csv"),
            "report_md": _hash(run_root / "REPORT.md"),
            **{f"frozen_{family}": digest for family, digest in frozen_hashes.items()},
        },
    }


__all__ = ["audit_final_run"]
