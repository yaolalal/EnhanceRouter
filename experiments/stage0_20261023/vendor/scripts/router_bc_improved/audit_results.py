#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eargtc.router_bc_improved.audit import audit_decision_rows
from eargtc.router_bc_improved.runner_utils import validate_selection_record


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 1e-6) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    aggregate = json.loads((args.result_root / "summary.json").read_text())["datasets"]
    aggregate_by_name = {row["dataset"]: row for row in aggregate}
    reports = []
    for dataset_dir in sorted(path for path in args.result_root.iterdir() if path.is_dir()):
        metrics_path = dataset_dir / "metrics.json"
        decisions_path = dataset_dir / "decisions.jsonl"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        dataset = metrics["dataset"]
        rows = [json.loads(line) for line in decisions_path.read_text().splitlines() if line]
        b_selection = metrics["scheme_b"]["selected_checkpoint"]
        c_selection = metrics["scheme_c"]["metrics"]["validation_selection"]
        validate_selection_record(b_selection)
        validate_selection_record(c_selection)
        selected_trust = c_selection["selected"]
        recomputed = audit_decision_rows(rows, selected_trust=selected_trust)
        stored = metrics["scheme_b"]["test"]
        checks = {
            "count": recomputed["count"] == metrics["counts"]["test"],
            "router_accuracy": close(recomputed["router_accuracy"], stored["router_accuracy"]),
            "gnn_accuracy": close(recomputed["gnn_accuracy"], stored["gnn_accuracy"]),
            "llm_accuracy": close(recomputed["llm_accuracy"], stored["llm_accuracy"]),
            "best_single_accuracy": close(recomputed["best_single_accuracy"], stored["best_single_accuracy"]),
            "oracle_accuracy": close(recomputed["oracle_accuracy"], stored["oracle_accuracy"]),
            "beats_best_single": recomputed["router_accuracy"] > recomputed["best_single_accuracy"],
            "scheme_c_route_identity": metrics["scheme_c"]["route_identity_with_b"] is True,
            "predictions_hash": sha256(Path(metrics["provenance"]["predictions"])) == metrics["provenance"]["predictions_sha256"],
            "oof_hash": sha256(Path(metrics["provenance"]["target_oof"])) == metrics["provenance"]["target_oof_sha256"],
            "aggregate_accuracy": close(aggregate_by_name[dataset]["scheme_b"], recomputed["router_accuracy"]),
        }
        if not all(checks.values()):
            raise AssertionError(f"{dataset} audit failed: {checks}")
        reports.append(
            {
                "dataset": dataset,
                "checks": checks,
                "recomputed": recomputed,
                "selected_b": metrics["scheme_b"]["selected_metadata"],
                "selected_c": selected_trust,
            }
        )
    expected = set(aggregate_by_name)
    actual = {row["dataset"] for row in reports}
    if actual != expected:
        raise AssertionError(f"Audit dataset mismatch: expected={expected}, actual={actual}")
    output = {"result_root": str(args.result_root.resolve()), "all_checks_pass": True, "datasets": reports}
    (args.result_root / "audit.json").write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
