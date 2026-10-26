#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eargtc.router_v7_decision.reporting import (
    method_summary_row,
    paired_bootstrap_difference,
)

DATASETS = ["cora", "citeseer", "pubmed", "arxiv", "ogbn-products"]
METHODS = ["A", "B", "C", "v6_scheme_b"]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fmt(value: float) -> str:
    return f"{float(value):.4f}"


def _fmt_optional(value: float | None) -> str:
    return "n/a" if value is None else _fmt(value)


def _accuracy_ci(row: dict) -> str:
    interval = row["bootstrap_95"].get("decisive_accuracy")
    if not interval:
        return "n/a"
    lower, upper = interval
    return f"[{_fmt(lower)}, {_fmt(upper)}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "runs/router_v7_decision_embeddings_20260827_v1",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    complete = json.loads(
        (run_root / "test/LOCKED_EVALUATION_COMPLETE.json").read_text()
    )
    metrics = json.loads(Path(complete["metrics_path"]).read_text())
    predictions = torch.load(
        complete["predictions_path"], map_location="cpu", weights_only=False
    )
    frozen = {
        family: json.loads((run_root / f"selection/{family}/FROZEN.json").read_text())
        for family in ("A", "B", "C")
    }
    frozen_audit = {}
    for family in ("A", "B", "C"):
        path = run_root / f"selection/{family}/FROZEN.json"
        digest = _hash(path)
        expected = complete["frozen_manifest_hashes"][family]
        if digest != expected:
            raise ValueError(f"frozen manifest changed before reporting: {path}")
        frozen_audit[family] = {
            "path": str(path),
            "sha256": digest,
        }

    rows = []
    paired = {}
    for dataset in DATASETS:
        paired[dataset] = {}
        for method in METHODS:
            if method == "v6_scheme_b":
                value = metrics[dataset][method]
                counts = dict(value["counts"])
                total = int(counts["total"])
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "gnn_only_recall": value["gnn_only_recall"],
                        "llm_only_recall": value["llm_only_recall"],
                        "decisive_accuracy": value["decisive_accuracy"],
                        "auroc": value["auroc"],
                        "average_precision": value["average_precision"],
                        "nll": value.get("nll"),
                        "gnn_only_count": counts["gnn_only"],
                        "llm_only_count": counts["llm_only"],
                        "validation_accuracy": None,
                        "validation_to_test_gap": None,
                        "selected": "frozen Router-v6",
                        "feature_key": "legacy",
                        "primary_threshold": None,
                        "balanced_threshold": None,
                        "validation_metrics": {},
                        "balanced_operating_point": {},
                        "bootstrap_95": value.get("bootstrap_95", {}),
                        "counts": counts,
                        "route_gnn_fraction": float(counts["route_gnn"] / total),
                        "selected_spec": {"source": "frozen Router-v6 Scheme B"},
                        "adapter": {},
                        "artifact_hashes": value.get("artifact_hashes", {}),
                    }
                )
                continue
            record = metrics[dataset][method]
            selection_path = run_root / f"selection/{method}/{dataset}/selection.json"
            selection = json.loads(selection_path.read_text())
            endpoint = frozen[method]["endpoints"][dataset]
            artifact_hashes = {
                key: value
                for key, value in endpoint.items()
                if key.endswith("_sha256")
            }
            rows.append(
                method_summary_row(
                    dataset,
                    method,
                    record,
                    selection=selection,
                    artifact_hashes=artifact_hashes,
                )
            )
        for first, second in (
            ("B", "A"),
            ("C", "A"),
            ("C", "B"),
            ("A", "v6_scheme_b"),
            ("B", "v6_scheme_b"),
            ("C", "v6_scheme_b"),
        ):
            paired[dataset][f"{first}_minus_{second}"] = paired_bootstrap_difference(
                predictions[dataset][first]["route_gnn"],
                predictions[dataset][second]["route_gnn"],
                predictions[dataset]["targets"],
                repetitions=args.bootstrap_repetitions,
                seed=42,
            )

    macro = {}
    for method in METHODS:
        selected_rows = [row for row in rows if row["method"] == method]
        macro[method] = {
            key: float(np.mean([row[key] for row in selected_rows]))
            for key in (
                "gnn_only_recall",
                "llm_only_recall",
                "decisive_accuracy",
                "auroc",
                "average_precision",
            )
        }
    summary = {
        "protocol": {
            "task": "decisive conflicts only: target 1=GNN-only, target 0=LLM-only",
            "selection": "source-only LODO for representation/model; target validation for threshold; locked test once",
            "A": "approved 20 logits/structural features",
            "B": "embedding-only paired low-rank evidence",
            "C": "A plus embedding evidence",
        },
        "artifact_audit": {
            "locked_evaluation_complete": complete,
            "frozen_manifests": frozen_audit,
            "report_generator": {
                "path": str(Path(__file__).resolve()),
                "sha256": _hash(Path(__file__).resolve()),
            },
        },
        "rows": rows,
        "macro": macro,
        "paired_differences": paired,
    }
    summary_path = run_root / "summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, summary_path)
    csv_path = run_root / "summary.csv"
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()}
            )
    os.replace(temporary_csv, csv_path)

    lines = [
        "# Router-v7 Scheme-B Embedding Decision Router Results",
        "",
        "Primary task: choose GNN on GNN-only nodes and LLM on LLM-only nodes. Agreement and neither-correct nodes are excluded throughout.",
        "",
        "## Locked-test primary operating point",
        "",
        "| Dataset | Method | Threshold | GNN-only recall | LLM-only recall | Decisive accuracy | Accuracy 95% CI | AUROC | AP | Route GNN | Selected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {method} | {threshold} | {rg} | {rl} | {acc} | {ci} | {auc} | {ap} | {route} | {selected} / {feature} |".format(
                dataset=row["dataset"],
                method=row["method"],
                threshold=_fmt_optional(row["primary_threshold"]),
                rg=_fmt(row["gnn_only_recall"]),
                rl=_fmt(row["llm_only_recall"]),
                acc=_fmt(row["decisive_accuracy"]),
                ci=_accuracy_ci(row),
                auc=_fmt(row["auroc"]),
                ap=_fmt(row["average_precision"]),
                route=_fmt(row["route_gnn_fraction"]),
                selected=row["selected"],
                feature=row["feature_key"],
            )
        )
    lines.extend(
        [
            "",
            "## Decisive counts and routing",
            "",
            "| Dataset | Method | GNN-only n | LLM-only n | Route GNN n | Route LLM n | Correct / total |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        counts = row["counts"]
        lines.append(
            f"| {row['dataset']} | {row['method']} | {counts['gnn_only']} | {counts['llm_only']} | {counts['route_gnn']} | {counts['route_llm']} | {counts['correct']} / {counts['total']} |"
        )
    lines.extend(
        [
            "",
            "## Macro average",
            "",
            "| Method | GNN-only recall | LLM-only recall | Decisive accuracy | AUROC | AP |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        value = macro[method]
        lines.append(
            f"| {method} | {_fmt(value['gnn_only_recall'])} | {_fmt(value['llm_only_recall'])} | {_fmt(value['decisive_accuracy'])} | {_fmt(value['auroc'])} | {_fmt(value['average_precision'])} |"
        )
    lines.extend(
        [
            "",
            "## Balanced validation operating point (locked test)",
            "",
            "| Dataset | Method | Threshold | GNN-only recall | LLM-only recall | Decisive accuracy |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        for method in ("A", "B", "C"):
            value = metrics[dataset][method]["balanced_operating_point"]
            lines.append(
                f"| {dataset} | {method} | {_fmt(value['threshold'])} | {_fmt(value['gnn_only_recall'])} | {_fmt(value['llm_only_recall'])} | {_fmt(value['decisive_accuracy'])} |"
            )
    lines.extend(
        [
            "",
            "## Validation-to-test decisive-accuracy gap",
            "",
            "| Dataset | Method | Validation | Test | Test minus validation |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        if row["method"] == "v6_scheme_b":
            continue
        lines.append(
            f"| {row['dataset']} | {row['method']} | {_fmt(row['validation_accuracy'])} | {_fmt(row['decisive_accuracy'])} | {_fmt(row['validation_to_test_gap'])} |"
        )
    lines.extend(
        [
            "",
            "## Paired decisive-accuracy differences (95% stratified bootstrap CI)",
            "",
            "| Dataset | Contrast | Difference | 95% CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        for contrast in (
            "B_minus_A",
            "C_minus_A",
            "C_minus_B",
            "C_minus_v6_scheme_b",
        ):
            value = paired[dataset][contrast]["decisive_accuracy"]
            lower, upper = value["ci95"]
            lines.append(
                f"| {dataset} | {contrast} | {_fmt(value['difference'])} | [{_fmt(lower)}, {_fmt(upper)}] |"
            )

    lines.extend(
        [
            "",
            "## Frozen artifact audit",
            "",
            "| Artifact | SHA256 |",
            "|---|---|",
        ]
    )
    for family in ("A", "B", "C"):
        lines.append(
            f"| {family} frozen manifest | `{frozen_audit[family]['sha256']}` |"
        )
    lines.extend(
        [
            f"| Locked metrics | `{complete['metrics_sha256']}` |",
            f"| Locked predictions | `{complete['predictions_sha256']}` |",
            f"| Locked evaluator code | `{complete.get('evaluator_code_sha256', 'not recorded')}` |",
            f"| Report generator code | `{_hash(Path(__file__).resolve())}` |",
            "",
            "Full endpoint selection/head/feature-bank hashes and selected specs are stored per row in summary.json.",
        ]
    )
    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- Thresholds are selected only on each target's official validation decisive conflicts.",
            "- Projectors use no correctness labels and exclude every target test ID.",
            "- Source supervision uses corrected OOF support predictions; test labels are accessed only by the locked evaluator.",
            "- Because five-shot decisive support can be as small as three nodes, conclusions about cross-episode generalization must be separated from this fixed-benchmark node-inference result.",
            "",
            "Machine-readable single-method and paired bootstrap intervals, counts, thresholds, selected specs, and artifact hashes are in summary.json.",
        ]
    )
    report = "\n".join(lines) + "\n"
    report_path = run_root / "REPORT.md"
    temporary_report = report_path.with_suffix(".md.tmp")
    temporary_report.write_text(report, encoding="utf-8")
    os.replace(temporary_report, report_path)
    docs_path = ROOT / "docs/router_v7_scheme_b_embeddings_method_and_results_2026-08-27.md"
    temporary_docs = docs_path.with_suffix(".md.tmp")
    temporary_docs.write_text(report, encoding="utf-8")
    os.replace(temporary_docs, docs_path)
    print(f"Wrote {summary_path}, {csv_path}, {report_path}, and {docs_path}")


if __name__ == "__main__":
    main()
