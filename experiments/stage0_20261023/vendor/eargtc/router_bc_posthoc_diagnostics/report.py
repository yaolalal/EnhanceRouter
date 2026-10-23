from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from collections.abc import Mapping, Sequence

import numpy as np

from eargtc.router_bc_supplementary.analysis import centered_logit_margin

from .analysis import (
    STATE_NAMES,
    classify_states,
    oracle_gap_decomposition,
    rank_biserial,
    rank_segments,
)
from .density import bounded_density
from .evidence import join_evidence, reconstruct_test_evidence
from .plotting import (
    DISPLAY_NAMES,
    plot_b_density,
    plot_c_density,
    plot_loss_effects,
    plot_rank_composition,
)


DATASETS = ("cora", "citeseer", "pubmed", "arxiv", "ogbn-products")
TOLERANCE = 2e-6


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid_jsonl:{path}:{line_number}") from error
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _stats(values) -> dict:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0, "mean": None, "std": None, "median": None,
            "q1": None, "q3": None,
        }
    q1, median, q3 = np.quantile(array, [0.25, 0.50, 0.75])
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
    }


def _stored_precision_at_25(metrics: Mapping) -> float | None:
    value = (
        metrics.get("scheme_c", {})
        .get("metrics", {})
        .get("selected", {})
        .get("precision_at_coverage", {})
        .get("0.25")
    )
    return None if value is None else float(value)


def _compare_saved(dataset: str, name: str, actual, expected) -> None:
    if expected is None:
        return
    if isinstance(expected, int):
        if int(actual) != int(expected):
            raise ValueError(f"{dataset}:{name}_mismatch:{actual}:{expected}")
    elif not math.isclose(float(actual), float(expected), abs_tol=TOLERANCE, rel_tol=0):
        raise ValueError(f"{dataset}:{name}_mismatch:{actual}:{expected}")


def build_dataset_analysis(
    dataset: str,
    rows: Sequence[Mapping],
    metrics: Mapping,
    evidence_by_id: Mapping[int, Mapping[str, float]],
) -> dict:
    states = classify_states(rows)
    segments, order, segment_assignment = rank_segments(rows, states)
    decomposition, gap_masks = oracle_gap_decomposition(rows)
    node_ids = np.asarray([int(row["node_id"]) for row in rows], dtype=np.int64)
    true = np.asarray([int(row["true_label"]) for row in rows], dtype=int)
    gnn = np.asarray([int(row["gnn_pred"]) for row in rows], dtype=int)
    llm = np.asarray([int(row["llm_pred"]) for row in rows], dtype=int)
    selected = np.asarray([int(row["selected_pred"]) for row in rows], dtype=int)
    trust = np.asarray([float(row["selected_trust"]) for row in rows])
    preference = np.asarray([float(row["preference_score"]) for row in rows])
    selected_correct = selected == true
    threshold = float(metrics["scheme_b"]["test"]["threshold"])
    margins = centered_logit_margin(preference, threshold)
    top25_count = int(math.ceil(0.25 * len(rows)))
    top25 = order[:top25_count]
    top25_precision = float(np.mean(selected_correct[top25]))
    top25_cutoff = float(trust[order[top25_count - 1]])

    stored_b = metrics.get("scheme_b", {}).get("test", {})
    stored_pair = stored_b.get("paired_vs_best_single", {})
    _compare_saved(dataset, "router_accuracy", decomposition["router_accuracy"], stored_b.get("router_accuracy"))
    _compare_saved(dataset, "oracle_accuracy", decomposition["oracle_accuracy"], stored_b.get("oracle_accuracy"))
    _compare_saved(dataset, "wins", decomposition["wins"], stored_pair.get("wins"))
    _compare_saved(dataset, "losses", decomposition["losses"], stored_pair.get("losses"))
    _compare_saved(dataset, "precision_at_25", top25_precision, _stored_precision_at_25(metrics))

    edges = np.linspace(0.0, 1.0, 31)
    grid = np.linspace(0.0, 1.0, 301)
    b_density = {
        "gnn_only": bounded_density(preference[states["gnn_only"]], bin_edges=edges, grid=grid),
        "llm_only": bounded_density(preference[states["llm_only"]], bin_edges=edges, grid=grid),
    }
    c_density = {
        "correct": bounded_density(trust[selected_correct], bin_edges=edges, grid=grid),
        "wrong": bounded_density(trust[~selected_correct], bin_edges=edges, grid=grid),
    }

    evidence_names = sorted(next(iter(evidence_by_id.values())).keys())
    evidence = {
        name: np.asarray([float(evidence_by_id[int(node_id)][name]) for node_id in node_ids])
        for name in evidence_names
    }
    evidence.update(
        {
            "preference_score": preference,
            "decision_margin": margins,
            "selected_trust": trust,
        }
    )
    comparisons = (
        ("loss_vs_protected", gap_masks["loss"], gap_masks["protected"]),
        (
            "missed_vs_recovered",
            gap_masks["missed_recovery"],
            gap_masks["recovered_other_only"],
        ),
    )
    effects = []
    for comparison, left_mask, right_mask in comparisons:
        for feature, values in evidence.items():
            left = values[left_mask]
            right = values[right_mask]
            left_stats = _stats(left)
            right_stats = _stats(right)
            effects.append(
                {
                    "dataset": dataset,
                    "comparison": comparison,
                    "feature": feature,
                    "rank_biserial": rank_biserial(left, right),
                    "loss_count": int(np.sum(left_mask)),
                    "reference_count": int(np.sum(right_mask)),
                    **{f"left_{key}": value for key, value in left_stats.items()},
                    **{f"right_{key}": value for key, value in right_stats.items()},
                }
            )

    gap_nodes = []
    for index, node_id in enumerate(node_ids):
        kind = (
            "loss" if gap_masks["loss"][index]
            else "missed_recovery" if gap_masks["missed_recovery"][index]
            else None
        )
        if kind is None:
            continue
        state = next(name for name in STATE_NAMES if states[name][index])
        gap_nodes.append(
            {
                "dataset": dataset,
                "node_id": int(node_id),
                "oracle_gap_kind": kind,
                "state": state,
                "true_label": int(true[index]),
                "gnn_pred": int(gnn[index]),
                "llm_pred": int(llm[index]),
                "selected_pred": int(selected[index]),
                "route": rows[index]["route"],
                "preference_score": float(preference[index]),
                "decision_margin": float(margins[index]),
                "selected_trust": float(trust[index]),
                "trust_segment": segments[int(segment_assignment[index])]["segment"],
                **evidence_by_id[int(node_id)],
            }
        )

    class_rows = []
    comparator_correct = (gnn == true) if decomposition["comparator"] == "gnn" else (llm == true)
    other_correct = (llm == true) if decomposition["comparator"] == "gnn" else (gnn == true)
    comparator_only = comparator_correct & ~other_correct
    other_only = other_correct & ~comparator_correct
    for label in sorted(np.unique(true).tolist()):
        mask = true == label
        comparator_count = int(np.sum(mask & comparator_only))
        other_count = int(np.sum(mask & other_only))
        loss_count = int(np.sum(mask & gap_masks["loss"]))
        missed_count = int(np.sum(mask & gap_masks["missed_recovery"]))
        class_rows.append(
            {
                "dataset": dataset,
                "true_label": int(label),
                "count": int(np.sum(mask)),
                "comparator_only": comparator_count,
                "losses": loss_count,
                "loss_rate_within_comparator_only": (
                    loss_count / comparator_count if comparator_count else None
                ),
                "other_only": other_count,
                "missed_recoveries": missed_count,
                "missed_rate_within_other_only": (
                    missed_count / other_count if other_count else None
                ),
            }
        )

    return {
        "dataset": dataset,
        "routing_threshold": threshold,
        "top25_count": top25_count,
        "top25_cutoff": top25_cutoff,
        "top25_precision": top25_precision,
        "states": {name: int(np.sum(mask)) for name, mask in states.items()},
        "segments": segments,
        "decomposition": decomposition,
        "b_density": b_density,
        "c_density": c_density,
        "loss_effects": effects,
        "gap_nodes": gap_nodes,
        "class_rows": class_rows,
    }


def _density_rows(analyses: Sequence[dict], kind: str) -> list[dict]:
    output = []
    groups = ("gnn_only", "llm_only") if kind == "b" else ("correct", "wrong")
    for analysis in analyses:
        densities = analysis["b_density"] if kind == "b" else analysis["c_density"]
        for group in groups:
            density = densities[group]
            for x, value in zip(density["grid"], density["kde_density"]):
                output.append(
                    {
                        "dataset": analysis["dataset"], "group": group,
                        "component": "kde", "x": float(x), "density": float(value),
                        "count": density["count"], "median": density["median"],
                        "bandwidth": density["bandwidth"],
                    }
                )
            edges = density["bin_edges"]
            for index, value in enumerate(density["hist_density"]):
                output.append(
                    {
                        "dataset": analysis["dataset"], "group": group,
                        "component": "histogram",
                        "bin_left": float(edges[index]),
                        "bin_right": float(edges[index + 1]),
                        "density": float(value), "count": density["count"],
                        "median": density["median"], "bandwidth": density["bandwidth"],
                    }
                )
    return output


def _display(dataset: str) -> str:
    return DISPLAY_NAMES.get(dataset, dataset)


def _render_report(analyses: Sequence[dict], figure_prefix: str) -> str:
    decomposition_lines = [
        "| Dataset | Comparator | Wins | Losses | Missed recoveries | Oracle-gap nodes | Loss share |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    segment_lines = [
        "| Dataset | Segment | N | Precision | Correct agr. | Wrong agr. | GNN-only | LLM-only | Conflict neither |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    findings = []
    figure_lines = []
    for analysis in analyses:
        dataset = analysis["dataset"]
        d = analysis["decomposition"]
        decomposition_lines.append(
            f"| {_display(dataset)} | {d['comparator'].upper()} | {d['wins']:,} "
            f"| {d['losses']:,} | {d['missed_recoveries']:,} "
            f"| {d['oracle_gap_count']:,} "
            f"| {d['losses'] / d['oracle_gap_count']:.1%} |"
        )
        for row in analysis["segments"]:
            segment_lines.append(
                f"| {_display(dataset)} | {row['segment']} | {row['count']:,} "
                f"| {row['precision']:.4f} "
                + "".join(
                    f"| {row[f'{name}_proportion']:.1%} " for name in STATE_NAMES
                )
                + "|"
            )
        top_segments = analysis["segments"][:2]
        correct_by_state = {
            name: sum(row[f"{name}_selected_correct_count"] for row in top_segments)
            for name in STATE_NAMES
        }
        top_correct = sum(correct_by_state.values())
        agreement_share = correct_by_state["correct_agreement"] / top_correct
        decisive_correct = correct_by_state["gnn_only"] + correct_by_state["llm_only"]
        strongest = sorted(
            [
                row for row in analysis["loss_effects"]
                if row["comparison"] == "loss_vs_protected"
                and row["rank_biserial"] is not None
            ],
            key=lambda row: abs(row["rank_biserial"]),
            reverse=True,
        )[:3]
        effect_text = ", ".join(
            f"`{row['feature']}` ({row['rank_biserial']:+.3f})" for row in strongest
        )
        findings.extend(
            [
                f"### {_display(dataset)}",
                "",
                f"- Oracle gap 共 `{d['oracle_gap_count']:,}` 个节点：loss "
                f"`{d['losses']:,}`，missed recovery `{d['missed_recoveries']:,}`。",
                f"- Top-25% precision 为 `{analysis['top25_precision']:.4f}`；其中 "
                f"correct-agreement 提供 `{agreement_share:.1%}` 的正确节点，"
                f"GNN-only/LLM-only 的正确路由合计 `{decisive_correct:,}` 个。"
                + (
                    " 因此高 Precision@25% 并非只由 agreement 造成。"
                    if decisive_correct > 0
                    else " 该数据集的高 Precision@25% 确实完全来自 correct agreement。"
                ),
                f"- Loss 相对 protected best-expert-only 的最大绝对秩效应：{effect_text}。",
                "",
            ]
        )
        for category, title in (
            ("scheme_b_density", "Scheme B GNN-only / LLM-only density"),
            ("scheme_c_density", "Scheme C correct / wrong density"),
            ("scheme_c_rank_composition", "Scheme C ranked composition"),
            ("loss_feature_effects", "Loss-node feature effects"),
        ):
            figure_lines.extend(
                [
                    f"### {_display(dataset)} — {title}",
                    "",
                    f"![{title}]({figure_prefix}/{category}/{dataset}.png)",
                    "",
                ]
            )
    return f"""# Router B/C Frozen Node Diagnostics

> Frozen post-selection analysis; no retraining, checkpoint selection, calibration fitting, or threshold selection.

## 1. 口径

Scheme B 图为 GNN-only 与 LLM-only 的归一化概率密度直方图和有界 KDE，
不是累积分布。Scheme C 图同样展示 selected-correct 与 selected-wrong 的概率密度。

Oracle gap 不仅来自 losses：

```text
oracle_gap_count = losses + missed_recoveries
```

Scheme C 按 trust 降序、node ID 升序打破 ties，并按
0–10%、10–25%、25–50%、50–75%、75–100% 分段。

## 2. Oracle-gap decomposition

{chr(10).join(decomposition_lines)}

## 3. Scheme C 排名区间组成

{chr(10).join(segment_lines)}

## 4. 数据集结论

{chr(10).join(findings)}

## 5. 每数据集图

{chr(10).join(figure_lines)}

## 6. 限制

这是使用 test labels 的 retrospective descriptive analysis。图、节点切片和
effect size 不能用于重新选择 Scheme B/C checkpoint、阈值或 evidence。
"""


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_manifest(root: Path) -> None:
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.sha256"
    )
    (root / "manifest.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(root)}\n" for path in files),
        encoding="utf-8",
    )


def run_diagnostics(
    bc_root: str | Path,
    corrected_root: str | Path,
    output_root: str | Path,
    document_path: str | Path,
) -> dict:
    bc = Path(bc_root).resolve()
    corrected = Path(corrected_root).resolve()
    output = Path(output_root).resolve()
    document = Path(document_path).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"nonempty_output_root:{output}")
    if document.exists() and document.stat().st_size > 0:
        raise FileExistsError(f"existing_document_path:{document}")
    if not bc.is_dir() or not corrected.is_dir():
        raise FileNotFoundError("missing_frozen_input_root")
    config = _read_json(bc / "config.json")
    dataset_root = Path(config["dataset_root"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    input_paths = [bc / "config.json", bc / "audit.json", corrected / "input_manifest.json"]
    analyses = []
    try:
        for dataset in DATASETS:
            decisions_path = bc / dataset / "decisions.jsonl"
            metrics_path = bc / dataset / "metrics.json"
            pipeline = corrected / dataset / "pipeline"
            split_path = pipeline / "splits" / f"{dataset}_5shot_seed42.json"
            input_paths.extend(
                [
                    decisions_path, metrics_path, pipeline / "predictions.pt",
                    split_path, pipeline / "router" / "oof" / "router_train.jsonl",
                    dataset_root / "datasets" / f"{dataset}.pt",
                ]
            )
            rows = _read_jsonl(decisions_path)
            metrics = _read_json(metrics_path)
            names, matrix, test_ids = reconstruct_test_evidence(
                dataset, pipeline, dataset_root, rows
            )
            evidence = join_evidence(rows, names, matrix, test_ids)
            analysis = build_dataset_analysis(dataset, rows, metrics, evidence)
            analyses.append(analysis)
            plot_b_density(analysis, stage / "figures")
            plot_c_density(analysis, stage / "figures")
            plot_rank_composition(analysis, stage / "figures")
            plot_loss_effects(analysis, stage / "figures")

        before_hashes = {str(path): _sha256(path) for path in input_paths}
        _write_csv(stage / "scheme_b_density.csv", _density_rows(analyses, "b"))
        _write_csv(stage / "scheme_c_density.csv", _density_rows(analyses, "c"))
        _write_csv(
            stage / "oracle_gap_decomposition.csv",
            [{"dataset": item["dataset"], **item["decomposition"]} for item in analyses],
        )
        _write_csv(
            stage / "oracle_gap_nodes.csv",
            [row for item in analyses for row in item["gap_nodes"]],
        )
        _write_csv(
            stage / "loss_feature_summary.csv",
            [row for item in analyses for row in item["loss_effects"]],
        )
        _write_csv(
            stage / "loss_class_summary.csv",
            [row for item in analyses for row in item["class_rows"]],
        )
        _write_csv(
            stage / "scheme_c_rank_segments.csv",
            [
                {"dataset": item["dataset"], **row}
                for item in analyses for row in item["segments"]
            ],
        )
        summaries = [
            {
                "dataset": item["dataset"],
                "routing_threshold": item["routing_threshold"],
                "top25_count": item["top25_count"],
                "top25_cutoff": item["top25_cutoff"],
                "top25_precision": item["top25_precision"],
                "states": item["states"],
                "decomposition": item["decomposition"],
                "segments": item["segments"],
            }
            for item in analyses
        ]
        (stage / "summary.json").write_text(
            json.dumps(_jsonable(summaries), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        after_hashes = {str(path): _sha256(path) for path in input_paths}
        checks = {
            "five_datasets": len(analyses) == 5,
            "source_hashes_unchanged": before_hashes == after_hashes,
            "oracle_identities": all(
                item["decomposition"]["oracle_gap_count"]
                == item["decomposition"]["losses"]
                + item["decomposition"]["missed_recoveries"]
                for item in analyses
            ),
            "segment_count": all(len(item["segments"]) == 5 for item in analyses),
            "segment_partition": all(
                all(
                    math.isclose(
                        sum(row[f"{name}_proportion"] for name in STATE_NAMES),
                        1.0,
                        abs_tol=1e-12,
                    )
                    for row in item["segments"]
                )
                for item in analyses
            ),
            "figure_count": len(list((stage / "figures").rglob("*.png"))) == 20
            and len(list((stage / "figures").rglob("*.pdf"))) == 20,
        }
        audit = {
            "all_checks_pass": all(checks.values()),
            "checks": checks,
            "input_hashes": before_hashes,
            "datasets": [
                {
                    "dataset": item["dataset"],
                    "top25_precision": item["top25_precision"],
                    "oracle_gap_count": item["decomposition"]["oracle_gap_count"],
                    "all_checks_pass": True,
                }
                for item in analyses
            ],
            "protocol": {
                "frozen_post_selection": True,
                "retraining": False,
                "checkpoint_reselection": False,
                "threshold_reselection": False,
                "raw_text_loaded": False,
            },
        }
        if not audit["all_checks_pass"]:
            raise ValueError("posthoc_audit_failed")
        (stage / "audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        internal_report = _render_report(analyses, "figures")
        (stage / "REPORT.md").write_text(internal_report, encoding="utf-8")
        _write_manifest(stage)
        if output.exists():
            output.rmdir()
        os.replace(stage, output)
        document.parent.mkdir(parents=True, exist_ok=True)
        external_prefix = f"../runs/{output.name}/figures"
        document.write_text(_render_report(analyses, external_prefix), encoding="utf-8")
        return audit
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise

