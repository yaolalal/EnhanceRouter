from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import torch

from .calibration import (
    ValidationSplit,
    deterministic_calibration_threshold_split,
    fit_identity_shrunk_platt,
    select_precision_threshold,
)
from .artifacts import (
    atomic_write_json,
    atomic_write_text,
    build_manifest,
    prepare_output_root,
    verify_manifest,
)
from .evaluation import evaluate_frozen_thresholds, probability_report


def _validate_split(split: ValidationSplit, count: int) -> None:
    calibration = torch.as_tensor(split.calibration, dtype=torch.bool).view(-1)
    threshold = torch.as_tensor(split.threshold, dtype=torch.bool).view(-1)
    if calibration.numel() != count or threshold.numel() != count:
        raise ValueError("validation split and scores must align")
    if bool((calibration & threshold).any()) or not bool(
        (calibration | threshold).all()
    ):
        raise ValueError("validation halves must be disjoint and exhaustive")
    if not bool(calibration.any()) or not bool(threshold.any()):
        raise ValueError("both validation halves must be non-empty")


def _thresholds(
    score: torch.Tensor,
    correct: torch.Tensor,
    precision_targets: tuple[float, ...],
    min_accepted: int,
) -> dict:
    return {
        f"precision_{precision:.2f}": select_precision_threshold(
            score,
            correct,
            target_precision=precision,
            min_accepted=min_accepted,
        )
        for precision in precision_targets
    }


def _variant(
    test_score: torch.Tensor,
    test_correct: torch.Tensor,
    selections: dict,
    *,
    calibrator: dict,
) -> dict:
    return {
        "calibrator": calibrator,
        "validation_thresholds": {
            name: value.to_dict() for name, value in selections.items()
        },
        "test_thresholds": evaluate_frozen_thresholds(
            test_score, test_correct, selections
        ),
        "test_metrics": probability_report(
            test_score, test_correct, include_curve=False
        ),
        "_test_score": test_score,
    }


def evaluate_expert_confidence_variants(
    validation_score: torch.Tensor,
    validation_correct: torch.Tensor,
    split: ValidationSplit,
    test_score: torch.Tensor,
    test_correct: torch.Tensor,
    *,
    precision_targets: tuple[float, ...] = (0.80, 0.85, 0.90),
    min_accepted: int = 10,
    calibration_regularization: float = 0.1,
) -> dict[str, dict]:
    """Evaluate raw and equal-budget Platt-calibrated expert confidence."""

    validation_score = torch.as_tensor(validation_score, dtype=torch.float64).view(-1)
    validation_correct = torch.as_tensor(
        validation_correct, dtype=torch.bool
    ).view(-1)
    test_score = torch.as_tensor(test_score, dtype=torch.float64).view(-1)
    test_correct = torch.as_tensor(test_correct, dtype=torch.bool).view(-1)
    if validation_score.numel() != validation_correct.numel():
        raise ValueError("validation scores and labels must align")
    if test_score.numel() != test_correct.numel():
        raise ValueError("test scores and labels must align")
    for name, score in (("validation", validation_score), ("test", test_score)):
        if not bool(torch.isfinite(score).all()) or bool(
            ((score < 0) | (score > 1)).any()
        ):
            raise ValueError(f"{name} expert confidence must lie in [0, 1]")
    _validate_split(split, validation_score.numel())

    raw_selections = _thresholds(
        validation_score[split.threshold],
        validation_correct[split.threshold],
        precision_targets,
        min_accepted,
    )
    raw = _variant(
        test_score,
        test_correct,
        raw_selections,
        calibrator={"type": "identity", "count": 0},
    )

    validation_logit = torch.logit(validation_score.clamp(1e-7, 1 - 1e-7))
    calibrator = fit_identity_shrunk_platt(
        validation_logit[split.calibration],
        validation_correct[split.calibration],
        regularization=calibration_regularization,
    )
    calibrated_validation = calibrator.probabilities(validation_score)
    calibrated_test = calibrator.probabilities(test_score)
    calibrated_selections = _thresholds(
        calibrated_validation[split.threshold],
        validation_correct[split.threshold],
        precision_targets,
        min_accepted,
    )
    calibrated = _variant(
        calibrated_test,
        test_correct,
        calibrated_selections,
        calibrator={"type": "identity_shrunk_platt", **calibrator.to_dict()},
    )
    return {
        "raw_expert_confidence": raw,
        "calibrated_expert_confidence": calibrated,
    }


VARIANTS = (
    "split_calibrated",
    "raw_expert_confidence",
    "calibrated_expert_confidence",
)
VARIANT_LABELS = {
    "split_calibrated": "Split-C",
    "raw_expert_confidence": "Raw expert confidence",
    "calibrated_expert_confidence": "Calibrated expert confidence",
}
DATASET_LABELS = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
    "arxiv": "Arxiv",
    "ogbn-products": "OGBN-Products",
}


def _csv_text(rows: list[dict]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _summary_rows(experiment: dict) -> list[dict]:
    rows = []
    for dataset, dataset_value in experiment["datasets"].items():
        for module, module_value in dataset_value["modules"].items():
            variants = module_value["variants"]
            c = variants["split_calibrated"]["test_metrics"]
            raw = variants["raw_expert_confidence"]["test_metrics"]
            calibrated = variants["calibrated_expert_confidence"]["test_metrics"]
            rows.append(
                {
                    "dataset": dataset,
                    "module": module,
                    "count": module_value["count"],
                    "base_accuracy": module_value["base_accuracy"],
                    "split_c_auroc": c["auroc"],
                    "raw_ec_auroc": raw["auroc"],
                    "raw_minus_c_auroc": raw["auroc"] - c["auroc"],
                    "calibrated_ec_auroc": calibrated["auroc"],
                    "calibrated_minus_c_auroc": calibrated["auroc"] - c["auroc"],
                    "split_c_ap": c["average_precision"],
                    "raw_ec_ap": raw["average_precision"],
                    "calibrated_ec_ap": calibrated["average_precision"],
                    "split_c_aurc": c["aurc"],
                    "raw_ec_aurc": raw["aurc"],
                    "raw_minus_c_aurc": raw["aurc"] - c["aurc"],
                    "calibrated_ec_aurc": calibrated["aurc"],
                    "calibrated_minus_c_aurc": calibrated["aurc"] - c["aurc"],
                    "calibrated_ec_scale": variants[
                        "calibrated_expert_confidence"
                    ]["calibrator"].get("scale"),
                    "calibrated_ec_bias": variants[
                        "calibrated_expert_confidence"
                    ]["calibrator"].get("bias"),
                }
            )
    return rows


def _threshold_rows(experiment: dict) -> list[dict]:
    rows = []
    for dataset, dataset_value in experiment["datasets"].items():
        for module, module_value in dataset_value["modules"].items():
            for variant in VARIANTS:
                value = module_value["variants"][variant]
                for name, validation in value["validation_thresholds"].items():
                    test = value["test_thresholds"][name]
                    rows.append(
                        {
                            "dataset": dataset,
                            "module": module,
                            "variant": variant,
                            "target_precision": validation["target_precision"],
                            "threshold": validation["threshold"],
                            "reject_all": validation["reject_all"],
                            "validation_accepted": validation["accepted_count"],
                            "validation_count": validation["validation_count"],
                            "validation_coverage": validation["coverage"],
                            "validation_precision": validation["empirical_precision"],
                            "validation_wilson_lower": validation[
                                "wilson_lower_bound"
                            ],
                            "test_accepted": test["accepted_count"],
                            "test_count": test["count"],
                            "test_coverage": test["coverage"],
                            "test_precision": test["precision"],
                            "test_risk": test["risk"],
                        }
                    )
    return rows


def _fmt(value) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def _report(summary: list[dict], thresholds: list[dict]) -> str:
    raw_wins = sum(row["raw_minus_c_auroc"] > 0 for row in summary)
    calibrated_wins = sum(row["calibrated_minus_c_auroc"] > 0 for row in summary)
    lines = [
        "# Expert Confidence as a Scheme-C Replacement",
        "",
        "Raw expert confidence and equal-budget Platt-calibrated expert confidence are compared with Split-C independently within conflict and agreement populations.",
        "",
        f"Raw expert confidence has higher test AUROC than Split-C in {raw_wins}/10 dataset-module populations; calibrated expert confidence does so in {calibrated_wins}/10.",
        "",
        "## Test ranking metrics",
        "",
        "| Dataset | Module | N | Split-C AUROC | Raw EC AUROC | Cal EC AUROC | Split-C AURC | Raw EC AURC | Cal EC AURC | Platt scale |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {row['module']} | {row['count']} | {_fmt(row['split_c_auroc'])} | {_fmt(row['raw_ec_auroc'])} | {_fmt(row['calibrated_ec_auroc'])} | {_fmt(row['split_c_aurc'])} | {_fmt(row['raw_ec_aurc'])} | {_fmt(row['calibrated_ec_aurc'])} | {_fmt(row['calibrated_ec_scale'])} |"
        )
    lines.extend(
        [
            "",
            "## Validation-selected P90 operating points on test",
            "",
            "| Dataset | Module | Score | Reject all | Validation coverage | Validation precision | Test coverage | Test precision | Test risk |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in thresholds:
        if float(row["target_precision"]) != 0.90:
            continue
        lines.append(
            f"| {row['dataset']} | {row['module']} | {VARIANT_LABELS[row['variant']]} | {row['reject_all']} | {_fmt(row['validation_coverage'])} | {_fmt(row['validation_precision'])} | {_fmt(row['test_coverage'])} | {_fmt(row['test_precision'])} | {_fmt(row['test_risk'])} |"
        )
    lines.extend(
        [
            "",
            "Thresholds are selected only on the frozen validation-threshold half. Test labels are used only for final evaluation. Reject-all has undefined precision and risk, not zero risk.",
            "",
        ]
    )
    return "\n".join(lines)


def _plot_risk_coverage(
    experiment: dict, curves: dict, output_root: Path
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "split_calibrated": "#3569b8",
        "raw_expert_confidence": "#df8500",
        "calibrated_expert_confidence": "#8b4ba8",
    }
    styles = {
        "split_calibrated": "-",
        "raw_expert_confidence": "-",
        "calibrated_expert_confidence": "--",
    }
    markers = {
        "precision_0.80": "o",
        "precision_0.85": "s",
        "precision_0.90": "^",
    }
    datasets = tuple(experiment["datasets"])
    modules = (
        ("conflict", "Disagreement / C_conflict"),
        ("agreement", "Agreement / C_agreement"),
    )
    figure, axes = plt.subplots(
        len(datasets), 2, figsize=(18, 23), sharex=True, squeeze=False
    )
    for row_index, dataset in enumerate(datasets):
        for column_index, (module, module_label) in enumerate(modules):
            axis = axes[row_index, column_index]
            maximum_risk = 0.0
            module_result = experiment["datasets"][dataset]["modules"][module]
            for variant in VARIANTS:
                curve = curves[dataset][module][variant]
                coverage = [float(row["coverage"]) for row in curve]
                risk = [float(row["risk"]) for row in curve]
                maximum_risk = max(maximum_risk, max(risk))
                axis.plot(
                    coverage,
                    risk,
                    color=colors[variant],
                    linestyle=styles[variant],
                    linewidth=2.0,
                    label=VARIANT_LABELS[variant],
                )
                for name, marker in markers.items():
                    point = module_result["variants"][variant]["test_thresholds"][name]
                    if not point["reject_all"] and point["risk"] is not None:
                        maximum_risk = max(maximum_risk, float(point["risk"]))
                        axis.scatter(
                            [point["coverage"]],
                            [point["risk"]],
                            marker=marker,
                            s=45,
                            color=colors[variant],
                            edgecolor="white",
                            linewidth=0.7,
                            zorder=4,
                        )
            scale = module_result["variants"]["calibrated_expert_confidence"][
                "calibrator"
            ].get("scale")
            axis.set_title(
                f"{DATASET_LABELS.get(dataset, dataset)} — {module_label}\n"
                f"n={module_result['count']:,}, EC Platt scale={_fmt(scale)}",
                fontsize=13,
            )
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, min(1.0, max(0.10, maximum_risk + 0.05)))
            axis.set_ylabel("Selective risk (1 - precision)")
            axis.grid(alpha=0.2)
            axis.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
            axis.text(
                0.99,
                0.02,
                "markers: ○ P80   □ P85   △ P90",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="#444444",
            )
            if row_index == len(datasets) - 1:
                axis.set_xlabel("Coverage")
    figure.suptitle(
        "Split-C vs. Expert Confidence — Test Risk–Coverage\n"
        "Validation-selected P80/P85/P90 operating points are overlaid",
        fontsize=18,
        y=0.998,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.985), h_pad=2.0, w_pad=1.3)
    png = output_root / "expert_confidence_risk_coverage_10panel.png"
    pdf = output_root / "expert_confidence_risk_coverage_10panel.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def write_expert_confidence_outputs(
    experiment: dict, curves: dict, output_root: str | Path
) -> dict:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = _summary_rows(experiment)
    thresholds = _threshold_rows(experiment)
    metrics_path = output_root / "metrics.json"
    curves_path = output_root / "curves.json"
    summary_path = output_root / "summary.csv"
    threshold_path = output_root / "threshold_operating_points.csv"
    report_path = output_root / "REPORT.md"
    atomic_write_json(experiment, metrics_path)
    atomic_write_json(curves, curves_path)
    atomic_write_text(_csv_text(summary), summary_path)
    atomic_write_text(_csv_text(thresholds), threshold_path)
    atomic_write_text(_report(summary, thresholds), report_path)
    png, pdf = _plot_risk_coverage(experiment, curves, output_root)
    manifest_path = output_root / "manifest.json"
    atomic_write_json(build_manifest(output_root), manifest_path)
    return {
        "summary_csv": str(summary_path),
        "threshold_csv": str(threshold_path),
        "metrics_json": str(metrics_path),
        "curves_json": str(curves_path),
        "report": str(report_path),
        "risk_coverage_png": str(png),
        "risk_coverage_pdf": str(pdf),
        "manifest": str(manifest_path),
    }


def run_expert_confidence_experiment(
    input_root: str | Path, output_root: str | Path
) -> dict:
    from eargtc.router_bc.data_impl import build_support_domain

    from .b_replay import replay_selected_b
    from .pipeline import (
        DATASETS,
        MODULES,
        _compact_curve,
        _heuristic_probability,
        _query_domain,
    )

    input_root, output_root = Path(input_root), Path(output_root)
    source_audit = verify_manifest(input_root)
    if not source_audit["all_checks_pass"]:
        raise RuntimeError(f"input run manifest verification failed: {source_audit}")
    source_config = json.loads(
        (input_root / "config.json").read_text(encoding="utf-8")
    )
    arguments = source_config["arguments"]
    destination = prepare_output_root(output_root, resume=False)
    input_run_root = Path(arguments["input_run_root"])
    frozen_b_root = Path(arguments["frozen_b_root"])
    dataset_root = Path(arguments["dataset_root"])
    precision_targets = tuple(
        float(value) for value in arguments["precision_targets"]
    )
    min_accepted = int(arguments["min_accepted"])
    regularization = float(arguments["calibration_regularization"])
    seed = int(arguments["seed"])
    supports = {
        dataset: build_support_domain(
            dataset,
            input_run_root
            / dataset
            / "pipeline/router/oof/router_train.jsonl",
        )
        for dataset in DATASETS
    }
    experiment = {
        "schema_version": 1,
        "protocol": {
            "baseline": "split_calibrated_reliability",
            "raw_expert_confidence": "selected expert top-1 probability for conflict; mean common-label probability for agreement",
            "calibrated_expert_confidence": "identity-shrunk Platt calibration fitted only on the original calibration half",
            "threshold_selection": "maximum coverage meeting empirical P80/P85/P90 on the original disjoint validation-threshold half",
            "score_populations": "conflict and agreement ranked independently",
            "test_labels_used_for": "final metrics only",
            "precision_targets": list(precision_targets),
            "min_accepted": min_accepted,
            "calibration_regularization": regularization,
        },
        "source": {
            "input_root": str(input_root.resolve()),
            "manifest_audit": source_audit,
            "arguments": arguments,
        },
        "datasets": {},
    }
    curves = {}
    for dataset in DATASETS:
        replay, target, b_audit = replay_selected_b(
            dataset=dataset,
            supports=supports,
            input_run_root=input_run_root,
            frozen_b_root=frozen_b_root,
            dataset_root=dataset_root,
            seed=seed,
        )
        if not b_audit["all_checks_pass"]:
            raise RuntimeError(f"frozen-B replay failed for {dataset}: {b_audit}")
        stored = json.loads(
            (input_root / dataset / "metrics.json").read_text(encoding="utf-8")
        )
        experiment["datasets"][dataset] = {
            "scheme_b_identity": b_audit,
            "modules": {},
        }
        curves[dataset] = {}
        raw_view = {"raw": (target.x, target.feature_names)}
        for module in MODULES:
            validation = _query_domain(
                target,
                replay,
                target.val_ids,
                view="raw",
                module=module,
                view_values=raw_view,
            )
            test = _query_domain(
                target,
                replay,
                target.test_ids,
                view="raw",
                module=module,
                view_values=raw_view,
            )
            split = deterministic_calibration_threshold_split(
                validation.node_ids, dataset, module
            )
            validation_heuristic = _heuristic_probability(
                target, replay, validation
            )
            test_heuristic = _heuristic_probability(target, replay, test)
            decision_rows = [
                json.loads(line)
                for line in (
                    input_root / dataset / f"decisions_{module}.jsonl"
                )
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            by_id = {int(row["node_id"]): row for row in decision_rows}
            if set(by_id) != set(test.node_ids.tolist()):
                raise AssertionError(f"decision IDs differ for {dataset}/{module}")
            split_c_score = torch.tensor(
                [
                    float(
                        by_id[int(node_id)]["split_calibrated_reliability"]
                    )
                    for node_id in test.node_ids.tolist()
                ],
                dtype=torch.float64,
            )
            stored_heuristic = torch.tensor(
                [
                    float(by_id[int(node_id)]["expert_confidence_heuristic"])
                    for node_id in test.node_ids.tolist()
                ],
                dtype=torch.float64,
            )
            stored_correct = torch.tensor(
                [
                    bool(by_id[int(node_id)]["correct"])
                    for node_id in test.node_ids.tolist()
                ],
                dtype=torch.bool,
            )
            if not torch.equal(stored_correct, test.target):
                raise AssertionError(
                    f"decision correctness differs for {dataset}/{module}"
                )
            heuristic_difference = float(
                (stored_heuristic - test_heuristic.to(torch.float64))
                .abs()
                .max()
            )
            if heuristic_difference > 1e-6:
                raise AssertionError(
                    f"expert-confidence replay drift for {dataset}/{module}: "
                    f"{heuristic_difference}"
                )
            variants = evaluate_expert_confidence_variants(
                validation_heuristic,
                validation.target,
                split,
                test_heuristic,
                test.target,
                precision_targets=precision_targets,
                min_accepted=min_accepted,
                calibration_regularization=regularization,
            )
            current = stored["modules"][module]
            split_c_metrics = probability_report(
                split_c_score, test.target, include_curve=False
            )
            expected = current["comparators"]["split_calibrated"]
            for metric in ("auroc", "average_precision", "aurc"):
                if (
                    abs(float(split_c_metrics[metric]) - float(expected[metric]))
                    > 1e-8
                ):
                    raise AssertionError(
                        f"Split-C metric drift for {dataset}/{module}/{metric}"
                    )
            c_calibration = current["calibration"]
            variants = {
                "split_calibrated": {
                    "calibrator": {
                        "type": "identity_shrunk_platt",
                        "scale": c_calibration["scale"],
                        "bias": c_calibration["bias"],
                        "regularization": c_calibration["regularization"],
                        "count": c_calibration["count"],
                    },
                    "validation_thresholds": current["thresholds"],
                    "test_thresholds": current["test_thresholds"],
                    "test_metrics": split_c_metrics,
                    "_test_score": split_c_score,
                },
                **variants,
            }
            curves[dataset][module] = {
                name: _compact_curve(value["_test_score"], test.target)
                for name, value in variants.items()
            }
            serializable_variants = {
                name: {
                    key: item
                    for key, item in value.items()
                    if key != "_test_score"
                }
                for name, value in variants.items()
            }
            experiment["datasets"][dataset]["modules"][module] = {
                "count": int(test.target.numel()),
                "base_accuracy": float(test.target.to(torch.float64).mean()),
                "validation_count": int(validation.target.numel()),
                "calibration_count": int(split.calibration.sum()),
                "threshold_count": int(split.threshold.sum()),
                "expert_confidence_replay_max_abs_diff": heuristic_difference,
                "variants": serializable_variants,
            }
    paths = write_expert_confidence_outputs(experiment, curves, destination)
    return {
        "output_root": str(destination),
        "dataset_count": len(experiment["datasets"]),
        "panel_count": sum(
            len(value["modules"])
            for value in experiment["datasets"].values()
        ),
        "source_manifest_ok": source_audit["all_checks_pass"],
        **paths,
    }


__all__ = [
    "evaluate_expert_confidence_variants",
    "run_expert_confidence_experiment",
    "write_expert_confidence_outputs",
]
