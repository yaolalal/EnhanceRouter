"""End-to-end orchestration for independently trained split Scheme C."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from eargtc.router_bc.data_impl import (
    SupportDomain,
    TargetArtifacts,
    build_support_domain,
)
from eargtc.router_bc.rank import percentile_features
from eargtc.router_bc_improved.runner_utils import validate_selection_record

from .artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    build_manifest,
    prepare_output_root,
    sha256_file,
    verify_manifest,
)
from .b_replay import FrozenBReplay, replay_selected_b
from .calibration import (
    deterministic_calibration_threshold_split,
    fit_identity_shrunk_platt,
    select_precision_threshold,
)
from .data import ReliabilityDomain, partition_population
from .evaluation import (
    evaluate_frozen_thresholds,
    paired_bootstrap_comparison,
    probability_report,
    risk_coverage_curve,
    subgroup_report,
)
from .features import agreement_features, conflict_features
from .model import ReliabilityConfig, fit_reliability_model, select_reliability_config


DATASETS = ("cora", "citeseer", "pubmed", "arxiv", "ogbn-products")
MODULES = ("conflict", "agreement")
VIEWS = ("raw", "rank")


@dataclass(frozen=True)
class PipelineConfig:
    input_run_root: Path
    frozen_b_root: Path
    dataset_root: Path
    output_root: Path
    datasets: tuple[str, ...] = DATASETS
    seed: int = 42
    l2_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    prior_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    calibration_regularization: float = 0.1
    precision_targets: tuple[float, ...] = (0.80, 0.85, 0.90)
    min_accepted: int = 10
    bootstrap_replicates: int = 1000
    resume: bool = False


def _rank_support(domain: SupportDomain) -> tuple[torch.Tensor, list[str]]:
    keep = [
        index
        for index, name in enumerate(domain.feature_names)
        if not name.startswith("ctx_")
    ]
    return (
        percentile_features(domain.x[:, keep]),
        [domain.feature_names[index] for index in keep],
    )


def _rank_target(target: TargetArtifacts) -> tuple[torch.Tensor, list[str]]:
    keep = [
        index
        for index, name in enumerate(target.feature_names)
        if not name.startswith("ctx_")
    ]
    return (
        percentile_features(target.x[:, keep]),
        [target.feature_names[index] for index in keep],
    )


def _feature_domain(domain: ReliabilityDomain, threshold: float) -> ReliabilityDomain:
    if domain.module == "conflict":
        x, names = conflict_features(
            domain.evidence,
            domain.feature_names,
            choose_g=domain.choose_g,
            b_score=domain.b_score,
            b_threshold=threshold,
        )
    else:
        x, names = agreement_features(domain.evidence, domain.feature_names)
    return ReliabilityDomain(
        dataset=domain.dataset,
        module=domain.module,
        node_ids=domain.node_ids,
        evidence=x,
        feature_names=names,
        target=domain.target,
        choose_g=domain.choose_g,
        b_score=domain.b_score,
        state=domain.state,
    )


def build_support_feature_domains(
    supports: dict[str, SupportDomain], replays: dict[str, FrozenBReplay]
) -> dict[str, dict[str, dict[str, ReliabilityDomain]]]:
    output = {view: {} for view in VIEWS}
    for dataset in DATASETS:
        support, replay = supports[dataset], replays[dataset]
        ids = torch.tensor(
            [int(row["node_id"]) for row in support.rows], dtype=torch.long
        )
        labels = torch.tensor(
            [int(row["y_true"]) for row in support.rows], dtype=torch.long
        )
        gnn = torch.tensor(
            [int(row["gnn_pred"]) for row in support.rows], dtype=torch.long
        )
        llm = torch.tensor(
            [int(row["llm_pred"]) for row in support.rows], dtype=torch.long
        )
        view_values = {
            "raw": (support.x, support.feature_names),
            "rank": _rank_support(support),
        }
        for view, (evidence, names) in view_values.items():
            populations = partition_population(
                dataset=dataset,
                node_ids=ids,
                evidence=evidence,
                feature_names=list(names),
                labels=labels,
                gnn_pred=gnn,
                llm_pred=llm,
                choose_g=replay.support_choose_g,
                b_score=replay.support_score,
            )
            output[view][dataset] = {
                module: _feature_domain(populations[module], replay.threshold)
                for module in MODULES
            }
    return output


def _query_domain(
    target: TargetArtifacts,
    replay: FrozenBReplay,
    node_ids: torch.Tensor,
    *,
    view: str,
    module: str,
    view_values: dict[str, tuple[torch.Tensor, list[str]]] | None = None,
) -> ReliabilityDomain:
    if view_values is not None:
        if view not in view_values:
            raise ValueError(f"missing cached evidence view: {view}")
        evidence, names = view_values[view]
    elif view == "raw":
        evidence, names = target.x, target.feature_names
    elif view == "rank":
        evidence, names = _rank_target(target)
    else:
        raise ValueError(f"unknown evidence view: {view}")
    ids = torch.as_tensor(node_ids, dtype=torch.long).view(-1)
    populations = partition_population(
        dataset=target.name,
        node_ids=ids,
        evidence=evidence[ids],
        feature_names=list(names),
        labels=target.labels[ids],
        gnn_pred=target.gnn_pred[ids],
        llm_pred=target.llm_pred[ids],
        choose_g=replay.full_choose_g[ids],
        b_score=replay.full_score[ids],
    )
    return _feature_domain(populations[module], replay.threshold)


def _hash_ids(ids: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for node_id in torch.as_tensor(ids, dtype=torch.long).view(-1).tolist():
        digest.update(f"{int(node_id)}\n".encode("utf-8"))
    return digest.hexdigest()


def _old_trust(frozen_root: Path, dataset: str, node_ids: torch.Tensor) -> torch.Tensor:
    rows = [
        json.loads(line)
        for line in (frozen_root / dataset / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    mapping = {int(row["node_id"]): float(row["selected_trust"]) for row in rows}
    if len(mapping) != len(rows):
        raise ValueError(f"duplicate frozen Scheme-C node IDs for {dataset}")
    missing = [
        int(node_id) for node_id in node_ids.tolist() if int(node_id) not in mapping
    ]
    if missing:
        raise ValueError(
            f"frozen Scheme-C comparator missing IDs for {dataset}: {missing[:5]}"
        )
    return torch.tensor([mapping[int(node_id)] for node_id in node_ids.tolist()])


def _heuristic_probability(
    target: TargetArtifacts,
    replay: FrozenBReplay,
    domain: ReliabilityDomain,
) -> torch.Tensor:
    ids = domain.node_ids
    if domain.module == "conflict":
        g_top = target.gnn_prob[ids].max(dim=1).values
        l_top = target.llm_prob[ids].max(dim=1).values
        return torch.where(replay.full_choose_g[ids], g_top, l_top)
    common = target.gnn_pred[ids]
    positions = torch.arange(ids.numel())
    g_common = target.gnn_prob[ids][positions, common]
    l_common = target.llm_prob[ids][positions, common]
    return (g_common + l_common) / 2.0


def _compact_curve(
    probability: torch.Tensor, correct: torch.Tensor, max_points: int = 1001
) -> list[dict]:
    curve = risk_coverage_curve(probability, correct)
    if len(curve) <= max_points:
        return curve
    indices = (
        torch.linspace(0, len(curve) - 1, max_points).round().long().unique().tolist()
    )
    return [curve[index] for index in indices]


def _base_rate(
    target_support: ReliabilityDomain, sources: list[ReliabilityDomain]
) -> tuple[float, str]:
    if target_support.target.numel():
        return float(target_support.target.to(torch.float64).mean()), "target_support"
    return (
        sum(float(domain.target.to(torch.float64).mean()) for domain in sources)
        / len(sources),
        "equal_source_domain_mean",
    )


def _module_run(
    *,
    config: PipelineConfig,
    dataset: str,
    module: str,
    supports_by_view: dict[str, dict[str, dict[str, ReliabilityDomain]]],
    replay: FrozenBReplay,
    target: TargetArtifacts,
    target_views: dict[str, tuple[torch.Tensor, list[str]]],
) -> tuple[dict, dict, list[dict]]:
    sources_by_view = {
        view: [
            supports_by_view[view][name][module] for name in DATASETS if name != dataset
        ]
        for view in VIEWS
    }
    selection = select_reliability_config(
        sources_by_view,
        l2_grid=config.l2_grid,
        prior_grid=config.prior_grid,
    )
    validate_selection_record(selection)
    best = selection["best"]
    view = str(best["view"])
    sources = sources_by_view[view]
    target_support = supports_by_view[view][dataset][module]
    fitted = fit_reliability_model(
        sources,
        target_support,
        ReliabilityConfig(float(best["l2"]), float(best["prior_precision"])),
    )

    validation = _query_domain(
        target,
        replay,
        target.val_ids,
        view=view,
        module=module,
        view_values=target_views,
    )
    split = deterministic_calibration_threshold_split(
        validation.node_ids, dataset, module
    )
    if bool((split.calibration & split.threshold).any()) or not bool(
        (split.calibration | split.threshold).all()
    ):
        raise AssertionError("invalid calibration/threshold split")
    validation_logits = fitted.logits(validation.evidence)
    calibrator = fit_identity_shrunk_platt(
        validation_logits[split.calibration],
        validation.target[split.calibration],
        regularization=config.calibration_regularization,
    )
    validation_probability = calibrator.probabilities_from_logits(validation_logits)
    thresholds = {
        f"precision_{precision:.2f}": select_precision_threshold(
            validation_probability[split.threshold],
            validation.target[split.threshold],
            target_precision=precision,
            min_accepted=config.min_accepted,
        )
        for precision in config.precision_targets
    }

    test = _query_domain(
        target,
        replay,
        target.test_ids,
        view=view,
        module=module,
        view_values=target_views,
    )
    raw_logits = fitted.logits(test.evidence)
    raw_probability = torch.sigmoid(raw_logits).to(torch.float32)
    calibrated = calibrator.probabilities_from_logits(raw_logits)
    old = _old_trust(config.frozen_b_root, dataset, test.node_ids)
    heuristic = _heuristic_probability(target, replay, test)
    rate, rate_source = _base_rate(target_support, sources)
    base = torch.full_like(calibrated, rate)
    comparators = {
        "split_calibrated": calibrated,
        "split_uncalibrated": raw_probability,
        "frozen_original_c": old,
        "expert_confidence_heuristic": heuristic,
        "base_rate": base,
    }
    comparator_metrics = {
        name: probability_report(probability, test.target, include_curve=False)
        for name, probability in comparators.items()
    }
    frozen_test_thresholds = evaluate_frozen_thresholds(
        calibrated, test.target, thresholds
    )
    matched_count = max(1, int(math.ceil(0.50 * test.target.numel())))
    bootstrap = paired_bootstrap_comparison(
        calibrated,
        old,
        test.target,
        matched_count=matched_count,
        replicates=config.bootstrap_replicates,
        seed=config.seed + DATASETS.index(dataset) * 17 + MODULES.index(module),
    )
    subgroups = {"state": subgroup_report(calibrated, test.target, test.state)}
    if module == "conflict":
        routes = [
            "b_route_gnn" if bool(value) else "b_route_llm" for value in test.choose_g
        ]
        subgroups["b_route"] = subgroup_report(calibrated, test.target, routes)

    metrics = {
        "dataset": dataset,
        "module": module,
        "semantics": (
            "P(frozen Scheme-B selected label is correct | disagreement)"
            if module == "conflict"
            else "P(common expert label is correct | agreement)"
        ),
        "population": {
            "support_count": int(target_support.target.numel()),
            "validation_count": int(validation.target.numel()),
            "test_count": int(test.target.numel()),
            "test_positive_count": int(test.target.sum()),
            "test_negative_count": int((~test.target).sum()),
        },
        "selection": selection,
        "model": fitted.to_dict(),
        "calibration": {
            **calibrator.to_dict(),
            "calibration_count": int(split.calibration.sum()),
            "threshold_count": int(split.threshold.sum()),
            "calibration_ids_sha256": _hash_ids(validation.node_ids[split.calibration]),
            "threshold_ids_sha256": _hash_ids(validation.node_ids[split.threshold]),
            "raw_calibration_half": probability_report(
                torch.sigmoid(validation_logits[split.calibration]),
                validation.target[split.calibration],
                include_curve=False,
            ),
            "calibrated_calibration_half": probability_report(
                validation_probability[split.calibration],
                validation.target[split.calibration],
                include_curve=False,
            ),
        },
        "thresholds": {name: value.to_dict() for name, value in thresholds.items()},
        "test_thresholds": frozen_test_thresholds,
        "comparators": comparator_metrics,
        "subgroups": subgroups,
        "paired_bootstrap_vs_frozen_original_c": {
            "unit": "node",
            "replicates": config.bootstrap_replicates,
            "matched_coverage": 0.50,
            "matched_count": matched_count,
            "delta_definition": "split_calibrated minus frozen_original_c",
            **bootstrap,
        },
        "base_rate_source": rate_source,
    }
    curves = {
        name: _compact_curve(probability, test.target)
        for name, probability in comparators.items()
    }
    decisions = []
    threshold_accept = {
        name: (
            torch.zeros_like(calibrated, dtype=torch.bool)
            if value.reject_all or value.threshold is None
            else calibrated >= float(value.threshold)
        )
        for name, value in thresholds.items()
    }
    for position, node_id in enumerate(test.node_ids.tolist()):
        choose_g = bool(test.choose_g[position])
        selected_prediction = int(
            target.gnn_pred[node_id] if choose_g else target.llm_pred[node_id]
        )
        row = {
            "node_id": int(node_id),
            "module": module,
            "state": test.state[position],
            "true_label": int(target.labels[node_id]),
            "gnn_pred": int(target.gnn_pred[node_id]),
            "llm_pred": int(target.llm_pred[node_id]),
            "b_route": "gnn" if choose_g else "llm",
            "b_score": float(test.b_score[position]),
            "selected_pred": selected_prediction,
            "correct": bool(test.target[position]),
            "split_raw_reliability": float(raw_probability[position]),
            "split_calibrated_reliability": float(calibrated[position]),
            "frozen_original_c_trust": float(old[position]),
            "expert_confidence_heuristic": float(heuristic[position]),
            "base_rate": float(base[position]),
            "accept": {
                name: bool(mask[position]) for name, mask in threshold_accept.items()
            },
        }
        decisions.append(row)
    return metrics, curves, decisions


def _provenance(config: PipelineConfig) -> dict:
    result = {}
    for dataset in DATASETS:
        paths = {
            "support_oof": config.input_run_root
            / dataset
            / "pipeline/router/oof/router_train.jsonl",
            "expert_predictions": config.input_run_root
            / dataset
            / "pipeline/predictions.pt",
            "official_split": config.input_run_root
            / dataset
            / "pipeline/splits"
            / f"{dataset}_5shot_seed42.json",
            "graph_dataset": config.dataset_root / "datasets" / f"{dataset}.pt",
            "frozen_b_metrics": config.frozen_b_root / dataset / "metrics.json",
            "frozen_b_decisions": config.frozen_b_root / dataset / "decisions.jsonl",
        }
        result[dataset] = {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in paths.items()
        }
    return result


def _summary_rows(metrics_by_dataset: dict[str, dict]) -> list[dict]:
    rows = []
    for dataset in sorted(metrics_by_dataset, key=lambda value: DATASETS.index(value)):
        for module in MODULES:
            value = metrics_by_dataset[dataset]["modules"][module]
            new = value["comparators"]["split_calibrated"]
            old = value["comparators"]["frozen_original_c"]
            primary = value["test_thresholds"]["precision_0.90"]
            rows.append(
                {
                    "dataset": dataset,
                    "module": module,
                    "count": new["count"],
                    "positive_rate": new["base_accuracy"],
                    "view": value["selection"]["best"]["view"],
                    "auroc_new": new["auroc"],
                    "auroc_old": old["auroc"],
                    "nll_new": new["nll"],
                    "nll_old": old["nll"],
                    "brier_new": new["brier"],
                    "brier_old": old["brier"],
                    "aurc_new": new["aurc"],
                    "aurc_old": old["aurc"],
                    "p90_test_coverage": primary["coverage"],
                    "p90_test_precision": primary["precision"],
                }
            )
    return rows


def _csv_text(rows: list[dict]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _report(rows: list[dict], audits: dict[str, dict]) -> str:
    lines = [
        "# Router-v6 Split Scheme-C Results",
        "",
        "Scheme B is frozen and identity-audited. C_conflict and C_agreement are trained, calibrated, thresholded, and ranked independently.",
        "",
        "| Dataset | Module | N | New AUROC | Old AUROC | New NLL | Old NLL | New AURC | Old AURC | P90 test coverage | P90 test precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        fmt = lambda value: "NA" if value is None else f"{float(value):.4f}"
        lines.append(
            f"| {row['dataset']} | {row['module']} | {row['count']} | {fmt(row['auroc_new'])} | {fmt(row['auroc_old'])} | {fmt(row['nll_new'])} | {fmt(row['nll_old'])} | {fmt(row['aurc_new'])} | {fmt(row['aurc_old'])} | {fmt(row['p90_test_coverage'])} | {fmt(row['p90_test_precision'])} |"
        )
    lines.extend(
        [
            "",
            "## Scheme-B identity",
            "",
            *(
                f"- {name}: {'PASS' if audit['all_checks_pass'] else 'FAIL'} ({audit['checked']} test nodes, max score diff {audit['max_score_abs_diff']:.3g})"
                for name, audit in audits.items()
            ),
            "",
            "No numeric score is used to rank agreement nodes against conflict nodes. Threshold feasibility is determined on each module's independent validation-threshold half; reject-all is retained when the precision constraint is infeasible.",
            "",
        ]
    )
    return "\n".join(lines)


def run_pipeline(config: PipelineConfig) -> dict:
    started = time.time()
    if not config.datasets or any(name not in DATASETS for name in config.datasets):
        raise ValueError(f"datasets must be a non-empty subset of {DATASETS}")
    root = prepare_output_root(config.output_root, resume=config.resume)
    supports = {
        name: build_support_domain(
            name,
            config.input_run_root / name / "pipeline/router/oof/router_train.jsonl",
        )
        for name in DATASETS
    }
    replays: dict[str, FrozenBReplay] = {}
    targets: dict[str, TargetArtifacts] = {}
    audits: dict[str, dict] = {}
    for name in DATASETS:
        replay, target, audit = replay_selected_b(
            dataset=name,
            supports=supports,
            input_run_root=config.input_run_root,
            frozen_b_root=config.frozen_b_root,
            dataset_root=config.dataset_root,
            seed=config.seed,
        )
        if not audit["all_checks_pass"]:
            raise RuntimeError(f"frozen Scheme-B identity failed for {name}: {audit}")
        replays[name], targets[name], audits[name] = replay, target, audit
    support_domains = build_support_feature_domains(supports, replays)

    protocol = {
        "schema_version": 1,
        "scheme_b": "frozen_selected_endpoint_exact_replay",
        "modules": list(MODULES),
        "views": list(VIEWS),
        "source_selection": "leave-one-source-dataset-out macro NLL/Brier/AURC",
        "target_adaptation": "scalar intercept on fixed support only",
        "validation": "label-blind disjoint calibration/threshold halves per module",
        "precision_targets": list(config.precision_targets),
        "min_accepted": config.min_accepted,
        "forbidden": "global or cross-population reliability ranking",
        "test_labels_used_for": "final evaluation and paired bootstrap only",
    }
    atomic_write_json(
        {
            "protocol": protocol,
            "arguments": {
                "input_run_root": str(config.input_run_root.resolve()),
                "frozen_b_root": str(config.frozen_b_root.resolve()),
                "dataset_root": str(config.dataset_root.resolve()),
                "output_root": str(config.output_root.resolve()),
                "datasets": list(config.datasets),
                "seed": config.seed,
                "l2_grid": list(config.l2_grid),
                "prior_grid": list(config.prior_grid),
                "calibration_regularization": config.calibration_regularization,
                "precision_targets": list(config.precision_targets),
                "min_accepted": config.min_accepted,
                "bootstrap_replicates": config.bootstrap_replicates,
            },
            "provenance": _provenance(config),
        },
        root / "config.json",
    )
    atomic_write_json({"scheme_b_identity": audits}, root / "audit.json")

    metrics_by_dataset = {}
    for dataset in config.datasets:
        destination = root / dataset
        if config.resume and (destination / "complete.json").exists():
            metrics_by_dataset[dataset] = json.loads(
                (destination / "metrics.json").read_text(encoding="utf-8")
            )
            continue
        if destination.exists():
            raise FileExistsError(
                f"refusing to replace incomplete dataset output: {destination}"
            )
        staging = Path(tempfile.mkdtemp(prefix=f".{dataset}.tmp-", dir=root))
        try:
            module_metrics, module_curves, all_decisions = {}, {}, []
            target_views = {
                "raw": (targets[dataset].x, targets[dataset].feature_names),
                "rank": _rank_target(targets[dataset]),
            }
            for module in MODULES:
                metrics, curves, decisions = _module_run(
                    config=config,
                    dataset=dataset,
                    module=module,
                    supports_by_view=support_domains,
                    replay=replays[dataset],
                    target=targets[dataset],
                    target_views=target_views,
                )
                module_metrics[module] = metrics
                module_curves[module] = curves
                all_decisions.extend(decisions)
                atomic_write_jsonl(decisions, staging / f"decisions_{module}.jsonl")
            conflict_ids = {
                row["node_id"] for row in all_decisions if row["module"] == "conflict"
            }
            agreement_ids = {
                row["node_id"] for row in all_decisions if row["module"] == "agreement"
            }
            expected_ids = set(targets[dataset].test_ids.tolist())
            if (
                conflict_ids & agreement_ids
                or conflict_ids | agreement_ids != expected_ids
            ):
                raise AssertionError(f"test populations do not partition {dataset}")
            dataset_metrics = {
                "dataset": dataset,
                "scheme_b_identity": audits[dataset],
                "population_partition": {
                    "conflict_count": len(conflict_ids),
                    "agreement_count": len(agreement_ids),
                    "test_count": len(expected_ids),
                    "disjoint": True,
                    "exhaustive": True,
                },
                "modules": module_metrics,
            }
            atomic_write_json(dataset_metrics, staging / "metrics.json")
            atomic_write_json(
                {module: value["model"] for module, value in module_metrics.items()},
                staging / "models.json",
            )
            atomic_write_json(module_curves, staging / "curves.json")
            atomic_write_jsonl(
                sorted(all_decisions, key=lambda row: (row["node_id"], row["module"])),
                staging / "decisions.jsonl",
            )
            atomic_write_json(
                {"complete": True, "dataset": dataset, "module_count": 2},
                staging / "complete.json",
            )
            os.replace(staging, destination)
            metrics_by_dataset[dataset] = dataset_metrics
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    rows = _summary_rows(metrics_by_dataset)
    atomic_write_json(rows, root / "summary.json")
    atomic_write_text(_csv_text(rows), root / "summary.csv")
    atomic_write_text(_report(rows, audits), root / "REPORT.md")
    atomic_write_json(
        {
            "complete": True,
            "datasets": list(config.datasets),
            "elapsed_seconds": time.time() - started,
        },
        root / "complete.json",
    )
    manifest = build_manifest(root)
    atomic_write_json(manifest, root / "manifest.json")
    manifest_audit = verify_manifest(root)
    if not manifest_audit["all_checks_pass"]:
        raise RuntimeError(f"manifest verification failed: {manifest_audit}")
    return {
        "output_root": str(root),
        "datasets": list(config.datasets),
        "summary": rows,
        "manifest_audit": manifest_audit,
        "scheme_b_identity": audits,
    }


__all__ = [
    "DATASETS",
    "MODULES",
    "PipelineConfig",
    "build_support_feature_domains",
    "run_pipeline",
]
