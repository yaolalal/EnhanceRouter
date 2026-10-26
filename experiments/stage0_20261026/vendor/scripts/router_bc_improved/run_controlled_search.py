#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eargtc.router_bc.data_impl import SupportDomain, build_support_domain, load_target_artifacts
from eargtc.router_bc.hybrid import blend_preference_logits
from eargtc.router_bc.io import guarded_output_directory
from eargtc.router_bc.metrics import probability_metrics, routing_metrics
from eargtc.router_bc.nonlinear import fit_preference_model, select_preference_model
from eargtc.router_bc.rank import percentile_features
from eargtc.router_bc.training import cross_validate_head, fit_head, head_targets
from eargtc.router_bc_improved.meta import (
    PreferenceConfig,
    PreferenceDomain,
    fit_preference_system,
    honest_all_row_scores,
    select_preference_config,
)
from eargtc.router_bc_improved.preference import configure_router_threads
from eargtc.router_bc_improved.runner_utils import (
    best_trial_for_family,
    exact_one_sided_binomial_p,
    preference_config_grid,
    validate_selection_record,
)
from eargtc.router_bc_improved.selection import CheckpointCandidate, select_b_checkpoint
from eargtc.router_bc_improved.trust import (
    TrustDomain,
    direct_trust_features,
    factorized_selected_correctness,
    fit_direct_trust,
    risk_coverage_metrics,
    select_direct_trust_config,
    select_trust_threshold,
)


DATASETS = ("cora", "citeseer", "pubmed", "arxiv", "ogbn-products")
FAMILIES = ("baseline", "ranking", "adapter", "combined")


def parse_grid(value: str) -> tuple[float, ...]:
    result = tuple(float(part) for part in value.split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("grid cannot be empty")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--input-run-root", type=Path, default=Path("runs/router_v5_no_calibration_reliability_only_fast_20260714"))
    result.add_argument("--dataset-root", type=Path, default=Path("/home/wujingyao/codes/TAG_Fewshot_Learning/GNN-as-Judge"))
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--datasets", default=",".join(DATASETS))
    result.add_argument("--legacy-l2-grid", type=parse_grid, default=parse_grid("0.01,0.1,1,10,100"))
    result.add_argument("--legacy-prior-grid", type=parse_grid, default=parse_grid("0.1,1,10,100"))
    result.add_argument("--trust-l2-grid", type=parse_grid, default=parse_grid("0.01,0.1,1,10"))
    result.add_argument("--trust-prior-grid", type=parse_grid, default=parse_grid("0.1,1,10,100"))
    result.add_argument("--trust-target-precision", type=float, default=0.90)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--resume", action="store_true")
    return result


def write_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def rank_support(domain: SupportDomain) -> SupportDomain:
    keep = [index for index, name in enumerate(domain.feature_names) if not name.startswith("ctx_")]
    return SupportDomain(
        name=domain.name,
        rows=domain.rows,
        x=percentile_features(domain.x[:, keep]),
        feature_names=[domain.feature_names[index] for index in keep],
        gnn_correct=domain.gnn_correct,
        llm_correct=domain.llm_correct,
        disagreement=domain.disagreement,
    )


def preference_domains(supports: dict[str, SupportDomain]) -> dict[str, PreferenceDomain]:
    result = {}
    for name, domain in supports.items():
        mask, target = head_targets(domain, "preference")
        result[name] = PreferenceDomain(name, domain.x[mask], target[mask])
    return result


def config_from_trial(row: dict) -> PreferenceConfig:
    return PreferenceConfig(
        l2=float(row["l2"]),
        rank_weight=float(row["rank_weight"]),
        adapter_rank=int(row["adapter_rank"]),
        prior_precision=float(row["prior_precision"]),
    )


def improved_candidates(
    view: str,
    supports: dict[str, SupportDomain],
    target_name: str,
    target_x: torch.Tensor,
) -> tuple[list[dict], dict]:
    domains = preference_domains(supports)
    source = [domains[name] for name in DATASETS if name != target_name]
    target_support = domains[target_name]
    meta = select_preference_config(source, preference_config_grid())
    candidates = []
    for family in FAMILIES:
        for objective in ("auc", "stable_auc", "nll"):
            trial = best_trial_for_family(
                meta["trials"], family, objective=objective, max_macro_nll=0.8
            )
            config = config_from_trial(trial)
            fitted = fit_preference_system(source, target_support, config)
            candidates.append(
                {
                    "name": f"improved_{view}_{family}_{objective}",
                    "view": view,
                    "family": family,
                    "complexity": int(config.adapter_rank > 0) + int(config.rank_weight > 0),
                    "score": fitted.probabilities(target_x),
                    "support_score": fitted.probabilities(supports[target_name].x),
                    "config": asdict(config),
                    "source_cv": {
                        "objective": objective,
                        "max_macro_nll": 0.8 if objective == "stable_auc" else None,
                        "macro_auroc": trial["macro_auroc"],
                        "macro_nll": trial["macro_nll"],
                    },
                }
            )
    return candidates, meta


def legacy_candidates(
    args,
    view: str,
    supports: dict[str, SupportDomain],
    target_name: str,
    target_x: torch.Tensor,
) -> tuple[list[dict], dict]:
    source = [supports[name] for name in DATASETS if name != target_name]
    target_support = supports[target_name]
    linear_cv = cross_validate_head(
        source,
        "preference",
        l2_grid=args.legacy_l2_grid,
        prior_grid=args.legacy_prior_grid,
    )
    nonlinear_cv = select_preference_model(source, seed=args.seed)
    linear = fit_head(source, target_support, "preference", linear_cv)
    nonlinear = fit_preference_model(source, target_support, nonlinear_cv, seed=args.seed)
    linear_score = linear.probabilities(target_x)
    nonlinear_score = nonlinear.probabilities(target_x)
    linear_support = linear.probabilities(target_support.x)
    nonlinear_support = nonlinear.probabilities(target_support.x)
    candidates = []
    for index, fraction in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
        candidates.append(
            {
                "name": f"legacy_{view}_lambda_{fraction:.2f}",
                "view": view,
                "family": "legacy_residual",
                "complexity": index,
                "score": blend_preference_logits(linear_score, nonlinear_score, fraction),
                "support_score": blend_preference_logits(linear_support, nonlinear_support, fraction),
                "config": {"lambda": fraction},
                "source_cv": {
                    "linear": linear_cv["best"],
                    "nonlinear": nonlinear_cv["best"],
                },
            }
        )
    return candidates, {"linear": linear_cv, "nonlinear": nonlinear_cv}


def paired_statistics(correct: torch.Tensor, comparator_correct: torch.Tensor, seed: int) -> dict:
    delta = correct.to(torch.int64) - comparator_correct.to(torch.int64)
    wins = int((delta == 1).sum())
    losses = int((delta == -1).sum())
    ties = int((delta == 0).sum())
    counts = np.array([losses, ties, wins], dtype=np.int64)
    values = np.array([-1.0, 0.0, 1.0])
    rng = np.random.default_rng(seed)
    samples = rng.multinomial(int(delta.numel()), counts / counts.sum(), size=5000)
    gains = samples @ values / delta.numel()
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "net_correct": wins - losses,
        "one_sided_exact_p": exact_one_sided_binomial_p(wins, losses),
        "gain_ci95": [float(np.quantile(gains, 0.025)), float(np.quantile(gains, 0.975))],
    }


def selective_test(q: torch.Tensor, correct: torch.Tensor, threshold: dict) -> dict:
    selected = q >= float(threshold["threshold"])
    return {
        "selected_count": int(selected.sum()),
        "coverage": float(selected.float().mean()),
        "precision": float(correct[selected].float().mean()) if bool(selected.any()) else None,
    }


def build_source_trust_domains(
    view_supports: dict[str, SupportDomain],
    target_name: str,
    config: PreferenceConfig,
) -> list[TrustDomain]:
    source_names = [name for name in DATASETS if name != target_name]
    preferences = preference_domains(view_supports)
    result = []
    for held_name in source_names:
        held = view_supports[held_name]
        train = [preferences[name] for name in source_names if name != held_name]
        mask, _ = head_targets(held, "preference")
        score = honest_all_row_scores(
            train,
            preferences[held_name],
            held.x,
            mask.nonzero().view(-1),
            config,
        )
        agreement = ~held.disagreement
        choose_g = agreement | (score >= 0.5)
        correct = torch.where(choose_g, held.gnn_correct, held.llm_correct)
        x = direct_trust_features(
            held.x,
            agreement=agreement,
            choose_g=choose_g,
            preference=score,
        )
        result.append(TrustDomain(held_name, x, correct.float()))
    return result


def run_target(args, target_name: str, raw_supports: dict[str, SupportDomain], destination: Path) -> dict:
    started = time.time()
    pipeline = args.input_run_root / target_name / "pipeline"
    target = load_target_artifacts(target_name, pipeline, args.dataset_root, raw_supports[target_name])
    support_views = {"raw": raw_supports, "rank": {name: rank_support(domain) for name, domain in raw_supports.items()}}
    target_views = {
        "raw": target.x,
        "rank": percentile_features(target.x[:, : len(support_views["rank"][target_name].feature_names)]),
    }

    candidates = []
    meta_by_view = {}
    legacy_by_view = {}
    for view in ("raw", "rank"):
        print(f"[{target_name}] {view}: full source meta-grid", flush=True)
        improved, meta = improved_candidates(view, support_views[view], target_name, target_views[view])
        print(f"[{target_name}] {view}: legacy residual endpoints", flush=True)
        legacy, legacy_cv = legacy_candidates(args, view, support_views[view], target_name, target_views[view])
        candidates.extend(improved + legacy)
        meta_by_view[view] = meta
        legacy_by_view[view] = legacy_cv

    validation_candidates = [
        CheckpointCandidate(
            candidate["name"],
            candidate["score"][target.val_ids],
            candidate["view"],
            candidate["complexity"],
        )
        for candidate in candidates
    ]
    selection = select_b_checkpoint(
        validation_candidates,
        target.labels[target.val_ids],
        target.gnn_pred[target.val_ids],
        target.llm_pred[target.val_ids],
    )
    validate_selection_record(selection)
    selected_candidate = next(row for row in candidates if row["name"] == selection["name"])
    score = selected_candidate["score"]
    threshold = float(selection["threshold"]["threshold"])
    b_val, _, val_correct = routing_metrics(
        target.val_ids, target.labels, target.gnn_pred, target.llm_pred, score, threshold
    )
    b_test, test_choose_g, test_correct = routing_metrics(
        target.test_ids, target.labels, target.gnn_pred, target.llm_pred, score, threshold
    )
    test_ids = target.test_ids
    if b_test["gnn_accuracy"] >= b_test["llm_accuracy"]:
        comparator_correct = target.gnn_pred[test_ids] == target.labels[test_ids]
        comparator = "gnn"
    else:
        comparator_correct = target.llm_pred[test_ids] == target.labels[test_ids]
        comparator = "llm"
    b_test["paired_vs_best_single"] = {"comparator": comparator, **paired_statistics(test_correct, comparator_correct, args.seed)}

    ablations = []
    trial_by_name = {row["name"]: row for row in selection["trials"]}
    for candidate in candidates:
        trial = trial_by_name[candidate["name"]]
        metric, _, _ = routing_metrics(
            target.test_ids,
            target.labels,
            target.gnn_pred,
            target.llm_pred,
            candidate["score"],
            float(trial["threshold"]["threshold"]),
        )
        ablations.append(
            {
                "name": candidate["name"],
                "view": candidate["view"],
                "family": candidate["family"],
                "config": candidate["config"],
                "source_cv": candidate["source_cv"],
                "validation_accuracy": trial["accuracy"],
                "test": metric,
            }
        )

    selected_view = selected_candidate["view"]
    view_supports = support_views[selected_view]
    source_supports = [view_supports[name] for name in DATASETS if name != target_name]
    target_support = view_supports[target_name]
    availability_cv = cross_validate_head(source_supports, "availability", l2_grid=args.trust_l2_grid, prior_grid=args.trust_prior_grid)
    agreement_cv = cross_validate_head(source_supports, "agreement", l2_grid=args.trust_l2_grid, prior_grid=args.trust_prior_grid)
    availability = fit_head(source_supports, target_support, "availability", availability_cv)
    agreement_head = fit_head(source_supports, target_support, "agreement", agreement_cv)
    a = availability.probabilities(target_views[selected_view])
    u = agreement_head.probabilities(target_views[selected_view])
    conflict = target.gnn_pred != target.llm_pred
    choose_g = (~conflict) | (score >= threshold)
    q_factorized = factorized_selected_correctness(conflict, choose_g, a, score, u)

    combined_trial = best_trial_for_family(
        meta_by_view[selected_view]["trials"],
        "combined",
        objective="stable_auc",
        max_macro_nll=0.8,
    )
    direct_source_domains = build_source_trust_domains(
        view_supports, target_name, config_from_trial(combined_trial)
    )
    support_score = selected_candidate["support_score"]
    support_choose_g = (~target_support.disagreement) | (support_score >= threshold)
    support_correct = torch.where(support_choose_g, target_support.gnn_correct, target_support.llm_correct)
    target_trust_support = TrustDomain(
        target_name,
        direct_trust_features(
            target_support.x,
            agreement=~target_support.disagreement,
            choose_g=support_choose_g,
            preference=support_score,
        ),
        support_correct.float(),
    )
    direct_cv = select_direct_trust_config(
        direct_source_domains,
        l2_grid=args.trust_l2_grid,
        prior_grid=args.trust_prior_grid,
    )
    direct = fit_direct_trust(
        direct_source_domains,
        target_trust_support,
        l2=direct_cv["best"]["l2"],
        prior_precision=direct_cv["best"]["prior_precision"],
    )
    direct_x = direct_trust_features(
        target_views[selected_view],
        agreement=~conflict,
        choose_g=choose_g,
        preference=score,
    )
    q_direct = direct.probabilities(direct_x)

    val_q_metrics = {
        "factorized": probability_metrics(q_factorized[target.val_ids], val_correct),
        "direct": probability_metrics(q_direct[target.val_ids], val_correct),
    }
    selected_trust = min(
        val_q_metrics,
        key=lambda name: (val_q_metrics[name]["nll"], 0 if name == "factorized" else 1),
    )
    q_final = q_factorized if selected_trust == "factorized" else q_direct
    trust_selection = {
        "selection": "official_validation_trust_checkpoint_early_stopping",
        "selected": selected_trust,
        "validation": val_q_metrics,
    }
    validate_selection_record(trust_selection)
    trust_threshold = select_trust_threshold(
        q_final[target.val_ids], val_correct, target_precision=args.trust_target_precision
    )
    factor_test = probability_metrics(q_factorized[test_ids], test_correct)
    direct_test = probability_metrics(q_direct[test_ids], test_correct)
    final_test = probability_metrics(q_final[test_ids], test_correct)
    trust_metrics = {
        "factorized": {**factor_test, **risk_coverage_metrics(q_factorized[test_ids], test_correct)},
        "direct": {**direct_test, **risk_coverage_metrics(q_direct[test_ids], test_correct)},
        "selected": {**final_test, **risk_coverage_metrics(q_final[test_ids], test_correct)},
        "validation_selection": trust_selection,
        "validation_threshold": trust_threshold,
        "test_at_validation_threshold": selective_test(q_final[test_ids], test_correct, trust_threshold),
    }

    summary = {
        "dataset": target_name,
        "protocol": {
            "shots": 5,
            "seed": args.seed,
            "source_domains": [name for name in DATASETS if name != target_name],
            "target_adapter_max_rank": 2,
            "validation_usage": "checkpoint_early_stopping_and_scalar_thresholds_only",
            "test_usage": "evaluation_only_after_selection_records_are_frozen",
            "class_permutation_invariant": True,
        },
        "counts": {
            "support": len(target_support.rows),
            "validation": int(target.val_ids.numel()),
            "test": int(target.test_ids.numel()),
        },
        "scheme_b": {
            "selected_checkpoint": selection,
            "selected_metadata": {key: selected_candidate[key] for key in ("name", "view", "family", "complexity", "config", "source_cv")},
            "validation": b_val,
            "test": b_test,
            "ablations": ablations,
        },
        "scheme_c": {
            "route_identity_with_b": True,
            "semantics": "selected_label_correctness_trust_not_an_independent_router",
            "metrics": trust_metrics,
            "factorized_models": {
                "availability": availability.to_dict(),
                "agreement": agreement_head.to_dict(),
            },
            "direct_model": {
                "shared_coef": direct.shared.coef.tolist(),
                "shared_intercept": direct.shared.intercept,
                "adaptation_bias": direct.adapted.adaptation_bias,
                "source_cv": direct_cv,
            },
        },
        "source_meta": meta_by_view,
        "legacy_source_cv": legacy_by_view,
        "elapsed_seconds": time.time() - started,
        "provenance": {
            "predictions": str(pipeline / "predictions.pt"),
            "predictions_sha256": sha256(pipeline / "predictions.pt"),
            "target_oof": str(pipeline / "router/oof/router_train.jsonl"),
            "target_oof_sha256": sha256(pipeline / "router/oof/router_train.jsonl"),
        },
    }
    write_json(summary, destination / "metrics.json")
    with (destination / "decisions.jsonl").open("w", encoding="utf-8") as handle:
        for offset, node_id in enumerate(test_ids.tolist()):
            same = bool(target.gnn_pred[node_id] == target.llm_pred[node_id])
            route_g = bool(test_choose_g[offset])
            row = {
                "node_id": node_id,
                "true_label": int(target.labels[node_id]),
                "gnn_pred": int(target.gnn_pred[node_id]),
                "llm_pred": int(target.llm_pred[node_id]),
                "route": "gnn" if route_g else "llm",
                "selected_pred": int(target.gnn_pred[node_id] if route_g else target.llm_pred[node_id]),
                "correct": bool(test_correct[offset]),
                "agreement": same,
                "preference_score": float(score[node_id]),
                "factorized_trust": float(q_factorized[node_id]),
                "direct_trust": float(q_direct[node_id]),
                "selected_trust": float(q_final[node_id]),
                "trust_checkpoint": selected_trust,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"[{target_name}] B={b_test['router_accuracy']:.4f} best={b_test['best_single_accuracy']:.4f} C={selected_trust} NLL={final_test['nll']:.4f}", flush=True)
    return summary


def aggregate(root: Path, names: list[str]) -> None:
    summaries = [json.loads((root / name / "metrics.json").read_text()) for name in names]
    rows = []
    for summary in summaries:
        b = summary["scheme_b"]["test"]
        c = summary["scheme_c"]["metrics"]["selected"]
        rows.append(
            {
                "dataset": summary["dataset"],
                "gnn": b["gnn_accuracy"],
                "llm": b["llm_accuracy"],
                "best_single": b["best_single_accuracy"],
                "scheme_b": b["router_accuracy"],
                "gain": b["gain_over_best_single"],
                "oracle": b["oracle_accuracy"],
                "oracle_gap": b["oracle_gap"],
                "checkpoint": summary["scheme_b"]["selected_metadata"]["name"],
                "c_checkpoint": summary["scheme_c"]["metrics"]["validation_selection"]["selected"],
                "c_nll": c["nll"],
                "c_brier": c["brier"],
                "c_ece": c["ece"],
                "c_aurc": c["aurc"],
            }
        )
    write_json({"datasets": rows}, root / "summary.json")
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Improved Router B/C Controlled Search",
        "",
        "| Dataset | Best single | Scheme B | Gain | Oracle | Oracle gap | B checkpoint | C trust | C NLL | C Brier | C ECE | C AURC |",
        "|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['best_single']:.4f} | {row['scheme_b']:.4f} | {row['gain']:+.4f} | {row['oracle']:.4f} | {row['oracle_gap']:.4f} | {row['checkpoint']} | {row['c_checkpoint']} | {row['c_nll']:.4f} | {row['c_brier']:.4f} | {row['c_ece']:.4f} | {row['c_aurc']:.4f} |"
        )
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parser().parse_args()
    configure_router_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    requested = [name for name in args.datasets.split(",") if name]
    unknown = sorted(set(requested) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing non-empty output root: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = {
        "input_run_root": str(args.input_run_root.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "datasets": requested,
        "fixed_5shot_seed": args.seed,
        "preference_grid": [asdict(value) for value in preference_config_grid()],
        "validation_only_selection": True,
        "router_threads": 1,
    }
    write_json(config, args.output_root / "config.json")
    raw_supports = {
        name: build_support_domain(name, args.input_run_root / name / "pipeline/router/oof/router_train.jsonl")
        for name in DATASETS
    }
    for name in requested:
        target_dir = args.output_root / name
        if args.resume and (target_dir / "metrics.json").exists():
            print(f"[{name}] complete; skipping", flush=True)
            continue
        with guarded_output_directory(target_dir) as staging:
            run_target(args, name, raw_supports, staging)
    aggregate(args.output_root, requested)


if __name__ == "__main__":
    main()
