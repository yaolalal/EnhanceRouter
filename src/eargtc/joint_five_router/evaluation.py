"""Locked, evaluation-only reporting for the Joint five-Router experiment.

This module deliberately keeps the validation/provenance pass separate from
opening the historical JSONL labels.  ``evaluate_all`` performs that pass for
every Router/dataset cell before it reads a single ``true_label`` field.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

import torch

from eargtc.joint_cross_repair.artifacts import read_json, sha256, verify_files, write_json
from eargtc.joint_cross_repair.training import blend, scheme_b_probability

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABEL_ROOT = ROOT / "runs/router_bc_improved_structural_oof_fixed_20260729_v1"


def labels_for_ids(rows: list[dict[str, Any]], ids: torch.Tensor) -> torch.Tensor:
    """Return historical labels in the exact requested node-ID order."""
    if not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.dtype != torch.long:
        raise ValueError("evaluation IDs must be an int64 vector")
    mapping = {int(row["node_id"]): int(row["true_label"]) for row in rows}
    if len(mapping) != len(rows):
        raise ValueError("duplicate evaluation labels")
    if ids.unique().numel() != ids.numel() or set(mapping) != set(ids.tolist()):
        raise ValueError("evaluation labels do not exactly cover test IDs")
    return torch.tensor([mapping[int(node_id)] for node_id in ids.tolist()], dtype=torch.long)


def _verify_historical_expert_identities(rows: list[dict[str, Any]], ids: torch.Tensor, gnn: torch.Tensor, llm: torch.Tensor) -> None:
    """Bind historical JSONL only to the unchanged expert predictions."""
    if not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.dtype != torch.long:
        raise ValueError("evaluation IDs must be an int64 vector")
    mapping = {int(row["node_id"]): row for row in rows}
    if len(mapping) != len(rows) or set(mapping) != set(ids.tolist()):
        raise ValueError("historical expert rows do not exactly cover test IDs")
    gnn, llm = _probability(gnn, "saved GNN probabilities"), _probability(llm, "saved LLM probabilities")
    if gnn.shape[0] != ids.numel() or llm.shape != gnn.shape:
        raise ValueError("saved expert predictions do not align with test IDs")
    expected_g = torch.tensor([int(mapping[int(node_id)]["gnn_pred"]) for node_id in ids.tolist()], dtype=torch.long)
    expected_l = torch.tensor([int(mapping[int(node_id)]["llm_pred"]) for node_id in ids.tolist()], dtype=torch.long)
    if not torch.equal(gnn.argmax(1), expected_g) or not torch.equal(llm.argmax(1), expected_l):
        raise ValueError("saved expert predictions differ from historical expert identities")


def _probability(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must be a nonempty [N,C] tensor")
    value = value.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
        raise ValueError(f"{name} contains invalid probabilities")
    if not bool(torch.allclose(value.sum(1), torch.ones(value.shape[0], dtype=value.dtype), atol=2e-5, rtol=0.0)):
        raise ValueError(f"{name} is not normalized")
    return value


def _native_probability(value: Any, name: str) -> torch.Tensor:
    _probability(value, name)
    return value.detach().to(device="cpu")


def classification_report(probability: torch.Tensor, y: torch.Tensor) -> dict[str, float | int | None]:
    probability = _probability(probability, "prediction")
    if not isinstance(y, torch.Tensor) or y.ndim != 1 or y.numel() != probability.shape[0]:
        raise ValueError("evaluation targets do not align with predictions")
    y = y.to(device="cpu", dtype=torch.long)
    if y.numel() == 0:
        return {"count": 0, "accuracy": None, "macro_f1": None, "nll": None}
    if bool((y < 0).any()) or bool((y >= probability.shape[1]).any()):
        raise ValueError("evaluation target is outside prediction class space")
    predicted = probability.argmax(1)
    classes = probability.shape[1]
    matrix = torch.bincount(y * classes + predicted, minlength=classes * classes).reshape(classes, classes).double()
    true_positive = matrix.diag()
    denominator = matrix.sum(0) + matrix.sum(1)
    f1 = torch.where(denominator > 0, 2 * true_positive / denominator, torch.zeros_like(denominator))
    rows = torch.arange(y.numel())
    return {"count": int(y.numel()), "accuracy": float((predicted == y).double().mean()),
            "macro_f1": float(f1.mean()), "nll": float(-probability[rows, y].clamp_min(1e-12).log().mean())}


def correction_damage(baseline: torch.Tensor, chosen: torch.Tensor, y: torch.Tensor) -> dict[str, int | float | None]:
    baseline, chosen, y = (torch.as_tensor(x, dtype=torch.long).view(-1) for x in (baseline, chosen, y))
    if baseline.numel() != chosen.numel() or chosen.numel() != y.numel():
        raise ValueError("correction/damage vectors must align")
    if not y.numel():
        return {"corrections": 0, "damages": 0, "stable_correct": 0, "stable_wrong": 0, "count": 0, "gain": None}
    before, after = baseline == y, chosen == y
    result = {"corrections": int((~before & after).sum()), "damages": int((before & ~after).sum()),
              "stable_correct": int((before & after).sum()), "stable_wrong": int((~before & ~after).sum()),
              "count": int(y.numel())}
    result["gain"] = (result["corrections"] - result["damages"]) / result["count"]
    return result


def _score(probability: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    report = classification_report(probability, y)
    return float(report["accuracy"]), float(report["nll"])


def _by_id(candidates: list[dict[str, Any]], branch: str) -> dict[str, dict[str, Any]]:
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"validation candidates for {branch} are missing")
    result = {candidate.get("id"): candidate for candidate in candidates if isinstance(candidate, dict)}
    if len(result) != len(candidates) or None in result:
        raise ValueError(f"validation candidate IDs for {branch} are invalid")
    return result


def _blend(g: torch.Tensor, l: torch.Tensor, pb: torch.Tensor, pick: dict[str, Any]) -> torch.Tensor:
    try:
        alpha, beta = float(pick["alpha"]), float(pick["beta"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("selection pick lacks numeric alpha/beta") from error
    if alpha not in (0.0, 0.5, 1.0) or beta not in (0.0, 0.25, 0.5, 1.0):
        raise ValueError("selection pick uses an unfrozen fusion parameter")
    return blend(g, l, pb, alpha, beta)


def _assert_close(observed: torch.Tensor, expected: torch.Tensor, message: str) -> None:
    observed, expected = _probability(observed, message), _probability(expected, message)
    if observed.shape != expected.shape or not torch.allclose(observed, expected, atol=3e-6, rtol=0.0):
        raise ValueError(message)
    if not torch.equal(observed.argmax(1), expected.argmax(1)):
        raise ValueError(message + " argmax")


def _positions(node_ids: torch.Tensor, requested: torch.Tensor) -> torch.Tensor:
    positions = {int(value): index for index, value in enumerate(node_ids.tolist())}
    if len(positions) != node_ids.numel() or any(int(value) not in positions for value in requested.tolist()):
        raise ValueError("prediction IDs do not cover requested node IDs")
    return torch.tensor([positions[int(value)] for value in requested.tolist()], dtype=torch.long)


def verify_selection_reconstruction(predictions: dict[str, Any], selection: dict[str, Any], validation: dict[str, Any], selection_ids: torch.Tensor, selection_targets: torch.Tensor) -> None:
    """Rebuild every selected validation fusion and compare it to saved tensors."""
    if not torch.equal(validation.get("node_ids"), selection_ids):
        raise ValueError("validation node IDs differ from safe selection IDs")
    node_ids = predictions.get("node_ids")
    if not isinstance(node_ids, torch.Tensor) or node_ids.ndim != 1:
        raise ValueError("prediction node IDs are missing")
    positions = _positions(node_ids, selection_ids)
    candidates = validation.get("candidates")
    if not isinstance(candidates, dict):
        raise ValueError("validation candidates are missing")
    g_by_id, l_by_id = _by_id(candidates.get("g"), "g"), _by_id(candidates.get("l"), "l")
    pb = _native_probability(validation.get("router"), "validation router")
    if pb.shape[0] != selection_ids.numel():
        raise ValueError("validation router does not align with selection IDs")
    families = predictions.get("families")
    if not isinstance(families, dict) or not isinstance(selection.get("families"), dict):
        raise ValueError("saved family selections are missing")

    def rebuilt(pick: dict[str, Any]) -> torch.Tensor:
        try:
            g, l = g_by_id[pick["g_id"]], l_by_id[pick["l_id"]]
        except KeyError as error:
            raise ValueError("selection pick references a missing validation candidate") from error
        value = _blend(_native_probability(g.get("probability"), "G candidate"), _native_probability(l.get("probability"), "L candidate"), pb, pick)
        accuracy, nll = _score(value, selection_targets)
        if not math.isclose(accuracy, float(pick["accuracy"]), abs_tol=2e-8) or not math.isclose(nll, float(pick["nll"]), abs_tol=2e-6):
            raise ValueError("saved selection score does not reproduce")
        return value

    rebuilt_by_family: dict[str, torch.Tensor] = {}
    for family in ("gold_only", "weighted", "scaled", "joint"):
        if family not in selection["families"] or family not in families:
            raise ValueError(f"missing {family} family")
        value = rebuilt(selection["families"][family]); rebuilt_by_family[family] = value
        saved = families[family]
        if not isinstance(saved, dict):
            raise ValueError(f"invalid saved {family} family")
        _assert_close(saved.get("ensemble")[positions], value, f"{family} full prediction fails validation reconstruction")
        seeds = saved.get("seeds")
        if not isinstance(seeds, torch.Tensor) or seeds.ndim != 3 or seeds.shape[0] != 3:
            raise ValueError(f"{family} must retain exactly three seed predictions")
        _assert_close(seeds.mean(0)[positions], value, f"{family} seed mean fails validation reconstruction")

    base_pick = selection.get("base_fusion")
    if not isinstance(base_pick, dict):
        raise ValueError("base fusion pick is missing")
    base = rebuilt(base_pick)
    baselines = predictions.get("baselines")
    if not isinstance(baselines, dict):
        raise ValueError("saved baselines are missing")
    _assert_close(baselines.get("base_fusion")[positions], base, "base fusion fails validation reconstruction")
    optimized = selection.get("optimized")
    if not isinstance(optimized, dict) or not isinstance(optimized.get("family"), str):
        raise ValueError("optimized selection is missing")
    name = optimized["family"]
    saved_optimized = families.get("optimized")
    if not isinstance(saved_optimized, dict):
        raise ValueError("optimized full prediction is missing")
    if name in rebuilt_by_family:
        if optimized != {"family": name, **selection["families"][name]}:
            raise ValueError("optimized record differs from its chosen family")
        expected = rebuilt_by_family[name]
        _assert_close(saved_optimized.get("ensemble")[positions], expected, "optimized family does not reproduce selected family")
        _assert_close(saved_optimized.get("ensemble"), families[name].get("ensemble"), "optimized full tensor differs from chosen family")
    elif name in ("router", "base_fusion"):
        expected = baselines[name]
        _assert_close(saved_optimized.get("ensemble"), expected, "optimized full tensor differs from chosen baseline")
    else:
        raise ValueError("optimized family is not an allowed Router fallback")


def _load_router_decisions(path: Path, dataset: str, router_id: str, test_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sanitize a mixed legacy artifact to the two fields permitted pre-label."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    try:
        record = raw[dataset][router_id]
        ids, accepted, selected = record["node_ids"], record["accepted"], record["selected_pred"]
    except (KeyError, TypeError) as error:
        raise ValueError("frozen Router decision artifact lacks whitelisted fields") from error
    if not all(isinstance(value, torch.Tensor) and value.ndim == 1 for value in (ids, accepted, selected)):
        raise ValueError("frozen Router decision fields are malformed")
    if not torch.equal(ids.to(dtype=torch.long), test_ids):
        raise ValueError("frozen Router decision IDs differ from safe test IDs")
    if accepted.dtype != torch.bool or selected.dtype == torch.bool or accepted.shape != selected.shape:
        raise ValueError("frozen Router accepted/prediction fields are malformed")
    return accepted, selected.to(dtype=torch.long)


def _safe_router_probability(features: dict[str, Any]) -> torch.Tensor:
    value = scheme_b_probability(features["logp_g"].exp(), features["logp_l"].exp(), features["choose_g"], features["agreement"])
    if not torch.equal(value.argmax(1), features["selected_pred"]):
        raise ValueError("safe Router probability does not reproduce selected predictions")
    return value


def _same_json(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(right, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _preflight_all(run_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate every complete cell without opening the test-label JSONL files."""
    from .adapter import ROUTERS, load_cell
    from .experiment import preflight, select_candidates

    config = preflight(run_root)
    if bool(config.get("smoke")):
        raise ValueError("smoke experiment cannot be evaluated as final")
    router_ids, datasets = list(config["routers"]), list(config["datasets"])
    if router_ids != list(ROUTERS) or len(router_ids) != 5 or len(datasets) != 5:
        raise ValueError("final evaluation requires the fixed five Routers and five datasets")
    decisions_path = Path(config.get("source_root", "")) / "test/decisions.pt"
    if not decisions_path.exists():
        raise ValueError("frozen Router decisions artifact is missing")
    cells = []
    for router_id in router_ids:
        for dataset in datasets:
            features, supervision, pools, _hidden, metadata = load_cell(config["export_root"], router_id, dataset, verify_sources=True)
            cell_root = run_root / router_id / dataset
            complete = read_json(cell_root / "complete.json")
            if not complete.get("all_checks_pass") or complete.get("test_labels_read") is not False:
                raise ValueError("cell was not completed without test labels")
            verify_files({str(cell_root / name): digest for name, digest in complete["artifacts"].items()})
            predictions = torch.load(cell_root / "predictions.pt", map_location="cpu", weights_only=True)
            if not isinstance(predictions.get("node_ids"), torch.Tensor) or not torch.equal(predictions["node_ids"], features["node_ids"]):
                raise ValueError("prediction node IDs must exactly equal safe feature node IDs")
            baselines = predictions.get("baselines")
            if not isinstance(baselines, dict):
                raise ValueError("saved baselines are missing")
            safe_gnn, safe_llm = features["logp_g"].exp(), features["logp_l"].exp()
            for name, expected in (("gnn", safe_gnn), ("llm", safe_llm)):
                if name not in baselines:
                    raise ValueError(f"saved {name} baseline is missing")
                _assert_close(baselines[name], expected, f"saved {name} baseline differs from safe probabilities")
            selection = read_json(cell_root / "selection.json")
            validation = torch.load(cell_root / "validation.pt", map_location="cpu", weights_only=True)
            router = _safe_router_probability(features)
            if "router" not in baselines:
                raise ValueError("saved router baseline is missing")
            _assert_close(baselines["router"], router, "saved router baseline differs from safe probabilities")
            if not torch.allclose(_probability(validation.get("router"), "validation router"), router[supervision["selection_ids"]].double(), atol=3e-6, rtol=0.0):
                raise ValueError("saved validation Router does not reproduce safe features")
            reconstructed = select_candidates(validation["candidates"], validation["router"], supervision["selection_targets"])
            if not _same_json(reconstructed, selection):
                raise ValueError("saved selection differs from deterministic candidate selection")
            verify_selection_reconstruction(predictions, selection, validation, supervision["selection_ids"], supervision["selection_targets"])
            accepted, selected = _load_router_decisions(decisions_path, dataset, router_id, supervision["test_ids"])
            test_positions = _positions(features["node_ids"], supervision["test_ids"])
            if not torch.equal(router[test_positions].argmax(1), selected):
                raise ValueError("safe Router output differs from frozen selected predictions")
            accepted_ids = torch.cat([pools["teach_g"], pools["teach_l"], pools["agreement"]]).unique()
            observed_accepted = torch.isin(supervision["test_ids"], accepted_ids)
            if not torch.equal(observed_accepted, accepted):
                raise ValueError("safe accepted mask differs from frozen Router decisions")
            cells.append({"router_id": router_id, "dataset": dataset, "features": features, "supervision": supervision,
                          "pools": pools, "metadata": metadata, "predictions": predictions, "router_probability": router,
                          "selection": selection, "accepted": accepted, "source_decisions": decisions_path})
    return config, cells


def _seed_summary(seed_probability: torch.Tensor, positions: torch.Tensor, y: torch.Tensor) -> dict[str, dict[str, float]]:
    if not isinstance(seed_probability, torch.Tensor) or seed_probability.ndim != 3 or seed_probability.shape[0] != 3:
        raise ValueError("optimized prediction must retain three seeds")
    reports = [classification_report(seed_probability[index][positions], y) for index in range(3)]
    return {key: {"mean": statistics.fmean(float(report[key]) for report in reports),
                  "std": statistics.stdev(float(report[key]) for report in reports)}
            for key in ("accuracy", "macro_f1", "nll")}


def _label_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not {"node_id", "true_label", "gnn_pred", "llm_pred"}.issubset(row) for row in rows):
        raise ValueError("historical label source lacks required fields")
    return rows


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _matrix_csv(path: Path, routers: list[str], datasets: list[str], values: dict[tuple[str, str], float]) -> None:
    rows = []
    for router in routers:
        row = {"router": router, **{dataset: values[(router, dataset)] for dataset in datasets}}
        row["mean"] = statistics.fmean(row[dataset] for dataset in datasets)
        rows.append(row)
    _write_csv(path, ["router", *datasets, "mean"], rows)


def _atomic_text(path: Path, text: str) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def _bound_label_root(config: dict[str, Any], label_root: str | Path) -> Path:
    if not isinstance(config.get("label_root"), str):
        raise ValueError("frozen CONFIG lacks label_root")
    value, expected = Path(label_root).resolve(), Path(config["label_root"]).resolve()
    if value != expected:
        raise ValueError("label root must match the frozen CONFIG label_root")
    return value


def evaluate_all(run_root: str | Path, *, label_root: str | Path = DEFAULT_LABEL_ROOT) -> dict[str, Any]:
    """Run the one permitted final label evaluation after a complete safe preflight."""
    run_root = Path(run_root).resolve()
    output = run_root / "evaluation"
    if output.exists():
        raise FileExistsError("evaluation output already exists and is immutable")
    config, cells = _preflight_all(run_root)  # This line is intentionally before _label_rows.
    label_root = _bound_label_root(config, label_root)
    detail, results = [], {}
    for cell in cells:
        test_ids = cell["supervision"]["test_ids"]
        labels_path = label_root / cell["dataset"] / "decisions.jsonl"
        rows = _label_rows(labels_path)
        y = labels_for_ids(rows, test_ids)
        positions = _positions(cell["predictions"]["node_ids"], test_ids)
        baseline = cell["predictions"]["baselines"]
        families = cell["predictions"]["families"]
        optimized = families["optimized"]
        router = cell["router_probability"][positions]
        base_fusion = baseline["base_fusion"][positions]
        optimized_probability = optimized["ensemble"][positions]
        all_metrics = {name: classification_report(probability[positions], y) for name, probability in baseline.items()}
        all_metrics.update({name: classification_report(record["ensemble"][positions], y) for name, record in families.items()})
        all_metrics["router"] = classification_report(router, y)
        _verify_historical_expert_identities(rows, test_ids, baseline["gnn"][positions], baseline["llm"][positions])
        accepted = cell["accepted"]
        groups = {"accepted": accepted, "unaccepted": ~accepted}
        group_metrics = {name: {model: classification_report(value[mask], y[mask]) for model, value in
                                (("router", router), ("base_fusion", base_fusion), ("optimized", optimized_probability))}
                         for name, mask in groups.items()}
        teacher = router.argmax(1)
        contextual_teacher_accuracy = float((teacher[accepted] == y[accepted]).double().mean()) if bool(accepted.any()) else None
        own_router = correction_damage(router.argmax(1), optimized_probability.argmax(1), y)
        own_base = correction_damage(base_fusion.argmax(1), optimized_probability.argmax(1), y)
        if not math.isclose(float(own_router["gain"]), float(all_metrics["optimized"]["accuracy"]) - float(all_metrics["router"]["accuracy"]), abs_tol=1e-10):
            raise ValueError("optimized/Router correction gain does not equal metric delta")
        if not math.isclose(float(own_base["gain"]), float(all_metrics["optimized"]["accuracy"]) - float(all_metrics["base_fusion"]["accuracy"]), abs_tol=1e-10):
            raise ValueError("optimized/base-fusion correction gain does not equal metric delta")
        seed_summary = _seed_summary(optimized["seeds"], positions, y)
        key = f"{cell['router_id']}/{cell['dataset']}"
        results[key] = {"router_id": cell["router_id"], "dataset": cell["dataset"], "selection": cell["selection"],
                        "metrics": all_metrics, "seed_metrics": seed_summary, "groups": group_metrics,
                        "optimized_vs_router": own_router, "optimized_vs_base_fusion": own_base,
                        "context_accepted_teacher_accuracy": contextual_teacher_accuracy,
                        "test_label_sha256": sha256(labels_path), "router_decisions_sha256": sha256(cell["source_decisions"]),
                        "accepted_count": int(accepted.sum()), "test_count": int(accepted.numel())}
        detail.append({"router": cell["router_id"], "dataset": cell["dataset"], "accuracy": all_metrics["optimized"]["accuracy"],
                       "macro_f1": all_metrics["optimized"]["macro_f1"], "nll": all_metrics["optimized"]["nll"],
                       "seed_accuracy_mean": seed_summary["accuracy"]["mean"], "seed_accuracy_std": seed_summary["accuracy"]["std"],
                       "accepted_count": int(accepted.sum()), "test_count": int(accepted.numel()),
                       "accepted_accuracy": group_metrics["accepted"]["optimized"]["accuracy"],
                       "unaccepted_accuracy": group_metrics["unaccepted"]["optimized"]["accuracy"],
                       "router_delta": own_router["gain"], "base_fusion_delta": own_base["gain"],
                       "context_accepted_teacher_accuracy": contextual_teacher_accuracy})
    routers, datasets = list(config["routers"]), list(config["datasets"])
    optimized_values = {(row["router"], row["dataset"]): float(row["accuracy"]) for row in detail}
    router_values = {(key.split("/")[0], key.split("/")[1]): float(value["metrics"]["router"]["accuracy"]) for key, value in results.items()}
    delta_values = {(row["router"], row["dataset"]): float(row["router_delta"]) for row in detail}
    aggregate = {"optimized_accuracy_mean": statistics.fmean(row["accuracy"] for row in detail),
                 "router_accuracy_mean": statistics.fmean(router_values[(router, dataset)] for router in routers for dataset in datasets),
                 "delta_vs_router_mean": statistics.fmean(row["router_delta"] for row in detail)}
    payload = {"config": {key: config[key] for key in ("routers", "datasets", "seeds", "export_root", "source_root") if key in config},
               "aggregate": aggregate, "cells": results}
    output.mkdir()
    write_json(output / "results.json", payload)
    fields = ["router", "dataset", "accuracy", "macro_f1", "nll", "seed_accuracy_mean", "seed_accuracy_std", "accepted_count", "test_count", "accepted_accuracy", "unaccepted_accuracy", "router_delta", "base_fusion_delta", "context_accepted_teacher_accuracy"]
    _write_csv(output / "performance.csv", fields, detail)
    _matrix_csv(output / "accuracy_matrix.csv", routers, datasets, optimized_values)
    _matrix_csv(output / "router_accuracy_matrix.csv", routers, datasets, router_values)
    _matrix_csv(output / "delta_accuracy_matrix.csv", routers, datasets, delta_values)
    lines = ["# Joint Five Router evaluation", "", "All 25 frozen validation selections and Router identities were reconstructed before the historical test labels were opened.", "",
             "| Router | Dataset | Accuracy | Macro-F1 | NLL | Seed accuracy (mean ± SD) | Δ Router | Δ base fusion |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in detail:
        lines.append(f"| {row['router']} | {row['dataset']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['nll']:.4f} | {row['seed_accuracy_mean']:.4f} ± {row['seed_accuracy_std']:.4f} | {row['router_delta']:+.4f} | {row['base_fusion_delta']:+.4f} |")
    lines += ["", f"Equal-cell mean optimized accuracy: {aggregate['optimized_accuracy_mean']:.4f}; mean change versus each cell's Router: {aggregate['delta_vs_router_mean']:+.4f}.",
              "Contextual accepted-teacher accuracy is reported only after selection; it was not an HPO objective."]
    _atomic_text(output / "REPORT.md", "\n".join(lines) + "\n")
    artifacts = {name: sha256(output / name) for name in ("results.json", "performance.csv", "accuracy_matrix.csv", "router_accuracy_matrix.csv", "delta_accuracy_matrix.csv", "REPORT.md")}
    write_json(output / "complete.json", {"all_checks_pass": True, "preflight_cells": len(cells), "artifacts": artifacts,
                                            "frozen_run_sha256": sha256(run_root / "FROZEN.json"),
                                            "frozen_source_sha256": sha256(Path(config["source_root"]) / "test/decisions.pt")})
    return payload


__all__ = ["DEFAULT_LABEL_ROOT", "labels_for_ids", "_bound_label_root", "classification_report", "correction_damage", "verify_selection_reconstruction", "evaluate_all"]
