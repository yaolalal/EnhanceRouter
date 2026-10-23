"""Five-cell, pre-truth audit and immutable evaluation for Joint-v5 Fixed."""
import csv
import itertools
import math
import statistics
from pathlib import Path

import torch

from eargtc.joint_cross_repair.artifacts import read_json, sha256, write_json
from eargtc.joint_cross_repair.training import blend, scheme_b_probability
from eargtc.joint_five_router.evaluation import (
    _assert_close,
    _bound_label_root,
    _label_rows,
    _positions,
    _seed_summary,
    _verify_historical_expert_identities,
    classification_report,
    correction_damage,
    labels_for_ids,
    verify_selection_reconstruction,
)
from eargtc.joint_five_router.experiment import select_candidates
from eargtc.joint_selector.evaluation import verify_seed_ensemble


def _load_outputs(path):
    path = Path(path)
    return (
        torch.load(path / "predictions.pt", weights_only=True, map_location="cpu"),
        read_json(path / "selection.json"),
        torch.load(path / "validation.pt", weights_only=True, map_location="cpu"),
    )


def verify_full_family(value, name, shape):
    if value["ensemble"].shape != shape or value["seeds"].shape != (3, *shape):
        if value["seeds"].shape[0] != 3:
            raise ValueError(name + " requires exactly three seeds")
        raise ValueError(name + " full node coverage differs")
    for probability in (value["ensemble"], value["seeds"]):
        sums = probability.sum(-1)
        if (
            not torch.isfinite(probability).all()
            or (probability < 0).any()
            or not torch.allclose(sums, torch.ones_like(sums), atol=2e-5, rtol=0)
        ):
            raise ValueError(name + " invalid probabilities")
    verify_seed_ensemble(value, name)


def _ids(value):
    return {int(x) for x in torch.as_tensor(value).tolist()}


def verify_v5_cell_contract(features, pools, metadata, predictions, old_predictions, old_pools=None):
    if not torch.equal(predictions["node_ids"], features["node_ids"]):
        raise ValueError("full node identity differs")
    if not torch.equal(old_predictions["node_ids"], features["node_ids"]):
        raise ValueError("v4 full node identity differs")
    if not torch.equal(predictions["baselines"]["router"], old_predictions["baselines"]["router"]):
        raise ValueError("v4 Router-B probability differs")
    if not torch.equal(
        predictions["baselines"]["router"].argmax(1), old_predictions["baselines"]["router"].argmax(1)
    ):
        raise ValueError("v4 Router-B selection differs")

    if old_pools is None:
        raw = metadata.get("v4_pool_ids")
        if raw is None:
            raise ValueError("v4 pool identities missing")
        old_pools = {name: torch.as_tensor(value, dtype=torch.long) for name, value in raw.items()}
    agreement = _ids(pools["agreement"])
    if agreement & (_ids(pools["teach_g"]) | _ids(pools["teach_l"])):
        raise ValueError("agreement overlaps conflict teaching pools")
    for name in ("teach_g", "teach_l"):
        if _ids(pools[name]) != _ids(old_pools[name]):
            raise ValueError(name + " conflict pool differs from v4")
    for name in ("keep_g", "keep_l"):
        if _ids(pools[name]) != _ids(old_pools[name]) | agreement:
            raise ValueError(name + " must equal v4 keep pool union accepted agreement")
    if _ids(pools["unselected"]) != _ids(old_pools["unselected"]) - agreement:
        raise ValueError("unselected pool must remove accepted agreement only")
    if int(metadata["agreement_count"]) != len(agreement):
        raise ValueError("agreement count differs")
    conflict = len(_ids(pools["teach_g"]) | _ids(pools["teach_l"]))
    if int(metadata["conflict_count"]) != conflict or int(metadata["accepted_count"]) != conflict + len(agreement):
        raise ValueError("accepted/conflict counts differ")


def verify_formal_population(routers, datasets, seeds, records):
    expected = set(itertools.product(routers, datasets))
    observed = {(r["router_id"], r["dataset"]) for r in records}
    if routers != ["v8_v5"] or len(datasets) != 5 or observed != expected or len(records) != 5:
        raise ValueError("all five frozen formal cells required")
    if seeds != [42, 43, 44]:
        raise ValueError("formal evaluation requires seeds 42/43/44")
    members = sum(int(r["member_count"]) for r in records)
    fresh = sum(int(r["trained_count"]) for r in records)
    reused = sum(int(r["reused_count"]) for r in records)
    if any(int(r["member_count"]) != 27 for r in records) or members != 135:
        raise ValueError("formal evaluation requires 27 members per cell and 135 total")
    if fresh != 120 or reused != 15:
        raise ValueError("formal evaluation requires 120 fresh and 15 reused members")
    return {"cells": 5, "members": members, "fresh": fresh, "reused": reused}


def audit_all(run_root):
    from .adapter import V4_EXPORT, load_cell
    from .experiment import ROUTERS, check_reference_config, preflight
    from eargtc.joint_v8.adapter import load_cell as load_v4_cell

    root = Path(run_root).resolve()
    config = preflight(root)
    check_reference_config(config)
    torch.set_num_threads(config["train"]["threads"])
    expected = list(itertools.product(config["routers"], config["datasets"]))
    if expected != list(itertools.product(ROUTERS, config["datasets"])) or len(expected) != 5:
        raise ValueError("all five frozen formal cells required")
    records = {}
    population = []
    history_root = Path(config["history_root"])
    for rid, dataset in expected:
        features, supervision, pools, hidden, metadata = load_cell(
            config["export_root"], rid, dataset, verify_sources=True
        )
        old_features, old_supervision, old_pools, old_hidden, old_metadata = load_v4_cell(
            V4_EXPORT, "v8_v4", dataset, verify_sources=True
        )
        predictions, selection, validation = _load_outputs(root / rid / dataset)
        old_predictions = torch.load(
            history_root / "v8_v4" / dataset / "predictions.pt", weights_only=True, map_location="cpu"
        )
        verify_v5_cell_contract(features, pools, metadata, predictions, old_predictions, old_pools)
        pg, pl = features["logp_g"].exp(), features["logp_l"].exp()
        router = scheme_b_probability(pg, pl, features["choose_g"], features["agreement"])
        for name, value in (("gnn", pg), ("llm", pl), ("router", router)):
            _assert_close(predictions["baselines"][name], value, "baseline " + name + " differs")
        _assert_close(validation["router"], router[supervision["selection_ids"]], "validation Router differs")
        rebuilt = select_candidates(
            validation["candidates"], validation["router"], supervision["selection_targets"]
        )
        if rebuilt != selection:
            raise ValueError("validation selection does not reproduce")
        verify_selection_reconstruction(
            predictions, selection, validation, supervision["selection_ids"], supervision["selection_targets"]
        )
        base = selection["base_fusion"]
        _assert_close(
            predictions["baselines"]["base_fusion"],
            blend(pg, pl, router, base["alpha"], base["beta"]),
            "full base fusion differs",
        )
        if set(predictions["families"]) != {"gold_only", "weighted", "scaled", "joint", "optimized"}:
            raise ValueError("missing candidate family")
        for name, value in predictions["families"].items():
            verify_full_family(value, name, pg.shape)
        cell_meta = read_json(root / rid / dataset / "metadata.json")
        markers = sorted((root / rid / dataset / "bank").glob("seed*/*.json"))
        if len(markers) != 27:
            raise ValueError("formal cell bank must contain 27 committed members")
        reuse_from_markers = sum(read_json(marker).get("reused_from") is not None for marker in markers)
        if (int(cell_meta["member_count"]) != len(markers)
                or int(cell_meta["reused_count"]) != reuse_from_markers
                or int(cell_meta["trained_count"]) != len(markers) - reuse_from_markers):
            raise ValueError("cell member metadata differs from committed bank")
        population.append(cell_meta)
        records[f"{rid}/{dataset}"] = {
            "accepted": int(metadata["accepted_count"]),
            "agreement": int(metadata["agreement_count"]),
            "conflict": int(metadata["conflict_count"]),
            "test": len(supervision["test_ids"]),
            "gold": len(supervision["gold_ids"]),
            "selection": len(supervision["selection_ids"]),
            "pool_counts": {name: len(value) for name, value in pools.items()},
            "validation_reproduced": True,
            "v4_router_probability_identical": True,
        }
        print("Pretest validated", rid, dataset, flush=True)
        del features, supervision, pools, hidden, predictions, validation, old_predictions
        del old_features, old_supervision, old_pools, old_hidden, old_metadata, pg, pl, router
    budget = verify_formal_population(config["routers"], config["datasets"], config["seeds"], population)
    audit = {
        "all_checks_pass": True,
        "cell_count": 5,
        "test_labels_read": False,
        "frozen_sha256": sha256(root / "FROZEN.json"),
        "budget": budget,
        "cells": records,
    }
    path = root / "PRETEST_AUDIT.json"
    if path.exists():
        if read_json(path) != audit:
            raise ValueError("immutable pretest audit differs")
    else:
        write_json(path, audit)
    return config, audit


def _csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_all(run_root):
    root = Path(run_root).resolve()
    output = root / "evaluation"
    if output.exists():
        raise FileExistsError("evaluation is immutable")
    stage = root / "evaluation.staging"
    if stage.exists():
        raise FileExistsError("evaluation staging exists; inspect previous attempt")
    config, audit = audit_all(root)
    stage.mkdir()
    from .adapter import load_cell

    label_root = _bound_label_root(config, config["label_root"])
    labels = {dataset: _label_rows(label_root / dataset / "decisions.jsonl") for dataset in config["datasets"]}
    rows, cells = [], {}
    for rid, dataset in itertools.product(config["routers"], config["datasets"]):
        features, supervision, pools, hidden, metadata = load_cell(
            config["export_root"], rid, dataset, verify_sources=True
        )
        predictions, selection, validation = _load_outputs(root / rid / dataset)
        ids = supervision["test_ids"]
        positions = _positions(predictions["node_ids"], ids)
        targets = labels_for_ids(labels[dataset], ids)
        baseline = predictions["baselines"]
        optimized = predictions["families"]["optimized"]
        probability = optimized["ensemble"][positions]
        _verify_historical_expert_identities(
            labels[dataset], ids, baseline["gnn"][positions], baseline["llm"][positions]
        )
        metrics = {name: classification_report(value[positions], targets) for name, value in baseline.items()}
        metrics.update(
            {name: classification_report(value["ensemble"][positions], targets) for name, value in predictions["families"].items()}
        )
        seed = _seed_summary(optimized["seeds"], positions, targets)
        accepted_ids = torch.cat([pools["teach_g"], pools["teach_l"], pools["agreement"]]).unique()
        accepted = torch.isin(ids, accepted_ids)
        groups = {
            name: classification_report(probability[mask], targets[mask])
            for name, mask in (("accepted", accepted), ("unaccepted", ~accepted))
        }
        damage = correction_damage(baseline["router"][positions].argmax(1), probability.argmax(1), targets)
        delta = metrics["optimized"]["accuracy"] - metrics["router"]["accuracy"]
        if not math.isclose(damage["gain"], delta, abs_tol=1e-10):
            raise ValueError("correction/damage mismatch")
        cells[f"{rid}/{dataset}"] = {
            "router_id": rid, "dataset": dataset, "metrics": metrics, "selection": selection,
            "seed_metrics": seed, "accepted_count": len(accepted_ids),
            "agreement_count": int(metadata["agreement_count"]), "conflict_count": int(metadata["conflict_count"]),
            "test_count": len(ids), "groups": groups,
            "pool_counts": {name: len(value) for name, value in pools.items()},
            "optimized_vs_router": damage,
            "test_label_sha256": sha256(label_root / dataset / "decisions.jsonl"),
        }
        rows.append({
            "router": rid, "dataset": dataset,
            **{key: metrics["optimized"][key] for key in ("accuracy", "macro_f1", "nll")},
            "router_accuracy": metrics["router"]["accuracy"], "delta_router": delta,
            "v4_router_accuracy": metrics["router"]["accuracy"],
            "gold_only_accuracy": metrics["gold_only"]["accuracy"],
            "base_fusion_accuracy": metrics["base_fusion"]["accuracy"],
            "seed_accuracy_mean": seed["accuracy"]["mean"], "seed_accuracy_std": seed["accuracy"]["std"],
            "accepted_count": len(accepted_ids), "conflict_count": int(metadata["conflict_count"]),
            "agreement_count": int(metadata["agreement_count"]), "test_count": len(ids),
        })
        print("Evaluated", rid, dataset, flush=True)
    payload = {
        "config": {key: config[key] for key in ("routers", "datasets", "architecture", "train", "seeds", "recipes", "protocol")},
        "pretest_audit": audit,
        "cells": cells,
    }
    write_json(stage / "results.json", payload)
    _csv(stage / "performance.csv", rows)
    by_key = {(row["router"], row["dataset"]): row for row in rows}
    matrices = {}
    for metric in ("accuracy", "macro_f1", "delta_router", "router_accuracy", "gold_only_accuracy", "base_fusion_accuracy"):
        matrix = []
        for rid in config["routers"]:
            row = {"router": rid, **{dataset: by_key[rid, dataset][metric] for dataset in config["datasets"]}}
            row["mean"] = statistics.fmean(row[dataset] for dataset in config["datasets"])
            matrix.append(row)
        _csv(stage / (metric + "_matrix.csv"), matrix)
        matrices[metric] = matrix
    write_report(stage / "REPORT.md", payload, matrices)
    write_json(stage / "complete.json", {
        "all_checks_pass": True, "preflight_cells": 5, "evaluated_cells": 5,
        "frozen_sha256": sha256(root / "FROZEN.json"),
        "pretest_audit_sha256": sha256(root / "PRETEST_AUDIT.json"),
        "artifacts": {path.name: sha256(path) for path in stage.iterdir() if path.is_file()},
    })
    stage.rename(output)
    return payload


def write_report(path, payload, matrices):
    datasets = payload["config"]["datasets"]
    lines = [
        "# Router-v8 v5 accepted-agreement into preserved Joint-Fixed",
        "",
        "All five datasets use the unchanged frozen V2 Joint model and validation-only selection. "
        "v5 adds accepted agreement examples only to each expert's keep-KL pool; it adds no agreement CE term. "
        "The full Router-B probability is byte-for-byte inherited from v4 and is audited before test labels are read.",
        "",
        "Accuracy uses argmax of the three-seed mean probability. Seed mean and standard deviation describe individual "
        "seed accuracy on the same split, so they can differ from ensemble accuracy and are not split-level confidence intervals.",
        "",
        "| Dataset | Joint Acc % | Macro-F1 % | NLL | Router-B Acc % | Delta B pp | Accepted | Conflict | Agreement | Selected family |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for dataset in datasets:
        cell = payload["cells"]["v8_v5/" + dataset]
        m, selection = cell["metrics"], cell["selection"]["optimized"]
        lines.append(
            f"| {dataset} | {100*m['optimized']['accuracy']:.2f} | {100*m['optimized']['macro_f1']:.2f} | "
            f"{m['optimized']['nll']:.6f} | {100*m['router']['accuracy']:.2f} | "
            f"{100*(m['optimized']['accuracy']-m['router']['accuracy']):+.2f} | {cell['accepted_count']} | "
            f"{cell['conflict_count']} | {cell['agreement_count']} | {selection['family']} |"
        )
    lines += [
        "", "The optimized candidate is selected on the frozen model-selection validation half, not by test accuracy. "
        "Historical validation and test results were previously observed; this is a continuation of that development protocol "
        "and is not a fresh held-out benchmark. Accepted-set accuracy is not full-test classification accuracy.", ""
    ]
    Path(path).write_text("\n".join(lines))
