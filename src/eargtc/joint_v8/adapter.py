"""Hash-bound, label-isolated Router-v8 adapter for unchanged Joint."""
from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from eargtc.joint_cross_repair.artifacts import (
    DATASETS, id_hash, read_json, save_tensor, sha256, validate_pools,
    validate_supervision, verify_files, write_json,
)
from eargtc.joint_five_router.adapter import (
    _safe_base_export, build_router_features_and_pools,
)


ROOT = Path(__file__).resolve().parents[2]
BASE_EXPORT = ROOT / "runs/joint_cross_repair_export_20260906_v2"
DEFAULT_EXPORT = ROOT / "runs/joint_v8_export_20260908_v1"
ROUTERS = ("v8_v3", "v8_v4")
SOURCE_ROOTS = {
    "v8_v3": ROOT / "runs/router_v8_calibration_selection_20260907_v3",
    "v8_v4": ROOT / "runs/router_v8_learned_20260907_v4",
}
_INPUT_SCHEMA = frozenset({"node_ids", "accepted_ids", "choose_g", "trust", "selected_pred", "module_code"})
_SCORE_FIELDS = ("node_ids", "gnn_pred", "llm_pred", "route", "selected_score")
_PIPELINE_SCHEMA = frozenset({"node_ids", "gnn_pred", "llm_pred", "route", "q", "selected_score",
                              "selected_action", "pseudo_label", "accepted_mask"})


def _tensor(value, *, dtype, name):
    result = torch.as_tensor(value, dtype=dtype).detach().cpu().contiguous().view(-1)
    if result.numel() == 0 or not bool(torch.isfinite(result.float()).all()):
        raise ValueError(f"invalid {name}")
    return result


def _score_whitelist(payload):
    """Immediately copy only prediction fields; mixed inputs are never traversed."""
    if not isinstance(payload, dict):
        raise ValueError("invalid score payload")
    try:
        result = {
            "node_ids": _tensor(payload["node_ids"], dtype=torch.long, name="node IDs"),
            "gnn_pred": _tensor(payload["gnn_pred"], dtype=torch.long, name="GNN predictions"),
            "llm_pred": _tensor(payload["llm_pred"], dtype=torch.long, name="LLM predictions"),
            "route": _tensor(payload["route"], dtype=torch.long, name="route"),
            "selected_score": _tensor(payload["selected_score"], dtype=torch.float64, name="selected score"),
        }
    except KeyError as exc:
        raise ValueError("missing safe prediction field") from exc
    n = result["node_ids"].numel()
    if any(value.numel() != n for value in result.values()):
        raise ValueError("misaligned score fields")
    return result


def align_scores(payload, required_ids, base):
    """Join a safe score table to exact required global IDs, independent of row order."""
    score = _score_whitelist(payload)
    ids = score["node_ids"]
    required = _tensor(required_ids, dtype=torch.long, name="required IDs")
    n = base["node_ids"].numel()
    if ids.unique().numel() != ids.numel():
        raise ValueError("duplicate score IDs")
    if required.unique().numel() != required.numel() or bool(((ids < 0) | (ids >= n)).any()):
        raise ValueError("duplicate or out-of-range coverage IDs")
    if ids.numel() != required.numel() or set(ids.tolist()) != set(required.tolist()):
        raise ValueError("score coverage differs from required IDs")
    location = {int(node): index for index, node in enumerate(ids.tolist())}
    order = torch.tensor([location[int(node)] for node in required.tolist()], dtype=torch.long)
    route = score["route"][order]
    if not bool(torch.isin(route, torch.tensor([0, 1])).all()):
        raise ValueError("invalid binary route")
    g = base["logp_g"].argmax(1)[required]
    l = base["logp_l"].argmax(1)[required]
    if not torch.equal(score["gnn_pred"][order], g) or not torch.equal(score["llm_pred"][order], l):
        raise ValueError("router expert predictions differ from Joint cache")
    choose_g = route == 0
    return {"node_ids": required, "choose_g": choose_g,
            "selected_pred": torch.where(choose_g, g, l), "trust": score["selected_score"][order].float()}


def replay_acceptance(node_ids, selected_score, policy):
    ids = _tensor(node_ids, dtype=torch.long, name="acceptance IDs")
    scores = _tensor(selected_score, dtype=torch.float64, name="acceptance scores")
    if ids.numel() != scores.numel() or ids.unique().numel() != ids.numel():
        raise ValueError("misaligned acceptance population")
    if not isinstance(policy, dict) or not isinstance(policy.get("reject_all"), bool):
        raise ValueError("invalid top-fraction policy")
    accepted = np.zeros(ids.numel(), dtype=bool)
    if policy["reject_all"]:
        return torch.from_numpy(accepted)
    fraction = policy.get("fraction")
    if not isinstance(fraction, (int, float)) or not 0 <= float(fraction) <= 1:
        raise ValueError("invalid top fraction")
    exact = Fraction(str(fraction))
    k = min(ids.numel(), max(0, (ids.numel() * exact.numerator + exact.denominator - 1) // exact.denominator))
    order = np.lexsort((ids.numpy(), -scores.numpy()))
    accepted[order[:k]] = True
    return torch.from_numpy(accepted)


def validate_pipeline(pipeline, frozen_scores, accepted, required_ids, base):
    if not isinstance(pipeline, dict) or set(pipeline) != _PIPELINE_SCHEMA:
        raise ValueError("unsafe pipeline schema")
    safe = _score_whitelist(pipeline)
    frozen = _score_whitelist(frozen_scores)
    for key in _SCORE_FIELDS:
        if not torch.equal(safe[key], frozen[key]):
            raise ValueError("pipeline differs from frozen score replay")
    aligned = align_scores(safe, required_ids, base)
    mask = _tensor(pipeline["accepted_mask"], dtype=torch.bool, name="accepted mask")
    action = _tensor(pipeline["selected_action"], dtype=torch.long, name="selected action")
    pseudo = _tensor(pipeline["pseudo_label"], dtype=torch.long, name="pseudo-label")
    expected = _tensor(accepted, dtype=torch.bool, name="expected acceptance")
    if any(value.numel() != safe["node_ids"].numel() for value in (mask, action, pseudo, expected)):
        raise ValueError("misaligned pipeline decisions")
    if not torch.equal(mask, expected):
        raise ValueError("pipeline acceptance differs from policy replay")
    routed = torch.where(safe["route"] == 0, safe["gnn_pred"], safe["llm_pred"])
    if not torch.equal(pseudo, torch.where(mask, routed, torch.full_like(routed, -1))):
        raise ValueError("invalid pipeline pseudo-label")
    if not torch.equal(action, torch.where(mask, safe["route"], torch.full_like(safe["route"], 2))):
        raise ValueError("invalid pipeline selected action")
    return aligned


def _disagreement_ids(base, ids):
    ids = _tensor(ids, dtype=torch.long, name="split IDs")
    g, l = base["logp_g"].argmax(1), base["logp_l"].argmax(1)
    return ids[g[ids] != l[ids]]


def validate_inputs(inputs, n):
    if not isinstance(inputs, dict) or set(inputs) != _INPUT_SCHEMA:
        raise ValueError("safe input schema forbids labels and unknown fields")
    if not torch.equal(inputs["node_ids"], torch.arange(n)):
        raise ValueError("input node identity mismatch")
    for key in ("choose_g", "trust", "selected_pred", "module_code"):
        if inputs[key].shape != (n,):
            raise ValueError("full-node input shape mismatch")
    if inputs["choose_g"].dtype != torch.bool or not bool(torch.isfinite(inputs["trust"]).all()):
        raise ValueError("invalid route/trust inputs")
    if bool(((inputs["trust"] < 0) | (inputs["trust"] > 1)).any()):
        raise ValueError("trust outside probability range")


def build_inputs(base, supervision, validation, test, accepted):
    n = base["node_ids"].numel()
    val_required = _disagreement_ids(base, torch.cat([supervision["selection_ids"], supervision["threshold_ids"]]))
    test_required = _disagreement_ids(base, supervision["test_ids"])
    if not torch.equal(validation["node_ids"], val_required) or not torch.equal(test["node_ids"], test_required):
        raise ValueError("validation/test disagreement coverage mismatch")
    mask = _tensor(accepted, dtype=torch.bool, name="accepted mask")
    if mask.numel() != test_required.numel():
        raise ValueError("acceptance differs from test disagreement population")
    choose_g = torch.ones(n, dtype=torch.bool)
    trust = torch.zeros(n, dtype=torch.float32)
    module_code = torch.full((n,), -1, dtype=torch.int8)
    for aligned in (validation, test):
        ids = aligned["node_ids"]
        choose_g[ids] = aligned["choose_g"]
        trust[ids] = aligned["trust"]
    module_code[supervision["test_ids"]] = 1
    module_code[test_required] = 2
    g, l = base["logp_g"].argmax(1), base["logp_l"].argmax(1)
    inputs = {"node_ids": base["node_ids"], "accepted_ids": test_required[mask], "choose_g": choose_g,
              "trust": trust, "selected_pred": torch.where(choose_g, g, l), "module_code": module_code}
    validate_inputs(inputs, n)
    return inputs


def _verify_source_records(router_id, dataset):
    source = SOURCE_ROOTS[router_id]
    frozen_path = source / "selection/FROZEN.json"
    audit_path = source / "FINAL_AUDIT.json"
    preserved_path = source / "PRESERVED_VERSION.json"
    frozen = read_json(frozen_path)
    audit = read_json(audit_path)
    preserved = read_json(preserved_path)
    frozen_hash = sha256(frozen_path)
    if frozen.get("schema") != "router_v8_freeze_v1" or audit.get("all_checks_pass") is not True:
        raise ValueError("invalid frozen Router audit")
    if audit.get("frozen_sha256") != frozen_hash or preserved.get("frozen_sha256") != frozen_hash:
        raise ValueError("Router freeze binding mismatch")
    relative = [f"targets/{dataset}/scores.pt", f"targets/{dataset}/validation_scores.pt",
                f"targets/{dataset}/selection.json"]
    if router_id == "v8_v3":
        relative.append(f"test/{dataset}_decisions.pt")
    else:
        relative.extend([f"pipeline/{dataset}.pt", "pipeline/MANIFEST.json", "slurm/export_pipeline.py"])
    records = {str(frozen_path.resolve()): frozen_hash, str(audit_path.resolve()): sha256(audit_path),
               str(preserved_path.resolve()): sha256(preserved_path)}
    for rel in relative:
        path = source / rel
        expected = frozen.get("hashes", {}).get(str(path.resolve()))
        if expected is None and router_id == "v8_v3":
            expected = audit.get("result_hashes", {}).get(str(path.resolve()))
        if expected is None and router_id == "v8_v4":
            expected = preserved.get("preserved_files", {}).get(str(path.relative_to(ROOT)))
        if expected is None:
            raise ValueError(f"source is absent from frozen/preserved manifests: {rel}")
        records[str(path.resolve())] = expected
    verify_files(records)
    if router_id == "v8_v4":
        manifest = read_json(source / "pipeline/MANIFEST.json")
        item = manifest.get("datasets", {}).get(dataset, {})
        if (manifest.get("schema") != "router_v8_learned_pipeline_v1" or manifest.get("seed") != 42
                or manifest.get("ground_truth_fields_exported") is not False
                or manifest.get("frozen_sha256") != frozen_hash
                or manifest.get("final_audit_sha256") != sha256(audit_path)
                or manifest.get("export_script_sha256") != sha256(source / "slurm/export_pipeline.py")
                or item.get("sha256") != sha256(source / f"pipeline/{dataset}.pt")):
            raise ValueError("invalid v4 pipeline manifest binding")
        if preserved.get("final_audit_sha256") != sha256(audit_path):
            raise ValueError("v4 preserved audit binding mismatch")
    return records


def _load_decision(router_id, dataset, base, supervision):
    source = SOURCE_ROOTS[router_id]
    score_raw = torch.load(source / f"targets/{dataset}/scores.pt", map_location="cpu", weights_only=True)
    validation_raw = torch.load(source / f"targets/{dataset}/validation_scores.pt", map_location="cpu", weights_only=True)
    score = _score_whitelist(score_raw)
    policy = read_json(source / f"targets/{dataset}/selection.json").get("top_fraction_policy")
    accepted = replay_acceptance(score["node_ids"], score["selected_score"], policy)
    val_ids = _disagreement_ids(base, torch.cat([supervision["selection_ids"], supervision["threshold_ids"]]))
    test_ids = _disagreement_ids(base, supervision["test_ids"])
    validation = align_scores(validation_raw, val_ids, base)
    test = align_scores(score, test_ids, base)
    if router_id == "v8_v3":
        mixed = torch.load(source / f"test/{dataset}_decisions.pt", map_location="cpu", weights_only=True)
        safe_decision = _score_whitelist(mixed)
        for key in _SCORE_FIELDS:
            if not torch.equal(safe_decision[key], score[key]):
                raise ValueError("v3 raw decision handoff differs from frozen scores")
        raw_action = _tensor(mixed["selected_action"], dtype=torch.long, name="v3 selected action")
        raw_pseudo = _tensor(mixed["pseudo_label"], dtype=torch.long, name="v3 pseudo-label")
        routed = torch.where(score["route"] == 0, score["gnn_pred"], score["llm_pred"])
        if not torch.equal(raw_action, torch.where(accepted, score["route"], torch.full_like(score["route"], 2))):
            raise ValueError("v3 raw decision action differs from safe replay")
        if not torch.equal(raw_pseudo, torch.where(accepted, routed, torch.full_like(routed, -1))):
            raise ValueError("v3 raw decision pseudo-label differs from safe replay")
    else:
        pipeline = torch.load(source / f"pipeline/{dataset}.pt", map_location="cpu", weights_only=True)
        test = validate_pipeline(pipeline, score_raw, accepted, test_ids, base)
        manifest_item = read_json(source / "pipeline/MANIFEST.json")["datasets"][dataset]
        if (manifest_item.get("rows") != score["node_ids"].numel()
                or manifest_item.get("accepted") != int(accepted.sum())
                or manifest_item.get("fraction") != policy.get("fraction")):
            raise ValueError("v4 pipeline manifest count/policy mismatch")
    accepted_by_id = {int(node): bool(value) for node, value in zip(score["node_ids"].tolist(), accepted.tolist())}
    accepted_aligned = torch.tensor([accepted_by_id[int(node)] for node in test_ids.tolist()], dtype=torch.bool)
    return validation, test, accepted_aligned


def _prepare_cell(export_root, router_id, dataset):
    dest = export_root / router_id / dataset
    if dest.exists():
        raise FileExistsError(f"refusing to overwrite export: {dest}")
    records = _verify_source_records(router_id, dataset)
    base_path = BASE_EXPORT / dataset
    base, supervision, _old_pools, _hidden, base_meta = _safe_base_export(base_path, verify_sources=True)
    validation, test, accepted = _load_decision(router_id, dataset, base, supervision)
    inputs = build_inputs(base, supervision, validation, test, accepted)
    decision = {"node_ids": supervision["test_ids"], "accepted": torch.isin(supervision["test_ids"], inputs["accepted_ids"]),
                "selected_pred": inputs["selected_pred"][supervision["test_ids"]],
                "module_code": inputs["module_code"][supervision["test_ids"]]}
    features, pools = build_router_features_and_pools(base, supervision["test_ids"], decision,
                                                       inputs["choose_g"], inputs["trust"])
    validate_supervision(supervision, features["node_ids"].numel(), features["logp_g"].shape[1])
    validate_pools(pools, features, supervision)
    base_sources = dict(base_meta["source_hashes"])
    meta = {"schema_version": 1, "router_id": router_id, "dataset": dataset,
            "base_export_dir": str(base_path.resolve()), "base_export_complete_sha256": sha256(base_path / "complete.json"),
            "base_feature_node_ids_sha256": id_hash(base["node_ids"]), "source_hashes": records,
            "base_source_hashes": base_sources, "accepted_count": int(accepted.sum()),
            "candidate_count": int(test["node_ids"].numel()),
            "pool_counts": {key: int(value.numel()) for key, value in pools.items()},
            "llm_hidden_path": base_meta["llm_hidden_path"], "llm_width": base_meta["llm_width"],
            "test_labels_exported": False,
            "default_route_note": "choose GNN on gold, expert-agreement, and any non-validation/non-test placeholder nodes"}
    dest.parent.mkdir(parents=True, exist_ok=True)
    stage = dest.with_name(dest.name + f".staging.{os.getpid()}")
    stage.mkdir()
    save_tensor(stage / "inputs.pt", inputs)
    write_json(stage / "metadata.json", meta)
    write_json(stage / "complete.json", {"schema_version": 1, "router_id": router_id, "dataset": dataset,
                                           "test_labels_exported": False,
                                           "artifacts": {name: sha256(stage / name) for name in ("inputs.pt", "metadata.json")}})
    os.rename(stage, dest)
    return meta


def prepare_all(export_root=DEFAULT_EXPORT):
    export_root = Path(export_root).resolve()
    if export_root.exists():
        raise FileExistsError(f"refusing to overwrite export root: {export_root}")
    export_root.mkdir(parents=True)
    metas = {}
    for router_id in ROUTERS:
        for dataset in DATASETS:
            metas[f"{router_id}/{dataset}"] = _prepare_cell(export_root, router_id, dataset)
    cells = {key: sha256(export_root / key / "complete.json") for key in metas}
    write_json(export_root / "COMPLETE.json", {"schema_version": 1, "cell_count": 10, "test_labels_exported": False, "cells": cells})
    return metas


def load_cell(export_root, router_id, dataset, verify_sources=True):
    if router_id not in ROUTERS or dataset not in DATASETS:
        raise ValueError("unknown router or dataset")
    root = Path(export_root).resolve()
    marker = read_json(root / "COMPLETE.json")
    expected_keys = {f"{router}/{data}" for router in ROUTERS for data in DATASETS}
    if (marker.get("cell_count") != 10 or marker.get("test_labels_exported") is not False
            or set(marker.get("cells", {})) != expected_keys):
        raise ValueError("export does not contain all ten Router-v8 cells")
    key, path = f"{router_id}/{dataset}", root / router_id / dataset
    if marker["cells"].get(key) != sha256(path / "complete.json"):
        raise ValueError("cell completion hash differs from root marker")
    complete = read_json(path / "complete.json")
    if (complete.get("schema_version") != 1 or complete.get("test_labels_exported") is not False
            or set(complete.get("artifacts", {})) != {"inputs.pt", "metadata.json"}):
        raise ValueError("invalid Router-v8 cell completion")
    verify_files({str(path / name): digest for name, digest in complete["artifacts"].items()})
    meta = read_json(path / "metadata.json")
    required = {"router_id", "dataset", "base_export_dir", "base_export_complete_sha256", "source_hashes",
                "base_source_hashes", "accepted_count", "candidate_count", "pool_counts", "llm_hidden_path", "llm_width"}
    if not required.issubset(meta) or meta["router_id"] != router_id or meta["dataset"] != dataset:
        raise ValueError("invalid Router-v8 metadata")
    if verify_sources:
        verify_files(meta["source_hashes"])
        verify_files(meta["base_source_hashes"])
    base_path = Path(meta["base_export_dir"])
    if sha256(base_path / "complete.json") != meta["base_export_complete_sha256"]:
        raise ValueError("base export completion hash mismatch")
    base, supervision, _old_pools, hidden, _base_meta = _safe_base_export(base_path, verify_sources=verify_sources)
    if meta.get("base_feature_node_ids_sha256") != id_hash(base["node_ids"]):
        raise ValueError("base node identity mismatch")
    inputs = torch.load(path / "inputs.pt", map_location="cpu", weights_only=True)
    validate_inputs(inputs, base["node_ids"].numel())
    test_disagreement = _disagreement_ids(base, supervision["test_ids"])
    if not bool(torch.isin(inputs["accepted_ids"], test_disagreement).all()):
        raise ValueError("accepted IDs outside historical test disagreement")
    decision = {"node_ids": supervision["test_ids"], "accepted": torch.isin(supervision["test_ids"], inputs["accepted_ids"]),
                "selected_pred": inputs["selected_pred"][supervision["test_ids"]],
                "module_code": inputs["module_code"][supervision["test_ids"]]}
    features, pools = build_router_features_and_pools(base, supervision["test_ids"], decision,
                                                       inputs["choose_g"], inputs["trust"])
    validate_pools(pools, features, supervision)
    if (inputs["accepted_ids"].numel() != meta["accepted_count"]
            or test_disagreement.numel() != meta["candidate_count"]
            or {key: int(value.numel()) for key, value in pools.items()} != meta["pool_counts"]):
        raise ValueError("Router-v8 metadata count mismatch")
    return features, supervision, pools, hidden, meta
