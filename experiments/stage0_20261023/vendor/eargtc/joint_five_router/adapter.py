"""Create strict safe inputs for the frozen five-router Joint experiment.

This module safely deserializes the hash-bound mixed historical artifacts, but
uses and exports only an explicit label-free field whitelist. Historical
correctness and metrics are never accessed or copied into an export.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from eargtc.joint_cross_repair.artifacts import (
    DATASETS, id_hash, read_json, save_tensor, sha256,
    validate_features, validate_hidden, validate_pools, validate_supervision, verify_files,
    write_json,
)
from eargtc.router_fiveway_topk.assembly import PolicyPopulation
from eargtc.router_fiveway_topk.policy import rank_exact
from eargtc.joint_cross_repair.policy import build_pools


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "runs/router_fiveway_bspecific_c_topk_20260907_v3"
BASE_EXPORT = ROOT / "runs/joint_cross_repair_export_20260906_v2"
DEFAULT_EXPORT = ROOT / "runs/joint_five_router_export_20260907_v1"
ROUTERS = OrderedDict((
    ("r6_legacy_c", "R6-LegacyC"),
    ("r6_split_c", "R6-SplitC"),
    ("r6_expert_conf_c", "R6-EC"),
    ("a20ms_split_c", "A20-SplitC"),
    ("a20emb_v2_split_c", "A20Emb-SplitC"),
))

_SAFE_DECISION = frozenset({"node_ids", "accepted", "selected_pred", "module_code"})
_INPUT_SCHEMA = frozenset({"node_ids", "accepted_ids", "choose_g", "trust", "selected_pred", "module_code"})
_MODULE_CODES = {"global": 0, "agreement": 1, "conflict": 2}
_A20_PROVIDER = {"a20ms_split_c": "a20ms", "a20emb_v2_split_c": "a20emb_v2"}


def _safe_base_export(path: Path, verify_sources: bool = True):
    """Equivalent to old load_export, but every tensor load is explicit weights-only."""
    complete = read_json(path / "complete.json")
    if complete.get("schema_version") != 1 or complete.get("test_labels_exported") is not False:
        raise ValueError("invalid safe base export completion")
    verify_files({str(path / name): digest for name, digest in complete.get("artifacts", {}).items()})
    meta = read_json(path / "metadata.json")
    if verify_sources:
        verify_files(meta["source_hashes"])
    features = torch.load(path / "features.pt", map_location="cpu", weights_only=True)
    supervision = torch.load(path / "supervision.pt", map_location="cpu", weights_only=True)
    pools = torch.load(path / "pools.pt", map_location="cpu", weights_only=True)
    hidden_payload = torch.load(meta["llm_hidden_path"], map_location="cpu", weights_only=True, mmap=True)
    n, c = features["node_ids"].numel(), features["logp_g"].shape[1]
    validate_features(features)
    validate_supervision(supervision, n, c)
    validate_hidden(hidden_payload, n, meta["llm_width"])
    validate_pools(pools, features, supervision)
    candidate = torch.tensor(meta["candidate_ids"], dtype=torch.long)
    expected = build_pools(supervision["test_ids"], features["agreement"], features["choose_g"], features["trust"],
                           read_json(path / "policies.json"), candidate_ids=candidate)
    if any(not torch.equal(pools[name], expected[name]) for name in pools):
        raise ValueError("safe base pools do not reproduce frozen policy")
    return features, supervision, pools, hidden_payload["hidden"], meta


def _source_records() -> dict[str, str]:
    """Verify every recorded frozen source file before deserializing any input."""
    manifest = read_json(SOURCE_ROOT / "manifest.json")
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("invalid frozen five-router manifest")
    records = {str((SOURCE_ROOT / relative).resolve()): item["sha256"]
               for relative, item in manifest["files"].items()
               if isinstance(item, dict) and isinstance(item.get("sha256"), str)}
    if len(records) != manifest.get("file_count"):
        raise ValueError("frozen source manifest is incomplete")
    records[str((SOURCE_ROOT / "manifest.json").resolve())] = sha256(SOURCE_ROOT / "manifest.json")
    verify_files(records)
    frozen = SOURCE_ROOT / "selection/FROZEN.json"
    expected = (SOURCE_ROOT / "selection/FROZEN.sha256").read_text(encoding="utf-8").strip()
    if sha256(frozen) != expected:
        raise ValueError("frozen policy hash mismatch")
    inputs = SOURCE_ROOT / "test/test_inputs.pt"
    if sha256(inputs) != (SOURCE_ROOT / "test/INPUTS.sha256").read_text(encoding="utf-8").strip():
        raise ValueError("frozen test input hash mismatch")
    return records


def _safe_test_inputs():
    """Load only a fixed local dataclass, never arbitrary pickle globals."""
    with torch.serialization.safe_globals([PolicyPopulation]):
        payload = torch.load(SOURCE_ROOT / "test/test_inputs.pt", map_location="cpu", weights_only=True)
    if set(payload) != {"populations"} or not isinstance(payload["populations"], dict):
        raise ValueError("unexpected frozen test-input schema")
    return payload["populations"]


def _safe_decisions():
    payload = torch.load(SOURCE_ROOT / "test/decisions.pt", map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != set(DATASETS):
        raise ValueError("unexpected frozen decision dataset schema")
    return payload


def _tensor(value, *, dtype, name: str) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=dtype).detach().cpu().contiguous().view(-1)
    if result.numel() == 0 or not bool(torch.isfinite(result.float()).all()):
        raise ValueError(f"invalid {name}")
    return result


def _decision_for(payload, dataset: str, router_id: str) -> dict[str, torch.Tensor]:
    item = payload[dataset][router_id]
    if not isinstance(item, dict) or not _SAFE_DECISION.issubset(item):
        raise ValueError("missing safe frozen decision fields")
    # Do not iterate the mixed record or access its correctness/metric values.
    result = {
        "node_ids": _tensor(item["node_ids"], dtype=torch.long, name="decision node IDs"),
        "accepted": _tensor(item["accepted"], dtype=torch.bool, name="accepted mask"),
        "selected_pred": _tensor(item["selected_pred"], dtype=torch.long, name="selected predictions"),
        "module_code": _tensor(item["module_code"], dtype=torch.int8, name="module code"),
    }
    n = result["node_ids"].numel()
    if any(value.numel() != n for value in result.values()) or result["node_ids"].unique().numel() != n:
        raise ValueError("misaligned frozen decisions")
    return result


def accepted_from_policies(node_ids: torch.Tensor, score: torch.Tensor, module_code: torch.Tensor, policies: dict) -> torch.Tensor:
    """Replay historical top-k quotas with Fraction-backed exact ranking."""
    ids = _tensor(node_ids, dtype=torch.long, name="policy IDs")
    scores = _tensor(score, dtype=torch.float64, name="policy scores")
    codes = _tensor(module_code, dtype=torch.int8, name="policy module codes")
    if not (ids.numel() == scores.numel() == codes.numel()) or ids.unique().numel() != ids.numel():
        raise ValueError("misaligned policy population")
    expected = {"global"} if set(codes.tolist()) == {0} else {"agreement", "conflict"}
    if set(policies) != expected:
        raise ValueError("frozen policy modules do not match decision modules")
    accepted = torch.zeros(ids.numel(), dtype=torch.bool)
    for module in expected:
        code = _MODULE_CODES[module]
        loc = torch.nonzero(codes == code, as_tuple=False).view(-1)
        if loc.numel() == 0:
            raise ValueError("empty frozen policy module")
        fraction = policies[module].get("k_fraction")
        if not isinstance(fraction, (float, int)):
            raise ValueError("invalid frozen k fraction")
        accepted[loc] = torch.from_numpy(rank_exact(ids[loc].numpy(), scores[loc].numpy(), float(fraction)))
    return accepted


def _policies(dataset: str, router_id: str) -> dict:
    root = SOURCE_ROOT / "topk_policy" / router_id
    names = ["global"] if router_id == "r6_legacy_c" else ["agreement", "conflict"]
    result = {}
    for name in names:
        value = read_json(root / f"{dataset}_{name}.json")
        if not {"k_fraction", "quota"}.issubset(value):
            raise ValueError("invalid frozen top-k policy")
        result[name] = value
    return result


def _population_scores(populations, dataset: str, router_id: str, decision: dict[str, torch.Tensor]) -> torch.Tensor:
    modules = populations.get(dataset, {}).get(router_id)
    expected_modules = {"global"} if router_id == "r6_legacy_c" else {"agreement", "conflict"}
    if not isinstance(modules, dict) or set(modules) != expected_modules:
        raise ValueError("unexpected frozen population modules")
    test_ids = decision["node_ids"]
    score = torch.empty(test_ids.numel(), dtype=torch.float32)
    seen = torch.zeros(test_ids.numel(), dtype=torch.bool)
    locations = {int(value): index for index, value in enumerate(test_ids.tolist())}
    for module, population in modules.items():
        if not isinstance(population, PolicyPopulation) or population.dataset != dataset or population.module != module:
            raise ValueError("invalid frozen policy population")
        # Whitelist only test IDs, score and selected predictions; never touch correctness fields.
        ids = _tensor(population.test_node_ids, dtype=torch.long, name="population test IDs")
        values = _tensor(population.test_score, dtype=torch.float32, name="population test score")
        pred = _tensor(population.test_selected_pred, dtype=torch.long, name="population selected prediction")
        if not (ids.numel() == values.numel() == pred.numel()) or ids.unique().numel() != ids.numel():
            raise ValueError("misaligned frozen policy population")
        loc = torch.tensor([locations.get(int(value), -1) for value in ids], dtype=torch.long)
        if bool((loc < 0).any()) or bool(seen[loc].any()):
            raise ValueError("population does not partition historical test IDs")
        code = _MODULE_CODES[module]
        if not bool((decision["module_code"][loc] == code).all()) or not torch.equal(decision["selected_pred"][loc], pred):
            raise ValueError("population prediction/module alignment mismatch")
        score[loc] = values
        seen[loc] = True
    if not bool(seen.all()):
        raise ValueError("population does not cover historical test IDs")
    replayed = accepted_from_policies(test_ids, score, decision["module_code"], _policies(dataset, router_id))
    if not torch.equal(replayed, decision["accepted"]):
        raise ValueError("frozen accepted mask drift")
    return score


def route_from_score(score: torch.Tensor, threshold: float, agreement: torch.Tensor) -> torch.Tensor:
    """Frozen A20 routing: agreement goes GNN; conflicts use float64 score threshold."""
    values = _tensor(score, dtype=torch.float32, name="A20 full score")
    same = _tensor(agreement, dtype=torch.bool, name="expert agreement")
    if values.numel() != same.numel() or not np.isfinite(float(threshold)):
        raise ValueError("invalid A20 routing inputs")
    return same | (values.double() >= float(threshold))


def _a20_route(dataset: str, router_id: str, base: dict) -> torch.Tensor:
    replay = torch.load(SOURCE_ROOT / "b_replay" / _A20_PROVIDER[router_id] / dataset / "replay.pt",
                        map_location="cpu", weights_only=True)
    if set(replay) != {"dataset", "endpoint", "full_score", "provenance", "support_node_ids", "support_score", "threshold"}:
        raise ValueError("unexpected A20 replay schema")
    n = base["node_ids"].numel()
    full_score = _tensor(replay["full_score"], dtype=torch.float32, name="A20 full score")
    support_ids = _tensor(replay["support_node_ids"], dtype=torch.long, name="A20 support IDs")
    threshold = float(replay["threshold"])
    if replay["dataset"] != dataset or full_score.numel() != n or not np.isfinite(threshold):
        raise ValueError("invalid A20 replay identity")
    if not torch.equal(support_ids, base["gold_ids"]):
        raise ValueError("A20 replay support IDs differ from original gold IDs")
    # Agreement routes to GNN; conflicts use the frozen score threshold.
    return route_from_score(full_score, threshold, base["agreement"])


def build_router_features_and_pools(base: dict, test_ids: torch.Tensor, decision: dict[str, torch.Tensor], choose_g: torch.Tensor, trust: torch.Tensor | None = None):
    """Apply a frozen route/decision to base features and form exact accepted pools."""
    n = base["node_ids"].numel()
    if not torch.equal(base["node_ids"], torch.arange(n)):
        raise ValueError("base node IDs are not canonical")
    test = _tensor(test_ids, dtype=torch.long, name="test IDs")
    if not torch.equal(decision["node_ids"], test):
        raise ValueError("historical decision IDs differ from base test IDs")
    route = _tensor(choose_g, dtype=torch.bool, name="router route")
    if route.numel() != n:
        raise ValueError("router route does not cover all nodes")
    g, l = base["logp_g"].argmax(1), base["logp_l"].argmax(1)
    agreement = g == l
    selected = torch.where(route, g, l)
    if not torch.equal(selected[test], decision["selected_pred"]):
        raise ValueError("selected prediction differs from frozen route")
    code = decision["module_code"]
    if not bool(torch.isin(code, torch.tensor([0, 1, 2], dtype=torch.int8)).all()):
        raise ValueError("invalid frozen module code")
    if bool((code == 1).any()) and not bool(agreement[test][code == 1].all()):
        raise ValueError("agreement module code differs from expert agreement")
    if bool((code == 2).any()) and not bool((~agreement[test][code == 2]).all()):
        raise ValueError("conflict module code differs from expert agreement")
    full_trust = torch.zeros(n, dtype=torch.float32) if trust is None else _tensor(trust, dtype=torch.float32, name="router trust")
    if full_trust.numel() != n or bool((full_trust < 0).any()) or bool((full_trust > 1).any()):
        raise ValueError("invalid router trust")
    features = {"node_ids": base["node_ids"], "gnn_embeddings": base["gnn_embeddings"],
                "logp_g": base["logp_g"], "logp_l": base["logp_l"], "choose_g": route,
                "agreement": agreement, "selected_pred": selected, "trust": full_trust}
    accepted = torch.zeros(n, dtype=torch.bool)
    accepted[test] = decision["accepted"]
    pools = {
        "teach_g": torch.nonzero(accepted & ~agreement & ~route, as_tuple=False).view(-1),
        "teach_l": torch.nonzero(accepted & ~agreement & route, as_tuple=False).view(-1),
        "agreement": torch.nonzero(accepted & agreement, as_tuple=False).view(-1),
    }
    pools["keep_g"] = torch.cat([pools["teach_l"], pools["agreement"]]).sort().values
    pools["keep_l"] = torch.cat([pools["teach_g"], pools["agreement"]]).sort().values
    selected_ids = torch.cat([pools["teach_g"], pools["teach_l"], pools["agreement"]])
    pools["unselected"] = test[~torch.isin(test, selected_ids)].sort().values
    validate_features(features)
    return features, pools


def _cell_metadata(base_meta: dict, dataset: str, router_id: str, records: dict[str, str], pools: dict, accepted_count: int) -> dict:
    return {
        "schema_version": 1, "router_id": router_id, "router": ROUTERS[router_id], "dataset": dataset,
        "base_export_dir": str((BASE_EXPORT / dataset).resolve()),
        "base_export_complete_sha256": sha256(BASE_EXPORT / dataset / "complete.json"),
        "base_feature_node_ids_sha256": id_hash(torch.arange(base_meta["node_count"])),
        "accepted_count": accepted_count, "pool_counts": {key: int(value.numel()) for key, value in pools.items()},
        "split_counts": {"gold": int(base_meta["gold_count"]), "selection": int(base_meta["model_selection_count"]),
                         "threshold": int(base_meta["threshold_count"]), "test": int(base_meta["test_count"])},
        "source_hashes": records, "test_labels_exported": False,
        "trust_note": "C reliability scores are present only on test IDs and are explicitly unused by Joint training.",
    }


def _prepare_cell(export_root: Path, router_id: str, dataset: str, records: dict[str, str], populations, decisions) -> dict:
    dest = export_root / router_id / dataset
    if dest.exists():
        raise FileExistsError(f"refusing to overwrite export: {dest}")
    base_features, supervision, _base_pools, _hidden, base_meta = _safe_base_export(BASE_EXPORT / dataset, verify_sources=True)
    decision = _decision_for(decisions, dataset, router_id)
    if not torch.equal(decision["node_ids"], supervision["test_ids"]):
        raise ValueError("frozen decision IDs differ from base test IDs")
    score = _population_scores(populations, dataset, router_id, decision)
    route = base_features["choose_g"] if router_id.startswith("r6_") else _a20_route(dataset, router_id, {**base_features, **supervision})
    trust = torch.zeros(base_features["node_ids"].numel(), dtype=torch.float32)
    trust[supervision["test_ids"]] = score
    features, pools = build_router_features_and_pools(base_features, supervision["test_ids"], decision, route, trust)
    validate_supervision(supervision, features["node_ids"].numel(), features["logp_g"].shape[1])
    validate_pools(pools, features, supervision)
    inputs = {"node_ids": features["node_ids"], "accepted_ids": decision["node_ids"][decision["accepted"]],
              "choose_g": features["choose_g"], "trust": trust, "selected_pred": features["selected_pred"],
              "module_code": torch.full_like(features["node_ids"], -1, dtype=torch.int8)}
    inputs["module_code"][supervision["test_ids"]] = decision["module_code"]
    metadata = _cell_metadata(base_meta, dataset, router_id, records, pools, int(decision["accepted"].sum()))
    dest.parent.mkdir(parents=True, exist_ok=True)
    stage = dest.with_name(dest.name + f".staging.{os.getpid()}")
    stage.mkdir()
    save_tensor(stage / "inputs.pt", inputs)
    write_json(stage / "metadata.json", metadata)
    names = ["inputs.pt", "metadata.json"]
    write_json(stage / "complete.json", {"schema_version": 1, "router_id": router_id, "dataset": dataset,
                                           "test_labels_exported": False,
                                           "artifacts": {name: sha256(stage / name) for name in names}})
    os.rename(stage, dest)
    return metadata


def prepare_all(export_root=DEFAULT_EXPORT):
    """Materialize all 25 strict router cells, then write one all-cells marker."""
    export_root = Path(export_root).resolve()
    if export_root.exists():
        raise FileExistsError(f"refusing to overwrite export root: {export_root}")
    records = _source_records()
    populations, decisions = _safe_test_inputs(), _safe_decisions()
    metas = {}
    export_root.mkdir(parents=True)
    try:
        for router_id in ROUTERS:
            for dataset in DATASETS:
                metas[f"{router_id}/{dataset}"] = _prepare_cell(export_root, router_id, dataset, records, populations, decisions)
        cells = {key: sha256(export_root / key / "complete.json") for key in metas}
        write_json(export_root / "COMPLETE.json", {"schema_version": 1, "cell_count": len(cells), "cells": cells,
                                                     "source_manifest_sha256": sha256(SOURCE_ROOT / "manifest.json")})
    except Exception:
        raise
    return metas


def load_cell(export_root, router_id: str, dataset: str, verify_sources: bool = True):
    """Load one strict cell, binding it to its safe base export and frozen source."""
    if router_id not in ROUTERS or dataset not in DATASETS:
        raise ValueError("unknown router or dataset")
    root = Path(export_root).resolve()
    complete_root = read_json(root / "COMPLETE.json")
    key = f"{router_id}/{dataset}"
    if complete_root.get("cell_count") != 25 or set(complete_root.get("cells", {})) != {f"{r}/{d}" for r in ROUTERS for d in DATASETS}:
        raise ValueError("export does not contain all frozen router cells")
    path = root / router_id / dataset
    if sha256(path / "complete.json") != complete_root["cells"].get(key):
        raise ValueError("cell completion hash differs from root marker")
    complete = read_json(path / "complete.json")
    if set(complete.get("artifacts", {})) != {"inputs.pt", "metadata.json"}:
        raise ValueError("invalid router cell artifact schema")
    if complete.get("schema_version") != 1 or complete.get("test_labels_exported") is not False:
        raise ValueError("invalid router cell completion")
    verify_files({str(path / name): digest for name, digest in complete.get("artifacts", {}).items()})
    meta = read_json(path / "metadata.json")
    required = {"base_export_dir", "base_export_complete_sha256", "router_id", "dataset", "accepted_count", "pool_counts", "split_counts", "source_hashes"}
    if not required.issubset(meta) or meta["router_id"] != router_id or meta["dataset"] != dataset:
        raise ValueError("invalid router cell metadata")
    if verify_sources:
        verify_files(meta["source_hashes"])
    base_path = Path(meta["base_export_dir"])
    if sha256(base_path / "complete.json") != meta["base_export_complete_sha256"]:
        raise ValueError("base export completion hash mismatch")
    base_features, supervision, _base_pools, hidden, _base_meta = _safe_base_export(base_path, verify_sources=verify_sources)
    if meta.get("base_feature_node_ids_sha256") != id_hash(base_features["node_ids"]):
        raise ValueError("base feature node identity mismatch")
    inputs = torch.load(path / "inputs.pt", map_location="cpu", weights_only=True)
    if set(inputs) != _INPUT_SCHEMA or not torch.equal(inputs["node_ids"], base_features["node_ids"]):
        raise ValueError("invalid router cell tensor schema")
    n = inputs["node_ids"].numel()
    accepted_ids = _tensor(inputs["accepted_ids"], dtype=torch.long, name="accepted IDs")
    if accepted_ids.unique().numel() != accepted_ids.numel() or not bool(torch.isin(accepted_ids, supervision["test_ids"]).all()):
        raise ValueError("accepted IDs outside test")
    route = _tensor(inputs["choose_g"], dtype=torch.bool, name="router route")
    trust = _tensor(inputs["trust"], dtype=torch.float32, name="router trust")
    selected = _tensor(inputs["selected_pred"], dtype=torch.long, name="router selected predictions")
    code = _tensor(inputs["module_code"], dtype=torch.int8, name="router module codes")
    if any(value.numel() != n for value in (route, trust, selected, code)):
        raise ValueError("router cell tensors are not full-node aligned")
    if bool((trust < 0).any()) or bool((trust > 1).any()) or bool((trust[~torch.isin(torch.arange(n), supervision["test_ids"])] != 0).any()):
        raise ValueError("router trust leaks outside test IDs")
    features, pools = build_router_features_and_pools(base_features, supervision["test_ids"],
        {"node_ids": supervision["test_ids"], "accepted": torch.isin(supervision["test_ids"], accepted_ids),
         "selected_pred": selected[supervision["test_ids"]], "module_code": code[supervision["test_ids"]]}, route, trust)
    validate_supervision(supervision, n, features["logp_g"].shape[1])
    validate_pools(pools, features, supervision)
    if int(accepted_ids.numel()) != meta["accepted_count"] or {k: int(v.numel()) for k, v in pools.items()} != meta["pool_counts"]:
        raise ValueError("router cell metadata count mismatch")
    return features, supervision, pools, hidden, meta

