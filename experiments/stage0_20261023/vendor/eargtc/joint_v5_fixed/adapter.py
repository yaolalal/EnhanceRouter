"""Hash-bound Router-v8 v5 handoff for the unchanged Joint-Fixed trainer."""
from __future__ import annotations

import os
from pathlib import Path

import torch

from eargtc.joint_cross_repair.artifacts import (
    DATASETS, id_hash, read_json, save_tensor, sha256, validate_pools,
    validate_supervision, verify_files, write_json,
)
from eargtc.joint_five_router.adapter import build_router_features_and_pools
from eargtc.joint_v8.adapter import load_cell as load_v4_cell
from eargtc.router_v8_v5.pipeline import load_pipeline, validate_payload


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT = ROOT / "runs/joint_v5_fixed_export_20260909_v1"
SOURCE_ROOT = ROOT / "runs/router_v8_merged_20260908_v5"
V4_EXPORT = ROOT / "runs/joint_v8_export_20260908_v1"
ROUTERS = ("v8_v5",)
_INPUT_SCHEMA = frozenset({"node_ids", "accepted_ids"})
_FEATURE_SCHEMA = frozenset({
    "node_ids", "gnn_embeddings", "logp_g", "logp_l", "choose_g",
    "agreement", "selected_pred", "trust",
})
_POOL_SCHEMA = frozenset({"teach_g", "teach_l", "agreement", "keep_g", "keep_l", "unselected"})
_EXPECTED_CANDIDATES = {
    "cora": 2173, "citeseer": 2656, "pubmed": 19202,
    "arxiv": 139344, "ogbn-products": 53378,
}
_EXPECTED_ACCEPTED = {
    "cora": 643, "citeseer": 665, "pubmed": 8729,
    "arxiv": 2691, "ogbn-products": 11471,
}
_PAYLOAD_ROWS = frozenset({
    "node_ids", "gnn_pred", "llm_pred", "selected_score", "accepted_mask",
    "pseudo_label", "population", "label_source",
})


def _ids(value, name):
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.long or value.ndim != 1:
        raise ValueError(f"invalid {name}")
    return value.detach().contiguous()


def _same_features(left, right):
    return set(left) == set(right) and all(
        isinstance(left[key], torch.Tensor) and torch.equal(left[key], right[key]) for key in left
    )


def _validate_payload_allow_shuffled(payload):
    """Apply the upstream validator after a lossless ID sort for explicit joins."""
    if not isinstance(payload, dict) or set(payload) != {
        "schema", "dataset", "node_count", "class_count", *_PAYLOAD_ROWS
    }:
        raise ValueError("unsafe v5 payload schema")
    ids = payload.get("node_ids")
    if not isinstance(ids, torch.Tensor) or ids.ndim != 1:
        raise ValueError("invalid v5 query IDs")
    order = ids.argsort()
    sorted_payload = {
        key: (value[order] if key in _PAYLOAD_ROWS else value)
        for key, value in payload.items()
    }
    validate_payload(sorted_payload, payload["dataset"])


def build_v5_inputs(payload, features, supervision, v4_pools):
    """Add accepted v5 agreements to v4 keep-KL pools without changing v4 state."""
    if not isinstance(features, dict) or set(features) != _FEATURE_SCHEMA:
        raise ValueError("invalid feature schema")
    if not isinstance(v4_pools, dict) or set(v4_pools) != _POOL_SCHEMA:
        raise ValueError("invalid v4 pool schema")
    _validate_payload_allow_shuffled(payload)
    n = features["node_ids"].numel()
    if payload["node_count"] != n or payload["class_count"] != features["logp_g"].shape[1]:
        raise ValueError("v5 node/class identity differs from v4")
    query = _ids(supervision["test_ids"], "historical query IDs")
    ids = _ids(payload["node_ids"], "v5 query IDs")
    if ids.unique().numel() != ids.numel():
        raise ValueError("duplicate v5 query IDs")
    if bool(((ids < 0) | (ids >= n)).any()):
        raise ValueError("v5 query ID out of range")
    if ids.numel() != query.numel() or set(ids.tolist()) != set(query.tolist()):
        protected = torch.cat([supervision["gold_ids"], supervision["selection_ids"], supervision["threshold_ids"]])
        if bool(torch.isin(ids, protected).any()):
            raise ValueError("v5 query contains gold or validation IDs")
        raise ValueError("v5 query differs from historical full query")

    location = {int(node): pos for pos, node in enumerate(ids.tolist())}
    order = torch.tensor([location[int(node)] for node in query.tolist()], dtype=torch.long)
    g = features["logp_g"].argmax(1)[query]
    l = features["logp_l"].argmax(1)[query]
    if not torch.equal(payload["gnn_pred"][order], g) or not torch.equal(payload["llm_pred"][order], l):
        raise ValueError("v5 expert predictions differ from v4")
    agreement = g == l
    if not torch.equal(payload["population"][order] == 1, agreement):
        raise ValueError("v5 population differs from expert agreement")

    accepted = payload["accepted_mask"][order]
    pseudo = payload["pseudo_label"][order]
    source = payload["label_source"][order]
    expected_pseudo = torch.where(accepted, torch.where(source == 1, l, g), -1)
    if not torch.equal(pseudo, expected_pseudo):
        raise ValueError("v5 pseudo-label differs from selected expert")

    old_teach_g = _ids(v4_pools["teach_g"], "v4 teach_g")
    old_teach_l = _ids(v4_pools["teach_l"], "v4 teach_l")
    if v4_pools["agreement"].numel() != 0:
        raise ValueError("v4 agreement pool must be empty")
    old_conflicts = torch.cat([old_teach_g, old_teach_l]).sort().values
    new_conflicts = query[accepted & ~agreement].sort().values
    if not torch.equal(new_conflicts, old_conflicts):
        raise ValueError("v5 accepted conflicts differ from v4")
    by_id = {int(node): pos for pos, node in enumerate(query.tolist())}
    if any(source[by_id[int(node)]].item() != 1 for node in old_teach_g.tolist()):
        raise ValueError("v5 conflict route differs from v4 teach_g")
    if any(source[by_id[int(node)]].item() != 0 for node in old_teach_l.tolist()):
        raise ValueError("v5 conflict route differs from v4 teach_l")

    decision = {
        "node_ids": query,
        "accepted": accepted,
        "selected_pred": features["selected_pred"][query],
        "module_code": torch.where(agreement, 1, 2).to(torch.int8),
    }
    rebuilt_features, pools = build_router_features_and_pools(
        features, query, decision, features["choose_g"], features["trust"]
    )
    if not _same_features(rebuilt_features, features):
        raise ValueError("v5 adaptation changed v4 features or routes")
    if not torch.equal(pools["teach_g"], old_teach_g) or not torch.equal(pools["teach_l"], old_teach_l):
        raise ValueError("v5 adaptation changed v4 conflict pools")
    expected_agreement = query[accepted & agreement].sort().values
    if not torch.equal(pools["agreement"], expected_agreement):
        raise ValueError("v5 agreement pool mismatch")
    return features, pools


def _v5_source_records(dataset):
    complete_path = SOURCE_ROOT / "COMPLETE.json"
    frozen_path = SOURCE_ROOT / "FROZEN.json"
    manifest_path = SOURCE_ROOT / "pipeline/MANIFEST.json"
    complete, frozen, manifest = read_json(complete_path), read_json(frozen_path), read_json(manifest_path)
    if complete.get("all_checks_pass") is not True or complete.get("frozen_sha256") != sha256(frozen_path):
        raise ValueError("invalid v5 completion/freeze binding")
    if frozen.get("schema") != "router_v8_v5_freeze_v1" or manifest.get("schema") != "router_v8_v5_manifest_v1":
        raise ValueError("invalid v5 frozen manifest schema")
    item = manifest.get("datasets", {}).get(dataset, {})
    pipeline_path = SOURCE_ROOT / item.get("path", "missing")
    records = {
        str(complete_path.resolve()): sha256(complete_path),
        str(frozen_path.resolve()): sha256(frozen_path),
        str(manifest_path.resolve()): frozen.get("files", {}).get(str(manifest_path)),
        str(pipeline_path.resolve()): item.get("sha256"),
    }
    records.update(frozen.get("source_files", {}))
    if any(not isinstance(value, str) for value in records.values()):
        raise ValueError("incomplete v5 source bindings")
    verify_files(records)
    return records, item


def _v4_records(dataset, meta):
    cell = V4_EXPORT / "v8_v4" / dataset
    records = {
        str((V4_EXPORT / "COMPLETE.json").resolve()): sha256(V4_EXPORT / "COMPLETE.json"),
        str((cell / "complete.json").resolve()): sha256(cell / "complete.json"),
        str((cell / "inputs.pt").resolve()): sha256(cell / "inputs.pt"),
        str((cell / "metadata.json").resolve()): sha256(cell / "metadata.json"),
        str((Path(meta["base_export_dir"]) / "complete.json").resolve()): meta["base_export_complete_sha256"],
    }
    records.update(meta["source_hashes"])
    records.update(meta["base_source_hashes"])
    verify_files(records)
    return records


def _prepare_cell(export_root, dataset):
    dest = export_root / "v8_v5" / dataset
    if dest.exists():
        raise FileExistsError(f"refusing to overwrite export: {dest}")
    source_hashes, manifest_item = _v5_source_records(dataset)
    payload = load_pipeline(SOURCE_ROOT, dataset)
    features, supervision, v4_pools, _hidden, old_meta = load_v4_cell(V4_EXPORT, "v8_v4", dataset)
    features, pools = build_v5_inputs(payload, features, supervision, v4_pools)
    if payload["node_ids"].numel() != _EXPECTED_CANDIDATES[dataset] or int(payload["accepted_mask"].sum()) != _EXPECTED_ACCEPTED[dataset]:
        raise ValueError("v5 frozen candidate/accepted count mismatch")
    base_source_hashes = _v4_records(dataset, old_meta)
    accepted_ids = payload["node_ids"][payload["accepted_mask"]].sort().values
    inputs = {"node_ids": supervision["test_ids"], "accepted_ids": accepted_ids}
    conflict_count = int((payload["accepted_mask"] & (payload["population"] == 0)).sum())
    agreement_count = int((payload["accepted_mask"] & (payload["population"] == 1)).sum())
    meta = {
        "schema_version": 1, "router_id": "v8_v5", "dataset": dataset,
        "base_export_dir": old_meta["base_export_dir"],
        "base_export_complete_sha256": old_meta["base_export_complete_sha256"],
        "base_feature_node_ids_sha256": id_hash(features["node_ids"]),
        "v4_export_dir": str(V4_EXPORT.resolve()),
        "source_hashes": source_hashes, "base_source_hashes": base_source_hashes,
        "candidate_count": int(payload["node_ids"].numel()),
        "accepted_count": int(payload["accepted_mask"].sum()),
        "conflict_count": conflict_count, "agreement_count": agreement_count,
        "pool_counts": {key: int(value.numel()) for key, value in pools.items()},
        "llm_hidden_path": old_meta["llm_hidden_path"], "llm_width": old_meta["llm_width"],
        "test_labels_exported": False,
        "trust_note": "All v4 trust values are retained byte-for-byte and are unused by Joint-Fixed.",
        "agreement_loss_note": "Accepted agreements enter keep_g and keep_l KL pools only; no agreement CE is added.",
        "manifest_counts": {key: manifest_item[key] for key in ("rows", "accepted", "disagreement_accepted", "agreement_accepted")},
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    stage = dest.with_name(dest.name + f".staging.{os.getpid()}")
    stage.mkdir()
    save_tensor(stage / "inputs.pt", inputs)
    write_json(stage / "metadata.json", meta)
    write_json(stage / "complete.json", {
        "schema_version": 1, "router_id": "v8_v5", "dataset": dataset,
        "test_labels_exported": False,
        "artifacts": {name: sha256(stage / name) for name in ("inputs.pt", "metadata.json")},
    })
    os.rename(stage, dest)
    return meta


def prepare_all(export_root=DEFAULT_EXPORT):
    export_root = Path(export_root).resolve()
    if export_root.exists():
        raise FileExistsError(f"refusing to overwrite export root: {export_root}")
    export_root.mkdir(parents=True)
    metas = {}
    for dataset in DATASETS:
        metas[f"v8_v5/{dataset}"] = _prepare_cell(export_root, dataset)
    cells = {key: sha256(export_root / key / "complete.json") for key in metas}
    write_json(export_root / "COMPLETE.json", {
        "schema_version": 1, "cell_count": 5, "test_labels_exported": False, "cells": cells,
    })
    return metas


def load_cell(export_root, router_id, dataset, verify_sources=True):
    if router_id not in ROUTERS or dataset not in DATASETS:
        raise ValueError("unknown router or dataset")
    root = Path(export_root).resolve()
    marker = read_json(root / "COMPLETE.json")
    expected = {f"v8_v5/{name}" for name in DATASETS}
    if marker.get("cell_count") != 5 or marker.get("test_labels_exported") is not False or set(marker.get("cells", {})) != expected:
        raise ValueError("export does not contain all five Router-v8 v5 cells")
    key, cell = f"v8_v5/{dataset}", root / "v8_v5" / dataset
    if marker["cells"].get(key) != sha256(cell / "complete.json"):
        raise ValueError("cell completion hash differs from root marker")
    complete = read_json(cell / "complete.json")
    if complete.get("schema_version") != 1 or complete.get("test_labels_exported") is not False or set(complete.get("artifacts", {})) != {"inputs.pt", "metadata.json"}:
        raise ValueError("invalid v5 cell completion")
    verify_files({str(cell / name): digest for name, digest in complete["artifacts"].items()})
    meta = read_json(cell / "metadata.json")
    required = {"router_id", "dataset", "base_export_dir", "base_export_complete_sha256", "source_hashes",
                "base_source_hashes", "candidate_count", "accepted_count", "conflict_count", "agreement_count",
                "pool_counts", "llm_hidden_path", "llm_width", "trust_note", "agreement_loss_note"}
    if not required.issubset(meta) or meta["router_id"] != router_id or meta["dataset"] != dataset:
        raise ValueError("invalid v5 cell metadata")
    if verify_sources:
        verify_files(meta["source_hashes"])
        verify_files(meta["base_source_hashes"])
    features, supervision, v4_pools, hidden, old_meta = load_v4_cell(V4_EXPORT, "v8_v4", dataset, verify_sources=verify_sources)
    if meta["base_export_dir"] != old_meta["base_export_dir"] or meta["base_export_complete_sha256"] != old_meta["base_export_complete_sha256"]:
        raise ValueError("v4/base export binding mismatch")
    inputs = torch.load(cell / "inputs.pt", map_location="cpu", weights_only=True)
    if not isinstance(inputs, dict) or set(inputs) != _INPUT_SCHEMA:
        raise ValueError("invalid safe input schema")
    query, accepted_ids = _ids(inputs["node_ids"], "query IDs"), _ids(inputs["accepted_ids"], "accepted IDs")
    if not torch.equal(query, supervision["test_ids"]) or accepted_ids.unique().numel() != accepted_ids.numel() or not bool(torch.isin(accepted_ids, query).all()):
        raise ValueError("invalid exported query/accepted identity")
    payload = load_pipeline(SOURCE_ROOT, dataset)
    if not torch.equal(accepted_ids, payload["node_ids"][payload["accepted_mask"]].sort().values):
        raise ValueError("exported acceptance differs from frozen v5 source")
    features, pools = build_v5_inputs(payload, features, supervision, v4_pools)
    validate_supervision(supervision, features["node_ids"].numel(), features["logp_g"].shape[1])
    validate_pools(pools, features, supervision)
    counts = {key: int(value.numel()) for key, value in pools.items()}
    conflict = int((payload["accepted_mask"] & (payload["population"] == 0)).sum())
    agreement = int((payload["accepted_mask"] & (payload["population"] == 1)).sum())
    if (meta.get("base_feature_node_ids_sha256") != id_hash(features["node_ids"])
            or meta["candidate_count"] != payload["node_ids"].numel()
            or meta["accepted_count"] != accepted_ids.numel()
            or meta["conflict_count"] != conflict or meta["agreement_count"] != agreement
            or meta["pool_counts"] != counts):
        raise ValueError("v5 cell metadata count or identity mismatch")
    return features, supervision, pools, hidden, meta

