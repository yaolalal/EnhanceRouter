"""Leakage-audited dense banks for the learned Router-v8 experiment."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
OLD_BANK_ROOT = REPO_ROOT / "runs/router_v8_trust_preference_20260906_v1/banks"
SAFE_EXPORT_ROOT = REPO_ROOT / "runs/joint_direct_numbered_router_export_20260813_v1"
HIDDEN_ROOT = REPO_ROOT / "runs/router_v7_decision_llm_hidden_20260827_v1"
_BASE_KEYS = frozenset({"dataset", "ids", "tab", "g", "l", "gnn_pred", "llm_pred"})
_LABELED_KEYS = _BASE_KEYS | {"state", "cal_mask", "policy_mask"}
_SPLITS = frozenset({"validation", "query", "unlabeled"})
_FORBIDDEN_LABEL_KEY = re.compile(r"(?:^|_)(?:gold|label|target|correct|ground_truth|y_true)(?=_|\Z)", re.IGNORECASE)
_OLD_BASE_KEYS = frozenset({
    "node_ids", "candidate_g", "candidate_l", "symmetric_global", "gnn_pred", "llm_pred",
    "candidate_names", "global_names",
})
_SAFE_TENSOR_KEYS = frozenset({
    "gnn_embeddings", "gnn_logits", "llm_logits", "router_evidence", "scheme_b_preference_score",
    "scheme_c_direct_trust", "scheme_c_factorized_trust", "token_ids",
})


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _hash_ids(ids: torch.Tensor | list[int]) -> str:
    values = torch.as_tensor(ids, dtype=torch.long).view(-1).tolist()
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def _hidden_row_hash(ids: torch.Tensor) -> str:
    return hashlib.sha256(ids.detach().cpu().to(torch.int64).contiguous().numpy().tobytes()).hexdigest()


def validate_bank(bank: dict[str, Any], labeled: bool = False) -> None:
    if not isinstance(bank, dict):
        raise ValueError("bank must be a plain dictionary")
    expected = _LABELED_KEYS if labeled else _BASE_KEYS
    if set(bank) != expected:
        raise ValueError(
            "bank schema forbids missing or extra fields: "
            f"missing={sorted(expected - set(bank))}, extra={sorted(set(bank) - expected)}"
        )
    if not isinstance(bank["dataset"], str) or not bank["dataset"]:
        raise ValueError("dataset must be a non-empty string")
    ids = bank["ids"]
    if not isinstance(ids, torch.Tensor) or ids.dtype != torch.long or ids.ndim != 1:
        raise ValueError("ids must be an int64 vector")
    if torch.unique(ids).numel() != ids.numel():
        raise ValueError("ids must be unique")
    if ids.numel() and bool((ids < 0).any()):
        raise ValueError("ids must be non-negative")
    n = ids.numel()
    for name, width, dtype in (
        ("tab", 70, torch.float32), ("g", 128, torch.float32), ("l", 4096, torch.float16),
    ):
        value = bank[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != dtype or value.ndim != 2
                or value.shape != (n, width)):
            raise ValueError(f"{name} must have shape ({n}, {width}) and dtype {dtype}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")
    for name in ("gnn_pred", "llm_pred"):
        value = bank[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.long or value.ndim != 1
                or value.shape != (n,)):
            raise ValueError(f"{name} must be an aligned int64 vector")
        if value.numel() and bool((value < 0).any()):
            raise ValueError(f"{name} contains a negative class ID")
    if bool((bank["gnn_pred"] == bank["llm_pred"]).any()):
        raise ValueError("banks may contain expert disagreement rows only")
    if labeled:
        state = bank["state"]
        if (not isinstance(state, torch.Tensor) or state.dtype != torch.long or state.ndim != 1
                or state.shape != (n,)):
            raise ValueError("state must be an aligned int64 vector")
        if state.numel() and bool(((state < 0) | (state > 2)).any()):
            raise ValueError("state values must be 0=G, 1=L, or 2=N")
        cal_mask, policy_mask = bank["cal_mask"], bank["policy_mask"]
        for name, value in (("cal_mask", cal_mask), ("policy_mask", policy_mask)):
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.bool
                    or value.ndim != 1 or value.shape != (n,)):
                raise ValueError(f"{name} must be an aligned bool vector")
        if not torch.equal(policy_mask, ~cal_mask):
            raise ValueError("calibration and policy masks must be exact complements")


def take(bank: dict[str, Any], index: Any) -> dict[str, Any]:
    """Slice every row-aligned tensor while retaining scalar domain metadata."""
    if not isinstance(bank, dict) or "ids" not in bank or not isinstance(bank["ids"], torch.Tensor):
        raise ValueError("bank must contain tensor ids")
    selected = torch.as_tensor(index)
    if selected.ndim != 1:
        raise ValueError("index must be one-dimensional")
    n = bank["ids"].shape[0]
    if selected.dtype == torch.bool and selected.numel() != n:
        raise ValueError("boolean index length differs from bank")
    if selected.dtype != torch.bool:
        selected = selected.long()
    return {
        key: value[selected]
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == n
        else value
        for key, value in bank.items()
    }


def _dataset_dir(root: str | Path, dataset: str) -> Path:
    root_path = Path(root).resolve()
    return root_path / dataset if root_path.name == "banks" else root_path / "banks" / dataset


def load_bank(root: str | Path, dataset: str, split: str) -> dict[str, Any]:
    if split not in _SPLITS:
        raise ValueError(f"unknown split: {split}")
    dataset_dir = _dataset_dir(root, dataset)
    manifest = _read_json(dataset_dir / "manifest.json")
    if manifest.get("schema") != "router_v8_learned_bank_v1":
        raise ValueError("invalid learned bank manifest schema")
    if manifest.get("dataset") != dataset:
        raise ValueError("manifest dataset mismatch")
    filename = f"{split}.pt"
    expected_hash = manifest.get("bank_hashes", {}).get(filename)
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"manifest lacks a valid hash for {filename}")
    bank_path = dataset_dir / filename
    if sha256_file(bank_path) != expected_hash:
        raise ValueError(f"bank hash mismatch: {bank_path}")
    bank = torch.load(bank_path, map_location="cpu", weights_only=True)
    validate_bank(bank, labeled=split == "validation")
    if bank["dataset"] != dataset:
        raise ValueError("bank dataset mismatch")
    return bank


def _validated_split_ids(value: Any, name: str, num_nodes: int) -> torch.Tensor:
    if not isinstance(value, list):
        raise ValueError(f"{name} split must be a list")
    ids = torch.tensor(value, dtype=torch.long)
    if ids.ndim != 1 or ids.unique().numel() != ids.numel():
        raise ValueError(f"{name} split IDs must be unique")
    if ids.numel() and bool(((ids < 0) | (ids >= num_nodes)).any()):
        raise ValueError(f"{name} split IDs out of range")
    return ids


def _historical_masks(ids: torch.Tensor, dataset: str) -> tuple[torch.Tensor, torch.Tensor]:
    hashes = [hashlib.sha256(f"{dataset}|conflict|{int(i)}".encode()).digest() for i in ids.tolist()]
    cal = torch.tensor([bool(value[0] & 1) for value in hashes], dtype=torch.bool)
    if cal.numel() and (bool(cal.all()) or not bool(cal.any())):
        order = sorted(range(ids.numel()), key=lambda index: hashes[index])
        cal[:] = False
        cal[order[: ids.numel() // 2]] = True
    return cal, ~cal


def _deterministic_subset_indices(ids: torch.Tensor, dataset: str, limit: int = 4096) -> torch.Tensor:
    """Select by ID hash and return positions in the unchanged query row order."""
    ids = torch.as_tensor(ids, dtype=torch.long).view(-1)
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if ids.numel() <= limit:
        return torch.arange(ids.numel(), dtype=torch.long)
    ranked = sorted(
        range(ids.numel()),
        key=lambda index: hashlib.sha256(
            f"{dataset}|unlabeled|{int(ids[index])}".encode()
        ).digest(),
    )[:limit]
    return torch.tensor(sorted(ranked), dtype=torch.long)


def _validate_old_bank(bank: Any, *, labeled: bool) -> None:
    expected = _OLD_BASE_KEYS | ({"state"} if labeled else set())
    if not isinstance(bank, dict) or set(bank) != expected:
        raise ValueError("old V8 bank schema mismatch or forbidden label field")
    n = bank["node_ids"].numel()
    if bank["node_ids"].dtype != torch.long or bank["node_ids"].ndim != 1:
        raise ValueError("old V8 bank IDs must be int64")
    if bank["node_ids"].unique().numel() != n:
        raise ValueError("old V8 bank IDs must be unique")
    for key, width in (("candidate_g", 27), ("candidate_l", 27), ("symmetric_global", 16)):
        value = bank[key]
        if value.dtype != torch.float32 or value.shape != (n, width) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"old V8 {key} schema mismatch")
    for key in ("gnn_pred", "llm_pred"):
        if bank[key].dtype != torch.long or bank[key].shape != (n,):
            raise ValueError(f"old V8 {key} schema mismatch")
    if bool((bank["gnn_pred"] == bank["llm_pred"]).any()):
        raise ValueError("old V8 bank contains agreement rows")
    if labeled:
        state = bank["state"]
        if state.dtype != torch.long or state.shape != (n,) or bool(((state < 0) | (state > 2)).any()):
            raise ValueError("old V8 state schema mismatch")


def _load_sources(dataset: str) -> dict[str, Any]:
    old_dir = Path(OLD_BANK_ROOT).resolve() / dataset
    old_manifest_path = old_dir / "manifest.json"
    old_manifest = _read_json(old_manifest_path)
    if old_manifest.get("schema") != "router_v8_feature_bank_v1" or old_manifest.get("dataset") != dataset:
        raise ValueError("old V8 manifest identity mismatch")
    old_paths = {name: old_dir / f"{name}.pt" for name in ("validation", "query")}
    for name, path in old_paths.items():
        if sha256_file(path) != old_manifest.get("bank_sha256", {}).get(name):
            raise ValueError(f"old V8 bank hash mismatch: {name}")
    validation_old = torch.load(old_paths["validation"], map_location="cpu", weights_only=True)
    query_old = torch.load(old_paths["query"], map_location="cpu", weights_only=True)
    _validate_old_bank(validation_old, labeled=True)
    _validate_old_bank(query_old, labeled=False)

    num_nodes = int(old_manifest.get("num_nodes", -1))
    if num_nodes <= 0:
        raise ValueError("old V8 manifest has invalid node count")
    split_raw = old_manifest.get("split_ids", {})
    train_ids = _validated_split_ids(split_raw.get("train"), "train", num_nodes)
    validation_ids = _validated_split_ids(split_raw.get("validation"), "validation", num_nodes)
    query_ids = _validated_split_ids(split_raw.get("test"), "test", num_nodes)
    if any(_hash_ids(ids) != old_manifest.get("split_hashes", {}).get(name)
           for name, ids in (("train", train_ids), ("validation", validation_ids), ("test", query_ids))):
        raise ValueError("old V8 historical split hash mismatch")
    owner = torch.zeros(num_nodes, dtype=torch.int8)
    for ids in (train_ids, validation_ids, query_ids):
        if bool(owner[ids].any()):
            raise ValueError("old V8 historical splits overlap")
        owner[ids] = 1
    expected_query = (owner == 0).nonzero(as_tuple=False).view(-1)
    # The first two splits mark owner; derive the historical ascending complement directly.
    excluded = torch.zeros(num_nodes, dtype=torch.bool)
    excluded[train_ids] = True
    excluded[validation_ids] = True
    expected_query = (~excluded).nonzero(as_tuple=False).view(-1)
    if not torch.equal(query_ids, expected_query):
        raise ValueError("old V8 test split is not the train/validation complement")
    if not torch.equal(validation_old["node_ids"], validation_ids[
        torch.isin(validation_ids, validation_old["node_ids"])
    ]):
        raise ValueError("old V8 validation conflict IDs differ from historical split/order")
    expected_query_conflicts = query_ids[torch.isin(query_ids, query_old["node_ids"])]
    if not torch.equal(query_old["node_ids"], expected_query_conflicts):
        raise ValueError("old V8 query conflict IDs differ from historical split/order")
    counts = old_manifest.get("counts", {})
    if int(counts.get("validation_conflicts", -1)) != validation_old["node_ids"].numel():
        raise ValueError("old V8 validation conflict count mismatch")
    if int(counts.get("query_conflicts", -1)) != query_old["node_ids"].numel():
        raise ValueError("old V8 query conflict count mismatch")

    safe_dir = Path(SAFE_EXPORT_ROOT).resolve() / dataset
    complete_path = safe_dir / "complete.json"
    complete = _read_json(complete_path)
    if complete.get("all_checks_pass") is not True or complete.get("test_labels_exported") is not False:
        raise ValueError("safe raw export did not pass label isolation")
    if complete.get("dataset") != dataset or int(complete.get("node_count", -1)) != num_nodes:
        raise ValueError("safe raw export completion identity mismatch")
    complete_artifacts = complete.get("artifacts")
    required_artifacts = {"router_tensors.pt", "split_ids.json", "safe_input_metadata.json"}
    if not isinstance(complete_artifacts, dict) or not required_artifacts.issubset(complete_artifacts):
        raise ValueError("safe raw export completion missing required artifacts")
    input_hashes = {str(old_manifest_path): sha256_file(old_manifest_path)}
    input_hashes.update({str(path): sha256_file(path) for path in old_paths.values()})
    input_hashes[str(complete_path)] = sha256_file(complete_path)
    for filename, record in complete_artifacts.items():
        artifact_path = safe_dir / filename
        expected = record.get("sha256") if isinstance(record, dict) else None
        if not isinstance(expected, str) or sha256_file(artifact_path) != expected:
            raise ValueError(f"safe raw export artifact hash mismatch: {filename}")
        input_hashes[str(artifact_path)] = expected
    safe_metadata_path = safe_dir / "safe_input_metadata.json"
    safe_metadata = _read_json(safe_metadata_path)
    if (safe_metadata.get("dataset") != dataset or int(safe_metadata.get("node_count", -1)) != num_nodes
            or safe_metadata.get("test_labels_exported") is not False):
        raise ValueError("safe raw metadata identity/isolation mismatch")
    safe_splits = _read_json(safe_dir / "split_ids.json")
    if (safe_splits.get("train") != train_ids.tolist()
            or safe_splits.get("validation") != validation_ids.tolist()
            or safe_splits.get("query") != query_ids.tolist()):
        raise ValueError("safe raw split IDs differ from old V8 historical split")
    tensors_path = safe_dir / "router_tensors.pt"
    raw = torch.load(tensors_path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or set(raw) != _SAFE_TENSOR_KEYS:
        raise ValueError("safe raw tensor schema mismatch or forbidden field")
    g = raw["gnn_embeddings"]
    gnn_logits, llm_logits = raw["gnn_logits"], raw["llm_logits"]
    if g.dtype != torch.float32 or g.shape != (num_nodes, 128) or not bool(torch.isfinite(g).all()):
        raise ValueError("safe raw GNN embedding shape/dtype mismatch")
    if (gnn_logits.ndim != 2 or gnn_logits.shape != llm_logits.shape
            or gnn_logits.shape[0] != num_nodes or not bool(torch.isfinite(gnn_logits).all())
            or not bool(torch.isfinite(llm_logits).all())):
        raise ValueError("safe expert logits shape/value mismatch")
    gnn_full, llm_full = gnn_logits.argmax(1).long(), llm_logits.argmax(1).long()
    for name, bank in (("validation", validation_old), ("query", query_old)):
        ids = bank["node_ids"]
        if not torch.equal(bank["gnn_pred"], gnn_full[ids]) or not torch.equal(bank["llm_pred"], llm_full[ids]):
            raise ValueError(f"expert argmax mismatch: {name}")

    audit_path = Path(HIDDEN_ROOT).resolve() / "AUDIT.json"
    audit = _read_json(audit_path)
    audit_row = audit.get("datasets", {}).get(dataset, {})
    hidden_path = Path(HIDDEN_ROOT).resolve() / dataset / "hidden.pt"
    if (audit_row.get("status") != "pass" or int(audit_row.get("rows", -1)) != num_nodes
            or int(audit_row.get("width", -1)) != 4096
            or audit_row.get("dtype") != "float16"
            or Path(audit_row.get("output", "")).resolve() != hidden_path):
        raise ValueError("raw hidden audit identity mismatch")
    hidden_payload = torch.load(hidden_path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(hidden_payload, dict) or set(hidden_payload) != {"node_ids", "hidden", "metadata"}:
        raise ValueError("raw hidden schema has forbidden or missing fields")
    hidden_metadata = hidden_payload["metadata"]
    if not isinstance(hidden_metadata, dict):
        raise ValueError("raw hidden metadata must be a dictionary")
    forbidden = [key for key in (*hidden_payload.keys(), *hidden_metadata.keys())
                 if _FORBIDDEN_LABEL_KEY.search(str(key))]
    if forbidden:
        raise ValueError(f"raw hidden metadata has forbidden label fields: {sorted(forbidden)}")
    hidden_ids, hidden = hidden_payload["node_ids"], hidden_payload["hidden"]
    if (hidden_ids.dtype != torch.long or not torch.equal(hidden_ids, torch.arange(num_nodes))
            or hidden.dtype != torch.float16 or hidden.shape != (num_nodes, 4096)):
        raise ValueError("raw hidden ID/shape/dtype mismatch")
    if hidden_metadata.get("dataset") != dataset or hidden_metadata.get("row_order_hash") != _hidden_row_hash(hidden_ids):
        raise ValueError("raw hidden metadata row-order mismatch")
    for start in range(0, num_nodes, 4096):
        if not bool(torch.isfinite(hidden[start:start + 4096]).all()):
            raise ValueError("raw hidden contains non-finite values")
    input_hashes[str(audit_path)] = sha256_file(audit_path)
    input_hashes[str(hidden_path)] = sha256_file(hidden_path)
    return {
        "old_manifest": old_manifest, "validation": validation_old, "query": query_old,
        "train_ids": train_ids, "validation_ids": validation_ids, "query_ids": query_ids,
        "g": g, "hidden": hidden, "hidden_metadata": hidden_metadata,
        "safe_metadata": safe_metadata, "input_hashes": dict(sorted(input_hashes.items())),
    }


def _new_bank(dataset: str, old: dict[str, Any], g: torch.Tensor, hidden: torch.Tensor,
              cal_mask: torch.Tensor | None = None) -> dict[str, Any]:
    ids = old["node_ids"].detach().cpu().long()
    bank: dict[str, Any] = {
        "dataset": dataset,
        "ids": ids,
        "tab": torch.cat((old["candidate_g"], old["candidate_l"], old["symmetric_global"]), dim=1).float(),
        "g": g[ids].detach().cpu().float(),
        "l": hidden[ids].detach().cpu().to(torch.float16),
        "gnn_pred": old["gnn_pred"].detach().cpu().long(),
        "llm_pred": old["llm_pred"].detach().cpu().long(),
    }
    if cal_mask is not None:
        bank.update(state=old["state"].detach().cpu().long(), cal_mask=cal_mask, policy_mask=~cal_mask)
    validate_bank(bank, labeled=cal_mask is not None)
    return bank


def prepare(root: str | Path, dataset: str) -> dict[str, Any]:
    """Build one dataset atomically from frozen, label-isolated sources."""
    if not isinstance(dataset, str) or not dataset or Path(dataset).name != dataset or dataset in {".", ".."}:
        raise ValueError("invalid dataset name")
    destination = Path(root).resolve() / "banks" / dataset
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite prepared bank: {destination}")
    source = _load_sources(dataset)
    validation_old, query_old = source["validation"], source["query"]
    if bool(torch.isin(validation_old["node_ids"], source["train_ids"]).any()):
        raise ValueError("gold rows found in validation raw-embedding bank")
    if bool(torch.isin(query_old["node_ids"], source["train_ids"]).any()):
        raise ValueError("gold rows found in query raw-embedding bank")
    cal_mask, _ = _historical_masks(validation_old["node_ids"], dataset)
    validation = _new_bank(dataset, validation_old, source["g"], source["hidden"], cal_mask)
    query = _new_bank(dataset, query_old, source["g"], source["hidden"])
    unlabeled = take(query, _deterministic_subset_indices(query["ids"], dataset))
    validate_bank(unlabeled, labeled=False)

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{dataset}.stage-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        banks = {"validation.pt": validation, "query.pt": query, "unlabeled.pt": unlabeled}
        for filename, bank in banks.items():
            torch.save(bank, stage / filename)
        bank_hashes = {filename: sha256_file(stage / filename) for filename in banks}
        old_manifest = source["old_manifest"]
        raw_meta = source["hidden_metadata"]
        manifest = {
            "schema": "router_v8_learned_bank_v1",
            "dataset": dataset,
            "input_hashes": source["input_hashes"],
            "bank_hashes": bank_hashes,
            "split_ids": old_manifest["split_ids"],
            "split_hashes": old_manifest["split_hashes"],
            "validation_conflict_ids": validation["ids"].tolist(),
            "calibration_ids": validation["ids"][validation["cal_mask"]].tolist(),
            "policy_ids": validation["ids"][validation["policy_mask"]].tolist(),
            "query_conflict_ids": query["ids"].tolist(),
            "unlabeled_ids": unlabeled["ids"].tolist(),
            "id_hashes": {
                "validation_conflicts": _hash_ids(validation["ids"]),
                "calibration": _hash_ids(validation["ids"][validation["cal_mask"]]),
                "policy": _hash_ids(validation["ids"][validation["policy_mask"]]),
                "query_conflicts": _hash_ids(query["ids"]),
                "unlabeled": _hash_ids(unlabeled["ids"]),
            },
            "counts": {
                "train_population": len(old_manifest["split_ids"]["train"]),
                "validation_population": len(old_manifest["split_ids"]["validation"]),
                "validation_conflicts": validation["ids"].numel(),
                "calibration_conflicts": int(validation["cal_mask"].sum()),
                "policy_conflicts": int(validation["policy_mask"].sum()),
                "test_population": len(old_manifest["split_ids"]["test"]),
                "query_conflicts": query["ids"].numel(),
                "unlabeled_conflicts": unlabeled["ids"].numel(),
            },
            "historical_validation_split": "sha256(dataset|conflict|node_id)[0]&1 with frozen fallback",
            "unlabeled_selection": "lowest sha256(dataset|unlabeled|node_id), capped at 4096; saved in query order",
            "label_provenance": {
                "source": "old_v8_validation_bank_state",
                "source_path": str((Path(OLD_BANK_ROOT).resolve() / dataset / "validation.pt")),
                "validation_only": True,
                "gold_rows_used": False,
                "query_labels_read": False,
            },
            "raw_hidden_metadata": {
                key: raw_meta.get(key) for key in (
                    "dataset", "dtype", "hidden_width", "row_order_hash", "model_hash", "tokenizer_hash",
                    "prompt_hash", "parity_argmax_match", "parity_max_abs_error", "parity_max_probability_error",
                )
            },
            "raw_cache_limitation": (
                "Raw LLM hidden states are historical final-token caches, not fresh decoder runs. The independent "
                "audit found 2/173 Arxiv LM-head argmax mismatches (nodes 38118 and 83420); original safe-export "
                "expert logits/predictions remain authoritative. Historical metadata lacks per-node text/token hashes "
                "and model_hash does not cover every model weight."
            ),
            "gnn_raw_provenance": "safe router export gnn_embeddings from the full gold-trained GNN",
            "gnn_gold_exclusion": "gold support rows are excluded because full-GNN embeddings depend on gold training labels",
            "test_labels_exported": False,
            "graph_labels_read": False,
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(stage, destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest
