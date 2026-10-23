"""Authenticated, label-isolated inputs for Joint-v6 Unified."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("cora", "citeseer", "pubmed", "arxiv", "ogbn-products")
BASE_ROOT = ROOT / "runs/joint_cross_repair_export_20260906_v2"
ROUTER_ROOT = ROOT / "runs/router_v8_v6_final"

_FEATURE_KEYS = frozenset({
    "node_ids", "gnn_embeddings", "logp_g", "logp_l", "choose_g",
    "agreement", "selected_pred", "trust",
})
_SUPERVISION_KEYS = frozenset({
    "gold_ids", "gold_targets", "selection_ids", "selection_targets",
    "threshold_ids", "test_ids",
})
_HIDDEN_KEYS = frozenset({"node_ids", "hidden", "metadata"})
_PIPELINE_KEYS = frozenset({
    "schema", "dataset", "node_count", "class_count", "node_ids",
    "gnn_pred", "llm_pred", "selected_score", "accepted_mask",
    "pseudo_label", "population", "label_source",
})
_SPLIT_KEYS = frozenset({"train", "validation", "query", "strict_holdout"})
_TRAIN_KEYS = frozenset({
    "graph", "text", "logp_g", "logp_l", "gold_ids", "gold_targets",
    "pseudo_ids", "pseudo_targets",
})
_VALIDATION_KEYS = frozenset({"ids", "targets"})
_INFERENCE_KEYS = frozenset({"node_ids", "query_ids", "threshold_ids"})
_METADATA_KEYS = frozenset({"input_hashes", "counts", "provenance"})
_COUNT_KEYS = frozenset({
    "nodes", "classes", "graph_width", "text_width", "gold",
    "validation", "threshold", "query", "pseudo",
})
_PROVENANCE_KEYS = frozenset({
    "dataset", "router", "base_export", "test_targets_loaded",
    "pseudo_weighting",
})
_FORBIDDEN_LABEL_KEY = re.compile(r"(^|_)(labels?|targets?|correct(ness)?|state|gold|ground_truth|y_true)($|_)", re.I)


def require_slurm() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Joint-v6 Unified numerical operations require Slurm")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _exact_dict(value: Any, keys: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} schema forbids missing or extra fields")
    return value


def _tensor(value: Any, *, dtype: torch.dtype, ndim: int, name: str) -> torch.Tensor:
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
            or value.dtype != dtype or value.ndim != ndim):
        raise ValueError(f"invalid {name} tensor")
    return value.detach()


def _ids(value: Any, n: int, name: str, *, allow_empty: bool = False) -> torch.Tensor:
    ids = _tensor(value, dtype=torch.long, ndim=1, name=name)
    if (not allow_empty and ids.numel() == 0) or ids.unique().numel() != ids.numel():
        raise ValueError(f"{name} has empty or duplicate IDs")
    if ids.numel() and bool(((ids < 0) | (ids >= n)).any()):
        raise ValueError(f"{name} has out-of-range IDs")
    return ids


def _finite_rows(value: torch.Tensor, name: str) -> None:
    for start in range(0, value.shape[0], 4096):
        if not bool(torch.isfinite(value[start:start + 4096]).all()):
            raise ValueError(f"nonfinite {name}")


def _validate_features(features: Any) -> tuple[int, int]:
    data = _exact_dict(features, _FEATURE_KEYS, "historical feature")
    node_ids = _tensor(data["node_ids"], dtype=torch.long, ndim=1, name="feature node IDs")
    n = node_ids.numel()
    if n == 0 or not torch.equal(node_ids, torch.arange(n)):
        raise ValueError("feature node identity must be the full global-ID order")
    graph = _tensor(data["gnn_embeddings"], dtype=torch.float32, ndim=2, name="graph features")
    logp_g = _tensor(data["logp_g"], dtype=torch.float32, ndim=2, name="GNN log probabilities")
    logp_l = _tensor(data["logp_l"], dtype=torch.float32, ndim=2, name="LLM log probabilities")
    if graph.shape[0] != n or graph.shape[1] == 0:
        raise ValueError("invalid graph feature shape")
    if logp_g.shape != logp_l.shape or logp_g.shape[0] != n or logp_g.shape[1] == 0:
        raise ValueError("expert class spaces or node counts differ")
    for value, name in ((graph, "graph features"), (logp_g, "GNN log probabilities"),
                        (logp_l, "LLM log probabilities")):
        _finite_rows(value, name)
    zeros = torch.zeros(n, dtype=logp_g.dtype)
    if (not torch.allclose(logp_g.logsumexp(1), zeros, atol=2e-5, rtol=0)
            or not torch.allclose(logp_l.logsumexp(1), zeros, atol=2e-5, rtol=0)):
        raise ValueError("expert log probabilities are not normalized")
    choose_g = _tensor(data["choose_g"], dtype=torch.bool, ndim=1, name="historical route")
    agreement = _tensor(data["agreement"], dtype=torch.bool, ndim=1, name="historical agreement")
    selected = _tensor(data["selected_pred"], dtype=torch.long, ndim=1, name="historical prediction")
    trust = data["trust"]
    if (not isinstance(trust, torch.Tensor) or trust.device.type != "cpu"
            or trust.ndim != 1 or not trust.dtype.is_floating_point):
        raise ValueError("invalid historical trust tensor")
    if any(value.shape != (n,) for value in (choose_g, agreement, selected, trust)):
        raise ValueError("historical Router metadata shape mismatch")
    _finite_rows(trust, "historical trust")
    if bool(((trust < 0) | (trust > 1)).any()):
        raise ValueError("historical trust outside probability range")
    gnn_pred, llm_pred = logp_g.argmax(1), logp_l.argmax(1)
    if not torch.equal(agreement, gnn_pred == llm_pred):
        raise ValueError("historical agreement differs from expert argmax")
    if not torch.equal(selected, torch.where(choose_g, gnn_pred, llm_pred)):
        raise ValueError("historical selected prediction differs from route")
    return n, logp_g.shape[1]


def _validate_supervision(supervision: Any, n: int, classes: int) -> dict[str, torch.Tensor]:
    data = _exact_dict(supervision, _SUPERVISION_KEYS, "historical supervision")
    result = {
        key: _ids(data[key], n, key)
        for key in ("gold_ids", "selection_ids", "threshold_ids", "test_ids")
    }
    for prefix in ("gold", "selection"):
        target = _tensor(data[f"{prefix}_targets"], dtype=torch.long, ndim=1,
                         name=f"{prefix} targets")
        if target.shape != result[f"{prefix}_ids"].shape:
            raise ValueError(f"{prefix} IDs and targets differ")
        if bool(((target < 0) | (target >= classes)).any()):
            raise ValueError(f"{prefix} target outside class space")
        result[f"{prefix}_targets"] = target
    owner = torch.zeros(n, dtype=torch.int8)
    for key in ("gold_ids", "selection_ids", "threshold_ids", "test_ids"):
        if bool(owner[result[key]].any()):
            raise ValueError("gold, validation, threshold, and query IDs overlap")
        owner[result[key]] = 1
    if not bool(owner.all()):
        raise ValueError("gold, validation, threshold, and query IDs do not partition nodes")
    return result


def _nested_keys(value: Any):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from _nested_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _nested_keys(nested)


def _validate_hidden(hidden: Any, n: int, dataset: str) -> torch.Tensor:
    payload = _exact_dict(hidden, _HIDDEN_KEYS, "hidden feature")
    ids = _tensor(payload["node_ids"], dtype=torch.long, ndim=1, name="hidden node IDs")
    if not torch.equal(ids, torch.arange(n)):
        raise ValueError("hidden node identity differs from global-ID order")
    value = payload["hidden"]
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
            or not value.dtype.is_floating_point or value.ndim != 2
            or value.shape[0] != n or value.shape[1] == 0):
        raise ValueError("invalid hidden feature tensor")
    _finite_rows(value, "hidden features")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("hidden metadata must be a dictionary")
    forbidden = [key for key in _nested_keys(metadata) if _FORBIDDEN_LABEL_KEY.search(key)]
    if forbidden:
        raise ValueError(f"hidden metadata contains forbidden target fields: {sorted(forbidden)}")
    if "dataset" in metadata and metadata["dataset"] != dataset:
        raise ValueError("hidden dataset identity mismatch")
    return value.detach()


def _json_ids(value: Any, n: int, name: str, *, allow_empty: bool = False) -> torch.Tensor:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError(f"invalid independent {name} split IDs")
    return _ids(torch.tensor(value, dtype=torch.long), n, f"independent {name}", allow_empty=allow_empty)


def _validate_splits(splits: Any, supervision: dict[str, torch.Tensor], n: int) -> None:
    data = _exact_dict(splits, _SPLIT_KEYS, "independent split")
    train = _json_ids(data["train"], n, "train")
    validation = _json_ids(data["validation"], n, "validation")
    query = _json_ids(data["query"], n, "query")
    holdout = _json_ids(data["strict_holdout"], n, "strict holdout", allow_empty=True)
    if not torch.equal(train, supervision["gold_ids"]):
        raise ValueError("gold IDs differ from independent train split")
    joined_validation = torch.cat([supervision["selection_ids"], supervision["threshold_ids"]])
    if not torch.equal(joined_validation.sort().values, validation.sort().values):
        raise ValueError("selection/threshold partition differs from independent validation split")
    if not torch.equal(query, supervision["test_ids"]):
        raise ValueError("query IDs differ from independent query split")
    if holdout.numel() and not bool(torch.isin(holdout, query).all()):
        raise ValueError("strict holdout lies outside independent query split")
    if not torch.equal(torch.cat([train, validation, query]).sort().values, torch.arange(n)):
        raise ValueError("independent splits do not partition the full node universe")


def _validate_pipeline(pipeline: Any, dataset: str, n: int, classes: int,
                       query_ids: torch.Tensor, logp_g: torch.Tensor,
                       logp_l: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    data = _exact_dict(pipeline, _PIPELINE_KEYS, "v6 pipeline")
    if (data["schema"] != "router_v8_v5_pipeline_v1" or data["dataset"] != dataset
            or type(data["node_count"]) is not int or data["node_count"] != n
            or type(data["class_count"]) is not int or data["class_count"] != classes):
        raise ValueError("v6 pipeline identity/schema mismatch")
    ids = _ids(data["node_ids"], n, "v6 query IDs")
    if not torch.equal(ids, query_ids):
        raise ValueError("v6 pseudo/query IDs differ from the independent query partition")
    rows = ids.numel()
    expected_types = {
        "gnn_pred": torch.long, "llm_pred": torch.long,
        "selected_score": torch.float64, "accepted_mask": torch.bool,
        "pseudo_label": torch.long, "population": torch.long,
        "label_source": torch.long,
    }
    vectors = {}
    for key, dtype in expected_types.items():
        vectors[key] = _tensor(data[key], dtype=dtype, ndim=1, name=f"v6 {key}")
        if vectors[key].shape != (rows,):
            raise ValueError(f"misaligned v6 {key}")
    _finite_rows(vectors["selected_score"], "v6 selected scores")
    expected_g = logp_g.argmax(1)[ids]
    expected_l = logp_l.argmax(1)[ids]
    if (not torch.equal(vectors["gnn_pred"], expected_g)
            or not torch.equal(vectors["llm_pred"], expected_l)):
        raise ValueError("v6 expert identity/argmax differs from historical features")
    population = vectors["population"]
    if bool(((population != 0) & (population != 1)).any()):
        raise ValueError("v6 pipeline contains an unknown population code")
    if not torch.equal(population == 1, expected_g == expected_l):
        raise ValueError("v6 population differs from expert agreement")
    accepted = vectors["accepted_mask"]
    source = vectors["label_source"]
    pseudo = vectors["pseudo_label"]
    if bool((source[~accepted] != -1).any()) or bool((pseudo[~accepted] != -1).any()):
        raise ValueError("rejected v6 rows expose an invalid pseudo label/source")
    if bool((source[accepted & (population == 1)] != 2).any()):
        raise ValueError("accepted v6 agreement has invalid source")
    conflict_source = source[accepted & (population == 0)]
    if bool(((conflict_source != 0) & (conflict_source != 1)).any()):
        raise ValueError("accepted v6 disagreement has invalid source")
    routed = torch.where(source == 1, expected_l, expected_g)
    if not torch.equal(pseudo, torch.where(accepted, routed, -1)):
        raise ValueError("v6 pseudo label differs from selected expert")
    if bool(((pseudo[accepted] < 0) | (pseudo[accepted] >= classes)).any()):
        raise ValueError("v6 pseudo target outside class space")
    return ids[accepted].detach(), pseudo[accepted].detach()


def validate_output(train: Any, validation: Any, inference: Any, metadata: Any) -> None:
    require_slurm()
    train = _exact_dict(train, _TRAIN_KEYS, "train output")
    validation = _exact_dict(validation, _VALIDATION_KEYS, "validation output")
    inference = _exact_dict(inference, _INFERENCE_KEYS, "inference output")
    metadata = _exact_dict(metadata, _METADATA_KEYS, "metadata output")
    counts = _exact_dict(metadata["counts"], _COUNT_KEYS, "metadata counts")
    provenance = _exact_dict(metadata["provenance"], _PROVENANCE_KEYS, "metadata provenance")
    hashes = metadata["input_hashes"]
    if (not isinstance(hashes, dict) or any(
            not isinstance(path, str) or not Path(path).is_absolute()
            or not isinstance(digest, str) or len(digest) != 64
            for path, digest in hashes.items())):
        raise ValueError("invalid metadata input_hashes")
    if provenance["test_targets_loaded"] is not False or provenance["pseudo_weighting"] != "uniform":
        raise ValueError("unsafe provenance")
    n, classes = counts["nodes"], counts["classes"]
    if any(type(value) is not int or value < 0 for value in counts.values()) or n <= 0 or classes <= 0:
        raise ValueError("invalid metadata counts")
    node_ids = _ids(inference["node_ids"], n, "inference node IDs")
    if not torch.equal(node_ids, torch.arange(n)):
        raise ValueError("inference does not preserve global node indexing")
    for key in ("graph", "text", "logp_g", "logp_l"):
        value = train[key]
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.ndim != 2 or value.shape[0] != n:
            raise ValueError(f"invalid train {key}")
    if train["logp_g"].shape != train["logp_l"].shape or train["logp_g"].shape[1] != classes:
        raise ValueError("train expert class spaces differ")
    for key, expected in (("gold_ids", counts["gold"]), ("pseudo_ids", counts["pseudo"])):
        if _ids(train[key], n, f"train {key}", allow_empty=key == "pseudo_ids").numel() != expected:
            raise ValueError(f"train {key} count mismatch")
    for ids_key, target_key in (("gold_ids", "gold_targets"), ("pseudo_ids", "pseudo_targets")):
        targets = _tensor(train[target_key], dtype=torch.long, ndim=1, name=f"train {target_key}")
        if targets.shape != train[ids_key].shape or bool(((targets < 0) | (targets >= classes)).any()):
            raise ValueError(f"invalid train {target_key}")
    val_ids = _ids(validation["ids"], n, "validation IDs")
    val_targets = _tensor(validation["targets"], dtype=torch.long, ndim=1, name="validation targets")
    query_ids = _ids(inference["query_ids"], n, "query IDs")
    threshold_ids = _ids(inference["threshold_ids"], n, "threshold IDs")
    if val_targets.shape != val_ids.shape or val_ids.numel() != counts["validation"]:
        raise ValueError("validation IDs/targets/count mismatch")
    if query_ids.numel() != counts["query"] or threshold_ids.numel() != counts["threshold"]:
        raise ValueError("inference split count mismatch")
    protected = torch.cat([train["gold_ids"], val_ids, threshold_ids])
    if bool(torch.isin(train["pseudo_ids"], protected).any()) or not bool(torch.isin(train["pseudo_ids"], query_ids).all()):
        raise ValueError("pseudo IDs cross the training/validation/query boundary")


def assemble_data(features: Any, supervision: Any, hidden: Any, pipeline: Any,
                  split_ids: Any, *, dataset: str, input_hashes: dict[str, str],
                  base_export: str = "synthetic", router: str = "router_v8_v6_final"):
    """Validate already-loaded records and expose only the registered safe interface."""
    require_slurm()
    if dataset not in DATASETS:
        raise ValueError("unknown dataset")
    n, classes = _validate_features(features)
    safe_supervision = _validate_supervision(supervision, n, classes)
    text = _validate_hidden(hidden, n, dataset)
    _validate_splits(split_ids, safe_supervision, n)
    pseudo_ids, pseudo_targets = _validate_pipeline(
        pipeline, dataset, n, classes, safe_supervision["test_ids"],
        features["logp_g"], features["logp_l"],
    )
    train = {
        "graph": features["gnn_embeddings"].detach(),
        "text": text,
        "logp_g": features["logp_g"].detach(),
        "logp_l": features["logp_l"].detach(),
        "gold_ids": safe_supervision["gold_ids"],
        "gold_targets": safe_supervision["gold_targets"],
        "pseudo_ids": pseudo_ids,
        "pseudo_targets": pseudo_targets,
    }
    validation = {
        "ids": safe_supervision["selection_ids"],
        "targets": safe_supervision["selection_targets"],
    }
    inference = {
        "node_ids": features["node_ids"].detach(),
        "query_ids": safe_supervision["test_ids"],
        "threshold_ids": safe_supervision["threshold_ids"],
    }
    metadata = {
        "input_hashes": dict(input_hashes),
        "counts": {
            "nodes": n, "classes": classes,
            "graph_width": features["gnn_embeddings"].shape[1],
            "text_width": text.shape[1],
            "gold": train["gold_ids"].numel(),
            "validation": validation["ids"].numel(),
            "threshold": inference["threshold_ids"].numel(),
            "query": inference["query_ids"].numel(),
            "pseudo": train["pseudo_ids"].numel(),
        },
        "provenance": {
            "dataset": dataset,
            "router": router,
            "base_export": base_export,
            "test_targets_loaded": False,
            "pseudo_weighting": "uniform",
        },
    }
    validate_output(train, validation, inference, metadata)
    return train, validation, inference, metadata


def _verify_hashes(records: dict[str, str]) -> dict[str, str]:
    normalized = {}
    for raw_path, expected in records.items():
        path = Path(raw_path).resolve()
        if not isinstance(expected, str) or len(expected) != 64 or sha256(path) != expected:
            raise ValueError(f"input hash mismatch: {path}")
        normalized[str(path)] = expected
    return normalized


def _load_base(dataset: str, base_root: Path):
    cell = base_root.resolve() / dataset
    complete_path = cell / "complete.json"
    complete = _exact_dict(
        _read_json(complete_path),
        frozenset({"schema_version", "dataset", "test_labels_exported", "artifacts"}),
        "historical export completion",
    )
    expected_artifacts = frozenset({"features.pt", "supervision.pt", "pools.pt", "policies.json", "metadata.json"})
    if (complete["schema_version"] != 1 or complete["dataset"] != dataset
            or complete["test_labels_exported"] is not False
            or not isinstance(complete["artifacts"], dict)
            or set(complete["artifacts"]) != expected_artifacts):
        raise ValueError("invalid historical safe export completion")
    artifact_hashes = {
        str((cell / name).resolve()): digest
        for name, digest in complete["artifacts"].items()
    }
    input_hashes = _verify_hashes(artifact_hashes)
    input_hashes[str(complete_path.resolve())] = sha256(complete_path)
    metadata_path = cell / "metadata.json"
    metadata = _read_json(metadata_path)
    expected_metadata = frozenset({
        "schema_version", "dataset", "node_count", "class_count", "gnn_width", "llm_width",
        "llm_hidden_path", "source_hashes", "router_root", "selected_b_checkpoint",
        "selected_c_checkpoint", "split_audit", "protocol", "candidate_ids",
        "pseudo_candidate_count", "test_labels_exported", "score", "policy_strata",
        "gold_count", "model_selection_count", "threshold_count", "test_count",
        "pool_counts", "upstream_validation_reuse", "llm_cache_limitation",
    })
    _exact_dict(metadata, expected_metadata, "historical metadata")
    if (metadata["schema_version"] != 1 or metadata["dataset"] != dataset
            or metadata["test_labels_exported"] is not False
            or not isinstance(metadata["source_hashes"], dict)):
        raise ValueError("invalid historical metadata identity/isolation")
    source_hashes = _verify_hashes(metadata["source_hashes"])
    input_hashes.update(source_hashes)
    split_paths = [Path(path) for path in source_hashes if Path(path).name == "split_ids.json"]
    if len(split_paths) != 1:
        raise ValueError("historical metadata does not bind one independent split_ids.json")
    hidden_path = Path(metadata["llm_hidden_path"]).resolve()
    if str(hidden_path) not in source_hashes:
        raise ValueError("historical metadata does not hash-bind hidden features")
    features = torch.load(cell / "features.pt", map_location="cpu", weights_only=True, mmap=True)
    supervision = torch.load(cell / "supervision.pt", map_location="cpu", weights_only=True)
    hidden = torch.load(hidden_path, map_location="cpu", weights_only=True, mmap=True)
    splits = _read_json(split_paths[0])
    return features, supervision, hidden, splits, metadata, input_hashes


def _load_v6(dataset: str, router_root: Path):
    root = router_root.resolve()
    complete_path = root / "COMPLETE.json"
    frozen_path = root / "FROZEN.json"
    audit_path = root / "FINAL_AUDIT.json"
    seal_path = root / "INPUT_SEAL.json"
    manifest_path = root / "pipeline/MANIFEST.json"
    complete = _exact_dict(
        _read_json(complete_path),
        frozenset({"all_checks_pass", "archival_origin_frozen_sha256", "frozen_sha256"}),
        "v6 completion",
    )
    frozen = _exact_dict(
        _read_json(frozen_path),
        frozenset({"archival_destination", "archival_origin", "files", "input_seal",
                   "original_frozen_sha256", "slurm_job_id"}),
        "v6 freeze",
    )
    audit = _read_json(audit_path)
    seal = _exact_dict(_read_json(seal_path), frozenset({"files"}), "v6 input seal")
    if (complete["all_checks_pass"] is not True or complete["frozen_sha256"] != sha256(frozen_path)
            or complete["archival_origin_frozen_sha256"] != frozen["original_frozen_sha256"]
            or audit.get("all_checks_pass") is not True
            or audit.get("archival_frozen_sha256") != sha256(frozen_path)
            or audit.get("original_frozen_sha256") != frozen["original_frozen_sha256"]
            or frozen["input_seal"] != sha256(seal_path)):
        raise ValueError("invalid v6 completion/freeze/audit binding")
    if not isinstance(seal["files"], dict) or not seal["files"]:
        raise ValueError("invalid v6 input seal records")
    if not isinstance(frozen["files"], dict):
        raise ValueError("invalid v6 frozen file records")
    frozen_manifest_hash = frozen["files"].get(str(manifest_path))
    if frozen_manifest_hash != sha256(manifest_path):
        raise ValueError("v6 frozen manifest hash mismatch")
    manifest = _exact_dict(_read_json(manifest_path), frozenset({"schema", "datasets"}), "v6 manifest")
    if (manifest["schema"] != "router_v8_v5_manifest_v1"
            or not isinstance(manifest["datasets"], dict)
            or set(manifest["datasets"]) != set(DATASETS)):
        raise ValueError("invalid v6 pipeline manifest")
    item = _exact_dict(manifest["datasets"][dataset], frozenset({"path", "sha256"}), "v6 dataset manifest")
    if item["path"] != f"pipeline/{dataset}.pt":
        raise ValueError("v6 pipeline path mismatch")
    pipeline_path = root / item["path"]
    if sha256(pipeline_path) != item["sha256"]:
        raise ValueError("v6 pipeline hash mismatch")
    pipeline = torch.load(pipeline_path, map_location="cpu", weights_only=True, mmap=True)
    paths = (complete_path, frozen_path, audit_path, seal_path, manifest_path, pipeline_path)
    return pipeline, {str(path.resolve()): sha256(path) for path in paths}


def load_data(dataset: str, *, base_root: str | Path = BASE_ROOT,
              router_root: str | Path = ROUTER_ROOT):
    """Load one authenticated dataset without exposing Router weights or test truth."""
    require_slurm()
    if dataset not in DATASETS:
        raise ValueError("unknown dataset")
    base_root, router_root = Path(base_root), Path(router_root)
    features, supervision, hidden, splits, old_meta, input_hashes = _load_base(dataset, base_root)
    pipeline, v6_hashes = _load_v6(dataset, router_root)
    input_hashes.update(v6_hashes)
    result = assemble_data(
        features, supervision, hidden, pipeline, splits,
        dataset=dataset,
        input_hashes=input_hashes,
        base_export=str((base_root.resolve() / dataset)),
        router=str(router_root.resolve()),
    )
    counts = result[3]["counts"]
    recorded = {
        "nodes": old_meta["node_count"], "classes": old_meta["class_count"],
        "graph_width": old_meta["gnn_width"], "text_width": old_meta["llm_width"],
        "gold": old_meta["gold_count"], "validation": old_meta["model_selection_count"],
        "threshold": old_meta["threshold_count"], "query": old_meta["test_count"],
    }
    if any(counts[key] != value for key, value in recorded.items()):
        raise ValueError("assembled audit counts differ from historical metadata")
    return result
