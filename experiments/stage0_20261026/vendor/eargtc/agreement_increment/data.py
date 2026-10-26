"""Hash-bound, agreement-only banks built without loading whole-graph truth."""
from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import torch

from eargtc.router_v8.evidence import (
    PROBABILITY_CANDIDATE_NAMES,
    PROBABILITY_GLOBAL_NAMES,
    STRUCTURAL_CANDIDATE_NAMES,
    STRUCTURAL_GLOBAL_NAMES,
    probability_evidence,
)
from eargtc.router_v8.prototypes import (
    GRAPH_CANDIDATE_NAMES,
    GRAPH_GLOBAL_NAMES,
    fit_class_prototypes,
    prototype_evidence,
)

from .artifacts import DATASETS, REPO, read_json, require_slurm, sha256, write_json


SCHEMA = "agreement_increment_bank_v1"
SAFE_ROOT = REPO / "runs/joint_direct_numbered_router_export_20260813_v1"
OLD_V8_ROOT = REPO / "runs/router_v8_trust_preference_20260906_v1/banks"
HIDDEN_ROOT = REPO / "runs/router_v7_decision_llm_hidden_20260827_v1"
SSL_ROOT = REPO / "runs/gcn_ssl_v2_selected_20260623_v1"
HISTORICAL_ROOT = REPO / "runs/router_bc_split_c_20260825_v3"

_COMMON_KEYS = frozenset({"dataset", "ids", "tab", "g", "l", "gnn_pred", "llm_pred"})
_VALIDATION_KEYS = _COMMON_KEYS | {"target", "cal_mask", "policy_mask"}
_QUERY_KEYS = _COMMON_KEYS
_SAFE_TENSOR_KEYS = frozenset({
    "gnn_embeddings", "gnn_logits", "llm_logits", "router_evidence", "scheme_b_preference_score",
    "scheme_c_direct_trust", "scheme_c_factorized_trust", "token_ids",
})
_SUPERVISION_KEYS = frozenset({
    "gold_support_ids", "gold_support_targets", "official_validation_ids", "official_validation_targets",
})
_STRUCTURAL_NAMES = ("log_degree", "neighbor_gnn_agreement", "neighbor_gnn_entropy")
_CANDIDATE_NAMES = tuple(PROBABILITY_CANDIDATE_NAMES + STRUCTURAL_CANDIDATE_NAMES + GRAPH_CANDIDATE_NAMES)
_GLOBAL_NAMES = tuple(PROBABILITY_GLOBAL_NAMES + STRUCTURAL_GLOBAL_NAMES + GRAPH_GLOBAL_NAMES)
_TAB_NAMES = tuple(f"candidate:{name}" for name in _CANDIDATE_NAMES) + tuple(
    f"symmetric:{name}" for name in _GLOBAL_NAMES
)


def _id_hash(ids: torch.Tensor) -> str:
    values = torch.as_tensor(ids, dtype=torch.long).view(-1).tolist()
    payload = json.dumps(values, separators=(",", ":")).encode("utf-8")
    import hashlib
    return hashlib.sha256(payload).hexdigest()


def _line_id_hash(ids: torch.Tensor) -> str:
    import hashlib
    payload = "".join(f"{int(value)}\n" for value in torch.as_tensor(ids).view(-1).tolist())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _record(records: dict[str, str], path: Path, expected: str | None = None) -> Path:
    path = path.resolve()
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise ValueError(f"source hash mismatch: {path}")
    records[str(path)] = actual
    return path


def _ids(value: Any, *, name: str, num_nodes: int) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.long).detach().cpu().view(-1)
    if result.unique().numel() != result.numel():
        raise ValueError(f"{name} IDs are not unique")
    if result.numel() and bool(((result < 0) | (result >= num_nodes)).any()):
        raise ValueError(f"{name} IDs are outside the node universe")
    return result


def validate_bank(bank: dict[str, Any], *, split: str, dataset: str | None = None) -> None:
    if split not in {"validation", "query"}:
        raise ValueError(f"unknown bank split: {split}")
    expected = _VALIDATION_KEYS if split == "validation" else _QUERY_KEYS
    if not isinstance(bank, dict) or set(bank) != expected:
        actual = set(bank) if isinstance(bank, dict) else set()
        raise ValueError(
            f"bank schema mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    if not isinstance(bank["dataset"], str) or not bank["dataset"]:
        raise ValueError("dataset must be a non-empty string")
    if dataset is not None and bank["dataset"] != dataset:
        raise ValueError("bank dataset mismatch")
    ids = bank["ids"]
    if not isinstance(ids, torch.Tensor) or ids.dtype != torch.long or ids.ndim != 1:
        raise ValueError("ids must be an int64 vector")
    if ids.unique().numel() != ids.numel() or bool((ids < 0).any()):
        raise ValueError("ids must be unique and non-negative")
    n = ids.numel()
    for name, width, dtype in (
        ("tab", 43, torch.float32), ("g", 128, torch.float32), ("l", 4096, torch.float16),
    ):
        value = bank[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != dtype or value.ndim != 2
                or value.shape != (n, width)):
            raise ValueError(f"{name} must have shape ({n}, {width}) and dtype {dtype}")
        step = 4096 if name == "l" else max(n, 1)
        for start in range(0, n, step):
            if not bool(torch.isfinite(value[start:start + step]).all()):
                raise ValueError(f"{name} contains non-finite values")
    for name in ("gnn_pred", "llm_pred"):
        value = bank[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.long or value.ndim != 1
                or value.shape != (n,) or bool((value < 0).any())):
            raise ValueError(f"{name} must be an aligned non-negative int64 vector")
    if not torch.equal(bank["gnn_pred"], bank["llm_pred"]):
        raise ValueError("agreement bank contains expert disagreement")
    if split == "validation":
        for name in ("target", "cal_mask", "policy_mask"):
            value = bank[name]
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.bool or value.ndim != 1
                    or value.shape != (n,)):
                raise ValueError(f"{name} must be an aligned bool vector")
        if not torch.equal(bank["policy_mask"], ~bank["cal_mask"]):
            raise ValueError("calibration and policy masks must be exact complements")


def _compose_agreement_tab(
    candidate_g: torch.Tensor, candidate_l: torch.Tensor, symmetric: torch.Tensor
) -> torch.Tensor:
    candidate_g = torch.as_tensor(candidate_g, dtype=torch.float32)
    candidate_l = torch.as_tensor(candidate_l, dtype=torch.float32)
    symmetric = torch.as_tensor(symmetric, dtype=torch.float32)
    if candidate_g.ndim != 2 or candidate_g.shape[1] != 27 or candidate_l.shape != candidate_g.shape:
        raise ValueError("candidate evidence must contain two aligned 27-column views")
    if symmetric.ndim != 2 or symmetric.shape != (candidate_g.shape[0], 16):
        raise ValueError("symmetric evidence must contain 16 aligned columns")
    if not torch.equal(candidate_g, candidate_l):
        raise ValueError("agreement candidate evidence differs between experts")
    result = torch.cat((candidate_g, symmetric), dim=1).contiguous()
    if result.shape[1] != 43 or not bool(torch.isfinite(result).all()):
        raise ValueError("invalid Build43 evidence")
    return result


def _historical_masks(
    ids: torch.Tensor, dataset: str, historical: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    from eargtc.router_bc_split_c.calibration import deterministic_calibration_threshold_split

    ids = torch.as_tensor(ids, dtype=torch.long).view(-1)
    split = deterministic_calibration_threshold_split(ids, dataset, "agreement")
    calibration, policy = split.calibration.bool(), split.threshold.bool()
    cal_ids, policy_ids = ids[calibration], ids[policy]
    try:
        expected = historical["modules"]["agreement"]["calibration"]
    except (KeyError, TypeError) as exc:
        raise ValueError("historical agreement split record is missing") from exc
    cal_hash, policy_hash = _line_id_hash(cal_ids), _line_id_hash(policy_ids)
    if (cal_hash != expected.get("calibration_ids_sha256")
            or policy_hash != expected.get("threshold_ids_sha256")):
        raise ValueError(f"historical agreement split mismatch: {dataset}")
    audit = {
        "calibration_ids_sha256": cal_hash,
        "policy_ids_sha256": policy_hash,
        "calibration_count": int(calibration.sum()),
        "policy_count": int(policy.sum()),
    }
    return calibration, policy, audit


def _load_sources(dataset: str) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")
    records: dict[str, str] = {}
    safe_dir = SAFE_ROOT / dataset
    complete_path = safe_dir / "complete.json"
    complete = read_json(complete_path)
    if (complete.get("all_checks_pass") is not True or complete.get("test_labels_exported") is not False
            or complete.get("dataset") != dataset):
        raise ValueError("safe export completion or label isolation failed")
    _record(records, complete_path)
    needed = {
        "router_tensors.pt", "training_supervision.pt", "split_ids.json",
        "safe_input_metadata.json", "replay_audit.json",
    }
    artifacts = complete.get("artifacts", {})
    if not needed.issubset(artifacts):
        raise ValueError("safe export is missing required artifacts")
    for name in sorted(needed):
        item = artifacts[name]
        expected = item.get("sha256") if isinstance(item, dict) else None
        if not isinstance(expected, str):
            raise ValueError(f"safe export lacks a hash for {name}")
        _record(records, safe_dir / name, expected)

    metadata = read_json(safe_dir / "safe_input_metadata.json")
    replay = read_json(safe_dir / "replay_audit.json")
    num_nodes = int(metadata.get("node_count", -1))
    num_classes = int(metadata.get("class_count", -1))
    if (metadata.get("dataset") != dataset or metadata.get("test_labels_exported") is not False
            or num_nodes <= 0 or num_classes < 2 or replay.get("all_checks_pass") is not True):
        raise ValueError("safe export metadata or replay audit failed")
    evidence_names = list(replay.get("router_evidence_names", []))
    if len(evidence_names) != 29 or len(set(evidence_names)) != 29:
        raise ValueError("safe Router evidence schema changed")
    if any(name not in evidence_names for name in _STRUCTURAL_NAMES):
        raise ValueError("safe Router evidence lacks structural columns")

    raw = torch.load(safe_dir / "router_tensors.pt", map_location="cpu", weights_only=True, mmap=True)
    supervision = torch.load(safe_dir / "training_supervision.pt", map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or set(raw) != _SAFE_TENSOR_KEYS:
        raise ValueError("safe tensor schema changed or contains a forbidden field")
    if not isinstance(supervision, dict) or set(supervision) != _SUPERVISION_KEYS:
        raise ValueError("safe supervision schema changed or contains query truth")
    g = raw["gnn_embeddings"]
    logits_g, logits_l, router_evidence = raw["gnn_logits"], raw["llm_logits"], raw["router_evidence"]
    if g.dtype != torch.float32 or g.shape != (num_nodes, 128):
        raise ValueError("safe GNN embedding identity changed")
    if (logits_g.ndim != 2 or logits_g.shape != (num_nodes, num_classes)
            or logits_l.shape != logits_g.shape or router_evidence.shape != (num_nodes, 29)):
        raise ValueError("safe logits/evidence alignment changed")
    for value in (g, logits_g, logits_l, router_evidence):
        if not bool(torch.isfinite(value).all()):
            raise ValueError("safe feature source contains non-finite values")

    splits = read_json(safe_dir / "split_ids.json")
    gold_ids = _ids(splits.get("train", []), name="gold", num_nodes=num_nodes)
    validation_ids = _ids(splits.get("validation", []), name="validation", num_nodes=num_nodes)
    query_ids = _ids(splits.get("query", []), name="query", num_nodes=num_nodes)
    owner = torch.zeros(num_nodes, dtype=torch.int8)
    for ids in (gold_ids, validation_ids, query_ids):
        if bool(owner[ids].any()):
            raise ValueError("historical split identities overlap")
        owner[ids] = 1
    if not bool(owner.all()):
        raise ValueError("historical splits do not retain the complete node universe")
    if (not torch.equal(gold_ids, supervision["gold_support_ids"])
            or not torch.equal(validation_ids, supervision["official_validation_ids"])):
        raise ValueError("safe label subsets do not match historical split identities")
    gold_targets = supervision["gold_support_targets"].long()
    validation_targets = supervision["official_validation_targets"].long()
    if (gold_targets.shape != gold_ids.shape or validation_targets.shape != validation_ids.shape
            or bool(((gold_targets < 0) | (gold_targets >= num_classes)).any())
            or bool(((validation_targets < 0) | (validation_targets >= num_classes)).any())):
        raise ValueError("safe support/validation labels are invalid")

    hidden_audit_path = HIDDEN_ROOT / "AUDIT.json"
    hidden_audit = read_json(hidden_audit_path)
    hidden_path = HIDDEN_ROOT / dataset / "hidden.pt"
    hidden_row = hidden_audit.get("datasets", {}).get(dataset, {})
    if (hidden_row.get("status") != "pass" or int(hidden_row.get("rows", -1)) != num_nodes
            or int(hidden_row.get("width", -1)) != 4096 or hidden_row.get("dtype") != "float16"):
        raise ValueError("LLM hidden audit identity failed")
    _record(records, hidden_audit_path)
    _record(records, hidden_path)
    hidden_payload = torch.load(hidden_path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(hidden_payload, dict) or set(hidden_payload) != {"node_ids", "hidden", "metadata"}:
        raise ValueError("LLM hidden schema changed or contains labels")
    hidden = hidden_payload["hidden"]
    if (hidden_payload["node_ids"].dtype != torch.long
            or not torch.equal(hidden_payload["node_ids"], torch.arange(num_nodes))
            or hidden.dtype != torch.float16 or hidden.shape != (num_nodes, 4096)):
        raise ValueError("LLM hidden row identity changed")
    for start in range(0, num_nodes, 4096):
        if not bool(torch.isfinite(hidden[start:start + 4096]).all()):
            raise ValueError("LLM hidden source contains non-finite values")

    ssl_manifest_path = SSL_ROOT / "manifest.json"
    ssl_config_path = SSL_ROOT / dataset / "ssl_config.json"
    ssl_path = SSL_ROOT / dataset / "encoder_and_embeddings.pt"
    ssl_manifest = read_json(ssl_manifest_path)
    declared = ssl_manifest.get("datasets", {}).get(dataset, {}).get("artifact_paths", {}).get(
        "encoder_and_embeddings"
    )
    if declared is None or Path(declared).resolve() != ssl_path.resolve():
        raise ValueError("selected SSL embedding manifest points elsewhere")
    _record(records, ssl_manifest_path)
    _record(records, ssl_config_path)
    _record(records, ssl_path)
    ssl_payload = torch.load(ssl_path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(ssl_payload, dict) or "embeddings" not in ssl_payload:
        raise ValueError("selected SSL artifact lacks embeddings")
    ssl = ssl_payload["embeddings"]
    if ssl.dtype != torch.float32 or ssl.shape != (num_nodes, 128) or not bool(torch.isfinite(ssl).all()):
        raise ValueError("selected SSL prototype embeddings are invalid")

    old_dir = OLD_V8_ROOT / dataset
    old_manifest_path = old_dir / "manifest.json"
    old_manifest = read_json(old_manifest_path)
    if old_manifest.get("schema") != "router_v8_feature_bank_v1" or old_manifest.get("dataset") != dataset:
        raise ValueError("old v8 evidence manifest identity failed")
    if (old_manifest.get("feature_schema", {}).get("candidate_width") != 27
            or old_manifest.get("feature_schema", {}).get("global_width") != 16
            or old_manifest.get("feature_schema", {}).get("candidate_names") != list(_CANDIDATE_NAMES)
            or old_manifest.get("feature_schema", {}).get("global_names") != list(_GLOBAL_NAMES)):
        raise ValueError("old v8 evidence schema is not Build43-compatible")
    old_query_path = old_dir / "query.pt"
    _record(records, old_manifest_path)
    _record(records, old_query_path, old_manifest.get("bank_sha256", {}).get("query"))
    old_query = torch.load(old_query_path, map_location="cpu", weights_only=True)
    old_expected = {
        "node_ids", "candidate_g", "candidate_l", "symmetric_global", "gnn_pred", "llm_pred",
        "candidate_names", "global_names",
    }
    if not isinstance(old_query, dict) or set(old_query) != old_expected:
        raise ValueError("old v8 query evidence schema changed or contains truth")

    historical_path = HISTORICAL_ROOT / dataset / "metrics.json"
    _record(records, historical_path)
    return {
        "num_nodes": num_nodes,
        "num_classes": num_classes,
        "raw": raw,
        "gold_ids": gold_ids,
        "gold_targets": gold_targets,
        "validation_ids": validation_ids,
        "validation_targets": validation_targets,
        "query_ids": query_ids,
        "hidden": hidden,
        "ssl": ssl,
        "evidence_names": evidence_names,
        "old_query": old_query,
        "old_manifest": old_manifest,
        "historical": read_json(historical_path),
        "input_hashes": dict(sorted(records.items())),
    }


def _build43(source: dict[str, Any], row_ids: torch.Tensor) -> torch.Tensor:
    ids = torch.as_tensor(row_ids, dtype=torch.long).view(-1)
    raw = source["raw"]
    prob_g = raw["gnn_logits"][ids].float().softmax(1)
    prob_l = raw["llm_logits"][ids].float().softmax(1)
    pred_g, pred_l = prob_g.argmax(1), prob_l.argmax(1)
    probability = probability_evidence(prob_g, prob_l, pred_g, pred_l)

    names = source["evidence_names"]
    evidence = raw["router_evidence"][ids].float()
    log_degree = evidence[:, names.index("log_degree")]
    neighbor_agreement = evidence[:, names.index("neighbor_gnn_agreement")]
    neighbor_entropy = evidence[:, names.index("neighbor_gnn_entropy")]
    structural_g = torch.stack(
        (neighbor_agreement, torch.zeros_like(neighbor_agreement), torch.ones_like(neighbor_agreement)), dim=1
    )
    structural_l = structural_g.clone() if torch.equal(pred_g, pred_l) else None
    structural_global = torch.stack(
        (neighbor_entropy, log_degree, (log_degree == 0).float()), dim=1
    )

    fitted = fit_class_prototypes(
        source["ssl"], source["gold_ids"], source["gold_targets"], num_classes=source["num_classes"]
    )
    prototype = prototype_evidence(
        embeddings=source["ssl"], row_ids=ids, candidate_g=pred_g, candidate_l=pred_l, fitted=fitted
    )
    candidate_g = torch.cat((probability[0], structural_g, prototype[0]), dim=1)
    if structural_l is None:
        # Safe Router evidence exports only the GNN-candidate neighbor vote. The L candidate
        # is unnecessary for agreement rows and is never synthesized for disagreements.
        candidate_l = candidate_g
    else:
        candidate_l = torch.cat((probability[1], structural_l, prototype[1]), dim=1)
    symmetric = torch.cat((probability[2], structural_global, prototype[2]), dim=1)
    return _compose_agreement_tab(candidate_g, candidate_l, symmetric)


def _audit_build43(source: dict[str, Any]) -> dict[str, Any]:
    old = source["old_query"]
    ids = old["node_ids"]
    if ids.numel() == 0:
        raise ValueError("old v8 query has no evidence rows for Build43 audit")
    raw = source["raw"]
    prob_g = raw["gnn_logits"][ids].float().softmax(1)
    prob_l = raw["llm_logits"][ids].float().softmax(1)
    probability = probability_evidence(prob_g, prob_l, old["gnn_pred"], old["llm_pred"])
    names = source["evidence_names"]
    evidence = raw["router_evidence"][ids].float()
    log_degree = evidence[:, names.index("log_degree")]
    neighbor_agreement = evidence[:, names.index("neighbor_gnn_agreement")]
    neighbor_entropy = evidence[:, names.index("neighbor_gnn_entropy")]
    structural_g = torch.stack(
        (neighbor_agreement, torch.zeros_like(neighbor_agreement), torch.ones_like(neighbor_agreement)), dim=1
    )
    structural_global = torch.stack((neighbor_entropy, log_degree, (log_degree == 0).float()), dim=1)
    fitted = fit_class_prototypes(
        source["ssl"], source["gold_ids"], source["gold_targets"], num_classes=source["num_classes"]
    )
    prototype = prototype_evidence(
        embeddings=source["ssl"], row_ids=ids, candidate_g=old["gnn_pred"],
        candidate_l=old["llm_pred"], fitted=fitted,
    )
    candidate_g = torch.cat((probability[0], structural_g, prototype[0]), dim=1)
    symmetric = torch.cat((probability[2], structural_global, prototype[2]), dim=1)
    expected_g, expected_global = old["candidate_g"], old["symmetric_global"]
    max_g = float((candidate_g - expected_g).abs().max())
    max_global = float((symmetric - expected_global).abs().max())
    if not torch.allclose(candidate_g, expected_g, atol=2e-6, rtol=2e-6):
        raise ValueError("Build43 candidate evidence differs from frozen v8")
    if not torch.allclose(symmetric, expected_global, atol=2e-6, rtol=2e-6):
        raise ValueError("Build43 symmetric evidence differs from frozen v8")
    return {
        "source": "frozen_router_v8_query_evidence",
        "rows": int(ids.numel()),
        "candidate_g_max_abs_error": max_g,
        "symmetric_max_abs_error": max_global,
        "all_checks_pass": True,
    }


def _bank(source: dict[str, Any], dataset: str, ids: torch.Tensor, *, target=None, masks=None):
    raw = source["raw"]
    ids = torch.as_tensor(ids, dtype=torch.long).view(-1)
    pred_g = raw["gnn_logits"][ids].argmax(1).long()
    pred_l = raw["llm_logits"][ids].argmax(1).long()
    if not torch.equal(pred_g, pred_l):
        raise ValueError("agreement selection contains expert disagreement")
    result = {
        "dataset": dataset,
        "ids": ids,
        "tab": _build43(source, ids),
        "g": raw["gnn_embeddings"][ids].float().contiguous(),
        "l": source["hidden"][ids].to(torch.float16).contiguous(),
        "gnn_pred": pred_g,
        "llm_pred": pred_l,
    }
    if target is not None:
        cal_mask, policy_mask = masks
        result.update(
            target=torch.as_tensor(target, dtype=torch.bool).view(-1),
            cal_mask=cal_mask.bool(),
            policy_mask=policy_mask.bool(),
        )
    validate_bank(result, split="validation" if target is not None else "query", dataset=dataset)
    return result


def prepare(root: Path, dataset: str) -> dict[str, Any]:
    """Write immutable validation/query agreement banks from audited safe sources."""
    require_slurm()
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")
    destination = Path(root).resolve() / "banks" / dataset
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite agreement bank: {destination}")
    source = _load_sources(dataset)
    raw = source["raw"]
    pred_g = raw["gnn_logits"].argmax(1)
    pred_l = raw["llm_logits"].argmax(1)
    validation_ids = source["validation_ids"]
    query_ids = source["query_ids"]
    validation_agreement = validation_ids[pred_g[validation_ids] == pred_l[validation_ids]]
    query_agreement = query_ids[pred_g[query_ids] == pred_l[query_ids]]
    lookup = torch.full((source["num_nodes"],), -1, dtype=torch.long)
    lookup[validation_ids] = source["validation_targets"]
    target = pred_g[validation_agreement] == lookup[validation_agreement]
    cal_mask, policy_mask, split_audit = _historical_masks(
        validation_agreement, dataset, source["historical"]
    )
    build43_audit = _audit_build43(source)
    validation = _bank(
        source, dataset, validation_agreement, target=target, masks=(cal_mask, policy_mask)
    )
    query = _bank(source, dataset, query_agreement)

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{dataset}.stage-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        torch.save(validation, stage / "validation.pt")
        torch.save(query, stage / "query.pt")
        bank_hashes = {name: sha256(stage / name) for name in ("validation.pt", "query.pt")}
        split_ids = {
            "gold": source["gold_ids"].tolist(),
            "validation": validation_ids.tolist(),
            "query": query_ids.tolist(),
        }
        manifest = {
            "schema": SCHEMA,
            "dataset": dataset,
            "bank_hashes": bank_hashes,
            "input_hashes": source["input_hashes"],
            "field_schema": {
                "validation": sorted(_VALIDATION_KEYS),
                "query": sorted(_QUERY_KEYS),
            },
            "feature_schema": {
                "tab_width": 43,
                "candidate_width": 27,
                "symmetric_width": 16,
                "tab_names": list(_TAB_NAMES),
                "candidate_names": list(_CANDIDATE_NAMES),
                "symmetric_names": list(_GLOBAL_NAMES),
                "gnn_width": 128,
                "llm_width": 4096,
            },
            "split_ids": split_ids,
            "split_hashes": {name: _id_hash(torch.tensor(ids)) for name, ids in split_ids.items()},
            "agreement_ids": {
                "validation": validation_agreement.tolist(),
                "query": query_agreement.tolist(),
            },
            "agreement_id_hashes": {
                "validation": _id_hash(validation_agreement),
                "query": _id_hash(query_agreement),
            },
            "historical_agreement_split": split_audit,
            "validation_target_counts": {
                "correct": int(target.sum()),
                "incorrect": int((~target).sum()),
            },
            "counts": {
                "gold_population": int(source["gold_ids"].numel()),
                "validation_population": int(validation_ids.numel()),
                "validation_agreement": int(validation_agreement.numel()),
                "calibration_agreement": int(cal_mask.sum()),
                "policy_agreement": int(policy_mask.sum()),
                "query_population": int(query_ids.numel()),
                "query_agreement": int(query_agreement.numel()),
            },
            "embedding_coverage": {
                "gnn_full_rows": source["num_nodes"],
                "llm_full_rows": source["num_nodes"],
                "ssl_prototype_full_rows": source["num_nodes"],
                "validation_rows": int(validation_agreement.numel()),
                "query_rows": int(query_agreement.numel()),
                "all_requested_rows_covered": True,
            },
            "prototype_fit": {
                "representation": "selected SSL graph embeddings",
                "fit_role": "gold support labels only",
                "fit_ids": source["gold_ids"].tolist(),
                "fit_ids_sha256": _id_hash(source["gold_ids"]),
                "query_fit_ids": [],
            },
            "build43_audit": build43_audit,
            "label_access": {
                "gold_support_labels_for_prototypes": True,
                "validation_labels_for_target": True,
                "query_labels_read": False,
                "query_truth_exported": False,
                "whole_graph_truth_loaded": False,
            },
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
        }
        write_json(stage / "manifest.json", manifest)
        os.replace(stage, destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def load_bank(root: Path, dataset: str, split: str) -> dict[str, Any]:
    """Verify the immutable bank hash and exact allowlisted schema before returning it."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")
    if split not in {"validation", "query"}:
        raise ValueError(f"unknown bank split: {split}")
    directory = Path(root).resolve() / "banks" / dataset
    manifest = read_json(directory / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("dataset") != dataset:
        raise ValueError("agreement bank manifest identity mismatch")
    filename = f"{split}.pt"
    expected = manifest.get("bank_hashes", {}).get(filename)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"agreement bank manifest lacks a valid hash for {filename}")
    path = directory / filename
    if sha256(path) != expected:
        raise ValueError(f"agreement bank hash mismatch: {path}")
    result = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    validate_bank(result, split=split, dataset=dataset)
    return result
