from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from pathlib import Path

import torch

from .router_replay import LABEL_FREE_NODE_KEYS
from .router_replay import RouterReplayResult


class ArtifactSchemaError(ValueError):
    pass


FORBIDDEN_KEYS = {
    "true_label",
    "gold_label",
    "correct",
    "pseudo_correct",
    "test_accuracy",
    "oracle",
}


def validate_label_free_row(row: dict) -> None:
    forbidden = sorted(set(row).intersection(FORBIDDEN_KEYS))
    if forbidden:
        raise ArtifactSchemaError(f"forbidden label/correctness key: {forbidden[0]}")
    missing = sorted(LABEL_FREE_NODE_KEYS - set(row))
    extra = sorted(set(row) - LABEL_FREE_NODE_KEYS)
    if missing or extra:
        raise ArtifactSchemaError(f"label-free schema mismatch: missing={missing}, extra={extra}")
    if row["scheme_b_route"] not in {"gnn", "llm"}:
        raise ArtifactSchemaError("scheme_b_route must be gnn or llm")
    if row["scheme_c_checkpoint"] not in {"factorized", "direct"}:
        raise ArtifactSchemaError("scheme_c_checkpoint must be factorized or direct")
    expected_pred = row["gnn_pred"] if row["scheme_b_route"] == "gnn" else row["llm_pred"]
    if int(row["scheme_b_selected_pred"]) != int(expected_pred):
        raise ArtifactSchemaError("scheme_b_selected_pred is inconsistent with scheme_b_route")
    expected_trust = (
        row["scheme_c_factorized_trust"]
        if row["scheme_c_checkpoint"] == "factorized"
        else row["scheme_c_direct_trust"]
    )
    if abs(float(row["scheme_c_selected_trust"]) - float(expected_trust)) > 1e-6:
        raise ArtifactSchemaError("scheme_c_selected_trust is inconsistent with its checkpoint")
    for key in (
        "scheme_b_preference_score",
        "scheme_b_threshold",
        "scheme_c_factorized_trust",
        "scheme_c_direct_trust",
        "scheme_c_selected_trust",
    ):
        value = float(row[key])
        if not math.isfinite(value):
            raise ArtifactSchemaError(f"non-finite value for {key}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(value: object, path: Path) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _scan_forbidden_keys(value: object, location: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in FORBIDDEN_KEYS:
                raise ArtifactSchemaError(f"forbidden key {key} in {location}")
            _scan_forbidden_keys(child, location)
    elif isinstance(value, list):
        for child in value:
            _scan_forbidden_keys(child, location)


def export_label_free(result: RouterReplayResult, output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    if (output_dir / "complete.json").exists():
        raise FileExistsError(f"completed export already exists: {output_dir / 'complete.json'}")
    if output_dir.exists():
        raise FileExistsError(f"refusing existing export directory: {output_dir}")
    staging = output_dir.parent / f".{result.dataset}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        rows = [node.as_dict() for node in result.nodes]
        for row in rows:
            validate_label_free_row(row)
        node_ids = [int(row["node_id"]) for row in rows]
        if len(node_ids) != len(set(node_ids)):
            raise ArtifactSchemaError("router nodes contain duplicate node ids")
        with (staging / "router_nodes.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

        forbidden_tensor_keys = sorted(set(result.tensors).intersection(FORBIDDEN_KEYS))
        if forbidden_tensor_keys:
            raise ArtifactSchemaError(
                f"forbidden tensor key: {forbidden_tensor_keys[0]}"
            )
        tensors = {}
        tensor_metadata = {}
        for name, value in result.tensors.items():
            tensor = torch.as_tensor(value).detach().cpu()
            if tensor.ndim == 0 or int(tensor.size(0)) != len(rows):
                raise ArtifactSchemaError(
                    f"tensor {name} must have first dimension equal to node count {len(rows)}"
                )
            tensors[name] = tensor
            tensor_metadata[name] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
        torch.save(tensors, staging / "router_tensors.pt")

        split_ids = {
            name: [int(value) for value in values]
            for name, values in result.split_ids.items()
        }
        _write_json(result.replay_audit, staging / "replay_audit.json")
        _write_json(split_ids, staging / "split_ids.json")
        for name, value in (
            ("replay_audit.json", result.replay_audit),
            ("split_ids.json", split_ids),
        ):
            _scan_forbidden_keys(value, name)

        artifacts = {}
        for name in (
            "router_nodes.jsonl",
            "router_tensors.pt",
            "replay_audit.json",
            "split_ids.json",
        ):
            path = staging / name
            artifacts[name] = {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        artifacts["router_tensors.pt"]["tensors"] = tensor_metadata
        complete = {
            "schema_version": 1,
            "dataset": result.dataset,
            "node_count": len(rows),
            "artifacts": artifacts,
            "all_checks_pass": bool(result.replay_audit.get("all_checks_pass", False)),
        }
        if not complete["all_checks_pass"]:
            raise ArtifactSchemaError("replay audit did not pass")
        _scan_forbidden_keys(complete, "complete.json")
        _write_json(complete, staging / "complete.json")
        os.rename(staging, output_dir)
        return complete
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
