from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def torch_save_atomic(path: str | Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def reject_test_information(payload: object) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key).lower()
            if lowered.startswith("test") or "test_" in lowered:
                raise ValueError(f"selection contains test-derived field: {key}")
            reject_test_information(value)
    elif isinstance(payload, list):
        for value in payload:
            reject_test_information(value)


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _verify(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    if sha256(path) != expected:
        raise ValueError(f"{label} hash mismatch: {path}")


def build_frozen_manifest(
    run_root: str | Path,
    *,
    datasets: tuple[str, ...],
    code_paths: tuple[str | Path, ...],
) -> dict:
    root = Path(run_root).resolve()
    endpoints = {}
    for dataset in datasets:
        complete_path = root / "selection" / dataset / "complete.json"
        if not complete_path.is_file():
            raise FileNotFoundError(f"missing complete endpoint: {dataset}")
        complete = _load_json(complete_path)
        if complete.get("status") != "complete":
            raise ValueError(f"endpoint is not complete: {dataset}")
        selection_path = Path(complete["selection_path"]).resolve()
        scores_path = Path(complete["scores_path"]).resolve()
        _verify(selection_path, complete["selection_sha256"], "selection")
        _verify(scores_path, complete["scores_sha256"], "scores")
        reject_test_information(_load_json(selection_path))
        endpoints[dataset] = {
            "complete_path": str(complete_path),
            "complete_sha256": sha256(complete_path),
            "selection_path": str(selection_path),
            "selection_sha256": complete["selection_sha256"],
            "scores_path": str(scores_path),
            "scores_sha256": complete["scores_sha256"],
        }
    code_hashes = {}
    for raw_path in code_paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen code: {path}")
        code_hashes[str(path)] = sha256(path)
    return {
        "status": "frozen_validation_selection",
        "datasets": list(datasets),
        "feature_family": "A20_logits_evidence",
        "training_backbone": "router_v6_full_preference_backbone",
        "endpoints": endpoints,
        "code_hashes": code_hashes,
    }


__all__ = [
    "build_frozen_manifest",
    "reject_test_information",
    "sha256",
    "torch_save_atomic",
    "write_json_atomic",
]
