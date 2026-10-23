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


def verified_complete(path: str | Path) -> dict | None:
    complete_path = Path(path)
    if not complete_path.is_file():
        return None
    payload = json.loads(complete_path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        return None
    for kind in ("selection", "scores"):
        artifact = Path(payload[f"{kind}_path"])
        if not artifact.is_file() or sha256(artifact) != payload[f"{kind}_sha256"]:
            return None
    reject_test_information(json.loads(Path(payload["selection_path"]).read_text()))
    return payload


__all__ = [
    "reject_test_information",
    "sha256",
    "torch_save_atomic",
    "verified_complete",
    "write_json_atomic",
]
