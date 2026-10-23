from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_prediction_identity(
    old_path: str | Path,
    new_path: str | Path,
) -> dict:
    old = torch.load(Path(old_path), map_location="cpu", weights_only=False)
    new = torch.load(Path(new_path), map_location="cpu", weights_only=False)
    checks = {
        "gnn_logits_identical": torch.equal(old["gnn_logits"], new["gnn_logits"]),
        "llm_logits_identical": torch.equal(old["llm_logits"], new["llm_logits"]),
        "label_names_identical": list(old["label_names"]) == list(new["label_names"]),
    }
    if not all(checks.values()):
        raise AssertionError(f"prediction identity failed: {checks}")
    return {**checks, "all_checks_pass": True}
