from __future__ import annotations

import json
from pathlib import Path

import torch

from eargtc.router_v7_decision.reporting import paired_bootstrap_difference

from .artifacts import sha256


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _verify_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen {label}: {path}")
    if sha256(path) != expected:
        raise ValueError(f"frozen {label} hash mismatch: {path}")


def verify_frozen_manifest(
    path: str | Path, *, datasets: tuple[str, ...]
) -> dict:
    manifest_path = Path(path).resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "frozen_validation_selection":
        raise ValueError("validation selections have not been frozen")
    if tuple(manifest.get("datasets", ())) != tuple(datasets):
        raise ValueError("frozen dataset order differs from evaluation contract")
    endpoints = manifest.get("endpoints", {})
    if set(endpoints) != set(datasets):
        raise ValueError("frozen endpoint set differs from evaluation contract")
    for dataset in datasets:
        endpoint = endpoints[dataset]
        for kind in ("complete", "selection", "scores"):
            artifact = Path(endpoint[f"{kind}_path"]).resolve()
            _verify_hash(artifact, endpoint[f"{kind}_sha256"], kind)
    for raw_path, digest in manifest.get("code_hashes", {}).items():
        _verify_hash(Path(raw_path).resolve(), digest, "code")
    return manifest


def align_frozen_comparator(
    payload: dict, node_ids: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_ids = torch.as_tensor(node_ids, dtype=torch.long).view(-1).cpu()
    expected_targets = torch.as_tensor(targets, dtype=torch.long).view(-1).cpu()
    actual_ids = torch.as_tensor(payload["node_ids"], dtype=torch.long).view(-1).cpu()
    actual_targets = (
        torch.as_tensor(payload["targets"], dtype=torch.long).view(-1).cpu()
    )
    if not torch.equal(actual_ids, expected_ids):
        raise ValueError("frozen comparator node IDs do not exactly align")
    if not torch.equal(actual_targets, expected_targets):
        raise ValueError("frozen comparator targets do not exactly align")
    method = payload["v6_scheme_b"]
    scores = torch.as_tensor(method["scores"], dtype=torch.float64).view(-1).cpu()
    routes = torch.as_tensor(method["route_gnn"], dtype=torch.bool).view(-1).cpu()
    if scores.numel() != expected_ids.numel() or routes.numel() != expected_ids.numel():
        raise ValueError("frozen comparator predictions do not align")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("frozen comparator scores contain non-finite values")
    return scores, routes


def paired_route_summary(
    first_routes: torch.Tensor,
    second_routes: torch.Tensor,
    targets: torch.Tensor,
    *,
    repetitions: int = 2000,
    seed: int = 42,
) -> dict:
    first = torch.as_tensor(first_routes, dtype=torch.bool).view(-1).cpu()
    second = torch.as_tensor(second_routes, dtype=torch.bool).view(-1).cpu()
    target = torch.as_tensor(targets, dtype=torch.long).view(-1).cpu()
    if first.numel() == 0 or first.shape != second.shape or first.shape != target.shape:
        raise ValueError("paired routes and targets must be nonempty and aligned")
    correct_first = first == target.bool()
    correct_second = second == target.bool()
    wins = int((correct_first & ~correct_second).sum().item())
    losses = int((~correct_first & correct_second).sum().item())
    return {
        "wins": wins,
        "losses": losses,
        "ties": int(first.numel() - wins - losses),
        "net_correct": wins - losses,
        "metrics": paired_bootstrap_difference(
            first,
            second,
            target,
            repetitions=repetitions,
            seed=seed,
        ),
    }


__all__ = [
    "align_frozen_comparator",
    "paired_route_summary",
    "verify_frozen_manifest",
]
