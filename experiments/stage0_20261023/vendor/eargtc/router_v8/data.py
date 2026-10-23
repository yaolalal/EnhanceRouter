from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


DATASETS = ("cora", "citeseer", "pubmed", "arxiv", "ogbn-products")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_ids(node_ids: torch.Tensor | list[int]) -> str:
    if isinstance(node_ids, torch.Tensor):
        values = [int(value) for value in node_ids.detach().cpu().view(-1).tolist()]
    else:
        values = [int(value) for value in node_ids]
    payload = json.dumps(values, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _validated_ids(values: Any, *, name: str, num_nodes: int) -> torch.Tensor:
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a JSON list")
    ids = torch.tensor([int(value) for value in values], dtype=torch.long)
    if ids.numel() != torch.unique(ids).numel():
        raise ValueError(f"{name} contains duplicate node IDs")
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= int(num_nodes)):
        raise ValueError(f"{name} contains node IDs outside [0, {num_nodes})")
    return ids


def build_split_ids(path: str | Path, *, num_nodes: int) -> dict[str, Any]:
    """Load the frozen split and derive the historical query complement exactly."""
    raw = read_json(path)
    train_ids = _validated_ids(raw.get("train_ids"), name="train_ids", num_nodes=num_nodes)
    validation_ids = _validated_ids(
        raw.get("val_ids"), name="val_ids", num_nodes=num_nodes
    )
    if torch.isin(train_ids, validation_ids).any():
        raise ValueError("train and validation IDs overlap")
    excluded = torch.zeros(int(num_nodes), dtype=torch.bool)
    excluded[train_ids] = True
    excluded[validation_ids] = True
    query_ids = (~excluded).nonzero(as_tuple=False).view(-1).long()
    return {
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "query_ids": query_ids,
        "declared_hash": raw.get("hash"),
        "source_path": str(Path(path).resolve()),
    }


def build_state(
    gnn_pred: torch.Tensor, llm_pred: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Return 0=G, 1=L, 2=N without encoding class IDs as features."""
    gnn_pred = gnn_pred.detach().cpu().view(-1).long()
    llm_pred = llm_pred.detach().cpu().view(-1).long()
    labels = labels.detach().cpu().view(-1).long()
    if not (gnn_pred.shape == llm_pred.shape == labels.shape):
        raise ValueError("prediction and label shapes differ")
    state = torch.full_like(labels, 2)
    state[gnn_pred == labels] = 0
    state[(gnn_pred != labels) & (llm_pred == labels)] = 1
    return state


def select_conflicts(
    node_ids: torch.Tensor, gnn_pred: torch.Tensor, llm_pred: torch.Tensor
) -> torch.Tensor:
    node_ids = node_ids.detach().cpu().view(-1).long()
    gnn_pred = gnn_pred.detach().cpu().view(-1).long()
    llm_pred = llm_pred.detach().cpu().view(-1).long()
    if not (node_ids.shape == gnn_pred.shape == llm_pred.shape):
        raise ValueError("node ID and prediction shapes differ")
    return node_ids[gnn_pred != llm_pred]


def validate_bank(bank: dict[str, Any], *, labeled: bool) -> None:
    required = {
        "node_ids",
        "candidate_g",
        "candidate_l",
        "symmetric_global",
        "gnn_pred",
        "llm_pred",
        "candidate_names",
        "global_names",
    }
    missing = required - set(bank)
    if missing:
        raise ValueError(f"bank missing keys: {sorted(missing)}")
    if labeled and "state" not in bank:
        raise ValueError("labeled bank must contain state")
    if not labeled and "state" in bank:
        raise ValueError("unlabeled bank must not contain state")
    node_ids = bank["node_ids"]
    if not isinstance(node_ids, torch.Tensor) or node_ids.dtype != torch.long:
        raise ValueError("node_ids must be a long tensor")
    node_ids = node_ids.view(-1)
    if torch.unique(node_ids).numel() != node_ids.numel():
        raise ValueError("bank node IDs must be unique")
    n = node_ids.numel()
    candidate_g = bank["candidate_g"]
    candidate_l = bank["candidate_l"]
    global_features = bank["symmetric_global"]
    if not all(isinstance(value, torch.Tensor) for value in (candidate_g, candidate_l, global_features)):
        raise ValueError("feature values must be tensors")
    if candidate_g.ndim != 2 or candidate_l.ndim != 2 or global_features.ndim != 2:
        raise ValueError("feature tensors must be rank two")
    if candidate_g.shape != candidate_l.shape:
        raise ValueError("candidate widths or row counts differ")
    if candidate_g.size(0) != n or global_features.size(0) != n:
        raise ValueError("feature row counts differ from node IDs")
    candidate_names = list(bank["candidate_names"])
    global_names = list(bank["global_names"])
    if len(candidate_names) != candidate_g.size(1):
        raise ValueError("candidate names and width differ")
    if len(global_names) != global_features.size(1):
        raise ValueError("global names and width differ")
    if len(set(candidate_names)) != len(candidate_names) or len(set(global_names)) != len(global_names):
        raise ValueError("feature names must be unique")
    for name in ("gnn_pred", "llm_pred"):
        value = bank[name]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.long or value.view(-1).numel() != n:
            raise ValueError(f"{name} must be an aligned long tensor")
    if torch.any(bank["gnn_pred"].view(-1) == bank["llm_pred"].view(-1)):
        raise ValueError("bank may contain conflict rows only")
    if labeled:
        state = bank["state"]
        if not isinstance(state, torch.Tensor) or state.dtype != torch.long or state.view(-1).numel() != n:
            raise ValueError("state must be an aligned long tensor")
        if state.numel() and (int(state.min()) < 0 or int(state.max()) > 2):
            raise ValueError("state values must be 0=G, 1=L, or 2=N")
    for name, value in (
        ("candidate_g", candidate_g),
        ("candidate_l", candidate_l),
        ("symmetric_global", global_features),
    ):
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite floating point")


def load_bank(path: str | Path) -> dict[str, Any]:
    """Load a plain Router V8 bank on CPU."""
    bank = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(bank, dict):
        raise ValueError(f"Expected plain dictionary bank: {path}")
    return bank
