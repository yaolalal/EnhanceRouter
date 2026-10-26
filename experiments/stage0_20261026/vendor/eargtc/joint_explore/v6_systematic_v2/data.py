"""Authenticated, label-isolated inputs for the v6 systematic study."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import os
from pathlib import Path
import sys
from typing import Iterable, Iterator

import torch


VALIDATION_SALT = "joint-v6-systematic-checkpoint-selection-v1"
_ROOT = Path(__file__).resolve().parents[3]


def _slurm_only() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Joint-v6 systematic numerical work must run under Slurm")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def restricted_torch_load(paths: Iterable[os.PathLike[str] | str]) -> Iterator[None]:
    """Temporarily allow ``torch.load`` only for exact canonical paths.

    File handles and other non-path inputs are rejected because they cannot be
    authenticated against the v6 producer manifest.  The original function is
    restored even when an import or load fails.
    """

    allowed = {Path(path).resolve(strict=False) for path in paths}
    original = torch.load

    def guarded(source, *args, **kwargs):
        if not isinstance(source, (str, os.PathLike)):
            raise ValueError("torch.load requires an authenticated filesystem path")
        if kwargs.get("weights_only") is not True:
            raise ValueError("authenticated torch.load requires weights_only=True")
        resolved = Path(source).resolve(strict=False)
        if resolved not in allowed:
            raise ValueError(f"torch.load path is outside the authenticated allowlist: {resolved}")
        return original(resolved, *args, **kwargs)

    torch.load = guarded
    try:
        yield
    finally:
        torch.load = original


def guarded_import(module_name: str):
    """Import a legacy adapter while denying every import-time tensor load."""

    with restricted_torch_load(()):
        return importlib.import_module(module_name)


def split_checkpoint_selection(
    ids: torch.Tensor, dataset: str, salt: str = VALIDATION_SALT
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split existing Joint-selection IDs by a label-blind stable hash rank."""

    _slurm_only()
    value = torch.as_tensor(ids)
    if value.dtype != torch.long or value.ndim != 1 or len(value) < 2:
        raise ValueError("Joint selection IDs must be an int64 vector with at least two rows")
    if len(value.unique()) != len(value) or bool((value < 0).any()):
        raise ValueError("Joint selection IDs must be unique non-negative IDs")
    ranked = sorted(
        value.tolist(),
        key=lambda node_id: (
            hashlib.sha256(f"{dataset}\0{node_id}\0{salt}".encode("utf-8")).hexdigest(),
            node_id,
        ),
    )
    cut = (len(ranked) + 1) // 2
    checkpoint = torch.tensor(sorted(ranked[:cut]), dtype=torch.long)
    selection = torch.tensor(sorted(ranked[cut:]), dtype=torch.long)
    return checkpoint, selection


def _id_hash(ids: torch.Tensor) -> str:
    value = torch.as_tensor(ids, dtype=torch.long).contiguous().cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _validate_hashes(records: dict[str, str], name: str) -> None:
    if not isinstance(records, dict):
        raise ValueError(f"{name} must be a hash mapping")
    for path, digest in records.items():
        if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid {name} record")


def build_safe_data(
    features: dict[str, torch.Tensor],
    supervision: dict[str, torch.Tensor],
    pools: dict[str, torch.Tensor],
    hidden: torch.Tensor,
    *,
    dataset: str,
    outer: int,
    input_hashes: dict[str, str],
    source_hashes: dict[str, str],
):
    """Convert frozen v6 arrays into strict train/checkpoint/select schemas."""

    _slurm_only()
    from eargtc.joint_cross_repair.artifacts import (
        validate_features,
        validate_pools,
        validate_supervision,
    )

    validate_features(features)
    n, classes = features["logp_g"].shape
    validate_supervision(supervision, n, classes)
    validate_pools(pools, features, supervision)
    if not torch.is_tensor(hidden) or hidden.ndim != 2 or hidden.shape[0] != n:
        raise ValueError("hidden features must be a rank-two node-aligned tensor")
    for start in range(0, n, 4096):
        if not bool(torch.isfinite(hidden[start : start + 4096]).all()):
            raise ValueError("hidden features contain nonfinite values")
    _validate_hashes(input_hashes, "input_hashes")
    _validate_hashes(source_hashes, "source_hashes")

    checkpoint_ids, selection_ids = split_checkpoint_selection(
        supervision["selection_ids"], dataset
    )
    target_by_id = dict(
        zip(supervision["selection_ids"].tolist(), supervision["selection_targets"].tolist())
    )
    checkpoint = {
        "ids": checkpoint_ids,
        "targets": torch.tensor([target_by_id[i] for i in checkpoint_ids.tolist()], dtype=torch.long),
    }
    selection = {
        "ids": selection_ids,
        "targets": torch.tensor([target_by_id[i] for i in selection_ids.tolist()], dtype=torch.long),
    }
    train = {
        "node_ids": features["node_ids"],
        "graph": features["gnn_embeddings"],
        "text": hidden,
        "logp_g": features["logp_g"],
        "logp_l": features["logp_l"],
        "choose_g": features["choose_g"],
        "agreement": features["agreement"],
        "selected_pred": features["selected_pred"],
        "gold_ids": supervision["gold_ids"],
        "gold_targets": supervision["gold_targets"],
        "query_ids": supervision["test_ids"],
        # In the frozen pool vocabulary, teach_l means selected GNN and
        # teach_g means selected LLM on accepted disagreements.
        "disagreement_choose_g_ids": pools["teach_l"].sort().values,
        "disagreement_choose_l_ids": pools["teach_g"].sort().values,
        "agreement_ids": pools["agreement"].sort().values,
    }
    inference = {
        "node_ids": features["node_ids"],
        "query_ids": supervision["test_ids"],
        "threshold_ids": supervision["threshold_ids"],
    }
    accepted = torch.cat(
        (train["disagreement_choose_g_ids"], train["disagreement_choose_l_ids"], train["agreement_ids"])
    ).sort().values
    metadata = {
        "dataset": str(dataset),
        "outer": int(outer),
        "input_hashes": dict(sorted(input_hashes.items())),
        "source_hashes": dict(sorted(source_hashes.items())),
        "identity_hashes": {
            "gold_ids": _id_hash(train["gold_ids"]),
            "checkpoint_ids": _id_hash(checkpoint["ids"]),
            "selection_ids": _id_hash(selection["ids"]),
            "threshold_ids": _id_hash(inference["threshold_ids"]),
            "query_ids": _id_hash(inference["query_ids"]),
            "accepted_ids": _id_hash(accepted),
        },
        "counts": {
            "nodes": n,
            "classes": classes,
            "graph_width": train["graph"].shape[1],
            "text_width": train["text"].shape[1],
            "gold": len(train["gold_ids"]),
            "checkpoint": len(checkpoint["ids"]),
            "selection": len(selection["ids"]),
            "threshold": len(inference["threshold_ids"]),
            "query": len(inference["query_ids"]),
            "accepted": len(accepted),
        },
        "provenance": {
            "router": "r8_v6",
            "test_targets_loaded": False,
            "validation_partition": "historical_joint_selection_label_blind_half",
        },
    }
    return train, checkpoint, selection, inference, metadata


def _loaded_eargtc_sources() -> dict[str, str]:
    """Hash every loaded Python source resolved inside this repository."""

    records: dict[str, str] = {}
    root = _ROOT.resolve()
    for _name, module in sorted(sys.modules.items()):
        raw = getattr(module, "__file__", None)
        if not raw:
            continue
        path = Path(raw).resolve(strict=False)
        if path.suffix == ".py" and path.is_relative_to(root) and path.is_file():
            records[str(path)] = _sha256(path)
    return records


def load_safe_data(outer: int, dataset: str, root: os.PathLike[str] | str | None = None):
    """Load only authenticated Router-v8(v6) arrays and adapt them safely."""

    _slurm_only()
    legacy = guarded_import("eargtc.joint_explore.v6_multiseed.experiment")
    run_root = Path(legacy.DEFAULT_RUN if root is None else root).resolve()
    paths = legacy.input_paths(run_root, int(outer), "r8_v6", str(dataset))
    tensor_keys = ("upstream", "embeddings", "supervision", "router", "hidden")
    allowed = {Path(paths[key]).resolve() for key in tensor_keys}

    config = legacy.read_json(legacy.config_path(run_root))
    legacy.validate_formal_config(config)
    parity = legacy.verify_router_parity(run_root, config)
    legacy.verify_files(config["source_code"])
    legacy.verify_files(config["preserved_snapshot"])
    with restricted_torch_load(allowed):
        features, supervision, pools, hidden, _split, bound = legacy.load_inputs(
            run_root, int(outer), "r8_v6", str(dataset)
        )
    config_path = Path(legacy.config_path(run_root)).resolve()
    combined = {**bound, str(config_path): _sha256(config_path)}
    for records in (parity, config["preserved_snapshot"]):
        for path, digest in records.items():
            if path in combined and combined[path] != digest:
                raise ValueError("conflicting authenticated input provenance")
            combined[path] = digest
    source_hashes = dict(config["source_code"])
    for path, digest in _loaded_eargtc_sources().items():
        if path in source_hashes and source_hashes[path] != digest:
            raise ValueError("conflicting loaded source provenance")
        source_hashes[path] = digest
    output = build_safe_data(
        features,
        supervision,
        pools,
        hidden,
        dataset=str(dataset),
        outer=int(outer),
        input_hashes=combined,
        source_hashes=source_hashes,
    )
    return output
