from __future__ import annotations

from collections.abc import Callable

import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .audit import assert_label_free_cache


_REQUIRED_METADATA = frozenset(
    {
        "dataset",
        "model_hash",
        "tokenizer_hash",
        "prompt_hash",
        "start",
        "end",
        "dtype",
        "hidden_width",
        "row_order_hash",
    }
)


@dataclass(frozen=True)
class HiddenManifest:
    dataset: str
    model_hash: str
    tokenizer_hash: str
    prompt_hash: str
    start: int
    end: int
    dtype: str
    hidden_width: int
    row_order_hash: str
    parity_max_abs_error: float = 0.0
    parity_max_probability_error: float = 0.0
    parity_argmax_match: bool = True

    @staticmethod
    def hash_ids(node_ids: torch.Tensor) -> str:
        ids = torch.as_tensor(node_ids, dtype=torch.int64).view(-1).cpu()
        return hashlib.sha256(ids.numpy().tobytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "HiddenManifest":
        missing = sorted(_REQUIRED_METADATA - set(value))
        if missing:
            raise ValueError(f"hidden metadata missing fields: {missing}")
        return cls(
            **{key: value[key] for key in cls.__dataclass_fields__ if key in value}
        )

    def stable_signature(self) -> tuple[str, str, str, str, str, int]:
        return (
            self.dataset,
            self.model_hash,
            self.tokenizer_hash,
            self.prompt_hash,
            self.dtype,
            int(self.hidden_width),
        )


@dataclass(frozen=True)
class HiddenShard:
    node_ids: torch.Tensor
    hidden: torch.Tensor
    manifest: HiddenManifest
    path: Path


def semantic_parity_is_valid(
    max_probability_error: float,
    argmax_match: bool,
) -> bool:
    return float(max_probability_error) <= 0.05 and bool(argmax_match)


def validate_hidden_shard(
    path: str | Path,
    expected: HiddenManifest | None = None,
) -> HiddenShard:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"hidden shard must be a dict: {path}")
    assert_label_free_cache(payload)
    node_ids = torch.as_tensor(payload["node_ids"], dtype=torch.long).view(-1).cpu()
    hidden = torch.as_tensor(payload["hidden"]).cpu().contiguous()
    manifest = HiddenManifest.from_dict(dict(payload["metadata"]))

    if node_ids.numel() == 0:
        raise ValueError(f"empty hidden shard: {path}")
    if node_ids.numel() != torch.unique(node_ids).numel():
        raise ValueError(f"duplicate node IDs in hidden shard: {path}")
    if hidden.ndim != 2 or hidden.shape[0] != node_ids.numel():
        raise ValueError(f"hidden rows do not align with IDs: {path}")
    if hidden.shape[1] != int(manifest.hidden_width):
        raise ValueError(f"hidden width does not match metadata: {path}")
    if not bool(torch.isfinite(hidden).all().item()):
        raise ValueError(f"non-finite hidden values: {path}")
    if manifest.row_order_hash != HiddenManifest.hash_ids(node_ids):
        raise ValueError(f"row-order hash mismatch: {path}")
    if int(manifest.start) != int(node_ids.min().item()):
        raise ValueError(f"manifest start does not match node IDs: {path}")
    if int(manifest.end) != int(node_ids.max().item()) + 1:
        raise ValueError(f"manifest end does not match node IDs: {path}")
    actual_dtype = str(hidden.dtype).replace("torch.", "")
    if actual_dtype != manifest.dtype:
        raise ValueError(f"dtype does not match metadata: {actual_dtype} != {manifest.dtype}")
    if not semantic_parity_is_valid(
        manifest.parity_max_probability_error, manifest.parity_argmax_match
    ):
        raise ValueError(f"hidden shard semantic parity failed: {path}")
    if expected is not None and manifest.stable_signature() != expected.stable_signature():
        raise ValueError(f"hidden shard provenance mismatch: {path}")
    return HiddenShard(node_ids=node_ids, hidden=hidden, manifest=manifest, path=path)


def atomic_torch_save(
    payload: Any,
    path: str | Path,
    *,
    validator: Callable[[Path], Any] | None = None,
) -> Any:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    serialized = buffer.getvalue()
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            written = handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        if written != len(serialized) or temporary.stat().st_size != len(serialized):
            raise OSError(f"incomplete torch artifact write: {temporary}")
        if validator is None:
            torch.load(temporary, map_location="cpu", weights_only=False)
        else:
            validator(temporary)
        os.replace(temporary, path)
        if validator is None:
            torch.load(path, map_location="cpu", weights_only=False)
            return path
        return validator(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_validated_hidden_shard(
    payload: dict[str, Any],
    path: str | Path,
) -> HiddenShard:
    result = atomic_torch_save(payload, path, validator=validate_hidden_shard)
    if not isinstance(result, HiddenShard):
        raise AssertionError("hidden validator returned an unexpected value")
    return result




def merge_hidden_shards(
    paths: list[str | Path],
    expected_ids: torch.Tensor,
) -> torch.Tensor:
    if not paths:
        raise ValueError("no hidden shards supplied")
    shards = [validate_hidden_shard(path) for path in paths]
    reference = shards[0].manifest.stable_signature()
    if any(shard.manifest.stable_signature() != reference for shard in shards[1:]):
        raise ValueError("hidden shard provenance mismatch")
    ids = torch.cat([shard.node_ids for shard in shards])
    hidden = torch.cat([shard.hidden for shard in shards])
    order = torch.argsort(ids, stable=True)
    ids = ids[order]
    hidden = hidden[order]
    expected = torch.as_tensor(expected_ids, dtype=torch.long).view(-1).cpu()
    if ids.numel() != torch.unique(ids).numel():
        raise ValueError("hidden shard coverage has duplicate node IDs")
    if not torch.equal(ids, expected):
        raise ValueError("hidden shard coverage does not exactly match expected IDs")
    return hidden.contiguous()


def provenance_hash(metadata: dict[str, Any]) -> str:
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "atomic_torch_save",
    "HiddenManifest",
    "HiddenShard",
    "merge_hidden_shards",
    "provenance_hash",
    "semantic_parity_is_valid",
    "write_validated_hidden_shard",
    "validate_hidden_shard",
]
