from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


DATASETS: tuple[str, ...] = (
    "cora",
    "citeseer",
    "pubmed",
    "arxiv",
    "ogbn-products",
)
VERSION_LABEL = "legacy_pre_structural_oof_fix_20260719"
BC_ROOT_NAME = "router_bc_improved_final_20260719_v1"
UPSTREAM_ROOT_NAME = "router_v5_no_calibration_reliability_only_fast_20260714"
EXPECTED_BC_HASHES: dict[str, str] = {
    "config.json": "797a6873a3b0706f87c3670c46eb8c94749cf45b613ef9068f855f5065c40f1d",
    "summary.json": "00ec4a138a80093be44477169541c918e009e9d0005cb1f15d4a809799ad3660",
    "audit.json": "511f63bec27f67810c86c666f3f30e99dfc912e91aaa319e3afcfd7ede1cf0cd",
    "cora/metrics.json": "c00766617841746abefa478dbf964676cc3e6bdef1dc3330ffa71fb7acdea8c9",
    "cora/decisions.jsonl": "81af090bbc467b4ad0f294d231aa4aa0526043991aac3eb03a60768eccd55b87",
    "citeseer/metrics.json": "152022311ca331fcaa9949f1af1c815cf57ece49764448fd263a029988f0a3d3",
    "citeseer/decisions.jsonl": "f7a46c98f83917d232867727debf9e5c8961676c81a610e3696352f3e97e8188",
    "pubmed/metrics.json": "eedd3718084a3cf871b367487e2929c47ffc1db4cad74c0a2c51419eba1ba877",
    "pubmed/decisions.jsonl": "df6aecc9e56c3cdae92341c9b3f287a2307699bc3300d322385fd4a733b59501",
    "arxiv/metrics.json": "e954e5efbbd30271b7151fec9ffec6b6d41c1b161282a331b5ac5bf160549067",
    "arxiv/decisions.jsonl": "8eca2432f60998484a3e11b6ab21c8082e0949345941d4d030dd547d6ca98610",
    "ogbn-products/metrics.json": "cee3ef5720eca68c614f927e20909ae20499234f5fc0918ca876c1384beec75c",
    "ogbn-products/decisions.jsonl": "0d1731a362ce575e7981edf6c2cfdcab856bd5481f15936a1cdfbf26fe3999a9",
}
REQUIRED_UPSTREAM_RELPATHS: tuple[str, ...] = (
    "predictions.pt",
    "config.json",
    "codebook.pt",
    "ssl_gnn_encoder.pt",
    "gnn_head.pt",
    "splits/{dataset}_5shot_seed42.json",
    "router/oof/router_train.jsonl",
)
_FORBIDDEN_PARTS = {
    "router_v5_structural_oof_fixed_20260729_v1",
    "router_bc_improved_structural_oof_fixed_20260729_v1",
    "router_bc_structural_fix",
}


class ProvenanceError(RuntimeError):
    """Raised when a frozen-input provenance check fails."""


@dataclass(frozen=True)
class FrozenRouterSource:
    bc_root: Path
    upstream_root: Path
    version_label: str


def reject_forbidden_source(path: Path) -> None:
    normalized_parts = {part.lower() for part in path.resolve(strict=False).parts}
    matched = sorted(normalized_parts.intersection(_FORBIDDEN_PARTS))
    if matched:
        raise ProvenanceError(
            f"forbidden structural-fix source path: {path} (matched {matched[0]})"
        )


def verify_frozen_source(source: FrozenRouterSource) -> dict:
    reject_forbidden_source(source.bc_root)
    reject_forbidden_source(source.upstream_root)
    if source.bc_root.name != BC_ROOT_NAME:
        raise ProvenanceError(
            f"unexpected Scheme B/C root {source.bc_root.name!r}; required {BC_ROOT_NAME!r}"
        )
    if source.upstream_root.name != UPSTREAM_ROOT_NAME:
        raise ProvenanceError(
            f"unexpected upstream root {source.upstream_root.name!r}; "
            f"required {UPSTREAM_ROOT_NAME!r}"
        )
    if source.version_label != VERSION_LABEL:
        raise ProvenanceError(
            f"unexpected version label {source.version_label!r}; required {VERSION_LABEL!r}"
        )
    if not source.bc_root.is_dir():
        raise ProvenanceError(f"Scheme B/C root is missing: {source.bc_root}")
    if not source.upstream_root.is_dir():
        raise ProvenanceError(f"upstream root is missing: {source.upstream_root}")

    bc_files: dict[str, dict[str, object]] = {}
    for relative, expected_hash in EXPECTED_BC_HASHES.items():
        record = _file_record(source.bc_root / relative)
        if record["sha256"] != expected_hash:
            raise ProvenanceError(
                f"SHA256 mismatch for {relative}: expected {expected_hash}, "
                f"got {record['sha256']}"
            )
        record["expected_sha256"] = expected_hash
        bc_files[relative] = record

    try:
        bc_config = json.loads((source.bc_root / "config.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ProvenanceError(f"cannot parse frozen B/C config.json: {exc}") from exc
    configured_upstream = bc_config.get("input_run_root")
    if not isinstance(configured_upstream, str):
        raise ProvenanceError("frozen B/C config input_run_root is missing or not a string")
    if Path(configured_upstream).resolve() != source.upstream_root.resolve():
        raise ProvenanceError(
            "frozen B/C config input_run_root does not match the requested upstream root: "
            f"{configured_upstream!r} != {str(source.upstream_root.resolve())!r}"
        )

    upstream_files: dict[str, list[dict[str, object]]] = {}
    for dataset in DATASETS:
        pipeline_root = source.upstream_root / dataset / "pipeline"
        records = []
        for relative_template in REQUIRED_UPSTREAM_RELPATHS:
            relative = relative_template.format(dataset=dataset)
            record = _file_record(pipeline_root / relative)
            record["relative_path"] = relative
            records.append(record)
        upstream_files[dataset] = records

    return {
        "schema_version": 1,
        "version_label": source.version_label,
        "scheme_bc_source": BC_ROOT_NAME,
        "upstream_source": UPSTREAM_ROOT_NAME,
        "bc_root": str(source.bc_root.resolve()),
        "upstream_root": str(source.upstream_root.resolve()),
        "datasets": list(DATASETS),
        "bc_files": bc_files,
        "upstream_files": upstream_files,
        "all_checks_pass": True,
    }


def write_manifest(source: FrozenRouterSource, output: Path) -> Path:
    manifest = verify_frozen_source(source)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"manifest already exists: {output}")

    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.tmp-",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise FileExistsError(f"manifest already exists: {output}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ProvenanceError(f"required frozen file is missing: {path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
