from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


DATASETS = ("cora", "citeseer", "pubmed", "arxiv", "ogbn-products")
BC_ROOT_NAME = "router_bc_improved_structural_oof_fixed_20260729_v1"
UPSTREAM_ROOT_NAME = "router_v5_structural_oof_fixed_20260729_v1"
NUMBERED_ROOT_NAME = "llm_numbered_token_logits_final_20260706"
VERSION_LABEL = "structural_oof_fixed_direct_numbered_20260813"


class ProvenanceError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenRouterSource:
    bc_root: Path
    upstream_root: Path
    numbered_root: Path
    version_label: str = VERSION_LABEL


def reject_forbidden_source(path: Path) -> None:
    parts = {part.lower() for part in Path(path).resolve(strict=False).parts}
    forbidden = {
        "router_bc_improved_final_20260719_v1",
        "router_v5_no_calibration_reliability_only_fast_20260714",
        "joint_bc_anchored",
    }
    match = sorted(parts & forbidden)
    if match:
        raise ProvenanceError(f"legacy/anchored source is forbidden: {match[0]}")


def verify_frozen_source(source: FrozenRouterSource) -> dict[str, object]:
    if source.version_label != VERSION_LABEL:
        raise ProvenanceError(f"unexpected version label: {source.version_label}")
    return verify_sources(source.bc_root, source.upstream_root, source.numbered_root)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, object]:
    value = Path(path)
    if not value.is_file():
        raise ProvenanceError(f"required input is missing: {value}")
    return {
        "path": str(value.resolve()),
        "size_bytes": value.stat().st_size,
        "sha256": sha256_file(value),
    }


def _json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"cannot parse required JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProvenanceError(f"required JSON is not an object: {path}")
    return value


def _require_root(path: Path, expected: str, role: str) -> Path:
    value = Path(path).resolve()
    if value.name != expected:
        raise ProvenanceError(
            f"{role} must use the structurally corrected/frozen root {expected!r}, "
            f"got {value.name!r}"
        )
    if not value.is_dir():
        raise ProvenanceError(f"{role} root is missing: {value}")
    return value


def verify_sources(
    bc_root: Path,
    upstream_root: Path,
    numbered_root: Path,
) -> dict[str, object]:
    bc = _require_root(bc_root, BC_ROOT_NAME, "Scheme B/C")
    upstream = _require_root(upstream_root, UPSTREAM_ROOT_NAME, "upstream")
    numbered = _require_root(numbered_root, NUMBERED_ROOT_NAME, "numbered prompt")

    bc_config = _json(bc / "config.json")
    bc_audit = _json(bc / "audit.json")
    upstream_audit = _json(upstream / "corrected_input_audit.json")
    numbered_manifest = _json(numbered / "manifest.json")
    if bc_audit.get("all_checks_pass") is not True:
        raise ProvenanceError("corrected Scheme B/C audit did not pass")
    if upstream_audit.get("all_checks_pass") is not True:
        raise ProvenanceError("corrected upstream audit did not pass")
    configured = Path(str(bc_config.get("input_run_root", "")))
    if not configured.is_absolute():
        configured = (Path.cwd() / configured).resolve()
    if configured != upstream:
        raise ProvenanceError("Scheme B/C config input_run_root does not match corrected upstream")

    dataset_checks: dict[str, dict[str, object]] = {}
    for dataset in DATASETS:
        pipeline = upstream / dataset / "pipeline"
        split = pipeline / "splits" / f"{dataset}_5shot_seed42.json"
        required = {
            "bc_decisions": bc / dataset / "decisions.jsonl",
            "bc_metrics": bc / dataset / "metrics.json",
            "upstream_predictions": pipeline / "predictions.pt",
            "upstream_config": pipeline / "config.json",
            "upstream_split": split,
            "upstream_dataset_manifest": upstream / dataset / "dataset_manifest.json",
        }
        complete = _json(upstream / dataset / "complete.json")
        if complete.get("all_checks_pass") is not True:
            raise ProvenanceError(f"corrected upstream dataset is incomplete: {dataset}")
        dataset_checks[dataset] = {
            "complete": True,
            "files": {name: _record(path) for name, path in required.items()},
        }

    return {
        "schema_version": 1,
        "version_label": VERSION_LABEL,
        "bc_root": str(bc),
        "upstream_root": str(upstream),
        "numbered_root": str(numbered),
        "datasets": list(DATASETS),
        "bc_config_input_matches_upstream": True,
        "root_files": {
            "bc_config": _record(bc / "config.json"),
            "bc_audit": _record(bc / "audit.json"),
            "upstream_audit": _record(upstream / "corrected_input_audit.json"),
            "numbered_manifest": _record(numbered / "manifest.json"),
        },
        "numbered_manifest_purpose": numbered_manifest.get("purpose"),
        "dataset_checks": dataset_checks,
        "all_checks_pass": True,
    }

