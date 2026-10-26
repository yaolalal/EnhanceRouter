from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(value, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@contextmanager
def atomic_output(destination: str | Path) -> Iterator[Path]:
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing existing Router-v7 output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.stage-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        yield stage
        if destination.exists():
            raise FileExistsError(f"Router-v7 destination appeared during staging: {destination}")
        os.replace(stage, destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def build_manifest(root: str | Path) -> dict:
    root = Path(root)
    files = []
    for path in sorted(value for value in root.rglob("*") if value.is_file() and value.name != "manifest.json"):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size": int(path.stat().st_size),
            }
        )
    return {"schema": "router_v7_manifest_v1", "file_count": len(files), "files": files}


def verify_manifest(root: str | Path, manifest: dict | None = None) -> dict:
    root = Path(root)
    if manifest is None:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    checks = []
    for record in manifest["files"]:
        path = root / record["path"]
        exists = path.is_file()
        digest = sha256_file(path) if exists else None
        checks.append(
            {
                "path": record["path"],
                "exists": exists,
                "sha256_matches": digest == record["sha256"],
                "size_matches": exists and path.stat().st_size == int(record["size"]),
            }
        )
    return {
        "checked": len(checks),
        "checks": checks,
        "all_checks_pass": all(
            row["exists"] and row["sha256_matches"] and row["size_matches"] for row in checks
        ) and len(checks) == int(manifest["file_count"]),
    }


def _resolve_project_file(project_root: Path, value: str | Path) -> tuple[Path, str]:
    root = project_root.resolve()
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"provenance path escapes project root: {value}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"missing provenance file: {relative}")
    return resolved, relative


def build_provenance(project_root: str | Path, paths) -> dict:
    """Hash an explicit, fail-closed set of files rooted in the project.

    Unlike build_manifest, this function may reference files spread across
    source, test, documentation, and immutable run directories.
    """

    root = Path(project_root)
    records = []
    seen = set()
    for value in paths:
        resolved, relative = _resolve_project_file(root, value)
        if relative in seen:
            raise ValueError(f"duplicate provenance path: {relative}")
        seen.add(relative)
        records.append(
            {
                "path": relative,
                "sha256": sha256_file(resolved),
                "size": int(resolved.stat().st_size),
            }
        )
    records.sort(key=lambda row: row["path"])
    if not records:
        raise ValueError("release provenance must contain at least one file")
    return {
        "schema": "router_v7_release_provenance_v1",
        "project_root_name": root.resolve().name,
        "file_count": len(records),
        "files": records,
    }


def verify_provenance(project_root: str | Path, provenance: dict) -> dict:
    root = Path(project_root)
    checks = []
    for record in provenance["files"]:
        try:
            path, relative = _resolve_project_file(root, record["path"])
        except (FileNotFoundError, ValueError):
            path, relative = None, str(record["path"])
        exists = path is not None
        checks.append(
            {
                "path": relative,
                "exists": exists,
                "sha256_matches": exists and sha256_file(path) == record["sha256"],
                "size_matches": exists and path.stat().st_size == int(record["size"]),
            }
        )
    return {
        "checked": len(checks),
        "checks": checks,
        "all_checks_pass": len(checks) == int(provenance["file_count"]) and all(
            row["exists"] and row["sha256_matches"] and row["size_matches"] for row in checks
        ),
    }
