"""Atomic, non-overwriting artifact helpers for split Scheme-C runs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(value, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(value: str, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(str(value), encoding="utf-8")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_jsonl(rows: list[dict], path: str | Path) -> None:
    lines = [json.dumps(row, sort_keys=True, allow_nan=False) for row in rows]
    atomic_write_text("\n".join(lines) + ("\n" if lines else ""), path)


def prepare_output_root(path: str | Path, *, resume: bool) -> Path:
    root = Path(path)
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(f"refusing to overwrite non-empty output root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _artifact_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json" and ".tmp-" not in path.name
    )


def build_manifest(root: str | Path) -> dict:
    base = Path(root)
    files = {
        path.relative_to(base).as_posix(): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in _artifact_files(base)
    }
    return {"schema_version": 1, "file_count": len(files), "files": files}


def verify_manifest(root: str | Path) -> dict:
    base = Path(root)
    path = base / "manifest.json"
    if not path.exists():
        return {
            "all_checks_pass": False,
            "missing_manifest": True,
            "missing": [],
            "unexpected": [],
            "mismatched": [],
        }
    expected = json.loads(path.read_text(encoding="utf-8"))["files"]
    actual = build_manifest(base)["files"]
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatched = sorted(
        name for name in set(expected) & set(actual) if expected[name] != actual[name]
    )
    return {
        "all_checks_pass": not missing and not unexpected and not mismatched,
        "missing_manifest": False,
        "missing": missing,
        "unexpected": unexpected,
        "mismatched": mismatched,
        "checked": len(expected),
    }


__all__ = [
    "atomic_write_json",
    "atomic_write_jsonl",
    "atomic_write_text",
    "build_manifest",
    "prepare_output_root",
    "sha256_file",
    "verify_manifest",
]
