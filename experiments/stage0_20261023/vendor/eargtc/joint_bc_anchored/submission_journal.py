from __future__ import annotations

import fcntl
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _decode(lines: list[str], identity: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not lines:
        raise RuntimeError("submission journal is empty")
    header = json.loads(lines[0])
    if header != {"record_type": "header", "identity": identity}:
        raise RuntimeError("submission journal identity mismatch")
    rows: dict[str, dict[str, Any]] = {}
    for line in lines[1:]:
        record = json.loads(line)
        if record.get("record_type") != "submission":
            raise RuntimeError("invalid submission journal record")
        key = record.get("key"); row = record.get("row")
        if not isinstance(key, str) or not isinstance(row, dict):
            raise RuntimeError("invalid submission journal row")
        if key in rows and rows[key] != row:
            raise RuntimeError(f"conflicting submission journal row: {key}")
        rows[key] = row
    return rows


@dataclass
class SubmissionJournal:
    path: Path
    identity: dict[str, Any]
    rows: dict[str, dict[str, Any]]

    @classmethod
    def open(
        cls, path: Path, identity: dict[str, Any],
    ) -> "SubmissionJournal":
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            header = {"record_type": "header", "identity": identity}
            try:
                with path.open("x", encoding="utf-8") as handle:
                    handle.write(json.dumps(header, sort_keys=True, allow_nan=False) + "\n")
                    handle.flush(); os.fsync(handle.fileno())
            except FileExistsError:
                pass
        rows = _decode(path.read_text(encoding="utf-8").splitlines(), identity)
        return cls(path=path, identity=identity, rows=rows)

    def record(self, key: str, row: dict[str, Any]) -> None:
        if not isinstance(key, str) or not key or not isinstance(row, dict):
            raise ValueError("submission key and row are required")
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            current = _decode(handle.read().splitlines(), self.identity)
            if key in current:
                if current[key] != row:
                    raise RuntimeError(f"conflicting submission journal row: {key}")
                self.rows = current
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                return
            handle.seek(0, os.SEEK_END)
            record = {"record_type": "submission", "key": key, "row": row}
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush(); os.fsync(handle.fileno())
            current[key] = row; self.rows = current
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def finalize(self, receipt: Path, *, expected_count: int) -> dict[str, Any]:
        receipt = Path(receipt)
        if receipt.exists():
            raise FileExistsError(f"refusing submission receipt overwrite: {receipt}")
        current = _decode(
            self.path.read_text(encoding="utf-8").splitlines(), self.identity,
        )
        if len(current) != int(expected_count):
            raise RuntimeError(
                f"submission journal incomplete: {len(current)} != {expected_count}"
            )
        payload = {
            "schema_version": 1,
            "state": "complete",
            "identity": self.identity,
            "journal": str(self.path),
            "count": len(current),
            "submitted": sum(row.get("status") == "submitted" for row in current.values()),
            "runs": list(current.values()),
        }
        receipt.parent.mkdir(parents=True, exist_ok=True)
        temporary = receipt.with_name(f".{receipt.name}.tmp-{os.getpid()}")
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        try:
            os.link(temporary, receipt)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing submission receipt overwrite: {receipt}"
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)
        self.rows = current
        return payload
