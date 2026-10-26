from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

from .utils import iter_jsonl


def _records_from_json(path: Path):
    if path.suffix == ".jsonl":
        yield from iter_jsonl(path)
        return
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        for rec in obj:
            yield rec
    elif isinstance(obj, dict):
        if "records" in obj and isinstance(obj["records"], list):
            for rec in obj["records"]:
                yield rec
        else:
            for key, value in obj.items():
                if isinstance(value, dict):
                    rec = dict(value)
                    rec.setdefault("node_id", key)
                    yield rec


def _scatter_class_logits(
    values,
    node_ids,
    num_nodes: int,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.float32, device=device)
    if values.dim() != 2 or values.size(-1) != num_classes:
        raise ValueError(f"Expected class logits with shape (*, {num_classes}), got {tuple(values.shape)}")
    if node_ids is None:
        if values.size(0) != num_nodes:
            raise ValueError(f"Dense LLM logits must have {num_nodes} rows, got {values.size(0)}")
        return values
    node_ids = torch.as_tensor(node_ids, dtype=torch.long, device=device)
    if node_ids.numel() != values.size(0):
        raise ValueError("node_ids and logits row count do not match")
    logits = torch.zeros(num_nodes, num_classes, dtype=torch.float32, device=device)
    valid = (node_ids >= 0) & (node_ids < num_nodes)
    logits[node_ids[valid]] = values[valid]
    return logits


def load_llm_logits(
    path: str | None,
    num_nodes: int,
    num_classes: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Load class-level LLM logits shaped as [num_nodes, num_classes].

    This loader intentionally does not reduce vocabulary-sized logits to class
    logits. For multi-token label names, that reduction must be done during LLM
    inference by scoring each complete label string and writing one score per
    class. JSON/JSONL records may use keys: logits, class_logits, scores.
    """
    if not path:
        return None
    file_path = Path(path)
    if file_path.suffix in (".pt", ".pth"):
        obj = torch.load(file_path, map_location=device, weights_only=False)
        if isinstance(obj, dict):
            values = obj.get("class_logits", obj.get("logits", obj.get("scores")))
            node_ids = obj.get("node_ids", obj.get("node_id"))
        else:
            values = obj
            node_ids = None
        if values is None:
            raise ValueError(f"No class logits found in {file_path}")
        return _scatter_class_logits(values, node_ids, num_nodes, num_classes, device)

    logits = torch.zeros(num_nodes, num_classes, device=device)
    seen = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    for offset, rec in enumerate(_records_from_json(file_path)):
        values = rec.get("class_logits", rec.get("logits", rec.get("scores")))
        if values is None:
            continue
        node_id = int(rec.get("node_id", rec.get("id", offset)))
        if not (0 <= node_id < num_nodes):
            continue
        row = torch.tensor(values, dtype=torch.float32, device=device)
        if row.numel() != num_classes:
            raise ValueError(
                f"Record for node {node_id} has {row.numel()} logits, expected {num_classes}. "
                "Provide class-level logits, not vocabulary logits."
            )
        logits[node_id] = row.view(num_classes)
        seen[node_id] = True
    return logits if seen.any() else None
