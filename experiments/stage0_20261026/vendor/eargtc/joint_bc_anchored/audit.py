from __future__ import annotations

from collections.abc import Iterable

import torch

from .artifacts import FORBIDDEN_KEYS


class AuditError(RuntimeError):
    pass


def scan_forbidden_training_content(value: object, location: str = "root") -> None:
    if isinstance(value, dict):
        if value.get("test_labels_available_to_process") not in (None, False):
            raise AuditError(f"test labels available to training process at {location}")
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_KEYS:
                raise AuditError(f"forbidden training key {key} at {location}")
            scan_forbidden_training_content(child, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            scan_forbidden_training_content(child, f"{location}[{index}]")


def _unique_set(record: dict, name: str) -> set[int]:
    values = [int(value) for value in record[name]]
    if len(values) != len(set(values)):
        raise AuditError(f"duplicate node ids in {name}")
    return set(values)


def audit_split_record(record: dict) -> dict:
    required = {
        "gold_support_ids", "official_validation_ids",
        "pseudo_training_candidate_ids", "heldout_evaluation_ids",
    }
    if not required.issubset(record):
        raise AuditError(f"split fields missing: {sorted(required - set(record))}")
    gold = _unique_set(record, "gold_support_ids")
    validation = _unique_set(record, "official_validation_ids")
    candidates = _unique_set(record, "pseudo_training_candidate_ids")
    heldout = _unique_set(record, "heldout_evaluation_ids")
    pairs = {
        "gold-validation": gold & validation,
        "gold-heldout": gold & heldout,
        "validation-candidates": validation & candidates,
        "validation-heldout": validation & heldout,
        "candidates-heldout": candidates & heldout,
        "gold-candidates": gold & candidates,
    }
    collisions = {name: sorted(values)[:10] for name, values in pairs.items() if values}
    if collisions:
        raise AuditError(f"split overlap detected: {collisions}")
    return {
        "gold_count": len(gold), "validation_count": len(validation),
        "pseudo_candidate_count": len(candidates), "heldout_count": len(heldout),
        "all_disjoint": True,
    }


def audit_step0_checkpoint(
    checkpoint: dict,
    *,
    effective_output: str = "b_anchored_residual",
) -> dict:
    if int(checkpoint.get("step", 0)) != 0:
        raise AuditError("step0 checkpoint has non-zero step")
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise AuditError("step0 checkpoint has no trainable model state")
    gate = {name: value for name, value in state.items() if "correction_gate." in name}
    if not gate or not {name.rsplit(".", 1)[-1] for name in gate}.issuperset({"weight", "bias"}):
        raise AuditError("step0 checkpoint lacks correction gate weight/bias")
    nonzero = [name for name, value in gate.items() if not torch.equal(torch.as_tensor(value), torch.zeros_like(torch.as_tensor(value)))]
    if nonzero:
        raise AuditError(f"step0 correction gate is not zero: {nonzero}")
    if effective_output not in {"b_anchored_residual", "absolute_graph_text"}:
        raise AuditError(f"unknown effective step0 output: {effective_output}")
    anchored = effective_output == "b_anchored_residual"
    return {
        "gate_tensor_count": len(gate),
        "effective_step0": (
            "scheme_b_by_zero_gate"
            if anchored else "absolute_graph_text_ablation"
        ),
        "step0_equals_b_by_zero_gate": anchored,
    }


def stitch_crossfit_predictions(folds: Iterable[dict], expected_node_ids: torch.Tensor) -> dict:
    rows = list(folds)
    if not rows:
        raise AuditError("no crossfit folds to stitch")
    ids = torch.cat([torch.as_tensor(row["node_ids"]).long().view(-1) for row in rows])
    if ids.unique().numel() != ids.numel():
        raise AuditError("duplicate node ids across crossfit folds")
    expected = torch.as_tensor(expected_node_ids).long().view(-1)
    if set(ids.tolist()) != set(expected.tolist()):
        missing = sorted(set(expected.tolist()) - set(ids.tolist()))[:10]
        extra = sorted(set(ids.tolist()) - set(expected.tolist()))[:10]
        raise AuditError(f"crossfit coverage mismatch: missing={missing}, extra={extra}")
    order = torch.argsort(ids)
    result: dict[str, object] = {"node_ids": ids[order]}
    shared = set.intersection(*(set(row) for row in rows)) - {"node_ids"}
    for key in sorted(shared):
        values = [row[key] for row in rows]
        if all(isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == row["node_ids"].numel() for value, row in zip(values, rows)):
            result[key] = torch.cat(values)[order]
        elif all(value == values[0] for value in values):
            result[key] = values[0]
        else:
            raise AuditError(f"cannot stitch inconsistent field {key}")
    return result
