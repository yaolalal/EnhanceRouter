from __future__ import annotations

import math
from collections.abc import Sequence

import torch


class ArtifactSchemaError(ValueError):
    pass


LABEL_FREE_KEYS = {
    "node_id",
    "agreement",
    "gnn_pred",
    "llm_pred",
    "scheme_b_preference_score",
    "scheme_b_route",
    "scheme_b_selected_pred",
    "scheme_c_factorized_trust",
    "scheme_c_direct_trust",
    "scheme_c_selected_trust",
    "scheme_c_checkpoint",
}


def sanitize_decision(source: dict) -> dict:
    route = str(source.get("route"))
    checkpoint = str(source.get("trust_checkpoint"))
    if route not in {"gnn", "llm"}:
        raise ArtifactSchemaError("Scheme B route must be gnn or llm")
    if checkpoint not in {"factorized", "direct"}:
        raise ArtifactSchemaError("Scheme C checkpoint must be factorized or direct")
    expected_pred = int(source["gnn_pred"] if route == "gnn" else source["llm_pred"])
    if int(source["selected_pred"]) != expected_pred:
        raise ArtifactSchemaError("Scheme B selected prediction disagrees with selected expert")
    expected_trust = float(
        source["factorized_trust"] if checkpoint == "factorized" else source["direct_trust"]
    )
    if not math.isclose(float(source["selected_trust"]), expected_trust, abs_tol=1e-6):
        raise ArtifactSchemaError("Scheme C selected trust disagrees with selected checkpoint")
    row = {
        "node_id": int(source["node_id"]),
        "agreement": bool(source["agreement"]),
        "gnn_pred": int(source["gnn_pred"]),
        "llm_pred": int(source["llm_pred"]),
        "scheme_b_preference_score": float(source["preference_score"]),
        "scheme_b_route": route,
        "scheme_b_selected_pred": int(source["selected_pred"]),
        "scheme_c_factorized_trust": float(source["factorized_trust"]),
        "scheme_c_direct_trust": float(source["direct_trust"]),
        "scheme_c_selected_trust": float(source["selected_trust"]),
        "scheme_c_checkpoint": checkpoint,
    }
    for key, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ArtifactSchemaError(f"non-finite label-free value: {key}")
    return row


def validate_label_free_rows(rows: Sequence[dict], expected_count: int) -> None:
    if len(rows) != int(expected_count):
        raise ArtifactSchemaError("label-free rows do not match expected node count")
    for index, row in enumerate(rows):
        if set(row) != LABEL_FREE_KEYS:
            raise ArtifactSchemaError("label-free row schema mismatch")
        if int(row["node_id"]) != index:
            raise ArtifactSchemaError("node IDs must be complete and ordered")


def _ids(value: torch.Tensor, name: str, node_count: int) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.long).view(-1).cpu()
    if result.numel() != torch.unique(result).numel():
        raise ArtifactSchemaError(f"{name} contains duplicate node IDs")
    if result.numel() and (int(result.min()) < 0 or int(result.max()) >= node_count):
        raise ArtifactSchemaError(f"{name} contains out-of-range node IDs")
    return result


def build_safe_supervision(
    *,
    labels: torch.Tensor,
    support_ids: torch.Tensor,
    validation_ids: torch.Tensor,
    query_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    target = torch.as_tensor(labels, dtype=torch.long).view(-1).cpu()
    support = _ids(support_ids, "support", target.numel())
    validation = _ids(validation_ids, "validation", target.numel())
    query = _ids(query_ids, "query", target.numel())
    if set(support.tolist()) & set(validation.tolist()):
        raise ArtifactSchemaError("support and validation overlap")
    supervised = set(support.tolist()) | set(validation.tolist())
    if supervised & set(query.tolist()):
        raise ArtifactSchemaError("query IDs overlap exported supervision")
    return {
        "gold_support_ids": support,
        "gold_support_targets": target[support],
        "official_validation_ids": validation,
        "official_validation_targets": target[validation],
    }
