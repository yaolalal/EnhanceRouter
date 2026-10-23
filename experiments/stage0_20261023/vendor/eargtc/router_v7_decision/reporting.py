from __future__ import annotations

import numpy as np
import torch


def _primary(routes: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    gnn = targets == 1
    llm = targets == 0
    return {
        "decisive_accuracy": float((routes == gnn).mean()),
        "gnn_only_recall": float(routes[gnn].mean()),
        "llm_only_recall": float((~routes[llm]).mean()),
    }


def method_summary_row(
    dataset: str,
    method: str,
    record: dict,
    *,
    selection: dict | None = None,
    artifact_hashes: dict | None = None,
) -> dict:
    primary = dict(record["accuracy_operating_point"])
    balanced = dict(record["balanced_operating_point"])
    validation = dict(record.get("validation", {}))
    counts = dict(primary["counts"])
    total = int(counts["total"])
    if total <= 0:
        raise ValueError("locked decisive result must be nonempty")
    validation_accuracy = validation.get("decisive_accuracy")
    selection = selection or {}
    return {
        "dataset": dataset,
        "method": method,
        "gnn_only_recall": primary["gnn_only_recall"],
        "llm_only_recall": primary["llm_only_recall"],
        "decisive_accuracy": primary["decisive_accuracy"],
        "auroc": primary["auroc"],
        "average_precision": primary["average_precision"],
        "nll": primary["nll"],
        "gnn_only_count": counts["gnn_only"],
        "llm_only_count": counts["llm_only"],
        "validation_accuracy": validation_accuracy,
        "validation_to_test_gap": (
            None
            if validation_accuracy is None
            else float(primary["decisive_accuracy"] - validation_accuracy)
        ),
        "selected": record["selected_name"],
        "feature_key": record["feature_key"],
        "primary_threshold": primary["threshold"],
        "balanced_threshold": balanced["threshold"],
        "validation_metrics": validation,
        "balanced_operating_point": balanced,
        "bootstrap_95": primary.get("bootstrap_95", {}),
        "counts": counts,
        "route_gnn_fraction": float(counts["route_gnn"] / total),
        "selected_spec": dict(selection.get("selected_spec", {})),
        "adapter": dict(selection.get("adapter", {})),
        "artifact_hashes": dict(artifact_hashes or {}),
    }


def paired_bootstrap_difference(
    first_routes: torch.Tensor,
    second_routes: torch.Tensor,
    targets: torch.Tensor,
    *,
    repetitions: int = 2000,
    seed: int = 42,
) -> dict[str, dict[str, float | list[float]]]:
    first = np.asarray(torch.as_tensor(first_routes, dtype=torch.bool).view(-1).cpu())
    second = np.asarray(torch.as_tensor(second_routes, dtype=torch.bool).view(-1).cpu())
    target = np.asarray(torch.as_tensor(targets, dtype=torch.long).view(-1).cpu())
    if first.size == 0 or first.shape != second.shape or first.shape != target.shape:
        raise ValueError("paired route arrays must be nonempty and aligned")
    if set(np.unique(target).tolist()) != {0, 1}:
        raise ValueError("paired bootstrap requires both decisive classes")
    point_first = _primary(first, target)
    point_second = _primary(second, target)
    keys = tuple(point_first)
    rng = np.random.default_rng(int(seed))
    groups = [np.flatnonzero(target == value) for value in (0, 1)]
    draws = {key: [] for key in keys}
    for _ in range(int(repetitions)):
        sample = np.concatenate(
            [rng.choice(group, size=group.size, replace=True) for group in groups]
        )
        one = _primary(first[sample], target[sample])
        two = _primary(second[sample], target[sample])
        for key in keys:
            draws[key].append(one[key] - two[key])
    return {
        key: {
            "difference": point_first[key] - point_second[key],
            "ci95": [
                float(np.quantile(draws[key], 0.025)),
                float(np.quantile(draws[key], 0.975)),
            ],
        }
        for key in keys
    }


__all__ = ["method_summary_row", "paired_bootstrap_difference"]
