from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence

import numpy as np


STATE_NAMES = (
    "correct_agreement",
    "wrong_agreement",
    "gnn_only",
    "llm_only",
    "conflict_neither",
)
SEGMENT_BOUNDS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
SEGMENT_LABELS = ("0–10%", "10–25%", "25–50%", "50–75%", "75–100%")


def _arrays(rows: Sequence[Mapping]) -> dict[str, np.ndarray]:
    required = {
        "node_id", "true_label", "gnn_pred", "llm_pred", "selected_pred",
        "selected_trust", "route", "preference_score",
    }
    if not rows:
        raise ValueError("empty_decision_rows")
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(f"missing_decision_fields:{','.join(missing)}")
    node_id = np.asarray([int(row["node_id"]) for row in rows], dtype=np.int64)
    if np.unique(node_id).size != node_id.size:
        raise ValueError("duplicate_node_id")
    trust = np.asarray([float(row["selected_trust"]) for row in rows], dtype=float)
    preference = np.asarray([float(row["preference_score"]) for row in rows], dtype=float)
    if np.any(~np.isfinite(trust)) or np.any((trust < 0) | (trust > 1)):
        raise ValueError("selected_trust_out_of_range")
    if np.any(~np.isfinite(preference)) or np.any((preference < 0) | (preference > 1)):
        raise ValueError("preference_score_out_of_range")
    return {
        "node_id": node_id,
        "true": np.asarray([int(row["true_label"]) for row in rows], dtype=int),
        "gnn": np.asarray([int(row["gnn_pred"]) for row in rows], dtype=int),
        "llm": np.asarray([int(row["llm_pred"]) for row in rows], dtype=int),
        "selected": np.asarray([int(row["selected_pred"]) for row in rows], dtype=int),
        "trust": trust,
        "preference": preference,
        "route_gnn": np.asarray([row["route"] == "gnn" for row in rows], dtype=bool),
    }


def classify_states(rows: Sequence[Mapping]) -> OrderedDict[str, np.ndarray]:
    a = _arrays(rows)
    g_correct = a["gnn"] == a["true"]
    l_correct = a["llm"] == a["true"]
    agreement = a["gnn"] == a["llm"]
    masks = OrderedDict(
        [
            ("correct_agreement", agreement & g_correct),
            ("wrong_agreement", agreement & ~g_correct),
            ("gnn_only", ~agreement & g_correct & ~l_correct),
            ("llm_only", ~agreement & ~g_correct & l_correct),
            ("conflict_neither", ~agreement & ~g_correct & ~l_correct),
        ]
    )
    coverage = sum(mask.astype(np.int8) for mask in masks.values())
    if not np.all(coverage == 1):
        raise ValueError("five_state_partition_failure")
    return masks


def _ceil_endpoint(fraction: float, count: int) -> int:
    return min(count, int(math.ceil(float(fraction) * count)))


def rank_segments(
    rows: Sequence[Mapping],
    state_masks: Mapping[str, np.ndarray],
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    a = _arrays(rows)
    count = a["node_id"].size
    if tuple(state_masks) != STATE_NAMES:
        raise ValueError("unexpected_state_names")
    order = np.lexsort((a["node_id"], -a["trust"]))
    endpoints = [_ceil_endpoint(value, count) for value in SEGMENT_BOUNDS]
    assignment = np.empty(count, dtype=np.int8)
    result = []
    selected_correct = a["selected"] == a["true"]
    for index, label in enumerate(SEGMENT_LABELS):
        start, stop = endpoints[index], endpoints[index + 1]
        ranked = order[start:stop]
        assignment[ranked] = index
        size = int(ranked.size)
        if size <= 0:
            raise ValueError(f"empty_rank_segment:{label}")
        row = {
            "segment_index": index,
            "segment": label,
            "start_rank": start,
            "end_rank": stop,
            "count": size,
            "precision": float(np.mean(selected_correct[ranked])),
            "gnn_route_rate": float(np.mean(a["route_gnn"][ranked])),
            "trust_mean": float(np.mean(a["trust"][ranked])),
            "trust_median": float(np.median(a["trust"][ranked])),
            "trust_min": float(np.min(a["trust"][ranked])),
            "trust_max": float(np.max(a["trust"][ranked])),
        }
        for name in STATE_NAMES:
            state_mask = np.asarray(state_masks[name], dtype=bool)
            state_count = int(np.sum(state_mask[ranked]))
            row[f"{name}_count"] = state_count
            row[f"{name}_proportion"] = state_count / size
            row[f"{name}_selected_correct_count"] = int(
                np.sum(state_mask[ranked] & selected_correct[ranked])
            )
        if sum(row[f"{name}_count"] for name in STATE_NAMES) != size:
            raise ValueError(f"rank_segment_partition_failure:{label}")
        result.append(row)
    if np.unique(order).size != count or np.any(np.bincount(assignment, minlength=5) == 0):
        raise ValueError("rank_segment_coverage_failure")
    return result, order, assignment


def oracle_gap_decomposition(
    rows: Sequence[Mapping],
) -> tuple[dict, dict[str, np.ndarray]]:
    a = _arrays(rows)
    g_correct = a["gnn"] == a["true"]
    l_correct = a["llm"] == a["true"]
    selected_correct = a["selected"] == a["true"]
    g_accuracy = float(np.mean(g_correct))
    l_accuracy = float(np.mean(l_correct))
    comparator = "gnn" if g_accuracy >= l_accuracy else "llm"
    comparator_correct = g_correct if comparator == "gnn" else l_correct
    other_correct = l_correct if comparator == "gnn" else g_correct
    comparator_only = comparator_correct & ~other_correct
    other_only = other_correct & ~comparator_correct
    masks = {
        "loss": comparator_correct & ~selected_correct,
        "win": selected_correct & ~comparator_correct,
        "protected": comparator_only & selected_correct,
        "missed_recovery": other_only & ~selected_correct,
        "recovered_other_only": other_only & selected_correct,
    }
    losses = int(np.sum(masks["loss"]))
    wins = int(np.sum(masks["win"]))
    missed = int(np.sum(masks["missed_recovery"]))
    oracle_gap = int(np.sum((g_correct | l_correct) & ~selected_correct))
    summary = {
        "count": int(a["true"].size),
        "comparator": comparator,
        "gnn_accuracy": g_accuracy,
        "llm_accuracy": l_accuracy,
        "router_accuracy": float(np.mean(selected_correct)),
        "oracle_accuracy": float(np.mean(g_correct | l_correct)),
        "losses": losses,
        "wins": wins,
        "missed_recoveries": missed,
        "protected": int(np.sum(masks["protected"])),
        "oracle_gap_count": oracle_gap,
        "oracle_headroom_count": int(np.sum(other_only)),
        "router_gain_count": wins - losses,
    }
    if oracle_gap != losses + missed:
        raise ValueError("oracle_gap_identity_failure")
    if summary["oracle_headroom_count"] != wins + missed:
        raise ValueError("oracle_headroom_identity_failure")
    return summary, masks


def rank_biserial(left, right) -> float | None:
    x = np.asarray(left, dtype=float).reshape(-1)
    y = np.asarray(right, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return None
    values = np.concatenate([x, y])
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    u = float(np.sum(ranks[: x.size]) - x.size * (x.size + 1) / 2)
    return 2.0 * u / (x.size * y.size) - 1.0
