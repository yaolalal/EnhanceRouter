"""Pure analysis functions for frozen Router B/C decisions.

This module never trains or selects a model.  It recomputes post-selection
metrics from saved node-level decisions and validates their internal
consistency.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np


EPS = 1e-6


def _as_1d(values: Iterable[float], dtype=float) -> np.ndarray:
    array = np.asarray(list(values), dtype=dtype)
    if array.ndim != 1:
        raise ValueError("expected_1d_array")
    return array


def _clip_probability(values: np.ndarray | Sequence[float]) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)


def _logit(values: np.ndarray | Sequence[float]) -> np.ndarray:
    clipped = _clip_probability(values)
    return np.log(clipped) - np.log1p(-clipped)


def centered_logit_margin(
    scores: np.ndarray | Sequence[float], threshold: float
) -> np.ndarray:
    """Return score logits centered on the frozen routing threshold."""

    return _logit(scores) - float(_logit([threshold])[0])


def tie_aware_auroc(
    labels: np.ndarray | Sequence[int], scores: np.ndarray | Sequence[float]
) -> float | None:
    """Compute Mann-Whitney AUROC with half credit for tied scores."""

    y = _as_1d(labels, dtype=int)
    s = _as_1d(scores, dtype=float)
    if y.size != s.size:
        raise ValueError("auroc_length_mismatch")
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(s.size, dtype=float)
    start = 0
    while start < s.size:
        stop = start + 1
        while stop < s.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        average_rank = 0.5 * ((start + 1) + stop)
        ranks[order[start:stop]] = average_rank
        start = stop
    rank_sum = float(np.sum(ranks[y == 1]))
    u_statistic = rank_sum - positives * (positives + 1) / 2.0
    return u_statistic / (positives * negatives)


def average_precision(
    labels: np.ndarray | Sequence[int], scores: np.ndarray | Sequence[float]
) -> float | None:
    """Compute grouped average precision so tied scores are order invariant."""

    y = _as_1d(labels, dtype=int)
    s = _as_1d(scores, dtype=float)
    if y.size != s.size:
        raise ValueError("average_precision_length_mismatch")
    positives = int(np.sum(y == 1))
    if positives == 0:
        return None

    order = np.argsort(-s, kind="mergesort")
    sorted_y = y[order]
    sorted_s = s[order]
    cumulative_positive = 0
    cumulative_count = 0
    result = 0.0
    start = 0
    while start < y.size:
        stop = start + 1
        while stop < y.size and sorted_s[stop] == sorted_s[start]:
            stop += 1
        group_positive = int(np.sum(sorted_y[start:stop] == 1))
        cumulative_positive += group_positive
        cumulative_count += stop - start
        precision = cumulative_positive / cumulative_count
        result += precision * group_positive / positives
        start = stop
    return result


def ks_distance(
    positive_scores: np.ndarray | Sequence[float],
    negative_scores: np.ndarray | Sequence[float],
) -> float | None:
    """Return the two-sample Kolmogorov-Smirnov distance."""

    positive = np.sort(_as_1d(positive_scores, dtype=float))
    negative = np.sort(_as_1d(negative_scores, dtype=float))
    if positive.size == 0 or negative.size == 0:
        return None
    values = np.unique(np.concatenate([positive, negative]))
    positive_cdf = np.searchsorted(positive, values, side="right") / positive.size
    negative_cdf = np.searchsorted(negative, values, side="right") / negative.size
    return float(np.max(np.abs(positive_cdf - negative_cdf)))


def wilson_interval(
    successes: int, count: int, z: float = 1.96
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    if count <= 0:
        raise ValueError("wilson_count_must_be_positive")
    if successes < 0 or successes > count:
        raise ValueError("wilson_successes_out_of_range")
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / count
            + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return center - radius, center + radius


def calibration_bins(
    labels: np.ndarray | Sequence[int],
    scores: np.ndarray | Sequence[float],
    n_bins: int = 10,
) -> list[dict]:
    """Build non-empty fixed-width reliability bins with Wilson intervals."""

    y = _as_1d(labels, dtype=int)
    s = _as_1d(scores, dtype=float)
    if y.size != s.size:
        raise ValueError("calibration_length_mismatch")
    if n_bins <= 0:
        raise ValueError("calibration_bins_must_be_positive")
    if np.any(~np.isfinite(s)) or np.any((s < 0.0) | (s > 1.0)):
        raise ValueError("calibration_score_out_of_range")

    assignments = np.minimum((s * n_bins).astype(int), n_bins - 1)
    result: list[dict] = []
    for index in range(n_bins):
        mask = assignments == index
        count = int(np.sum(mask))
        if count == 0:
            continue
        successes = int(np.sum(y[mask] == 1))
        lower, upper = wilson_interval(successes, count)
        result.append(
            {
                "bin": index,
                "lower": index / n_bins,
                "upper": (index + 1) / n_bins,
                "count": count,
                "mean_score": float(np.mean(s[mask])),
                "empirical_rate": successes / count,
                "wilson_lower": lower,
                "wilson_upper": upper,
            }
        )
    return result


def _calibration_intercept_slope(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[float | None, float | None]:
    if labels.size == 0 or np.unique(labels).size < 2:
        return None, None
    x = _logit(scores)
    design = np.column_stack([np.ones(x.size), x])
    coefficients = np.zeros(2, dtype=float)
    ridge = np.diag([1e-8, 1e-6])
    for _ in range(100):
        logits = np.clip(design @ coefficients, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (probabilities - labels) + ridge @ coefficients
        weights = np.maximum(probabilities * (1.0 - probabilities), 1e-10)
        hessian = design.T @ (design * weights[:, None]) + ridge
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        coefficients -= step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return float(coefficients[0]), float(coefficients[1])


def binary_metrics(
    labels: np.ndarray | Sequence[int],
    scores: np.ndarray | Sequence[float],
    n_bins: int = 10,
) -> dict:
    """Return threshold-free discrimination and calibration metrics."""

    y = _as_1d(labels, dtype=int)
    s = _as_1d(scores, dtype=float)
    if y.size != s.size:
        raise ValueError("binary_metrics_length_mismatch")
    if y.size == 0:
        raise ValueError("binary_metrics_empty")
    clipped = _clip_probability(s)
    bins = calibration_bins(y, s, n_bins=n_bins)
    ece = sum(
        item["count"] / y.size
        * abs(item["mean_score"] - item["empirical_rate"])
        for item in bins
    )
    intercept, slope = _calibration_intercept_slope(y.astype(float), s)
    return {
        "count": int(y.size),
        "positive_rate": float(np.mean(y)),
        "auroc": tie_aware_auroc(y, s),
        "average_precision": average_precision(y, s),
        "nll": float(
            -np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
        ),
        "brier": float(np.mean((s - y) ** 2)),
        "ece": float(ece),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def _required_threshold(metrics: Mapping) -> float:
    try:
        threshold = float(metrics["scheme_b"]["test"]["threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("missing_frozen_threshold") from error
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError("invalid_frozen_threshold")
    return threshold


def _saved_trust_threshold(metrics: Mapping) -> float | None:
    try:
        threshold = float(
            metrics["scheme_c"]["metrics"]["validation_threshold"]["threshold"]
        )
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError("invalid_frozen_trust_threshold")
    return threshold


def _validate_rows(rows: Sequence[Mapping], threshold: float, dataset: str) -> None:
    required = {
        "agreement",
        "correct",
        "gnn_pred",
        "llm_pred",
        "node_id",
        "preference_score",
        "route",
        "selected_pred",
        "selected_trust",
        "true_label",
    }
    seen: set[int] = set()
    for index, row in enumerate(rows):
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"{dataset}:missing_fields:{index}:{','.join(missing)}")
        node_id = int(row["node_id"])
        if node_id in seen:
            raise ValueError(f"{dataset}:duplicate_node_id:{node_id}")
        seen.add(node_id)
        score = float(row["preference_score"])
        trust = float(row["selected_trust"])
        if not math.isfinite(score) or score < 0.0 or score > 1.0:
            raise ValueError(f"{dataset}:preference_score_out_of_range:{node_id}")
        if not math.isfinite(trust) or trust < 0.0 or trust > 1.0:
            raise ValueError(f"{dataset}:trust_score_out_of_range:{node_id}")
        agreement = row["gnn_pred"] == row["llm_pred"]
        if bool(row["agreement"]) != agreement:
            raise ValueError(f"{dataset}:agreement_invariant:{node_id}")
        expected_route = (
            "gnn"
            if agreement or score >= threshold
            else "llm"
        )
        if row["route"] != expected_route:
            raise ValueError(f"{dataset}:frozen_threshold_route:{node_id}")
        expected_prediction = (
            row["gnn_pred"] if expected_route == "gnn" else row["llm_pred"]
        )
        if row["selected_pred"] != expected_prediction:
            raise ValueError(f"{dataset}:selected_prediction:{node_id}")
        if bool(row["correct"]) != (expected_prediction == row["true_label"]):
            raise ValueError(f"{dataset}:selected_correctness:{node_id}")


def _accuracy(values: np.ndarray) -> float:
    return float(np.mean(values))


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _trust_checkpoint(metrics: Mapping, rows: Sequence[Mapping]) -> str | None:
    try:
        return str(
            metrics["scheme_c"]["metrics"]["validation_selection"]["selected"]
        )
    except (KeyError, TypeError):
        values = {str(row.get("trust_checkpoint")) for row in rows}
        return values.pop() if len(values) == 1 else None


def analyze_dataset(rows: Sequence[Mapping], metrics: Mapping) -> dict:
    """Validate and analyze one frozen target-dataset decision file."""

    dataset = str(metrics.get("dataset", "unknown"))
    threshold = _required_threshold(metrics)
    trust_threshold = _saved_trust_threshold(metrics)
    if not rows:
        raise ValueError(f"{dataset}:empty_decisions")
    _validate_rows(rows, threshold, dataset)

    node_id = np.asarray([int(row["node_id"]) for row in rows], dtype=np.int64)
    true_label = np.asarray([int(row["true_label"]) for row in rows], dtype=int)
    gnn_pred = np.asarray([int(row["gnn_pred"]) for row in rows], dtype=int)
    llm_pred = np.asarray([int(row["llm_pred"]) for row in rows], dtype=int)
    selected_pred = np.asarray(
        [int(row["selected_pred"]) for row in rows], dtype=int
    )
    scores = np.asarray(
        [float(row["preference_score"]) for row in rows], dtype=float
    )
    trust = np.asarray([float(row["selected_trust"]) for row in rows], dtype=float)
    routes_gnn = np.asarray([row["route"] == "gnn" for row in rows], dtype=bool)

    gnn_correct = gnn_pred == true_label
    llm_correct = llm_pred == true_label
    selected_correct = selected_pred == true_label
    agreement = gnn_pred == llm_pred
    conflict = ~agreement
    both_correct = gnn_correct & llm_correct
    gnn_only = gnn_correct & ~llm_correct
    llm_only = ~gnn_correct & llm_correct
    both_wrong = ~gnn_correct & ~llm_correct
    decisive = gnn_only | llm_only

    if not np.all(decisive <= conflict):
        raise ValueError(f"{dataset}:decisive_must_be_conflict")
    if not np.all(
        both_correct.astype(int)
        + gnn_only.astype(int)
        + llm_only.astype(int)
        + both_wrong.astype(int)
        == 1
    ):
        raise ValueError(f"{dataset}:four_state_partition")

    margins = centered_logit_margin(scores, threshold)
    decisive_labels = gnn_only[decisive].astype(int)
    decisive_scores = scores[decisive]
    preference_metrics = binary_metrics(decisive_labels, decisive_scores)
    gnn_only_count = int(np.sum(gnn_only))
    llm_only_count = int(np.sum(llm_only))
    gnn_only_correct_route = int(np.sum(gnn_only & routes_gnn))
    llm_only_correct_route = int(np.sum(llm_only & ~routes_gnn))
    gnn_only_recall = _safe_rate(gnn_only_correct_route, gnn_only_count)
    llm_only_recall = _safe_rate(llm_only_correct_route, llm_only_count)
    balanced_accuracy = (
        0.5 * (gnn_only_recall + llm_only_recall)
        if gnn_only_recall is not None and llm_only_recall is not None
        else None
    )
    preference_metrics.update(
        {
            "ks_distance": ks_distance(scores[gnn_only], scores[llm_only]),
            "gnn_only_median_margin": float(np.median(margins[gnn_only])),
            "llm_only_median_margin": float(np.median(margins[llm_only])),
            "gnn_only_recall": gnn_only_recall,
            "llm_only_recall": llm_only_recall,
            "decisive_accuracy": (
                (gnn_only_correct_route + llm_only_correct_route)
                / int(np.sum(decisive))
            ),
            "balanced_accuracy": balanced_accuracy,
        }
    )

    gnn_accuracy = _accuracy(gnn_correct)
    llm_accuracy = _accuracy(llm_correct)
    router_accuracy = _accuracy(selected_correct)
    oracle_accuracy = _accuracy(gnn_correct | llm_correct)
    if gnn_accuracy >= llm_accuracy:
        comparator = "gnn"
        comparator_correct = gnn_correct
        best_single_accuracy = gnn_accuracy
    else:
        comparator = "llm"
        comparator_correct = llm_correct
        best_single_accuracy = llm_accuracy
    wins = int(np.sum(selected_correct & ~comparator_correct))
    losses = int(np.sum(~selected_correct & comparator_correct))
    ties = int(selected_correct.size - wins - losses)
    headroom = oracle_accuracy - best_single_accuracy
    oracle_recovery = (
        (router_accuracy - best_single_accuracy) / headroom
        if headroom > 0.0
        else None
    )

    counts = {
        "all": int(node_id.size),
        "agreement": int(np.sum(agreement)),
        "conflict": int(np.sum(conflict)),
        "decisive": int(np.sum(decisive)),
        "both_correct": int(np.sum(both_correct)),
        "gnn_only": gnn_only_count,
        "llm_only": llm_only_count,
        "both_wrong": int(np.sum(both_wrong)),
    }
    routing = {
        "threshold": threshold,
        "trust_threshold": trust_threshold,
        "gnn_accuracy": gnn_accuracy,
        "llm_accuracy": llm_accuracy,
        "best_single_accuracy": best_single_accuracy,
        "comparator": comparator,
        "router_accuracy": router_accuracy,
        "gain_over_best_single": router_accuracy - best_single_accuracy,
        "oracle_accuracy": oracle_accuracy,
        "oracle_gap": oracle_accuracy - router_accuracy,
        "oracle_recovery": oracle_recovery,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "net_correct": wins - losses,
        "gnn_route_rate_on_conflict": float(np.mean(routes_gnn[conflict])),
    }
    trust_metrics = binary_metrics(selected_correct.astype(int), trust)

    arrays = {
        "node_id": node_id,
        "true_label": true_label,
        "gnn_pred": gnn_pred,
        "llm_pred": llm_pred,
        "selected_pred": selected_pred,
        "preference_score": scores,
        "decision_margin": margins,
        "selected_trust": trust,
        "route_gnn": routes_gnn,
        "gnn_correct": gnn_correct,
        "llm_correct": llm_correct,
        "selected_correct": selected_correct,
        "agreement": agreement,
        "conflict": conflict,
        "decisive": decisive,
        "both_correct": both_correct,
        "gnn_only": gnn_only,
        "llm_only": llm_only,
        "both_wrong": both_wrong,
    }
    analysis = {
        "dataset": dataset,
        "checkpoint": metrics.get("scheme_b", {})
        .get("selected_checkpoint", {})
        .get("name"),
        "view": metrics.get("scheme_b", {})
        .get("selected_metadata", {})
        .get("view"),
        "trust_checkpoint": _trust_checkpoint(metrics, rows),
        "counts": counts,
        "preference": preference_metrics,
        "preference_calibration_bins": calibration_bins(
            decisive_labels, decisive_scores
        ),
        "routing": routing,
        "trust": trust_metrics,
        "trust_calibration_bins": calibration_bins(
            selected_correct.astype(int), trust
        ),
        "arrays": arrays,
    }
    trust_segments = rank_segments(trust, node_id, n_segments=10)
    arrays["trust_rank_decile"] = trust_segments
    analysis["trust_rank_composition"] = trust_rank_composition(
        arrays, trust_segments
    )
    analysis["loss_nodes"] = loss_node_rows(analysis)
    analysis["loss_summary"] = loss_summary(
        analysis, analysis["loss_nodes"]
    )
    return analysis


def threshold_sensitivity(
    arrays: Mapping[str, np.ndarray],
    frozen_threshold: float,
    n_quantiles: int = 201,
) -> list[dict]:
    """Retrospective test sensitivity at deterministic score thresholds."""

    if n_quantiles < 2:
        raise ValueError("threshold_quantiles_too_small")
    scores = np.asarray(arrays["preference_score"], dtype=float)
    decisive = np.asarray(arrays["decisive"], dtype=bool)
    conflict = np.asarray(arrays["conflict"], dtype=bool)
    gnn_correct = np.asarray(arrays["gnn_correct"], dtype=bool)
    llm_correct = np.asarray(arrays["llm_correct"], dtype=bool)
    gnn_only = np.asarray(arrays["gnn_only"], dtype=bool)
    llm_only = np.asarray(arrays["llm_only"], dtype=bool)
    decisive_scores = scores[decisive]
    quantiles = np.quantile(decisive_scores, np.linspace(0.0, 1.0, n_quantiles))
    outside_low = float(np.nextafter(np.min(scores), -np.inf))
    outside_high = float(np.nextafter(np.max(scores), np.inf))
    thresholds = np.unique(
        np.concatenate(
            [
                quantiles,
                np.asarray([outside_low, outside_high, 0.5, frozen_threshold]),
            ]
        )
    )
    gnn_accuracy = _accuracy(gnn_correct)
    llm_accuracy = _accuracy(llm_correct)
    best_single = max(gnn_accuracy, llm_accuracy)
    gnn_only_count = int(np.sum(gnn_only))
    llm_only_count = int(np.sum(llm_only))
    result: list[dict] = []
    for threshold in thresholds:
        route_gnn = ~conflict | (scores >= threshold)
        selected_correct = np.where(route_gnn, gnn_correct, llm_correct)
        gnn_recall = _safe_rate(int(np.sum(gnn_only & route_gnn)), gnn_only_count)
        llm_recall = _safe_rate(
            int(np.sum(llm_only & ~route_gnn)), llm_only_count
        )
        balanced = (
            0.5 * (gnn_recall + llm_recall)
            if gnn_recall is not None and llm_recall is not None
            else None
        )
        result.append(
            {
                "threshold": float(threshold),
                "is_frozen": bool(threshold == frozen_threshold),
                "router_accuracy": _accuracy(selected_correct),
                "gain_over_best_single": _accuracy(selected_correct) - best_single,
                "decisive_accuracy": _accuracy(selected_correct[decisive]),
                "balanced_decisive_accuracy": balanced,
                "gnn_only_recall": gnn_recall,
                "llm_only_recall": llm_recall,
                "gnn_route_rate_on_conflict": float(np.mean(route_gnn[conflict])),
            }
        )
    return result


def risk_coverage(
    correctness: np.ndarray | Sequence[int],
    trust: np.ndarray | Sequence[float],
) -> list[dict]:
    """Return every prefix point of a deterministic risk-coverage curve."""

    correct = _as_1d(correctness, dtype=int)
    scores = _as_1d(trust, dtype=float)
    if correct.size != scores.size:
        raise ValueError("risk_coverage_length_mismatch")
    order = np.lexsort((np.arange(scores.size), -scores))
    cumulative = np.cumsum(correct[order])
    counts = np.arange(1, scores.size + 1)
    precision = cumulative / counts
    return [
        {
            "count": int(count),
            "coverage": float(count / scores.size),
            "precision": float(value),
            "risk": float(1.0 - value),
        }
        for count, value in zip(counts, precision)
    ]


def rank_segments(
    scores: np.ndarray | Sequence[float],
    node_ids: np.ndarray | Sequence[int],
    n_segments: int = 10,
) -> np.ndarray:
    """Assign deterministic one-based, near-equal descending-score segments."""

    values = _as_1d(scores, dtype=float)
    identifiers = _as_1d(node_ids, dtype=np.int64)
    if values.size != identifiers.size:
        raise ValueError("rank_segment_length_mismatch")
    if values.size == 0:
        raise ValueError("rank_segment_empty")
    if n_segments <= 0:
        raise ValueError("rank_segment_count_must_be_positive")
    if np.any(~np.isfinite(values)):
        raise ValueError("rank_segment_nonfinite_score")
    order = np.lexsort((identifiers, -values))
    result = np.empty(values.size, dtype=int)
    for segment, indices in enumerate(np.array_split(order, n_segments), 1):
        result[indices] = segment
    return result


def trust_rank_composition(
    arrays: Mapping[str, np.ndarray], rank_segment: np.ndarray | Sequence[int]
) -> list[dict]:
    """Summarize exclusive node states within trust-ranked segments."""

    segments = _as_1d(rank_segment, dtype=int)
    trust = np.asarray(arrays["selected_trust"], dtype=float)
    selected_correct = np.asarray(arrays["selected_correct"], dtype=bool)
    agreement = np.asarray(arrays["agreement"], dtype=bool)
    conflict = np.asarray(arrays["conflict"], dtype=bool)
    gnn_only = np.asarray(arrays["gnn_only"], dtype=bool)
    llm_only = np.asarray(arrays["llm_only"], dtype=bool)
    both_wrong = np.asarray(arrays["both_wrong"], dtype=bool)
    route_gnn = np.asarray(arrays["route_gnn"], dtype=bool)
    if any(
        values.size != segments.size
        for values in (
            trust,
            selected_correct,
            agreement,
            conflict,
            gnn_only,
            llm_only,
            both_wrong,
            route_gnn,
        )
    ):
        raise ValueError("trust_rank_composition_length_mismatch")

    categories = {
        "agreement_correct": agreement & selected_correct,
        "agreement_wrong": agreement & ~selected_correct,
        "gnn_only": gnn_only,
        "llm_only": llm_only,
        "conflict_neither": conflict & both_wrong,
    }
    category_total = sum(mask.astype(int) for mask in categories.values())
    if not np.all(category_total == 1):
        raise ValueError("trust_rank_category_partition")

    result: list[dict] = []
    for segment in sorted(np.unique(segments).tolist()):
        mask = segments == segment
        count = int(np.sum(mask))
        row = {
            "segment": int(segment),
            "count": count,
            "precision": float(np.mean(selected_correct[mask])),
            "agreement_fraction": float(np.mean(agreement[mask])),
            "route_gnn_fraction": float(np.mean(route_gnn[mask])),
            "trust_min": float(np.min(trust[mask])),
            "trust_max": float(np.max(trust[mask])),
        }
        for name, category in categories.items():
            category_count = int(np.sum(category & mask))
            row[f"{name}_count"] = category_count
            row[f"{name}_fraction"] = category_count / count
        result.append(row)
    return result


def _four_state(arrays: Mapping[str, np.ndarray], index: int) -> str:
    for state in ("both_correct", "gnn_only", "llm_only", "both_wrong"):
        if bool(np.asarray(arrays[state], dtype=bool)[index]):
            return state
    raise ValueError("missing_four_state")


def loss_node_rows(analysis: Mapping) -> list[dict]:
    """Return node-level paired losses against the saved best-single comparator."""

    arrays = analysis["arrays"]
    routing = analysis["routing"]
    comparator = str(routing["comparator"])
    comparator_correct = np.asarray(
        arrays[f"{comparator}_correct"], dtype=bool
    )
    selected_correct = np.asarray(arrays["selected_correct"], dtype=bool)
    loss_mask = comparator_correct & ~selected_correct
    trust_threshold = routing.get("trust_threshold")
    result: list[dict] = []
    for index in np.flatnonzero(loss_mask):
        trust = float(arrays["selected_trust"][index])
        state = _four_state(arrays, int(index))
        if state not in {"gnn_only", "llm_only"}:
            raise ValueError("paired_loss_must_be_decisive")
        margin = float(arrays["decision_margin"][index])
        route = "gnn" if bool(arrays["route_gnn"][index]) else "llm"
        result.append(
            {
                "dataset": str(analysis["dataset"]),
                "node_id": int(arrays["node_id"][index]),
                "true_label": int(arrays["true_label"][index]),
                "gnn_pred": int(arrays["gnn_pred"][index]),
                "llm_pred": int(arrays["llm_pred"][index]),
                "selected_pred": int(arrays["selected_pred"][index]),
                "comparator": comparator,
                "route": route,
                "state": state,
                "comparator_correct": bool(comparator_correct[index]),
                "selected_correct": bool(selected_correct[index]),
                "preference_score": float(arrays["preference_score"][index]),
                "routing_threshold": float(routing["threshold"]),
                "decision_margin": margin,
                "absolute_decision_margin": abs(margin),
                "selected_trust": trust,
                "trust_threshold": (
                    float(trust_threshold)
                    if trust_threshold is not None
                    else None
                ),
                "accepted_by_trust_threshold": (
                    bool(trust >= float(trust_threshold))
                    if trust_threshold is not None
                    else None
                ),
                "trust_rank_decile": int(
                    arrays["trust_rank_decile"][index]
                ),
            }
        )
    if len(result) != int(routing["losses"]):
        raise ValueError("paired_loss_count_mismatch")
    return result


def loss_summary(analysis: Mapping, rows: Sequence[Mapping]) -> dict:
    """Aggregate paired-loss score, margin, trust, and rank diagnostics."""

    result = {
        "dataset": str(analysis["dataset"]),
        "comparator": str(analysis["routing"]["comparator"]),
        "loss_count": len(rows),
        "state": None,
        "accepted_count": 0,
        "accepted_fraction": None,
    }
    for segment in range(1, 11):
        result[f"trust_decile_{segment}_count"] = int(
            sum(int(row["trust_rank_decile"]) == segment for row in rows)
        )
    if not rows:
        for prefix in (
            "preference_score",
            "absolute_decision_margin",
            "selected_trust",
        ):
            for suffix in ("q25", "median", "q75"):
                result[f"{prefix}_{suffix}"] = None
        return result

    states = sorted({str(row["state"]) for row in rows})
    result["state"] = states[0] if len(states) == 1 else "mixed"
    for key in (
        "preference_score",
        "absolute_decision_margin",
        "selected_trust",
    ):
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
        result[f"{key}_q25"] = float(q25)
        result[f"{key}_median"] = float(median)
        result[f"{key}_q75"] = float(q75)
    accepted = [
        row["accepted_by_trust_threshold"]
        for row in rows
        if row["accepted_by_trust_threshold"] is not None
    ]
    if accepted:
        result["accepted_count"] = int(sum(bool(value) for value in accepted))
        result["accepted_fraction"] = result["accepted_count"] / len(accepted)
    return result
