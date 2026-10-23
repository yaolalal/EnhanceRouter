"""Population-specific reliability metrics and paired uncertainty estimates."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from eargtc.router_bc.metrics import binary_auroc, probability_metrics

from .calibration import ThresholdSelection, apply_threshold


def _aligned(
    probability: torch.Tensor, correct: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    p = torch.as_tensor(probability, dtype=torch.float64).view(-1)
    y = torch.as_tensor(correct, dtype=torch.bool).view(-1)
    if p.numel() == 0 or p.numel() != y.numel():
        raise ValueError("evaluation requires aligned non-empty inputs")
    if not bool(torch.isfinite(p).all()) or bool(((p < 0) | (p > 1)).any()):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    return p, y


def average_precision(probability: torch.Tensor, correct: torch.Tensor) -> float | None:
    p, y = _aligned(probability, correct)
    positives = int(y.sum())
    if positives == 0:
        return None
    order = torch.argsort(p, descending=True, stable=True)
    ranked_p = p[order]
    ranked = y[order].to(torch.float64)
    cumulative_true = torch.cumsum(ranked, 0)
    endpoints = torch.ones(ranked.numel(), dtype=torch.bool)
    endpoints[:-1] = ranked_p[:-1] > ranked_p[1:]
    endpoint_ids = endpoints.nonzero().view(-1)
    true_at_endpoint = cumulative_true[endpoint_ids]
    previous_true = torch.cat(
        [torch.zeros(1, dtype=torch.float64), true_at_endpoint[:-1]]
    )
    precision_at_endpoint = true_at_endpoint / (endpoint_ids.to(torch.float64) + 1.0)
    recall_increment = (true_at_endpoint - previous_true) / positives
    return float((precision_at_endpoint * recall_increment).sum())


def risk_coverage_curve(probability: torch.Tensor, correct: torch.Tensor) -> list[dict]:
    p, y = _aligned(probability, correct)
    order = torch.argsort(p, descending=True, stable=True)
    ranked_p = p[order]
    cumulative_correct = torch.cumsum(y[order].to(torch.int64), 0)
    result = []
    for index in range(p.numel()):
        count = index + 1
        successes = int(cumulative_correct[index])
        result.append(
            {
                "accepted_count": count,
                "coverage": count / p.numel(),
                "threshold": float(ranked_p[index]),
                "precision": successes / count,
                "risk": 1.0 - successes / count,
            }
        )
    return result


def _aurc(probability: torch.Tensor, correct: torch.Tensor) -> float:
    order = torch.argsort(probability, descending=True, stable=True)
    ranked = correct[order].to(torch.float64)
    risk = 1.0 - torch.cumsum(ranked, 0) / torch.arange(
        1, ranked.numel() + 1, dtype=torch.float64
    )
    return float(risk.mean())


def _oracle_aurc(correct: torch.Tensor) -> float:
    y = torch.as_tensor(correct, dtype=torch.bool).view(-1)
    oracle_score = y.to(torch.float64)
    return _aurc(oracle_score, y)


def _fixed_coverage_rows(curve: list[dict]) -> dict[str, dict]:
    result = {}
    count = len(curve)
    for coverage in (0.10, 0.25, 0.50, 0.75, 1.00):
        accepted = min(count, max(1, int(math.ceil(coverage * count))))
        row = curve[accepted - 1]
        result[f"{coverage:.2f}"] = dict(row)
    return result


def _reliability_bins(
    probability: torch.Tensor, correct: torch.Tensor, bins: int = 10
) -> list[dict]:
    p, y = _aligned(probability, correct)
    output = []
    for index in range(int(bins)):
        low, high = index / bins, (index + 1) / bins
        mask = (p >= low) & ((p < high) if index + 1 < bins else (p <= high))
        output.append(
            {
                "low": low,
                "high": high,
                "count": int(mask.sum()),
                "mean_probability": float(p[mask].mean()) if bool(mask.any()) else None,
                "accuracy": float(y[mask].to(torch.float64).mean())
                if bool(mask.any())
                else None,
            }
        )
    return output


def probability_report(
    probability: torch.Tensor, correct: torch.Tensor, *, include_curve: bool = True
) -> dict:
    p, y = _aligned(probability, correct)
    base = probability_metrics(p, y)
    curve = risk_coverage_curve(p, y)
    aurc = sum(float(row["risk"]) for row in curve) / len(curve)
    oracle = _oracle_aurc(y)
    report = {
        **base,
        "positive_count": int(y.sum()),
        "negative_count": int((~y).sum()),
        "base_accuracy": float(y.to(torch.float64).mean()),
        "average_precision": average_precision(p, y),
        "aurc": aurc,
        "oracle_aurc": oracle,
        "excess_aurc": aurc - oracle,
        "fixed_coverage": _fixed_coverage_rows(curve),
        "reliability_bins": _reliability_bins(p, y),
    }
    if include_curve:
        report["risk_coverage_curve"] = curve
    return report


def subgroup_report(
    probability: torch.Tensor, correct: torch.Tensor, groups: Sequence[str]
) -> dict[str, dict]:
    p, y = _aligned(probability, correct)
    if len(groups) != p.numel():
        raise ValueError("group labels must align with probabilities")
    result = {}
    for group in sorted(set(str(value) for value in groups)):
        mask = torch.tensor([str(value) == group for value in groups], dtype=torch.bool)
        result[group] = probability_report(p[mask], y[mask], include_curve=False)
    return result


def evaluate_frozen_thresholds(
    probability: torch.Tensor,
    correct: torch.Tensor,
    selections: dict[str, ThresholdSelection],
) -> dict[str, dict]:
    p, y = _aligned(probability, correct)
    output = {}
    for name, selection in sorted(selections.items()):
        accepted = apply_threshold(p, selection)
        count = int(accepted.sum())
        successes = int(y[accepted].sum()) if count else 0
        precision = successes / count if count else None
        output[name] = {
            "threshold": selection.threshold,
            "reject_all": selection.reject_all,
            "accepted_count": count,
            "count": int(p.numel()),
            "coverage": count / p.numel(),
            "correct_count": successes,
            "precision": precision,
            "risk": 1.0 - precision if precision is not None else None,
            "validation_selection": selection.to_dict(),
        }
    return output


def _nll(probability: torch.Tensor, target: torch.Tensor) -> float:
    p = probability.clamp(1e-7, 1 - 1e-7)
    y = target.to(torch.float64)
    return float((-(y * p.log() + (1 - y) * (1 - p).log())).mean())


def _brier(probability: torch.Tensor, target: torch.Tensor) -> float:
    return float((probability - target.to(torch.float64)).square().mean())


def _matched_precision(
    probability: torch.Tensor, target: torch.Tensor, count: int
) -> float:
    accepted = min(max(int(count), 1), probability.numel())
    order = torch.argsort(probability, descending=True, stable=True)
    return float(target[order[:accepted]].to(torch.float64).mean())


def _weighted_metrics(
    probability: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    matched_count: int,
    harmonic: torch.Tensor,
    order: torch.Tensor | None = None,
) -> dict[str, float]:
    """Exact metrics for a bootstrap sample represented by node multiplicities.

    Sorting is performed once per score vector; harmonic-number differences
    reproduce every repeated node's cumulative-risk contribution without
    materializing or re-sorting the expanded bootstrap sample.
    """

    p = torch.as_tensor(probability, dtype=torch.float64).view(-1)
    y = torch.as_tensor(target, dtype=torch.bool).view(-1)
    weight = torch.as_tensor(weights, dtype=torch.long).view(-1)
    if not (p.numel() == y.numel() == weight.numel()) or bool((weight < 0).any()):
        raise ValueError("weighted metrics require aligned non-negative multiplicities")
    count = int(weight.sum())
    if count <= 0 or harmonic.numel() <= count:
        raise ValueError("harmonic table must cover the non-empty weighted sample")
    clipped = p.clamp(1e-7, 1 - 1e-7)
    binary = y.to(torch.float64)
    nll_values = -(binary * clipped.log() + (1 - binary) * (1 - clipped).log())
    brier_values = (p - binary).square()

    if order is None:
        order = torch.argsort(p, descending=True, stable=True)
    ranked_y = y[order]
    ranked_weight = weight[order]
    accepted_before = torch.cumsum(ranked_weight, 0) - ranked_weight
    errors = (~ranked_y).to(torch.long)
    error_weight = ranked_weight * errors
    errors_before = torch.cumsum(error_weight, 0) - error_weight
    accepted_after = accepted_before + ranked_weight
    harmonic_difference = harmonic[accepted_after] - harmonic[accepted_before]
    correct_before = accepted_before - errors_before
    risk_contribution = torch.where(
        ranked_y,
        errors_before.to(torch.float64) * harmonic_difference,
        ranked_weight.to(torch.float64)
        - correct_before.to(torch.float64) * harmonic_difference,
    )
    target_count = min(max(int(matched_count), 1), count)
    take = (target_count - accepted_before).clamp(min=0)
    take = torch.minimum(take, ranked_weight)
    matched_successes = (take * ranked_y.to(torch.long)).sum()
    return {
        "nll": float((weight.to(torch.float64) * nll_values).sum() / count),
        "brier": float((weight.to(torch.float64) * brier_values).sum() / count),
        "aurc": float(risk_contribution.sum() / count),
        "matched_precision": float(matched_successes / target_count),
    }


def _interval(values: list[float], estimate: float) -> dict:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "estimate": float(estimate),
        "ci95_low": float(torch.quantile(tensor, 0.025)),
        "ci95_high": float(torch.quantile(tensor, 0.975)),
        "probability_better": float((tensor < 0).to(torch.float64).mean()),
    }


def paired_bootstrap_comparison(
    new_probability: torch.Tensor,
    old_probability: torch.Tensor,
    correct: torch.Tensor,
    *,
    matched_count: int,
    replicates: int = 1000,
    seed: int = 42,
) -> dict[str, dict]:
    """Paired node bootstrap; deltas are new minus old (lower is better except precision)."""

    new, y = _aligned(new_probability, correct)
    old, old_y = _aligned(old_probability, correct)
    if new.numel() != old.numel() or not torch.equal(y, old_y):
        raise ValueError("paired bootstrap inputs must be node-aligned")
    if int(replicates) <= 0:
        raise ValueError("replicates must be positive")
    matched_count = min(max(int(matched_count), 1), new.numel())

    harmonic = torch.zeros(new.numel() + 1, dtype=torch.float64)
    harmonic[1:] = torch.cumsum(
        1.0 / torch.arange(1, new.numel() + 1, dtype=torch.float64), 0
    )
    new_order = torch.argsort(new, descending=True, stable=True)
    old_order = torch.argsort(old, descending=True, stable=True)

    def deltas(weight: torch.Tensor) -> dict[str, float]:
        new_metrics = _weighted_metrics(
            new,
            y,
            weight,
            matched_count=matched_count,
            harmonic=harmonic,
            order=new_order,
        )
        old_metrics = _weighted_metrics(
            old,
            y,
            weight,
            matched_count=matched_count,
            harmonic=harmonic,
            order=old_order,
        )
        return {
            "nll_delta": new_metrics["nll"] - old_metrics["nll"],
            "brier_delta": new_metrics["brier"] - old_metrics["brier"],
            "aurc_delta": new_metrics["aurc"] - old_metrics["aurc"],
            "matched_precision_delta": new_metrics["matched_precision"]
            - old_metrics["matched_precision"],
        }

    observed = deltas(torch.ones(new.numel(), dtype=torch.long))
    samples = {key: [] for key in observed}
    generator = torch.Generator().manual_seed(int(seed))
    for _ in range(int(replicates)):
        indices = torch.randint(new.numel(), (new.numel(),), generator=generator)
        weight = torch.bincount(indices, minlength=new.numel())
        values = deltas(weight)
        for key, value in values.items():
            samples[key].append(value)
    output = {key: _interval(samples[key], value) for key, value in observed.items()}
    output["matched_precision_delta"]["probability_better"] = float(
        (torch.tensor(samples["matched_precision_delta"]) > 0).to(torch.float64).mean()
    )
    return output


__all__ = [
    "average_precision",
    "evaluate_frozen_thresholds",
    "paired_bootstrap_comparison",
    "probability_report",
    "risk_coverage_curve",
    "subgroup_report",
]
