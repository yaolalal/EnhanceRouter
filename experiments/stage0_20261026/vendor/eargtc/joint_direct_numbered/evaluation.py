from __future__ import annotations

import math
import statistics
from collections.abc import Iterable

import torch
import torch.nn.functional as F


def _f1_by_class(pred: torch.Tensor, target: torch.Tensor, class_count: int) -> tuple[list[float], list[int]]:
    scores: list[float] = []
    supports: list[int] = []
    for label in range(int(class_count)):
        predicted = pred == label
        actual = target == label
        tp = int((predicted & actual).sum())
        fp = int((predicted & ~actual).sum())
        fn = int((~predicted & actual).sum())
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else (2 * tp) / denominator)
        supports.append(int(actual.sum()))
    return scores, supports


def expected_calibration_error(logits: torch.Tensor, target: torch.Tensor, bins: int = 15) -> float:
    probability = torch.softmax(torch.as_tensor(logits).float(), dim=-1)
    confidence, pred = probability.max(dim=-1)
    correct = (pred == torch.as_tensor(target).long()).float()
    result = 0.0
    for index in range(int(bins)):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (confidence > lower) & (confidence <= upper)
        if index == 0:
            mask |= confidence == 0
        if mask.any():
            result += float(mask.float().mean() * (correct[mask].mean() - confidence[mask].mean()).abs())
    return result


def classification_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    class_count: int | None = None,
    ece_bins: int = 15,
) -> dict[str, float]:
    value = torch.as_tensor(logits).float()
    gold = torch.as_tensor(target).long().view(-1)
    if value.ndim != 2 or value.size(0) != gold.numel() or not gold.numel():
        raise ValueError("classification inputs must be non-empty [N,C] logits and N targets")
    if not torch.isfinite(value).all():
        raise ValueError("classification logits contain NaN/Inf")
    classes = value.size(1) if class_count is None else int(class_count)
    if classes != value.size(1):
        raise ValueError("class_count differs from logit width")
    pred = value.argmax(dim=-1)
    f1, support = _f1_by_class(pred, gold, classes)
    count = gold.numel()
    return {
        "accuracy": float((pred == gold).float().mean()),
        "macro_f1": sum(f1) / classes,
        "weighted_f1": sum(score * size for score, size in zip(f1, support)) / count,
        "nll": float(F.cross_entropy(value, gold)),
        "ece": expected_calibration_error(value, gold, ece_bins),
    }


def correction_damage(
    baseline_pred: torch.Tensor,
    joint_pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, int | float]:
    baseline = torch.as_tensor(baseline_pred).long().view(-1)
    joint = torch.as_tensor(joint_pred).long().view(-1)
    gold = torch.as_tensor(target).long().view(-1)
    if baseline.numel() != joint.numel() or joint.numel() != gold.numel() or not gold.numel():
        raise ValueError("paired predictions/targets must have the same non-zero length")
    b = baseline == gold
    j = joint == gold
    corrections = int((~b & j).sum())
    damages = int((b & ~j).sum())
    stable_correct = int((b & j).sum())
    stable_wrong = int((~b & ~j).sum())
    count = gold.numel()
    if corrections + damages + stable_correct + stable_wrong != count:
        raise AssertionError("correction-damage partition is incomplete")
    gain = (corrections - damages) / count
    observed = float(j.float().mean() - b.float().mean())
    if not math.isclose(observed, gain, abs_tol=1e-7):
        raise AssertionError("correction-damage gain identity failed")
    return {
        "corrections": corrections,
        "damages": damages,
        "stable_correct": stable_correct,
        "stable_wrong": stable_wrong,
        "count": count,
        "gain": gain,
    }


def paired_exact_p(corrections: int, damages: int) -> float:
    correction = int(corrections); damage = int(damages)
    if correction < 0 or damage < 0:
        raise ValueError("discordant counts cannot be negative")
    total = correction + damage
    if total == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(min(correction, damage), total, 0.5, alternative="two-sided").pvalue)
    except ImportError:
        tail = sum(math.comb(total, index) for index in range(min(correction, damage) + 1)) / (2 ** total)
        return min(1.0, 2.0 * tail)


def paired_bootstrap_gain(
    baseline_correct: torch.Tensor,
    joint_correct: torch.Tensor,
    *,
    samples: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    baseline = torch.as_tensor(baseline_correct, dtype=torch.float32).view(-1)
    joint = torch.as_tensor(joint_correct, dtype=torch.float32).view(-1)
    if baseline.numel() != joint.numel() or not baseline.numel() or samples <= 0:
        raise ValueError("bootstrap inputs must be equal, non-empty, with positive samples")
    difference = joint - baseline
    generator = torch.Generator(device="cpu"); generator.manual_seed(int(seed))
    values = []
    remaining = int(samples)
    while remaining:
        chunk = min(remaining, 128)
        indices = torch.randint(difference.numel(), (chunk, difference.numel()), generator=generator)
        values.append(difference[indices].mean(dim=1))
        remaining -= chunk
    gains = torch.cat(values).sort().values
    lower = float(torch.quantile(gains, 0.025))
    upper = float(torch.quantile(gains, 0.975))
    return lower, upper


def aggregate_seed_metrics(rows: Iterable[dict[str, float]]) -> dict[str, dict[str, float]]:
    values = list(rows)
    if not values:
        raise ValueError("cannot aggregate zero seeds")
    keys = set(values[0])
    if any(set(row) != keys for row in values):
        raise ValueError("seed metric schemas differ")
    result = {}
    for key in sorted(keys):
        column = [float(row[key]) for row in values]
        result[key] = {
            "mean": statistics.fmean(column),
            "std": 0.0 if len(column) == 1 else statistics.stdev(column),
            "min": min(column),
            "max": max(column),
        }
    return result
