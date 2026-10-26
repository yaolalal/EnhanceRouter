from __future__ import annotations

import numpy as np


def _array(values, name: str) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).reshape(-1)
    if output.size == 0 or not np.isfinite(output).all():
        raise ValueError(f"{name} must be non-empty and finite")
    return output


def empirical_ks(left, right) -> float:
    left = np.sort(_array(left, "left"))
    right = np.sort(_array(right, "right"))
    values = np.unique(np.concatenate([left, right]))
    left_cdf = np.searchsorted(left, values, side="right") / left.size
    right_cdf = np.searchsorted(right, values, side="right") / right.size
    return float(np.max(np.abs(left_cdf - right_cdf)))


def empirical_wasserstein_1d(left, right) -> float:
    left = np.sort(_array(left, "left"))
    right = np.sort(_array(right, "right"))
    values = np.unique(np.concatenate([left, right]))
    if values.size == 1:
        return 0.0
    intervals = np.diff(values)
    left_cdf = np.searchsorted(left, values[:-1], side="right") / left.size
    right_cdf = np.searchsorted(right, values[:-1], side="right") / right.size
    return float(np.sum(np.abs(left_cdf - right_cdf) * intervals))


def _summary(values: np.ndarray) -> dict:
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def distribution_diagnostics(
    *, oof_values, aligned_query_values, test_query_values
) -> dict:
    oof = _array(oof_values, "oof_values")
    aligned = _array(aligned_query_values, "aligned_query_values")
    test = _array(test_query_values, "test_query_values")
    if oof.size != aligned.size:
        raise ValueError("OOF and aligned query values must have equal length")
    return {
        "counts": {
            "oof": int(oof.size),
            "aligned_query": int(aligned.size),
            "test_query": int(test.size),
        },
        "oof": _summary(oof),
        "aligned_query": _summary(aligned),
        "test_query": _summary(test),
        "aligned_mae": float(np.mean(np.abs(oof - aligned))),
        "oof_vs_test_ks": empirical_ks(oof, test),
        "oof_vs_test_wasserstein": empirical_wasserstein_1d(oof, test),
    }
