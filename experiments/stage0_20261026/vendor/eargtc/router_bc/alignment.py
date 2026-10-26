from __future__ import annotations

import math


def choose_alignment_checkpoint(
    raw_validation_accuracy: float,
    rank_validation_accuracy: float,
    validation_count: int,
    *,
    z: float = 1.645,
) -> dict:
    """Conservative validation early stopping on the raw→rank trajectory.

    The independent-proportion standard error is deliberately conservative for
    paired predictions.  Rank alignment is accepted only when its validation
    gain clears a one-sided uncertainty margin; otherwise the raw checkpoint is
    retained.
    """
    count = int(validation_count)
    if count <= 0:
        raise ValueError("validation_count must be positive")
    raw = float(raw_validation_accuracy)
    rank = float(rank_validation_accuracy)
    pooled = min(max(0.5 * (raw + rank), 0.0), 1.0)
    standard_error = math.sqrt(max(2.0 * pooled * (1.0 - pooled) / count, 0.0))
    required_gain = float(z) * standard_error
    gain = rank - raw
    return {
        "selected": "rank" if gain > required_gain else "raw",
        "raw_validation_accuracy": raw,
        "rank_validation_accuracy": rank,
        "validation_count": count,
        "rank_gain": gain,
        "standard_error_conservative": standard_error,
        "z": float(z),
        "required_gain": required_gain,
        "rule": "rank_gain_gt_z_times_independent_proportion_se_else_raw",
    }
