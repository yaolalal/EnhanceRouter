from __future__ import annotations

import math

import numpy as np


def _bandwidth(values: np.ndarray, grid: np.ndarray) -> float:
    if values.size > 1:
        standard = float(np.std(values, ddof=1))
        q1, q3 = np.quantile(values, [0.25, 0.75])
        robust = float((q3 - q1) / 1.34)
        scale_candidates = [value for value in (standard, robust) if value > 0]
        if scale_candidates:
            result = 0.9 * min(scale_candidates) * values.size ** (-0.2)
            if math.isfinite(result) and result > 0:
                return max(result, 2.0 * float(np.min(np.diff(grid))))
    return max(0.02, 2.0 * float(np.min(np.diff(grid))))


def bounded_density(
    values,
    *,
    bin_edges,
    grid,
) -> dict:
    sample = np.asarray(values, dtype=float).reshape(-1)
    edges = np.asarray(bin_edges, dtype=float).reshape(-1)
    points = np.asarray(grid, dtype=float).reshape(-1)
    if sample.size == 0:
        raise ValueError("density_empty")
    if np.any(~np.isfinite(sample)) or np.any((sample < 0) | (sample > 1)):
        raise ValueError("density_score_out_of_range")
    if (
        edges.size < 2
        or points.size < 2
        or np.any(np.diff(edges) <= 0)
        or np.any(np.diff(points) <= 0)
        or edges[0] != 0
        or edges[-1] != 1
        or points[0] != 0
        or points[-1] != 1
    ):
        raise ValueError("density_grid_must_span_unit_interval")
    histogram, _ = np.histogram(sample, bins=edges, density=True)
    bandwidth = _bandwidth(sample, points)
    density = np.zeros(points.size, dtype=float)
    normalizer = 1.0 / (sample.size * bandwidth * math.sqrt(2.0 * math.pi))
    for start in range(0, sample.size, 4096):
        chunk = sample[start : start + 4096]
        for reflected in (chunk, -chunk, 2.0 - chunk):
            z = (points[:, None] - reflected[None, :]) / bandwidth
            density += np.exp(-0.5 * z * z).sum(axis=1)
    density *= normalizer
    density = np.maximum(density, 0.0)
    integral = float(np.trapezoid(density, points))
    if not math.isfinite(integral) or integral <= 0:
        raise ValueError("density_kde_normalization_failure")
    density /= integral
    return {
        "count": int(sample.size),
        "median": float(np.median(sample)),
        "mean": float(np.mean(sample)),
        "bandwidth": float(bandwidth),
        "bin_edges": edges,
        "hist_density": histogram,
        "grid": points,
        "kde_density": density,
    }
