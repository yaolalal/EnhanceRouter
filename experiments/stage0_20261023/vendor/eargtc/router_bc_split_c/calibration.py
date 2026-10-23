"""Independent calibration and selective thresholds for split Scheme C."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class ValidationSplit:
    calibration: torch.Tensor
    threshold: torch.Tensor


@dataclass(frozen=True)
class PlattCalibrator:
    scale: float
    bias: float
    regularization: float
    count: int

    def logits(self, raw_logits: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(raw_logits, dtype=torch.float64)
        return value * float(self.scale) + float(self.bias)

    def probabilities_from_logits(self, raw_logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(raw_logits)).to(torch.float32)

    def probabilities(self, raw_probability: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(raw_probability, dtype=torch.float64).clamp(
            1e-7, 1 - 1e-7
        )
        return self.probabilities_from_logits(torch.logit(value))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ThresholdSelection:
    target_precision: float
    min_accepted: int
    threshold: float | None
    reject_all: bool
    accepted_count: int
    validation_count: int
    empirical_precision: float | None
    coverage: float
    wilson_lower_bound: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def deterministic_calibration_threshold_split(
    node_ids: torch.Tensor, dataset: str, module: str
) -> ValidationSplit:
    """Create a stable, label-blind 50/50 validation split."""

    ids = torch.as_tensor(node_ids, dtype=torch.long).view(-1)
    if ids.numel() < 2:
        raise ValueError(
            "calibration/threshold split needs at least two validation rows"
        )
    if ids.numel() != torch.unique(ids).numel():
        raise ValueError("validation node IDs must be unique")
    digests = [
        hashlib.sha256(f"{dataset}|{module}|{int(node_id)}".encode("utf-8")).digest()
        for node_id in ids.tolist()
    ]
    calibration = torch.tensor(
        [bool(digest[0] & 1) for digest in digests], dtype=torch.bool
    )
    if not bool(calibration.any()) or bool(calibration.all()):
        order = sorted(range(ids.numel()), key=lambda index: digests[index])
        calibration = torch.zeros(ids.numel(), dtype=torch.bool)
        calibration[order[: ids.numel() // 2]] = True
    return ValidationSplit(calibration=calibration, threshold=~calibration)


def _platt_objective(
    design: torch.Tensor,
    target: torch.Tensor,
    params: torch.Tensor,
    regularization: float,
) -> torch.Tensor:
    logits = design @ params
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    center = torch.tensor([1.0, 0.0], dtype=torch.float64)
    return loss + 0.5 * float(regularization) * (params - center).square().sum()


def fit_identity_shrunk_platt(
    raw_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    regularization: float = 0.1,
    max_iter: int = 100,
) -> PlattCalibrator:
    """Fit affine Platt scaling shrunk toward the identity mapping.

    The slope is intentionally unconstrained: a negative value is an auditable
    target-calibration signal that the source-transferred ordering is reversed.
    """

    logits = torch.as_tensor(raw_logits, dtype=torch.float64).view(-1)
    y = torch.as_tensor(target, dtype=torch.float64).view(-1)
    if logits.numel() == 0 or logits.numel() != y.numel():
        raise ValueError("Platt calibration requires aligned non-empty inputs")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("raw logits contain non-finite values")
    if bool(((y != 0) & (y != 1)).any()):
        raise ValueError("calibration targets must be binary")
    if float(regularization) < 0.0:
        raise ValueError("regularization must be non-negative")

    design = torch.column_stack([logits, torch.ones_like(logits)])
    params = torch.tensor([1.0, 0.0], dtype=torch.float64)
    center = params.clone()
    regularization = float(regularization)
    for _ in range(max(int(max_iter), 1)):
        probability = torch.sigmoid(design @ params).clamp(1e-9, 1 - 1e-9)
        gradient = design.T @ (probability - y) / y.numel()
        gradient += regularization * (params - center)
        curvature = probability * (1.0 - probability) / y.numel()
        hessian = design.T @ (curvature[:, None] * design)
        hessian += torch.eye(2, dtype=torch.float64) * (regularization + 1e-8)
        try:
            step = torch.linalg.solve(hessian, gradient)
        except RuntimeError:
            step = torch.linalg.pinv(hessian) @ gradient

        old_objective = _platt_objective(design, y, params, regularization)
        step_size = 1.0
        candidate = params - step
        while (
            _platt_objective(design, y, candidate, regularization) > old_objective
            and step_size > 1e-6
        ):
            step_size *= 0.5
            candidate = params - step_size * step
        if float((candidate - params).abs().max()) < 1e-10:
            params = candidate
            break
        params = candidate
    return PlattCalibrator(
        scale=float(params[0]),
        bias=float(params[1]),
        regularization=regularization,
        count=int(y.numel()),
    )


def wilson_lower_bound(
    successes: int, count: int, *, z: float = 1.6448536269514722
) -> float | None:
    """One-sided Wilson lower confidence bound (95% by default)."""

    successes, count = int(successes), int(count)
    if count <= 0:
        return None
    if successes < 0 or successes > count:
        raise ValueError("successes must lie in [0, count]")
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = proportion + z * z / (2.0 * count)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)
    )
    return max(0.0, (center - radius) / denominator)


def select_precision_threshold(
    probability: torch.Tensor,
    correct: torch.Tensor,
    *,
    target_precision: float,
    min_accepted: int = 10,
) -> ThresholdSelection:
    """Choose maximum validation coverage satisfying empirical precision."""

    p = torch.as_tensor(probability, dtype=torch.float64).view(-1)
    y = torch.as_tensor(correct, dtype=torch.bool).view(-1)
    if p.numel() == 0 or p.numel() != y.numel():
        raise ValueError("threshold selection requires aligned non-empty inputs")
    if not bool(torch.isfinite(p).all()) or bool(((p < 0) | (p > 1)).any()):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    if not 0.0 < float(target_precision) <= 1.0:
        raise ValueError("target_precision must lie in (0, 1]")
    if int(min_accepted) <= 0:
        raise ValueError("min_accepted must be positive")

    order = torch.argsort(p, descending=True, stable=True)
    sorted_p, sorted_y = p[order], y[order].to(torch.int64)
    cumulative = torch.cumsum(sorted_y, dim=0)
    endpoints = torch.ones(p.numel(), dtype=torch.bool)
    endpoints[:-1] = sorted_p[:-1] > sorted_p[1:]
    best: tuple[int, int, float] | None = None
    for endpoint in endpoints.nonzero().view(-1).tolist():
        accepted = endpoint + 1
        successes = int(cumulative[endpoint])
        precision = successes / accepted
        if accepted >= int(min_accepted) and precision >= float(target_precision):
            if best is None or accepted > best[0]:
                best = (accepted, successes, float(sorted_p[endpoint]))
    if best is None:
        return ThresholdSelection(
            target_precision=float(target_precision),
            min_accepted=int(min_accepted),
            threshold=None,
            reject_all=True,
            accepted_count=0,
            validation_count=int(p.numel()),
            empirical_precision=None,
            coverage=0.0,
            wilson_lower_bound=None,
        )
    accepted, successes, threshold = best
    return ThresholdSelection(
        target_precision=float(target_precision),
        min_accepted=int(min_accepted),
        threshold=threshold,
        reject_all=False,
        accepted_count=accepted,
        validation_count=int(p.numel()),
        empirical_precision=successes / accepted,
        coverage=accepted / p.numel(),
        wilson_lower_bound=wilson_lower_bound(successes, accepted),
    )


def apply_threshold(
    probability: torch.Tensor, selection: ThresholdSelection
) -> torch.Tensor:
    p = torch.as_tensor(probability, dtype=torch.float64).view(-1)
    if selection.reject_all or selection.threshold is None:
        return torch.zeros(p.numel(), dtype=torch.bool)
    return p >= float(selection.threshold)


__all__ = [
    "PlattCalibrator",
    "ThresholdSelection",
    "ValidationSplit",
    "apply_threshold",
    "deterministic_calibration_threshold_split",
    "fit_identity_shrunk_platt",
    "select_precision_threshold",
    "wilson_lower_bound",
]
