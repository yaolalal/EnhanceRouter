from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch



def _correct_curve(
    probability: torch.Tensor,
    g_correct: torch.Tensor,
    l_correct: torch.Tensor,
    thresholds: torch.Tensor,
) -> torch.Tensor:
    """Exact correctness at every threshold in O(n log n + k log n)."""
    p = torch.as_tensor(probability, dtype=torch.float64).view(-1)
    g = torch.as_tensor(g_correct, dtype=torch.int64).view(-1)
    l = torch.as_tensor(l_correct, dtype=torch.int64).view(-1)
    if not (p.numel() == g.numel() == l.numel()):
        raise ValueError("threshold curve inputs must have equal length")
    order = torch.argsort(p, stable=True)
    delta = (g - l)[order]
    prefix = torch.cat([torch.zeros(1, dtype=torch.int64), delta.cumsum(0)])
    first_ge = torch.searchsorted(p[order], thresholds, right=False)
    return l.sum() + delta.sum() - prefix[first_ge]


def _environment_curve(
    record: Mapping[str, torch.Tensor], thresholds: torch.Tensor
) -> torch.Tensor:
    labels = torch.as_tensor(record["labels"]).view(-1)
    return _correct_curve(
        record["probability"],
        torch.as_tensor(record["gnn_pred"]).view(-1) == labels,
        torch.as_tensor(record["llm_pred"]).view(-1) == labels,
        thresholds,
    )


def _flip_curve(
    flip_pairs: Sequence[
        tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]
    ],
    thresholds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    curves, totals = [], []
    for base, shifted in flip_pairs:
        base_a = torch.as_tensor(base["advantage"], dtype=torch.float64).view(-1)
        shifted_a = torch.as_tensor(
            shifted["advantage"], dtype=torch.float64
        ).view(-1)
        valid = (base_a >= 0) != (shifted_a >= 0)
        valid &= (shifted_a - base_a).abs() >= 0.1
        if bool(valid.any()):
            p = torch.cat([
                torch.as_tensor(base["probability"])[valid],
                torch.as_tensor(shifted["probability"])[valid],
            ])
            target = torch.cat([(base_a >= 0)[valid], (shifted_a >= 0)[valid]])
            curves.append(_correct_curve(p, target, ~target, thresholds))
            totals.append(int(target.numel()))
    if not curves:
        return torch.zeros_like(thresholds), torch.zeros_like(thresholds, dtype=torch.int64), 0
    stacked = torch.stack(curves)
    macro = torch.stack([
        curve.double() / total for curve, total in zip(curves, totals)
    ]).mean(0)
    return macro, stacked.sum(0), sum(totals)


def select_competence_threshold(
    environments: Sequence[Mapping[str, torch.Tensor]],
    flip_pairs: Sequence[
        tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]
    ],
    *,
    accuracy_tolerance: float = 0.01,
) -> dict[str, float | int]:
    """Maximize competence response inside a validation-accuracy tolerance band."""
    if not environments:
        raise ValueError("threshold selection requires validation environments")
    tolerance = float(accuracy_tolerance)
    if tolerance < 0:
        raise ValueError("accuracy_tolerance must be non-negative")
    probabilities = torch.cat(
        [
            torch.as_tensor(record["probability"], dtype=torch.float64).view(-1)
            for record in environments
        ]
    )
    thresholds = torch.unique(
        torch.cat(
            [
                probabilities,
                torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
            ]
        ),
        sorted=True,
    )
    per_environment = torch.stack(
        [_environment_curve(record, thresholds) for record in environments]
    )
    totals = torch.tensor(
        [torch.as_tensor(record["labels"]).numel() for record in environments],
        dtype=torch.int64,
    )
    correct = per_environment.sum(0)
    count = int(totals.sum())
    accuracy = (per_environment.double() / totals.double().view(-1, 1)).mean(0)
    clean_accuracy = per_environment[0].double() / int(totals[0])
    flip_accuracy, flip_correct, flip_total = _flip_curve(flip_pairs, thresholds)
    best_accuracy = float(accuracy.max())
    best_clean = float(clean_accuracy.max())
    required_tolerance = torch.maximum(
        best_accuracy - accuracy, best_clean - clean_accuracy
    )
    effective_tolerance = max(tolerance, float(required_tolerance.min()))
    eligible = required_tolerance <= effective_tolerance + 1e-12
    eligible_indices = torch.nonzero(eligible, as_tuple=False).view(-1).tolist()
    chosen = min(
        eligible_indices,
        key=lambda index: (
            -float(flip_accuracy[index]),
            -float(accuracy[index]),
            abs(float(thresholds[index]) - 0.5),
            float(thresholds[index]),
        ),
    )
    return {
        "threshold": float(thresholds[chosen]),
        "correct": int(correct[chosen]),
        "count": count,
        "accuracy": float(accuracy[chosen]),
        "clean_accuracy": float(clean_accuracy[chosen]),
        "requested_accuracy_tolerance": tolerance,
        "effective_accuracy_tolerance": effective_tolerance,
        "flip_correct": int(flip_correct[chosen]),
        "flip_total": flip_total,
        "flip_accuracy": float(flip_accuracy[chosen]),
    }
