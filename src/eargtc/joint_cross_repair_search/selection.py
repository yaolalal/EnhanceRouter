"""Deterministic validation-only selection for cross-repair candidates."""
from __future__ import annotations

import math
from typing import Any

import torch

ALPHAS = (0.0, 0.5, 1.0)
BETAS = (0.0, 0.25, 0.5, 1.0)
_TOL = 1e-6


def residual_probability(base_logp: torch.Tensor, repaired_logits: torch.Tensor, scale: float) -> torch.Tensor:
    if not isinstance(base_logp, torch.Tensor) or not isinstance(repaired_logits, torch.Tensor):
        raise ValueError("residual inputs must be tensors")
    if base_logp.ndim != 2 or base_logp.shape != repaired_logits.shape or base_logp.shape[1] == 0:
        raise ValueError("residual inputs must have matching [rows, classes] shape")
    if not base_logp.is_floating_point() or not repaired_logits.is_floating_point():
        raise ValueError("residual inputs must be floating point")
    if not bool(torch.isfinite(base_logp).all() and torch.isfinite(repaired_logits).all()):
        raise ValueError("residual inputs must be finite")
    try:
        scale = float(scale)
    except (TypeError, ValueError) as error:
        raise ValueError("scale must be finite") from error
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    return torch.softmax(base_logp + scale * (repaired_logits - base_logp), dim=1)


def _probability(value: Any, name: str, shape: tuple[int, int] | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or not all(value.shape):
        raise ValueError(f"{name} must be a nonempty [rows, classes] tensor")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain finite floating-point probabilities")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} shape must match candidates")
    if bool((value < 0).any()) or not bool(torch.allclose(value.sum(1), torch.ones(value.shape[0], device=value.device, dtype=value.dtype), atol=_TOL, rtol=0.0)):
        raise ValueError(f"{name} must be normalized nonnegative probabilities")
    return value


def _targets(y: Any, shape: tuple[int, int]) -> torch.Tensor:
    if not isinstance(y, torch.Tensor) or y.ndim != 1 or y.shape[0] != shape[0]:
        raise ValueError("targets must be a rank-one tensor aligned with probabilities")
    if y.dtype == torch.bool or y.dtype.is_floating_point or y.dtype.is_complex:
        raise ValueError("targets must be integer class indices")
    if y.numel() == 0 or bool((y < 0).any()) or bool((y >= shape[1]).any()):
        raise ValueError("targets must be within the probability class range")
    return y.to(device="cpu", dtype=torch.long)


def _candidates(candidates: Any, y: Any, name: str) -> tuple[list[dict[str, Any]], torch.Tensor]:
    if not isinstance(candidates, (list, tuple)) or not candidates:
        raise ValueError(f"{name} candidates must be a nonempty list")
    checked, seen, shape = [], set(), None
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("id"), str) or not candidate["id"]:
            raise ValueError(f"{name} candidate must have a nonempty string id")
        if candidate["id"] in seen or not isinstance(candidate.get("metadata"), dict):
            raise ValueError(f"{name} candidates require unique ids and dictionary metadata")
        probability = _probability(candidate.get("probability"), f"{name} candidate probability", shape)
        shape = tuple(probability.shape)
        checked.append(candidate)
        seen.add(candidate["id"])
    return checked, _targets(y, shape)


def _score(probability: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    probability = probability.detach().to(device="cpu", dtype=torch.float64)
    rows = torch.arange(y.numel())
    return float((probability.argmax(1) == y).double().mean().item()), float(-probability[rows, y].clamp_min(1e-12).log().mean().item())


def _candidate_key(candidate: dict[str, Any], y: torch.Tensor) -> tuple[float, float, str]:
    accuracy, nll = _score(candidate["probability"], y)
    return -accuracy, nll, candidate["id"]


def choose_independent(candidates: list[dict[str, Any]], y: torch.Tensor) -> dict[str, Any]:
    checked, targets = _candidates(candidates, y, "independent")
    return min(checked, key=lambda candidate: _candidate_key(candidate, targets))


def _active(g: dict[str, Any], l: dict[str, Any], alpha: float, beta: float) -> int:
    if beta == 0.0:
        return 0
    g_repaired = bool(g["metadata"].get("active_repair", not g["metadata"].get("base", False)))
    l_repaired = bool(l["metadata"].get("active_repair", not l["metadata"].get("base", False)))
    return int(alpha > 0.0 and g_repaired) + int(alpha < 1.0 and l_repaired)


def _trials(g: dict[str, Any], l: dict[str, Any], pb: torch.Tensor, y: torch.Tensor) -> list[dict[str, Any]]:
    trials = []
    for alpha in ALPHAS:
        for beta in BETAS:
            p = (1.0 - beta) * pb + beta * (alpha * g["probability"] + (1.0 - alpha) * l["probability"])
            accuracy, nll = _score(p, y)
            trials.append({"g_id": g["id"], "l_id": l["id"], "alpha": float(alpha), "beta": float(beta),
                           "accuracy": accuracy, "nll": nll, "active_repairs": _active(g, l, alpha, beta)})
    return trials


def _trial_key(trial: dict[str, Any]) -> tuple[float, float, int, float, float, float, str, str]:
    return (-trial["accuracy"], trial["nll"], trial["active_repairs"], trial["beta"],
            abs(trial["alpha"] - 0.5), trial["alpha"], trial["g_id"], trial["l_id"])


def choose_fusion_pair(g_candidate: dict[str, Any], l_candidate: dict[str, Any], pb: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:
    g, targets = _candidates([g_candidate], y, "g")
    l, _ = _candidates([l_candidate], targets, "l")
    if tuple(g[0]["probability"].shape) != tuple(l[0]["probability"].shape):
        raise ValueError("G and L candidate probability shapes must match")
    baseline = _probability(pb, "baseline probability", tuple(g[0]["probability"].shape))
    selected = min(_trials(g[0], l[0], baseline, targets), key=_trial_key)
    return {**selected, "trial_count": len(ALPHAS) * len(BETAS)}


def _shortlist(candidates: list[dict[str, Any]], y: torch.Tensor, size: int) -> list[dict[str, Any]]:
    bases = [candidate for candidate in candidates if bool(candidate["metadata"].get("base", False))]
    if len(bases) != 1:
        raise ValueError("joint selection requires exactly one base candidate per branch")
    ranked = sorted(candidates, key=lambda candidate: _candidate_key(candidate, y))
    nll_best = min(candidates, key=lambda candidate: (_score(candidate["probability"], y)[1], candidate["id"]))
    ids = {bases[0]["id"], nll_best["id"], *(candidate["id"] for candidate in ranked[:size])}
    return sorted((candidate for candidate in candidates if candidate["id"] in ids), key=lambda candidate: candidate["id"])


def choose_joint(g_candidates: list[dict[str, Any]], l_candidates: list[dict[str, Any]], pb: torch.Tensor, y: torch.Tensor, shortlist_size: int = 4) -> dict[str, Any]:
    if isinstance(shortlist_size, bool) or not isinstance(shortlist_size, int) or shortlist_size < 1:
        raise ValueError("shortlist_size must be a positive integer")
    g, targets = _candidates(g_candidates, y, "g")
    l, _ = _candidates(l_candidates, targets, "l")
    if tuple(g[0]["probability"].shape) != tuple(l[0]["probability"].shape):
        raise ValueError("G and L candidate probability shapes must match")
    baseline = _probability(pb, "baseline probability", tuple(g[0]["probability"].shape))
    gs, ls = _shortlist(g, targets, shortlist_size), _shortlist(l, targets, shortlist_size)
    trials = [trial for gc in gs for lc in ls for trial in _trials(gc, lc, baseline, targets)]
    selected = min(trials, key=_trial_key)
    return {**selected, "trial_count": len(trials),
            "g_shortlist_ids": [candidate["id"] for candidate in gs],
            "l_shortlist_ids": [candidate["id"] for candidate in ls]}


__all__ = ["ALPHAS", "BETAS", "residual_probability", "choose_independent", "choose_fusion_pair", "choose_joint"]
