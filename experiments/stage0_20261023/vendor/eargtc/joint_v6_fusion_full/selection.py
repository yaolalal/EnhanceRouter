"""Exhaustive validation-only fusion over the registered Joint-v6 candidates."""
from __future__ import annotations

import math

import torch

from eargtc.joint_v6_unified.experiment import require_slurm, score


ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
BRANCHES = ("g", "l")
CANDIDATES_PER_BRANCH = 73
SCALES = (0.25, 0.5, 1.0)


def _validate_inputs(candidates, targets, incumbent):
    if not isinstance(candidates, dict) or set(candidates) != set(BRANCHES):
        raise ValueError("Only GNN and LLM candidate branches are permitted")
    if (not torch.is_tensor(targets) or targets.device.type != "cpu"
            or targets.dtype != torch.long or targets.ndim != 1 or not len(targets)):
        raise ValueError("Validation targets must be a nonempty CPU torch.long vector")
    if not torch.is_tensor(incumbent) or incumbent.device.type != "cpu":
        raise ValueError("Incumbent probability must be a CPU tensor")
    baseline = score(incumbent, targets)

    branch_config_ids = {}
    for branch in BRANCHES:
        values = candidates[branch]
        if not isinstance(values, list) or len(values) != CANDIDATES_PER_BRANCH:
            raise ValueError(
                f"Registered {branch} candidate population must contain exactly "
                f"{CANDIDATES_PER_BRANCH} entries"
            )
        identities = []
        raw_count = 0
        residual_combinations = set()
        for candidate in values:
            if not isinstance(candidate, dict) or set(candidate) != {"id", "probability", "metadata"}:
                raise ValueError("Invalid candidate schema")
            identity = candidate["id"]
            if not isinstance(identity, str) or not identity:
                raise ValueError("Invalid candidate identity")
            identities.append(identity)
            probability = candidate["probability"]
            if (not torch.is_tensor(probability) or probability.device.type != "cpu"
                    or probability.shape != incumbent.shape):
                raise ValueError("Candidate probability shape or device differs from incumbent")
            score(probability, targets)

            metadata = candidate["metadata"]
            if (not isinstance(metadata, dict) or type(metadata.get("raw")) is not bool
                    or metadata.get("branch") != branch):
                raise ValueError("Candidate branch metadata is invalid")
            if metadata["raw"]:
                raw_count += 1
                if set(metadata) != {"raw", "branch"} or identity != f"{branch}/raw":
                    raise ValueError("Raw candidate identity or metadata is invalid")
            else:
                if set(metadata) != {"raw", "branch", "config_id", "criterion", "scale"}:
                    raise ValueError("Residual candidate metadata is invalid")
                if (not isinstance(metadata["config_id"], str) or not metadata["config_id"]
                        or metadata["criterion"] not in ("accuracy", "nll")
                        or not isinstance(metadata["scale"], (int, float))
                        or isinstance(metadata["scale"], bool)
                        or not math.isfinite(float(metadata["scale"]))
                        or float(metadata["scale"]) not in SCALES):
                    raise ValueError("Residual candidate metadata is invalid")
                config_id = metadata["config_id"]
                criterion = metadata["criterion"]
                scale = float(metadata["scale"])
                expected_id = f"{branch}/{config_id}/{criterion}/{scale:g}"
                if identity != expected_id:
                    raise ValueError("Residual candidate identity differs from metadata")
                combination = (
                    metadata["config_id"], metadata["criterion"], float(metadata["scale"])
                )
                if combination in residual_combinations:
                    raise ValueError("Duplicate residual candidate combination")
                residual_combinations.add(combination)
        if len(set(identities)) != len(identities):
            raise ValueError("Invalid candidate identities")
        if raw_count != 1:
            raise ValueError("Exactly one raw expert is required per branch")
        config_ids = {combination[0] for combination in residual_combinations}
        if len(config_ids) != 12:
            raise ValueError("Residual population must contain exactly 12 config IDs")
        expected_combinations = {
            (config_id, criterion, scale)
            for config_id in config_ids
            for criterion in ("accuracy", "nll")
            for scale in SCALES
        }
        if residual_combinations != expected_combinations:
            raise ValueError("Residual candidate cross is incomplete")
        branch_config_ids[branch] = config_ids
    if branch_config_ids["g"] != branch_config_ids["l"]:
        raise ValueError("GNN and LLM candidate config sets differ")
    return baseline


def _trial(g, l, alpha, targets):
    probability = alpha * g["probability"] + (1.0 - alpha) * l["probability"]
    active_heads = int(alpha > 0 and not g["metadata"]["raw"])
    active_heads += int(alpha < 1 and not l["metadata"]["raw"])
    return {
        "g_id": g["id"],
        "l_id": l["id"],
        "alpha": alpha,
        "active_heads": active_heads,
        **score(probability, targets),
    }


def select_full(candidates, targets, incumbent):
    """Enumerate all 73 x 73 x 5 registered validation fusion trials."""
    require_slurm()
    baseline = _validate_inputs(candidates, targets, incumbent)
    trials = [
        _trial(g, l, alpha, targets)
        for g in candidates["g"]
        for l in candidates["l"]
        for alpha in ALPHAS
    ]
    best = min(
        trials,
        key=lambda trial: (
            -trial["accuracy"],
            trial["nll"],
            trial["active_heads"],
            trial["g_id"],
            trial["l_id"],
            trial["alpha"],
        ),
    )
    winner = "incumbent" if (
        -baseline["accuracy"], baseline["nll"]
    ) <= (-best["accuracy"], best["nll"]) else "repair"
    selected = baseline if winner == "incumbent" else {
        "accuracy": best["accuracy"], "nll": best["nll"]
    }
    return {
        "winner": winner,
        **selected,
        "g_id": best["g_id"],
        "l_id": best["l_id"],
        "alpha": best["alpha"],
        "active_heads": 0 if winner == "incumbent" else best["active_heads"],
        "best_repair": best,
        "incumbent_score": baseline,
        "shortlist": {
            branch: [candidate["id"] for candidate in candidates[branch]]
            for branch in BRANCHES
        },
        "trials": trials,
    }


__all__ = ["ALPHAS", "select_full"]
