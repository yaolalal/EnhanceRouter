from __future__ import annotations

import math

from .meta import PreferenceConfig


def preference_config_grid() -> list[PreferenceConfig]:
    return [
        PreferenceConfig(l2, rank_weight, rank, prior)
        for l2 in (0.01, 0.1, 1.0, 10.0)
        for rank_weight in (0.0, 0.25, 1.0, 4.0)
        for rank in (0, 1, 2)
        for prior in (0.1, 1.0, 10.0, 100.0)
    ]


def _family_match(row: dict, family: str) -> bool:
    ranked = float(row["rank_weight"]) > 0.0
    adapted = int(row["adapter_rank"]) > 0
    matches = {
        "baseline": not ranked and not adapted,
        "ranking": ranked and not adapted,
        "adapter": not ranked and adapted,
        "combined": ranked and adapted,
    }
    if family not in matches:
        raise ValueError(f"Unknown preference family: {family}")
    return matches[family]


def best_trial_for_family(
    trials: list[dict],
    family: str,
    *,
    objective: str = "auc",
    max_macro_nll: float = 0.8,
) -> dict:
    eligible = [row for row in trials if _family_match(row, family)]
    if not eligible:
        raise ValueError(f"No source-CV trials for family {family}")
    if objective == "stable_auc":
        stable = [row for row in eligible if row["macro_nll"] <= float(max_macro_nll)]
        eligible = stable if stable else eligible
        objective = "auc"
    if objective == "auc":
        key = lambda row: (
            -(row["macro_auroc"] if row["macro_auroc"] is not None else -1.0),
            row["macro_nll"],
            row["adapter_rank"],
            row["rank_weight"],
            row["l2"],
            row["prior_precision"],
        )
    elif objective == "nll":
        key = lambda row: (
            row["macro_nll"],
            -(row["macro_auroc"] if row["macro_auroc"] is not None else -1.0),
            row["adapter_rank"],
            row["rank_weight"],
            row["l2"],
            row["prior_precision"],
        )
    else:
        raise ValueError(f"Unknown source selection objective: {objective}")
    best = min(eligible, key=key)
    fields = ("l2", "rank_weight", "adapter_rank", "prior_precision", "macro_auroc", "macro_nll")
    return {name: best[name] for name in fields}


def validate_selection_record(value) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if (lowered.startswith("test") or "test_" in lowered) and child is not False:
                raise ValueError(f"Selection record contains test-derived field: {key}")
            validate_selection_record(child)
    elif isinstance(value, list):
        for child in value:
            validate_selection_record(child)


def exact_one_sided_binomial_p(wins: int, losses: int) -> float:
    wins, losses = int(wins), int(losses)
    if wins < 0 or losses < 0:
        raise ValueError("wins and losses must be non-negative")
    total = wins + losses
    if total == 0:
        return 1.0
    return sum(math.comb(total, value) for value in range(wins, total + 1)) / (1 << total)
