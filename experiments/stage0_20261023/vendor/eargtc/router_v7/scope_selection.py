from __future__ import annotations


def selected_validation_metrics(metrics: dict, candidates: list[dict]) -> dict[str, float]:
    selected = str(metrics["selected_candidate"])
    matches = [row for row in candidates if str(row.get("name")) == selected]
    if len(matches) != 1:
        raise ValueError(f"expected one validation record for selected candidate={selected!r}")
    validation = matches[0].get("validation", {})
    if "worst_regret" not in validation or "response_accuracy" not in validation:
        raise ValueError("selected candidate lacks validation regret/response metrics")
    return {
        "worst_regret": float(validation["worst_regret"]),
        "response_accuracy": float(validation["response_accuracy"]),
    }


def choose_training_scope(
    *,
    pooled_metrics: dict,
    pooled_candidates: list[dict],
    target_metrics: dict,
    target_candidates: list[dict],
) -> dict:
    pooled = selected_validation_metrics(pooled_metrics, pooled_candidates)
    target = selected_validation_metrics(target_metrics, target_candidates)
    target_pareto_dominates = (
        target["worst_regret"] < pooled["worst_regret"]
        and target["response_accuracy"] >= pooled["response_accuracy"]
    )
    scope = "target" if target_pareto_dominates else "pooled"
    return {
        "scope": scope,
        "pooled_selected_validation": pooled,
        "target_selected_validation": target,
        "target_pareto_dominates": target_pareto_dominates,
        "tie_break": "pooled",
    }
