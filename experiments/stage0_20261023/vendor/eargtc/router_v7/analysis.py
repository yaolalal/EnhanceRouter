from __future__ import annotations


def _reject_test_fields(value, path: str = "") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered.startswith("test") or "_test" in lowered or "test_" in lowered:
                raise ValueError(f"selection candidate contains test-derived field: {path}{key}")
            _reject_test_fields(child, f"{path}{key}.")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_test_fields(child, f"{path}{index}.")


def select_validation_candidate(
    candidates: list[dict], *, regret_tolerance: float = 0.005
) -> dict:
    if not candidates:
        raise ValueError("at least one Router-v7 candidate is required")
    if float(regret_tolerance) < 0:
        raise ValueError("regret_tolerance must be non-negative")
    _reject_test_fields(candidates)
    for candidate in candidates:
        validation = candidate.get("validation", {})
        if "flip_accuracy" not in validation or "worst_regret" not in validation:
            raise ValueError("candidate lacks validation flip_accuracy/worst_regret")
    best_regret = min(
        float(candidate["validation"]["worst_regret"])
        for candidate in candidates
    )
    feasible = [
        candidate for candidate in candidates
        if float(candidate["validation"]["worst_regret"])
        <= best_regret + float(regret_tolerance) + 1e-12
    ]
    return min(
        feasible,
        key=lambda row: (
            -float(row["validation"].get("response_accuracy", row["validation"]["flip_accuracy"])),
            -float(row["validation"]["flip_accuracy"]),
            float(row["validation"]["worst_regret"]),
            int(row.get("complexity", 0)),
            str(row["name"]),
        ),
    )


def evaluate_claim_gates(
    *,
    paired_vs_erm_flip_ci: list[float],
    v7_vs_v6_regret_ci: list[float],
    clean_accuracy_delta: float,
    clean_tolerance: float = 0.01,
) -> dict:
    if len(paired_vs_erm_flip_ci) != 2 or len(v7_vs_v6_regret_ci) != 2:
        raise ValueError("claim confidence intervals must have two endpoints")
    mechanism = float(paired_vs_erm_flip_ci[0]) > 0.0
    robust = float(v7_vs_v6_regret_ci[0]) > 0.0
    clean = float(clean_accuracy_delta) >= -float(clean_tolerance)
    return {
        "mechanism": mechanism,
        "robust": robust,
        "clean_noninferiority": clean,
        "overall": mechanism and robust and clean,
    }
