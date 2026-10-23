from __future__ import annotations

import math


class AuditError(RuntimeError):
    pass


_DIRECT_CONTRACT = {
    "method": "direct_graph_conditioned_number_logits",
    "loss_weights": {
        "gold_ce": 1.0,
        "hard_ce": 1.0,
        "agreement_kd": 0.1,
        "conflict_kd": 0.05,
    },
    "high_precision_target": 0.85,
    "medium_precision_target": 0.70,
    "prediction_semantics": "direct_last_position_number_token_logits",
    "logits_to_keep": 1,
    "scheme_b_role": "pseudo_label_selection_only",
    "scheme_c_role": "high_medium_abstain_assignment_only",
    "anchor_used": False,
    "neutral_branch_used": False,
    "difference_residual_used": False,
    "candidate_class_loop_used": False,
    "test_labels_available_to_process": False,
    "selection_data": "official_validation_only",
}


def validate_direct_config(config: dict, *, allow_smoke: bool = False) -> None:
    for key, expected in _DIRECT_CONTRACT.items():
        if config.get(key) != expected:
            raise AuditError(
                f"direct-numbered config mismatch for {key}: "
                f"expected {expected!r}, observed {config.get(key)!r}"
            )
    if config.get("smoke") is not False and not allow_smoke:
        raise AuditError("full result is marked as smoke")
    high = config.get("tau_high", {})
    medium = config.get("tau_medium", {})
    if not all(
        isinstance(row, dict)
        and int(row.get("selected_count", 0)) > 0
        and math.isfinite(float(row.get("threshold", float("nan"))))
        for row in (high, medium)
    ):
        raise AuditError("validation-derived trust thresholds are invalid")
    if float(high["threshold"]) < float(medium["threshold"]):
        raise AuditError("high threshold is below medium threshold")


def audit_evaluation_payload(payload: dict, *, expected_count: int) -> None:
    if (
        payload.get("all_checks_pass") is not True
        or payload.get("training_artifact_contains_targets") is not False
        or int(payload.get("count", -1)) != int(expected_count)
    ):
        raise AuditError("evaluation completion/count/target-isolation contract failed")
    try:
        metrics = payload["metrics"]
        b_accuracy = float(metrics["Scheme-B"]["accuracy"])
        joint_accuracy = float(metrics["Joint-Direct-Numbered"]["accuracy"])
        paired = payload["correction_damage"]
        count = int(paired["count"])
        corrections = int(paired["corrections"])
        damages = int(paired["damages"])
        partition = corrections + damages + int(paired["stable_correct"]) + int(
            paired["stable_wrong"]
        )
        reported = float(payload["gain_over_scheme_b"])
        paired_gain = float(paired["gain"])
        exact_p = float(paired["paired_exact_p"])
        interval = [float(value) for value in paired["paired_bootstrap_ci95"]]
    except (KeyError, TypeError, ValueError) as error:
        raise AuditError("evaluation payload schema is incomplete") from error
    expected_gain = (corrections - damages) / count
    values = (reported, paired_gain, joint_accuracy - b_accuracy, expected_gain)
    if count != expected_count or partition != count or not all(
        math.isclose(value, expected_gain, rel_tol=1e-6, abs_tol=1e-7)
        for value in values
    ):
        raise AuditError("evaluation correction/damage gain identity failed")
    if not 0.0 <= exact_p <= 1.0 or len(interval) != 2 or interval[0] > interval[1]:
        raise AuditError("evaluation paired statistics are invalid")
