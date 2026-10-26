from __future__ import annotations

import math

import torch

from eargtc.router_bc.metrics import probability_metrics
from eargtc.router_bc_improved.audit import audit_decision_rows
from eargtc.router_bc_improved.runner_utils import validate_selection_record
from eargtc.router_bc_improved.trust import risk_coverage_metrics


def _close(left, right, tolerance: float) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(float(left), float(right), abs_tol=tolerance, rel_tol=0.0)


def _metrics_match(stored: dict, recomputed: dict, tolerance: float) -> bool:
    for key, value in recomputed.items():
        if key not in stored:
            return False
        actual = stored[key]
        if isinstance(value, dict):
            if not isinstance(actual, dict) or not _metrics_match(actual, value, tolerance):
                return False
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if not _close(actual, value, tolerance):
                return False
        elif actual != value:
            return False
    return True


def audit_bc_dataset(
    *,
    metrics: dict,
    decision_rows: list[dict],
    aggregate_row: dict,
    tolerance: float = 1e-6,
) -> dict:
    b_selection = metrics["scheme_b"]["selected_checkpoint"]
    c_selection = metrics["scheme_c"]["metrics"]["validation_selection"]
    validate_selection_record(b_selection)
    validate_selection_record(c_selection)
    selected_checkpoint = str(c_selection["selected"])
    recomputed = audit_decision_rows(
        decision_rows, selected_trust=selected_checkpoint
    )
    stored_b = metrics["scheme_b"]["test"]
    count = int(recomputed["count"])

    correct = torch.tensor(
        [bool(row["correct"]) for row in decision_rows], dtype=torch.bool
    )
    trust_reports = {}
    for name, field in (
        ("factorized", "factorized_trust"),
        ("direct", "direct_trust"),
    ):
        probability = torch.tensor(
            [float(row[field]) for row in decision_rows], dtype=torch.float64
        )
        trust_reports[name] = {
            **probability_metrics(probability, correct),
            **risk_coverage_metrics(probability, correct),
        }
    trust_reports["selected"] = trust_reports[selected_checkpoint]

    threshold = float(
        metrics["scheme_c"]["metrics"]["validation_threshold"]["threshold"]
    )
    selected_probability = torch.tensor(
        [float(row["selected_trust"]) for row in decision_rows],
        dtype=torch.float64,
    )
    accepted = selected_probability >= threshold
    threshold_report = {
        "selected_count": int(accepted.sum().item()),
        "coverage": float(accepted.to(torch.float64).mean().item()),
        "precision": (
            float(correct[accepted].to(torch.float64).mean().item())
            if bool(accepted.any().item())
            else None
        ),
    }

    paired = stored_b["paired_vs_best_single"]
    comparator = paired.get(
        "comparator",
        "gnn"
        if recomputed["gnn_accuracy"] >= recomputed["llm_accuracy"]
        else "llm",
    )
    comparator_correct = torch.tensor(
        [
            int(row[f"{comparator}_pred"]) == int(row["true_label"])
            for row in decision_rows
        ],
        dtype=torch.bool,
    )
    delta = correct.to(torch.int64) - comparator_correct.to(torch.int64)
    paired_recomputed = {
        "wins": int((delta == 1).sum().item()),
        "losses": int((delta == -1).sum().item()),
        "ties": int((delta == 0).sum().item()),
    }
    paired_recomputed["net_correct"] = (
        paired_recomputed["wins"] - paired_recomputed["losses"]
    )
    gain = recomputed["router_accuracy"] - recomputed["best_single_accuracy"]
    oracle_gap = recomputed["oracle_accuracy"] - recomputed["router_accuracy"]
    checks = {
        "count": count == int(metrics["counts"]["test"]),
        "router_accuracy": _close(
            recomputed["router_accuracy"], stored_b["router_accuracy"], tolerance
        ),
        "gnn_accuracy": _close(
            recomputed["gnn_accuracy"], stored_b["gnn_accuracy"], tolerance
        ),
        "llm_accuracy": _close(
            recomputed["llm_accuracy"], stored_b["llm_accuracy"], tolerance
        ),
        "best_single_accuracy": _close(
            recomputed["best_single_accuracy"],
            stored_b["best_single_accuracy"],
            tolerance,
        ),
        "oracle_accuracy": _close(
            recomputed["oracle_accuracy"], stored_b["oracle_accuracy"], tolerance
        ),
        "gain_identity": _close(gain, stored_b["gain_over_best_single"], tolerance),
        "oracle_gap_identity": _close(
            oracle_gap, stored_b["oracle_gap"], tolerance
        ),
        "beats_best_single_truthful": bool(stored_b["beats_best_single"])
        == (gain > 0.0),
        "scheme_c_route_identity": metrics["scheme_c"][
            "route_identity_with_b"
        ]
        is True,
        "paired_counts": all(
            int(paired[key]) == value for key, value in paired_recomputed.items()
        )
        and sum(paired_recomputed[key] for key in ("wins", "losses", "ties"))
        == count,
        "wins_losses_gain_identity": _close(
            paired_recomputed["net_correct"] / count, gain, tolerance
        ),
        "factorized_trust_metrics": _metrics_match(
            metrics["scheme_c"]["metrics"]["factorized"],
            trust_reports["factorized"],
            tolerance,
        ),
        "direct_trust_metrics": _metrics_match(
            metrics["scheme_c"]["metrics"]["direct"],
            trust_reports["direct"],
            tolerance,
        ),
        "selected_trust_metrics": _metrics_match(
            metrics["scheme_c"]["metrics"]["selected"],
            trust_reports["selected"],
            tolerance,
        ),
        "validation_threshold_test_metrics": _metrics_match(
            metrics["scheme_c"]["metrics"]["test_at_validation_threshold"],
            threshold_report,
            tolerance,
        ),
        "aggregate_scheme_b": _close(
            aggregate_row["scheme_b"], recomputed["router_accuracy"], tolerance
        ),
        "aggregate_best_single": _close(
            aggregate_row["best_single"],
            recomputed["best_single_accuracy"],
            tolerance,
        ),
        "aggregate_oracle": _close(
            aggregate_row["oracle"], recomputed["oracle_accuracy"], tolerance
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"corrected B/C metric audit failed: {checks}")
    return {
        "checks": checks,
        "recomputed": recomputed,
        "paired": {**paired_recomputed, "comparator": comparator},
        "beats_best_single": gain > 0.0,
        "gain_over_best_single": gain,
        "oracle_gap": oracle_gap,
        "trust_recomputed": trust_reports,
        "test_at_validation_threshold_recomputed": threshold_report,
        "all_checks_pass": True,
    }
