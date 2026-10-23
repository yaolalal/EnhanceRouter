from __future__ import annotations


def audit_decision_rows(rows: list[dict], *, selected_trust: str) -> dict:
    if not rows:
        raise ValueError("Decision audit requires non-empty rows")
    if selected_trust not in {"factorized", "direct"}:
        raise ValueError("selected_trust must be factorized or direct")
    node_ids = [int(row["node_id"]) for row in rows]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Decision rows contain duplicate node ids")
    router_correct = 0
    gnn_correct = 0
    llm_correct = 0
    oracle_correct = 0
    for row in rows:
        route = row["route"]
        if route not in {"gnn", "llm"}:
            raise ValueError(f"Unknown route: {route}")
        expected_prediction = int(row["gnn_pred"] if route == "gnn" else row["llm_pred"])
        if int(row["selected_pred"]) != expected_prediction:
            raise ValueError("selected_pred is inconsistent with route")
        expected_correct = expected_prediction == int(row["true_label"])
        if bool(row["correct"]) != expected_correct:
            raise ValueError("correct flag is inconsistent with selected_pred")
        factor = float(row["factorized_trust"])
        direct = float(row["direct_trust"])
        selected = float(row["selected_trust"])
        if not all(0.0 <= value <= 1.0 for value in (factor, direct, selected)):
            raise ValueError("Trust probability is outside [0,1]")
        expected_trust = factor if selected_trust == "factorized" else direct
        if abs(selected - expected_trust) > 1e-6:
            raise ValueError("selected_trust does not match its checkpoint")
        g_ok = int(row["gnn_pred"]) == int(row["true_label"])
        l_ok = int(row["llm_pred"]) == int(row["true_label"])
        router_correct += int(expected_correct)
        gnn_correct += int(g_ok)
        llm_correct += int(l_ok)
        oracle_correct += int(g_ok or l_ok)
    count = len(rows)
    return {
        "count": count,
        "router_accuracy": router_correct / count,
        "gnn_accuracy": gnn_correct / count,
        "llm_accuracy": llm_correct / count,
        "best_single_accuracy": max(gnn_correct, llm_correct) / count,
        "oracle_accuracy": oracle_correct / count,
        "all_checks_pass": True,
    }
