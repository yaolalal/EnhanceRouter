from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch


def _optional_mean(mask: torch.Tensor, value: torch.Tensor) -> float | None:
    return float(value[mask].float().mean()) if bool(mask.any()) else None


def evaluate_environment(
    *,
    labels: torch.Tensor,
    gnn_pred: torch.Tensor,
    llm_pred: torch.Tensor,
    preference_probability: torch.Tensor,
    threshold: float,
) -> dict:
    y = torch.as_tensor(labels).view(-1)
    g = torch.as_tensor(gnn_pred).view(-1)
    l = torch.as_tensor(llm_pred).view(-1)
    p = torch.as_tensor(preference_probability).view(-1)
    if not (y.numel() == g.numel() == l.numel() == p.numel()) or y.numel() == 0:
        raise ValueError("environment metric inputs must be equal and non-empty")
    choose_g = p >= float(threshold)
    selected = torch.where(choose_g, g, l)
    g_correct, l_correct = g == y, l == y
    router_correct = selected == y
    oracle_correct = g_correct | l_correct
    g_only = g_correct & ~l_correct
    l_only = l_correct & ~g_correct
    router_accuracy = float(router_correct.float().mean())
    oracle_accuracy = float(oracle_correct.float().mean())
    return {
        "count": int(y.numel()),
        "router_accuracy": router_accuracy,
        "gnn_accuracy": float(g_correct.float().mean()),
        "llm_accuracy": float(l_correct.float().mean()),
        "best_single_accuracy": max(float(g_correct.float().mean()), float(l_correct.float().mean())),
        "oracle_accuracy": oracle_accuracy,
        "oracle_regret": oracle_accuracy - router_accuracy,
        "gnn_only_count": int(g_only.sum()),
        "llm_only_count": int(l_only.sum()),
        "gnn_only_recall": _optional_mean(g_only, choose_g),
        "llm_only_recall": _optional_mean(l_only, ~choose_g),
    }


def _ranks(value: torch.Tensor) -> torch.Tensor:
    array = torch.as_tensor(value, dtype=torch.float64).view(-1)
    if array.numel() == 0:
        return array.clone()
    order = torch.argsort(array, stable=True)
    sorted_value = array[order]
    starts = torch.ones(array.numel(), dtype=torch.bool)
    starts[1:] = sorted_value[1:] != sorted_value[:-1]
    group = starts.cumsum(0) - 1
    counts = torch.bincount(group)
    first = counts.cumsum(0) - counts
    average = first.double() + 0.5 * (counts.double() - 1.0)
    result = torch.empty_like(array)
    result[order] = average[group]
    return result


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if left.numel() < 2:
        return None
    x, y = _ranks(left), _ranks(right)
    x, y = x - x.mean(), y - y.mean()
    denominator = torch.sqrt((x.square().sum()) * (y.square().sum()))
    if float(denominator) == 0:
        return None
    return float((x * y).sum() / denominator)


def competence_response_metrics(
    *,
    base_advantage: torch.Tensor,
    shifted_advantage: torch.Tensor,
    base_score: torch.Tensor,
    shifted_score: torch.Tensor,
    base_route_g: torch.Tensor,
    shifted_route_g: torch.Tensor,
    base_oracle_g: torch.Tensor,
    shifted_oracle_g: torch.Tensor,
) -> dict:
    values = [
        torch.as_tensor(value).view(-1)
        for value in (
            base_advantage, shifted_advantage, base_score, shifted_score,
            base_route_g, shifted_route_g, base_oracle_g, shifted_oracle_g,
        )
    ]
    if len({value.numel() for value in values}) != 1 or values[0].numel() == 0:
        raise ValueError("competence-response inputs must be equal and non-empty")
    delta_a = values[1].float() - values[0].float()
    delta_s = values[3].float() - values[2].float()
    material = delta_a != 0
    direction = _optional_mean(material, torch.sign(delta_a) == torch.sign(delta_s))
    base_route, shifted_route = values[4].bool(), values[5].bool()
    base_oracle, shifted_oracle = values[6].bool(), values[7].bool()
    oracle_changed = base_oracle != shifted_oracle
    route_changed = base_route != shifted_route
    true_positive = route_changed & oracle_changed
    flip_accuracy = None
    if bool(oracle_changed.any()):
        correct = torch.cat([
            base_route[oracle_changed] == base_oracle[oracle_changed],
            shifted_route[oracle_changed] == shifted_oracle[oracle_changed],
        ])
        flip_accuracy = float(correct.float().mean())
    precision = (
        float(true_positive.sum() / route_changed.sum()) if bool(route_changed.any()) else None
    )
    recall = (
        float(true_positive.sum() / oracle_changed.sum()) if bool(oracle_changed.any()) else None
    )
    non_flip = ~oracle_changed
    return {
        "count": int(delta_a.numel()),
        "delta_s_delta_advantage_spearman": _spearman(delta_s, delta_a),
        "shift_direction_accuracy": direction,
        "oracle_flip_count": int(oracle_changed.sum()),
        "oracle_flip_routing_accuracy": flip_accuracy,
        "route_switch_precision": precision,
        "route_switch_recall": recall,
        "non_flip_stability": _optional_mean(non_flip, ~route_changed),
    }


def flip_endpoint_route_score(
    *,
    base_advantage: torch.Tensor,
    shifted_advantage: torch.Tensor,
    base_probability: torch.Tensor,
    shifted_probability: torch.Tensor,
    threshold: float,
    epsilon: float = 0.1,
) -> dict[str, torch.Tensor | int]:
    """Return one same-node routing score for every material oracle flip.

    A node scores the mean correctness of its clean and shifted routing
    decisions. Keeping one value per node makes the subsequent bootstrap use
    the node, rather than the two correlated endpoints, as its sampling unit.
    """
    values = [
        torch.as_tensor(value).view(-1)
        for value in (
            base_advantage, shifted_advantage,
            base_probability, shifted_probability,
        )
    ]
    if len({value.numel() for value in values}) != 1:
        raise ValueError("flip endpoint inputs must have equal length")
    base_a, shifted_a = values[0].float(), values[1].float()
    base_oracle, shifted_oracle = base_a >= 0, shifted_a >= 0
    mask = (
        (base_oracle != shifted_oracle)
        & ((shifted_a - base_a).abs() >= float(epsilon))
    )
    base_route = values[2].float() >= float(threshold)
    shifted_route = values[3].float() >= float(threshold)
    score = 0.5 * (
        (base_route[mask] == base_oracle[mask]).float()
        + (shifted_route[mask] == shifted_oracle[mask]).float()
    )
    return {"mask": mask, "score": score, "node_count": int(mask.sum())}


def risk_coverage(
    probability: torch.Tensor,
    correct: torch.Tensor,
    *,
    coverages: tuple[float, ...] = (0.10, 0.25, 0.33, 0.50, 0.75, 1.00),
) -> dict:
    p = torch.as_tensor(probability, dtype=torch.float64).view(-1)
    y = torch.as_tensor(correct, dtype=torch.bool).view(-1)
    if p.numel() != y.numel() or p.numel() == 0:
        raise ValueError("risk-coverage inputs must be equal and non-empty")
    order = torch.argsort(p, descending=True, stable=True)
    cumulative = y[order].double().cumsum(0) / torch.arange(1, p.numel() + 1)
    precision = {}
    for coverage in coverages:
        count = max(1, min(p.numel(), int(math.ceil(float(coverage) * p.numel()))))
        precision[f"{coverage:.2f}"] = float(cumulative[count - 1])
    return {
        "aurc": float((1.0 - cumulative).mean()),
        "precision_at_coverage": precision,
    }


def cluster_bootstrap_delta(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    node_ids: torch.Tensor,
    samples: int = 5000,
    seed: int = 42,
) -> dict:
    a = torch.as_tensor(left, dtype=torch.float64).view(-1)
    b = torch.as_tensor(right, dtype=torch.float64).view(-1)
    node = torch.as_tensor(node_ids, dtype=torch.long).view(-1)
    if not (a.numel() == b.numel() == node.numel()) or a.numel() == 0:
        raise ValueError("cluster bootstrap inputs must be equal and non-empty")
    order = torch.argsort(node, stable=True)
    sorted_node = node[order]
    _, counts = torch.unique_consecutive(sorted_node, return_counts=True)
    group = torch.repeat_interleave(torch.arange(counts.numel()), counts)
    sums = torch.zeros(counts.numel(), dtype=torch.float64)
    sums.scatter_add_(0, group, (a - b)[order])
    cluster_delta = (sums / counts.double()).numpy()
    rng = np.random.default_rng(int(seed))
    sample_count = max(1, int(samples))
    values, frequencies = np.unique(cluster_delta, return_counts=True)
    if len(values) <= 256:
        draws = rng.multinomial(
            len(cluster_delta), frequencies / frequencies.sum(), size=sample_count
        )
        estimates = draws @ values / len(cluster_delta)
    else:
        estimates = np.empty(sample_count, dtype=np.float64)
        batch_size = max(1, 2_000_000 // len(cluster_delta))
        for start in range(0, sample_count, batch_size):
            stop = min(sample_count, start + batch_size)
            draw = rng.integers(
                0, len(cluster_delta), size=(stop - start, len(cluster_delta))
            )
            estimates[start:stop] = cluster_delta[draw].mean(1)
    return {
        "cluster_count": int(len(cluster_delta)),
        "delta": float(cluster_delta.mean()),
        "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
        "samples": sample_count,
        "seed": int(seed),
    }


def macro_dataset_bootstrap_delta(
    records: Mapping[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    samples: int = 5000,
    seed: int = 42,
) -> dict:
    """Two-level bootstrap: resample benchmark datasets, then node clusters."""
    if not records:
        raise ValueError("macro bootstrap requires at least one dataset")
    sample_count = max(1, int(samples))
    rng = np.random.default_rng(int(seed))
    names = sorted(records)
    deltas: dict[str, np.ndarray] = {}
    for name, (left, right, node_ids) in records.items():
        a = torch.as_tensor(left, dtype=torch.float64).view(-1)
        b = torch.as_tensor(right, dtype=torch.float64).view(-1)
        node = torch.as_tensor(node_ids, dtype=torch.long).view(-1)
        if not (a.numel() == b.numel() == node.numel()) or a.numel() == 0:
            raise ValueError(f"invalid macro bootstrap rows for dataset={name}")
        order = torch.argsort(node, stable=True)
        sorted_node = node[order]
        _, counts = torch.unique_consecutive(sorted_node, return_counts=True)
        group = torch.repeat_interleave(torch.arange(counts.numel()), counts)
        sums = torch.zeros(counts.numel(), dtype=torch.float64)
        sums.scatter_add_(0, group, (a - b)[order])
        deltas[name] = (sums / counts.double()).numpy()
    estimates = np.zeros(sample_count, dtype=np.float64)
    for _ in range(len(names)):
        selected = rng.integers(0, len(names), size=sample_count)
        for dataset_index, name in enumerate(names):
            mask = selected == dataset_index
            count = int(mask.sum())
            if count == 0:
                continue
            values, frequencies = np.unique(deltas[name], return_counts=True)
            draws = rng.multinomial(
                len(deltas[name]), frequencies / frequencies.sum(), size=count
            )
            estimates[mask] += draws @ values / len(deltas[name]) / len(names)
    point = float(np.mean([deltas[name].mean() for name in names]))
    return {
        "dataset_count": len(names),
        "delta": point,
        "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
        "samples": sample_count,
        "seed": int(seed),
    }


def final_population_comparison(
    decisions: Mapping[str, dict],
    *,
    environments: tuple[str, ...],
    samples: int = 5000,
    seed: int = 42,
    minimum_flip_nodes: int = 50,
) -> dict:
    """Compute benchmark-population paired-vs-ERM effects from final decisions."""
    if not decisions:
        raise ValueError("final population comparison requires decisions")
    output = {}
    for environment_id in environments:
        accuracy_records = {}
        flip_records = {}
        flip_counts = {}
        for dataset, record in decisions.items():
            environment = record["environments"][environment_id]
            paired, matched = environment["paired"], environment["matched_erm"]
            node_ids = torch.as_tensor(paired["node_ids"], dtype=torch.long)
            if not torch.equal(node_ids, torch.as_tensor(matched["node_ids"], dtype=torch.long)):
                raise ValueError(f"paired/ERM node mismatch: {dataset}/{environment_id}")
            for key in ("labels", "gnn_pred", "llm_pred"):
                if not torch.equal(torch.as_tensor(paired[key]), torch.as_tensor(matched[key])):
                    raise ValueError(
                        f"paired/ERM {key} mismatch: {dataset}/{environment_id}"
                    )
            labels = torch.as_tensor(paired["labels"])
            paired_selected = torch.where(
                torch.as_tensor(paired["probability"]) >= float(record["paired_threshold"]),
                torch.as_tensor(paired["gnn_pred"]), torch.as_tensor(paired["llm_pred"]),
            )
            matched_selected = torch.where(
                torch.as_tensor(matched["probability"]) >= float(record["matched_erm_threshold"]),
                torch.as_tensor(matched["gnn_pred"]), torch.as_tensor(matched["llm_pred"]),
            )
            accuracy_records[dataset] = (
                (paired_selected == labels).float(),
                (matched_selected == labels).float(),
                node_ids,
            )
            pair = record["heldout_pairs"][environment_id]
            paired_flip = flip_endpoint_route_score(
                base_advantage=pair["paired_base"]["advantage"],
                shifted_advantage=pair["paired_shifted"]["advantage"],
                base_probability=pair["paired_base"]["probability"],
                shifted_probability=pair["paired_shifted"]["probability"],
                threshold=float(record["paired_threshold"]),
            )
            matched_flip = flip_endpoint_route_score(
                base_advantage=pair["matched_erm_base"]["advantage"],
                shifted_advantage=pair["matched_erm_shifted"]["advantage"],
                base_probability=pair["matched_erm_base"]["probability"],
                shifted_probability=pair["matched_erm_shifted"]["probability"],
                threshold=float(record["matched_erm_threshold"]),
            )
            if not torch.equal(paired_flip["mask"], matched_flip["mask"]):
                raise ValueError(f"paired/ERM flip mismatch: {dataset}/{environment_id}")
            count = int(paired_flip["node_count"])
            flip_counts[dataset] = count
            if count >= int(minimum_flip_nodes):
                mask = paired_flip["mask"]
                flip_records[dataset] = (
                    paired_flip["score"], matched_flip["score"],
                    torch.as_tensor(pair["paired_shifted"]["node_ids"])[mask],
                )
        flip_result = {
            "available": False,
            "minimum_flip_nodes": int(minimum_flip_nodes),
            "dataset_flip_counts": flip_counts,
        }
        if flip_records:
            flip_result.update({
                "available": True,
                **macro_dataset_bootstrap_delta(
                    flip_records, samples=samples, seed=seed + 1000
                ),
            })
        output[environment_id] = {
            "router_accuracy": macro_dataset_bootstrap_delta(
                accuracy_records, samples=samples, seed=seed
            ),
            "flip_endpoint_routing": flip_result,
        }
    return output
