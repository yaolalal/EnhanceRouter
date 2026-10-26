from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from .pairs import build_counterfactual_pairs, node_environment_weights


def advantage_flip_record(
    base_advantage: torch.Tensor,
    shifted_advantage: torch.Tensor,
    base_route_g: torch.Tensor,
    shifted_route_g: torch.Tensor,
    *,
    epsilon: float = 0.1,
) -> dict[str, int]:
    """Score endpoints where the higher true-label-probability expert flips."""
    base = torch.as_tensor(base_advantage, dtype=torch.float32).view(-1)
    shifted = torch.as_tensor(shifted_advantage, dtype=torch.float32).view(-1)
    base_route = torch.as_tensor(base_route_g, dtype=torch.bool).view(-1)
    shifted_route = torch.as_tensor(shifted_route_g, dtype=torch.bool).view(-1)
    if len({base.numel(), shifted.numel(), base_route.numel(), shifted_route.numel()}) != 1:
        raise ValueError("advantage flip inputs must have equal length")
    base_oracle = base >= 0
    shifted_oracle = shifted >= 0
    valid = (base_oracle != shifted_oracle) & ((shifted - base).abs() >= float(epsilon))
    correct = (base_route[valid] == base_oracle[valid]).sum() + (
        shifted_route[valid] == shifted_oracle[valid]
    ).sum()
    return {"flip_correct": int(correct), "flip_total": int(2 * valid.sum())}


def _dataset_balanced_weights(rows: list[dict]) -> torch.Tensor:
    weight = node_environment_weights(rows).to(torch.float64)
    datasets = sorted({str(row["dataset"]) for row in rows})
    target_mass = len(rows) / max(len(datasets), 1)
    for dataset in datasets:
        mask = torch.tensor([str(row["dataset"]) == dataset for row in rows])
        mass = weight[mask].sum()
        if float(mass) <= 0:
            raise ValueError(f"non-positive training mass for dataset={dataset}")
        weight[mask] *= target_mass / mass
    return weight.to(torch.float32)


def build_training_batch(
    artifacts: Mapping[str, dict],
    environment_ids: Sequence[str],
    *,
    include_context: bool = True,
    feature_indices: Sequence[int] | None = None,
    pair_epsilon: float = 0.1,
) -> dict[str, object]:
    """Stack OOF support evidence without allowing query/test labels into training."""
    rows: list[dict] = []
    local, context, g_correct, l_correct = [], [], [], []
    for dataset in sorted(artifacts):
        artifact = artifacts[dataset]
        for environment_id in environment_ids:
            environment = artifact["environments"][environment_id]
            support = environment["support"]
            spec = environment["spec"]
            node_ids = torch.as_tensor(support["node_ids"]).view(-1)
            count = int(node_ids.numel())
            tensors = [
                torch.as_tensor(support[key]).view(-1)
                for key in ("gnn_correct", "llm_correct", "advantage")
            ]
            if any(value.numel() != count for value in tensors):
                raise ValueError(f"misaligned support evidence: {dataset}/{environment_id}")
            features = torch.as_tensor(support["local_features"], dtype=torch.float32)
            if feature_indices is not None:
                features = features[:, list(feature_indices)]
            ctx = torch.as_tensor(support["context_features"], dtype=torch.float32)
            if not include_context:
                ctx = ctx.new_empty((count, 0))
            if features.size(0) != count or ctx.size(0) != count:
                raise ValueError(f"misaligned feature evidence: {dataset}/{environment_id}")
            local.append(features)
            context.append(ctx)
            g_correct.append(tensors[0].bool())
            l_correct.append(tensors[1].bool())
            for index, node_id in enumerate(node_ids.tolist()):
                rows.append(
                    {
                        "dataset": dataset,
                        "node_id": int(node_id),
                        "environment_id": str(spec["environment_id"]),
                        "channel": str(spec["channel"]),
                        "severity": float(spec["severity"]),
                        "advantage": float(tensors[2][index]),
                    }
                )
    if not rows:
        raise ValueError("no Router-v7 support evidence selected")
    return {
        "rows": rows,
        "local_features": torch.cat(local),
        "context_features": torch.cat(context),
        "gnn_correct": torch.cat(g_correct),
        "llm_correct": torch.cat(l_correct),
        "trust_target": None,
        "row_weight": _dataset_balanced_weights(rows),
        "pairs": build_counterfactual_pairs(rows, epsilon=pair_epsilon),
        "clean_mask": torch.tensor(
            [row["environment_id"] == "clean" for row in rows], dtype=torch.bool
        ),
    }


def validation_summary(records: Sequence[dict]) -> dict[str, float]:
    if not records:
        raise ValueError("validation records must be non-empty")
    flip_records = [record for record in records if int(record.get("flip_total", 0)) > 0]
    direction_records = [record for record in records if int(record.get("direction_total", 0)) > 0]
    flip_total = sum(int(record["flip_total"]) for record in flip_records)
    direction_total = sum(int(record["direction_total"]) for record in direction_records)
    regrets = [float(record["oracle_regret"]) for record in records if "oracle_regret" in record]
    if not regrets:
        raise ValueError("validation records contain no oracle regret")
    return {
        "flip_accuracy": (
            sum(int(record["flip_correct"]) / int(record["flip_total"]) for record in flip_records)
            / len(flip_records) if flip_records else 0.0
        ),
        "response_accuracy": (
            sum(int(record["direction_correct"]) / int(record["direction_total"]) for record in direction_records)
            / len(direction_records) if direction_records else 0.0
        ),
        "response_count": direction_total,
        "worst_regret": max(regrets),
    }
