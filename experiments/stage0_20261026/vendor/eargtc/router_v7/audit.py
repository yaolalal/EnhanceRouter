from __future__ import annotations

from collections import Counter

import torch


def audit_channel_isolation(
    *,
    clean_gnn: torch.Tensor,
    clean_llm: torch.Tensor,
    shifted_gnn: torch.Tensor,
    shifted_llm: torch.Tensor,
    channel: str,
) -> dict:
    tensors = [torch.as_tensor(value) for value in (clean_gnn, clean_llm, shifted_gnn, shifted_llm)]
    if tensors[0].shape != tensors[2].shape or tensors[1].shape != tensors[3].shape:
        raise ValueError("clean and shifted expert shapes differ")
    if channel == "graph":
        untouched = torch.equal(tensors[1], tensors[3])
        changed_name = "gnn"
        untouched_name = "llm"
        changed = not torch.equal(tensors[0], tensors[2])
    elif channel == "text":
        untouched = torch.equal(tensors[0], tensors[2])
        changed_name = "llm"
        untouched_name = "gnn"
        changed = not torch.equal(tensors[1], tensors[3])
    else:
        raise ValueError("isolation channel must be graph or text")
    return {
        "channel": channel,
        "changed_channel": changed_name,
        "untouched_channel": untouched_name,
        "changed_channel_differs": bool(changed),
        "untouched_channel_exact_identity": bool(untouched),
        "all_checks_pass": bool(changed and untouched),
    }


def audit_counterfactual_rows(
    rows: list[dict],
    *,
    dataset: str,
    node_ids: list[int],
    environment_ids: list[str],
) -> dict:
    expected = {
        (str(dataset), int(node_id), str(environment_id))
        for node_id in node_ids
        for environment_id in environment_ids
    }
    keys = [
        (str(row["dataset"]), int(row["node_id"]), str(row["environment_id"]))
        for row in rows
    ]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate counterfactual rows: {duplicates[:3]}")
    actual = set(keys)
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    return {
        "dataset": str(dataset),
        "expected_count": len(expected),
        "actual_count": len(keys),
        "missing": missing,
        "extra": extra,
        "all_checks_pass": not missing and not extra,
    }
