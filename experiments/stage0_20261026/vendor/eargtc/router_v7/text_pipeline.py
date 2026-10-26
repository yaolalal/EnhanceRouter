from __future__ import annotations

import hashlib

import torch

from .perturbations import delete_span, mask_tokens, mixed_assignment, truncate_text
from .schemas import EnvironmentSpec


def align_text_replay(
    *,
    frozen_clean: torch.Tensor,
    runtime_clean: torch.Tensor,
    runtime_shifted: torch.Tensor,
    changed_mask: torch.Tensor,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Transfer only the same-runtime text intervention onto frozen clean logits.

    The log-probability residual removes software/hardware replay drift. Rows whose
    input text was not changed are overwritten with the frozen clean distribution
    exactly, which makes the node-level isolation contract auditable bit-for-bit.
    """
    if frozen_clean.shape != runtime_clean.shape or frozen_clean.shape != runtime_shifted.shape:
        raise ValueError("clean and shifted probability tensors must have identical shapes")
    if frozen_clean.ndim != 2:
        raise ValueError("probability tensors must have shape [nodes, classes]")
    changed_mask = torch.as_tensor(changed_mask, dtype=torch.bool, device=frozen_clean.device)
    if changed_mask.shape != (frozen_clean.shape[0],):
        raise ValueError("changed_mask must contain one value per node")
    tensors = (frozen_clean, runtime_clean, runtime_shifted)
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("probability tensors must be finite")
    if any(bool((value < 0).any()) for value in tensors):
        raise ValueError("probability tensors must be non-negative")
    log_aligned = (
        frozen_clean.clamp_min(epsilon).log()
        + runtime_shifted.clamp_min(epsilon).log()
        - runtime_clean.clamp_min(epsilon).log()
    )
    aligned = torch.softmax(log_aligned, dim=1)
    aligned[~changed_mask] = frozen_clean[~changed_mask]
    return aligned


def _hash_order(node_ids: list[int], seed: int, split: str) -> list[int]:
    return sorted(
        {int(node_id) for node_id in node_ids},
        key=lambda node_id: hashlib.sha256(
            f"router-v7-probe:{int(seed)}:{split}:{node_id}".encode()
        ).digest(),
    )


def select_probe_nodes(
    *,
    num_nodes: int,
    train_ids: list[int],
    val_ids: list[int],
    val_cap: int,
    test_cap: int,
    seed: int,
) -> dict[str, list[int]]:
    train = sorted({int(value) for value in train_ids})
    val_set = {int(value) for value in val_ids}
    if set(train) & val_set:
        raise ValueError("probe train/validation overlap")
    if any(value < 0 or value >= int(num_nodes) for value in set(train) | val_set):
        raise ValueError("probe split contains node outside graph")
    test = [
        node_id for node_id in range(int(num_nodes))
        if node_id not in set(train) and node_id not in val_set
    ]
    selected_val = _hash_order(list(val_set), seed, "validation")[: max(0, int(val_cap))]
    selected_test = _hash_order(test, seed, "test")[: max(0, int(test_cap))]
    return {
        "train_ids": train,
        "val_ids": selected_val,
        "test_ids": selected_test,
        "node_ids": train + selected_val + selected_test,
    }


def transform_texts(texts: list[str], spec: EnvironmentSpec) -> list[str]:
    if spec.channel not in {"clean", "text", "mixed"}:
        raise ValueError("text transformation requires clean/text/mixed environment")
    assignments = mixed_assignment(range(len(texts)), seed=spec.seed)
    result = []
    for node_id, raw in enumerate(texts):
        text = str(raw or "")
        node_seed = int(spec.seed) * 1_000_003 + node_id
        apply = spec.channel != "mixed" or assignments[node_id] == "text"
        if spec.channel == "clean" or not apply:
            shifted = text
        elif spec.family == "truncate":
            shifted = truncate_text(text, spec.severity)
        elif spec.family in {"mask", "mixed_mask"}:
            shifted = mask_tokens(text, spec.severity, seed=node_seed)
        elif spec.family in {"span_delete", "section_delete"}:
            shifted = delete_span(text, spec.severity, seed=node_seed)
        else:
            raise ValueError(f"unsupported text shift family: {spec.family}")
        result.append(shifted)
    return result
