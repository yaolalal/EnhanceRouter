from __future__ import annotations

from dataclasses import replace

import torch

from .features import beta_accuracy_context, leave_one_out_accuracy_context
from .replay import EnvironmentEvidence


_TENSOR_FIELDS = (
    "local_features",
    "gnn_prob",
    "llm_prob",
    "gnn_pred",
    "llm_pred",
    "gnn_correct",
    "llm_correct",
    "advantage",
)


def combine_fold_evidence(
    fold_parts: list[EnvironmentEvidence], *, support_ids: list[int]
) -> EnvironmentEvidence:
    if not fold_parts:
        raise ValueError("at least one fold evidence part is required")
    by_node: dict[int, tuple[EnvironmentEvidence, int]] = {}
    for part in fold_parts:
        if part.local_feature_names != fold_parts[0].local_feature_names:
            raise ValueError("fold local feature names differ")
        for position, node_id in enumerate(part.node_ids.tolist()):
            if int(node_id) in by_node:
                raise ValueError(f"duplicate support node across folds: {node_id}")
            by_node[int(node_id)] = (part, position)
    expected = [int(node_id) for node_id in support_ids]
    if set(by_node) != set(expected) or len(expected) != len(set(expected)):
        raise ValueError("fold support coverage mismatch")
    stacked: dict[str, list[torch.Tensor]] = {name: [] for name in _TENSOR_FIELDS}
    for node_id in expected:
        part, position = by_node[node_id]
        for name in _TENSOR_FIELDS:
            stacked[name].append(getattr(part, name)[position])
    values = {name: torch.stack(items) for name, items in stacked.items()}
    context, context_names = leave_one_out_accuracy_context(
        values["gnn_correct"], values["llm_correct"]
    )
    return EnvironmentEvidence(
        node_ids=torch.tensor(expected, dtype=torch.long),
        context_features=context,
        local_feature_names=list(fold_parts[0].local_feature_names),
        context_feature_names=context_names,
        **values,
    )


def contextualize_queries(
    evidence: EnvironmentEvidence, support: EnvironmentEvidence
) -> EnvironmentEvidence:
    context, names = beta_accuracy_context(support.gnn_correct, support.llm_correct)
    expanded = context.view(1, -1).expand(evidence.node_ids.numel(), -1).clone()
    return replace(
        evidence,
        context_features=expanded,
        context_feature_names=names,
    )


def selective_replay(
    clean: EnvironmentEvidence,
    shifted: EnvironmentEvidence,
    *,
    shifted_node_ids: list[int] | set[int],
) -> EnvironmentEvidence:
    """Compose a node-level counterfactual while keeping unassigned rows exact."""
    if not torch.equal(clean.node_ids.cpu(), shifted.node_ids.cpu()):
        raise ValueError("clean and shifted replay node IDs/order differ")
    if clean.local_feature_names != shifted.local_feature_names:
        raise ValueError("clean and shifted local feature schemas differ")
    available = {int(value) for value in clean.node_ids.tolist()}
    selected = {int(value) for value in shifted_node_ids} & available
    mask = torch.tensor(
        [int(value) in selected for value in clean.node_ids.tolist()],
        dtype=torch.bool,
        device=clean.local_features.device,
    )

    def choose(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_value = torch.as_tensor(left)
        right_value = torch.as_tensor(right, device=left_value.device)
        if left_value.shape != right_value.shape or left_value.size(0) != mask.numel():
            raise ValueError("clean and shifted replay tensors differ")
        field_mask = mask.to(left_value.device)
        view = field_mask.view(field_mask.numel(), *([1] * (left_value.ndim - 1)))
        return torch.where(view, right_value, left_value)

    values = {
        name: choose(getattr(clean, name), getattr(shifted, name))
        for name in _TENSOR_FIELDS
    }
    return replace(
        clean,
        context_features=choose(clean.context_features, shifted.context_features),
        **values,
    )
