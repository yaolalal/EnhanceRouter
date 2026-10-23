from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .features import (
    beta_accuracy_context,
    leave_one_out_accuracy_context,
    local_evidence,
)


@dataclass(frozen=True)
class EnvironmentEvidence:
    node_ids: torch.Tensor
    local_features: torch.Tensor
    context_features: torch.Tensor
    local_feature_names: list[str]
    context_feature_names: list[str]
    gnn_prob: torch.Tensor
    llm_prob: torch.Tensor
    gnn_pred: torch.Tensor
    llm_pred: torch.Tensor
    gnn_correct: torch.Tensor
    llm_correct: torch.Tensor
    advantage: torch.Tensor


def replay_gnn(encoder, head, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    encoder.eval()
    head.eval()
    with torch.no_grad():
        embedding = encoder(node_features, edge_index)
        logits = head(embedding)
    if logits.ndim != 2 or logits.size(0) != node_features.size(0):
        raise ValueError("GNN replay must return full-graph logits")
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("GNN replay produced non-finite logits")
    return logits.detach()


def _structural_evidence(
    edge_index: torch.Tensor, prediction: torch.Tensor, num_nodes: int, num_classes: int
) -> dict[str, torch.Tensor]:
    edge = torch.as_tensor(edge_index, dtype=torch.long, device=prediction.device)
    if edge.ndim != 2 or edge.size(0) != 2:
        raise ValueError("edge_index must have shape [2,E]")
    src, dst = edge[0], edge[1]
    degree = torch.bincount(src, minlength=num_nodes).float()
    same_count = torch.zeros(num_nodes, dtype=torch.float32, device=prediction.device)
    if src.numel():
        same_count.index_add_(0, src, (prediction[src] == prediction[dst]).float())
    agreement = torch.where(degree > 0, same_count / degree.clamp_min(1), torch.zeros_like(degree))
    class_count = torch.zeros((num_nodes, num_classes), dtype=torch.float32, device=prediction.device)
    if src.numel():
        class_count.index_put_(
            (src, prediction[dst]), torch.ones_like(src, dtype=torch.float32), accumulate=True
        )
    q = class_count / degree.clamp_min(1).unsqueeze(1)
    entropy = -(q * q.clamp_min(1e-12).log()).sum(1) / torch.log(
        torch.tensor(float(num_classes), device=prediction.device)
    )
    entropy = torch.where(degree > 0, entropy, torch.ones_like(entropy))
    return {
        "log_degree": torch.log1p(degree),
        "neighbor_gnn_agreement": agreement,
        "neighbor_gnn_entropy": entropy,
    }


def build_environment_evidence(
    *,
    node_ids: list[int],
    support_ids: list[int],
    edge_index: torch.Tensor,
    gnn_logits: torch.Tensor,
    llm_logits: torch.Tensor,
    labels: torch.Tensor,
    include_soft_geometry: bool = True,
) -> EnvironmentEvidence:
    g_logits = torch.as_tensor(gnn_logits, dtype=torch.float32)
    l_logits = torch.as_tensor(llm_logits, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long).view(-1)
    if g_logits.shape != l_logits.shape or g_logits.ndim != 2:
        raise ValueError("expert logits must be aligned [N,C] matrices")
    if g_logits.size(0) != y.numel():
        raise ValueError("expert logits and labels must cover the same graph")
    node = torch.tensor([int(value) for value in node_ids], dtype=torch.long)
    support = torch.tensor([int(value) for value in support_ids], dtype=torch.long)
    if torch.unique(node).numel() != node.numel() or torch.unique(support).numel() != support.numel():
        raise ValueError("node and support IDs must be unique")
    g_prob, l_prob = F.softmax(g_logits, 1), F.softmax(l_logits, 1)
    g_pred, l_pred = g_prob.argmax(1), l_prob.argmax(1)
    structural = _structural_evidence(
        edge_index, g_pred, g_logits.size(0), g_logits.size(1)
    )
    full_local, local_names = local_evidence(
        g_prob, l_prob, structural, include_soft_geometry=include_soft_geometry
    )
    g_correct_full, l_correct_full = g_pred == y, l_pred == y
    support_context, context_names = beta_accuracy_context(
        g_correct_full[support], l_correct_full[support]
    )
    context = support_context.view(1, -1).expand(node.numel(), -1).clone()
    if support.numel():
        loo, _ = leave_one_out_accuracy_context(
            g_correct_full[support], l_correct_full[support]
        )
        support_position = {int(node_id): index for index, node_id in enumerate(support.tolist())}
        for row, node_id in enumerate(node.tolist()):
            if node_id in support_position:
                context[row] = loo[support_position[node_id]]
    rows = torch.arange(node.numel())
    true_probability_g = g_prob[node, y[node]].clamp_min(1e-12)
    true_probability_l = l_prob[node, y[node]].clamp_min(1e-12)
    return EnvironmentEvidence(
        node_ids=node,
        local_features=full_local[node],
        context_features=context,
        local_feature_names=local_names,
        context_feature_names=context_names,
        gnn_prob=g_prob[node],
        llm_prob=l_prob[node],
        gnn_pred=g_pred[node],
        llm_pred=l_pred[node],
        gnn_correct=g_correct_full[node],
        llm_correct=l_correct_full[node],
        advantage=true_probability_g.log() - true_probability_l.log(),
    )
