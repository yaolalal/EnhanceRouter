from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


class NeutralCacheMismatchError(ValueError):
    pass


class _ValueLockedFrozenLlmChunkedScore(torch.autograd.Function):
    """Exact configured-batch values with memory-bounded frozen-LLM VJPs.

    Forward scoring runs without an activation graph.  Backward recomputes one
    label microbatch at a time and applies the incoming class gradient.  This
    preserves the cache/inference value semantics while keeping the frozen 8B
    LLM backward graph within the verified 24GB memory bound.
    """

    @staticmethod
    def forward(
        ctx, struct_vec, backbone, text, value_batch_size,
        gradient_microbatch_size,
    ):
        batch_size = max(int(value_batch_size), 1)
        ctx.backbone = backbone
        ctx.text = text
        ctx.gradient_microbatch_size = max(int(gradient_microbatch_size), 1)
        ctx.save_for_backward(struct_vec.detach())
        with torch.no_grad():
            pieces = []
            class_count = len(backbone.label_names)
            for start in range(0, class_count, batch_size):
                end = min(start + batch_size, class_count)
                indices = torch.arange(
                    start, end, dtype=torch.long, device=struct_vec.device,
                )
                pieces.append(backbone._score_full_one_label_indices(
                    text,
                    struct_vec.detach(),
                    indices,
                    label_batch_size=batch_size,
                ))
            return torch.cat(pieces, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        (saved_vector,) = ctx.saved_tensors
        backbone = ctx.backbone
        class_count = len(backbone.label_names)
        batch_size = ctx.gradient_microbatch_size
        total_gradient = torch.zeros_like(saved_vector)
        for start in range(0, class_count, batch_size):
            end = min(start + batch_size, class_count)
            vector = saved_vector.detach().requires_grad_(True)
            indices = torch.arange(
                start, end, dtype=torch.long, device=vector.device,
            )
            with torch.enable_grad():
                scores = backbone._score_full_one_label_indices(
                    ctx.text, vector, indices, label_batch_size=batch_size,
                )
                (chunk_gradient,) = torch.autograd.grad(
                    scores,
                    vector,
                    grad_outputs=grad_output[start:end].to(scores.dtype),
                    retain_graph=False,
                    create_graph=False,
                )
            total_gradient.add_(chunk_gradient.detach())
        return total_gradient, None, None, None, None


@dataclass
class JointOutput:
    logits: torch.Tensor
    graph_text_logits: torch.Tensor
    residual: torch.Tensor
    gamma: torch.Tensor


def validate_neutral_cache_metadata(expected: dict, actual: dict) -> None:
    for key, expected_value in expected.items():
        if key not in actual or actual[key] != expected_value:
            raise NeutralCacheMismatchError(
                f"neutral cache metadata mismatch for {key}: "
                f"expected {expected_value!r}, got {actual.get(key)!r}"
            )


class AnchoredResidualJointClassifier(nn.Module):
    """B-anchored residual head around an isolated frozen-LLM Joint backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        gate_dim: int = 10,
        gamma_max: float = 0.5,
        gradient_label_pair_batch_size: int = 1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.gamma_max = float(gamma_max)
        self.gradient_label_pair_batch_size = max(
            int(gradient_label_pair_batch_size), 1,
        )
        backbone_device = backbone.llm.get_input_embeddings().weight.device
        self.correction_gate = nn.Linear(int(gate_dim), 1).to(backbone_device)
        nn.init.zeros_(self.correction_gate.weight)
        nn.init.zeros_(self.correction_gate.bias)
        input_weight = backbone.llm.get_input_embeddings().weight.detach().float()
        self.register_buffer("_neutral_vector", input_weight.mean(dim=0), persistent=True)

    @classmethod
    def from_joint_config(
        cls,
        *,
        gate_dim: int = 10,
        gamma_max: float = 0.5,
        gradient_label_pair_batch_size: int = 1,
        **joint_kwargs,
    ) -> "AnchoredResidualJointClassifier":
        from .backbone import IsolatedJointGraphTextBackbone

        return cls(
            IsolatedJointGraphTextBackbone(**joint_kwargs),
            gate_dim=gate_dim,
            gamma_max=gamma_max,
            gradient_label_pair_batch_size=gradient_label_pair_batch_size,
        )

    def neutral_vector(self) -> torch.Tensor:
        return self._neutral_vector.detach()

    def graph_text_logits(
        self,
        token_ids: torch.Tensor,
        texts: Sequence[str],
        gnn_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            torch.is_grad_enabled()
            and bool(getattr(self.backbone, "freeze_llm", False))
            and hasattr(self.backbone, "_structural_vectors")
            and hasattr(self.backbone, "_score_full_one_label_indices")
        ):
            return self._frozen_llm_gradient_cached_logits(token_ids, texts, gnn_embeddings)
        return self.backbone(token_ids, texts, gnn_embeddings)

    def _frozen_llm_gradient_cached_logits(
        self,
        token_ids: torch.Tensor,
        texts: Sequence[str],
        gnn_embeddings: torch.Tensor | None,
    ) -> torch.Tensor:
        """Exact configured-batch values and VJPs with one live label chunk."""
        llm = getattr(self.backbone, "llm", None)
        if llm is None or any(parameter.requires_grad for parameter in llm.parameters()):
            raise RuntimeError("gradient caching requires a fully frozen LLM")
        if len(texts) != int(token_ids.numel()):
            raise ValueError("token_ids and texts batch size do not match")
        vectors = self.backbone._structural_vectors(token_ids, gnn_embeddings)
        class_count = len(self.backbone.label_names)
        label_batch_size = max(
            int(getattr(self.backbone, "label_pair_batch_size", 1)), 1,
        )
        rows = []
        for text, vector in zip(texts, vectors):
            rows.append(_ValueLockedFrozenLlmChunkedScore.apply(
                vector,
                self.backbone,
                text,
                label_batch_size,
                self.gradient_label_pair_batch_size,
            ))
        return torch.stack(rows) if rows else torch.empty(
            (0, class_count), device=vectors.device, dtype=vectors.dtype,
        )

    def score_neutral(self, texts: Sequence[str]) -> torch.Tensor:
        """Score identical prompts with the one fixed neutral structural vector."""
        if not hasattr(self.backbone, "_score_full_one"):
            raise TypeError("backbone does not expose complete label-string scoring")
        batch_size = int(getattr(self.backbone, "label_pair_batch_size", 1))
        vector = self.neutral_vector().to(self.backbone.device)
        rows = [
            self.backbone._score_full_one(text, vector, label_batch_size=batch_size)
            for text in texts
        ]
        class_count = len(self.backbone.label_names)
        return torch.stack(rows) if rows else torch.empty((0, class_count), device=vector.device)

    @staticmethod
    def residual_logits(
        graph_text_logits: torch.Tensor,
        neutral_logits: torch.Tensor,
    ) -> torch.Tensor:
        graph = torch.as_tensor(graph_text_logits)
        neutral = torch.as_tensor(neutral_logits, device=graph.device, dtype=graph.dtype)
        if graph.shape != neutral.shape or graph.ndim != 2:
            raise ValueError("graph and neutral logits must have identical [N,C] shapes")
        difference = graph - neutral
        return difference - difference.mean(dim=-1, keepdim=True)

    def forward(
        self,
        token_ids: torch.Tensor,
        texts: Sequence[str],
        gnn_embeddings: torch.Tensor | None,
        anchor_logits: torch.Tensor,
        gate_features: torch.Tensor,
        neutral_logits: torch.Tensor,
        runtime_neutral_logits: torch.Tensor | None = None,
    ) -> JointOutput:
        graph_logits = self.graph_text_logits(token_ids, texts, gnn_embeddings)
        cached_neutral = torch.as_tensor(
            neutral_logits, device=graph_logits.device, dtype=graph_logits.dtype,
        )
        if cached_neutral.shape != graph_logits.shape or not torch.isfinite(
            cached_neutral
        ).all():
            raise ValueError("cached neutral logits must be finite and match graph logits")
        if runtime_neutral_logits is None:
            with torch.no_grad():
                runtime_neutral = self.score_neutral(texts).to(
                    device=graph_logits.device, dtype=graph_logits.dtype,
                )
        else:
            runtime_neutral = torch.as_tensor(
                runtime_neutral_logits,
                device=graph_logits.device, dtype=graph_logits.dtype,
            )
            if runtime_neutral.shape != graph_logits.shape or not torch.isfinite(
                runtime_neutral
            ).all():
                raise ValueError(
                    "runtime neutral logits must be finite and match graph logits"
                )
        residual = self.residual_logits(graph_logits, runtime_neutral)
        anchor = torch.as_tensor(anchor_logits, device=residual.device, dtype=residual.dtype)
        features = torch.as_tensor(
            gate_features,
            device=self.correction_gate.weight.device,
            dtype=self.correction_gate.weight.dtype,
        )
        if anchor.shape != residual.shape or features.size(0) != residual.size(0):
            raise ValueError("anchor/gate batch dimensions do not match residual logits")
        gamma = self.gamma_max * torch.tanh(self.correction_gate(features).squeeze(-1))
        logits = anchor + gamma.to(anchor.dtype).unsqueeze(-1) * residual
        return JointOutput(logits, graph_logits, residual, gamma)
