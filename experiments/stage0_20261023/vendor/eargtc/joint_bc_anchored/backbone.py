from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class IsolatedJointGraphTextBackbone(nn.Module):
    """Isolated minimal full-label scorer for the anchored Joint pipeline.

    This is the full-scoring subset of the frozen upstream Joint semantics.
    Prefix-cache code is deliberately excluded because every locked upstream
    dataset uses ``joint_scoring_mode=full``.
    """

    def __init__(
        self,
        model_path: str,
        codebook_size: int,
        label_names: Sequence[str],
        task_prompt: str,
        token_dim: int | None = None,
        dropout: float = 0.0,
        max_input_length: int = 2048,
        score_normalization: str = "mean",
        dtype: str = "float16",
        device_map: str | None = None,
        freeze_llm: bool = True,
        label_pair_batch_size: int = 64,
        structural_init: str = "zero",
        scoring_mode: str = "full",
        gnn_embedding_dim: int | None = None,
        residual_gate_init: float = 0.0,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.label_names = list(label_names)
        self.task_prompt = task_prompt.strip()
        self.max_input_length = int(max_input_length)
        self.label_pair_batch_size = int(label_pair_batch_size)
        self.structural_init = structural_init
        self.scoring_mode = scoring_mode
        self.score_normalization = score_normalization
        if self.score_normalization not in {"mean", "sum"}:
            raise ValueError("score_normalization must be 'mean' or 'sum'")
        if self.structural_init not in {
            "zero", "small_random", "random", "llm_manifold",
        }:
            raise ValueError(
                "structural_init must be 'zero', 'small_random', 'random', or 'llm_manifold'"
            )
        if self.scoring_mode != "full":
            raise ValueError("isolated anchored backbone supports only locked full scoring")

        torch_dtype = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.freeze_llm = bool(freeze_llm)
        if self.freeze_llm:
            for parameter in self.llm.parameters():
                parameter.requires_grad_(False)
            self.llm.eval()
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if self.gradient_checkpointing and hasattr(
            self.llm, "gradient_checkpointing_enable"
        ):
            self.llm.config.use_cache = False
            self.llm.gradient_checkpointing_enable()

        hidden_size = self.llm.get_input_embeddings().embedding_dim
        self.token_dim = hidden_size if token_dim is None else int(token_dim)
        self.token_embedding = nn.Embedding(codebook_size, self.token_dim).to(self.device)
        self.token_projection = (
            nn.Identity()
            if self.token_dim == hidden_size
            else nn.Linear(self.token_dim, hidden_size)
        ).to(self.device)
        self.gnn_embedding_dim = (
            None if gnn_embedding_dim is None else int(gnn_embedding_dim)
        )
        if self.gnn_embedding_dim is not None and self.gnn_embedding_dim > 0:
            self.residual_norm = nn.LayerNorm(self.gnn_embedding_dim).to(self.device)
            self.residual_projector = nn.Linear(
                self.gnn_embedding_dim, hidden_size
            ).to(self.device)
            self.residual_gate = nn.Parameter(
                torch.tensor(float(residual_gate_init), device=self.device)
            )
        else:
            self.residual_norm = None
            self.residual_projector = None
            self.residual_gate = None
        self._init_structural_parameters()
        self.dropout = nn.Dropout(dropout)
        self._label_ids = [
            self._ids(" " + label, add_special_tokens=False)
            or self._ids(label, add_special_tokens=False)
            for label in self.label_names
        ]
        if any(len(ids) == 0 for ids in self._label_ids):
            raise ValueError("At least one class label tokenized to an empty sequence.")

    @property
    def device(self) -> torch.device:
        return self.llm.get_input_embeddings().weight.device

    def _init_structural_parameters(self) -> None:
        if self.structural_init == "random":
            self._init_residual_parameters()
            return
        if self.structural_init == "llm_manifold":
            self._init_llm_manifold_code_tokens()
            self._init_residual_parameters()
            return
        if self.structural_init == "small_random":
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=1e-3)
            if isinstance(self.token_projection, nn.Linear):
                nn.init.normal_(self.token_projection.weight, mean=0.0, std=1e-3)
                nn.init.zeros_(self.token_projection.bias)
            self._init_residual_parameters()
            return
        if isinstance(self.token_projection, nn.Linear):
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.token_projection.weight)
            nn.init.zeros_(self.token_projection.bias)
        else:
            nn.init.zeros_(self.token_embedding.weight)
        self._init_residual_parameters()

    def _init_llm_manifold_code_tokens(self) -> None:
        input_weight = self.llm.get_input_embeddings().weight.detach().float()
        generator = torch.Generator(device=input_weight.device)
        generator.manual_seed(0)
        sample_ids = torch.randperm(
            input_weight.size(0), generator=generator, device=input_weight.device
        )[: self.token_embedding.num_embeddings]
        anchors = input_weight.index_select(0, sample_ids).to(device=self.device)
        if isinstance(self.token_projection, nn.Identity):
            self.token_embedding.weight.data.copy_(
                anchors.to(dtype=self.token_embedding.weight.dtype)
            )
            return
        mean = anchors.mean(dim=0)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.token_projection.weight, mean=0.0, std=1e-4)
        self.token_projection.bias.data.copy_(
            mean.to(device=self.device, dtype=self.token_projection.bias.dtype)
        )

    def _init_residual_parameters(self) -> None:
        if self.residual_projector is None:
            return
        nn.init.xavier_uniform_(self.residual_projector.weight)
        nn.init.zeros_(self.residual_projector.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_llm:
            self.llm.eval()
        return self

    def _prompt_parts(self, text: str) -> tuple[str, str]:
        return (
            "Structural information of the center node: ",
            f"\n{text}\n{self.task_prompt}",
        )

    def _ids(self, text: str, add_special_tokens: bool) -> List[int]:
        return self.tokenizer(
            text, add_special_tokens=add_special_tokens
        ).input_ids

    def _prompt_embeddings(
        self, text: str, struct_vec: torch.Tensor
    ) -> torch.Tensor:
        prefix, suffix = self._prompt_parts(text)
        prefix_ids = self._ids(prefix, add_special_tokens=True)
        suffix_ids = self._ids(suffix, add_special_tokens=False)
        max_label_len = max(len(ids) for ids in self._label_ids)
        suffix_budget = max(
            self.max_input_length - max_label_len - len(prefix_ids) - 1, 1
        )
        suffix_ids = suffix_ids[-suffix_budget:]
        input_embedding = self.llm.get_input_embeddings()
        struct_vec = struct_vec.to(dtype=input_embedding.weight.dtype)
        pieces = []
        if prefix_ids:
            pieces.append(
                input_embedding(
                    torch.tensor(prefix_ids, dtype=torch.long, device=self.device)
                )
            )
        pieces.append(struct_vec.view(1, -1))
        if suffix_ids:
            pieces.append(
                input_embedding(
                    torch.tensor(suffix_ids, dtype=torch.long, device=self.device)
                )
            )
        return torch.cat(pieces, dim=0)

    def _structural_vectors(
        self,
        token_ids: torch.Tensor,
        gnn_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_ids = token_ids.long().to(self.device)
        struct_vecs = self.token_projection(
            self.dropout(self.token_embedding(token_ids))
        )
        if self.residual_projector is not None and gnn_embeddings is not None:
            residual = gnn_embeddings.to(
                device=self.device, dtype=self.residual_projector.weight.dtype
            )
            residual = self.residual_projector(self.residual_norm(residual))
            struct_vecs = struct_vecs + self.residual_gate.to(
                dtype=residual.dtype
            ) * residual
        return struct_vecs

    def forward(
        self,
        token_ids: torch.Tensor,
        texts: Sequence[str],
        gnn_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.score_full(
            token_ids,
            texts,
            gnn_embeddings,
            label_batch_size=self.label_pair_batch_size,
        )

    def score_full(
        self,
        token_ids: torch.Tensor,
        texts: Sequence[str],
        gnn_embeddings: torch.Tensor | None = None,
        label_batch_size: int = 1,
    ) -> torch.Tensor:
        if len(texts) != int(token_ids.numel()):
            raise ValueError("token_ids and texts batch size do not match")
        struct_vecs = self._structural_vectors(token_ids, gnn_embeddings)
        rows = [
            self._score_full_one(
                text, struct_vec, label_batch_size=label_batch_size
            )
            for text, struct_vec in zip(texts, struct_vecs)
        ]
        return (
            torch.stack(rows, dim=0)
            if rows
            else torch.empty(0, len(self.label_names), device=self.device)
        )

    def _score_full_one(
        self,
        text: str,
        struct_vec: torch.Tensor,
        label_batch_size: int = 1,
    ) -> torch.Tensor:
        label_indices = torch.arange(
            len(self._label_ids), dtype=torch.long, device=self.device
        )
        return self._score_full_one_label_indices(
            text,
            struct_vec,
            label_indices,
            label_batch_size=label_batch_size,
        )

    def score_label_indices(
        self,
        token_id: torch.Tensor,
        text: str,
        label_indices: torch.Tensor,
        gnn_embedding: torch.Tensor | None = None,
        label_batch_size: int = 1,
    ) -> torch.Tensor:
        token_id = token_id.reshape(1).long().to(self.device)
        if gnn_embedding is not None:
            gnn_embedding = gnn_embedding.reshape(1, -1)
        struct_vec = self._structural_vectors(token_id, gnn_embedding)[0]
        return self._score_full_one_label_indices(
            text,
            struct_vec,
            label_indices,
            label_batch_size=label_batch_size,
        )

    def _score_full_one_label_indices(
        self,
        text: str,
        struct_vec: torch.Tensor,
        label_indices: torch.Tensor,
        label_batch_size: int = 1,
    ) -> torch.Tensor:
        indices = label_indices.long().detach().cpu().tolist()
        prompt = self._prompt_embeddings(text, struct_vec)
        prompt_len = prompt.size(0)
        input_embedding = self.llm.get_input_embeddings()
        label_scores = []
        batch_size = max(int(label_batch_size), 1)
        for start in range(0, len(indices), batch_size):
            chunk_indices = indices[start : start + batch_size]
            label_ids_batch = [self._label_ids[index] for index in chunk_indices]
            max_label_len = max(len(ids) for ids in label_ids_batch)
            total_len = prompt_len + max_label_len
            embeds = torch.zeros(
                len(label_ids_batch),
                total_len,
                prompt.size(-1),
                dtype=prompt.dtype,
                device=self.device,
            )
            attention = torch.zeros(
                len(label_ids_batch), total_len, dtype=torch.long, device=self.device
            )
            target_ids = torch.full(
                (len(label_ids_batch), max_label_len),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            target_mask = torch.zeros_like(target_ids, dtype=torch.bool)
            for row_index, ids in enumerate(label_ids_batch):
                ids_tensor = torch.tensor(
                    ids, dtype=torch.long, device=self.device
                )
                embeds[row_index, :prompt_len] = prompt
                embeds[
                    row_index, prompt_len : prompt_len + len(ids)
                ] = input_embedding(ids_tensor).to(dtype=prompt.dtype)
                attention[row_index, : prompt_len + len(ids)] = 1
                target_ids[row_index, : len(ids)] = ids_tensor
                target_mask[row_index, : len(ids)] = True

            position_ids = attention.long().cumsum(dim=1) - 1
            position_ids = position_ids.masked_fill(attention == 0, 0)
            keep_positions = torch.arange(
                prompt_len - 1,
                prompt_len - 1 + max_label_len,
                dtype=torch.long,
                device=self.device,
            )
            output = self.llm(
                inputs_embeds=embeds,
                attention_mask=attention,
                position_ids=position_ids,
                logits_to_keep=keep_positions,
                use_cache=False,
            ).logits.float()
            log_probs = F.log_softmax(output, dim=-1).gather(
                -1, target_ids.unsqueeze(-1)
            ).squeeze(-1)
            masked = log_probs.masked_fill(~target_mask, 0.0)
            scores = masked.sum(dim=-1)
            if self.score_normalization == "mean":
                scores = scores / target_mask.sum(dim=-1).clamp_min(1)
            label_scores.append(scores)
        return torch.cat(label_scores, dim=0)
