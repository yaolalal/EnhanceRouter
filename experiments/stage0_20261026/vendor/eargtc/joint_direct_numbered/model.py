from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

import torch
import torch.nn as nn

from .numbered import NumberedPromptSpec, extract_class_logits


STRUCTURE_MARKER = "<<EARGTC_GRAPH_VECTOR_91C53A7B>>"


@dataclass
class DirectOutput:
    logits: torch.Tensor
    vocabulary_logits: torch.Tensor
    structural_vectors: torch.Tensor
    attention_mask: torch.Tensor
    sequence_lengths: torch.Tensor


class DirectNumberedJointClassifier(nn.Module):
    """Frozen LLM with one trainable graph vector and direct number-token logits."""

    def __init__(
        self,
        *,
        llm: nn.Module,
        tokenizer,
        prompt_spec: NumberedPromptSpec,
        codebook_size: int,
        gnn_embedding_dim: int,
        gnn_scale_init: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.prompt_spec = prompt_spec
        for parameter in self.llm.parameters():
            parameter.requires_grad_(False)
        self.llm.eval()
        embedding = self.llm.get_input_embeddings()
        hidden = int(embedding.embedding_dim)
        self.code_embedding = nn.Embedding(int(codebook_size), hidden).to(
            device=embedding.weight.device, dtype=torch.float32
        )
        self.gnn_norm = nn.LayerNorm(int(gnn_embedding_dim)).to(embedding.weight.device)
        self.gnn_projector = nn.Linear(int(gnn_embedding_dim), hidden).to(
            embedding.weight.device
        )
        self.gnn_scale = nn.Parameter(
            torch.tensor(float(gnn_scale_init), device=embedding.weight.device)
        )
        self.dropout = nn.Dropout(float(dropout))
        self._init_graph_parameters()

    @property
    def device(self) -> torch.device:
        return self.llm.get_input_embeddings().weight.device

    def train(self, mode: bool = True):
        super().train(mode)
        self.llm.eval()
        return self

    def _init_graph_parameters(self) -> None:
        source = self.llm.get_input_embeddings().weight.detach().float()
        generator = torch.Generator(device=source.device)
        generator.manual_seed(0)
        ids = torch.randperm(source.size(0), generator=generator, device=source.device)
        ids = ids[: self.code_embedding.num_embeddings]
        self.code_embedding.weight.data.copy_(source.index_select(0, ids))
        nn.init.xavier_uniform_(self.gnn_projector.weight)
        nn.init.zeros_(self.gnn_projector.bias)

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_path: str,
        prompt_spec: NumberedPromptSpec,
        codebook_size: int,
        gnn_embedding_dim: int,
        dtype: str = "bfloat16",
        device_map: str | None = "auto",
        gnn_scale_init: float = 0.1,
        gradient_checkpointing: bool = True,
    ) -> "DirectNumberedJointClassifier":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": "auto",
        }[dtype]
        llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        llm.config.use_cache = False
        if gradient_checkpointing and hasattr(llm, "gradient_checkpointing_enable"):
            llm.gradient_checkpointing_enable()
        return cls(
            llm=llm,
            tokenizer=tokenizer,
            prompt_spec=prompt_spec,
            codebook_size=codebook_size,
            gnn_embedding_dim=gnn_embedding_dim,
            gnn_scale_init=gnn_scale_init,
        )

    def structural_vectors(
        self, token_ids: torch.Tensor, gnn_embeddings: torch.Tensor
    ) -> torch.Tensor:
        codes = torch.as_tensor(token_ids, dtype=torch.long, device=self.device).view(-1)
        graph = torch.as_tensor(
            gnn_embeddings,
            dtype=self.gnn_projector.weight.dtype,
            device=self.device,
        )
        if graph.ndim != 2 or graph.size(0) != codes.numel():
            raise ValueError("token IDs and GNN embeddings must have matching batch size")
        code = self.code_embedding(codes)
        continuous = self.gnn_projector(self.gnn_norm(graph))
        return self.dropout(code) + self.gnn_scale * continuous

    def _render_parts(self, text: str) -> tuple[list[int], list[int]]:
        human = (
            "Structural information of the center node:\n"
            f"{STRUCTURE_MARKER}\n{text}\n{self.prompt_spec.task_prompt}"
        )
        if self.prompt_spec.use_chat_template and getattr(
            self.tokenizer, "chat_template", None
        ):
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": human}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = human
        if rendered.count(STRUCTURE_MARKER) != 1:
            raise ValueError("rendered prompt must contain exactly one structure marker")
        prefix, suffix = rendered.split(STRUCTURE_MARKER)
        prefix_ids = list(self.tokenizer.encode(prefix, add_special_tokens=False))
        suffix_ids = list(self.tokenizer.encode(suffix, add_special_tokens=False))
        budget = int(self.prompt_spec.max_input_length) - len(prefix_ids) - 1
        if budget <= 0:
            raise ValueError("numbered prompt prefix exceeds max input length")
        suffix_ids = suffix_ids[-budget:]
        if not suffix_ids:
            raise ValueError("numbered prompt suffix tokenized to empty")
        return prefix_ids, suffix_ids

    def _input_batch(
        self, structural_vectors: torch.Tensor, texts: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(texts) != structural_vectors.size(0):
            raise ValueError("texts and structural vectors must have matching batch size")
        input_embedding = self.llm.get_input_embeddings()
        sequences = []
        lengths = []
        for text, vector in zip(texts, structural_vectors):
            prefix_ids, suffix_ids = self._render_parts(str(text))
            prefix = input_embedding(torch.tensor(prefix_ids, device=self.device))
            suffix = input_embedding(torch.tensor(suffix_ids, device=self.device))
            sequence = torch.cat(
                [prefix, vector.to(prefix.dtype).view(1, -1), suffix], dim=0
            )
            sequences.append(sequence)
            lengths.append(sequence.size(0))
        max_length = max(lengths)
        hidden = input_embedding.embedding_dim
        values = torch.zeros(
            len(sequences), max_length, hidden,
            device=self.device, dtype=input_embedding.weight.dtype,
        )
        mask = torch.zeros(
            len(sequences), max_length, device=self.device, dtype=torch.long
        )
        for index, sequence in enumerate(sequences):
            length = sequence.size(0)
            values[index, -length:] = sequence
            mask[index, -length:] = 1
        return values, mask, torch.tensor(lengths, dtype=torch.long)

    def forward(
        self,
        token_ids: torch.Tensor,
        texts: Sequence[str],
        gnn_embeddings: torch.Tensor,
    ) -> DirectOutput:
        structural = self.structural_vectors(token_ids, gnn_embeddings)
        inputs_embeds, attention_mask, lengths = self._input_batch(structural, texts)
        output = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=1,
        )
        vocabulary_logits = output.logits[:, -1, :]
        logits = extract_class_logits(
            vocabulary_logits,
            self.prompt_spec.number_token_ids,
            self.prompt_spec.number_to_label_ids,
        )
        return DirectOutput(
            logits=logits,
            vocabulary_logits=vocabulary_logits,
            structural_vectors=structural,
            attention_mask=attention_mask,
            sequence_lengths=lengths,
        )
