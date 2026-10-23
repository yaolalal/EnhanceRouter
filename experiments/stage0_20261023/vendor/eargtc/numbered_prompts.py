from __future__ import annotations

import re
from typing import Sequence

import torch
import torch.nn.functional as F


def strip_answer_suffix(prompt: str) -> str:
    """Remove the final Answer cue so the numbered instruction can replace it."""
    return re.sub(r"\s*Answer\s*:\s*$", "", str(prompt), flags=re.IGNORECASE).rstrip()


def _numbered_options(label_names: Sequence[str]) -> str:
    return "\n".join(f"{idx}. {label}" for idx, label in enumerate(label_names, start=1))


def strip_numbered_redundancy(prompt: str) -> str:
    """Compatibility helper for the original append-only numbered prompt."""
    return strip_answer_suffix(prompt)


def build_numbered_prompt(*, text: str, base_prompt: str, label_names: Sequence[str]) -> str:
    if not label_names:
        raise ValueError("label_names must not be empty")
    base = strip_answer_suffix(base_prompt)
    options = _numbered_options(label_names)
    return (
        f"{text}\n{base}\n\n"
        f"Category number options:\n{options}\n\n"
        f"Reply only one category number from 1 to {len(label_names)}. "
        "Do not output the category name or any explanation words.\n\n"
        "Answer: "
    )

def number_token_ids(tokenizer, num_classes: int) -> list[int]:
    if int(num_classes) <= 0:
        raise ValueError("num_classes must be positive")
    ids: list[int] = []
    bad: list[dict] = []
    for number in range(1, int(num_classes) + 1):
        pieces = tokenizer(str(number), add_special_tokens=False).input_ids
        if len(pieces) != 1:
            bad.append({"number": number, "token_ids": pieces})
            continue
        ids.append(int(pieces[0]))
    if bad:
        raise ValueError(f"Category numbers must each be a single tokenizer token: {bad}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Category number token ids are not unique: {ids}")
    return ids


def class_distribution_from_token_logits(
    vocab_logits: torch.Tensor,
    number_ids: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if vocab_logits.dim() != 1:
        raise ValueError(f"Expected 1D vocabulary logits, got shape {tuple(vocab_logits.shape)}")
    token_ids = torch.tensor([int(i) for i in number_ids], dtype=torch.long, device=vocab_logits.device)
    class_logits = vocab_logits.float().index_select(0, token_ids)
    probabilities = F.softmax(class_logits, dim=-1)
    return class_logits, probabilities
