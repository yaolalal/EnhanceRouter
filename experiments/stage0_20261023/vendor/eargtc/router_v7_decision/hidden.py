from __future__ import annotations

import torch


def _answer_positions(
    mask: torch.Tensor,
    *,
    target_device: torch.device,
) -> torch.Tensor:
    """Return each row's final attended position on the consumer device."""

    positions = (
        torch.arange(mask.shape[1], device=mask.device)
        .view(1, -1)
        .expand_as(mask)
        .masked_fill(~mask, -1)
        .max(dim=1)
        .values
    )
    if bool((positions < 0).any().item()):
        raise ValueError("empty prompt encountered")
    return positions.to(device=target_device)


def answer_hidden_and_logits(
    model,
    enc: dict[str, torch.Tensor],
    number_token_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return final prompt-token hidden states and numbered next-token logits."""

    if "input_ids" not in enc or "attention_mask" not in enc:
        raise ValueError("encoded batch requires input_ids and attention_mask")
    base_model = getattr(model, "model", None)
    if base_model is None:
        output = model(**enc, output_hidden_states=True, use_cache=False)
        hidden_states = output.hidden_states[-1]
    else:
        output = base_model(**enc, use_cache=False, return_dict=True)
        hidden_states = output.last_hidden_state
    mask = enc["attention_mask"].to(dtype=torch.bool)
    if hidden_states.ndim != 3 or mask.shape != hidden_states.shape[:2]:
        raise ValueError("model hidden states do not align with attention mask")
    positions = _answer_positions(mask, target_device=hidden_states.device)
    batch = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    answer_hidden = hidden_states[batch, positions]
    vocab_logits = model.lm_head(answer_hidden)
    token_ids = torch.as_tensor(
        number_token_ids, dtype=torch.long, device=vocab_logits.device
    )
    return answer_hidden, vocab_logits.index_select(1, token_ids)


__all__ = ["answer_hidden_and_logits"]
