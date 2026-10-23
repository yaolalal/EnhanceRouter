from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from eargtc.numbered_prompts import build_numbered_prompt
from eargtc.prompts import DIRECT_PROMPTS


class NumberedPromptError(ValueError):
    pass


@dataclass(frozen=True)
class NumberedPromptSpec:
    dataset: str
    label_names: tuple[str, ...]
    task_prompt: str
    number_token_ids: tuple[int, ...]
    number_to_label_ids: tuple[int, ...]
    use_chat_template: bool
    max_input_length: int
    source_metadata_sha256: str

    @property
    def num_classes(self) -> int:
        return len(self.label_names)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_mapping(
    number_token_ids: Sequence[int], number_to_label_ids: Sequence[int]
) -> None:
    count = len(number_token_ids)
    if count <= 0 or len(number_to_label_ids) != count:
        raise NumberedPromptError("number-token and label mappings must be non-empty and equal")
    if len(set(map(int, number_token_ids))) != count:
        raise NumberedPromptError("number token IDs must be unique")
    if sorted(map(int, number_to_label_ids)) != list(range(count)):
        raise NumberedPromptError("number-to-label mapping must be a permutation")


def extract_class_logits(
    vocabulary_logits: torch.Tensor,
    number_token_ids: Sequence[int],
    number_to_label_ids: Sequence[int],
) -> torch.Tensor:
    value = torch.as_tensor(vocabulary_logits)
    if value.ndim != 2:
        raise NumberedPromptError("vocabulary logits must have shape [N,V]")
    _validate_mapping(number_token_ids, number_to_label_ids)
    token_ids = torch.tensor(
        list(map(int, number_token_ids)), dtype=torch.long, device=value.device
    )
    if int(token_ids.min()) < 0 or int(token_ids.max()) >= value.size(1):
        raise NumberedPromptError("number token ID is outside the vocabulary")
    by_number = value.index_select(-1, token_ids)
    class_logits = torch.empty_like(by_number)
    label_ids = torch.tensor(
        list(map(int, number_to_label_ids)), dtype=torch.long, device=value.device
    )
    class_logits.index_copy_(-1, label_ids, by_number)
    return class_logits


def load_numbered_prompt_spec(dataset: str, numbered_root: Path) -> NumberedPromptSpec:
    root = Path(numbered_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise NumberedPromptError(f"numbered root manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        source_value = manifest["datasets"][dataset]["source_dir"]
    except KeyError as exc:
        raise NumberedPromptError(f"dataset missing from numbered manifest: {dataset}") from exc
    source = Path(source_value)
    if not source.is_absolute():
        source = root.resolve().parents[1] / source
    metadata_paths = sorted((source / "chunks").glob("*.meta.json"))
    if not metadata_paths:
        raise NumberedPromptError(f"no numbered-logits metadata for {dataset}: {root}")
    metadata_path = metadata_paths[0]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("dataset") != dataset:
        raise NumberedPromptError("numbered metadata dataset mismatch")
    label_names = tuple(map(str, metadata["label_names"]))
    number_token_ids = tuple(map(int, metadata["number_token_ids"]))
    number_to_label_ids = tuple(
        map(int, metadata.get("number_to_label_ids", range(len(label_names))))
    )
    _validate_mapping(number_token_ids, number_to_label_ids)
    if len(label_names) != len(number_token_ids):
        raise NumberedPromptError("label names and number tokens differ in count")

    prompt_path_value = metadata.get("prompt_template_path")
    if prompt_path_value:
        prompt_path = Path(prompt_path_value)
        if not prompt_path.is_file():
            raise NumberedPromptError(f"frozen prompt template is missing: {prompt_path}")
        task_prompt = prompt_path.read_text(encoding="utf-8").rstrip()
    else:
        base_prompt = DIRECT_PROMPTS.get(dataset)
        if base_prompt is None:
            raise NumberedPromptError(f"no base prompt for dataset {dataset}")
        task_prompt = build_numbered_prompt(
            text="", base_prompt=base_prompt, label_names=label_names
        ).lstrip("\n")
    if "category number" not in task_prompt.lower():
        raise NumberedPromptError("frozen prompt does not request one category number")
    return NumberedPromptSpec(
        dataset=dataset,
        label_names=label_names,
        task_prompt=task_prompt,
        number_token_ids=number_token_ids,
        number_to_label_ids=number_to_label_ids,
        use_chat_template=bool(metadata.get("use_chat_template", True)),
        max_input_length=int(metadata.get("max_input_length", 2048)),
        source_metadata_sha256=_sha256(metadata_path),
    )

