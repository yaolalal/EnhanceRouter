#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eargtc.data import create_few_shot_dataset, get_texts, label_space
from eargtc.numbered_prompts import build_numbered_prompt, number_token_ids
from eargtc.prompts import DIRECT_PROMPTS
from eargtc.router_v7_decision.cache import (
    HiddenManifest,
    semantic_parity_is_valid,
    validate_hidden_shard,
    write_validated_hidden_shard,
)
from eargtc.router_v7_decision.hidden import answer_hidden_and_logits
from eargtc.utils import set_seed


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _artifact_hash(root: Path, names: list[str]) -> str:
    records = []
    for name in names:
        path = root / name
        if path.exists():
            stat = path.stat()
            records.append((name, stat.st_size, _sha256_bytes(path.read_bytes())))
    if not records:
        raise ValueError(f"none of the provenance files exist below {root}: {names}")
    return _sha256_bytes(json.dumps(records, sort_keys=True).encode())


def _prompt_text(tokenizer, human: str, use_chat_template: bool) -> str:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": human}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return human


def _reference_rows(path: Path, wanted_ids: set[int]) -> dict[int, list[float]]:
    found: dict[int, list[float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            node_id = int(row["node_id"])
            if node_id in wanted_ids:
                found[node_id] = [float(v) for v in row["logits"]]
                if len(found) == len(wanted_ids):
                    break
    missing = wanted_ids - set(found)
    if missing:
        raise ValueError(f"reference logits missing node IDs: {sorted(missing)}")
    return found


def _class_logits(
    numbered_logits: torch.Tensor, number_to_label_ids: list[int]
) -> torch.Tensor:
    result = torch.empty_like(numbered_logits)
    for number_index, label_id in enumerate(number_to_label_ids):
        result[:, int(label_id)] = numbered_logits[:, number_index]
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--reference-meta-json", required=True, type=Path)
    parser.add_argument("--reference-logits-jsonl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-index", required=True, type=int)
    parser.add_argument("--end-index", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device-map", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reference = _json(args.reference_meta_json)
    if reference["dataset"] != args.dataset or reference.get("node_source") != "all":
        raise ValueError("reference metadata does not describe this full dataset")
    set_seed(int(reference["seed"]))
    data = create_few_shot_dataset(
        args.dataset,
        args.dataset_root,
        int(reference["shots"]),
        int(reference["seed"]),
        device="cpu",
    )
    texts = get_texts(data)
    num_classes, label_names = label_space(data)
    if list(label_names) != list(reference["label_names"]):
        raise ValueError("label-space mismatch with frozen numbered-logit metadata")
    end = min(int(args.end_index), int(data.num_nodes))
    node_ids = list(range(int(args.start_index), end))
    if not node_ids:
        raise ValueError("empty extraction range")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    token_ids = number_token_ids(tokenizer, int(num_classes))
    if token_ids != [int(v) for v in reference["number_token_ids"]]:
        raise ValueError("number-token IDs differ from frozen reference")

    prompt_template_path = reference.get("prompt_template_path")
    prompt_template = (
        Path(prompt_template_path).read_text(encoding="utf-8").rstrip()
        if prompt_template_path
        else None
    )
    number_to_label_ids = [
        int(v) for v in reference.get("number_to_label_ids", range(int(num_classes)))
    ]
    prompt_protocol = {
        "dataset": args.dataset,
        "base_prompt": DIRECT_PROMPTS.get(args.dataset),
        "label_names": list(label_names),
        "prompt_template": prompt_template,
        "use_chat_template": bool(reference["use_chat_template"]),
        "max_input_length": int(reference["max_input_length"]),
        "number_to_label_ids": number_to_label_ids,
    }
    prompt_hash = _sha256_bytes(
        json.dumps(prompt_protocol, sort_keys=True, ensure_ascii=False).encode()
    )
    model_hash = _artifact_hash(
        args.model_path, ["config.json", "generation_config.json", "model.safetensors.index.json"]
    )
    tokenizer_hash = _artifact_hash(
        args.model_path,
        ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
    )

    expected_metadata = {
        "dataset": args.dataset,
        "model_hash": model_hash,
        "tokenizer_hash": tokenizer_hash,
        "prompt_hash": prompt_hash,
        "start": node_ids[0],
        "end": node_ids[-1] + 1,
        "dtype": args.dtype,
        "hidden_width": int(reference.get("hidden_width", 4096)),
        "row_order_hash": HiddenManifest.hash_ids(torch.tensor(node_ids)),
    }
    if args.output.exists():
        expected = HiddenManifest.from_dict(expected_metadata)
        try:
            validate_hidden_shard(args.output, expected)
        except Exception as error:
            quarantine = args.output.with_name(f".{args.output.name}.invalid.{os.getpid()}")
            os.replace(args.output, quarantine)
            print(f"Quarantined invalid shard ({type(error).__name__}): {quarantine}", flush=True)
        else:
            print(f"Valid shard already exists: {args.output}", flush=True)
            return

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()
    model_device = model.get_input_embeddings().weight.device
    rows: list[torch.Tensor] = []
    parity_ids = {node_ids[0], node_ids[len(node_ids) // 2], node_ids[-1]}
    reference_logits = _reference_rows(args.reference_logits_jsonl, parity_ids)
    parity_error = 0.0
    parity_probability_error = 0.0
    parity_argmax_match = True

    step = max(int(args.batch_size), 1)
    with torch.inference_mode():
        for start in range(0, len(node_ids), step):
            batch_ids = node_ids[start : start + step]
            human = []
            for node_id in batch_ids:
                if prompt_template is None:
                    value = build_numbered_prompt(
                        text=texts[node_id],
                        base_prompt=DIRECT_PROMPTS[args.dataset],
                        label_names=label_names,
                    )
                else:
                    value = f"{texts[node_id]}\n{prompt_template}"
                human.append(value)
            prompts = [
                _prompt_text(tokenizer, value, bool(reference["use_chat_template"]))
                for value in human
            ]
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(reference["max_input_length"]),
            )
            enc = {key: value.to(model_device) for key, value in enc.items()}
            hidden, numbered = answer_hidden_and_logits(model, enc, token_ids)
            class_logits = _class_logits(numbered.float(), number_to_label_ids)
            for row_index, node_id in enumerate(batch_ids):
                if node_id in parity_ids:
                    expected = torch.tensor(
                        reference_logits[node_id],
                        dtype=torch.float32,
                        device=class_logits.device,
                    )
                    actual = class_logits[row_index]
                    parity_error = max(
                        parity_error,
                        float((actual - expected).abs().max().item()),
                    )
                    actual_probability = torch.softmax(actual, dim=0)
                    expected_probability = torch.softmax(expected, dim=0)
                    parity_probability_error = max(
                        parity_probability_error,
                        float(
                            (actual_probability - expected_probability)
                            .abs()
                            .max()
                            .item()
                        ),
                    )
                    parity_argmax_match = parity_argmax_match and bool(
                        actual.argmax().item() == expected.argmax().item()
                    )
            rows.append(hidden.detach().to(dtype).cpu())
            done = min(start + step, len(node_ids))
            if done == step or done % 100 == 0 or done == len(node_ids):
                print(f"Extracted {done}/{len(node_ids)}", flush=True)

    if not semantic_parity_is_valid(
        parity_probability_error, parity_argmax_match
    ):
        raise ValueError(
            f"LM-head semantic parity failed: abs={parity_error:.6g}, "
            f"prob={parity_probability_error:.6g}, argmax={parity_argmax_match}"
        )
    hidden = torch.cat(rows, dim=0).contiguous()
    expected_metadata["hidden_width"] = int(hidden.shape[1])
    expected_metadata.update(
        {
            "max_input_length": int(reference["max_input_length"]),
            "number_token_ids": token_ids,
            "number_to_class_ids": number_to_label_ids,
            "reference_meta": str(args.reference_meta_json.resolve()),
            "reference_logits": str(args.reference_logits_jsonl.resolve()),
            "parity_sample_ids": sorted(parity_ids),
            "parity_max_abs_error": parity_error,
            "parity_max_probability_error": parity_probability_error,
            "parity_argmax_match": parity_argmax_match,
        }
    )
    payload = {
        "node_ids": torch.tensor(node_ids, dtype=torch.long),
        "hidden": hidden,
        "metadata": expected_metadata,
    }
    write_validated_hidden_shard(payload, args.output)
    print(f"Wrote validated label-free shard: {args.output}", flush=True)


if __name__ == "__main__":
    main()
