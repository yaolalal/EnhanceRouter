#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eargtc.router_v7_decision.cache import (
    HiddenManifest,
    merge_hidden_shards,
    validate_hidden_shard,
    write_validated_hidden_shard,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "runs/router_v7_decision_llm_hidden_20260827_v1",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cora", "citeseer", "pubmed", "arxiv", "ogbn-products"],
    )
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    submission = json.loads(
        (run_root / "submission_manifest.json").read_text(encoding="utf-8")
    )
    audit = {"run_root": str(run_root), "datasets": {}, "status": "pass"}
    for dataset in args.datasets:
        records = [row for row in submission["jobs"] if row["dataset"] == dataset]
        if not records:
            raise ValueError(f"submission manifest has no shards for {dataset}")
        records.sort(key=lambda row: int(row["start"]))
        paths = [Path(row["output"]) for row in records]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"{dataset} has {len(missing)} missing shards; first={missing[0]}"
            )
        expected_ids = torch.arange(int(records[-1]["end"]), dtype=torch.long)
        hidden = merge_hidden_shards(paths, expected_ids)
        shards = [validate_hidden_shard(path) for path in paths]
        reference = shards[0].manifest
        parity_errors = [
            float(shard.manifest.parity_max_abs_error)
            for shard in shards
        ]
        probability_errors = [
            float(shard.manifest.parity_max_probability_error)
            for shard in shards
        ]
        argmax_match = all(bool(shard.manifest.parity_argmax_match) for shard in shards)
        metadata = {
            "dataset": dataset,
            "model_hash": reference.model_hash,
            "tokenizer_hash": reference.tokenizer_hash,
            "prompt_hash": reference.prompt_hash,
            "start": 0,
            "end": int(expected_ids.numel()),
            "dtype": str(hidden.dtype).replace("torch.", ""),
            "hidden_width": int(hidden.shape[1]),
            "row_order_hash": HiddenManifest.hash_ids(expected_ids),
            "shard_count": len(paths),
            "source_shards": [str(path.resolve()) for path in paths],
            "parity_max_abs_error": max(parity_errors, default=0.0),
            "parity_max_probability_error": max(probability_errors, default=0.0),
            "parity_argmax_match": argmax_match,
        }
        payload = {"node_ids": expected_ids, "hidden": hidden, "metadata": metadata}
        output = run_root / dataset / "hidden.pt"
        validated = write_validated_hidden_shard(payload, output)
        audit["datasets"][dataset] = {
            "status": "pass",
            "rows": int(validated.node_ids.numel()),
            "width": int(validated.hidden.shape[1]),
            "dtype": str(validated.hidden.dtype).replace("torch.", ""),
            "shards": len(paths),
            "parity_max_abs_error": metadata["parity_max_abs_error"],
            "parity_max_probability_error": metadata["parity_max_probability_error"],
            "parity_argmax_match": metadata["parity_argmax_match"],
            "output": str(output),
        }
        print(json.dumps(audit["datasets"][dataset], ensure_ascii=False), flush=True)
    path = run_root / "AUDIT.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
