#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "cora": {
        "nodes": 2708,
        "chunk": 2708,
        "partition": "RTX3090",
        "time": "1-00:00:00",
        "meta": "runs/llm_numbered_token_logits_20260629/cora/chunks/logits_0_1000.jsonl.meta.json",
    },
    "citeseer": {
        "nodes": 3186,
        "chunk": 3186,
        "partition": "RTX3090",
        "time": "1-00:00:00",
        "meta": "runs/llm_numbered_token_logits_20260629/citeseer/chunks/logits_0_1000.jsonl.meta.json",
    },
    "pubmed": {
        "nodes": 19717,
        "chunk": 4000,
        "partition": "RTX3090",
        "time": "2-00:00:00",
        "meta": "runs/llm_numbered_token_logits_20260629/pubmed/chunks/logits_0_4000.jsonl.meta.json",
    },
    "arxiv": {
        "nodes": 169343,
        "chunk": 12000,
        "partition": "RTX4090",
        "time": "2-00:00:00",
        "meta": "runs/llm_numbered_token_logits_arxiv_v1_explanation_order_full_20260703/arxiv/chunks/logits_0_42336.jsonl.meta.json",
    },
    "ogbn-products": {
        "nodes": 54025,
        "chunk": 8000,
        "partition": "RTX4090",
        "time": "2-00:00:00",
        "meta": "runs/llm_numbered_token_logits_20260629/ogbn-products/chunks/logits_0_4000.jsonl.meta.json",
    },
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "runs/router_v7_decision_llm_hidden_20260827_v1",
    )
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for dataset, spec in DATASETS.items():
        meta = (ROOT / spec["meta"]).resolve()
        logits = (
            ROOT
            / "runs/llm_numbered_token_logits_final_20260706"
            / dataset
            / "llm_logits.jsonl"
        ).resolve()
        if not meta.exists() or not logits.exists():
            raise FileNotFoundError(f"missing frozen reference for {dataset}")
        for start in range(0, int(spec["nodes"]), int(spec["chunk"])):
            end = min(start + int(spec["chunk"]), int(spec["nodes"]))
            output = run_root / dataset / "shards" / f"hidden_{start}_{end}.pt"
            record = {
                "dataset": dataset,
                "start": start,
                "end": end,
                "output": str(output),
                "partition": spec["partition"],
                "time": spec["time"],
                "reference_meta": str(meta),
                "reference_meta_sha256": _hash(meta),
                "reference_logits": str(logits),
                "status": "existing" if output.exists() else "planned",
                "job_id": None,
            }
            if args.submit and not output.exists():
                env = os.environ.copy()
                env.update(
                    {
                        "ROOT_DIR": str(ROOT),
                        "RUN_ROOT": str(run_root),
                        "DATASET": dataset,
                        "START_INDEX": str(start),
                        "END_INDEX": str(end),
                        "REFERENCE_META": str(meta),
                        "REFERENCE_LOGITS": str(logits),
                    }
                )
                command = [
                    "sbatch",
                    "--parsable",
                    f"--partition={spec['partition']}",
                    f"--time={spec['time']}",
                    "--export=ALL",
                    "slurm/router_v7_decision_hidden_chunk.sbatch",
                ]
                record["job_id"] = subprocess.check_output(
                    command, cwd=ROOT, env=env, text=True
                ).strip()
                record["status"] = "submitted"
                print(json.dumps(record, ensure_ascii=False), flush=True)
            jobs.append(record)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "submitted": bool(args.submit),
        "script_sha256": _hash(Path(__file__)),
        "sbatch_sha256": _hash(ROOT / "slurm/router_v7_decision_hidden_chunk.sbatch"),
        "jobs": jobs,
    }
    path = run_root / (
        "submission_manifest.json" if args.submit else "planned_manifest.json"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    print(f"Wrote {path} with {len(jobs)} shards")


if __name__ == "__main__":
    main()
