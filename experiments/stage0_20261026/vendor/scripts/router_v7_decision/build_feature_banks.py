#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eargtc.router_v7_decision.data import load_base_dataset_bundle
from eargtc.router_v7_decision.cache import atomic_torch_save
from eargtc.router_v7_decision.feature_bank import build_dataset_feature_bank

DATASETS = ["cora", "citeseer", "pubmed", "arxiv", "ogbn-products"]


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "runs/router_v7_decision_embeddings_20260827_v1",
    )
    parser.add_argument(
        "--hidden-root",
        type=Path,
        default=ROOT / "runs/router_v7_decision_llm_hidden_20260827_v1",
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=ROOT / "runs/router_v5_structural_oof_fixed_20260729_v1",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=ROOT.parent / "GNN-as-Judge"
    )
    parser.add_argument(
        "--gnn-embedding-root",
        type=Path,
        default=ROOT / "runs/gcn_ssl_v2_selected_20260623_v1",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-projection-rows", type=int, default=512)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    records = {}
    for dataset in DATASETS:
        output = output_root / "features" / dataset / "feature_bank.pt"
        if output.exists():
            bank = torch.load(output, map_location="cpu", weights_only=False)
            if bank.dataset != dataset:
                raise ValueError(f"existing feature bank dataset mismatch: {output}")
            print(f"Reusing {output}", flush=True)
        else:
            bundle = load_base_dataset_bundle(
                dataset,
                upstream_root=args.upstream_root,
                dataset_root=args.dataset_root,
                gnn_embedding_root=args.gnn_embedding_root,
            )
            bank = build_dataset_feature_bank(
                bundle,
                hidden_path=args.hidden_root / dataset / "hidden.pt",
                seed=args.seed,
                max_projection_rows=args.max_projection_rows,
            )
            atomic_torch_save(bank, output)
            print(f"Wrote {output}", flush=True)
        records[dataset] = {
            "path": str(output),
            "sha256": _hash(output),
            "input_hashes": {
                "llm_hidden": _hash(args.hidden_root / dataset / "hidden.pt"),
                "gnn_ssl_embedding": _hash(
                    args.gnn_embedding_root / dataset / "encoder_and_embeddings.pt"
                ),
                "frozen_predictions": _hash(
                    args.upstream_root / dataset / "pipeline" / "predictions.pt"
                ),
                "corrected_oof": _hash(
                    args.upstream_root / dataset / "pipeline/router/oof/router_train.jsonl"
                ),
                "graph": _hash(args.dataset_root / "datasets" / f"{dataset}.pt"),
            },
            "feature_keys": bank.feature_keys(),
            "projection_fit_count": int(bank.projection_fit_ids.numel()),
            "projection_fit_ids": bank.projection_fit_ids.tolist(),
            "test_overlap": int(
                torch.isin(bank.projection_fit_ids, bank.test_ids).sum().item()
            ),
            "effective_dims": {
                name: int(projection.effective_dim)
                for name, projection in bank.projections.items()
            },
        }
        if records[dataset]["test_overlap"] != 0:
            raise ValueError(f"projection/test overlap detected for {dataset}")
    manifest = {
        "status": "complete",
        "seed": args.seed,
        "max_projection_rows": args.max_projection_rows,
        "projection_protocol": "local CCA/PCA/random controls plus a fixed shared LLM random basis with dataset-local label-free GNN ridge alignment; deterministic train+validation IDs; test excluded",
        "datasets": records,
        "code_hashes": {
            str(path.relative_to(ROOT)): _hash(path)
            for path in [
                ROOT / "eargtc/router_v7_decision/audit.py",
                ROOT / "eargtc/router_v7_decision/cache.py",
                ROOT / "eargtc/router_v7_decision/feature_bank.py",
                ROOT / "eargtc/router_v7_decision/projection.py",
                Path(__file__),
            ]
        },
    }
    path = output_root / "features" / "MANIFEST.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
