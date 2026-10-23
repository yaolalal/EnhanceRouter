#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eargtc.router_v7_decision.data import (
    decisive_partition,
    load_base_dataset_bundle,
)
from eargtc.router_v7_decision.experiment import (
    PreparedDecisionDataset,
    select_fit_validate,
)
from eargtc.router_v7_decision.metrics import decisive_metrics
from eargtc.router_v7_decision.models import CandidateSpec

DATASETS = ["cora", "citeseer", "pubmed", "arxiv", "ogbn-products"]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _ordinary_grid() -> list[CandidateSpec]:
    candidates = []
    for balanced in (False, True):
        suffix = "balanced" if balanced else "natural"
        for c in (0.1, 1.0, 10.0):
            candidates.append(CandidateSpec(f"logistic_c{c:g}_{suffix}", "logistic", c=c, class_balanced=balanced))
        for c in (0.3, 3.0):
            candidates.append(CandidateSpec(f"rbf_c{c:g}_{suffix}", "rbf", c=c, class_balanced=balanced))
        for leaf in (1, 3):
            candidates.append(CandidateSpec(f"extra_leaf{leaf}_{suffix}", "extra_trees", class_balanced=balanced, min_samples_leaf=leaf))
    for c in (0.3, 3.0):
        candidates.append(CandidateSpec(f"mlp16_c{c:g}", "mlp", c=c, hidden_dim=16))
    return candidates


def _prepare(dataset: str, bank, bundle, keys: list[str]) -> PreparedDecisionDataset:
    support = decisive_partition(
        bundle.support_labels, bundle.support_gnn_pred, bundle.support_llm_pred
    )
    validation = decisive_partition(
        bundle.labels[bundle.val_ids],
        bundle.gnn_pred[bundle.val_ids],
        bundle.llm_pred[bundle.val_ids],
    )
    validation_ids = bundle.val_ids[validation.node_ids]
    return PreparedDecisionDataset(
        name=dataset,
        support_features={
            key: bank.support_feature(key, support.node_ids) for key in keys
        },
        support_targets=support.target.long(),
        validation_features={
            key: bank.full_feature(key, validation_ids) for key in keys
        },
        validation_targets=validation.target.long(),
        validation_node_ids=validation_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "runs/router_v7_decision_embeddings_20260827_v1",
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
    parser.add_argument("--families", nargs="+", choices=["B", "C"], default=["B", "C"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    banks = {
        dataset: torch.load(
            output_root / "features" / dataset / "feature_bank.pt",
            map_location="cpu",
            weights_only=False,
        )
        for dataset in DATASETS
    }
    bundles = {
        dataset: load_base_dataset_bundle(
            dataset,
            upstream_root=args.upstream_root,
            dataset_root=args.dataset_root,
            gnn_embedding_root=args.gnn_embedding_root,
        )
        for dataset in DATASETS
    }
    ordinary = _ordinary_grid()
    for family in args.families:
        keys = [
            key
            for key in banks[DATASETS[0]].feature_keys()
            if key.startswith(f"{family}_")
        ]
        candidates = list(ordinary)
        if family == "C":
            for balanced in (False, True):
                suffix = "balanced" if balanced else "natural"
                for c in (0.1, 1.0, 10.0):
                    candidates.append(
                        CandidateSpec(
                            f"residual_c{c:g}_{suffix}",
                            "residual_logistic",
                            c=c,
                            class_balanced=balanced,
                            base_width=20,
                            correction_scale=0.25,
                        )
                    )
        prepared = {
            dataset: _prepare(dataset, banks[dataset], bundles[dataset], keys)
            for dataset in DATASETS
        }
        endpoints = {}
        for target in DATASETS:
            endpoint = select_fit_validate(
                target,
                prepared,
                keys,
                candidates,
                seed=args.seed,
                adapt_target_bias=True,
            )
            target_root = output_root / "selection" / family / target
            target_root.mkdir(parents=True, exist_ok=True)
            head_path = target_root / "head.pkl"
            temporary = head_path.with_name(f".{head_path.name}.tmp.{os.getpid()}")
            with temporary.open("wb") as handle:
                pickle.dump(endpoint.head, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, head_path)
            validation = decisive_metrics(
                endpoint.validation_scores,
                endpoint.validation_targets,
                endpoint.accuracy_threshold.threshold,
            )
            selection = {
                "target": target,
                "family": family,
                "feature_key": endpoint.feature_key,
                "selected_name": endpoint.selected_name,
                "selected_spec": endpoint.spec.__dict__,
                "adapter": {
                    "kind": "bounded_scalar_bias",
                    "bias": endpoint.head.adapter_bias,
                    "max_abs_bias": 1.0,
                    "ridge": 5.0,
                },
                "accuracy_threshold": endpoint.accuracy_threshold.__dict__,
                "balanced_threshold": endpoint.balanced_threshold.__dict__,
                "validation_metrics_at_accuracy_threshold": validation,
                "validation_node_ids": endpoint.validation_node_ids.tolist(),
                "validation_scores": endpoint.validation_scores.tolist(),
                "validation_targets": endpoint.validation_targets.tolist(),
                "source_lodo_ranking": endpoint.ranking,
                "head_path": str(head_path),
                "feature_bank": str(
                    output_root / "features" / target / "feature_bank.pt"
                ),
            }
            selection_path = target_root / "selection.json"
            _atomic_json(selection_path, selection)
            endpoints[target] = {
                "selection_path": str(selection_path),
                "selection_sha256": _hash(selection_path),
                "head_path": str(head_path),
                "head_sha256": _hash(head_path),
                "feature_bank_path": selection["feature_bank"],
                "feature_bank_sha256": _hash(Path(selection["feature_bank"])),
            }
            print(
                family,
                target,
                endpoint.selected_name,
                f"val_acc={validation['decisive_accuracy']:.4f}",
                f"R_G={validation['gnn_only_recall']:.4f}",
                f"R_L={validation['llm_only_recall']:.4f}",
                flush=True,
            )
        manifest = {
            "status": "frozen_validation_selection",
            "family": family,
            "seed": args.seed,
            "datasets": DATASETS,
            "feature_keys": keys,
            "candidate_grid": [candidate.__dict__ for candidate in candidates],
            "adaptation": {"kind": "bounded_scalar_bias", "max_abs_bias": 1.0, "ridge": 5.0},
            "selection_rule": {
                "source": "joint projection/model LODO macro decisive accuracy; 0.5pp band then minimum recall, NLL, complexity, name",
                "target": "validation accuracy; minimum recall, distance to 0.5, larger threshold",
            },
            "endpoints": endpoints,
            "feature_manifest": {
                "path": str(output_root / "features" / "MANIFEST.json"),
                "sha256": _hash(output_root / "features" / "MANIFEST.json"),
            },
            "code_hashes": {
                str(path.relative_to(ROOT)): _hash(path)
                for path in [
                    ROOT / "eargtc/router_v7_decision/audit.py",
                    ROOT / "eargtc/router_v7_decision/cache.py",
                    ROOT / "eargtc/router_v7_decision/data.py",
                    ROOT / "eargtc/router_v7_decision/features.py",
                    ROOT / "eargtc/router_v7_decision/feature_bank.py",
                    ROOT / "eargtc/router_v7_decision/projection.py",
                    ROOT / "eargtc/router_v7_decision/experiment.py",
                    ROOT / "eargtc/router_v7_decision/models.py",
                    ROOT / "eargtc/router_v7_decision/metrics.py",
                    ROOT / "eargtc/router_v7_decision/selection.py",
                    Path(__file__),
                ]
            },
        }
        _atomic_json(output_root / "selection" / family / "FROZEN.json", manifest)
        print(f"Frozen {family} selection")


if __name__ == "__main__":
    main()
