from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from eargtc.router_a20_v6backbone.evaluation import paired_route_summary

from .artifacts import sha256


def verify_frozen_manifest(path: str | Path, *, datasets: tuple[str, ...], endpoints: tuple[str, ...]) -> dict:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text())
    if payload.get("status") != "frozen_validation_selection":
        raise ValueError("selection is not globally frozen")
    if tuple(payload.get("datasets", ())) != datasets or tuple(payload.get("endpoint_names", ())) != endpoints:
        raise ValueError("frozen experiment coverage differs from contract")
    for dataset in datasets:
        record = payload["dataset_endpoints"][dataset]
        complete_path = Path(record["complete_path"])
        if sha256(complete_path) != record["complete_sha256"]:
            raise ValueError(f"target completion hash mismatch: {dataset}")
        for endpoint in endpoints:
            frozen = record["endpoints"][endpoint]
            for kind in ("selection", "scores"):
                artifact = Path(frozen[f"{kind}_path"])
                if not artifact.is_file() or sha256(artifact) != frozen[f"{kind}_sha256"]:
                    raise ValueError(f"artifact hash mismatch: {dataset}/{endpoint}/{kind}")
    for raw_path, digest in payload.get("code_hashes", {}).items():
        if sha256(raw_path) != digest:
            raise ValueError(f"frozen code hash mismatch: {raw_path}")
    return payload


def align_baseline(payload: dict, node_ids: torch.Tensor, targets: torch.Tensor, method: str) -> tuple[torch.Tensor, torch.Tensor]:
    expected_ids = torch.as_tensor(node_ids, dtype=torch.long).cpu()
    expected_targets = torch.as_tensor(targets, dtype=torch.long).cpu()
    if not torch.equal(torch.as_tensor(payload["node_ids"]).long(), expected_ids):
        raise ValueError(f"{method}: baseline node IDs do not align")
    if not torch.equal(torch.as_tensor(payload["targets"]).long(), expected_targets):
        raise ValueError(f"{method}: baseline targets do not align")
    record = payload[method]
    scores = torch.as_tensor(record["scores"], dtype=torch.float64).view(-1)
    routes = torch.as_tensor(record["route_gnn"], dtype=torch.bool).view(-1)
    if scores.numel() != targets.numel() or routes.numel() != targets.numel() or not bool(torch.isfinite(scores).all()):
        raise ValueError(f"{method}: invalid baseline predictions")
    return scores, routes


def route_metrics(scores: torch.Tensor, routes: torch.Tensor, targets: torch.Tensor) -> dict:
    score = np.asarray(torch.as_tensor(scores, dtype=torch.float64).cpu())
    route = np.asarray(torch.as_tensor(routes, dtype=torch.bool).cpu())
    target = np.asarray(torch.as_tensor(targets, dtype=torch.long).cpu())
    gnn, llm = target == 1, target == 0
    correct = route == gnn
    clipped = np.clip(score, 1e-7, 1 - 1e-7)
    g_recall = float(route[gnn].mean())
    l_recall = float((~route[llm]).mean())
    return {
        "gnn_only_recall": g_recall,
        "llm_only_recall": l_recall,
        "decisive_accuracy": float(correct.mean()),
        "balanced_accuracy": 0.5 * (g_recall + l_recall),
        "auroc": float(roc_auc_score(target, score)),
        "average_precision": float(average_precision_score(target, score)),
        "nll": float(-np.mean(target * np.log(clipped) + (1 - target) * np.log1p(-clipped))),
        "counts": {
            "gnn_only": int(gnn.sum()),
            "llm_only": int(llm.sum()),
            "route_gnn": int(route.sum()),
            "route_llm": int((~route).sum()),
            "correct": int(correct.sum()),
            "total": int(target.size),
        },
    }


__all__ = ["align_baseline", "paired_route_summary", "route_metrics", "verify_frozen_manifest"]
