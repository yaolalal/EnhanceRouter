"""Controlled B-specific training for Split Scheme-C conflict reliability."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import torch

from eargtc.router_bc.data_impl import build_support_domain, load_target_artifacts
from eargtc.router_bc_split_c.artifacts import atomic_write_json, sha256_file
from eargtc.router_bc_split_c.b_replay import FrozenBReplay
from eargtc.router_bc_split_c.calibration import (
    deterministic_calibration_threshold_split,
    fit_identity_shrunk_platt,
)
from eargtc.router_bc_split_c.model import (
    ReliabilityConfig,
    fit_reliability_model,
    select_reliability_config,
)
from eargtc.router_bc_split_c.pipeline import (
    DATASETS,
    _query_domain,
    _rank_target,
    build_support_feature_domains,
)

from .providers import BReplay


L2_GRID = (0.01, 0.1, 1.0, 10.0)
PRIOR_GRID = (0.1, 1.0, 10.0, 100.0)
CALIBRATION_REGULARIZATION = 0.1


def selected_correctness_target(
    labels: torch.Tensor,
    gnn_pred: torch.Tensor,
    llm_pred: torch.Tensor,
    choose_g: torch.Tensor,
) -> torch.Tensor:
    y = torch.as_tensor(labels).view(-1)
    g = torch.as_tensor(gnn_pred).view(-1)
    l = torch.as_tensor(llm_pred).view(-1)
    route = torch.as_tensor(choose_g, dtype=torch.bool).view(-1)
    if not (y.numel() == g.numel() == l.numel() == route.numel()):
        raise ValueError("conflict target inputs must align")
    return torch.where(route, g, l) == y


def _support_predictions(support) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([int(row["gnn_pred"]) for row in support.rows]),
        torch.tensor([int(row["llm_pred"]) for row in support.rows]),
    )


def to_frozen_replay(replay: BReplay, support) -> FrozenBReplay:
    expected_ids = torch.tensor([int(row["node_id"]) for row in support.rows])
    if not torch.equal(expected_ids, replay.support_node_ids):
        raise ValueError(f"support node IDs differ for {replay.dataset}")
    g, l = _support_predictions(support)
    return FrozenBReplay(
        dataset=replay.dataset,
        threshold=replay.threshold,
        full_score=replay.full_score,
        support_score=replay.support_score,
        full_choose_g=torch.empty(0, dtype=torch.bool),
        support_choose_g=replay.support_choose_g(g, l),
        selected_metadata=replay.provenance or {},
    )


def attach_full_routes(replay: FrozenBReplay, target) -> FrozenBReplay:
    from eargtc.router_bc_split_c.b_replay import route_from_score

    return FrozenBReplay(
        dataset=replay.dataset,
        threshold=replay.threshold,
        full_score=replay.full_score,
        support_score=replay.support_score,
        full_choose_g=route_from_score(
            replay.full_score,
            threshold=replay.threshold,
            gnn_pred=target.gnn_pred,
            llm_pred=target.llm_pred,
        ),
        support_choose_g=replay.support_choose_g,
        selected_metadata=replay.selected_metadata,
    )


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _rows(domain, probability: torch.Tensor, target) -> dict:
    ids = domain.node_ids
    selected = torch.where(
        domain.choose_g,
        target.gnn_pred[ids],
        target.llm_pred[ids],
    )
    return {
        "node_ids": ids.cpu(),
        "score": probability.float().cpu(),
        "correct": domain.target.bool().cpu(),
        "choose_g": domain.choose_g.bool().cpu(),
        "selected_pred": selected.long().cpu(),
        "state": list(domain.state),
    }


def train_conflict_models(
    *,
    provider_id: str,
    original_supports: dict,
    targets: dict,
    replays: dict[str, FrozenBReplay],
    output_root: str | Path,
    resume: bool = True,
) -> dict:
    """Train five conflict modules with the exact finalized Split-C v3 contract."""

    root = Path(output_root) / "c_conflict" / provider_id
    root.mkdir(parents=True, exist_ok=True)
    support_domains = build_support_feature_domains(original_supports, replays)
    summary = {}
    for dataset in DATASETS:
        payload_path = root / dataset / "scores.pt"
        metrics_path = root / dataset / "metrics.json"
        if resume and payload_path.exists() and metrics_path.exists():
            summary[dataset] = json.loads(metrics_path.read_text(encoding="utf-8"))
            continue
        started = time.time()
        sources_by_view = {
            view: [support_domains[view][name]["conflict"] for name in DATASETS if name != dataset]
            for view in ("raw", "rank")
        }
        selection = select_reliability_config(
            sources_by_view, l2_grid=L2_GRID, prior_grid=PRIOR_GRID
        )
        best = selection["best"]
        view = str(best["view"])
        model = fit_reliability_model(
            sources_by_view[view],
            support_domains[view][dataset]["conflict"],
            ReliabilityConfig(float(best["l2"]), float(best["prior_precision"])),
        )
        target = targets[dataset]
        if bool((target.labels[target.test_ids] >= 0).any()):
            raise RuntimeError(
                f"sealed training requires masked test labels: {provider_id}/{dataset}"
            )
        replay = replays[dataset]
        target_views = {
            "raw": (target.x, target.feature_names),
            "rank": _rank_target(target),
        }
        validation = _query_domain(
            target, replay, target.val_ids, view=view, module="conflict", view_values=target_views
        )
        split = deterministic_calibration_threshold_split(validation.node_ids, dataset, "conflict")
        validation_logits = model.logits(validation.evidence)
        calibrator = fit_identity_shrunk_platt(
            validation_logits[split.calibration],
            validation.target[split.calibration],
            regularization=CALIBRATION_REGULARIZATION,
        )
        validation_score = calibrator.probabilities_from_logits(validation_logits)
        payload = {
            "schema_version": 1,
            "provider_id": provider_id,
            "dataset": dataset,
            "module": "conflict",
            "validation": _rows(validation, validation_score, target),
            "validation_calibration_mask": split.calibration.cpu(),
            "validation_threshold_mask": split.threshold.cpu(),
            "selection": selection,
            "model": model.to_dict(),
            "calibrator": calibrator.to_dict(),
            "control": {
                "l2_grid": list(L2_GRID),
                "prior_grid": list(PRIOR_GRID),
                "calibration_regularization": CALIBRATION_REGULARIZATION,
                "source_selection": "source_lodo_macro_nll",
                "target_adaptation": "scalar_intercept",
                "validation_split": "sha256(dataset|module|node_id)",
                "sealed_phase": "source+target-validation only; test labels masked",
            },
        }
        _atomic_torch_save(payload, payload_path)
        metric = {
            "provider_id": provider_id,
            "dataset": dataset,
            "elapsed_seconds": time.time() - started,
            "selection": selection,
            "calibrator": calibrator.to_dict(),
            "validation_count": int(validation.target.numel()),
            "calibration_count": int(split.calibration.sum()),
            "threshold_count": int(split.threshold.sum()),
            "test_metrics": "ABSENT_BY_DESIGN_UNTIL_FREEZE",
            "scores_sha256": sha256_file(payload_path),
        }
        atomic_write_json(metric, metrics_path)
        summary[dataset] = metric
        print(
            f"[{provider_id}/{dataset}] conflict selection artifact complete "
            f"view={view} elapsed={metric['elapsed_seconds']:.1f}s",
            flush=True,
        )
    atomic_write_json(summary, root / "summary.json")
    return summary


__all__ = [
    "CALIBRATION_REGULARIZATION", "L2_GRID", "PRIOR_GRID", "attach_full_routes",
    "selected_correctness_target", "to_frozen_replay", "train_conflict_models",
]
