"""Read-only replay adapters for frozen A20 Scheme-B endpoints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from eargtc.router_a20_embeddings_v6backbone.data import build_combined_domains
from eargtc.router_a20_embeddings_v6backbone.random_correction import build_corrected_random_domains
from eargtc.router_a20_v6backbone.data import load_a20_domain
from eargtc.router_bc.data_impl import SupportDomain, TargetArtifacts, load_target_artifacts
from eargtc.router_bc.rank import percentile_features
from eargtc.router_bc_split_c.b_replay import (
    FrozenBReplay, _rank_support, _replay_improved, _replay_legacy, route_from_score,
)


DATASETS = ("cora", "citeseer", "pubmed", "arxiv", "ogbn-products")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class EndpointSpec:
    provider_id: str
    dataset: str
    endpoint: str
    scores_path: Path
    scores_sha256: str
    selection_path: Path
    selection_sha256: str


@dataclass(frozen=True)
class BReplay:
    dataset: str
    endpoint: str
    threshold: float
    full_node_ids: torch.Tensor
    full_score: torch.Tensor
    support_node_ids: torch.Tensor
    support_score: torch.Tensor
    provenance: dict | None = None

    def __post_init__(self) -> None:
        full_ids = torch.as_tensor(self.full_node_ids, dtype=torch.long).view(-1).cpu()
        full_score = torch.as_tensor(self.full_score, dtype=torch.float32).view(-1).cpu()
        support_ids = torch.as_tensor(self.support_node_ids, dtype=torch.long).view(-1).cpu()
        support_score = torch.as_tensor(self.support_score, dtype=torch.float32).view(-1).cpu()
        if full_ids.numel() != full_score.numel() or support_ids.numel() != support_score.numel():
            raise ValueError("B replay node IDs and scores must align")
        if torch.unique(full_ids).numel() != full_ids.numel() or torch.unique(support_ids).numel() != support_ids.numel():
            raise ValueError("B replay node IDs must be unique")
        if not bool(torch.isfinite(full_score).all()) or not bool(torch.isfinite(support_score).all()):
            raise ValueError("B replay scores must be finite")
        if not 0.0 < float(self.threshold) < 1.0:
            raise ValueError("B replay threshold must be in (0,1)")
        object.__setattr__(self, "full_node_ids", full_ids)
        object.__setattr__(self, "full_score", full_score)
        object.__setattr__(self, "support_node_ids", support_ids)
        object.__setattr__(self, "support_score", support_score)

    def full_choose_g(self, gnn_pred: torch.Tensor, llm_pred: torch.Tensor) -> torch.Tensor:
        return route_from_score(self.full_score, threshold=self.threshold, gnn_pred=gnn_pred, llm_pred=llm_pred)

    def support_choose_g(self, gnn_pred: torch.Tensor, llm_pred: torch.Tensor) -> torch.Tensor:
        return route_from_score(self.support_score, threshold=self.threshold, gnn_pred=gnn_pred, llm_pred=llm_pred)


def endpoint_spec(provider_id: str, dataset: str, frozen_path: str | Path) -> EndpointSpec:
    manifest = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
    if dataset not in manifest["dataset_endpoints"]:
        raise ValueError(f"unknown frozen dataset: {dataset}")
    if provider_id == "a20ms":
        endpoint = "A20_grid_control"
    elif provider_id == "a20emb_v2":
        endpoint = manifest["dataset_endpoints"][dataset]["primary_source_only"]["config_name"]
    else:
        raise ValueError(f"unknown B provider: {provider_id}")
    row = manifest["dataset_endpoints"][dataset]["endpoints"][endpoint]
    return EndpointSpec(
        provider_id=provider_id,
        dataset=dataset,
        endpoint=endpoint,
        scores_path=Path(row["scores_path"]),
        scores_sha256=str(row["scores_sha256"]),
        selection_path=Path(row["selection_path"]),
        selection_sha256=str(row["selection_sha256"]),
    )


def load_a20_inputs(
    *, upstream_root: str | Path, dataset_root: str | Path, gnn_embedding_root: str | Path,
    mask_test_labels: bool = False,
) -> tuple[dict[str, SupportDomain], dict[str, TargetArtifacts]]:
    supports: dict[str, SupportDomain] = {}
    targets: dict[str, TargetArtifacts] = {}
    for dataset in DATASETS:
        supports[dataset], target = load_a20_domain(
            dataset,
            upstream_root=upstream_root,
            dataset_root=dataset_root,
            gnn_embedding_root=gnn_embedding_root,
        )
        if mask_test_labels:
            labels = target.labels.clone()
            labels[target.test_ids] = -1
            target = replace(target, labels=labels)
        targets[dataset] = target
    return supports, targets


def reconstruct_v6_replay_sealed(
    dataset: str,
    *,
    supports: dict[str, SupportDomain],
    input_run_root: str | Path,
    frozen_b_root: str | Path,
    dataset_root: str | Path,
    seed: int = 42,
    mask_test_labels: bool = True,
) -> tuple[FrozenBReplay, TargetArtifacts, dict]:
    """Replay frozen v6 B without reading test labels or decisions."""

    input_root, frozen_root = Path(input_run_root), Path(frozen_b_root)
    metrics_path = frozen_root / dataset / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metadata = metrics["scheme_b"]["selected_metadata"]
    threshold = float(metrics["scheme_b"]["selected_checkpoint"]["threshold"]["threshold"])
    target = load_target_artifacts(
        dataset, input_root / dataset / "pipeline", dataset_root, supports[dataset]
    )
    views = {"raw": supports, "rank": {name: _rank_support(domain) for name, domain in supports.items()}}
    selected_supports = views[str(metadata["view"])]
    target_x = target.x if metadata["view"] == "raw" else percentile_features(
        target.x[:, :len(selected_supports[dataset].feature_names)]
    )
    if metadata["family"] == "legacy_residual":
        full_score, support_score = _replay_legacy(dataset, selected_supports, target_x, metadata, int(seed))
    else:
        full_score, support_score = _replay_improved(dataset, selected_supports, target_x, metadata)
    support = supports[dataset]
    support_g = torch.tensor([int(row["gnn_pred"]) for row in support.rows])
    support_l = torch.tensor([int(row["llm_pred"]) for row in support.rows])
    replay = FrozenBReplay(
        dataset=dataset,
        threshold=threshold,
        full_score=full_score,
        support_score=support_score,
        full_choose_g=route_from_score(full_score, threshold=threshold, gnn_pred=target.gnn_pred, llm_pred=target.llm_pred),
        support_choose_g=route_from_score(support_score, threshold=threshold, gnn_pred=support_g, llm_pred=support_l),
        selected_metadata=metadata,
    )
    audit = {
        "dataset": dataset,
        "phase": "pre-freeze label-blind v6 B replay",
        "metrics_sha256": sha256_file(metrics_path),
        "test_metrics": "DEFERRED_UNTIL_AFTER_FREEZE",
        "all_checks_pass": True,
    }
    if mask_test_labels:
        labels = target.labels.clone()
        labels[target.test_ids] = -1
        target = replace(target, labels=labels)
    return replay, target, audit


def _selected_views(
    supports: dict[str, SupportDomain], target: TargetArtifacts, metadata: dict
) -> tuple[dict[str, SupportDomain], torch.Tensor]:
    view = str(metadata["view"])
    if view == "raw":
        return supports, target.x
    if view != "rank":
        raise ValueError(f"unknown frozen endpoint view: {view}")
    ranked = {
        name: SupportDomain(
            name=domain.name,
            rows=domain.rows,
            x=percentile_features(domain.x),
            feature_names=domain.feature_names,
            gnn_correct=domain.gnn_correct,
            llm_correct=domain.llm_correct,
            disagreement=domain.disagreement,
        )
        for name, domain in supports.items()
    }
    return ranked, percentile_features(target.x)


def reconstruct_b_replay(
    spec: EndpointSpec,
    *,
    a20_supports: dict[str, SupportDomain],
    a20_targets: dict[str, TargetArtifacts],
    banks: dict | None = None,
    seed: int = 42,
) -> tuple[BReplay, dict]:
    for path, expected in ((spec.scores_path, spec.scores_sha256), (spec.selection_path, spec.selection_sha256)):
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"frozen endpoint hash mismatch: {path}")
    selection = json.loads(spec.selection_path.read_text(encoding="utf-8"))
    cached = torch.load(spec.scores_path, map_location="cpu", weights_only=False)
    if spec.endpoint == "A20_grid_control":
        supports, target = a20_supports, a20_targets[spec.dataset]
        alignment = None
    else:
        if banks is None:
            raise ValueError("embedding replay requires frozen feature banks")
        family, dim_text = spec.endpoint.rsplit("_d", 1)
        dim = int(dim_text)
        if family == "random" and "corrected" in str(spec.scores_path):
            supports, target, alignment = build_corrected_random_domains(
                a20_supports, a20_targets, banks, target_name=spec.dataset, dim=dim
            )
        else:
            supports, target, alignment = build_combined_domains(
                a20_supports, a20_targets, banks,
                target_name=spec.dataset, family=family, dim=dim,
            )
    metadata = selection["selected_metadata"]
    view_supports, target_x = _selected_views(supports, target, metadata)
    if metadata["family"] == "legacy_residual":
        full_score, support_score = _replay_legacy(
            spec.dataset, view_supports, target_x, metadata, int(seed)
        )
    else:
        full_score, support_score = _replay_improved(
            spec.dataset, view_supports, target_x, metadata
        )
    cached_ids = torch.as_tensor(cached["node_ids"], dtype=torch.long).view(-1)
    cached_score = torch.as_tensor(cached["preference_score"], dtype=torch.float32).view(-1)
    if not torch.equal(cached_ids, torch.arange(target.labels.numel())):
        raise RuntimeError("frozen full score IDs are not canonical")
    max_diff = float((full_score.float() - cached_score).abs().max())
    if max_diff > 1e-5:
        raise RuntimeError(f"reconstructed B score mismatch: {spec.dataset}/{spec.endpoint} diff={max_diff}")
    threshold = float(selection["selected_checkpoint"]["threshold"]["threshold"])
    replay = BReplay(
        dataset=spec.dataset,
        endpoint=spec.endpoint,
        threshold=threshold,
        full_node_ids=cached_ids,
        full_score=cached_score,
        support_node_ids=torch.tensor([int(row["node_id"]) for row in supports[spec.dataset].rows]),
        support_score=support_score,
        provenance={
            "provider_id": spec.provider_id,
            "scores_sha256": spec.scores_sha256,
            "selection_sha256": spec.selection_sha256,
            "selected_metadata": metadata,
            "reconstruction_max_abs_diff": max_diff,
            "alignment": alignment,
        },
    )
    return replay, {"supports": supports, "target": target, "selection": selection}


__all__ = [
    "BReplay", "DATASETS", "EndpointSpec", "endpoint_spec", "load_a20_inputs",
    "reconstruct_b_replay", "reconstruct_v6_replay_sealed", "sha256_file",
]
