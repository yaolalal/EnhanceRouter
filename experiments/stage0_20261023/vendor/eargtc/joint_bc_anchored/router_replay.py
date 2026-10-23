from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from eargtc.joint_bc_anchored.provenance import (
    DATASETS,
    FrozenRouterSource,
    reject_forbidden_source,
    verify_frozen_source,
)


LABEL_FREE_NODE_KEYS: set[str] = {
    "node_id",
    "gnn_pred",
    "llm_pred",
    "agreement",
    "scheme_b_route",
    "scheme_b_selected_pred",
    "scheme_b_preference_score",
    "scheme_b_threshold",
    "scheme_c_factorized_trust",
    "scheme_c_direct_trust",
    "scheme_c_selected_trust",
    "scheme_c_checkpoint",
}


class ReplayMismatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenRouterNode:
    node_id: int
    gnn_pred: int
    llm_pred: int
    agreement: bool
    scheme_b_route: str
    scheme_b_selected_pred: int
    scheme_b_preference_score: float
    scheme_b_threshold: float
    scheme_c_factorized_trust: float
    scheme_c_direct_trust: float
    scheme_c_selected_trust: float
    scheme_c_checkpoint: str

    def as_dict(self) -> dict:
        return {
            "node_id": int(self.node_id),
            "gnn_pred": int(self.gnn_pred),
            "llm_pred": int(self.llm_pred),
            "agreement": bool(self.agreement),
            "scheme_b_route": self.scheme_b_route,
            "scheme_b_selected_pred": int(self.scheme_b_selected_pred),
            "scheme_b_preference_score": float(self.scheme_b_preference_score),
            "scheme_b_threshold": float(self.scheme_b_threshold),
            "scheme_c_factorized_trust": float(self.scheme_c_factorized_trust),
            "scheme_c_direct_trust": float(self.scheme_c_direct_trust),
            "scheme_c_selected_trust": float(self.scheme_c_selected_trust),
            "scheme_c_checkpoint": self.scheme_c_checkpoint,
        }


@dataclass(frozen=True)
class RouterReplayResult:
    dataset: str
    nodes: tuple[FrozenRouterNode, ...]
    split_ids: dict[str, list[int]]
    tensors: dict[str, torch.Tensor]
    replay_audit: dict


def build_nodes_from_scores(
    *,
    gnn_pred: torch.Tensor,
    llm_pred: torch.Tensor,
    preference_score: torch.Tensor,
    threshold: float,
    factorized_trust: torch.Tensor,
    direct_trust: torch.Tensor,
    selected_checkpoint: str,
) -> tuple[FrozenRouterNode, ...]:
    if selected_checkpoint not in {"factorized", "direct"}:
        raise ValueError("selected_checkpoint must be 'factorized' or 'direct'")
    tensors = [
        torch.as_tensor(value).view(-1)
        for value in (
            gnn_pred,
            llm_pred,
            preference_score,
            factorized_trust,
            direct_trust,
        )
    ]
    if not tensors or len({int(value.numel()) for value in tensors}) != 1:
        raise ValueError("all Router score tensors must have the same row count")
    gnn, llm, preference, factorized, direct = tensors
    agreement = gnn == llm
    choose_gnn = agreement | (preference >= float(threshold))
    selected = factorized if selected_checkpoint == "factorized" else direct
    nodes = []
    for node_id in range(gnn.numel()):
        route_gnn = bool(choose_gnn[node_id])
        values = (preference[node_id], factorized[node_id], direct[node_id], selected[node_id])
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"non-finite Router score at node {node_id}")
        nodes.append(
            FrozenRouterNode(
                node_id=node_id,
                gnn_pred=int(gnn[node_id]),
                llm_pred=int(llm[node_id]),
                agreement=bool(agreement[node_id]),
                scheme_b_route="gnn" if route_gnn else "llm",
                scheme_b_selected_pred=int(gnn[node_id] if route_gnn else llm[node_id]),
                scheme_b_preference_score=float(preference[node_id]),
                scheme_b_threshold=float(threshold),
                scheme_c_factorized_trust=float(factorized[node_id]),
                scheme_c_direct_trust=float(direct[node_id]),
                scheme_c_selected_trust=float(selected[node_id]),
                scheme_c_checkpoint=selected_checkpoint,
            )
        )
    return tuple(nodes)


def compare_saved_test_rows(result: RouterReplayResult, saved_path: Path) -> dict:
    saved_rows = []
    with Path(saved_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                saved_rows.append(json.loads(line))
    saved_by_id = {int(row["node_id"]): row for row in saved_rows}
    if len(saved_by_id) != len(saved_rows):
        raise ReplayMismatchError("saved decisions contain duplicate node ids")
    query_ids = {int(value) for value in result.split_ids["query"]}
    if set(saved_by_id) != query_ids:
        missing = sorted(query_ids - set(saved_by_id))[:10]
        extra = sorted(set(saved_by_id) - query_ids)[:10]
        raise ReplayMismatchError(
            f"node set mismatch: missing={missing}, extra={extra}, "
            f"expected_count={len(query_ids)}, saved_count={len(saved_by_id)}"
        )
    result_by_id = {node.node_id: node for node in result.nodes}
    exact_fields = {
        "gnn_pred": "gnn_pred",
        "llm_pred": "llm_pred",
        "agreement": "agreement",
        "route": "scheme_b_route",
        "selected_pred": "scheme_b_selected_pred",
        "trust_checkpoint": "scheme_c_checkpoint",
    }
    float_fields = {
        "preference_score": "scheme_b_preference_score",
        "factorized_trust": "scheme_c_factorized_trust",
        "direct_trust": "scheme_c_direct_trust",
        "selected_trust": "scheme_c_selected_trust",
    }
    for saved_name, node_name in exact_fields.items():
        mismatches = [
            node_id
            for node_id, row in saved_by_id.items()
            if row[saved_name] != getattr(result_by_id[node_id], node_name)
        ]
        if mismatches:
            raise ReplayMismatchError(
                f"exact mismatch for {saved_name} at node ids {mismatches[:10]}"
            )
    max_abs_error = {}
    for saved_name, node_name in float_fields.items():
        errors = {
            node_id: abs(float(row[saved_name]) - float(getattr(result_by_id[node_id], node_name)))
            for node_id, row in saved_by_id.items()
        }
        maximum = max(errors.values(), default=0.0)
        max_abs_error[saved_name] = maximum
        if maximum > 1e-6:
            offending = max(errors, key=errors.get)
            raise ReplayMismatchError(
                f"{saved_name} max_abs_error={maximum:.12g} exceeds 1e-6 at node {offending}"
            )
    return {
        "dataset": result.dataset,
        "node_count": len(saved_rows),
        "node_set_exact": True,
        "exact_fields": sorted(exact_fields),
        "max_abs_error": max_abs_error,
        "tolerance": 1e-6,
        "all_checks_pass": True,
    }


def replay_dataset(dataset: str, source: FrozenRouterSource) -> RouterReplayResult:
    """Refit the frozen legacy Router procedure and return label-free full-node scores."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")
    verify_frozen_source(source)

    # Import only the frozen pre-structural-fix implementation.  The prohibited
    # structural-fix packages are deliberately absent from this module.
    from scripts.router_bc_improved import run_controlled_search as frozen_runner

    frozen_runner.configure_router_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    bc_config = json.loads((source.bc_root / "config.json").read_text(encoding="utf-8"))
    args = SimpleNamespace(
        input_run_root=source.upstream_root,
        dataset_root=Path(bc_config["dataset_root"]),
        legacy_l2_grid=(0.01, 0.1, 1.0, 10.0, 100.0),
        legacy_prior_grid=(0.1, 1.0, 10.0, 100.0),
        trust_l2_grid=(0.01, 0.1, 1.0, 10.0),
        trust_prior_grid=(0.1, 1.0, 10.0, 100.0),
        trust_target_precision=0.90,
        seed=42,
    )
    raw_supports = {
        name: frozen_runner.build_support_domain(
            name,
            source.upstream_root / name / "pipeline/router/oof/router_train.jsonl",
        )
        for name in DATASETS
    }
    pipeline = source.upstream_root / dataset / "pipeline"
    target = frozen_runner.load_target_artifacts(
        dataset,
        pipeline,
        args.dataset_root,
        raw_supports[dataset],
    )
    support_views = {
        "raw": raw_supports,
        "rank": {
            name: frozen_runner.rank_support(domain)
            for name, domain in raw_supports.items()
        },
    }
    target_views = {
        "raw": target.x,
        "rank": frozen_runner.percentile_features(
            target.x[:, : len(support_views["rank"][dataset].feature_names)]
        ),
    }

    candidates = []
    meta_by_view = {}
    for view in ("raw", "rank"):
        improved, meta = frozen_runner.improved_candidates(
            view,
            support_views[view],
            dataset,
            target_views[view],
        )
        legacy, _ = frozen_runner.legacy_candidates(
            args,
            view,
            support_views[view],
            dataset,
            target_views[view],
        )
        candidates.extend(improved + legacy)
        meta_by_view[view] = meta

    validation_candidates = [
        frozen_runner.CheckpointCandidate(
            candidate["name"],
            candidate["score"][target.val_ids],
            candidate["view"],
            candidate["complexity"],
        )
        for candidate in candidates
    ]
    selection = frozen_runner.select_b_checkpoint(
        validation_candidates,
        target.labels[target.val_ids],
        target.gnn_pred[target.val_ids],
        target.llm_pred[target.val_ids],
    )
    frozen_runner.validate_selection_record(selection)
    selected_candidate = next(
        row for row in candidates if row["name"] == selection["name"]
    )
    score = selected_candidate["score"]
    threshold = float(selection["threshold"]["threshold"])
    conflict = target.gnn_pred != target.llm_pred
    choose_gnn = (~conflict) | (score >= threshold)
    selected_pred = torch.where(choose_gnn, target.gnn_pred, target.llm_pred)
    validation_correct = selected_pred[target.val_ids] == target.labels[target.val_ids]

    selected_view = selected_candidate["view"]
    view_supports = support_views[selected_view]
    source_supports = [view_supports[name] for name in DATASETS if name != dataset]
    target_support = view_supports[dataset]
    availability_cv = frozen_runner.cross_validate_head(
        source_supports,
        "availability",
        l2_grid=args.trust_l2_grid,
        prior_grid=args.trust_prior_grid,
    )
    agreement_cv = frozen_runner.cross_validate_head(
        source_supports,
        "agreement",
        l2_grid=args.trust_l2_grid,
        prior_grid=args.trust_prior_grid,
    )
    availability = frozen_runner.fit_head(
        source_supports, target_support, "availability", availability_cv
    )
    agreement_head = frozen_runner.fit_head(
        source_supports, target_support, "agreement", agreement_cv
    )
    availability_score = availability.probabilities(target_views[selected_view])
    agreement_score = agreement_head.probabilities(target_views[selected_view])
    q_factorized = frozen_runner.factorized_selected_correctness(
        conflict,
        choose_gnn,
        availability_score,
        score,
        agreement_score,
    )

    combined_trial = frozen_runner.best_trial_for_family(
        meta_by_view[selected_view]["trials"],
        "combined",
        objective="stable_auc",
        max_macro_nll=0.8,
    )
    direct_source_domains = frozen_runner.build_source_trust_domains(
        view_supports,
        dataset,
        frozen_runner.config_from_trial(combined_trial),
    )
    support_score = selected_candidate["support_score"]
    support_choose_gnn = (~target_support.disagreement) | (support_score >= threshold)
    support_correct = torch.where(
        support_choose_gnn,
        target_support.gnn_correct,
        target_support.llm_correct,
    )
    target_trust_support = frozen_runner.TrustDomain(
        dataset,
        frozen_runner.direct_trust_features(
            target_support.x,
            agreement=~target_support.disagreement,
            choose_g=support_choose_gnn,
            preference=support_score,
        ),
        support_correct.float(),
    )
    direct_cv = frozen_runner.select_direct_trust_config(
        direct_source_domains,
        l2_grid=args.trust_l2_grid,
        prior_grid=args.trust_prior_grid,
    )
    direct = frozen_runner.fit_direct_trust(
        direct_source_domains,
        target_trust_support,
        l2=direct_cv["best"]["l2"],
        prior_precision=direct_cv["best"]["prior_precision"],
    )
    direct_x = frozen_runner.direct_trust_features(
        target_views[selected_view],
        agreement=~conflict,
        choose_g=choose_gnn,
        preference=score,
    )
    q_direct = direct.probabilities(direct_x)
    validation_metrics = {
        "factorized": frozen_runner.probability_metrics(
            q_factorized[target.val_ids], validation_correct
        ),
        "direct": frozen_runner.probability_metrics(
            q_direct[target.val_ids], validation_correct
        ),
    }
    selected_trust = min(
        validation_metrics,
        key=lambda name: (
            validation_metrics[name]["nll"],
            0 if name == "factorized" else 1,
        ),
    )
    nodes = build_nodes_from_scores(
        gnn_pred=target.gnn_pred,
        llm_pred=target.llm_pred,
        preference_score=score,
        threshold=threshold,
        factorized_trust=q_factorized,
        direct_trust=q_direct,
        selected_checkpoint=selected_trust,
    )

    prediction = torch.load(
        pipeline / "predictions.pt", map_location="cpu", weights_only=False
    )
    embeddings, embedding_metadata = _load_selected_gnn_embeddings(
        dataset, pipeline, source
    )
    tensors = {
        "gnn_logits": prediction["gnn_logits"].detach().cpu(),
        "llm_logits": prediction["llm_logits"].detach().cpu(),
        "token_ids": prediction["token_ids"].detach().cpu(),
        "gnn_embeddings": embeddings,
        "router_evidence": target.x.detach().cpu(),
        "scheme_b_preference_score": score.detach().cpu(),
        "scheme_c_factorized_trust": q_factorized.detach().cpu(),
        "scheme_c_direct_trust": q_direct.detach().cpu(),
    }
    split_ids = {
        "train": target.train_ids.tolist(),
        "validation": target.val_ids.tolist(),
        "query": target.test_ids.tolist(),
    }
    provisional = RouterReplayResult(
        dataset=dataset,
        nodes=nodes,
        split_ids=split_ids,
        tensors=tensors,
        replay_audit={},
    )
    saved_metrics = json.loads(
        (source.bc_root / dataset / "metrics.json").read_text(encoding="utf-8")
    )
    expected_b_name = saved_metrics["scheme_b"]["selected_metadata"]["name"]
    expected_threshold = float(
        saved_metrics["scheme_b"]["selected_checkpoint"]["threshold"]["threshold"]
    )
    expected_c_name = saved_metrics["scheme_c"]["metrics"]["validation_selection"][
        "selected"
    ]
    if selection["name"] != expected_b_name:
        raise ReplayMismatchError(
            f"selected B checkpoint mismatch: {selection['name']} != {expected_b_name}"
        )
    threshold_error = abs(threshold - expected_threshold)
    if threshold_error > 1e-12:
        raise ReplayMismatchError(
            f"selected B threshold mismatch: error={threshold_error:.12g}"
        )
    if selected_trust != expected_c_name:
        raise ReplayMismatchError(
            f"selected C checkpoint mismatch: {selected_trust} != {expected_c_name}"
        )
    row_audit = compare_saved_test_rows(
        provisional,
        source.bc_root / dataset / "decisions.jsonl",
    )
    replay_audit = {
        **row_audit,
        "candidate_count": len(candidates),
        "selected_b_checkpoint": selection["name"],
        "selected_b_view": selected_view,
        "selected_b_threshold": threshold,
        "selected_b_threshold_abs_error": threshold_error,
        "selected_c_checkpoint": selected_trust,
        "router_evidence_names": list(target.feature_names),
        "embedding_source": embedding_metadata,
        "all_checks_pass": True,
    }
    return RouterReplayResult(
        dataset=dataset,
        nodes=nodes,
        split_ids=split_ids,
        tensors=tensors,
        replay_audit=replay_audit,
    )


def _load_selected_gnn_embeddings(
    dataset: str,
    pipeline: Path,
    source: FrozenRouterSource,
) -> tuple[torch.Tensor, dict[str, str]]:
    config = json.loads((pipeline / "config.json").read_text(encoding="utf-8"))
    manifest_path = Path(config["selected_gnn_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = source.bc_root.resolve().parents[1] / manifest_path
    reject_forbidden_source(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_path = Path(
        manifest["datasets"][dataset]["artifact_paths"]["encoder_and_embeddings"]
    )
    if not artifact_path.is_absolute():
        artifact_path = source.bc_root.resolve().parents[1] / artifact_path
    reject_forbidden_source(artifact_path)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    embeddings = torch.as_tensor(artifact["embeddings"]).detach().cpu()
    return embeddings, {
        "selected_gnn_manifest": str(manifest_path.resolve()),
        "encoder_and_embeddings": str(artifact_path.resolve()),
    }
