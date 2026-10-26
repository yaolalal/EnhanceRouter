from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from eargtc.router_bc.data_impl import SupportDomain, TargetArtifacts
from eargtc.router_bc.hybrid import blend_preference_logits
from eargtc.router_bc.nonlinear import fit_preference_model, select_preference_model
from eargtc.router_bc.training import cross_validate_head, fit_head, head_targets
from eargtc.router_bc_improved.meta import (
    PreferenceConfig,
    PreferenceDomain,
    fit_preference_system,
    select_preference_config,
)
from eargtc.router_bc_improved.runner_utils import (
    best_trial_for_family,
    preference_config_grid,
    validate_selection_record,
)
from eargtc.router_bc_improved.selection import (
    CheckpointCandidate,
    select_b_checkpoint,
)

from .data import rank_a20_query, rank_a20_support


DATASETS = ("cora", "citeseer", "pubmed", "arxiv", "ogbn-products")
FAMILIES = ("baseline", "ranking", "adapter", "combined")
OBJECTIVES = ("auc", "stable_auc", "nll")
RESIDUAL_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class SearchSettings:
    legacy_l2_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    legacy_prior_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    seed: int = 42


@dataclass(frozen=True)
class TargetSelection:
    selection: dict
    selected_score: torch.Tensor
    selected_metadata: dict
    source_meta: dict
    legacy_source_cv: dict


def candidate_blueprint_names() -> list[str]:
    names = []
    for view in ("raw", "rank"):
        names.extend(
            f"improved_{view}_{family}_{objective}"
            for family in FAMILIES
            for objective in OBJECTIVES
        )
        names.extend(
            f"legacy_{view}_lambda_{fraction:.2f}"
            for fraction in RESIDUAL_FRACTIONS
        )
    return names


def preference_domains(
    supports: dict[str, SupportDomain],
) -> dict[str, PreferenceDomain]:
    result = {}
    for name, domain in supports.items():
        mask, target = head_targets(domain, "preference")
        result[name] = PreferenceDomain(name, domain.x[mask], target[mask])
    return result


def _config_from_trial(row: dict) -> PreferenceConfig:
    return PreferenceConfig(
        l2=float(row["l2"]),
        rank_weight=float(row["rank_weight"]),
        adapter_rank=int(row["adapter_rank"]),
        prior_precision=float(row["prior_precision"]),
    )


def _improved_candidates(
    view: str,
    supports: dict[str, SupportDomain],
    target_name: str,
    target_x: torch.Tensor,
    dataset_order: tuple[str, ...],
) -> tuple[list[dict], dict]:
    domains = preference_domains(supports)
    source = [domains[name] for name in dataset_order if name != target_name]
    target_support = domains[target_name]
    meta = select_preference_config(source, preference_config_grid())
    candidates = []
    for family in FAMILIES:
        for objective in OBJECTIVES:
            trial = best_trial_for_family(
                meta["trials"], family, objective=objective, max_macro_nll=0.8
            )
            config = _config_from_trial(trial)
            fitted = fit_preference_system(source, target_support, config)
            candidates.append(
                {
                    "name": f"improved_{view}_{family}_{objective}",
                    "view": view,
                    "family": family,
                    "complexity": int(config.adapter_rank > 0)
                    + int(config.rank_weight > 0),
                    "score": fitted.probabilities(target_x),
                    "config": asdict(config),
                    "source_cv": {
                        "objective": objective,
                        "max_macro_nll": 0.8
                        if objective == "stable_auc"
                        else None,
                        "macro_auroc": trial["macro_auroc"],
                        "macro_nll": trial["macro_nll"],
                    },
                }
            )
    return candidates, meta


def _legacy_candidates(
    settings: SearchSettings,
    view: str,
    supports: dict[str, SupportDomain],
    target_name: str,
    target_x: torch.Tensor,
    dataset_order: tuple[str, ...],
) -> tuple[list[dict], dict]:
    source = [supports[name] for name in dataset_order if name != target_name]
    target_support = supports[target_name]
    linear_cv = cross_validate_head(
        source,
        "preference",
        l2_grid=settings.legacy_l2_grid,
        prior_grid=settings.legacy_prior_grid,
    )
    nonlinear_cv = select_preference_model(source, seed=settings.seed)
    linear = fit_head(source, target_support, "preference", linear_cv)
    nonlinear = fit_preference_model(
        source, target_support, nonlinear_cv, seed=settings.seed
    )
    linear_score = linear.probabilities(target_x)
    nonlinear_score = nonlinear.probabilities(target_x)
    candidates = []
    for index, fraction in enumerate(RESIDUAL_FRACTIONS):
        candidates.append(
            {
                "name": f"legacy_{view}_lambda_{fraction:.2f}",
                "view": view,
                "family": "legacy_residual",
                "complexity": index,
                "score": blend_preference_logits(
                    linear_score, nonlinear_score, fraction
                ),
                "config": {"lambda": fraction},
                "source_cv": {
                    "linear": linear_cv["best"],
                    "nonlinear": nonlinear_cv["best"],
                },
            }
        )
    return candidates, {"linear": linear_cv, "nonlinear": nonlinear_cv}


def select_validation_candidate(
    target: TargetArtifacts, candidates: list[dict]
) -> tuple[dict, dict]:
    validation = [
        CheckpointCandidate(
            row["name"],
            torch.as_tensor(row["score"])[target.val_ids],
            row["view"],
            int(row["complexity"]),
        )
        for row in candidates
    ]
    selection = select_b_checkpoint(
        validation,
        target.labels[target.val_ids],
        target.gnn_pred[target.val_ids],
        target.llm_pred[target.val_ids],
    )
    validate_selection_record(selection)
    selected = next(row for row in candidates if row["name"] == selection["name"])
    return selection, selected


def train_select_target(
    target_name: str,
    raw_supports: dict[str, SupportDomain],
    target: TargetArtifacts,
    *,
    settings: SearchSettings = SearchSettings(),
    dataset_order: tuple[str, ...] = DATASETS,
) -> TargetSelection:
    if tuple(raw_supports) != tuple(dataset_order):
        raise ValueError("support domains must follow the frozen dataset order")
    if target_name not in raw_supports or target.name != target_name:
        raise ValueError("target name does not align with support/query artifacts")
    support_views = {
        "raw": raw_supports,
        "rank": {name: rank_a20_support(domain) for name, domain in raw_supports.items()},
    }
    target_views = {"raw": target.x, "rank": rank_a20_query(target.x)}
    candidates = []
    source_meta = {}
    legacy_cv = {}
    for view in ("raw", "rank"):
        improved, meta = _improved_candidates(
            view,
            support_views[view],
            target_name,
            target_views[view],
            dataset_order,
        )
        legacy_candidates, legacy_record = _legacy_candidates(
            settings,
            view,
            support_views[view],
            target_name,
            target_views[view],
            dataset_order,
        )
        candidates.extend(improved + legacy_candidates)
        source_meta[view] = meta
        legacy_cv[view] = legacy_record
    actual_names = [row["name"] for row in candidates]
    if actual_names != candidate_blueprint_names():
        raise AssertionError("A20 v6 candidate set differs from the frozen blueprint")
    selection, selected = select_validation_candidate(target, candidates)
    metadata = {
        key: selected[key]
        for key in ("name", "view", "family", "complexity", "config", "source_cv")
    }
    return TargetSelection(
        selection=selection,
        selected_score=torch.as_tensor(selected["score"], dtype=torch.float32).cpu(),
        selected_metadata=metadata,
        source_meta=source_meta,
        legacy_source_cv=legacy_cv,
    )


__all__ = [
    "DATASETS",
    "SearchSettings",
    "TargetSelection",
    "candidate_blueprint_names",
    "preference_domains",
    "select_validation_candidate",
    "train_select_target",
]
