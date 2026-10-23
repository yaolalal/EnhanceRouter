from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import torch

from .metrics import decisive_metrics
from .models import (
    CandidateSpec,
    DomainData,
    FittedDecisionHead,
    adapt_on_target_support,
    fit_candidate,
)
from .selection import (
    ThresholdRecord,
    choose_accuracy_threshold,
    choose_balanced_threshold,
)


@dataclass(frozen=True)
class PreparedDecisionDataset:
    name: str
    support_features: dict[str, torch.Tensor]
    support_targets: torch.Tensor
    validation_features: dict[str, torch.Tensor]
    validation_targets: torch.Tensor
    validation_node_ids: torch.Tensor

    def __post_init__(self) -> None:
        support_count = int(torch.as_tensor(self.support_targets).numel())
        validation_count = int(torch.as_tensor(self.validation_targets).numel())
        if validation_count != int(torch.as_tensor(self.validation_node_ids).numel()):
            raise ValueError("validation targets and IDs do not align")
        for value in self.support_features.values():
            if int(value.shape[0]) != support_count:
                raise ValueError("support feature rows do not align")
        for value in self.validation_features.values():
            if int(value.shape[0]) != validation_count:
                raise ValueError("validation feature rows do not align")


@dataclass(frozen=True)
class ValidationEndpoint:
    target: str
    selected_name: str
    feature_key: str
    spec: CandidateSpec
    head: FittedDecisionHead
    accuracy_threshold: ThresholdRecord
    balanced_threshold: ThresholdRecord
    validation_scores: np.ndarray
    validation_targets: torch.Tensor
    validation_node_ids: torch.Tensor
    ranking: list[dict]


def _candidate_name(feature_key: str, spec: CandidateSpec) -> str:
    return f"{feature_key}::{spec.name}"


def select_fit_validate(
    target: str,
    datasets: dict[str, PreparedDecisionDataset],
    feature_keys: list[str],
    candidate_specs: list[CandidateSpec],
    *,
    seed: int = 42,
    adapt_target_bias: bool = True,
) -> ValidationEndpoint:
    if target not in datasets or len(datasets) < 3:
        raise ValueError("selection requires a target and at least two sources")
    sources = [value for name, value in datasets.items() if name != target]
    complexity = {
        "logistic": 1,
        "residual_logistic": 2,
        "rbf": 3,
        "extra_trees": 4,
        "mlp": 5,
    }
    ranking = []
    for feature_key in feature_keys:
        for source in sources + [datasets[target]]:
            if feature_key not in source.support_features:
                raise ValueError(f"{source.name} lacks feature key {feature_key}")
        for spec in candidate_specs:
            if spec.family == "residual_logistic":
                if spec.base_width is None:
                    continue
                width = int(sources[0].support_features[feature_key].shape[1])
                if not 0 < int(spec.base_width) < width:
                    continue
            folds = []
            for held_index, held in enumerate(sources):
                train_domains = [
                    DomainData(
                        source.name,
                        source.support_features[feature_key],
                        source.support_targets,
                    )
                    for index, source in enumerate(sources)
                    if index != held_index
                ]
                head = fit_candidate(spec, train_domains, seed=seed + held_index)
                scores = head.predict_proba(held.support_features[feature_key])
                result = decisive_metrics(scores, held.support_targets, 0.5)
                recalls = [
                    value
                    for value in (
                        result["gnn_only_recall"],
                        result["llm_only_recall"],
                    )
                    if isfinite(value)
                ]
                folds.append(
                    {
                        "domain": held.name,
                        "decisive_accuracy": result["decisive_accuracy"],
                        "minimum_recall": min(recalls) if recalls else 0.0,
                        "nll": result["nll"],
                    }
                )
            ranking.append(
                {
                    "name": _candidate_name(feature_key, spec),
                    "feature_key": feature_key,
                    "spec": spec,
                    "macro_accuracy": float(
                        np.mean([fold["decisive_accuracy"] for fold in folds])
                    ),
                    "macro_minimum_recall": float(
                        np.mean([fold["minimum_recall"] for fold in folds])
                    ),
                    "macro_nll": float(np.mean([fold["nll"] for fold in folds])),
                    "complexity": complexity[spec.family],
                    "folds": folds,
                }
            )
    best_accuracy = max(row["macro_accuracy"] for row in ranking)
    eligible = [row for row in ranking if row["macro_accuracy"] >= best_accuracy - 0.005]
    winner = sorted(
        eligible,
        key=lambda row: (
            -row["macro_minimum_recall"],
            row["macro_nll"],
            row["complexity"],
            row["name"],
        ),
    )[0]
    feature_key = winner["feature_key"]
    spec = winner["spec"]
    source_domains = [
        DomainData(
            source.name,
            source.support_features[feature_key],
            source.support_targets,
        )
        for source in sources
    ]
    head = fit_candidate(spec, source_domains, seed=seed)
    target_data = datasets[target]
    if adapt_target_bias:
        head = adapt_on_target_support(
            head,
            target_data.support_features[feature_key],
            target_data.support_targets,
            max_abs_bias=1.0,
            ridge=5.0,
        )
    validation_scores = head.predict_proba(
        target_data.validation_features[feature_key]
    )
    accuracy_threshold = choose_accuracy_threshold(
        validation_scores, target_data.validation_targets
    )
    balanced_threshold = choose_balanced_threshold(
        validation_scores, target_data.validation_targets
    )
    safe_ranking = [
        {
            **{key: value for key, value in row.items() if key != "spec"},
            "spec": row["spec"].__dict__,
        }
        for row in ranking
    ]
    return ValidationEndpoint(
        target=target,
        selected_name=winner["name"],
        feature_key=feature_key,
        spec=spec,
        head=head,
        accuracy_threshold=accuracy_threshold,
        balanced_threshold=balanced_threshold,
        validation_scores=validation_scores,
        validation_targets=target_data.validation_targets,
        validation_node_ids=target_data.validation_node_ids,
        ranking=safe_ranking,
    )


__all__ = [
    "PreparedDecisionDataset",
    "ValidationEndpoint",
    "select_fit_validate",
]
