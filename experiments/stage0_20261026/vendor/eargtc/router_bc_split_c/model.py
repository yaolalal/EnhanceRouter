"""Regularized cross-domain models for split Scheme-C reliability."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from eargtc.router_bc.logistic import (
    LogisticModel,
    adapt_scalar_intercept,
    domain_balanced_weights,
    fit_logistic,
)
from eargtc.router_bc.metrics import binary_auroc, probability_metrics
from eargtc.router_bc.training import Standardizer, fit_standardizer

from .data import ReliabilityDomain


@dataclass(frozen=True)
class ReliabilityConfig:
    l2: float
    prior_precision: float

    def __post_init__(self) -> None:
        if float(self.l2) < 0.0 or float(self.prior_precision) < 0.0:
            raise ValueError("regularization strengths must be non-negative")


@dataclass(frozen=True)
class FittedReliabilityModel:
    model: LogisticModel
    standardizer: Standardizer
    config: ReliabilityConfig
    module: str
    feature_names: list[str]

    def logits(self, evidence: torch.Tensor) -> torch.Tensor:
        value = _validate_prediction_matrix(evidence, len(self.feature_names))
        return self.model.logits(self.standardizer.transform(value))

    def probabilities(self, evidence: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(evidence)).to(torch.float32)

    def to_dict(self) -> dict:
        return {
            "module": self.module,
            "feature_names": list(self.feature_names),
            **asdict(self.config),
            "coef": self.model.coef.tolist(),
            "intercept": float(self.model.intercept),
            "adaptation_bias": float(self.model.adaptation_bias),
            "standardizer_mean": self.standardizer.mean.tolist(),
            "standardizer_scale": self.standardizer.scale.tolist(),
        }


def _validate_prediction_matrix(evidence: torch.Tensor, width: int) -> torch.Tensor:
    value = torch.as_tensor(evidence, dtype=torch.float64)
    if value.ndim != 2 or value.size(1) != width:
        raise ValueError(f"expected evidence [N,{width}], got {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("evidence contains non-finite values")
    return value


def _validate_domains(domains: list[ReliabilityDomain]) -> tuple[str, list[str]]:
    if not domains:
        raise ValueError("at least one reliability domain is required")
    module = domains[0].module
    names = list(domains[0].feature_names)
    seen: set[str] = set()
    for domain in domains:
        if domain.module != module:
            raise ValueError("reliability modules cannot be mixed")
        if domain.feature_names != names:
            raise ValueError("feature names differ across reliability domains")
        if domain.dataset in seen:
            raise ValueError(f"duplicate reliability domain: {domain.dataset}")
        if domain.target.numel() == 0:
            raise ValueError(f"empty reliability domain: {domain.dataset}/{module}")
        seen.add(domain.dataset)
    return module, names


def _stack(
    domains: list[ReliabilityDomain],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    _validate_domains(domains)
    return (
        torch.cat([domain.evidence for domain in domains]),
        torch.cat([domain.target.to(torch.float64) for domain in domains]),
        [domain.dataset for domain in domains for _ in range(domain.target.numel())],
    )


def _fit_shared(
    source_domains: list[ReliabilityDomain], config: ReliabilityConfig
) -> tuple[Standardizer, LogisticModel]:
    x, y, domain_names = _stack(source_domains)
    weights = domain_balanced_weights(domain_names)
    standardizer = fit_standardizer(x, weights)
    model = fit_logistic(
        standardizer.transform(x),
        y,
        sample_weight=weights,
        l2=float(config.l2),
    )
    return standardizer, model


def _aurc(probability: torch.Tensor, target: torch.Tensor) -> float:
    order = torch.argsort(probability, descending=True, stable=True)
    ranked = target[order].to(torch.float64)
    risks = 1.0 - torch.cumsum(ranked, 0) / torch.arange(
        1, ranked.numel() + 1, dtype=torch.float64
    )
    return float(risks.mean())


def honest_heldout_probabilities(
    source_domains: list[ReliabilityDomain],
    held_domain: ReliabilityDomain,
    config: ReliabilityConfig,
) -> torch.Tensor:
    """Predict held support rows while excluding each row's own target label."""

    module, names = _validate_domains(source_domains)
    if held_domain.module != module or held_domain.feature_names != names:
        raise ValueError("held domain is incompatible with source domains")
    standardizer, shared = _fit_shared(source_domains, config)
    held_x = standardizer.transform(held_domain.evidence)
    output = []
    for index in range(held_domain.target.numel()):
        keep = torch.ones(held_domain.target.numel(), dtype=torch.bool)
        keep[index] = False
        adapted = adapt_scalar_intercept(
            shared,
            held_x[keep],
            held_domain.target[keep].to(torch.float64),
            prior_precision=float(config.prior_precision),
        )
        output.append(float(adapted.probabilities(held_x[index : index + 1])[0]))
    return torch.tensor(output, dtype=torch.float32)


def fit_reliability_model(
    source_domains: list[ReliabilityDomain],
    target_support: ReliabilityDomain,
    config: ReliabilityConfig,
) -> FittedReliabilityModel:
    module, names = _validate_domains(source_domains)
    if target_support.module != module or target_support.feature_names != names:
        raise ValueError("target support is incompatible with source domains")
    standardizer, shared = _fit_shared(source_domains, config)
    adapted = adapt_scalar_intercept(
        shared,
        standardizer.transform(target_support.evidence),
        target_support.target.to(torch.float64),
        prior_precision=float(config.prior_precision),
    )
    return FittedReliabilityModel(
        model=adapted,
        standardizer=standardizer,
        config=config,
        module=module,
        feature_names=names,
    )


def select_reliability_config(
    domains_by_view: dict[str, list[ReliabilityDomain]],
    *,
    l2_grid: tuple[float, ...],
    prior_grid: tuple[float, ...],
) -> dict:
    """Select view and regularization using source-only leave-domain-out NLL."""

    if not domains_by_view or not l2_grid or not prior_grid:
        raise ValueError("selection requires views and non-empty parameter grids")
    trials = []
    for view, domains in sorted(domains_by_view.items()):
        if len(domains) < 2:
            raise ValueError("source LODO selection needs at least two domains")
        _validate_domains(domains)
        expected_names = {domain.dataset for domain in domains}
        for l2 in l2_grid:
            for prior in prior_grid:
                config = ReliabilityConfig(float(l2), float(prior))
                folds = []
                for held in domains:
                    train = [
                        domain for domain in domains if domain.dataset != held.dataset
                    ]
                    probability = honest_heldout_probabilities(train, held, config)
                    metrics = probability_metrics(probability, held.target)
                    folds.append(
                        {
                            "heldout_domain": held.dataset,
                            "aurc": _aurc(probability, held.target),
                            **metrics,
                        }
                    )
                if {fold["heldout_domain"] for fold in folds} != expected_names:
                    raise AssertionError("incomplete source LODO folds")
                aurocs = [fold["auroc"] for fold in folds if fold["auroc"] is not None]
                trials.append(
                    {
                        "view": str(view),
                        **asdict(config),
                        "macro_nll": sum(float(fold["nll"]) for fold in folds)
                        / len(folds),
                        "macro_brier": sum(float(fold["brier"]) for fold in folds)
                        / len(folds),
                        "macro_aurc": sum(float(fold["aurc"]) for fold in folds)
                        / len(folds),
                        "macro_auroc": (
                            sum(float(value) for value in aurocs) / len(aurocs)
                            if aurocs
                            else None
                        ),
                        "folds": folds,
                    }
                )
    best = min(
        trials,
        key=lambda row: (
            row["macro_nll"],
            row["macro_brier"],
            row["macro_aurc"],
            -(row["macro_auroc"] if row["macro_auroc"] is not None else -1.0),
            row["view"],
            row["l2"],
            row["prior_precision"],
        ),
    )
    return {
        "selection": "source_lodo_macro_nll",
        "best": {
            "view": best["view"],
            "l2": best["l2"],
            "prior_precision": best["prior_precision"],
        },
        "best_metrics": {
            "macro_nll": best["macro_nll"],
            "macro_brier": best["macro_brier"],
            "macro_aurc": best["macro_aurc"],
            "macro_auroc": best["macro_auroc"],
        },
        "trials": trials,
    }


__all__ = [
    "FittedReliabilityModel",
    "ReliabilityConfig",
    "fit_reliability_model",
    "honest_heldout_probabilities",
    "select_reliability_config",
]
