from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from eargtc.router_bc.metrics import binary_auroc
from eargtc.router_bc.training import Standardizer, fit_standardizer

from .adapter import AdaptedPreference, DomainBasis, adapt_low_rank, build_domain_basis
from .preference import PreferenceModel, fit_pairwise_preference, pairwise_domain_weights


@dataclass(frozen=True)
class PreferenceDomain:
    name: str
    x: torch.Tensor
    y: torch.Tensor

    def __post_init__(self):
        value = torch.as_tensor(self.x, dtype=torch.float64)
        target = torch.as_tensor(self.y, dtype=torch.float64).view(-1)
        if value.ndim != 2 or value.size(0) != target.numel() or target.numel() == 0:
            raise ValueError("PreferenceDomain expects non-empty X[N,D] and y[N]")
        object.__setattr__(self, "x", value)
        object.__setattr__(self, "y", target)


@dataclass(frozen=True)
class PreferenceConfig:
    l2: float
    rank_weight: float
    adapter_rank: int
    prior_precision: float


@dataclass(frozen=True)
class PreferenceSystem:
    scaler: Standardizer
    base: PreferenceModel
    basis: DomainBasis
    adapted: AdaptedPreference

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapted.probabilities(self.scaler.transform(x))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapted.logits(self.scaler.transform(x))


def _stack(domains: list[PreferenceDomain]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    x = torch.cat([domain.x for domain in domains])
    y = torch.cat([domain.y for domain in domains])
    names = [domain.name for domain in domains for _ in range(domain.y.numel())]
    return x, y, names


def _fit_shared(
    train_domains: list[PreferenceDomain], config: PreferenceConfig
) -> tuple[Standardizer, PreferenceModel, DomainBasis]:
    x, y, names = _stack(train_domains)
    scaler = fit_standardizer(x, pairwise_domain_weights(names))
    standardized = scaler.transform(x)
    base = fit_pairwise_preference(
        standardized,
        y,
        names,
        l2=config.l2,
        rank_weight=config.rank_weight,
    )
    domain_coefs = []
    for domain in train_domains:
        if int(torch.unique(domain.y).numel()) < 2:
            domain_coefs.append(base.coef.clone())
            continue
        local = fit_pairwise_preference(
            scaler.transform(domain.x),
            domain.y,
            [domain.name] * domain.y.numel(),
            l2=config.l2,
            rank_weight=config.rank_weight,
        )
        domain_coefs.append(local.coef)
    basis = build_domain_basis(base.coef, domain_coefs, config.adapter_rank)
    return scaler, base, basis


def fit_preference_system(
    train_domains: list[PreferenceDomain],
    target_support: PreferenceDomain,
    config: PreferenceConfig,
) -> PreferenceSystem:
    scaler, base, basis = _fit_shared(train_domains, config)
    adapted = adapt_low_rank(
        base,
        basis,
        scaler.transform(target_support.x),
        target_support.y,
        precision=config.prior_precision,
    )
    return PreferenceSystem(scaler, base, basis, adapted)


def honest_heldout_scores(
    train_domains: list[PreferenceDomain],
    held_domain: PreferenceDomain,
    config: PreferenceConfig,
) -> torch.Tensor:
    scaler, base, basis = _fit_shared(train_domains, config)
    return _honest_scores_from_shared(
        scaler, base, basis, held_domain, config.prior_precision
    )


def _honest_scores_from_shared(
    scaler: Standardizer,
    base: PreferenceModel,
    basis: DomainBasis,
    held_domain: PreferenceDomain,
    prior_precision: float,
) -> torch.Tensor:
    held_x = scaler.transform(held_domain.x)
    probability = []
    for row in range(held_domain.y.numel()):
        keep = torch.ones(held_domain.y.numel(), dtype=torch.bool)
        keep[row] = False
        adapted = adapt_low_rank(
            base,
            basis,
            held_x[keep],
            held_domain.y[keep],
            precision=prior_precision,
        )
        probability.append(float(adapted.probabilities(held_x[row : row + 1])[0]))
    return torch.tensor(probability, dtype=torch.float32)


def honest_all_row_scores(
    train_domains: list[PreferenceDomain],
    held_decisive: PreferenceDomain,
    held_all_x: torch.Tensor,
    decisive_row_ids: torch.Tensor,
    config: PreferenceConfig,
) -> torch.Tensor:
    """Score all held rows while excluding each decisive row's own label."""
    all_x = torch.as_tensor(held_all_x, dtype=torch.float64)
    row_ids = torch.as_tensor(decisive_row_ids, dtype=torch.long).view(-1)
    if all_x.ndim != 2 or row_ids.numel() != held_decisive.y.numel():
        raise ValueError("decisive_row_ids must map every decisive target row")
    scaler, base, basis = _fit_shared(train_domains, config)
    decisive_x = scaler.transform(held_decisive.x)
    full = adapt_low_rank(
        base,
        basis,
        decisive_x,
        held_decisive.y,
        precision=config.prior_precision,
    )
    probability = full.probabilities(scaler.transform(all_x))
    for position, row_id in enumerate(row_ids.tolist()):
        keep = torch.ones(held_decisive.y.numel(), dtype=torch.bool)
        keep[position] = False
        adapted = adapt_low_rank(
            base,
            basis,
            decisive_x[keep],
            held_decisive.y[keep],
            precision=config.prior_precision,
        )
        probability[row_id] = adapted.probabilities(
            scaler.transform(all_x[row_id : row_id + 1])
        )[0]
    return probability


def _nll(probability: torch.Tensor, target: torch.Tensor) -> float:
    p = probability.to(torch.float64).clamp(1e-7, 1.0 - 1e-7)
    y = target.to(torch.float64)
    return float((-(y * p.log() + (1.0 - y) * (1.0 - p).log())).mean().item())


def select_preference_config(
    domains: list[PreferenceDomain], configs: list[PreferenceConfig]
) -> dict:
    if len(domains) < 2 or not configs:
        raise ValueError("Meta-selection needs at least two domains and one config")
    trials = []
    shared_cache = {}
    for config in configs:
        folds = []
        for held in domains:
            train = [domain for domain in domains if domain.name != held.name]
            cache_key = (
                held.name,
                float(config.l2),
                float(config.rank_weight),
                int(config.adapter_rank),
            )
            if cache_key not in shared_cache:
                shared_cache[cache_key] = _fit_shared(train, config)
            scaler, base, basis = shared_cache[cache_key]
            score = _honest_scores_from_shared(
                scaler, base, basis, held, config.prior_precision
            )
            auc = binary_auroc(score, held.y.bool())
            folds.append(
                {
                    "heldout_domain": held.name,
                    "count": int(held.y.numel()),
                    "auroc": auc,
                    "nll": _nll(score, held.y),
                }
            )
        valid_auc = [fold["auroc"] for fold in folds if fold["auroc"] is not None]
        trials.append(
            {
                **asdict(config),
                "macro_auroc": sum(valid_auc) / len(valid_auc) if valid_auc else None,
                "macro_nll": sum(fold["nll"] for fold in folds) / len(folds),
                "folds": folds,
            }
        )
    best = min(
        trials,
        key=lambda row: (
            -(row["macro_auroc"] if row["macro_auroc"] is not None else -1.0),
            row["macro_nll"],
            row["adapter_rank"],
            row["rank_weight"],
            row["l2"],
            row["prior_precision"],
        ),
    )
    fields = asdict(configs[0]).keys()
    return {
        "selection": "nested_source_lodo_macro_auroc",
        "best": {key: best[key] for key in fields},
        "best_metrics": {
            "macro_auroc": best["macro_auroc"],
            "macro_nll": best["macro_nll"],
        },
        "trials": trials,
    }
