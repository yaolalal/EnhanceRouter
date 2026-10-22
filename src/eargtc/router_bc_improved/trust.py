from __future__ import annotations

from dataclasses import dataclass

import torch

from eargtc.router_bc.logistic import LogisticModel, adapt_scalar_intercept, domain_balanced_weights, fit_logistic
from eargtc.router_bc.metrics import probability_metrics
from eargtc.router_bc.training import Standardizer, fit_standardizer


def factorized_selected_correctness(conflict, choose_g, availability, preference, agreement_correctness):
    conflict = torch.as_tensor(conflict, dtype=torch.bool)
    choose_g = torch.as_tensor(choose_g, dtype=torch.bool)
    a = torch.as_tensor(availability).clamp(0.0, 1.0)
    c = torch.as_tensor(preference).clamp(0.0, 1.0)
    u = torch.as_tensor(agreement_correctness).clamp(0.0, 1.0)
    return torch.where(conflict, torch.where(choose_g, a * c, a * (1.0 - c)), u)


def direct_trust_features(evidence, *, agreement, choose_g, preference):
    x = torch.as_tensor(evidence, dtype=torch.float64)
    same = torch.as_tensor(agreement, dtype=torch.float64).view(-1)
    route_g = torch.as_tensor(choose_g, dtype=torch.float64).view(-1)
    c = torch.as_tensor(preference, dtype=torch.float64).view(-1)
    if x.ndim != 2 or not (x.size(0) == same.numel() == route_g.numel() == c.numel()):
        raise ValueError("Direct trust feature inputs must have equal row counts")
    selected = torch.where(route_g.bool(), c, 1.0 - c)
    return torch.cat([x, same[:, None], route_g[:, None], c[:, None], selected[:, None]], dim=1).to(torch.float32)


@dataclass(frozen=True)
class TrustDomain:
    name: str
    x: torch.Tensor
    y: torch.Tensor

    def __post_init__(self):
        value = torch.as_tensor(self.x, dtype=torch.float64)
        target = torch.as_tensor(self.y, dtype=torch.float64).view(-1)
        if value.ndim != 2 or value.size(0) != target.numel() or target.numel() == 0:
            raise ValueError("TrustDomain expects non-empty X[N,D] and y[N]")
        object.__setattr__(self, "x", value)
        object.__setattr__(self, "y", target)


@dataclass(frozen=True)
class FittedDirectTrust:
    scaler: Standardizer
    shared: LogisticModel
    adapted: LogisticModel
    l2: float
    prior_precision: float

    def probabilities(self, x):
        return self.adapted.probabilities(self.scaler.transform(x))


def _stack(domains):
    x = torch.cat([domain.x for domain in domains])
    y = torch.cat([domain.y for domain in domains])
    names = [domain.name for domain in domains for _ in range(domain.y.numel())]
    return x, y, names


def fit_direct_trust(source_domains, target_support, *, l2, prior_precision):
    if not source_domains:
        raise ValueError("At least one source trust domain is required")
    x, y, names = _stack(source_domains)
    weights = domain_balanced_weights(names)
    scaler = fit_standardizer(x, weights)
    shared = fit_logistic(scaler.transform(x), y, sample_weight=weights, l2=l2)
    adapted = adapt_scalar_intercept(shared, scaler.transform(target_support.x), target_support.y, prior_precision=prior_precision)
    return FittedDirectTrust(scaler, shared, adapted, float(l2), float(prior_precision))


def honest_direct_trust_scores(source_domains, held_domain, *, l2, prior_precision):
    x, y, names = _stack(source_domains)
    weights = domain_balanced_weights(names)
    scaler = fit_standardizer(x, weights)
    shared = fit_logistic(scaler.transform(x), y, sample_weight=weights, l2=l2)
    held_x = scaler.transform(held_domain.x)
    result = []
    for row in range(held_domain.y.numel()):
        keep = torch.ones(held_domain.y.numel(), dtype=torch.bool)
        keep[row] = False
        adapted = adapt_scalar_intercept(shared, held_x[keep], held_domain.y[keep], prior_precision=prior_precision)
        result.append(float(adapted.probabilities(held_x[row : row + 1])[0]))
    return torch.tensor(result, dtype=torch.float32)


def select_direct_trust_config(domains, *, l2_grid, prior_grid):
    trials = []
    for l2 in l2_grid:
        for prior in prior_grid:
            folds = []
            for held in domains:
                source = [domain for domain in domains if domain.name != held.name]
                probability = honest_direct_trust_scores(source, held, l2=l2, prior_precision=prior)
                folds.append({"heldout_domain": held.name, **probability_metrics(probability, held.y)})
            trials.append({
                "l2": float(l2),
                "prior_precision": float(prior),
                "macro_nll": sum(fold["nll"] for fold in folds) / len(folds),
                "macro_brier": sum(fold["brier"] for fold in folds) / len(folds),
                "folds": folds,
            })
    best = min(trials, key=lambda row: (row["macro_nll"], row["macro_brier"], row["l2"], row["prior_precision"]))
    return {
        "selection": "source_lodo_macro_nll",
        "best": {"l2": best["l2"], "prior_precision": best["prior_precision"]},
        "best_metrics": {"macro_nll": best["macro_nll"], "macro_brier": best["macro_brier"]},
        "trials": trials,
    }


def risk_coverage_metrics(probability, correct, *, coverages=(0.10, 0.25, 0.50, 0.75, 1.00)):
    p = torch.as_tensor(probability, dtype=torch.float64).view(-1)
    y = torch.as_tensor(correct, dtype=torch.bool).view(-1)
    if p.numel() != y.numel() or p.numel() == 0:
        raise ValueError("Risk-coverage requires equal non-empty inputs")
    order = torch.argsort(p, descending=True, stable=True)
    count = torch.arange(1, p.numel() + 1, dtype=torch.float64)
    precision = y[order].to(torch.float64).cumsum(0) / count
    precision_at = {}
    for coverage in coverages:
        selected = max(1, min(p.numel(), int(torch.ceil(torch.tensor(coverage * p.numel())).item())))
        precision_at[f"{coverage:.2f}"] = float(precision[selected - 1].item())
    return {"count": int(p.numel()), "aurc": float((1.0 - precision).mean().item()), "precision_at_coverage": precision_at}


def select_trust_threshold(probability, correct, *, target_precision):
    p = torch.as_tensor(probability, dtype=torch.float64).view(-1)
    y = torch.as_tensor(correct, dtype=torch.bool).view(-1)
    if p.numel() != y.numel() or p.numel() == 0:
        raise ValueError("Trust threshold requires equal non-empty inputs")
    target = float(target_precision)
    if not 0.0 <= target <= 1.0:
        raise ValueError("target_precision must be in [0,1]")
    trials = []
    for threshold in torch.unique(p).sort(descending=True).values.tolist():
        mask = p >= float(threshold)
        selected = int(mask.sum().item())
        trials.append({
            "threshold": float(threshold),
            "selected_count": selected,
            "coverage": selected / p.numel(),
            "precision": float(y[mask].float().mean().item()),
        })
    feasible = [row for row in trials if row["precision"] >= target]
    if feasible:
        best = min(feasible, key=lambda row: (-row["selected_count"], -row["precision"], -row["threshold"]))
        source = "official_validation_precision_constraint"
    else:
        best = min(trials, key=lambda row: (-row["precision"], -row["selected_count"], -row["threshold"]))
        source = "official_validation_best_precision_fallback"
    return {**best, "target_precision": target, "source": source}
