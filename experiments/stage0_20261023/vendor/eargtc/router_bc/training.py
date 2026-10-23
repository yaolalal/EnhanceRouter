from __future__ import annotations

from dataclasses import dataclass

import torch

from .data_impl import SupportDomain
from .logistic import LogisticModel, adapt_scalar_intercept, domain_balanced_weights, fit_logistic


@dataclass(frozen=True)
class Standardizer:
    mean: torch.Tensor
    scale: torch.Tensor

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x.to(torch.float64) - self.mean) / self.scale


@dataclass(frozen=True)
class FittedHead:
    model: LogisticModel
    standardizer: Standardizer
    l2: float
    prior_precision: float
    cv: dict

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.probabilities(self.standardizer.transform(x))

    def to_dict(self) -> dict:
        return {
            "coef": self.model.coef.tolist(),
            "intercept": self.model.intercept,
            "adaptation_bias": self.model.adaptation_bias,
            "mean": self.standardizer.mean.tolist(),
            "scale": self.standardizer.scale.tolist(),
            "l2": self.l2,
            "prior_precision": self.prior_precision,
            "cv": self.cv,
        }


def fit_standardizer(x: torch.Tensor, weight: torch.Tensor | None = None) -> Standardizer:
    x = x.to(torch.float64)
    if weight is None:
        weight = torch.ones(x.size(0), dtype=torch.float64)
    weight = weight.to(torch.float64)
    weight = weight / weight.sum().clamp_min(1e-12)
    mean = (weight[:, None] * x).sum(dim=0)
    variance = (weight[:, None] * (x - mean).square()).sum(dim=0)
    return Standardizer(mean=mean, scale=variance.sqrt().clamp_min(1e-6))


def head_targets(domain: SupportDomain, head: str) -> tuple[torch.Tensor, torch.Tensor]:
    g, l, conflict = domain.gnn_correct, domain.llm_correct, domain.disagreement
    if head == "preference":
        mask, target = conflict & (g != l), g.to(torch.float32)
    elif head == "availability":
        mask, target = conflict, (g | l).to(torch.float32)
    elif head == "agreement":
        mask, target = ~conflict, g.to(torch.float32)
    else:
        raise ValueError(f"Unknown head: {head}")
    return mask, target


def _stack(domains: list[SupportDomain], head: str):
    xs, ys, names = [], [], []
    for domain in domains:
        mask, target = head_targets(domain, head)
        if bool(mask.any().item()):
            xs.append(domain.x[mask])
            ys.append(target[mask])
            names.extend([domain.name] * int(mask.sum().item()))
    if not xs:
        raise ValueError(f"No training rows for head {head}")
    return torch.cat(xs), torch.cat(ys), names


def _log_loss(prob: torch.Tensor, target: torch.Tensor) -> float:
    p = prob.to(torch.float64).clamp(1e-7, 1.0 - 1e-7)
    y = target.to(torch.float64)
    return float((-(y * p.log() + (1.0 - y) * (1.0 - p).log())).mean().item())


def cross_validate_head(
    domains: list[SupportDomain],
    head: str,
    *,
    l2_grid: tuple[float, ...],
    prior_grid: tuple[float, ...],
) -> dict:
    trials = []
    for l2 in l2_grid:
        for prior in prior_grid:
            fold_rows, all_prob, all_target = [], [], []
            for held in domains:
                train_domains = [domain for domain in domains if domain.name != held.name]
                x_train, y_train, train_names = _stack(train_domains, head)
                weights = domain_balanced_weights(train_names)
                scaler = fit_standardizer(x_train, weights)
                model = fit_logistic(
                    scaler.transform(x_train), y_train, sample_weight=weights, l2=l2
                )
                mask, held_target_all = head_targets(held, head)
                held_x, held_y = held.x[mask], held_target_all[mask]
                fold_prob = []
                for i in range(held_y.numel()):
                    keep = torch.ones(held_y.numel(), dtype=torch.bool)
                    keep[i] = False
                    adapted = adapt_scalar_intercept(
                        model,
                        scaler.transform(held_x[keep]),
                        held_y[keep],
                        prior_precision=prior,
                    )
                    value = adapted.probabilities(scaler.transform(held_x[i : i + 1]))[0]
                    fold_prob.append(float(value))
                probability = torch.tensor(fold_prob, dtype=torch.float32)
                all_prob.append(probability)
                all_target.append(held_y)
                fold_rows.append(
                    {
                        "heldout_domain": held.name,
                        "count": int(held_y.numel()),
                        "nll": _log_loss(probability, held_y),
                        "accuracy": float(
                            ((probability >= 0.5) == held_y.bool()).float().mean().item()
                        ),
                    }
                )
            probability, target = torch.cat(all_prob), torch.cat(all_target)
            trials.append(
                {
                    "l2": float(l2),
                    "prior_precision": float(prior),
                    "nll": _log_loss(probability, target),
                    "accuracy": float(
                        ((probability >= 0.5) == target.bool()).float().mean().item()
                    ),
                    "count": int(target.numel()),
                    "folds": fold_rows,
                }
            )
    best = min(
        trials,
        key=lambda row: (
            row["nll"],
            -row["accuracy"],
            row["l2"],
            row["prior_precision"],
        ),
    )
    return {"head": head, "selection": "source_lodo_nll", "best": best, "trials": trials}


def fit_head(
    source_domains: list[SupportDomain],
    target_support: SupportDomain,
    head: str,
    cv: dict,
) -> FittedHead:
    best = cv["best"]
    x, y, names = _stack(source_domains, head)
    weights = domain_balanced_weights(names)
    scaler = fit_standardizer(x, weights)
    model = fit_logistic(
        scaler.transform(x), y, sample_weight=weights, l2=float(best["l2"])
    )
    mask, target_all = head_targets(target_support, head)
    adapted = adapt_scalar_intercept(
        model,
        scaler.transform(target_support.x[mask]),
        target_all[mask],
        prior_precision=float(best["prior_precision"]),
    )
    return FittedHead(
        model=adapted,
        standardizer=scaler,
        l2=float(best["l2"]),
        prior_precision=float(best["prior_precision"]),
        cv=cv,
    )
