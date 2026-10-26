from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


@dataclass(frozen=True)
class DomainData:
    name: str
    x: torch.Tensor
    y: torch.Tensor

    def __post_init__(self) -> None:
        x = torch.as_tensor(self.x)
        y = torch.as_tensor(self.y).view(-1)
        if x.ndim != 2 or x.shape[0] != y.numel() or y.numel() == 0:
            raise ValueError(f"invalid domain data: {self.name}")
        values = set(y.long().tolist())
        if not values <= {0, 1}:
            raise ValueError("decision targets must be binary")


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    c: float = 1.0
    class_balanced: bool = False
    min_samples_leaf: int = 1
    hidden_dim: int = 32
    base_width: int | None = None
    correction_scale: float = 0.25

    def __post_init__(self) -> None:
        allowed = {"logistic", "rbf", "extra_trees", "mlp", "residual_logistic"}
        if self.family not in allowed:
            raise ValueError(f"unknown decision family: {self.family}")
        if self.c <= 0 or self.min_samples_leaf <= 0 or self.hidden_dim <= 0:
            raise ValueError("model hyperparameters must be positive")
        if self.family == "residual_logistic" and self.base_width is None:
            raise ValueError("residual logistic requires base_width")


@dataclass(frozen=True)
class FittedDecisionHead:
    spec: CandidateSpec
    estimator: Any
    input_scale: np.ndarray
    adapter_bias: float
    domain_weight_totals: dict[str, float]
    seed: int

    def predict_proba(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        values = np.asarray(torch.as_tensor(x, dtype=torch.float32).cpu())
        if values.ndim != 2 or values.shape[1] != self.input_scale.shape[0]:
            raise ValueError("decision input shape differs from fitted head")
        raw = self.estimator.predict_proba(values * self.input_scale)
        classes = list(getattr(self.estimator, "classes_", [0, 1]))
        if 1 in classes:
            probability = raw[:, classes.index(1)]
        else:
            probability = np.full(values.shape[0], float(classes[0] == 1))
        probability = np.clip(probability, 1e-7, 1 - 1e-7)
        logits = np.log(probability) - np.log1p(-probability) + self.adapter_bias
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float64)


def _stack_domains(
    domains: list[DomainData], class_balanced: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    if not domains:
        raise ValueError("at least one source domain is required")
    width = int(torch.as_tensor(domains[0].x).shape[1])
    xs, ys, weights, names = [], [], [], []
    for domain in domains:
        x = torch.as_tensor(domain.x, dtype=torch.float32).cpu().numpy()
        y = torch.as_tensor(domain.y, dtype=torch.long).view(-1).cpu().numpy()
        if x.shape[1] != width:
            raise ValueError("source domains have different feature widths")
        weight = np.full(y.shape[0], 1.0 / y.shape[0], dtype=np.float64)
        if class_balanced:
            counts = np.bincount(y, minlength=2).astype(np.float64)
            factors = np.where(counts > 0, 1.0 / counts, 0.0)
            weight *= factors[y]
            weight /= weight.sum()
        xs.append(x)
        ys.append(y)
        weights.append(weight)
        names.extend([domain.name] * len(y))
    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    weight_all = np.concatenate(weights)
    totals = {
        domain.name: float(weight_all[np.asarray(names) == domain.name].sum())
        for domain in domains
    }
    return x_all, y_all, weight_all, totals


def fit_candidate(
    spec: CandidateSpec,
    domains: list[DomainData],
    seed: int,
) -> FittedDecisionHead:
    x, y, sample_weight, totals = _stack_domains(domains, spec.class_balanced)
    input_scale = np.ones(x.shape[1], dtype=np.float64)
    family = spec.family
    if family == "residual_logistic":
        base = int(spec.base_width)
        if not 0 < base < x.shape[1]:
            raise ValueError("residual base_width must split the input columns")
        input_scale[base:] = float(spec.correction_scale)
        family = "logistic"
    scaled = x * input_scale

    if np.unique(y).size < 2:
        estimator = DummyClassifier(strategy="constant", constant=int(y[0]))
    elif family == "logistic":
        estimator = LogisticRegression(
            C=float(spec.c),
            max_iter=5000,
            solver="lbfgs",
            random_state=int(seed),
        )
    elif family == "rbf":
        estimator = SVC(
            C=float(spec.c),
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=int(seed),
        )
    elif family == "extra_trees":
        estimator = ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=int(spec.min_samples_leaf),
            max_features="sqrt",
            random_state=int(seed),
            n_jobs=1,
        )
    elif family == "mlp":
        estimator = MLPClassifier(
            hidden_layer_sizes=(int(spec.hidden_dim),),
            activation="relu",
            alpha=1.0 / float(spec.c),
            batch_size=min(32, len(y)),
            learning_rate_init=1e-3,
            max_iter=1500,
            random_state=int(seed),
            early_stopping=False,
        )
    else:
        raise AssertionError(family)
    estimator.fit(scaled, y, sample_weight=sample_weight)
    return FittedDecisionHead(
        spec=spec,
        estimator=estimator,
        input_scale=input_scale,
        adapter_bias=0.0,
        domain_weight_totals=totals,
        seed=int(seed),
    )


def adapt_on_target_support(
    head: FittedDecisionHead,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    max_abs_bias: float = 2.0,
    ridge: float = 1.0,
) -> FittedDecisionHead:
    targets = np.asarray(torch.as_tensor(y, dtype=torch.float64).view(-1).cpu())
    if targets.size == 0:
        return head
    scores = np.clip(head.predict_proba(x), 1e-7, 1 - 1e-7)
    logits = np.log(scores) - np.log1p(-scores)
    bias = 0.0
    for _ in range(50):
        probability = 1.0 / (1.0 + np.exp(-(logits + bias)))
        gradient = float(np.sum(probability - targets) + float(ridge) * bias)
        curvature = float(
            np.sum(probability * (1.0 - probability)) + float(ridge)
        )
        update = gradient / max(curvature, 1e-12)
        bias = float(np.clip(bias - update, -max_abs_bias, max_abs_bias))
        if abs(update) < 1e-8:
            break
    return replace(head, adapter_bias=float(head.adapter_bias + bias))


__all__ = [
    "CandidateSpec",
    "DomainData",
    "FittedDecisionHead",
    "adapt_on_target_support",
    "fit_candidate",
]
