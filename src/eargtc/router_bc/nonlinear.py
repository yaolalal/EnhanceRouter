from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data_impl import SupportDomain
from .logistic import domain_balanced_weights
from .training import head_targets


def _feature_profiles(names: list[str]) -> dict[str, list[int]]:
    evidence = [i for i, name in enumerate(names) if not name.startswith("ctx_")]
    graph = {"log_degree", "neighbor_gnn_agreement", "neighbor_gnn_entropy"}
    core_names = {
        "g_entropy", "l_entropy", "g_margin", "l_margin", "g_top1", "l_top1",
        "p_g_at_l", "p_l_at_g", "js_divergence", "bhattacharyya_overlap",
        "log_degree", "neighbor_gnn_agreement", "neighbor_gnn_entropy",
    }
    return {
        "full": evidence,
        "no_graph": [i for i in evidence if names[i] not in graph],
        "core": [i for i in evidence if names[i] in core_names],
    }


def _factories(seed: int):
    result = {
        "logistic_full": ("full", lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=0.03, max_iter=2000, random_state=seed)
        )),
        "rbf_no_graph": ("no_graph", lambda: make_pipeline(
            StandardScaler(), SVC(C=1.0, gamma=0.1, probability=True, random_state=seed)
        )),
        "rf_core_d1": ("core", lambda: RandomForestClassifier(
            n_estimators=300, max_depth=1, min_samples_leaf=5, max_features="sqrt",
            random_state=seed, n_jobs=1,
        )),
    }
    for profile in ("full", "no_graph", "core"):
        for depth in (1, 2, 3):
            result[f"extra_{profile}_d{depth}"] = (profile, lambda depth=depth: ExtraTreesClassifier(
                n_estimators=300, max_depth=depth, min_samples_leaf=5,
                max_features="sqrt", random_state=seed, n_jobs=1,
            ))
    return result


def _fit(estimator, x, y, weights):
    kwargs = {"sample_weight": weights}
    if hasattr(estimator, "steps"):
        kwargs = {f"{estimator.steps[-1][0]}__sample_weight": weights}
    estimator.fit(x, y, **kwargs)
    return estimator


def select_preference_model(domains: list[SupportDomain], seed: int = 42) -> dict:
    profiles = _feature_profiles(domains[0].feature_names)
    trials = []
    for model_name, (profile, factory) in _factories(seed).items():
        folds = []
        for held in domains:
            train_domains = [domain for domain in domains if domain.name != held.name]
            xs, ys, domain_names = [], [], []
            for domain in train_domains:
                mask, target = head_targets(domain, "preference")
                xs.append(domain.x[mask][:, profiles[profile]].numpy())
                ys.append(target[mask].numpy())
                domain_names.extend([domain.name] * int(mask.sum()))
            weights = domain_balanced_weights(domain_names).numpy()
            estimator = _fit(factory(), np.concatenate(xs), np.concatenate(ys), weights)
            mask, target = head_targets(held, "preference")
            y = target[mask].numpy()
            probability = estimator.predict_proba(held.x[mask][:, profiles[profile]].numpy())[:, 1]
            folds.append({
                "heldout_domain": held.name,
                "count": int(len(y)),
                "auroc": float(roc_auc_score(y, probability)) if len(set(y)) == 2 else None,
                "nll": float(log_loss(y, probability, labels=[0, 1])),
                "accuracy": float(np.mean((probability >= 0.5) == y)),
            })
        valid_auc = [fold["auroc"] for fold in folds if fold["auroc"] is not None]
        trials.append({
            "model_name": model_name,
            "profile": profile,
            "macro_auroc": sum(valid_auc) / len(valid_auc),
            "macro_nll": sum(fold["nll"] for fold in folds) / len(folds),
            "macro_accuracy": sum(fold["accuracy"] for fold in folds) / len(folds),
            "folds": folds,
        })
    best = min(trials, key=lambda row: (-row["macro_auroc"], row["macro_nll"], row["model_name"]))
    return {
        "head": "preference",
        "selection": "nested_source_lodo_macro_auroc",
        "best": best,
        "trials": trials,
    }


def _adapt_logit_bias(probability: np.ndarray, target: np.ndarray, precision: float = 1.0) -> float:
    if len(target) == 0:
        return 0.0
    base = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1))
    delta = 0.0
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-np.clip(base + delta, -30, 30)))
        gradient = np.mean(p - target) + precision * delta
        hessian = np.mean(p * (1.0 - p)) + precision + 1e-8
        step = gradient / hessian
        delta -= step
        if abs(step) < 1e-10:
            break
    return float(delta)


@dataclass
class NonlinearPreferenceHead:
    estimator: object
    columns: list[int]
    selected_feature_names: list[str]
    adaptation_bias: float
    cv: dict

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        probability = self.estimator.predict_proba(x[:, self.columns].numpy())[:, 1]
        logit = np.log(np.clip(probability, 1e-6, 1 - 1e-6) / np.clip(1 - probability, 1e-6, 1))
        adapted = 1.0 / (1.0 + np.exp(-np.clip(logit + self.adaptation_bias, -30, 30)))
        return torch.tensor(adapted, dtype=torch.float32)

    def to_dict(self) -> dict:
        return {
            "model_name": self.cv["best"]["model_name"],
            "profile": self.cv["best"]["profile"],
            "selected_feature_names": self.selected_feature_names,
            "adaptation_bias": self.adaptation_bias,
            "target_trainable_parameters": 1,
            "cv": self.cv,
        }


def fit_preference_model(
    source_domains: list[SupportDomain], target_support: SupportDomain, cv: dict, seed: int = 42
) -> NonlinearPreferenceHead:
    best = cv["best"]
    profile, factory = _factories(seed)[best["model_name"]]
    columns = _feature_profiles(source_domains[0].feature_names)[profile]
    xs, ys, domain_names = [], [], []
    for domain in source_domains:
        mask, target = head_targets(domain, "preference")
        xs.append(domain.x[mask][:, columns].numpy())
        ys.append(target[mask].numpy())
        domain_names.extend([domain.name] * int(mask.sum()))
    weights = domain_balanced_weights(domain_names).numpy()
    estimator = _fit(factory(), np.concatenate(xs), np.concatenate(ys), weights)
    target_mask, target_all = head_targets(target_support, "preference")
    target_x = target_support.x[target_mask][:, columns].numpy()
    target_y = target_all[target_mask].numpy()
    base_probability = estimator.predict_proba(target_x)[:, 1]
    bias = _adapt_logit_bias(base_probability, target_y, precision=1.0)
    return NonlinearPreferenceHead(
        estimator=estimator,
        columns=columns,
        selected_feature_names=[source_domains[0].feature_names[i] for i in columns],
        adaptation_bias=bias,
        cv=cv,
    )
