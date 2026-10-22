from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from eargtc.router_bc.data_impl import (
    SupportDomain,
    TargetArtifacts,
    load_target_artifacts,
)
from eargtc.router_bc.hybrid import blend_preference_logits
from eargtc.router_bc.logistic import (
    adapt_scalar_intercept,
    domain_balanced_weights,
    fit_logistic,
)
from eargtc.router_bc.nonlinear import (
    _adapt_logit_bias,
    _factories,
    _feature_profiles,
    _fit,
    fit_preference_model,
)
from eargtc.router_bc.rank import percentile_features
from eargtc.router_bc.training import fit_head, fit_standardizer, head_targets
from eargtc.router_bc_improved.meta import (
    PreferenceConfig,
    PreferenceDomain,
    fit_preference_system,
    honest_all_row_scores,
)


@dataclass(frozen=True)
class FrozenBReplay:
    dataset: str
    threshold: float
    full_score: torch.Tensor
    support_score: torch.Tensor
    full_choose_g: torch.Tensor
    support_choose_g: torch.Tensor
    selected_metadata: dict


def route_from_score(
    score: torch.Tensor,
    *,
    threshold: float,
    gnn_pred: torch.Tensor,
    llm_pred: torch.Tensor,
) -> torch.Tensor:
    value = torch.as_tensor(score, dtype=torch.float64).view(-1)
    g = torch.as_tensor(gnn_pred).view(-1)
    l = torch.as_tensor(llm_pred).view(-1)
    if not (value.numel() == g.numel() == l.numel()):
        raise ValueError("Scheme B route inputs must have equal row counts")
    return (g == l) | (value >= float(threshold))


def _rank_support(domain: SupportDomain) -> SupportDomain:
    keep = [
        index
        for index, name in enumerate(domain.feature_names)
        if not name.startswith("ctx_")
    ]
    return SupportDomain(
        name=domain.name,
        rows=domain.rows,
        x=percentile_features(domain.x[:, keep]),
        feature_names=[domain.feature_names[index] for index in keep],
        gnn_correct=domain.gnn_correct,
        llm_correct=domain.llm_correct,
        disagreement=domain.disagreement,
    )


def _preference_domain(domain: SupportDomain) -> tuple[PreferenceDomain, torch.Tensor]:
    mask, target = head_targets(domain, "preference")
    return PreferenceDomain(
        domain.name, domain.x[mask], target[mask]
    ), mask.nonzero().view(-1)


def _config(value: dict) -> PreferenceConfig:
    return PreferenceConfig(
        l2=float(value["l2"]),
        rank_weight=float(value["rank_weight"]),
        adapter_rank=int(value["adapter_rank"]),
        prior_precision=float(value["prior_precision"]),
    )


def _honest_linear_support_scores(
    source: list[SupportDomain], target: SupportDomain, best: dict
) -> torch.Tensor:
    xs, ys, names = [], [], []
    for domain in source:
        mask, label = head_targets(domain, "preference")
        xs.append(domain.x[mask])
        ys.append(label[mask])
        names.extend([domain.name] * int(mask.sum().item()))
    x, y = torch.cat(xs), torch.cat(ys)
    weights = domain_balanced_weights(names)
    scaler = fit_standardizer(x, weights)
    shared = fit_logistic(
        scaler.transform(x), y, sample_weight=weights, l2=float(best["l2"])
    )
    mask, target_all = head_targets(target, "preference")
    row_ids = mask.nonzero().view(-1)
    decisive_x = scaler.transform(target.x[mask])
    decisive_y = target_all[mask]
    full = adapt_scalar_intercept(
        shared,
        decisive_x,
        decisive_y,
        prior_precision=float(best["prior_precision"]),
    )
    probability = full.probabilities(scaler.transform(target.x))
    for position, row_id in enumerate(row_ids.tolist()):
        keep = torch.ones(decisive_y.numel(), dtype=torch.bool)
        keep[position] = False
        adapted = adapt_scalar_intercept(
            shared,
            decisive_x[keep],
            decisive_y[keep],
            prior_precision=float(best["prior_precision"]),
        )
        probability[row_id] = adapted.probabilities(
            scaler.transform(target.x[row_id : row_id + 1])
        )[0]
    return probability


def _apply_numpy_bias(probability: np.ndarray, bias: float) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + np.exp(-np.clip(logit + float(bias), -30.0, 30.0)))


def _honest_nonlinear_support_scores(
    source: list[SupportDomain], target: SupportDomain, best: dict, seed: int
) -> torch.Tensor:
    profile, factory = _factories(int(seed))[best["model_name"]]
    columns = _feature_profiles(source[0].feature_names)[profile]
    xs, ys, names = [], [], []
    for domain in source:
        mask, label = head_targets(domain, "preference")
        xs.append(domain.x[mask][:, columns].numpy())
        ys.append(label[mask].numpy())
        names.extend([domain.name] * int(mask.sum().item()))
    estimator = _fit(
        factory(),
        np.concatenate(xs),
        np.concatenate(ys),
        domain_balanced_weights(names).numpy(),
    )
    mask, target_all = head_targets(target, "preference")
    row_ids = mask.nonzero().view(-1)
    target_y = target_all[mask].numpy()
    base_all = estimator.predict_proba(target.x[:, columns].numpy())[:, 1]
    full_bias = _adapt_logit_bias(base_all[row_ids.numpy()], target_y, precision=1.0)
    probability = _apply_numpy_bias(base_all, full_bias)
    for position, row_id in enumerate(row_ids.tolist()):
        keep = np.ones(len(target_y), dtype=bool)
        keep[position] = False
        bias = _adapt_logit_bias(
            base_all[row_ids.numpy()][keep], target_y[keep], precision=1.0
        )
        probability[row_id] = _apply_numpy_bias(base_all[row_id : row_id + 1], bias)[0]
    return torch.tensor(probability, dtype=torch.float32)


def _replay_improved(
    target_name: str,
    view_supports: dict[str, SupportDomain],
    target_x: torch.Tensor,
    metadata: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    domains = {
        name: _preference_domain(domain)[0] for name, domain in view_supports.items()
    }
    target_domain, row_ids = _preference_domain(view_supports[target_name])
    source = [domains[name] for name in view_supports if name != target_name]
    config = _config(metadata["config"])
    fitted = fit_preference_system(source, target_domain, config)
    full_score = fitted.probabilities(target_x)
    honest_support = honest_all_row_scores(
        source, target_domain, view_supports[target_name].x, row_ids, config
    )
    return full_score, honest_support


def _replay_legacy(
    target_name: str,
    view_supports: dict[str, SupportDomain],
    target_x: torch.Tensor,
    metadata: dict,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = [view_supports[name] for name in view_supports if name != target_name]
    target_support = view_supports[target_name]
    linear_best = metadata["source_cv"]["linear"]
    nonlinear_best = metadata["source_cv"]["nonlinear"]
    linear = fit_head(source, target_support, "preference", {"best": linear_best})
    nonlinear = fit_preference_model(
        source, target_support, {"best": nonlinear_best}, seed=int(seed)
    )
    fraction = float(metadata["config"]["lambda"])
    full_score = blend_preference_logits(
        linear.probabilities(target_x), nonlinear.probabilities(target_x), fraction
    )
    honest_score = blend_preference_logits(
        _honest_linear_support_scores(source, target_support, linear_best),
        _honest_nonlinear_support_scores(
            source, target_support, nonlinear_best, int(seed)
        ),
        fraction,
    )
    return full_score, honest_score


def audit_frozen_b_identity(
    replay: FrozenBReplay,
    *,
    node_ids: torch.Tensor,
    labels: torch.Tensor,
    gnn_pred: torch.Tensor,
    llm_pred: torch.Tensor,
    frozen_rows: list[dict],
    score_tolerance: float = 1e-6,
) -> dict:
    ids = torch.as_tensor(node_ids, dtype=torch.long).view(-1)
    y = torch.as_tensor(labels).view(-1)
    g = torch.as_tensor(gnn_pred).view(-1)
    l = torch.as_tensor(llm_pred).view(-1)
    if not (ids.numel() == y.numel() == g.numel() == l.numel()):
        raise ValueError("B identity inputs must have equal row counts")
    frozen = {int(row["node_id"]): row for row in frozen_rows}
    if len(frozen) != len(frozen_rows):
        raise ValueError("duplicate node IDs in frozen Scheme B decisions")
    node_set_match = set(ids.tolist()) == set(frozen)
    counts = {
        "expert_mismatches": 0,
        "label_mismatches": 0,
        "route_mismatches": 0,
        "selected_prediction_mismatches": 0,
        "correctness_mismatches": 0,
        "score_mismatches": 0,
    }
    max_score_diff = 0.0
    for node_id, label, g_pred, l_pred in zip(
        ids.tolist(), y.tolist(), g.tolist(), l.tolist()
    ):
        if node_id not in frozen:
            continue
        row = frozen[node_id]
        counts["expert_mismatches"] += int(
            int(row["gnn_pred"]) != int(g_pred) or int(row["llm_pred"]) != int(l_pred)
        )
        counts["label_mismatches"] += int(int(row["true_label"]) != int(label))
        choose_g = bool(replay.full_choose_g[node_id])
        counts["route_mismatches"] += int((row["route"] == "gnn") != choose_g)
        selected = int(g_pred if choose_g else l_pred)
        counts["selected_prediction_mismatches"] += int(
            int(row["selected_pred"]) != selected
        )
        counts["correctness_mismatches"] += int(
            bool(row["correct"]) != (selected == int(label))
        )
        difference = abs(
            float(row["preference_score"]) - float(replay.full_score[node_id])
        )
        max_score_diff = max(max_score_diff, difference)
        counts["score_mismatches"] += int(difference > float(score_tolerance))
    return {
        "dataset": replay.dataset,
        "checked": int(ids.numel()),
        "node_set_matches": node_set_match,
        **counts,
        "max_score_abs_diff": max_score_diff,
        "score_tolerance": float(score_tolerance),
        "all_checks_pass": node_set_match
        and all(value == 0 for value in counts.values()),
    }


def replay_selected_b(
    *,
    dataset: str,
    supports: dict[str, SupportDomain],
    input_run_root: str | Path,
    frozen_b_root: str | Path,
    dataset_root: str | Path,
    seed: int = 42,
) -> tuple[FrozenBReplay, TargetArtifacts, dict]:
    input_root, frozen_root = Path(input_run_root), Path(frozen_b_root)
    metrics = json.loads(
        (frozen_root / dataset / "metrics.json").read_text(encoding="utf-8")
    )
    metadata = metrics["scheme_b"]["selected_metadata"]
    threshold = float(
        metrics["scheme_b"]["selected_checkpoint"]["threshold"]["threshold"]
    )
    target = load_target_artifacts(
        dataset, input_root / dataset / "pipeline", dataset_root, supports[dataset]
    )
    views = {
        "raw": supports,
        "rank": {name: _rank_support(domain) for name, domain in supports.items()},
    }
    selected_supports = views[metadata["view"]]
    target_x = (
        target.x
        if metadata["view"] == "raw"
        else percentile_features(
            target.x[:, : len(selected_supports[dataset].feature_names)]
        )
    )
    if metadata["family"] == "legacy_residual":
        full_score, support_score = _replay_legacy(
            dataset, selected_supports, target_x, metadata, int(seed)
        )
    else:
        full_score, support_score = _replay_improved(
            dataset, selected_supports, target_x, metadata
        )
    full_choose = route_from_score(
        full_score,
        threshold=threshold,
        gnn_pred=target.gnn_pred,
        llm_pred=target.llm_pred,
    )
    support = supports[dataset]
    support_g = torch.tensor([int(row["gnn_pred"]) for row in support.rows])
    support_l = torch.tensor([int(row["llm_pred"]) for row in support.rows])
    support_choose = route_from_score(
        support_score,
        threshold=threshold,
        gnn_pred=support_g,
        llm_pred=support_l,
    )
    replay = FrozenBReplay(
        dataset=str(dataset),
        threshold=threshold,
        full_score=full_score,
        support_score=support_score,
        full_choose_g=full_choose,
        support_choose_g=support_choose,
        selected_metadata=metadata,
    )
    frozen_rows = [
        json.loads(line)
        for line in (frozen_root / dataset / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    audit = audit_frozen_b_identity(
        replay,
        node_ids=target.test_ids,
        labels=target.labels[target.test_ids],
        gnn_pred=target.gnn_pred[target.test_ids],
        llm_pred=target.llm_pred[target.test_ids],
        frozen_rows=frozen_rows,
    )
    return replay, target, audit
