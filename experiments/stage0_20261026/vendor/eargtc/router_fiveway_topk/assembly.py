"""Replay frozen/new reliability scores into comparable policy populations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from eargtc.router_bc.data_impl import SupportDomain
from eargtc.router_bc.logistic import LogisticModel
from eargtc.router_bc.rank import percentile_features
from eargtc.router_bc.training import Standardizer, fit_head, head_targets
from eargtc.router_bc_improved.meta import PreferenceDomain, fit_preference_system
from eargtc.router_bc_improved.runner_utils import best_trial_for_family
from eargtc.router_bc_improved.trust import (
    direct_trust_features,
    factorized_selected_correctness,
    fit_direct_trust,
)
from eargtc.router_bc_split_c.b_replay import _replay_legacy
from eargtc.router_bc_split_c.calibration import (
    PlattCalibrator,
    deterministic_calibration_threshold_split,
)
from eargtc.router_bc_split_c.pipeline import _heuristic_probability, _query_domain, _rank_target
from scripts.router_bc_improved.run_controlled_search import (
    build_source_trust_domains,
    config_from_trial,
    rank_support,
)


@dataclass(frozen=True)
class PolicyPopulation:
    dataset: str
    module: str
    validation_node_ids: torch.Tensor
    validation_score: torch.Tensor
    validation_correct: torch.Tensor
    test_node_ids: torch.Tensor
    test_score: torch.Tensor
    test_correct: torch.Tensor
    test_selected_pred: torch.Tensor
    provenance: dict

    def __post_init__(self) -> None:
        for prefix in ("validation", "test"):
            ids = torch.as_tensor(getattr(self, f"{prefix}_node_ids"), dtype=torch.long).view(-1)
            score = torch.as_tensor(getattr(self, f"{prefix}_score"), dtype=torch.float32).view(-1)
            correct = torch.as_tensor(getattr(self, f"{prefix}_correct"), dtype=torch.bool).view(-1)
            if not (ids.numel() == score.numel() == correct.numel()):
                raise ValueError(f"{prefix} population fields must align")
            if ids.numel() != torch.unique(ids).numel() or not bool(torch.isfinite(score).all()):
                raise ValueError(f"invalid {prefix} policy population")
            object.__setattr__(self, f"{prefix}_node_ids", ids)
            object.__setattr__(self, f"{prefix}_score", score)
            object.__setattr__(self, f"{prefix}_correct", correct)
        selected = torch.as_tensor(self.test_selected_pred, dtype=torch.long).view(-1)
        if selected.numel() != self.test_node_ids.numel():
            raise ValueError("test selected predictions must align")
        object.__setattr__(self, "test_selected_pred", selected)


def _serialized_split_model(record: dict):
    model = LogisticModel(
        coef=torch.tensor(record["coef"], dtype=torch.float64),
        intercept=float(record["intercept"]),
        adaptation_bias=float(record["adaptation_bias"]),
    )
    scaler = Standardizer(
        mean=torch.tensor(record["standardizer_mean"], dtype=torch.float64),
        scale=torch.tensor(record["standardizer_scale"], dtype=torch.float64),
    )
    return model, scaler


def split_c_population(
    *, dataset: str, module: str, target, replay, split_root: str | Path,
    validation_only: bool = False,
) -> PolicyPopulation:
    root = Path(split_root) / dataset
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))["modules"][module]
    model_record = json.loads((root / "models.json").read_text(encoding="utf-8"))[module]
    model, scaler = _serialized_split_model(model_record)
    view = str(metrics["selection"]["best"]["view"])
    views = {"raw": (target.x, target.feature_names), "rank": _rank_target(target)}
    validation = _query_domain(target, replay, target.val_ids, view=view, module=module, view_values=views)
    split = deterministic_calibration_threshold_split(validation.node_ids, dataset, module)
    calibrator = PlattCalibrator(
        scale=float(metrics["calibration"]["scale"]),
        bias=float(metrics["calibration"]["bias"]),
        regularization=float(metrics["calibration"]["regularization"]),
        count=int(metrics["calibration"]["count"]),
    )
    validation_score = calibrator.probabilities_from_logits(model.logits(scaler.transform(validation.evidence)))
    if validation_only:
        empty = torch.empty(0)
        return PolicyPopulation(
            dataset, module,
            validation.node_ids[split.threshold], validation_score[split.threshold], validation.target[split.threshold],
            empty.long(), empty.float(), empty.bool(), empty.long(),
            {"source": "split_c_v3", "view": view, "sealed_validation_only": True},
        )
    test = _query_domain(target, replay, target.test_ids, view=view, module=module, view_values=views)
    test_score = calibrator.probabilities_from_logits(model.logits(scaler.transform(test.evidence)))
    rows = [json.loads(line) for line in (root / f"decisions_{module}.jsonl").read_text().splitlines() if line]
    by_id = {int(row["node_id"]): row for row in rows}
    stored = torch.tensor([float(by_id[int(i)]["split_calibrated_reliability"]) for i in test.node_ids])
    max_diff = float((stored - test_score).abs().max())
    if max_diff > 1e-6:
        raise RuntimeError(f"Split-C score replay mismatch: {dataset}/{module}/{max_diff}")
    selected_pred = torch.where(test.choose_g, target.gnn_pred[test.node_ids], target.llm_pred[test.node_ids])
    return PolicyPopulation(
        dataset, module,
        validation.node_ids[split.threshold], validation_score[split.threshold], validation.target[split.threshold],
        test.node_ids, stored, test.target, selected_pred,
        {"source": "split_c_v3", "replay_max_abs_diff": max_diff, "view": view},
    )


def expert_confidence_population(
    *, dataset: str, module: str, target, replay, expert_root: str | Path,
    validation_only: bool = False,
) -> PolicyPopulation:
    validation = _query_domain(
        target, replay, target.val_ids, view="raw", module=module,
        view_values={"raw": (target.x, target.feature_names)},
    )
    split = deterministic_calibration_threshold_split(validation.node_ids, dataset, module)
    raw_val = _heuristic_probability(target, replay, validation)
    metrics = json.loads((Path(expert_root) / "metrics.json").read_text())["datasets"][dataset]["modules"][module]
    record = metrics["variants"]["calibrated_expert_confidence"]["calibrator"]
    calibrator = PlattCalibrator(float(record["scale"]), float(record["bias"]), float(record["regularization"]), int(record["count"]))
    score_val = calibrator.probabilities(raw_val)
    if validation_only:
        empty = torch.empty(0)
        return PolicyPopulation(
            dataset, module,
            validation.node_ids[split.threshold], score_val[split.threshold], validation.target[split.threshold],
            empty.long(), empty.float(), empty.bool(), empty.long(),
            {"source": "calibrated_expert_confidence", "scale": calibrator.scale, "bias": calibrator.bias, "sealed_validation_only": True},
        )
    test = _query_domain(
        target, replay, target.test_ids, view="raw", module=module,
        view_values={"raw": (target.x, target.feature_names)},
    )
    raw_test = _heuristic_probability(target, replay, test)
    score_test = calibrator.probabilities(raw_test)
    selected_pred = torch.where(test.choose_g, target.gnn_pred[test.node_ids], target.llm_pred[test.node_ids])
    return PolicyPopulation(
        dataset, module,
        validation.node_ids[split.threshold], score_val[split.threshold], validation.target[split.threshold],
        test.node_ids, score_test, test.target, selected_pred,
        {"source": "calibrated_expert_confidence", "scale": calibrator.scale, "bias": calibrator.bias},
    )


def new_conflict_population(path: str | Path, *, target=None, replay=None) -> PolicyPopulation:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validation = payload["validation"]
    mask = torch.as_tensor(payload["validation_threshold_mask"], dtype=torch.bool)
    if target is None or replay is None:
        empty = torch.empty(0)
        return PolicyPopulation(
            str(payload["dataset"]), "conflict",
            validation["node_ids"][mask], validation["score"][mask], validation["correct"][mask],
            empty.long(), empty.float(), empty.bool(), empty.long(),
            {"source": str(payload["provider_id"]), "selection": payload["selection"]["best"], "calibrator": payload["calibrator"], "sealed_validation_only": True},
        )
    model, scaler = _serialized_split_model(payload["model"])
    view = str(payload["selection"]["best"]["view"])
    views = {"raw": (target.x, target.feature_names), "rank": _rank_target(target)}
    domain = _query_domain(target, replay, target.test_ids, view=view, module="conflict", view_values=views)
    record = payload["calibrator"]
    calibrator = PlattCalibrator(float(record["scale"]), float(record["bias"]), float(record["regularization"]), int(record["count"]))
    score = calibrator.probabilities_from_logits(model.logits(scaler.transform(domain.evidence)))
    selected = torch.where(domain.choose_g, target.gnn_pred[domain.node_ids], target.llm_pred[domain.node_ids])
    return PolicyPopulation(
        str(payload["dataset"]), "conflict",
        validation["node_ids"][mask], validation["score"][mask], validation["correct"][mask],
        domain.node_ids, score, domain.target, selected,
        {"source": str(payload["provider_id"]), "selection": payload["selection"]["best"], "calibrator": payload["calibrator"]},
    )


def _head_probability(record: dict, x: torch.Tensor) -> torch.Tensor:
    model = LogisticModel(
        torch.tensor(record["coef"], dtype=torch.float64),
        float(record["intercept"]),
        float(record["adaptation_bias"]),
    )
    scaler = Standardizer(torch.tensor(record["mean"], dtype=torch.float64), torch.tensor(record["scale"], dtype=torch.float64))
    return model.probabilities(scaler.transform(x))


def _preference_domain(domain: SupportDomain) -> PreferenceDomain:
    mask, target = head_targets(domain, "preference")
    return PreferenceDomain(domain.name, domain.x[mask], target[mask])


def _in_sample_support_score(view_supports: dict[str, SupportDomain], target_name: str, metadata: dict) -> torch.Tensor:
    source = [view_supports[name] for name in view_supports if name != target_name]
    target_support = view_supports[target_name]
    if metadata["family"] == "legacy_residual":
        # Full replay score on support uses the same target adaptation but no per-row exclusion.
        linear = fit_head(source, target_support, "preference", {"best": metadata["source_cv"]["linear"]})
        from eargtc.router_bc.nonlinear import fit_preference_model
        from eargtc.router_bc.hybrid import blend_preference_logits
        nonlinear = fit_preference_model(source, target_support, {"best": metadata["source_cv"]["nonlinear"]}, seed=42)
        return blend_preference_logits(
            linear.probabilities(target_support.x), nonlinear.probabilities(target_support.x), float(metadata["config"]["lambda"])
        )
    domains = {name: _preference_domain(domain) for name, domain in view_supports.items()}
    config = config_from_trial(metadata["config"])
    fitted = fit_preference_system(
        [domains[name] for name in view_supports if name != target_name], domains[target_name], config
    )
    return fitted.probabilities(target_support.x)


def legacy_c_population(
    *, dataset: str, raw_supports: dict[str, SupportDomain], target, replay, legacy_root: str | Path,
    validation_only: bool = False,
) -> PolicyPopulation:
    metrics = json.loads((Path(legacy_root) / dataset / "metrics.json").read_text())
    selected_view = metrics["scheme_b"]["selected_metadata"]["view"]
    supports = raw_supports if selected_view == "raw" else {name: rank_support(domain) for name, domain in raw_supports.items()}
    target_x = target.x if selected_view == "raw" else percentile_features(target.x[:, :len(supports[dataset].feature_names)])
    conflict = target.gnn_pred != target.llm_pred
    choose_g = replay.full_choose_g
    factor = metrics["scheme_c"]["factorized_models"]
    availability = _head_probability(factor["availability"], target_x)
    agreement = _head_probability(factor["agreement"], target_x)
    q_factor = factorized_selected_correctness(conflict, choose_g, availability, replay.full_score, agreement)

    combined_trial = best_trial_for_family(
        metrics["source_meta"][selected_view]["trials"], "combined", objective="stable_auc", max_macro_nll=0.8
    )
    source_domains = build_source_trust_domains(supports, dataset, config_from_trial(combined_trial))
    support_score = _in_sample_support_score(supports, dataset, metrics["scheme_b"]["selected_metadata"])
    target_support = supports[dataset]
    support_choose = (~target_support.disagreement) | (support_score >= replay.threshold)
    support_correct = torch.where(support_choose, target_support.gnn_correct, target_support.llm_correct)
    from eargtc.router_bc_improved.trust import TrustDomain
    target_domain = TrustDomain(
        dataset,
        direct_trust_features(target_support.x, agreement=~target_support.disagreement, choose_g=support_choose, preference=support_score),
        support_correct.float(),
    )
    direct_record = metrics["scheme_c"]["direct_model"]
    direct = fit_direct_trust(
        source_domains, target_domain,
        l2=float(direct_record["source_cv"]["best"]["l2"]),
        prior_precision=float(direct_record["source_cv"]["best"]["prior_precision"]),
    )
    direct_x = direct_trust_features(target_x, agreement=~conflict, choose_g=choose_g, preference=replay.full_score)
    q_direct = direct.probabilities(direct_x)
    selected_kind = metrics["scheme_c"]["metrics"]["validation_selection"]["selected"]
    q = q_factor if selected_kind == "factorized" else q_direct

    val_ids = target.val_ids
    val_conflict = conflict[val_ids]
    masks = []
    for module, population in (("conflict", val_conflict), ("agreement", ~val_conflict)):
        ids = val_ids[population]
        split = deterministic_calibration_threshold_split(ids, dataset, module)
        masks.append(ids[split.threshold])
    policy_ids = torch.cat(masks)
    val_selected = torch.where(choose_g[policy_ids], target.gnn_pred[policy_ids], target.llm_pred[policy_ids])
    val_correct = val_selected == target.labels[policy_ids]
    if validation_only:
        empty = torch.empty(0)
        return PolicyPopulation(
            dataset, "global", policy_ids, q[policy_ids], val_correct,
            empty.long(), empty.float(), empty.bool(), empty.long(),
            {"source": "structural_oof_v6_legacy_c", "selected": selected_kind, "sealed_validation_only": True},
        )
    test_ids = target.test_ids
    test_selected = torch.where(choose_g[test_ids], target.gnn_pred[test_ids], target.llm_pred[test_ids])
    test_correct = test_selected == target.labels[test_ids]
    rows = [json.loads(line) for line in (Path(legacy_root) / dataset / "decisions.jsonl").read_text().splitlines() if line]
    by_id = {int(row["node_id"]): row for row in rows}
    stored = torch.tensor([float(by_id[int(i)]["selected_trust"]) for i in test_ids])
    max_diff = float((stored - q[test_ids]).abs().max())
    if max_diff > 2e-5:
        raise RuntimeError(f"legacy C score replay mismatch: {dataset}/{max_diff}")
    return PolicyPopulation(
        dataset, "global",
        policy_ids, q[policy_ids], val_correct,
        test_ids, stored, test_correct, test_selected,
        {"source": "structural_oof_v6_legacy_c", "selected": selected_kind, "replay_max_abs_diff": max_diff},
    )


__all__ = [
    "PolicyPopulation", "expert_confidence_population", "legacy_c_population",
    "new_conflict_population", "split_c_population",
]
