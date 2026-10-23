from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.gnn import MLPHead
from .train import train_classifier
from .utils import ensure_dir, write_jsonl


@dataclass
class RouterArtifacts:
    labels: torch.Tensor
    idx: torch.Tensor
    label_sources: dict[int, str]
    sample_weights: torch.Tensor
    stats: dict
    feature_names: list[str]


class TwoHeadRouter(nn.Module):
    """Predict per-node reliability for GNN and LLM experts.

    The router does not predict classes. It predicts:
      r_g = P(GNN argmax is correct | evidence)
      r_l = P(LLM argmax is correct | evidence)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        mid_dim = max(int(hidden_dim) // 2, 8)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), mid_dim),
            nn.ReLU(),
        )
        self.gnn_head = nn.Linear(mid_dim, 1)
        self.llm_head = nn.Linear(mid_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x.float())
        return self.gnn_head(h).squeeze(-1), self.llm_head(h).squeeze(-1)


SCALAR_FEATURE_NAMES = [
    "gnn_conf",
    "llm_conf",
    "gnn_entropy",
    "llm_entropy",
    "gnn_margin",
    "llm_margin",
    "agreement",
    "confidence_gap",
    "signed_confidence_gap",
    "log_degree",
    "neighbor_gnn_agreement",
    "neighbor_gnn_entropy",
    "text_length",
]

FEATURE_NAMES = SCALAR_FEATURE_NAMES


def router_feature_names(num_classes: int) -> list[str]:
    num_classes = int(num_classes)
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    evidence = [f"gnn_logit_{i}" for i in range(num_classes)]
    evidence += [f"llm_logit_{i}" for i in range(num_classes)]
    return evidence + list(SCALAR_FEATURE_NAMES)


def _calibrate_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    temp = max(float(temperature), 1e-6)
    return logits.float() / temp


def fit_temperature_grid(
    logits: torch.Tensor,
    labels: torch.Tensor,
    idx: Iterable[int] | torch.Tensor,
    candidates: Iterable[float] | None = None,
) -> dict:
    """Fit a scalar temperature by minimizing NLL on a calibration split."""
    if candidates is None:
        candidates = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0)
    if isinstance(idx, torch.Tensor):
        ids = idx.detach().cpu().long().tolist()
    else:
        ids = [int(i) for i in idx]
    ids = [i for i in ids if 0 <= int(i) < int(logits.size(0))]
    candidate_list = [float(t) for t in candidates]
    if not ids:
        return {"temperature": 1.0, "nll": None, "count": 0, "candidates": candidate_list}
    device = logits.device
    idx_t = torch.tensor(ids, dtype=torch.long, device=device)
    y = labels.to(device).long()[idx_t]
    best_temp = 1.0
    best_nll = float("inf")
    tried: list[dict] = []
    with torch.no_grad():
        for temp in candidate_list:
            if temp <= 0:
                continue
            nll = F.cross_entropy(_calibrate_logits(logits[idx_t], temp), y).item()
            tried.append({"temperature": float(temp), "nll": float(nll)})
            if nll < best_nll:
                best_nll = float(nll)
                best_temp = float(temp)
    return {"temperature": float(best_temp), "nll": float(best_nll), "count": int(len(ids)), "candidates": tried}


def _temperature_metadata(
    oof_gnn: dict,
    router_train_llm: dict,
    final_gnn: dict,
    final_llm: dict,
    *,
    final_calibration_split: str | None = None,
) -> dict:
    metadata = {
        "oof_gnn": oof_gnn,
        "router_train_llm": router_train_llm,
        "final_gnn": final_gnn,
        "final_llm": final_llm,
    }
    if final_calibration_split is not None:
        metadata["final_calibration_split"] = str(final_calibration_split)
    return metadata


def _identity_temperature_metadata(split_name: str) -> dict:
    return {
        "temperature": 1.0,
        "nll": None,
        "count": 0,
        "candidates": [],
        "source": "identity_no_logit_calibration",
        "split": str(split_name),
    }


def _prepare_router_logits(
    gnn_logits: torch.Tensor,
    llm_logits: torch.Tensor,
    labels: torch.Tensor,
    idx: Iterable[int] | torch.Tensor,
    *,
    calibration_enabled: bool,
    split_name: str,
) -> dict:
    if not bool(calibration_enabled):
        return {
            "gnn_logits": gnn_logits,
            "llm_logits": llm_logits,
            "gnn_temperature": _identity_temperature_metadata(f"{split_name}_gnn"),
            "llm_temperature": _identity_temperature_metadata(f"{split_name}_llm"),
            "logit_calibration": "disabled",
        }
    gnn_temperature = fit_temperature_grid(gnn_logits.detach(), labels, idx)
    llm_temperature = fit_temperature_grid(llm_logits.detach(), labels, idx)
    return {
        "gnn_logits": _calibrate_logits(gnn_logits, gnn_temperature["temperature"]),
        "llm_logits": _calibrate_logits(llm_logits, llm_temperature["temperature"]),
        "gnn_temperature": gnn_temperature,
        "llm_temperature": llm_temperature,
        "logit_calibration": "temperature_grid",
    }


def _normalized_entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    num_classes = probs.size(-1)
    if num_classes <= 1:
        return torch.zeros(probs.size(0), dtype=probs.dtype, device=probs.device)
    entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=-1)
    return entropy / math.log(num_classes)


def _margin(probs: torch.Tensor) -> torch.Tensor:
    if probs.size(-1) <= 1:
        return torch.ones(probs.size(0), dtype=probs.dtype, device=probs.device)
    top2 = probs.topk(k=2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


def _read_generation_metadata(path: str | Path, num_nodes: int) -> dict[int, dict]:
    path = Path(path)
    meta: dict[int, dict] = {}
    if not path.exists() or path.suffix.lower() != ".jsonl":
        return meta
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "node_id" not in row:
                continue
            node_id = int(row["node_id"])
            if node_id < 0 or node_id >= num_nodes:
                continue
            conf = row.get("generative_confidence", row.get("llm_confidence"))
            label_id = row.get("generative_label_id", row.get("label_id"))
            valid = row.get("valid")
            if valid is None:
                valid = label_id is not None
            mean_logprob = row.get("mean_logprob")
            if mean_logprob is None and conf is not None and float(conf) > 0:
                mean_logprob = math.log(max(float(conf), 1e-12))
            meta[node_id] = {
                "valid": bool(valid),
                "confidence": 0.0 if conf is None else float(conf),
                "mean_logprob": -50.0 if mean_logprob is None else float(mean_logprob),
            }
    return meta


def _token_length(text: str) -> int:
    # Lightweight, dependency-free approximation. It is stable across runs and
    # avoids loading the LLM tokenizer during router feature extraction.
    return len(re.findall(r"\S+", text or ""))


def _degree_and_neighbor_features(edge_index: torch.Tensor, gnn_pred: torch.Tensor, num_nodes: int, num_classes: int):
    device = gnn_pred.device
    src = edge_index[0].long().to(device)
    dst = edge_index[1].long().to(device)
    degree = torch.bincount(src, minlength=num_nodes).float()

    same = (gnn_pred[src] == gnn_pred[dst]).float()
    same_count = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    same_count.index_add_(0, src, same)
    neighbor_agreement = torch.where(degree > 0, same_count / degree.clamp_min(1.0), torch.zeros_like(degree))

    class_counts = torch.zeros(num_nodes, num_classes, dtype=torch.float32, device=device)
    class_counts.index_put_((src, gnn_pred[dst].long()), torch.ones_like(src, dtype=torch.float32), accumulate=True)
    q = class_counts / degree.clamp_min(1.0).unsqueeze(-1)
    if num_classes <= 1:
        neighbor_entropy = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    else:
        neighbor_entropy = -(q * q.clamp_min(1e-12).log()).sum(dim=-1) / math.log(num_classes)
    neighbor_entropy = torch.where(degree > 0, neighbor_entropy, torch.ones_like(neighbor_entropy))
    return degree, neighbor_agreement, neighbor_entropy


def build_router_feature_rows(
    *,
    node_ids: Iterable[int],
    data,
    texts: list[str],
    gnn_logits: torch.Tensor,
    llm_logits: torch.Tensor,
    llm_metadata: dict[int, dict] | None = None,
    max_text_tokens: int | None = None,
) -> list[dict]:
    del llm_metadata, max_text_tokens
    node_ids = [int(i) for i in node_ids]
    num_nodes = int(data.num_nodes)
    num_classes = int(gnn_logits.size(-1))
    feature_names = router_feature_names(num_classes)
    gnn_logits = gnn_logits.float()
    llm_logits = llm_logits.float()
    gnn_probs = F.softmax(gnn_logits, dim=-1)
    llm_probs = F.softmax(llm_logits, dim=-1)
    gnn_pred = gnn_probs.argmax(dim=-1)
    llm_pred = llm_probs.argmax(dim=-1)

    gnn_entropy = _normalized_entropy(gnn_probs).clamp(0.0, 1.0)
    llm_entropy = _normalized_entropy(llm_probs).clamp(0.0, 1.0)
    gnn_conf = (1.0 - gnn_entropy).clamp(0.0, 1.0)
    llm_conf = (1.0 - llm_entropy).clamp(0.0, 1.0)
    gnn_margin = _margin(gnn_probs).clamp(0.0, 1.0)
    llm_margin = _margin(llm_probs).clamp(0.0, 1.0)
    degree, neighbor_agreement, neighbor_entropy = _degree_and_neighbor_features(
        data.edge_index, gnn_pred, num_nodes, num_classes
    )

    rows = []
    for node_id in node_ids:
        text_length = _token_length(texts[node_id] if node_id < len(texts) else "")
        g_conf = float(gnn_conf[node_id].item())
        l_conf = float(llm_conf[node_id].item())
        features = {f"gnn_logit_{i}": float(gnn_logits[node_id, i].item()) for i in range(num_classes)}
        features.update({f"llm_logit_{i}": float(llm_logits[node_id, i].item()) for i in range(num_classes)})
        features.update(
            {
                "gnn_conf": g_conf,
                "llm_conf": l_conf,
                "gnn_entropy": float(gnn_entropy[node_id].item()),
                "llm_entropy": float(llm_entropy[node_id].item()),
                "gnn_margin": float(gnn_margin[node_id].item()),
                "llm_margin": float(llm_margin[node_id].item()),
                "agreement": int(int(gnn_pred[node_id].item()) == int(llm_pred[node_id].item())),
                "confidence_gap": abs(g_conf - l_conf),
                "signed_confidence_gap": g_conf - l_conf,
                "log_degree": math.log1p(float(degree[node_id].item())),
                "neighbor_gnn_agreement": float(neighbor_agreement[node_id].item()),
                "neighbor_gnn_entropy": float(neighbor_entropy[node_id].item()),
                "text_length": int(text_length),
            }
        )
        missing = [name for name in feature_names if name not in features]
        if missing:
            raise ValueError(f"Router feature construction missed fields: {missing[:5]}")
        rows.append(
            {
                "node_id": node_id,
                "features": features,
                "gnn_pred": int(gnn_pred[node_id].item()),
                "llm_pred": int(llm_pred[node_id].item()),
                "gnn_prob": gnn_probs[node_id].detach().cpu().tolist(),
                "llm_prob": llm_probs[node_id].detach().cpu().tolist(),
            }
        )
    return rows

def _assert_feature_ranges(rows: list[dict]) -> None:
    bounded = [
        "gnn_conf", "llm_conf", "gnn_entropy", "llm_entropy", "gnn_margin", "llm_margin",
        "agreement", "confidence_gap", "neighbor_gnn_agreement", "neighbor_gnn_entropy",
    ]
    for row in rows:
        features = row["features"]
        for name in bounded:
            value = float(features[name])
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"Router feature {name} out of [0, 1] for node {row['node_id']}: {value}")
        if abs(float(features["gnn_conf"]) + float(features["gnn_entropy"]) - 1.0) > 1e-4:
            raise ValueError(f"gnn_conf + gnn_entropy sanity check failed for node {row['node_id']}")


def _feature_matrix(rows: list[dict], feature_names: list[str]) -> torch.Tensor:
    return torch.tensor([[float(row["features"][name]) for name in feature_names] for row in rows], dtype=torch.float32)


def _stratified_folds(train_idx: torch.Tensor, labels: torch.Tensor, num_folds: int, seed: int) -> list[list[int]]:
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    by_class: dict[int, list[int]] = defaultdict(list)
    for node_id in train_idx.detach().cpu().long().tolist():
        by_class[int(labels[int(node_id)].item())].append(int(node_id))
    min_count = min((len(v) for v in by_class.values() if v), default=1)
    k = max(2, min(int(num_folds), max(min_count, 2)))
    folds: list[list[int]] = [[] for _ in range(k)]
    for ids in by_class.values():
        order = torch.randperm(len(ids), generator=rng).tolist()
        shuffled = [ids[i] for i in order]
        for i, node_id in enumerate(shuffled):
            folds[i % k].append(node_id)
    return [fold for fold in folds if fold]


def _gnn_class_weights(labels: torch.Tensor, train_idx: torch.Tensor, num_classes: int, mode: str) -> torch.Tensor | None:
    mode = str(mode or "none")
    if mode == "none":
        return None
    if mode != "balanced":
        raise ValueError(f"Unknown GNN head class_weight mode: {mode}")
    counts = torch.bincount(labels[train_idx].view(-1).long().detach().cpu(), minlength=int(num_classes)).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    weights[counts == 0] = 0.0
    positive = weights > 0
    if positive.any():
        weights = weights / weights[positive].mean().clamp_min(1e-12)
    return weights.to(labels.device)


def _router_oof_head_config(args, gnn_artifact_info: dict | None = None) -> dict:
    trial = (gnn_artifact_info or {}).get("mlp_trial_config") or {}
    if trial:
        return {
            "source": "selected_gnn_mlp_trial",
            "name": str(trial.get("name", (gnn_artifact_info or {}).get("mlp_trial", "selected"))),
            "head_hidden_dim": int(trial.get("head_hidden_dim", args.head_hidden_dim)),
            "dropout": float(trial.get("dropout", args.dropout)),
            "lr": float(trial.get("lr", args.head_lr)),
            "weight_decay": float(trial.get("weight_decay", args.weight_decay)),
            "epochs": int(trial.get("epochs", args.router_oof_head_epochs)),
            "class_weight": str(trial.get("class_weight", "none")),
            "patience": None if trial.get("patience") is None else int(trial.get("patience")),
            "select_by": str(trial.get("select_by", "val_loss")),
        }
    return {
        "source": "router_args_fallback",
        "name": "router_args_fallback",
        "head_hidden_dim": int(args.head_hidden_dim),
        "dropout": float(args.dropout),
        "lr": float(args.head_lr),
        "weight_decay": float(args.weight_decay),
        "epochs": int(args.router_oof_head_epochs),
        "class_weight": "none",
        "patience": None,
        "select_by": "val_loss",
    }


def _train_two_head_router(
    rows: list[dict],
    feature_names: list[str],
    *,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
):
    if not rows:
        raise ValueError("Cannot train router without OOF rows.")
    torch.manual_seed(int(seed))
    x = _feature_matrix(rows, feature_names).to(device)
    y_g = torch.tensor([float(row["gnn_correct"]) for row in rows], dtype=torch.float32, device=device)
    y_l = torch.tensor([float(row["llm_correct"]) for row in rows], dtype=torch.float32, device=device)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    xz = (x - mean) / std

    model = TwoHeadRouter(input_dim=x.size(1), hidden_dim=hidden_dim, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    pos_g = y_g.sum().clamp_min(1.0)
    neg_g = (y_g.numel() - y_g.sum()).clamp_min(1.0)
    pos_l = y_l.sum().clamp_min(1.0)
    neg_l = (y_l.numel() - y_l.sum()).clamp_min(1.0)
    loss_g = nn.BCEWithLogitsLoss(pos_weight=(neg_g / pos_g).detach())
    loss_l = nn.BCEWithLogitsLoss(pos_weight=(neg_l / pos_l).detach())
    for _ in range(max(int(epochs), 1)):
        model.train()
        opt.zero_grad(set_to_none=True)
        rg_logit, rl_logit = model(xz)
        loss = loss_g(rg_logit, y_g) + loss_l(rl_logit, y_l)
        loss.backward()
        opt.step()
    model.eval()
    return model, mean.detach().cpu(), std.detach().cpu()


def _route_node(
    row: dict,
    r_g: float,
    r_l: float,
    *,
    tau_hard: float,
    tau_soft: float,
    margin: float,
    conflict_to_abstain: bool,
) -> dict:
    gnn_pred = int(row["gnn_pred"])
    llm_pred = int(row["llm_pred"])
    same = gnn_pred == llm_pred
    tau_hard = float(tau_hard)
    tau_soft = float(tau_soft)
    margin = float(margin)

    if bool(conflict_to_abstain) and (not same) and r_g >= tau_hard and r_l >= tau_hard:
        return {"pool": "A", "action": "high_conflict", "pseudo_label": None, "weight": 0.0}
    if same and min(r_g, r_l) >= tau_soft:
        return {"pool": "H", "action": "consensus", "pseudo_label": gnn_pred, "weight": float((r_g + r_l) / 2.0)}
    if r_g >= tau_hard and (r_g - r_l) >= margin:
        return {"pool": "H", "action": "trust_gnn", "pseudo_label": gnn_pred, "weight": float(r_g)}
    if r_l >= tau_hard and (r_l - r_g) >= margin:
        return {"pool": "H", "action": "trust_llm", "pseudo_label": llm_pred, "weight": float(r_l)}
    if max(r_g, r_l) >= tau_soft:
        g = torch.tensor(row["gnn_prob"], dtype=torch.float32)
        l = torch.tensor(row["llm_prob"], dtype=torch.float32)
        q = r_g * g + r_l * l
        q = q / q.sum().clamp_min(1e-12)
        return {"pool": "S", "action": "soft_mix", "pseudo_label": None, "weight": float(max(r_g, r_l)), "soft_target": q.tolist()}
    return {"pool": "A", "action": "trust_neither", "pseudo_label": None, "weight": 0.0}


def _score_rows_with_router(
    model: TwoHeadRouter,
    mean: torch.Tensor,
    std: torch.Tensor,
    rows: list[dict],
    feature_names: list[str],
    device: torch.device,
) -> tuple[list[float], list[float]]:
    if not rows:
        return [], []
    x = _feature_matrix(rows, feature_names).to(device)
    xz = (x - mean.to(device)) / std.to(device).clamp_min(1e-6)
    model.eval()
    with torch.no_grad():
        r_g_logit, r_l_logit = model(xz)
        r_g = torch.sigmoid(r_g_logit).detach().cpu().tolist()
        r_l = torch.sigmoid(r_l_logit).detach().cpu().tolist()
    return [float(v) for v in r_g], [float(v) for v in r_l]


def _make_decisions(
    rows: list[dict],
    r_g: list[float],
    r_l: list[float],
    *,
    tau_hard: float,
    tau_soft: float,
    margin: float,
    conflict_to_abstain: bool,
) -> list[dict]:
    decisions: list[dict] = []
    for row, rg, rl in zip(rows, r_g, r_l):
        decision = _route_node(
            row,
            float(rg),
            float(rl),
            tau_hard=tau_hard,
            tau_soft=tau_soft,
            margin=margin,
            conflict_to_abstain=conflict_to_abstain,
        )
        record = {
            "node_id": int(row["node_id"]),
            "pool": decision["pool"],
            "action": decision["action"],
            "pseudo_label": decision.get("pseudo_label"),
            "weight": float(decision.get("weight", 0.0)),
            "r_g": float(rg),
            "r_l": float(rl),
            "gnn_pred": int(row["gnn_pred"]),
            "llm_pred": int(row["llm_pred"]),
        }
        if "soft_target" in decision:
            record["soft_target"] = decision["soft_target"]
        decisions.append(record)
    return decisions


def _wilson_lower_bound(correct: int, total: int, z: float = 1.96) -> float:
    total = int(total)
    correct = int(correct)
    if total <= 0:
        return 0.0
    phat = correct / total
    z2 = float(z) ** 2
    denom = 1.0 + z2 / total
    centre = phat + z2 / (2.0 * total)
    radius = float(z) * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * total)) / total)
    return max(0.0, float((centre - radius) / denom))


def _parse_float_grid(value, default: Iterable[float]) -> list[float]:
    if value is None:
        return [float(v) for v in default]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    text = str(value).strip()
    if not text:
        return [float(v) for v in default]
    return [float(part.strip()) for part in text.split(",") if part.strip()]

def _select_hard_pool(items: list[dict], *, target_precision: float, total_cap: int, per_class_cap: int) -> tuple[list[dict], dict]:
    items = sorted(items, key=lambda row: (-float(row["weight"]), int(row["node_id"])))
    selected: list[dict] = []
    class_counts: Counter[int] = Counter()
    skipped_class_cap = 0
    for item in items:
        label = int(item["pseudo_label"])
        if per_class_cap > 0 and class_counts[label] >= int(per_class_cap):
            skipped_class_cap += 1
            continue
        selected.append(item)
        class_counts[label] += 1
        if total_cap > 0 and len(selected) >= int(total_cap):
            break
    return selected, {
        "target_precision_reference": float(target_precision),
        "expected_precision": float(sum(float(x["weight"]) for x in selected) / max(len(selected), 1)),
        "skipped_by_class_cap": int(skipped_class_cap),
        "class_counts": {str(k): int(v) for k, v in sorted(class_counts.items())},
    }

def _diagnostic_accuracy(items: list[dict], labels: torch.Tensor, num_classes: int) -> dict:
    if not items:
        return {"count": 0, "accuracy": 0.0}
    correct = 0
    by_action: dict[str, list[int]] = defaultdict(list)
    for item in items:
        node_id = int(item["node_id"])
        pred = int(item["pseudo_label"])
        ok = int(pred == int(labels[node_id].item()))
        correct += ok
        by_action[str(item["action"])].append(ok)
    return {
        "count": int(len(items)),
        "accuracy": float(correct / max(len(items), 1)),
        "by_action": {action: {"count": len(vals), "accuracy": float(sum(vals) / max(len(vals), 1))} for action, vals in sorted(by_action.items())},
    }


def _quality_for_decisions(items: list[dict], labels: torch.Tensor, *, num_nodes: int | None = None) -> dict:
    if not items:
        return {"count": 0, "precision": 0.0, "precision_lcb": 0.0, "coverage": 0.0}
    correct = 0
    for item in items:
        node_id = int(item["node_id"])
        correct += int(int(item["pseudo_label"]) == int(labels[node_id].item()))
    n = len(items)
    return {
        "count": int(n),
        "correct": int(correct),
        "precision": float(correct / max(n, 1)),
        "precision_lcb": float(_wilson_lower_bound(correct, n)),
        "coverage": float(n / max(int(num_nodes or n), 1)),
    }


def _per_source_precision(items: list[dict], labels: torch.Tensor) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        grouped[str(item["action"])].append(item)
    out = {}
    for source in ("consensus", "trust_gnn", "trust_llm"):
        vals = grouped.get(source, [])
        out[source] = _quality_for_decisions(vals, labels)
    return out


def _per_class_precision(items: list[dict], labels: torch.Tensor, num_classes: int) -> dict:
    grouped: dict[int, list[int]] = defaultdict(list)
    for item in items:
        grouped[int(item["pseudo_label"])].append(int(item["node_id"]))
    out = {}
    for cls in range(int(num_classes)):
        ids = grouped.get(cls, [])
        correct = sum(int(cls == int(labels[node_id].item())) for node_id in ids)
        out[str(cls)] = {
            "selected_count": int(len(ids)),
            "correct": int(correct),
            "precision": float(correct / max(len(ids), 1)) if ids else 0.0,
            "precision_lcb": float(_wilson_lower_bound(correct, len(ids))) if ids else 0.0,
        }
    return out


def _calibration_report(scores: list[float], targets: list[int], *, bins: int = 10) -> dict:
    if not scores:
        return {"count": 0, "ece": 0.0, "bins": []}
    bins = max(int(bins), 1)
    bucket_rows = []
    ece = 0.0
    n = len(scores)
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        include_hi = b == bins - 1
        ids = [i for i, s in enumerate(scores) if (s >= lo and (s <= hi if include_hi else s < hi))]
        if not ids:
            bucket_rows.append({"bin": b, "lo": lo, "hi": hi, "count": 0, "confidence": 0.0, "accuracy": 0.0, "gap": 0.0})
            continue
        conf = sum(float(scores[i]) for i in ids) / len(ids)
        acc = sum(int(targets[i]) for i in ids) / len(ids)
        gap = abs(conf - acc)
        ece += (len(ids) / n) * gap
        bucket_rows.append({"bin": b, "lo": lo, "hi": hi, "count": len(ids), "confidence": float(conf), "accuracy": float(acc), "gap": float(gap)})
    return {"count": int(n), "ece": float(ece), "bins": bucket_rows}


def _expected_vs_actual(items: list[dict], labels: torch.Tensor) -> dict:
    if not items:
        return {"count": 0, "expected_precision": 0.0, "actual_precision": 0.0, "gap": 0.0}
    expected = sum(float(item["weight"]) for item in items) / len(items)
    actual = _quality_for_decisions(items, labels)["precision"]
    return {"count": int(len(items)), "expected_precision": float(expected), "actual_precision": float(actual), "gap": float(expected - actual)}


def _conflict_case_accuracy(decisions: list[dict], labels: torch.Tensor) -> dict:
    conflict = [item for item in decisions if int(item["gnn_pred"]) != int(item["llm_pred"])]
    hard = [item for item in conflict if item["pool"] == "H"]
    gnn_correct = sum(int(int(item["gnn_pred"]) == int(labels[int(item["node_id"])].item())) for item in conflict)
    llm_correct = sum(int(int(item["llm_pred"]) == int(labels[int(item["node_id"])].item())) for item in conflict)
    return {
        "conflict_count": int(len(conflict)),
        "selected_count": int(len(hard)),
        "selected_quality": _quality_for_decisions(hard, labels, num_nodes=max(len(conflict), 1)),
        "gnn_accuracy_on_conflicts": float(gnn_correct / max(len(conflict), 1)),
        "llm_accuracy_on_conflicts": float(llm_correct / max(len(conflict), 1)),
    }


def _abstain_quality(decisions: list[dict], labels: torch.Tensor) -> dict:
    abstained = [item for item in decisions if item["pool"] == "A"]
    if not abstained:
        return {"count": 0, "gnn_accuracy": 0.0, "llm_accuracy": 0.0, "oracle_expert_accuracy": 0.0, "both_wrong_rate": 0.0}
    g_ok = [int(int(item["gnn_pred"]) == int(labels[int(item["node_id"])].item())) for item in abstained]
    l_ok = [int(int(item["llm_pred"]) == int(labels[int(item["node_id"])].item())) for item in abstained]
    oracle = [int(g or l) for g, l in zip(g_ok, l_ok)]
    both_wrong = [int((not g) and (not l)) for g, l in zip(g_ok, l_ok)]
    return {
        "count": int(len(abstained)),
        "gnn_accuracy": float(sum(g_ok) / len(abstained)),
        "llm_accuracy": float(sum(l_ok) / len(abstained)),
        "oracle_expert_accuracy": float(sum(oracle) / len(abstained)),
        "both_wrong_rate": float(sum(both_wrong) / len(abstained)),
    }


def _precision_coverage_curve(items: list[dict], labels: torch.Tensor, total_nodes: int) -> list[dict]:
    ordered = sorted(items, key=lambda row: (-float(row["weight"]), int(row["node_id"])))
    curve = []
    correct = 0
    checkpoints = set()
    for frac in (0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
        checkpoints.add(max(1, int(math.ceil(len(ordered) * frac))))
    for i, item in enumerate(ordered, start=1):
        correct += int(int(item["pseudo_label"]) == int(labels[int(item["node_id"])].item()))
        if i in checkpoints or i == len(ordered):
            curve.append(
                {
                    "selected_count": int(i),
                    "coverage": float(i / max(int(total_nodes), 1)),
                    "precision": float(correct / i),
                    "precision_lcb": float(_wilson_lower_bound(correct, i)),
                    "min_weight": float(item["weight"]),
                }
            )
    return curve


def _router_validation_report(
    rows: list[dict],
    decisions: list[dict],
    labels: torch.Tensor,
    num_classes: int,
    r_g: list[float],
    r_l: list[float],
    selected_hard: list[dict],
) -> dict:
    n = len(rows)
    hard_candidates = [item for item in decisions if item["pool"] == "H"]
    g_targets = [int(int(row["gnn_pred"]) == int(labels[int(row["node_id"])].item())) for row in rows]
    l_targets = [int(int(row["llm_pred"]) == int(labels[int(row["node_id"])].item())) for row in rows]
    return {
        "val_count": int(n),
        "selected_pseudo_precision": _quality_for_decisions(selected_hard, labels, num_nodes=n),
        "hard_candidate_precision": _quality_for_decisions(hard_candidates, labels, num_nodes=n),
        "per_source_precision": _per_source_precision(selected_hard, labels),
        "per_class_precision": _per_class_precision(selected_hard, labels, num_classes),
        "precision_coverage_curve": _precision_coverage_curve(hard_candidates, labels, n),
        "calibration": {
            "gnn_head": _calibration_report(r_g, g_targets),
            "llm_head": _calibration_report(r_l, l_targets),
            "scale_check": {
                "r_g_mean": float(sum(r_g) / max(len(r_g), 1)),
                "r_l_mean": float(sum(r_l) / max(len(r_l), 1)),
                "r_g_min": float(min(r_g) if r_g else 0.0),
                "r_l_min": float(min(r_l) if r_l else 0.0),
                "r_g_max": float(max(r_g) if r_g else 0.0),
                "r_l_max": float(max(r_l) if r_l else 0.0),
            },
        },
        "expected_vs_actual_precision": _expected_vs_actual(selected_hard, labels),
        "conflict_case_accuracy": _conflict_case_accuracy(decisions, labels),
        "abstain_quality": _abstain_quality(decisions, labels),
    }


def _optimize_thresholds_on_validation(
    rows: list[dict],
    r_g: list[float],
    r_l: list[float],
    labels: torch.Tensor,
    num_classes: int,
    args,
) -> dict:
    del num_classes
    hard_cap = int(args.router_hard_cap if args.router_hard_cap > 0 else args.joint_max_pseudo_train)
    per_class_cap = int(args.router_per_class_cap if args.router_per_class_cap > 0 else args.joint_pseudo_per_class_cap)
    targets = _parse_float_grid(getattr(args, "router_target_precision_grid", None), (0.85, 0.90, 0.95))
    tau_soft_grid = _parse_float_grid(getattr(args, "router_tau_soft_grid", None), (0.45, 0.50, 0.55, 0.60, 0.65))
    tau_hard_grid = _parse_float_grid(getattr(args, "router_tau_hard_grid", None), (0.65, 0.70, 0.75, 0.80, 0.85, 0.90))
    margin_grid = _parse_float_grid(getattr(args, "router_margin_grid", None), (0.00, 0.05, 0.10, 0.15, 0.20))
    z = float(getattr(args, "router_wilson_z", 1.96))
    best = None
    grid_rows = []
    for target_precision in targets:
        for tau_soft in tau_soft_grid:
            for tau_hard in tau_hard_grid:
                if float(tau_soft) > float(tau_hard):
                    continue
                for margin in margin_grid:
                    decisions = _make_decisions(
                        rows,
                        r_g,
                        r_l,
                        tau_hard=float(tau_hard),
                        tau_soft=float(tau_soft),
                        margin=float(margin),
                        conflict_to_abstain=bool(args.router_conflict_to_abstain),
                    )
                    hard_candidates = [item for item in decisions if item["pool"] == "H"]
                    selected, _ = _select_hard_pool(
                        hard_candidates,
                        target_precision=float(target_precision),
                        total_cap=hard_cap,
                        per_class_cap=per_class_cap,
                    )
                    n = len(selected)
                    if n == 0:
                        row = {
                            "target_precision": float(target_precision),
                            "tau_soft": float(tau_soft),
                            "tau_hard": float(tau_hard),
                            "margin": float(margin),
                            "selected_count": 0,
                            "coverage": 0.0,
                            "precision": 0.0,
                            "precision_lcb": 0.0,
                            "accepted": False,
                        }
                        grid_rows.append(row)
                        continue
                    k = sum(int(int(item["pseudo_label"]) == int(labels[int(item["node_id"])].item())) for item in selected)
                    precision = k / n
                    precision_lcb = _wilson_lower_bound(k, n, z=z)
                    coverage = n / max(len(rows), 1)
                    accepted = precision_lcb >= float(target_precision)
                    row = {
                        "target_precision": float(target_precision),
                        "tau_soft": float(tau_soft),
                        "tau_hard": float(tau_hard),
                        "margin": float(margin),
                        "selected_count": int(n),
                        "correct": int(k),
                        "coverage": float(coverage),
                        "precision": float(precision),
                        "precision_lcb": float(precision_lcb),
                        "accepted": bool(accepted),
                    }
                    grid_rows.append(row)
                    if not accepted:
                        continue
                    score = coverage
                    if best is None or score > best["score"]:
                        best = dict(row, score=float(score))
    if best is None:
        fallback = {
            "score": 0.0,
            "target_precision": float(args.router_target_precision),
            "tau_soft": float(args.router_soft_min_reliability),
            "tau_hard": float(args.router_hard_min_reliability),
            "margin": float(args.router_reliability_margin),
            "selected_count": 0,
            "coverage": 0.0,
            "precision": 0.0,
            "precision_lcb": 0.0,
            "accepted": False,
            "fallback": True,
        }
        best = fallback
    else:
        best["fallback"] = False
    return {"best": best, "grid": grid_rows}


def build_learnable_router_artifacts(
    *,
    data,
    texts: list[str],
    embeddings: torch.Tensor,
    final_gnn_logits: torch.Tensor,
    llm_logits: torch.Tensor,
    candidate_idx: torch.Tensor,
    gnn_artifact_info: dict | None = None,
    num_classes: int,
    args,
    output_dir: Path,
) -> RouterArtifacts:
    router_dir = ensure_dir(Path(args.router_output_dir) if args.router_output_dir else output_dir / "router")
    device = embeddings.device
    train_idx = data.train_mask.bool().nonzero().view(-1).long()
    val_idx = data.val_mask.bool().nonzero().view(-1).long() if hasattr(data, "val_mask") else torch.empty(0, dtype=torch.long, device=device)
    llm_metadata = None
    feature_names = router_feature_names(num_classes)
    oof_head_config = _router_oof_head_config(args, gnn_artifact_info)

    folds = _stratified_folds(train_idx, data.y, args.router_num_folds, args.seed)
    ensure_dir(router_dir / "oof")
    oof_gnn_logits = torch.zeros(int(data.num_nodes), int(num_classes), dtype=torch.float32, device=device)
    train_ids_all = train_idx.detach().cpu().long().tolist()
    for fold_id, holdout_ids in enumerate(folds):
        holdout_idx = torch.tensor(holdout_ids, dtype=torch.long, device=device)
        holdout_set = set(holdout_ids)
        fold_train_ids = [int(i) for i in train_ids_all if int(i) not in holdout_set]
        fold_train_idx = torch.tensor(fold_train_ids, dtype=torch.long, device=device)
        head = MLPHead(
            input_dim=embeddings.size(1),
            hidden_dim=int(oof_head_config["head_hidden_dim"]),
            output_dim=num_classes,
            dropout=float(oof_head_config["dropout"]),
        ).to(device)
        fold_class_weights = _gnn_class_weights(
            data.y,
            fold_train_idx,
            num_classes,
            str(oof_head_config["class_weight"]),
        )
        _, fold_logits = head, train_classifier(
            head,
            embeddings,
            data.y,
            fold_train_idx,
            val_idx,
            epochs=int(oof_head_config["epochs"]),
            lr=float(oof_head_config["lr"]),
            weight_decay=float(oof_head_config["weight_decay"]),
            class_weights=fold_class_weights,
            patience=oof_head_config.get("patience"),
            select_by=str(oof_head_config["select_by"]),
        )
        oof_gnn_logits[holdout_idx] = fold_logits.detach()[holdout_idx].float().to(device)

    train_prepared_logits = _prepare_router_logits(
        oof_gnn_logits.detach(),
        llm_logits.detach(),
        data.y,
        train_idx,
        calibration_enabled=bool(getattr(args, "router_logit_calibration", True)),
        split_name="router_train",
    )
    oof_gnn_temperature = train_prepared_logits["gnn_temperature"]
    router_train_llm_temperature = train_prepared_logits["llm_temperature"]

    oof_rows = build_router_feature_rows(
        node_ids=train_ids_all,
        data=data,
        texts=texts,
        gnn_logits=train_prepared_logits["gnn_logits"].detach(),
        llm_logits=train_prepared_logits["llm_logits"].detach(),
        llm_metadata=llm_metadata,
        max_text_tokens=args.router_max_text_tokens or args.joint_max_input_length,
    )
    _assert_feature_ranges(oof_rows)
    fold_by_node = {int(node_id): int(fold_id) for fold_id, holdout_ids in enumerate(folds) for node_id in holdout_ids}
    rows_by_fold: dict[int, list[dict]] = defaultdict(list)
    for row in oof_rows:
        node_id = int(row["node_id"])
        row["fold"] = int(fold_by_node.get(node_id, -1))
        row["y_true"] = int(data.y[node_id].item())
        row["gnn_correct"] = int(int(row["gnn_pred"]) == row["y_true"])
        row["llm_correct"] = int(int(row["llm_pred"]) == row["y_true"])
        rows_by_fold[int(row["fold"])].append(row)
    for fold_id, rows in sorted(rows_by_fold.items()):
        write_jsonl(rows, router_dir / "oof" / f"fold_{fold_id}_predictions.jsonl")
    write_jsonl(oof_rows, router_dir / "oof" / "router_train.jsonl")

    model, mean, std = _train_two_head_router(
        oof_rows,
        feature_names,
        hidden_dim=args.router_hidden_dim,
        dropout=args.router_dropout,
        epochs=args.router_epochs,
        lr=args.router_lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
    )

    calibration_idx = val_idx if int(val_idx.numel()) > 0 else train_idx
    final_prepared_logits = _prepare_router_logits(
        final_gnn_logits.detach(),
        llm_logits.detach(),
        data.y,
        calibration_idx,
        calibration_enabled=bool(getattr(args, "router_logit_calibration", True)),
        split_name="final_validation" if int(val_idx.numel()) > 0 else "final_train",
    )
    final_gnn_temperature = final_prepared_logits["gnn_temperature"]
    final_llm_temperature = final_prepared_logits["llm_temperature"]
    final_gnn_for_router = final_prepared_logits["gnn_logits"]
    final_llm_for_router = final_prepared_logits["llm_logits"]
    logit_calibration_status = final_prepared_logits["logit_calibration"]

    hard_cap = int(args.router_hard_cap if args.router_hard_cap > 0 else args.joint_max_pseudo_train)
    per_class_cap = int(args.router_per_class_cap if args.router_per_class_cap > 0 else args.joint_pseudo_per_class_cap)
    threshold_config = {
        "target_precision": float(args.router_target_precision),
        "tau_soft": float(args.router_soft_min_reliability),
        "tau_hard": float(args.router_hard_min_reliability),
        "margin": float(args.router_reliability_margin),
        "source": "fixed_args",
    }
    validation_report = None
    threshold_search = None
    val_rows: list[dict] = []
    if bool(getattr(args, "router_optimize_thresholds", True)) and int(val_idx.numel()) > 0:
        val_ids = val_idx.detach().cpu().long().tolist()
        val_rows = build_router_feature_rows(
            node_ids=val_ids,
            data=data,
            texts=texts,
            gnn_logits=final_gnn_for_router.detach(),
            llm_logits=final_llm_for_router.detach(),
            llm_metadata=llm_metadata,
            max_text_tokens=args.router_max_text_tokens or args.joint_max_input_length,
        )
        _assert_feature_ranges(val_rows)
        val_r_g, val_r_l = _score_rows_with_router(model, mean, std, val_rows, feature_names, device)
        threshold_search = _optimize_thresholds_on_validation(val_rows, val_r_g, val_r_l, data.y, num_classes, args)
        best = threshold_search["best"]
        threshold_config = {
            "target_precision": float(best["target_precision"]),
            "tau_soft": float(best["tau_soft"]),
            "tau_hard": float(best["tau_hard"]),
            "margin": float(best["margin"]),
            "source": "validation_grid_search" if not best.get("fallback") else "fallback_fixed_args",
            "validation_precision": float(best.get("precision", 0.0)),
            "validation_precision_lcb": float(best.get("precision_lcb", 0.0)),
            "validation_coverage": float(best.get("coverage", 0.0)),
            "validation_selected_count": int(best.get("selected_count", 0)),
        }
        val_decisions = _make_decisions(
            val_rows,
            val_r_g,
            val_r_l,
            tau_hard=threshold_config["tau_hard"],
            tau_soft=threshold_config["tau_soft"],
            margin=threshold_config["margin"],
            conflict_to_abstain=bool(args.router_conflict_to_abstain),
        )
        val_hard_candidates = [item for item in val_decisions if item["pool"] == "H"]
        val_selected_hard, val_hard_select_stats = _select_hard_pool(
            val_hard_candidates,
            target_precision=threshold_config["target_precision"],
            total_cap=hard_cap,
            per_class_cap=per_class_cap,
        )
        validation_report = _router_validation_report(
            val_rows,
            val_decisions,
            data.y,
            num_classes,
            val_r_g,
            val_r_l,
            val_selected_hard,
        )
        validation_report["hard_selection"] = val_hard_select_stats
        write_jsonl(val_rows, router_dir / "validation_router_features.jsonl")
        write_jsonl(val_decisions, router_dir / "validation_router_outputs.jsonl")
        write_jsonl(val_selected_hard, router_dir / "validation_hard_pool.jsonl")
        write_jsonl(threshold_search["grid"], router_dir / "validation_threshold_grid.jsonl")

    candidate_ids = candidate_idx.detach().cpu().long().tolist()
    feature_rows = build_router_feature_rows(
        node_ids=candidate_ids,
        data=data,
        texts=texts,
        gnn_logits=final_gnn_for_router.detach(),
        llm_logits=final_llm_for_router.detach(),
        llm_metadata=llm_metadata,
        max_text_tokens=args.router_max_text_tokens or args.joint_max_input_length,
    )
    _assert_feature_ranges(feature_rows)
    r_g, r_l = _score_rows_with_router(model, mean, std, feature_rows, feature_names, device)

    if bool(getattr(args, "router_reliability_only", False)):
        reliability_rows = []
        for row, rg, rl in zip(feature_rows, r_g, r_l):
            reliability_rows.append({
                "node_id": int(row["node_id"]),
                "r_g": float(rg),
                "r_l": float(rl),
                "gnn_pred": int(row["gnn_pred"]),
                "llm_pred": int(row["llm_pred"]),
            })
        decisions = [
            {
                "node_id": item["node_id"],
                "pool": "R",
                "action": "reliability_only",
                "pseudo_label": None,
                "weight": float(max(item["r_g"], item["r_l"])),
                "r_g": item["r_g"],
                "r_l": item["r_l"],
                "gnn_pred": item["gnn_pred"],
                "llm_pred": item["llm_pred"],
            }
            for item in reliability_rows
        ]

        validation_report = None
        if int(val_idx.numel()) > 0:
            val_ids = val_idx.detach().cpu().long().tolist()
            val_rows = build_router_feature_rows(
                node_ids=val_ids,
                data=data,
                texts=texts,
                gnn_logits=final_gnn_for_router.detach(),
                llm_logits=final_llm_for_router.detach(),
                llm_metadata=llm_metadata,
                max_text_tokens=args.router_max_text_tokens or args.joint_max_input_length,
            )
            _assert_feature_ranges(val_rows)
            val_r_g, val_r_l = _score_rows_with_router(model, mean, std, val_rows, feature_names, device)
            validation_report = _router_validation_report(
                val_rows,
                [
                    {
                        "node_id": int(row["node_id"]),
                        "pool": "R",
                        "action": "reliability_only",
                        "pseudo_label": None,
                        "weight": float(max(rg, rl)),
                        "r_g": float(rg),
                        "r_l": float(rl),
                        "gnn_pred": int(row["gnn_pred"]),
                        "llm_pred": int(row["llm_pred"]),
                    }
                    for row, rg, rl in zip(val_rows, val_r_g, val_r_l)
                ],
                data.y,
                num_classes,
                val_r_g,
                val_r_l,
                [],
            )
            write_jsonl(val_rows, router_dir / "validation_router_features.jsonl")
            write_jsonl(
                [
                    {
                        "node_id": int(row["node_id"]),
                        "r_g": float(rg),
                        "r_l": float(rl),
                        "gnn_pred": int(row["gnn_pred"]),
                        "llm_pred": int(row["llm_pred"]),
                    }
                    for row, rg, rl in zip(val_rows, val_r_g, val_r_l)
                ],
                router_dir / "validation_reliability_scores.jsonl",
            )

        write_jsonl(feature_rows, router_dir / "router_features.jsonl")
        write_jsonl(reliability_rows, router_dir / "router_reliability_scores.jsonl")
        write_jsonl(decisions, router_dir / "router_outputs.jsonl")
        write_jsonl([], router_dir / "hard_pool.jsonl")
        write_jsonl([], router_dir / "soft_pool.jsonl")
        write_jsonl([], router_dir / "abstain_pool.jsonl")

        threshold_config = {
            "target_precision": float(args.router_target_precision),
            "tau_soft": None,
            "tau_hard": None,
            "margin": None,
            "source": "disabled_reliability_only",
        }
        temperature_metadata = _temperature_metadata(
            oof_gnn_temperature,
            router_train_llm_temperature,
            final_gnn_temperature,
            final_llm_temperature,
            final_calibration_split="validation" if int(val_idx.numel()) > 0 else "train",
        )
        torch.save(
            {
                "model_state": model.state_dict(),
                "feature_names": feature_names,
                "mean": mean,
                "std": std,
                "config": {
                    "hidden_dim": int(args.router_hidden_dim),
                    "dropout": float(args.router_dropout),
                    "policy": "learnable_reliability_only",
                    "oof_gnn_head_config": oof_head_config,
                    "thresholds": threshold_config,
                    "temperatures": temperature_metadata,
                    "logit_calibration": logit_calibration_status,
                },
            },
            router_dir / "router.pt",
        )

        labels = torch.full_like(data.y, fill_value=-1)
        labels[train_idx] = data.y[train_idx]
        label_sources: dict[int, str] = {int(node_id): "true" for node_id in train_ids_all}
        sample_weights = torch.ones(data.num_nodes, dtype=torch.float32, device=data.y.device)
        idx = train_idx.detach().long()
        stats = {
            "policy": "learnable_router_v5_no_calibration_reliability_only" if not bool(getattr(args, "router_logit_calibration", True)) else "learnable_router_reliability_only",
            "router_dir": str(router_dir),
            "feature_names": feature_names,
            "scalar_feature_names": SCALAR_FEATURE_NAMES,
            "oof_train_count": int(len(oof_rows)),
            "oof_folds": int(len(folds)),
            "oof_gnn_head_config": oof_head_config,
            "candidate_count": int(len(feature_rows)),
            "reliability_score_count": int(len(reliability_rows)),
            "router_reliability_only": True,
            "logit_calibration": logit_calibration_status,
            "pool_counts_before_hard_selection": {"R": int(len(reliability_rows))},
            "action_counts_before_hard_selection": {"reliability_only": int(len(reliability_rows))},
            "hard_candidates": 0,
            "soft_pool": 0,
            "abstain_pool": 0,
            "pseudo_selected": 0,
            "true_count": int(len(train_ids_all)),
            "total_selected": int(idx.numel()),
            "selected_pseudo_sources": {},
            "selected_action_counts": {},
            "hard_selection": {
                "target_precision_reference": float(args.router_target_precision),
                "total_cap": 0,
                "per_class_cap": 0,
                "source": "disabled_reliability_only",
            },
            "selected_pseudo_quality": _diagnostic_accuracy([], data.y, num_classes),
            "thresholds": threshold_config,
            "threshold_search": None,
            "validation_report": validation_report,
            "temperatures": temperature_metadata,
            "target_precision": float(args.router_target_precision),
            "hard_min_reliability": None,
            "soft_min_reliability": None,
            "reliability_margin": None,
            "hard_cap": 0,
            "per_class_cap": 0,
        }
        (router_dir / "router_summary.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return RouterArtifacts(labels=labels, idx=idx, label_sources=label_sources, sample_weights=sample_weights, stats=stats, feature_names=feature_names)

    decisions = _make_decisions(
        feature_rows,
        r_g,
        r_l,
        tau_hard=threshold_config["tau_hard"],
        tau_soft=threshold_config["tau_soft"],
        margin=threshold_config["margin"],
        conflict_to_abstain=bool(args.router_conflict_to_abstain),
    )

    hard_candidates: list[dict] = [item for item in decisions if item["pool"] == "H"]
    soft_pool: list[dict] = [item for item in decisions if item["pool"] == "S"]
    abstain_pool: list[dict] = [item for item in decisions if item["pool"] == "A"]
    selected_hard, hard_select_stats = _select_hard_pool(
        hard_candidates,
        target_precision=threshold_config["target_precision"],
        total_cap=hard_cap,
        per_class_cap=per_class_cap,
    )
    selected_ids = {int(item["node_id"]) for item in selected_hard}
    hard_rejected = [dict(item, pool="A", action="hard_pool_rejected_by_cap") for item in hard_candidates if int(item["node_id"]) not in selected_ids]
    abstain_pool = abstain_pool + hard_rejected

    write_jsonl(feature_rows, router_dir / "router_features.jsonl")
    write_jsonl(decisions, router_dir / "router_outputs.jsonl")
    write_jsonl(selected_hard, router_dir / "hard_pool.jsonl")
    write_jsonl(soft_pool, router_dir / "soft_pool.jsonl")
    write_jsonl(abstain_pool, router_dir / "abstain_pool.jsonl")

    torch.save(
        {
            "model_state": model.state_dict(),
            "feature_names": feature_names,
            "mean": mean,
            "std": std,
            "config": {
                "hidden_dim": int(args.router_hidden_dim),
                "dropout": float(args.router_dropout),
                "policy": "learnable",
                "oof_gnn_head_config": oof_head_config,
                "thresholds": threshold_config,
                "temperatures": _temperature_metadata(
                    oof_gnn_temperature,
                    router_train_llm_temperature,
                    final_gnn_temperature,
                    final_llm_temperature,
                ),
            },
        },
        router_dir / "router.pt",
    )

    labels = torch.full_like(data.y, fill_value=-1)
    label_sources: dict[int, str] = {}
    labels[train_idx] = data.y[train_idx]
    sample_weights = torch.ones(data.num_nodes, dtype=torch.float32, device=data.y.device)
    for node_id in train_ids_all:
        label_sources[int(node_id)] = "true"
    for item in selected_hard:
        node_id = int(item["node_id"])
        labels[node_id] = int(item["pseudo_label"])
        label_sources[node_id] = f"router_{item['action']}"
        sample_weights[node_id] = float(item["weight"])
    idx = (labels >= 0).nonzero().view(-1)

    action_counts = Counter(str(item["action"]) for item in decisions)
    pool_counts = Counter(str(item["pool"]) for item in decisions)
    selected_action_counts = Counter(str(item["action"]) for item in selected_hard)
    selected_sources = Counter("consensus" if item["action"] == "consensus" else "gnn" if item["action"] == "trust_gnn" else "llm" for item in selected_hard)
    stats = {
        "policy": "learnable_router_v4_calibrated_evidence",
        "router_dir": str(router_dir),
        "feature_names": feature_names,
        "scalar_feature_names": SCALAR_FEATURE_NAMES,
        "oof_train_count": int(len(oof_rows)),
        "oof_folds": int(len(folds)),
        "oof_gnn_head_config": oof_head_config,
        "candidate_count": int(len(feature_rows)),
        "pool_counts_before_hard_selection": dict(pool_counts),
        "action_counts_before_hard_selection": dict(action_counts),
        "hard_candidates": int(len(hard_candidates)),
        "soft_pool": int(len(soft_pool)),
        "abstain_pool": int(len(abstain_pool)),
        "pseudo_selected": int(len(selected_hard)),
        "true_count": int(len(train_ids_all)),
        "total_selected": int(idx.numel()),
        "selected_pseudo_sources": dict(selected_sources),
        "selected_action_counts": dict(selected_action_counts),
        "hard_selection": hard_select_stats,
        "selected_pseudo_quality": _diagnostic_accuracy(selected_hard, data.y, num_classes),
        "thresholds": threshold_config,
        "threshold_search": threshold_search,
        "validation_report": validation_report,
        "temperatures": _temperature_metadata(
            oof_gnn_temperature,
            router_train_llm_temperature,
            final_gnn_temperature,
            final_llm_temperature,
            final_calibration_split="validation" if int(val_idx.numel()) > 0 else "train",
        ),
        "target_precision": float(threshold_config["target_precision"]),
        "hard_min_reliability": float(threshold_config["tau_hard"]),
        "soft_min_reliability": float(threshold_config["tau_soft"]),
        "reliability_margin": float(threshold_config["margin"]),
        "hard_cap": int(hard_cap),
        "per_class_cap": int(per_class_cap),
    }
    (router_dir / "router_summary.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return RouterArtifacts(labels=labels, idx=idx, label_sources=label_sources, sample_weights=sample_weights, stats=stats, feature_names=feature_names)
