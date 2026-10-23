from __future__ import annotations

import argparse
import html
import json
import random
import re
from dataclasses import asdict
from pathlib import Path

import torch

from .codebook import fit_codebook
from .data import create_few_shot_dataset, get_texts, label_space, masks_dict
from .llm_io import load_llm_logits
from .metrics import accuracy, macro_f1, split_metrics, weighted_f1
from .models.gnn import GNNEncoder, MLPHead
from .models.joint import JointGraphTextClassifier
from .models.ssl import SSLStats, pretrain_ssl_gnn
from .prompts import DIRECT_PROMPTS
from .pseudolabel import build_pseudo_label_tensor, partition_trust_regions, summarize_regions
from .router import build_learnable_router_artifacts
from .train import joint_llm_logits_for_indices, train_classifier, train_joint_llm_model
from .utils import choose_device, ensure_dir, save_json, set_seed, write_jsonl


ROOT = Path(__file__).resolve().parents[1]


def _idx_from_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask.bool().nonzero().view(-1)


def _candidate_indices(data, mode: str) -> torch.Tensor:
    if mode == "all":
        return torch.arange(data.num_nodes, device=data.y.device)
    if mode == "unlabeled":
        return (~data.train_mask.bool()).nonzero().view(-1)
    if mode == "no-val":
        val_mask = getattr(data, "val_mask", torch.zeros_like(data.train_mask)).bool()
        return (~(data.train_mask.bool() | val_mask)).nonzero().view(-1)
    raise ValueError(f"Unknown candidate mode: {mode}")


def _train_gnn_head(data, embeddings, num_classes: int, args):
    head = MLPHead(
        input_dim=embeddings.size(1),
        hidden_dim=args.head_hidden_dim,
        output_dim=num_classes,
        dropout=args.dropout,
    ).to(embeddings.device)
    val_idx = _idx_from_mask(data.val_mask) if hasattr(data, "val_mask") else None
    return head, train_classifier(
        head,
        embeddings,
        data.y,
        _idx_from_mask(data.train_mask),
        val_idx,
        epochs=args.head_epochs,
        lr=args.head_lr,
        weight_decay=args.weight_decay,
    )


def _resolve_artifact_path(path: str | Path, artifact_root: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    candidates = [ROOT / path, Path.cwd() / path, artifact_root / path.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_selected_gnn_artifacts(data, num_classes: int, label_names: list[str], args, device):
    artifact_root = Path(args.selected_gnn_root)
    manifest_path = Path(args.selected_gnn_manifest) if args.selected_gnn_manifest else artifact_root / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    try:
        entry = manifest["datasets"][args.dataset]
    except KeyError as exc:
        raise KeyError(f"No selected GNN manifest entry for dataset={args.dataset}: {manifest_path}") from exc

    paths = entry["artifact_paths"]
    encoder_embedding_path = _resolve_artifact_path(paths["encoder_and_embeddings"], artifact_root)
    head_path = _resolve_artifact_path(paths["mlp_head"], artifact_root)
    prediction_path = _resolve_artifact_path(paths["mlp_predictions"], artifact_root)
    metrics_path = _resolve_artifact_path(paths["mlp_metrics"], artifact_root) if "mlp_metrics" in paths else None

    encoder_payload = torch.load(encoder_embedding_path, map_location=device, weights_only=False)
    embeddings = encoder_payload["embeddings"].to(device=device, dtype=torch.float32)
    if embeddings.size(0) != data.num_nodes:
        raise ValueError(
            f"Selected GNN embeddings node count mismatch for {args.dataset}: "
            f"{embeddings.size(0)} != {data.num_nodes}"
        )

    encoder = GNNEncoder(
        input_dim=data.x.size(1),
        hidden_dim=args.gnn_hidden_dim,
        embedding_dim=embeddings.size(1),
        n_layers=args.gnn_layers,
        gnn_type=args.gnn_type,
        dropout=args.dropout,
    ).to(device)
    encoder_state = encoder_payload.get("encoder_state")
    if encoder_state is None:
        encoder_state = torch.load(_resolve_artifact_path(paths["ssl_gnn_encoder"], artifact_root), map_location=device, weights_only=False)
    encoder.load_state_dict(encoder_state)
    encoder.eval()

    head_payload = torch.load(head_path, map_location=device, weights_only=False)
    saved_label_names = head_payload.get("label_names")
    if saved_label_names is not None and list(saved_label_names) != list(label_names):
        raise ValueError(f"Selected GNN label names mismatch for {args.dataset}")
    head_state = head_payload["head_state"]
    head_hidden_dim = int(head_state["net.0.weight"].size(0))
    gnn_head = MLPHead(
        input_dim=embeddings.size(1),
        hidden_dim=head_hidden_dim,
        output_dim=num_classes,
        dropout=0.0,
    ).to(device)
    gnn_head.load_state_dict(head_state)
    gnn_head.eval()

    pred_payload = torch.load(prediction_path, map_location=device, weights_only=False)
    gnn_logits = pred_payload["logits"].to(device=device, dtype=torch.float32)
    if tuple(gnn_logits.shape) != (data.num_nodes, num_classes):
        raise ValueError(
            f"Selected GNN logits shape mismatch for {args.dataset}: "
            f"{tuple(gnn_logits.shape)} != {(data.num_nodes, num_classes)}"
        )

    mlp_metrics_payload = {}
    if metrics_path is not None and metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            mlp_metrics_payload = json.load(f)

    ssl_stats = SSLStats(final_loss=0.0, final_feature_loss=0.0, final_edge_loss=0.0)
    artifact_info = {
        "source": "selected_gcn_ssl_v2_artifacts",
        "manifest": str(manifest_path),
        "artifact_root": str(artifact_root),
        "ssl_config": entry.get("ssl_config"),
        "mlp_config": entry.get("mlp_config"),
        "mlp_trial": entry.get("mlp_trial"),
        "metrics": entry.get("metrics"),
        "mlp_trial_config": mlp_metrics_payload.get("trial"),
        "mlp_train_stats": mlp_metrics_payload.get("train_stats"),
        "artifact_paths": {
            "encoder_and_embeddings": str(encoder_embedding_path),
            "mlp_head": str(head_path),
            "mlp_predictions": str(prediction_path),
            "mlp_metrics": None if metrics_path is None else str(metrics_path),
        },
    }
    return encoder, embeddings.detach(), ssl_stats, gnn_head, gnn_logits.detach(), artifact_info


def _llm_logits(data, num_classes: int, args, device):
    logits = load_llm_logits(args.llm_logits_path, data.num_nodes, num_classes, device)
    if logits is None:
        raise ValueError(
            "LLM class logits are required. Pass --llm-logits-path pointing to a .pt/.json/.jsonl "
            "file with one num_classes-length logit vector per scored node. Use "
            "scripts/hf_label_logprob_infer.py to compute class logits for multi-token label names."
        )
    return logits, "external_class_logits"


def _pseudo_label_quality(
    pseudo_labels: torch.Tensor,
    pseudo_idx: torch.Tensor,
    label_sources: dict[int, str],
    y: torch.Tensor,
    num_classes: int,
):
    pseudo_only = [
        int(node_id)
        for node_id in pseudo_idx.detach().cpu().tolist()
        if label_sources.get(int(node_id)) != "true"
    ]

    def score(node_ids: list[int]):
        if not node_ids:
            return {"count": 0, "accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
        idx = torch.tensor(node_ids, dtype=torch.long, device=y.device)
        pred = pseudo_labels[idx].long()
        gold = y[idx].long()
        return {
            "count": int(idx.numel()),
            "accuracy": accuracy(pred, gold),
            "macro_f1": macro_f1(pred, gold, num_classes),
            "weighted_f1": weighted_f1(pred, gold, num_classes),
        }

    by_source: dict[str, dict] = {}
    for node_id in pseudo_only:
        source = label_sources.get(node_id, "unknown")
        by_source.setdefault(source, {"node_ids": []})["node_ids"].append(node_id)
    by_source = {source: score(values["node_ids"]) for source, values in sorted(by_source.items())}
    return {"overall_pseudo": score(pseudo_only), "by_source": by_source}


def _score_node_ids(pred: torch.Tensor, y: torch.Tensor, node_ids: list[int], num_classes: int):
    if not node_ids:
        return {"count": 0, "accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
    idx = torch.tensor(node_ids, dtype=torch.long, device=y.device)
    pred = pred[idx].long()
    gold = y[idx].long()
    return {
        "count": int(idx.numel()),
        "accuracy": accuracy(pred, gold),
        "macro_f1": macro_f1(pred, gold, num_classes),
        "weighted_f1": weighted_f1(pred, gold, num_classes),
    }


def _trust_region_quality(trust_records, y: torch.Tensor, num_classes: int):
    by_region: dict[str, dict] = {}
    gnn_pred = torch.full_like(y, -1)
    llm_pred = torch.full_like(y, -1)
    pseudo_pred = torch.full_like(y, -1)
    pseudo_node_ids: list[int] = []
    for rec in trust_records:
        node_id = int(rec.node_id)
        by_region.setdefault(rec.region, {"node_ids": [], "pseudo_node_ids": []})["node_ids"].append(node_id)
        gnn_pred[node_id] = int(rec.gnn_pred)
        llm_pred[node_id] = int(rec.llm_pred)
        if rec.pseudo_label is not None:
            pseudo_pred[node_id] = int(rec.pseudo_label)
            by_region[rec.region]["pseudo_node_ids"].append(node_id)
            pseudo_node_ids.append(node_id)

    summary = {}
    for region, values in sorted(by_region.items()):
        node_ids = values["node_ids"]
        pseudo_ids = values["pseudo_node_ids"]
        summary[region] = {
            "count": int(len(node_ids)),
            "pseudo_count": int(len(pseudo_ids)),
            "gnn_pred_quality": _score_node_ids(gnn_pred, y, node_ids, num_classes),
            "llm_pred_quality": _score_node_ids(llm_pred, y, node_ids, num_classes),
            "pseudo_label_quality": _score_node_ids(pseudo_pred, y, pseudo_ids, num_classes),
        }
    summary["pseudo_labeled_overall"] = _score_node_ids(pseudo_pred, y, pseudo_node_ids, num_classes)
    return summary


def _selected_pseudo_quality(
    joint_train_idx: torch.Tensor,
    pseudo_labels: torch.Tensor,
    label_sources: dict[int, str],
    y: torch.Tensor,
    num_classes: int,
):
    selected = [
        int(node_id)
        for node_id in joint_train_idx.detach().cpu().tolist()
        if label_sources.get(int(node_id)) != "true"
    ]
    return _score_node_ids(pseudo_labels, y, selected, num_classes)


def _codebook_coverage(token_ids: torch.Tensor, train_idx: torch.Tensor, eval_idx: torch.Tensor, codebook_size: int, val_idx=None):
    token_cpu = token_ids.detach().cpu().long()

    def token_set(idx):
        if idx is None or idx.numel() == 0:
            return set()
        return set(int(v) for v in token_cpu[idx.detach().cpu().long()].tolist())

    train_tokens = token_set(train_idx)
    eval_tokens = token_set(eval_idx)
    val_tokens = token_set(val_idx)
    unseen_eval_tokens = sorted(eval_tokens - train_tokens)
    eval_cpu = eval_idx.detach().cpu().long()
    unseen_eval_nodes = [int(i) for i in eval_cpu.tolist() if int(token_cpu[int(i)].item()) in set(unseen_eval_tokens)]
    return {
        "codebook_size": int(codebook_size),
        "train_unique_tokens": int(len(train_tokens)),
        "val_unique_tokens": int(len(val_tokens)),
        "eval_unique_tokens": int(len(eval_tokens)),
        "eval_unseen_token_count": int(len(unseen_eval_tokens)),
        "eval_unseen_tokens": unseen_eval_tokens,
        "eval_nodes_with_unseen_token": int(len(unseen_eval_nodes)),
        "eval_nodes_with_unseen_token_ratio": float(len(unseen_eval_nodes) / max(int(eval_idx.numel()), 1)),
    }


def _split_accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor, idx: torch.Tensor) -> float:
    if idx is None or idx.numel() == 0:
        return 0.0
    return accuracy(logits[idx].argmax(dim=-1), y[idx])


def _val_gate_baseline(args, gnn_logits: torch.Tensor, llm_logits: torch.Tensor, y: torch.Tensor, val_idx: torch.Tensor):
    gnn_val = _split_accuracy_from_logits(gnn_logits, y, val_idx)
    llm_val = _split_accuracy_from_logits(llm_logits, y, val_idx)
    if args.joint_val_gate_baseline == "gnn":
        return gnn_val, {"name": "gnn", "gnn_val_accuracy": gnn_val, "llm_val_accuracy": llm_val}
    if args.joint_val_gate_baseline == "llm":
        return llm_val, {"name": "llm", "gnn_val_accuracy": gnn_val, "llm_val_accuracy": llm_val}
    return max(gnn_val, llm_val), {"name": "best_single", "gnn_val_accuracy": gnn_val, "llm_val_accuracy": llm_val}




def _joint_supervision(data, pseudo_labels: torch.Tensor, pseudo_idx: torch.Tensor, args):
    labels = pseudo_labels.clone()
    train_idx = pseudo_idx.long()
    val_idx = None
    val_mask = getattr(data, "val_mask", None)

    if val_mask is not None:
        val_mask = val_mask.bool()
        labels[val_mask] = data.y[val_mask]
        val_idx = _idx_from_mask(val_mask)
        if args.exclude_val_from_joint_train:
            train_idx = train_idx[~val_mask[train_idx]]

    train_idx = train_idx[labels[train_idx] >= 0]
    val_train_overlap = 0
    if val_idx is not None and val_idx.numel() > 0:
        train_set = set(train_idx.detach().cpu().tolist())
        val_train_overlap = sum(1 for node_id in val_idx.detach().cpu().tolist() if node_id in train_set)
    if val_train_overlap != 0:
        raise ValueError(f"Validation leakage detected: {val_train_overlap} validation nodes are in joint train_idx.")

    stats = {
        "train_size_before_selection": int(train_idx.numel()),
        "val_size": 0 if val_idx is None else int(val_idx.numel()),
        "val_label_source": "true",
        "exclude_val_from_train": bool(args.exclude_val_from_joint_train),
        "val_train_overlap": int(val_train_overlap),
    }
    return labels, train_idx, val_idx, stats


def _record_confidence_for_source(rec) -> float:
    if rec.label_source == "consensus":
        return min(rec.gnn_conf, rec.llm_conf)
    if rec.label_source in {"gnn", "gnn_high_conflict"}:
        return rec.gnn_conf
    if rec.label_source in {"llm", "llm_high_conflict"}:
        return rec.llm_conf
    return max(rec.gnn_conf, rec.llm_conf)


def _split_csv_words(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for chunk in str(value).split(",") for item in chunk.split() if item.strip()]


def _parse_source_float_map(value: str | None, defaults: dict[str, float]) -> dict[str, float]:
    parsed = dict(defaults)
    for item in _split_csv_words(value):
        if "=" not in item:
            raise ValueError(f"Expected source=value item, got: {item}")
        source, raw = item.split("=", 1)
        parsed[source.strip()] = float(raw)
    return parsed


def _source_priority_map(value: str | None) -> dict[str, int]:
    defaults = ["consensus", "llm", "gnn", "llm_high_conflict", "gnn_high_conflict"]
    sources = _split_csv_words(value) or defaults
    return {source: rank for rank, source in enumerate(sources)}


def _quota_targets(total: int, quotas: dict[str, float], priority_map: dict[str, int]) -> dict[str, int]:
    if total <= 0 or not quotas:
        return {}
    positive = {source: max(float(weight), 0.0) for source, weight in quotas.items() if float(weight) > 0}
    weight_sum = sum(positive.values())
    if weight_sum <= 0:
        return {}
    raw = {source: total * weight / weight_sum for source, weight in positive.items()}
    targets = {source: int(value) for source, value in raw.items()}
    remainder = total - sum(targets.values())
    order = sorted(
        positive,
        key=lambda source: (-(raw[source] - targets[source]), priority_map.get(source, len(priority_map)), source),
    )
    for source in order[:remainder]:
        targets[source] += 1
    return targets


def _select_joint_train_idx(
    train_idx: torch.Tensor,
    labels: torch.Tensor,
    label_sources: dict[int, str],
    trust_records,
    args,
    num_classes: int,
):
    record_by_node = {int(rec.node_id): rec for rec in trust_records}
    true_ids: list[int] = []
    pseudo_items: list[tuple[int, float, int, int, str]] = []
    before_by_source: dict[str, int] = {}
    rejected_by_source: dict[str, int] = {}
    allowed_sources = set(_split_csv_words(args.joint_allowed_sources))
    source_thresholds = _parse_source_float_map(
        args.joint_source_min_confidence,
        {"consensus": 0.45, "llm": 0.75, "gnn": 0.80, "llm_high_conflict": 0.95, "gnn_high_conflict": 0.95},
    )
    priority_map = _source_priority_map(args.joint_source_priority)
    source_quotas = _parse_source_float_map(args.joint_source_quotas, {}) if args.joint_source_quotas else {}

    for node_id in train_idx.detach().cpu().tolist():
        node_id = int(node_id)
        source = label_sources.get(node_id, "unknown")
        if source == "true":
            true_ids.append(node_id)
            continue
        rec = record_by_node.get(node_id)
        if rec is None:
            continue
        before_by_source[source] = before_by_source.get(source, 0) + 1
        confidence = _record_confidence_for_source(rec)

        if args.joint_selection_policy == "confidence":
            if confidence < args.joint_pseudo_min_confidence:
                rejected_by_source[source] = rejected_by_source.get(source, 0) + 1
                continue
            priority = 0
        elif args.joint_selection_policy in {"source_threshold", "source_quota"}:
            if source not in allowed_sources:
                rejected_by_source[source] = rejected_by_source.get(source, 0) + 1
                continue
            threshold = source_thresholds.get(source, args.joint_pseudo_min_confidence)
            if confidence < threshold:
                rejected_by_source[source] = rejected_by_source.get(source, 0) + 1
                continue
            priority = priority_map.get(source, len(priority_map))
        else:
            raise ValueError(f"Unknown joint_selection_policy: {args.joint_selection_policy}")

        pseudo_items.append((priority, float(confidence), node_id, int(labels[node_id].item()), source))

    pseudo_items.sort(key=lambda item: (item[0], -item[1], item[2]))
    after_confidence = len(pseudo_items)
    if args.joint_selection_policy == "source_quota" and args.joint_max_pseudo_train > 0:
        max_items = int(args.joint_max_pseudo_train)
        targets = _quota_targets(max_items, source_quotas, priority_map)
        by_source: dict[str, list[tuple[int, float, int, int, str]]] = {}
        for item in pseudo_items:
            by_source.setdefault(item[4], []).append(item)
        quota_selected: list[tuple[int, float, int, int, str]] = []
        used_nodes: set[int] = set()
        for source in sorted(targets, key=lambda src: priority_map.get(src, len(priority_map))):
            for item in by_source.get(source, [])[: targets[source]]:
                quota_selected.append(item)
                used_nodes.add(item[2])
        if len(quota_selected) < max_items:
            for item in pseudo_items:
                if item[2] in used_nodes:
                    continue
                quota_selected.append(item)
                used_nodes.add(item[2])
                if len(quota_selected) >= max_items:
                    break
        for item in pseudo_items:
            if item[2] not in used_nodes:
                rejected_by_source[item[4]] = rejected_by_source.get(item[4], 0) + 1
        pseudo_items = quota_selected[:max_items]

    if args.joint_pseudo_per_class_cap > 0:
        class_counts = [0 for _ in range(num_classes)]
        capped = []
        for item in pseudo_items:
            label = item[3]
            if class_counts[label] >= args.joint_pseudo_per_class_cap:
                rejected_by_source[item[4]] = rejected_by_source.get(item[4], 0) + 1
                continue
            class_counts[label] += 1
            capped.append(item)
        pseudo_items = capped
    else:
        class_counts = None
    after_class_cap = len(pseudo_items)
    if args.joint_max_pseudo_train > 0 and args.joint_selection_policy != "source_quota":
        overflow = pseudo_items[args.joint_max_pseudo_train :]
        for item in overflow:
            rejected_by_source[item[4]] = rejected_by_source.get(item[4], 0) + 1
        pseudo_items = pseudo_items[: args.joint_max_pseudo_train]

    selected_ids = sorted(true_ids) + [item[2] for item in pseudo_items]
    selected = torch.tensor(selected_ids, dtype=torch.long, device=train_idx.device)
    selected_sources: dict[str, int] = {}
    for _, _, _, _, source in pseudo_items:
        selected_sources[source] = selected_sources.get(source, 0) + 1
    stats = {
        "policy": args.joint_selection_policy,
        "true_count": int(len(true_ids)),
        "pseudo_before_selection": int(sum(1 for node_id in train_idx.detach().cpu().tolist() if label_sources.get(int(node_id)) != "true")),
        "pseudo_before_selection_by_source": before_by_source,
        "pseudo_after_confidence": int(after_confidence),
        "pseudo_after_class_cap": int(after_class_cap),
        "pseudo_selected": int(len(pseudo_items)),
        "total_selected": int(selected.numel()),
        "min_confidence": float(args.joint_pseudo_min_confidence),
        "source_min_confidence": source_thresholds,
        "source_quotas": source_quotas,
        "allowed_sources": sorted(allowed_sources),
        "source_priority": priority_map,
        "max_pseudo_train": int(args.joint_max_pseudo_train),
        "per_class_cap": int(args.joint_pseudo_per_class_cap),
        "selected_pseudo_sources": selected_sources,
        "rejected_pseudo_sources": rejected_by_source,
    }
    if class_counts is not None:
        stats["selected_pseudo_class_counts"] = class_counts
    return selected, stats

def _joint_eval_masks(data, split: str):
    base = masks_dict(data)
    if split == "all":
        return base
    if split not in base:
        raise ValueError(f"Unknown joint eval split: {split}")
    return {split: base[split]}


def _idx_from_masks(masks: dict[str, torch.Tensor]) -> torch.Tensor:
    mask_iter = iter(masks.values())
    union = next(mask_iter).bool().clone()
    for mask in mask_iter:
        union |= mask.bool()
    return _idx_from_mask(union)


def _sample_idx_like_gnn_as_judge(idx: torch.Tensor, max_samples: int, seed: int) -> torch.Tensor:
    idx_cpu = idx.long().detach().cpu().tolist()
    if max_samples <= 0 or len(idx_cpu) <= max_samples:
        return idx.long()
    sampled = random.Random(int(seed)).sample(idx_cpu, int(max_samples))
    return torch.tensor(sampled, dtype=torch.long, device=idx.device)


def _sample_mask_like_gnn_as_judge(mask: torch.Tensor, max_samples: int, seed: int) -> torch.Tensor:
    idx = _sample_idx_like_gnn_as_judge(_idx_from_mask(mask), max_samples, seed)
    sampled = torch.zeros_like(mask, dtype=torch.bool)
    sampled[idx] = True
    return sampled


def _gnn_as_judge_sft_test_path(dataset_root: str | Path, dataset: str) -> Path | None:
    names = {
        "arxiv": "arxiv_sft_5_shot_test.json",
        "ogbn-products": "ogbn-products_sft_5_shot_test.json",
    }
    file_name = names.get(dataset.lower())
    if file_name is None:
        return None
    path = Path(dataset_root) / "LLaMA-Factory" / "data" / file_name
    return path if path.exists() else None


def _conversation_human_text(record: dict) -> str:
    for turn in record.get("conversations", []):
        if turn.get("from") == "human":
            return str(turn.get("value", ""))
    return ""


def _match_text(text: str) -> str:
    text = html.unescape(str(text)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _default_recovered_test_node_ids_path(args) -> Path:
    return Path("runs") / "recovered_test_samples" / f"{args.dataset}_{args.shots}shot_seed{args.seed}_test_node_ids.json"


def _ids_from_recovered_test_node_ids(path: str | Path, data, max_samples: int) -> torch.Tensor:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    node_ids = payload.get("node_ids", payload) if isinstance(payload, dict) else payload
    if not isinstance(node_ids, list):
        raise ValueError(f"Recovered eval node ids file must contain a list or a dict with node_ids: {path}")
    if max_samples > 0 and len(node_ids) != int(max_samples):
        raise ValueError(f"{path} contains {len(node_ids)} node ids, expected {max_samples}")
    ids = torch.tensor([int(node_id) for node_id in node_ids], dtype=torch.long, device=data.y.device)
    if ids.numel() == 0:
        raise ValueError(f"Recovered eval node ids file is empty: {path}")
    if int(ids.min().item()) < 0 or int(ids.max().item()) >= int(data.num_nodes):
        raise ValueError(f"Recovered eval node ids out of range for data.num_nodes={data.num_nodes}: {path}")
    if torch.unique(ids.detach().cpu()).numel() != ids.numel():
        raise ValueError(f"Recovered eval node ids contain duplicates: {path}")
    return ids


def _eval_node_ids_for_split_guard(args, data) -> tuple[torch.Tensor | None, str]:
    if args.joint_eval_split != "test" or args.joint_eval_max_samples <= 0:
        return None, ""
    path = Path(args.joint_eval_node_ids_path) if args.joint_eval_node_ids_path else _default_recovered_test_node_ids_path(args)
    if not path.exists():
        return None, str(path)
    return _ids_from_recovered_test_node_ids(path, data, args.joint_eval_max_samples), str(path)


def _remove_eval_ids_from_train_val(data, eval_idx: torch.Tensor | None) -> dict:
    if eval_idx is None or eval_idx.numel() == 0:
        return {"enabled": False}
    train_mask = data.train_mask.bool().clone()
    val_mask = getattr(data, "val_mask", torch.zeros_like(train_mask)).bool().clone()
    test_mask = getattr(data, "test_mask", torch.zeros_like(train_mask)).bool()
    eval_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    eval_mask[eval_idx.long()] = True
    stats = {
        "enabled": True,
        "eval_count": int(eval_mask.sum().item()),
        "removed_train_count": int((train_mask & eval_mask).sum().item()),
        "removed_val_count": int((val_mask & eval_mask).sum().item()),
        "original_eval_test_overlap": int((test_mask & eval_mask).sum().item()),
    }
    if stats["removed_train_count"] or stats["removed_val_count"]:
        data.train_mask = train_mask & ~eval_mask
        data.val_mask = val_mask & ~eval_mask
    stats["final_eval_train_overlap"] = int((data.train_mask.bool() & eval_mask).sum().item())
    stats["final_eval_val_overlap"] = int((data.val_mask.bool() & eval_mask).sum().item())
    return stats


def _ids_from_gnn_as_judge_sft_test(
    data,
    texts: list[str],
    dataset_root: str | Path,
    dataset: str,
    max_samples: int,
) -> torch.Tensor | None:
    path = _gnn_as_judge_sft_test_path(dataset_root, dataset)
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if max_samples > 0 and len(records) != int(max_samples):
        return None

    test_ids = _idx_from_mask(data.test_mask).detach().cpu().long().tolist()
    match_texts = [_match_text(text) for text in texts]
    prefix_lengths = (256, 192, 128, 96, 64, 32, 20)
    prefix_maps: dict[int, dict[str, int | None]] = {}
    for length in prefix_lengths:
        mapping: dict[str, int | None] = {}
        for node_id in test_ids:
            text = match_texts[node_id]
            if len(text) < length:
                continue
            key = text[:length]
            mapping[key] = node_id if key not in mapping else None
        prefix_maps[length] = mapping

    matched: list[int] = []
    for record in records:
        prompt = _match_text(_conversation_human_text(record))
        node_id = None
        for length in prefix_lengths:
            if len(prompt) < length:
                continue
            candidate = prefix_maps[length].get(prompt[:length])
            if candidate is not None:
                node_id = int(candidate)
                break
        if node_id is None:
            candidates = []
            for idx in test_ids:
                text = match_texts[int(idx)]
                prefix = text[: min(128, len(text))]
                if prefix and prompt.startswith(prefix):
                    candidates.append((len(text), int(idx)))
            if not candidates:
                return None
            max_len = max(length for length, _ in candidates)
            best = [idx for length, idx in candidates if length == max_len]
            exact = [idx for idx in best if match_texts[idx] and prompt.startswith(match_texts[idx])]
            node_id = min(exact or best)
        matched.append(node_id)

    if len(set(matched)) != len(matched):
        return None
    return torch.tensor(matched, dtype=torch.long, device=data.y.device)


def _mask_from_idx_like(mask: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    sampled = torch.zeros_like(mask, dtype=torch.bool)
    sampled[idx.long()] = True
    return sampled


def _split_integrity_stats(data, candidate_idx: torch.Tensor, eval_idx: torch.Tensor):
    train_mask = data.train_mask.bool()
    val_mask = getattr(data, "val_mask", torch.zeros_like(train_mask)).bool()
    test_mask = getattr(data, "test_mask", torch.zeros_like(train_mask)).bool()
    candidate_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    candidate_mask[candidate_idx.long()] = True
    eval_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    eval_mask[eval_idx.long()] = True
    return {
        "train_count": int(train_mask.sum().item()),
        "val_count": int(val_mask.sum().item()),
        "test_count": int(test_mask.sum().item()),
        "candidate_count": int(candidate_mask.sum().item()),
        "eval_count": int(eval_mask.sum().item()),
        "candidate_minus_test_count": int((candidate_mask & ~test_mask).sum().item()),
        "test_minus_candidate_count": int((test_mask & ~candidate_mask).sum().item()),
        "eval_train_overlap": int((eval_mask & train_mask).sum().item()),
        "eval_val_overlap": int((eval_mask & val_mask).sum().item()),
        "eval_test_overlap": int((eval_mask & test_mask).sum().item()),
        "train_val_overlap": int((train_mask & val_mask).sum().item()),
        "train_test_overlap": int((train_mask & test_mask).sum().item()),
        "val_test_overlap": int((val_mask & test_mask).sum().item()),
        "eval_is_subset_of_test": bool((eval_mask & ~test_mask).sum().item() == 0),
    }


def run(args):
    set_seed(args.seed)
    device = choose_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    split_dir = ensure_dir(output_dir / "splits")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    save_json(config, output_dir / "config.json")

    data = create_few_shot_dataset(
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        shots=args.shots,
        seed=args.seed,
        device=device,
        split_output_dir=split_dir,
    )
    data.x = data.x.float()
    num_classes, label_names = label_space(data)
    texts = get_texts(data)
    eval_split_guard_idx, eval_split_guard_path = _eval_node_ids_for_split_guard(args, data)
    eval_split_guard_stats = _remove_eval_ids_from_train_val(data, eval_split_guard_idx)
    eval_split_guard_stats["node_ids_path"] = eval_split_guard_path
    task_prompt = DIRECT_PROMPTS.get(args.dataset)
    if task_prompt is None:
        raise ValueError(f"No GNN-as-Judge prompt configured for dataset={args.dataset}")

    gnn_artifact_info = None
    if args.selected_gnn_root:
        encoder, embeddings, ssl_stats, gnn_head, gnn_logits, gnn_artifact_info = _load_selected_gnn_artifacts(
            data=data,
            num_classes=num_classes,
            label_names=label_names,
            args=args,
            device=device,
        )
    else:
        encoder, embeddings, ssl_stats = pretrain_ssl_gnn(
            data=data,
            hidden_dim=args.gnn_hidden_dim,
            embedding_dim=args.embedding_dim,
            n_layers=args.gnn_layers,
            gnn_type=args.gnn_type,
            epochs=args.ssl_epochs,
            lr=args.ssl_lr,
            weight_decay=args.weight_decay,
            feature_mask_rate=args.feature_mask_rate,
            edge_loss_weight=args.edge_loss_weight,
        )
        gnn_head, gnn_logits = _train_gnn_head(data, embeddings, num_classes, args)

    llm_logits, llm_source = _llm_logits(data, num_classes, args, device)

    candidate_idx = _candidate_indices(data, args.pseudo_candidate_mode)
    trust_records = partition_trust_regions(
        gnn_logits=gnn_logits,
        llm_logits=llm_logits,
        candidate_idx=candidate_idx.detach().cpu().tolist(),
        theta_g=args.theta_g,
        theta_l=args.theta_l,
        confidence_metric=args.confidence_metric,
        agreement_tie_delta=args.agreement_tie_delta,
    )
    rule_pseudo_labels, rule_pseudo_idx, rule_label_sources = build_pseudo_label_tensor(data, trust_records)

    codebook = fit_codebook(embeddings, size=args.codebook_size, iters=args.codebook_iters, seed=args.seed)
    token_ids = codebook.encode(embeddings)

    router_artifacts = None
    joint_sample_weights = None
    if args.router_policy == "learnable":
        router_artifacts = build_learnable_router_artifacts(
            data=data,
            texts=texts,
            embeddings=embeddings,
            final_gnn_logits=gnn_logits,
            llm_logits=llm_logits,
            candidate_idx=candidate_idx,
            gnn_artifact_info=gnn_artifact_info,
            num_classes=num_classes,
            args=args,
            output_dir=output_dir,
        )
        pseudo_labels = router_artifacts.labels
        pseudo_idx = router_artifacts.idx
        label_sources = router_artifacts.label_sources
        joint_sample_weights = router_artifacts.sample_weights
        joint_labels, joint_train_idx, val_idx, joint_supervision_stats = _joint_supervision(data, pseudo_labels, pseudo_idx, args)
        joint_train_selection_stats = dict(router_artifacts.stats)
        joint_train_selection_stats["rule_router_baseline"] = {
            "pseudo_label_count": int(rule_pseudo_idx.numel()),
            "label_sources": {source: sum(1 for value in rule_label_sources.values() if value == source) for source in sorted(set(rule_label_sources.values()))},
        }
    else:
        pseudo_labels = rule_pseudo_labels
        pseudo_idx = rule_pseudo_idx
        label_sources = rule_label_sources
        joint_labels, joint_train_idx, val_idx, joint_supervision_stats = _joint_supervision(data, pseudo_labels, pseudo_idx, args)
        joint_train_idx, joint_train_selection_stats = _select_joint_train_idx(
            joint_train_idx,
            joint_labels,
            label_sources,
            trust_records,
            args,
            num_classes,
        )
    joint_supervision_stats["train_size"] = int(joint_train_idx.numel())
    if bool(args.router_only):
        masks = masks_dict(data)
        pseudo_only_count = sum(1 for source in label_sources.values() if source != "true")
        metrics = {
            "dataset": args.dataset,
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.edge_index.size(1)),
            "num_classes": int(num_classes),
            "shots": int(args.shots),
            "seed": int(args.seed),
            "mode": "router_only",
            "llm_source": llm_source,
            "ssl": asdict(ssl_stats),
            "gnn_artifact": gnn_artifact_info,
            "candidate_mode": args.pseudo_candidate_mode,
            "trust_regions": summarize_regions(trust_records),
            "trust_region_quality": _trust_region_quality(trust_records, data.y, num_classes),
            "pseudo_label_count": int(pseudo_idx.numel()),
            "pseudo_label_quality": _pseudo_label_quality(pseudo_labels, pseudo_idx, label_sources, data.y, num_classes),
            "selected_pseudo_quality": _selected_pseudo_quality(joint_train_idx, pseudo_labels, label_sources, data.y, num_classes),
            "joint_supervision": joint_supervision_stats,
            "joint_train_selection": joint_train_selection_stats,
            "router": None if router_artifacts is None else router_artifacts.stats,
            "pseudo_label_count_excluding_true": int(pseudo_only_count),
            "pseudo_label_coverage_over_candidates": float(pseudo_only_count / max(candidate_idx.numel(), 1)),
            "gnn_head": split_metrics(gnn_logits, data.y, masks, num_classes),
            "llm_output_token": split_metrics(llm_logits, data.y, masks, num_classes),
        }

        trust_rows = []
        for rec in trust_records:
            row = asdict(rec)
            if row["pseudo_label"] is not None:
                row["pseudo_label_name"] = label_names[row["pseudo_label"]]
            trust_rows.append(row)

        struct_rows = []
        for node_id in range(data.num_nodes):
            token_id = int(token_ids[node_id].item())
            struct_rows.append({"node_id": int(node_id), "struct_token_id": token_id, "struct_codebook_id": token_id})

        pseudo_rows = []
        for node_id in pseudo_idx.detach().cpu().tolist():
            label = int(pseudo_labels[node_id].item())
            pseudo_rows.append(
                {
                    "node_id": int(node_id),
                    "label": label,
                    "label_name": label_names[label],
                    "source": label_sources.get(int(node_id), "unknown"),
                    "is_correct": bool(label == int(data.y[node_id].item())),
                    "gold_label": int(data.y[node_id].item()),
                    "gold_label_name": label_names[int(data.y[node_id].item())],
                    "struct_token_id": int(token_ids[node_id].item()),
                }
            )

        write_jsonl(trust_rows, output_dir / "trust_regions.jsonl")
        write_jsonl(struct_rows, output_dir / "node_struct_tokens.jsonl")
        write_jsonl(pseudo_rows, output_dir / "pseudo_labels.jsonl")
        save_json(metrics, output_dir / "metrics.json")
        torch.save(
            {
                "gnn_logits": gnn_logits.detach().cpu(),
                "llm_logits": llm_logits.detach().cpu(),
                "token_ids": token_ids.detach().cpu(),
                "pseudo_labels": pseudo_labels.detach().cpu(),
                "joint_train_idx": joint_train_idx.detach().cpu(),
                "joint_sample_weights": None if joint_sample_weights is None else joint_sample_weights.detach().cpu(),
                "router_policy": args.router_policy,
                "joint_val_idx": None if val_idx is None else val_idx.detach().cpu(),
                "label_names": label_names,
                "mode": "router_only",
            },
            output_dir / "predictions.pt",
        )
        torch.save(codebook.state_dict(), output_dir / "codebook.pt")
        torch.save(encoder.state_dict(), output_dir / "ssl_gnn_encoder.pt")
        torch.save(gnn_head.state_dict(), output_dir / "gnn_head.pt")
        return metrics

    requested_joint_eval_masks = _joint_eval_masks(data, args.joint_eval_split)
    joint_eval_sample_source = "full_split"
    if args.joint_eval_max_samples > 0 and args.joint_eval_split != "all":
        sampled_masks = {}
        for name, mask in requested_joint_eval_masks.items():
            exact_idx = None
            recovered_path = None
            if name == "test":
                recovered_path = Path(args.joint_eval_node_ids_path) if args.joint_eval_node_ids_path else _default_recovered_test_node_ids_path(args)
                if eval_split_guard_idx is not None:
                    exact_idx = eval_split_guard_idx
                elif recovered_path.exists():
                    exact_idx = _ids_from_recovered_test_node_ids(recovered_path, data, args.joint_eval_max_samples)
                else:
                    exact_idx = _ids_from_gnn_as_judge_sft_test(
                        data=data,
                        texts=texts,
                        dataset_root=args.dataset_root,
                        dataset=args.dataset,
                        max_samples=args.joint_eval_max_samples,
                    )
            if exact_idx is not None:
                sampled_masks[name] = _mask_from_idx_like(mask.bool(), exact_idx)
                joint_eval_sample_source = "recovered_test_node_ids" if recovered_path and recovered_path.exists() else "gnn_as_judge_sft_test_json"
            else:
                sampled_masks[name] = _sample_mask_like_gnn_as_judge(mask.bool(), args.joint_eval_max_samples, args.seed)
                joint_eval_sample_source = "random_sample_seed"
        requested_joint_eval_masks = sampled_masks
    requested_joint_eval_idx = _idx_from_masks(requested_joint_eval_masks)
    val_eval_mask = data.val_mask.bool()
    val_eval_idx = val_idx
    if val_idx is not None and args.joint_val_max_samples > 0:
        val_eval_mask = _sample_mask_like_gnn_as_judge(data.val_mask.bool(), args.joint_val_max_samples, args.seed)
        val_eval_idx = _idx_from_mask(val_eval_mask)
    joint_supervision_stats["val_split_size"] = int(0 if val_idx is None else val_idx.numel())
    joint_supervision_stats["val_size"] = int(0 if val_eval_idx is None else val_eval_idx.numel())
    joint_supervision_stats["val_eval_size"] = int(0 if val_eval_idx is None else val_eval_idx.numel())
    joint_supervision_stats["val_eval_sample_seed"] = int(args.seed)
    joint_supervision_stats["val_eval_sample_source"] = (
        "gnn_as_judge_random_sample"
        if val_idx is not None and args.joint_val_max_samples > 0 and val_idx.numel() > args.joint_val_max_samples
        else "full_val_split"
    )

    joint_model = JointGraphTextClassifier(
        model_path=args.llm_model_path,
        codebook_size=codebook.size,
        label_names=label_names,
        task_prompt=task_prompt,
        token_dim=args.token_dim,
        dropout=args.dropout,
        max_input_length=args.joint_max_input_length,
        score_normalization=args.joint_score_normalization,
        dtype=args.joint_llm_dtype,
        device_map=args.joint_llm_device_map,
        freeze_llm=True,
        label_pair_batch_size=args.joint_label_pair_batch_size,
        structural_init=args.joint_structural_init,
        scoring_mode=args.joint_scoring_mode,
        gnn_embedding_dim=embeddings.size(1),
        residual_gate_init=args.joint_residual_gate_init,
        gradient_checkpointing=args.joint_gradient_checkpointing,
    )

    joint_val_gate = {"enabled": bool(args.joint_val_gate), "passed": None}
    if args.joint_val_gate and val_eval_idx is not None and val_eval_idx.numel() > 0:
        joint_eval_masks = {"val": val_eval_mask}
        joint_eval_idx = val_eval_idx
        joint_logits = train_joint_llm_model(
            joint_model,
            token_ids,
            texts,
            joint_labels,
            joint_train_idx,
            val_eval_idx,
            joint_eval_idx,
            epochs=args.joint_epochs,
            lr=args.joint_lr,
            weight_decay=args.weight_decay,
            batch_size=args.joint_batch_size,
            eval_batch_size=args.joint_eval_batch_size,
            val_every=args.joint_val_every,
            gnn_embeddings=embeddings,
            gnn_teacher_logits=gnn_logits,
            llm_teacher_logits=llm_logits,
            ce_weight=args.joint_ce_weight,
            distill_weight=args.joint_distill_weight,
            distill_temperature=args.joint_distill_temperature,
            memory_efficient_backward=args.joint_memory_efficient_backward,
            joint_sample_weights=joint_sample_weights,
        )
        joint_val_accuracy = _split_accuracy_from_logits(joint_logits, data.y, val_eval_idx)
        baseline_accuracy, baseline_info = _val_gate_baseline(args, gnn_logits, llm_logits, data.y, val_eval_idx)
        passed = bool(joint_val_accuracy >= baseline_accuracy + args.joint_val_gate_min_delta)
        joint_val_gate.update(
            {
                "passed": passed,
                "joint_val_accuracy": float(joint_val_accuracy),
                "baseline_accuracy": float(baseline_accuracy),
                "baseline": baseline_info,
                "min_delta": float(args.joint_val_gate_min_delta),
                "requested_eval_split": args.joint_eval_split,
            }
        )
        if passed and args.joint_eval_split != "val":
            torch.save(
                {
                    "token_embedding": joint_model.token_embedding.state_dict(),
                    "token_projection": joint_model.token_projection.state_dict(),
                    "residual_norm": None if getattr(joint_model, "residual_norm", None) is None else joint_model.residual_norm.state_dict(),
                    "residual_projector": None if getattr(joint_model, "residual_projector", None) is None else joint_model.residual_projector.state_dict(),
                    "residual_gate": None if getattr(joint_model, "residual_gate", None) is None else joint_model.residual_gate.detach().cpu(),
                    "label_names": label_names,
                    "llm_model_path": args.llm_model_path,
                    "joint_model": "frozen_llm_with_llm_manifold_code_token_and_gated_gnn_residual",
                    "joint_structural_init": args.joint_structural_init,
                    "joint_residual_gate_init": float(args.joint_residual_gate_init),
                    "joint_scoring_mode": args.joint_scoring_mode,
                    "gnn_embedding_dim": int(embeddings.size(1)),
                    "stage": "after_val_gate_before_requested_eval",
                },
                output_dir / "joint_model_after_val.pt",
            )
            eval_batch_size = 1 if args.joint_eval_batch_size <= 0 else int(args.joint_eval_batch_size)
            with torch.no_grad():
                requested_logits = joint_llm_logits_for_indices(
                    joint_model, token_ids, texts, requested_joint_eval_idx, eval_batch_size, gnn_embeddings=embeddings
                )
            joint_logits[requested_joint_eval_idx.to(joint_logits.device)] = requested_logits.to(joint_logits.device)
            joint_eval_masks = requested_joint_eval_masks
            joint_eval_idx = requested_joint_eval_idx
        elif not passed:
            joint_val_gate["test_skipped"] = True
    else:
        joint_eval_masks = requested_joint_eval_masks
        joint_eval_idx = requested_joint_eval_idx
        joint_logits = train_joint_llm_model(
            joint_model,
            token_ids,
            texts,
            joint_labels,
            joint_train_idx,
            val_eval_idx,
            joint_eval_idx,
            epochs=args.joint_epochs,
            lr=args.joint_lr,
            weight_decay=args.weight_decay,
            batch_size=args.joint_batch_size,
            eval_batch_size=args.joint_eval_batch_size,
            val_every=args.joint_val_every,
            gnn_embeddings=embeddings,
            gnn_teacher_logits=gnn_logits,
            llm_teacher_logits=llm_logits,
            ce_weight=args.joint_ce_weight,
            distill_weight=args.joint_distill_weight,
            distill_temperature=args.joint_distill_temperature,
            memory_efficient_backward=args.joint_memory_efficient_backward,
            joint_sample_weights=joint_sample_weights,
        )

    masks = masks_dict(data)
    pseudo_only_count = sum(1 for source in label_sources.values() if source != "true")
    metrics = {
        "dataset": args.dataset,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "num_classes": int(num_classes),
        "shots": int(args.shots),
        "seed": int(args.seed),
        "llm_source": llm_source,
        "joint_model": "frozen_llm_with_llm_manifold_code_token_and_gated_gnn_residual",
        "joint_structural_init": args.joint_structural_init,
        "joint_residual_gate_init": float(args.joint_residual_gate_init),
        "joint_scoring_mode": args.joint_scoring_mode,
        "joint_loss": {
            "ce_weight": float(args.joint_ce_weight),
            "distill_weight": float(args.joint_distill_weight),
            "distill_temperature": float(args.joint_distill_temperature),
            "teachers": ["gnn_head", "llm_output_token"],
            "memory_efficient_backward": bool(args.joint_memory_efficient_backward),
        },
        "joint_prompt": "one graph soft token after an English structural-context prefix + raw node text + dataset-specific GNN-as-Judge prompt",
        "text_input": "raw_text_in_llm_prompt",
        "ssl": asdict(ssl_stats),
        "gnn_artifact": gnn_artifact_info,
        "candidate_mode": args.pseudo_candidate_mode,
        "trust_regions": summarize_regions(trust_records),
        "trust_region_quality": _trust_region_quality(trust_records, data.y, num_classes),
        "pseudo_label_count": int(pseudo_idx.numel()),
        "pseudo_label_quality": _pseudo_label_quality(pseudo_labels, pseudo_idx, label_sources, data.y, num_classes),
        "selected_pseudo_quality": _selected_pseudo_quality(joint_train_idx, pseudo_labels, label_sources, data.y, num_classes),
        "joint_supervision": joint_supervision_stats,
        "joint_train_selection": joint_train_selection_stats,
        "router": None if router_artifacts is None else router_artifacts.stats,
        "joint_val_gate": joint_val_gate,
        "joint_requested_eval_split": args.joint_eval_split,
        "joint_eval_split": "+".join(joint_eval_masks.keys()),
        "joint_eval_count": int(joint_eval_idx.numel()),
        "joint_eval_max_samples": int(args.joint_eval_max_samples),
        "joint_val_max_samples": int(args.joint_val_max_samples),
        "joint_eval_sample_seed": int(args.seed),
        "joint_eval_sample_source": joint_eval_sample_source,
        "joint_eval_node_ids_path": str(args.joint_eval_node_ids_path or _default_recovered_test_node_ids_path(args)),
        "eval_split_guard": eval_split_guard_stats,
        "split_integrity": _split_integrity_stats(data, candidate_idx, joint_eval_idx),
        "codebook_coverage": _codebook_coverage(token_ids, joint_train_idx, joint_eval_idx, codebook.size, val_idx),
        "pseudo_label_count_excluding_true": int(pseudo_only_count),
        "pseudo_label_coverage_over_candidates": float(pseudo_only_count / max(candidate_idx.numel(), 1)),
        "gnn_head": split_metrics(gnn_logits, data.y, masks, num_classes),
        "llm_output_token": split_metrics(llm_logits, data.y, masks, num_classes),
        "joint": split_metrics(joint_logits, data.y, joint_eval_masks, num_classes),
    }

    trust_rows = []
    for rec in trust_records:
        row = asdict(rec)
        if row["pseudo_label"] is not None:
            row["pseudo_label_name"] = label_names[row["pseudo_label"]]
        trust_rows.append(row)

    struct_rows = []
    for node_id in range(data.num_nodes):
        token_id = int(token_ids[node_id].item())
        struct_rows.append({"node_id": int(node_id), "struct_token_id": token_id, "struct_codebook_id": token_id})

    pseudo_rows = []
    for node_id in pseudo_idx.detach().cpu().tolist():
        label = int(pseudo_labels[node_id].item())
        pseudo_rows.append(
            {
                "node_id": int(node_id),
                "label": label,
                "label_name": label_names[label],
                "source": label_sources.get(int(node_id), "unknown"),
                "is_correct": bool(label == int(data.y[node_id].item())),
                "gold_label": int(data.y[node_id].item()),
                "gold_label_name": label_names[int(data.y[node_id].item())],
                "struct_token_id": int(token_ids[node_id].item()),
            }
        )

    write_jsonl(trust_rows, output_dir / "trust_regions.jsonl")
    write_jsonl(struct_rows, output_dir / "node_struct_tokens.jsonl")
    write_jsonl(pseudo_rows, output_dir / "pseudo_labels.jsonl")
    save_json(metrics, output_dir / "metrics.json")

    torch.save(
        {
            "joint_logits": joint_logits.detach().cpu(),
            "gnn_logits": gnn_logits.detach().cpu(),
            "llm_logits": llm_logits.detach().cpu(),
            "token_ids": token_ids.detach().cpu(),
            "pseudo_labels": pseudo_labels.detach().cpu(),
            "joint_train_idx": joint_train_idx.detach().cpu(),
            "joint_sample_weights": None if joint_sample_weights is None else joint_sample_weights.detach().cpu(),
            "router_policy": args.router_policy,
            "joint_val_idx": None if val_idx is None else val_idx.detach().cpu(),
            "joint_eval_idx": joint_eval_idx.detach().cpu(),
            "label_names": label_names,
        },
        output_dir / "predictions.pt",
    )
    torch.save(codebook.state_dict(), output_dir / "codebook.pt")
    torch.save(encoder.state_dict(), output_dir / "ssl_gnn_encoder.pt")
    torch.save(gnn_head.state_dict(), output_dir / "gnn_head.pt")
    torch.save(
        {
            "token_embedding": joint_model.token_embedding.state_dict(),
            "token_projection": joint_model.token_projection.state_dict(),
            "residual_norm": None if getattr(joint_model, "residual_norm", None) is None else joint_model.residual_norm.state_dict(),
            "residual_projector": None if getattr(joint_model, "residual_projector", None) is None else joint_model.residual_projector.state_dict(),
            "residual_gate": None if getattr(joint_model, "residual_gate", None) is None else joint_model.residual_gate.detach().cpu(),
            "label_names": label_names,
            "llm_model_path": args.llm_model_path,
            "joint_model": "frozen_llm_with_llm_manifold_code_token_and_gated_gnn_residual",
            "joint_structural_init": args.joint_structural_init,
            "joint_residual_gate_init": float(args.joint_residual_gate_init),
            "joint_scoring_mode": args.joint_scoring_mode,
            "gnn_embedding_dim": int(embeddings.size(1)),
        },
        output_dir / "joint_model.pt",
    )
    return metrics


def build_parser():
    parser = argparse.ArgumentParser(description="Run EAR-GTC on a text-attributed graph.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, default=".")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/default"))
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--gnn-type", type=str, default="GCN", choices=["GCN", "SAGE", "SGConv"])
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--gnn-hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--selected-gnn-root", type=str, default="", help="Optional selected GNN artifact root; when set, load frozen selected GNN embeddings/logits instead of retraining SSL GNN.")
    parser.add_argument("--selected-gnn-manifest", type=str, default="", help="Optional selected GNN manifest path. Defaults to <selected-gnn-root>/manifest.json.")
    parser.add_argument("--ssl-epochs", type=int, default=100)
    parser.add_argument("--ssl-lr", type=float, default=1e-3)
    parser.add_argument("--feature-mask-rate", type=float, default=0.2)
    parser.add_argument("--edge-loss-weight", type=float, default=0.2)

    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--head-epochs", type=int, default=200)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--llm-model-path", type=str, default="/data1/wujingyao-20354/wujingyao/codes/models/llama-3-8b-instruct")
    parser.add_argument("--llm-logits-path", type=str, required=True)

    parser.add_argument("--confidence-metric", type=str, default="entropy", choices=["entropy", "max_prob", "margin"])
    parser.add_argument("--theta-g", type=float, default=0.45)
    parser.add_argument("--theta-l", type=float, default=0.45)
    parser.add_argument("--agreement-tie-delta", type=float, default=0.05)
    parser.add_argument("--pseudo-candidate-mode", type=str, default="unlabeled", choices=["no-val", "unlabeled", "all"])

    parser.add_argument("--codebook-size", type=int, default=128)
    parser.add_argument("--codebook-iters", type=int, default=25)
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument("--joint-epochs", type=int, default=3)
    parser.add_argument("--joint-lr", type=float, default=1e-3)
    parser.add_argument("--joint-batch-size", type=int, default=0, help="0 selects an automatic node batch size from num_classes.")
    parser.add_argument("--joint-eval-batch-size", type=int, default=0, help="0 selects an automatic eval node batch size from num_classes.")
    parser.add_argument("--joint-label-pair-batch-size", type=int, default=64)
    parser.add_argument("--joint-val-every", type=int, default=1)
    parser.add_argument("--joint-eval-split", type=str, default="test", choices=["test", "val", "train", "all"])
    parser.add_argument("--joint-eval-max-samples", type=int, default=0, help="If >0, evaluate this many nodes from the requested eval split.")
    parser.add_argument("--joint-eval-node-ids-path", type=str, default="", help="Optional JSON file containing recovered exact eval node_ids. If omitted, runs/recovered_test_samples/{dataset}_{shots}shot_seed{seed}_test_node_ids.json is used when present.")
    parser.add_argument("--joint-val-max-samples", type=int, default=0, help="If >0, sample this many validation nodes for joint val-gate/training validation.")
    parser.add_argument("--joint-val-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--joint-val-gate-baseline", type=str, default="best_single", choices=["best_single", "gnn", "llm"])
    parser.add_argument("--joint-val-gate-min-delta", type=float, default=0.0)
    parser.add_argument("--joint-selection-policy", type=str, default="source_threshold", choices=["source_threshold", "source_quota", "confidence"])
    parser.add_argument("--joint-allowed-sources", type=str, default="consensus,llm,gnn")
    parser.add_argument("--joint-source-min-confidence", type=str, default="consensus=0.45,llm=0.75,gnn=0.80")
    parser.add_argument("--joint-source-priority", type=str, default="consensus,llm,gnn")
    parser.add_argument("--joint-source-quotas", type=str, default="", help="Source quota weights for --joint-selection-policy source_quota, e.g. consensus=0.7,llm=0.2,gnn=0.1.")
    parser.add_argument("--joint-pseudo-min-confidence", type=float, default=0.5)
    parser.add_argument("--joint-max-pseudo-train", type=int, default=2048, help="Maximum pseudo-labeled nodes used for joint training; <=0 keeps all after filtering.")
    parser.add_argument("--joint-pseudo-per-class-cap", type=int, default=256, help="Per-class cap for pseudo-labeled joint training nodes; <=0 disables it.")
    parser.add_argument("--joint-max-input-length", type=int, default=2048)
    parser.add_argument("--joint-score-normalization", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument("--joint-ce-weight", type=float, default=1.0)
    parser.add_argument("--joint-distill-weight", type=float, default=0.2)
    parser.add_argument("--joint-distill-temperature", type=float, default=2.0)
    parser.add_argument("--joint-structural-init", type=str, default="llm_manifold", choices=["zero", "small_random", "random", "llm_manifold"])
    parser.add_argument("--joint-residual-gate-init", type=float, default=0.0)
    parser.add_argument("--joint-scoring-mode", type=str, default="full", choices=["full", "prefix_cache"])
    parser.add_argument("--joint-llm-dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--joint-llm-device-map", type=str, default="auto")
    parser.add_argument("--joint-gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--joint-memory-efficient-backward", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--router-policy", type=str, default="rule", choices=["rule", "learnable"], help="Pseudo-label router policy. 'rule' preserves the source-threshold baseline; 'learnable' trains the two-head reliability router.")
    parser.add_argument("--router-only", action=argparse.BooleanOptionalAction, default=False, help="Stop after router training, validation diagnostics, and pseudo-label export; skip joint model training/evaluation.")
    parser.add_argument("--router-model", type=str, default="two_head_mlp", choices=["two_head_mlp"])
    parser.add_argument("--router-feature-set", type=str, default="v1", choices=["v1"])
    parser.add_argument("--router-output-dir", type=str, default="", help="Optional router artifact directory. Defaults to <output-dir>/router.")
    parser.add_argument("--router-logit-calibration", action=argparse.BooleanOptionalAction, default=True, help="Apply temperature scaling to GNN/LLM logits before router feature extraction. Disable for Router V5 reliability-only experiments.")
    parser.add_argument("--router-reliability-only", action=argparse.BooleanOptionalAction, default=False, help="Train and export only r_g/r_l reliability scores without H/S/A pseudo-label selection.")
    parser.add_argument("--router-num-folds", type=int, default=5)
    parser.add_argument("--router-oof-head-epochs", type=int, default=200)
    parser.add_argument("--router-hidden-dim", type=int, default=64)
    parser.add_argument("--router-dropout", type=float, default=0.2)
    parser.add_argument("--router-epochs", type=int, default=300)
    parser.add_argument("--router-lr", type=float, default=1e-3)
    parser.add_argument("--router-target-precision", type=float, default=0.90)
    parser.add_argument("--router-hard-cap", type=int, default=512)
    parser.add_argument("--router-per-class-cap", type=int, default=128)
    parser.add_argument("--router-soft-min-reliability", type=float, default=0.55)
    parser.add_argument("--router-hard-min-reliability", type=float, default=0.75)
    parser.add_argument("--router-reliability-margin", type=float, default=0.15)
    parser.add_argument("--router-optimize-thresholds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--router-target-precision-grid", type=str, default="0.85,0.90,0.95")
    parser.add_argument("--router-tau-soft-grid", type=str, default="0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--router-tau-hard-grid", type=str, default="0.65,0.70,0.75,0.80,0.85,0.90")
    parser.add_argument("--router-margin-grid", type=str, default="0.00,0.05,0.10,0.15,0.20")
    parser.add_argument("--router-wilson-z", type=float, default=1.96)
    parser.add_argument("--router-conflict-to-abstain", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--router-max-text-tokens", type=int, default=0, help="Deprecated compatibility option; router text_length is no longer truncated.")

    parser.add_argument("--exclude-val-from-joint-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser


def main():
    args = build_parser().parse_args()
    metrics = run(args)
    print(metrics)


if __name__ == "__main__":
    main()
