#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eargtc.data import create_few_shot_dataset, label_space, masks_dict
from eargtc.metrics import accuracy, macro_f1, split_metrics, weighted_f1
from eargtc.models.gnn import GNNEncoder, MLPHead
from eargtc.models.ssl import pretrain_ssl_gnn
from eargtc.utils import choose_device, ensure_dir, save_json, set_seed


@dataclass(frozen=True)
class HeadTrial:
    name: str
    head_hidden_dim: int = 128
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 400
    patience: int = 100
    select_by: str = "val_loss"
    class_weight: str = "none"


DEFAULT_HEAD_TRIALS = [
    HeadTrial("baseline_like", 128, 0.3, 1e-3, 1e-4, 200, 0, "val_loss", "none"),
    HeadTrial("longer_loss", 128, 0.3, 1e-3, 1e-4, 800, 150, "val_loss", "none"),
    HeadTrial("wider_loss", 256, 0.3, 1e-3, 1e-4, 800, 150, "val_loss", "none"),
    HeadTrial("wider_dropout_loss", 256, 0.5, 1e-3, 1e-4, 800, 150, "val_loss", "none"),
    HeadTrial("fast_wider_loss", 256, 0.3, 3e-3, 1e-4, 800, 150, "val_loss", "none"),
    HeadTrial("regularized_loss", 128, 0.5, 3e-3, 1e-3, 800, 150, "val_loss", "none"),
    HeadTrial("val_acc", 128, 0.3, 1e-3, 1e-4, 800, 150, "val_acc", "none"),
    HeadTrial("wider_val_acc", 256, 0.3, 1e-3, 1e-4, 800, 150, "val_acc", "none"),
    HeadTrial("balanced_loss", 128, 0.3, 1e-3, 1e-4, 800, 150, "val_loss", "balanced"),
    HeadTrial("balanced_val_acc", 128, 0.3, 1e-3, 1e-4, 800, 150, "val_acc", "balanced"),
]


def _idx(mask: torch.Tensor) -> torch.Tensor:
    return mask.bool().nonzero().view(-1)


def _read_node_ids(path: Path, device: torch.device) -> torch.Tensor:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    node_ids = payload.get("node_ids", payload) if isinstance(payload, dict) else payload
    if not isinstance(node_ids, list):
        raise ValueError(f"Expected a list of node ids or a dict with node_ids: {path}")
    ids = torch.tensor([int(node_id) for node_id in node_ids], dtype=torch.long, device=device)
    if ids.numel() == 0:
        raise ValueError(f"Empty node id file: {path}")
    if torch.unique(ids.detach().cpu()).numel() != ids.numel():
        raise ValueError(f"Duplicate node ids in {path}")
    return ids


def _score_idx(logits: torch.Tensor, y: torch.Tensor, idx: torch.Tensor, num_classes: int) -> dict:
    idx = idx.long()
    pred = logits[idx].argmax(dim=-1)
    gold = y[idx].view(-1).long()
    return {
        "count": int(idx.numel()),
        "accuracy": accuracy(pred, gold),
        "macro_f1": macro_f1(pred, gold, num_classes),
        "weighted_f1": weighted_f1(pred, gold, num_classes),
    }


def _class_weights(y: torch.Tensor, train_idx: torch.Tensor, num_classes: int, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode != "balanced":
        raise ValueError(f"Unknown class_weight mode: {mode}")
    counts = torch.bincount(y[train_idx].view(-1).long().detach().cpu(), minlength=num_classes).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    weights[counts == 0] = 0.0
    weights = weights / weights[weights > 0].mean().clamp_min(1e-12)
    return weights.to(y.device)


def _is_better(select_by: str, val_loss: float, val_acc: float, best_val_loss: float, best_val_acc: float) -> bool:
    if select_by == "val_loss":
        return val_loss < best_val_loss
    if select_by == "val_acc":
        return val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss)
    raise ValueError(f"Unknown select_by: {select_by}")


def train_head(
    embeddings: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    num_classes: int,
    trial: HeadTrial,
) -> tuple[torch.Tensor, dict, dict]:
    model = MLPHead(
        input_dim=embeddings.size(1),
        hidden_dim=trial.head_hidden_dim,
        output_dim=num_classes,
        dropout=trial.dropout,
    ).to(embeddings.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=trial.lr, weight_decay=trial.weight_decay)
    weights = _class_weights(y, train_idx, num_classes, trial.class_weight)
    best_state = deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_epoch = 0
    wait = 0
    history = []

    for epoch in range(1, max(trial.epochs, 1) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(embeddings)
        train_loss = F.cross_entropy(logits[train_idx], y[train_idx], weight=weights)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(embeddings)
            val_loss = F.cross_entropy(logits[val_idx], y[val_idx]).item() if val_idx.numel() else 0.0
            val_acc = accuracy(logits[val_idx].argmax(dim=-1), y[val_idx]) if val_idx.numel() else 0.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss.detach().cpu().item()),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc),
            }
        )
        if _is_better(trial.select_by, val_loss, val_acc, best_val_loss, best_val_acc):
            best_state = deepcopy(model.state_dict())
            best_val_loss = float(val_loss)
            best_val_acc = float(val_acc)
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        if trial.patience > 0 and wait >= trial.patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(embeddings).detach()
    stats = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_accuracy": float(best_val_acc),
        "ran_epochs": int(len(history)),
    }
    return logits, stats, {"history": history, "state_dict": deepcopy(model.state_dict())}


def _load_saved_encoder(data, path: Path, hidden_dim: int, embedding_dim: int, layers: int, device: torch.device):
    encoder = GNNEncoder(
        input_dim=data.x.size(1),
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        n_layers=layers,
        gnn_type="GCN",
        dropout=0.3,
    ).to(device)
    state = torch.load(path, map_location=device, weights_only=False)
    encoder.load_state_dict(state)
    encoder.eval()
    with torch.no_grad():
        embeddings = encoder(data.x, data.edge_index).detach()
    return encoder, embeddings, {"source": str(path), "trained_in_this_run": False}


def _train_encoder(data, args):
    encoder, embeddings, ssl_stats = pretrain_ssl_gnn(
        data=data,
        hidden_dim=args.gnn_hidden_dim,
        embedding_dim=args.embedding_dim,
        n_layers=args.gnn_layers,
        gnn_type="GCN",
        epochs=args.ssl_epochs,
        lr=args.ssl_lr,
        weight_decay=args.ssl_weight_decay,
        feature_mask_rate=args.feature_mask_rate,
        edge_loss_weight=args.edge_loss_weight,
    )
    return encoder, embeddings, {"trained_in_this_run": True, "ssl": asdict(ssl_stats)}


def _load_trials(path: Path | None) -> list[HeadTrial]:
    if path is None:
        return DEFAULT_HEAD_TRIALS
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return [HeadTrial(**row) for row in rows]


def build_parser():
    parser = argparse.ArgumentParser(description="Isolated GCN SSL + classifier tuning on recovered 1000 eval nodes.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--node-ids-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--saved-encoder-path", type=Path, default=None)
    parser.add_argument("--trials-json", type=Path, default=None)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--gnn-hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--ssl-epochs", type=int, default=100)
    parser.add_argument("--ssl-lr", type=float, default=1e-3)
    parser.add_argument("--ssl-weight-decay", type=float, default=1e-4)
    parser.add_argument("--feature-mask-rate", type=float, default=0.2)
    parser.add_argument("--edge-loss-weight", type=float, default=0.2)
    return parser


def main():
    args = build_parser().parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    device = choose_device(args.device)
    out_dir = ensure_dir(args.output_dir)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    save_json(config, out_dir / "config.json")

    data = create_few_shot_dataset(args.dataset, args.dataset_root, args.shots, args.seed, device)
    data.x = data.x.float()
    y = data.y.view(-1).long()
    num_classes, label_names = label_space(data)
    eval_idx = _read_node_ids(args.node_ids_path, device)
    eval_mask = torch.zeros(int(data.num_nodes), dtype=torch.bool, device=device)
    eval_mask[eval_idx] = True
    split_integrity = {
        "eval_count": int(eval_idx.numel()),
        "eval_train_overlap": int((eval_mask & data.train_mask.bool()).sum().item()),
        "eval_val_overlap": int((eval_mask & data.val_mask.bool()).sum().item()),
        "eval_test_overlap": int((eval_mask & data.test_mask.bool()).sum().item()),
        "eval_is_subset_of_test": bool((eval_mask & ~data.test_mask.bool()).sum().item() == 0),
    }
    if split_integrity["eval_train_overlap"] or split_integrity["eval_val_overlap"]:
        raise ValueError(f"Recovered eval ids overlap train/val: {split_integrity}")

    if args.saved_encoder_path is not None:
        encoder, embeddings, encoder_info = _load_saved_encoder(
            data,
            args.saved_encoder_path,
            args.gnn_hidden_dim,
            args.embedding_dim,
            args.gnn_layers,
            device,
        )
    else:
        encoder, embeddings, encoder_info = _train_encoder(data, args)

    train_idx = _idx(data.train_mask)
    val_idx = _idx(data.val_mask)
    trials = _load_trials(args.trials_json)
    trial_summaries = []
    best_by_val = None
    oracle_diagnostic_best_by_eval = None
    for trial_no, trial in enumerate(trials, start=1):
        set_seed(args.seed + trial_no)
        print(f"[{args.dataset}] trial {trial_no}/{len(trials)} {trial.name}", flush=True)
        logits, train_stats, artifact = train_head(embeddings, y, train_idx, val_idx, num_classes, trial)
        metrics = {
            "trial": asdict(trial),
            "train": _score_idx(logits, y, train_idx, num_classes),
            "val": _score_idx(logits, y, val_idx, num_classes),
            "test_full": split_metrics(logits, y, {"test": data.test_mask.bool()}, num_classes)["test"],
            "recovered_eval": _score_idx(logits, y, eval_idx, num_classes),
            "train_stats": train_stats,
        }
        trial_dir = ensure_dir(out_dir / f"{trial_no:03d}_{trial.name}")
        save_json(metrics, trial_dir / "metrics.json")
        torch.save(
            {
                "head_state": artifact["state_dict"],
                "logits": logits.detach().cpu(),
                "eval_idx": eval_idx.detach().cpu(),
                "label_names": label_names,
            },
            trial_dir / "predictions.pt",
        )
        save_json(artifact["history"], trial_dir / "history.json")
        trial_summaries.append(metrics)
        if best_by_val is None or (
            metrics["val"]["accuracy"] > best_by_val["val"]["accuracy"]
            or (
                metrics["val"]["accuracy"] == best_by_val["val"]["accuracy"]
                and metrics["train_stats"]["best_val_loss"] < best_by_val["train_stats"]["best_val_loss"]
            )
        ):
            best_by_val = metrics
        if (
            oracle_diagnostic_best_by_eval is None
            or metrics["recovered_eval"]["accuracy"] > oracle_diagnostic_best_by_eval["recovered_eval"]["accuracy"]
        ):
            oracle_diagnostic_best_by_eval = metrics
        print(
            json.dumps(
                {
                    "trial": trial.name,
                    "val_acc": metrics["val"]["accuracy"],
                    "eval_acc": metrics["recovered_eval"]["accuracy"],
                    "full_test_acc": metrics["test_full"]["accuracy"],
                    "best_epoch": metrics["train_stats"]["best_epoch"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = {
        "dataset": args.dataset,
        "baseline": "isolated_gcn_ssl_encoder_plus_few_shot_mlp_head",
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "num_classes": int(num_classes),
        "shots": int(args.shots),
        "seed": int(args.seed),
        "train_count": int(train_idx.numel()),
        "val_count": int(val_idx.numel()),
        "split_integrity": split_integrity,
        "encoder": encoder_info,
        "selection_rule": "selected_by_validation_only; recovered_eval is never used for training, validation, early stopping, or legal model selection",
        "selected_by_validation": best_by_val,
        "oracle_diagnostic_best_by_recovered_eval": oracle_diagnostic_best_by_eval,
        "trials": trial_summaries,
    }
    save_json(summary, out_dir / "summary.json")
    torch.save({"encoder_state": encoder.state_dict(), "embeddings": embeddings.detach().cpu()}, out_dir / "encoder_and_embeddings.pt")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
