from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    if y.numel() == 0:
        return 0.0
    return float((pred == y).float().mean().item())


def macro_f1(pred: torch.Tensor, y: torch.Tensor, num_classes: int) -> float:
    if y.numel() == 0:
        return 0.0
    pred_np = pred.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    scores = []
    for c in range(num_classes):
        tp = np.logical_and(pred_np == c, y_np == c).sum()
        fp = np.logical_and(pred_np == c, y_np != c).sum()
        fn = np.logical_and(pred_np != c, y_np == c).sum()
        denom = (2 * tp) + fp + fn
        scores.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(np.mean(scores))


def weighted_f1(pred: torch.Tensor, y: torch.Tensor, num_classes: int) -> float:
    if y.numel() == 0:
        return 0.0
    weights = torch.bincount(y.detach().cpu(), minlength=num_classes).float()
    if weights.sum() == 0:
        return 0.0
    pred_np = pred.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    scores = []
    for c in range(num_classes):
        tp = np.logical_and(pred_np == c, y_np == c).sum()
        fp = np.logical_and(pred_np == c, y_np != c).sum()
        fn = np.logical_and(pred_np != c, y_np == c).sum()
        denom = (2 * tp) + fp + fn
        scores.append(0.0 if denom == 0 else (2 * tp) / denom)
    score_tensor = torch.tensor(scores, dtype=torch.float32)
    return float((score_tensor * weights / weights.sum()).sum().item())


@torch.no_grad()
def split_metrics(logits: torch.Tensor, y: torch.Tensor, masks: Dict[str, torch.Tensor], num_classes: int):
    pred = logits.argmax(dim=-1)
    out = {}
    for name, mask in masks.items():
        mask = mask.bool()
        out[name] = {
            "accuracy": accuracy(pred[mask], y[mask]),
            "macro_f1": macro_f1(pred[mask], y[mask], num_classes),
            "weighted_f1": weighted_f1(pred[mask], y[mask], num_classes),
            "count": int(mask.sum().item()),
        }
    return out
