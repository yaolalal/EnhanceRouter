from __future__ import annotations

import torch


def factorized_probabilities(a: torch.Tensor, c: torch.Tensor) -> dict[str, torch.Tensor]:
    a = torch.as_tensor(a).clamp(0.0, 1.0)
    c = torch.as_tensor(c).clamp(0.0, 1.0)
    if a.shape != c.shape:
        raise ValueError("a and c must have identical shapes")
    return {"gnn_only": a * c, "llm_only": a * (1.0 - c), "neither": 1.0 - a}


def select_preference_threshold(
    score: torch.Tensor,
    gnn_correct: torch.Tensor,
    llm_correct: torch.Tensor,
    disagreement: torch.Tensor,
    *,
    default_threshold: float = 0.5,
) -> dict:
    score = torch.as_tensor(score, dtype=torch.float64).view(-1)
    g = torch.as_tensor(gnn_correct, dtype=torch.bool).view(-1)
    l = torch.as_tensor(llm_correct, dtype=torch.bool).view(-1)
    conflict = torch.as_tensor(disagreement, dtype=torch.bool).view(-1)
    if not (score.numel() == g.numel() == l.numel() == conflict.numel()):
        raise ValueError("Threshold inputs must have the same length")
    decisive = conflict & (g != l)
    if not bool(decisive.any().item()):
        return {
            "threshold": float(default_threshold),
            "correct": 0,
            "decisive_count": 0,
            "accuracy": None,
            "source": "default_no_decisive_validation_conflict",
        }
    s = score[decisive]
    yg = g[decisive]
    unique = torch.unique(s).sort().values
    candidates = [float(unique[0].item()) - 1e-12]
    candidates.extend(float((unique[i - 1] + unique[i]).item() / 2.0) for i in range(1, unique.numel()))
    candidates.append(float(unique[-1].item()) + 1e-12)
    candidates.append(float(default_threshold))
    best = None
    for threshold in candidates:
        pred_g = s >= threshold
        correct = int((pred_g == yg).sum().item())
        key = (correct, -abs(float(threshold) - float(default_threshold)))
        if best is None or key > best[0]:
            best = (key, float(threshold), correct)
    assert best is not None
    return {
        "threshold": best[1],
        "correct": best[2],
        "decisive_count": int(decisive.sum().item()),
        "accuracy": best[2] / int(decisive.sum().item()),
        "source": "official_validation_decisive_conflicts",
    }

