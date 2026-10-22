from __future__ import annotations

import torch


CONTEXT_NAMES = [
    "ctx_gnn_accuracy",
    "ctx_llm_accuracy",
    "ctx_conflict_any_correct",
    "ctx_conflict_gnn_win",
    "ctx_agreement_correct",
    "ctx_conflict_rate",
]


def _agreement(row: dict) -> bool:
    if "agreement" in row:
        return bool(row["agreement"])
    return int(row["gnn_pred"]) == int(row["llm_pred"])


def _posterior(successes: int, trials: int, alpha: float, beta: float) -> float:
    return (float(successes) + float(alpha)) / (float(trials) + float(alpha) + float(beta))


def beta_context(rows: list[dict], alpha: float = 1.0, beta: float = 1.0) -> tuple[torch.Tensor, list[str]]:
    g = [bool(row["gnn_correct"]) for row in rows]
    l = [bool(row["llm_correct"]) for row in rows]
    agree = [_agreement(row) for row in rows]
    conflict_ids = [i for i, same in enumerate(agree) if not same]
    agreement_ids = [i for i, same in enumerate(agree) if same]
    decisive_ids = [i for i in conflict_ids if g[i] != l[i]]

    values = [
        _posterior(sum(g), len(rows), alpha, beta),
        _posterior(sum(l), len(rows), alpha, beta),
        _posterior(sum(g[i] or l[i] for i in conflict_ids), len(conflict_ids), alpha, beta),
        _posterior(sum(g[i] and not l[i] for i in decisive_ids), len(decisive_ids), alpha, beta),
        _posterior(sum(g[i] for i in agreement_ids), len(agreement_ids), alpha, beta),
        _posterior(len(conflict_ids), len(rows), alpha, beta),
    ]
    return torch.tensor(values, dtype=torch.float32), list(CONTEXT_NAMES)


def leave_one_out_contexts(
    rows: list[dict], alpha: float = 1.0, beta: float = 1.0
) -> tuple[torch.Tensor, list[str]]:
    if not rows:
        return torch.empty((0, len(CONTEXT_NAMES)), dtype=torch.float32), list(CONTEXT_NAMES)
    values = [beta_context(rows[:i] + rows[i + 1 :], alpha=alpha, beta=beta)[0] for i in range(len(rows))]
    return torch.stack(values), list(CONTEXT_NAMES)

