from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import torch

from .losses import router_v7_loss
from .model import RouterV7
from .pairs import PairBatch


@dataclass(frozen=True)
class RouterV7Config:
    hidden_dim: int = 32
    dropout: float = 0.1
    epochs: int = 500
    lr: float = 0.01
    weight_decay: float = 1e-4
    shift_weight: float = 1.0
    clean_weight: float = 0.5
    trust_weight: float = 1.0
    temperature: float = 1.0
    shift_objective: str = "huber"
    delta_clip: float = 4.0
    patience: int = 100
    grad_clip: float = 5.0


@dataclass
class FittedRouterV7:
    model: RouterV7
    local_mean: torch.Tensor
    local_std: torch.Tensor
    context_mean: torch.Tensor
    context_std: torch.Tensor
    config: RouterV7Config
    history: list[dict]

    def predict(
        self, local_features: torch.Tensor, context_features: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local = torch.as_tensor(local_features, dtype=torch.float32)
        local = (local - self.local_mean) / self.local_std
        if self.model.context_dim:
            context = torch.as_tensor(context_features, dtype=torch.float32)
            context = (context - self.context_mean) / self.context_std
        else:
            context = local.new_empty((local.size(0), 0))
        self.model.eval()
        with torch.no_grad():
            output = self.model(local, context)
        return output.preference_logit.cpu(), torch.sigmoid(output.trust_logit).cpu()

    def to_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "local_mean": self.local_mean.tolist(),
            "local_std": self.local_std.tolist(),
            "context_mean": self.context_mean.tolist(),
            "context_std": self.context_std.tolist(),
            "history": self.history,
        }


def _standardize(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if value.size(1) == 0:
        empty = value.new_empty((0,))
        return value, empty, empty
    mean = value.mean(0)
    std = value.std(0, unbiased=False).clamp_min(1e-6)
    return (value - mean) / std, mean, std


def fit_router_v7(
    *,
    local_features: torch.Tensor,
    context_features: torch.Tensor,
    gnn_correct: torch.Tensor,
    llm_correct: torch.Tensor,
    trust_target: torch.Tensor | None,
    row_weight: torch.Tensor,
    pairs: PairBatch,
    clean_mask: torch.Tensor,
    config: RouterV7Config = RouterV7Config(),
    seed: int = 42,
) -> FittedRouterV7:
    local = torch.as_tensor(local_features, dtype=torch.float32)
    context = torch.as_tensor(context_features, dtype=torch.float32)
    if local.ndim != 2 or context.ndim != 2 or local.size(0) != context.size(0):
        raise ValueError("training features must be aligned matrices")
    if local.size(0) == 0 or not bool(torch.isfinite(local).all()) or not bool(torch.isfinite(context).all()):
        raise ValueError("training features must be non-empty and finite")
    local_z, local_mean, local_std = _standardize(local)
    context_z, context_mean, context_std = _standardize(context)
    torch.manual_seed(int(seed))
    model = RouterV7(
        local_dim=local.size(1),
        context_dim=context.size(1),
        hidden_dim=int(config.hidden_dim),
        dropout=float(config.dropout),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay)
    )
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict] = []
    stale = 0
    for epoch in range(max(1, int(config.epochs))):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(local_z, context_z)
        losses = router_v7_loss(
            output,
            gnn_correct=gnn_correct,
            llm_correct=llm_correct,
            trust_target=trust_target,
            row_weight=row_weight,
            pairs=pairs,
            clean_mask=clean_mask,
            shift_weight=config.shift_weight,
            clean_weight=config.clean_weight,
            shift_objective=config.shift_objective,
            delta_clip=config.delta_clip,
            trust_weight=config.trust_weight,
            temperature=config.temperature,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip))
        optimizer.step()
        total = float(losses["total"].detach())
        if epoch == 0 or epoch == int(config.epochs) - 1 or epoch % 10 == 0:
            history.append({"epoch": epoch, **{key: float(value.detach()) for key, value in losses.items()}})
        if total < best_loss - 1e-8:
            best_loss = total
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= max(1, int(config.patience)):
                break
    model.load_state_dict(best_state)
    model.eval()
    return FittedRouterV7(
        model=model,
        local_mean=local_mean,
        local_std=local_std,
        context_mean=context_mean,
        context_std=context_std,
        config=config,
        history=history,
    )


def select_route_threshold(
    probability: torch.Tensor,
    *,
    labels: torch.Tensor,
    gnn_pred: torch.Tensor,
    llm_pred: torch.Tensor,
) -> dict:
    p = torch.as_tensor(probability, dtype=torch.float64).view(-1)
    y = torch.as_tensor(labels).view(-1)
    g = torch.as_tensor(gnn_pred).view(-1)
    l = torch.as_tensor(llm_pred).view(-1)
    if not (p.numel() == y.numel() == g.numel() == l.numel()) or p.numel() == 0:
        raise ValueError("threshold inputs must be equal and non-empty")
    candidates = sorted(set([0.0, 0.5, 1.0, *p.tolist()]))
    trials = []
    for threshold in candidates:
        choose_g = p >= float(threshold)
        selected = torch.where(choose_g, g, l)
        correct = int((selected == y).sum().item())
        decisive = (g != l) & ((g == y) != (l == y))
        decisive_correct = int((selected[decisive] == y[decisive]).sum().item())
        trials.append(
            {
                "threshold": float(threshold),
                "correct": correct,
                "accuracy": correct / p.numel(),
                "decisive_correct": decisive_correct,
            }
        )
    return min(
        trials,
        key=lambda row: (
            -row["correct"],
            -row["decisive_correct"],
            abs(row["threshold"] - 0.5),
            row["threshold"],
        ),
    )
