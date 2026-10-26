from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1000
    validate_every: int = 50
    patience_evals: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    gradient_accumulation: int = 1


@dataclass(frozen=True)
class TrainResult:
    optimizer_steps: int
    best_step: int
    best_metrics: dict[str, float]
    stopped_early: bool
    history: list[dict]


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _save(value: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def train_direct_joint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[nn.Module, int], torch.Tensor | Iterable[torch.Tensor]],
    validate_fn: Callable[[nn.Module], dict[str, float]],
    output_dir: Path,
    config: TrainConfig,
    initial_semantics_check: Callable[[nn.Module], bool],
) -> TrainResult:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = ("initial.pt", "best_validation.pt", "raw_final.pt")
    if any((output / name).exists() for name in names):
        raise FileExistsError(f"refusing to overwrite checkpoints in {output}")
    if not initial_semantics_check(model):
        raise RuntimeError("initial model failed direct numbered-logit semantics")
    _save({"step": 0, "model": trainable_state_dict(model)}, output / "initial.pt")

    accumulation = max(int(config.gradient_accumulation), 1)
    validate_every = max(int(config.validate_every), 1)
    optimizer_steps = 0
    microstep = 0
    best_key = None
    best_step = 0
    best_metrics: dict[str, float] = {}
    stale = 0
    stopped = False
    history = []
    optimizer.zero_grad(set_to_none=True)
    while optimizer_steps < int(config.steps):
        for _ in range(accumulation):
            produced = loss_fn(model, microstep)
            terms = (produced,) if isinstance(produced, torch.Tensor) else produced
            count = 0
            for term in terms:
                if not isinstance(term, torch.Tensor) or term.ndim != 0:
                    raise TypeError("loss_fn must produce scalar tensors")
                (term / accumulation).backward()
                count += 1
            if count == 0:
                raise RuntimeError("loss_fn produced no loss terms")
            microstep += 1
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        if optimizer_steps % validate_every:
            continue
        metrics = {name: float(value) for name, value in validate_fn(model).items()}
        if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
            raise RuntimeError("validation metrics contain NaN/Inf")
        row = {"step": optimizer_steps, **metrics}
        history.append(row)
        key = (-metrics["accuracy"], metrics["nll"], optimizer_steps)
        if best_key is None or key < best_key:
            best_key = key
            best_step = optimizer_steps
            best_metrics = metrics
            stale = 0
            _save(
                {"step": optimizer_steps, "metrics": metrics, "model": trainable_state_dict(model)},
                output / "best_validation.pt",
            )
        else:
            stale += 1
            if stale >= int(config.patience_evals):
                stopped = True
                break
    _save(
        {"step": optimizer_steps, "model": trainable_state_dict(model)},
        output / "raw_final.pt",
    )
    record = {
        "selection_data": "official_validation_only",
        "best_step": best_step,
        "best_metrics": best_metrics,
        "optimizer_steps": optimizer_steps,
        "stopped_early": stopped,
        "config": asdict(config),
        "history": history,
    }
    (output / "checkpoint_selection.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return TrainResult(
        optimizer_steps, best_step, best_metrics, stopped, history
    )
