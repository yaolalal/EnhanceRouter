from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Iterable
from typing import Callable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1000
    validate_every: int = 50
    patience_evals: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    gamma_max: float = 0.5
    gradient_accumulation: int = 1


@dataclass(frozen=True)
class TrainResult:
    optimizer_steps: int
    best_step: int
    best_metrics: dict[str, float]
    stopped_early: bool
    history: list[dict]


def _atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Materialize only optimized parameters, detached on CPU."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _better(metrics: dict[str, float], step: int, best: tuple | None) -> bool:
    key = (-float(metrics["accuracy"]), float(metrics["nll"]), int(step))
    return best is None or key < best


def train_anchored_joint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[nn.Module, int], torch.Tensor | Iterable[torch.Tensor]],
    validate_fn: Callable[[nn.Module], dict[str, float]],
    output_dir: Path,
    config: TrainConfig,
    step0_check: Callable[[nn.Module], bool],
) -> TrainResult:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("step0.pt", "best_validation.pt", "raw_final.pt")):
        raise FileExistsError(f"refusing to overwrite checkpoints in {output}")
    if not step0_check(model):
        raise RuntimeError("step 0 checkpoint is not Scheme B-equivalent")
    _atomic_torch_save({"step": 0, "model": trainable_state_dict(model)}, output / "step0.pt")

    accumulation = max(int(config.gradient_accumulation), 1)
    validate_every = max(int(config.validate_every), 1)
    optimizer_steps = 0
    microstep = 0
    best_key = None
    best_step = 0
    best_metrics: dict[str, float] = {}
    stale = 0
    history = []
    stopped_early = False
    optimizer.zero_grad(set_to_none=True)
    while optimizer_steps < int(config.steps):
        for _ in range(accumulation):
            produced = loss_fn(model, microstep)
            terms = (produced,) if isinstance(produced, torch.Tensor) else produced
            term_count = 0
            for loss in terms:
                if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                    raise TypeError("loss_fn must produce scalar tensors")
                (loss / accumulation).backward()
                term_count += 1
            if term_count == 0:
                raise RuntimeError("loss_fn produced no loss terms")
            microstep += 1
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        if optimizer_steps % validate_every != 0:
            continue
        metrics = {key: float(value) for key, value in validate_fn(model).items()}
        row = {"step": optimizer_steps, **metrics}
        history.append(row)
        if _better(metrics, optimizer_steps, best_key):
            best_key = (-metrics["accuracy"], metrics["nll"], optimizer_steps)
            best_step = optimizer_steps
            best_metrics = metrics
            stale = 0
            _atomic_torch_save(
                {"step": optimizer_steps, "metrics": metrics, "model": trainable_state_dict(model)},
                output / "best_validation.pt",
            )
        else:
            stale += 1
            if stale >= int(config.patience_evals):
                stopped_early = True
                break

    _atomic_torch_save(
        {"step": optimizer_steps, "model": trainable_state_dict(model)}, output / "raw_final.pt"
    )
    record = {
        "selection_data": "official_validation_only",
        "best_step": best_step,
        "best_metrics": best_metrics,
        "optimizer_steps": optimizer_steps,
        "stopped_early": stopped_early,
        "config": asdict(config),
        "history": history,
    }
    (output / "checkpoint_selection.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return TrainResult(
        optimizer_steps,
        best_step,
        best_metrics,
        stopped_early,
        history,
    )
