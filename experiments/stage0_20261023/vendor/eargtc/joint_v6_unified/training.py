"""Leakage-safe full-coverage training for Joint-v6 unified models."""
from __future__ import annotations

import copy
import math
import os
import random

import torch
import torch.nn.functional as F

from .model import FAMILIES, Model


TRAIN_KEYS = {
    "graph",
    "text",
    "logp_g",
    "logp_l",
    "gold_ids",
    "gold_targets",
    "pseudo_ids",
    "pseudo_targets",
}
VALIDATION_KEYS = {"ids", "targets"}
CONFIG_KEYS = {
    "family",
    "width",
    "dropout",
    "pseudo_weight",
    "lr",
    "weight_decay",
    "batch_size",
    "max_epochs",
    "validate_every",
    "patience",
}
OPTIONAL_CONFIG_KEYS = {"id"}
GRADIENT_CLIP = 5.0
RESIDUAL_AUX_WEIGHT = 0.25
PREDICTION_BATCH_SIZE = 4096


def _require_slurm() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Joint-v6 numerical operations require a real Slurm job")


def _exact_schema(value, expected, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a plain dictionary")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} schema mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("config must be a plain dictionary")
    actual = set(config)
    missing = CONFIG_KEYS - actual
    extra = actual - CONFIG_KEYS - OPTIONAL_CONFIG_KEYS
    if missing or extra:
        raise ValueError(f"config schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    if "id" in config and (not isinstance(config["id"], str) or not config["id"]):
        raise ValueError("config id must be a non-empty string")
    family = config["family"]
    if family not in FAMILIES:
        raise ValueError("unknown model family")
    for key in ("width", "batch_size", "max_epochs", "validate_every", "patience"):
        value = config[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    dropout = config["dropout"]
    if not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("invalid dropout")
    for key in ("pseudo_weight", "weight_decay"):
        value = config[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid {key}")
    lr = config["lr"]
    if not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("invalid lr")


def _validate_ids(ids, name, rows, allow_empty):
    if not torch.is_tensor(ids) or ids.dtype != torch.long or ids.ndim != 1:
        raise ValueError(f"{name} must be a rank-one torch.long tensor")
    if not allow_empty and ids.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= rows):
        raise ValueError(f"{name} are out of range")
    if ids.unique().numel() != ids.numel():
        raise ValueError(f"{name} contain duplicates")


def _validate_targets(targets, name, count, classes):
    if not torch.is_tensor(targets) or targets.dtype != torch.long or targets.ndim != 1:
        raise ValueError(f"{name} must be a rank-one torch.long tensor")
    if targets.numel() != count:
        raise ValueError(f"{name} length mismatch")
    if targets.numel() and (int(targets.min()) < 0 or int(targets.max()) >= classes):
        raise ValueError(f"{name} are out of class range")


def _validate_train(train):
    _exact_schema(train, TRAIN_KEYS, "train")
    for key in ("graph", "text", "logp_g", "logp_l"):
        value = train[key]
        if not torch.is_tensor(value) or value.ndim != 2 or not value.is_floating_point():
            raise ValueError(f"train {key} must be a rank-two floating tensor")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"nonfinite train {key}")
    rows = train["graph"].shape[0]
    if rows < 1 or train["text"].shape[0] != rows or train["logp_g"].shape[0] != rows or train["logp_l"].shape[0] != rows:
        raise ValueError("all-node input rows do not align")
    if train["graph"].shape[1] < 1 or train["text"].shape[1] < 1:
        raise ValueError("feature dimensions must be positive")
    classes = train["logp_g"].shape[1]
    if classes < 2 or train["logp_l"].shape[1] != classes:
        raise ValueError("expert class dimensions do not align")
    for key in ("logp_g", "logp_l"):
        normalizer = train[key].detach().float().logsumexp(1)
        if not torch.allclose(normalizer, torch.zeros_like(normalizer), atol=1e-4, rtol=1e-4):
            raise ValueError(f"{key} must contain normalized log probabilities")
    _validate_ids(train["gold_ids"], "gold IDs", rows, allow_empty=False)
    _validate_ids(train["pseudo_ids"], "pseudo IDs", rows, allow_empty=True)
    _validate_targets(train["gold_targets"], "gold targets", len(train["gold_ids"]), classes)
    _validate_targets(train["pseudo_targets"], "pseudo targets", len(train["pseudo_ids"]), classes)
    if bool(torch.isin(train["gold_ids"], train["pseudo_ids"]).any()):
        raise ValueError("gold and pseudo IDs overlap")
    return rows, classes


def _validate_validation(validation, rows, classes, train):
    _exact_schema(validation, VALIDATION_KEYS, "validation")
    _validate_ids(validation["ids"], "validation IDs", rows, allow_empty=False)
    _validate_targets(validation["targets"], "validation targets", len(validation["ids"]), classes)
    if bool(torch.isin(validation["ids"], train["gold_ids"]).any()) or bool(
        torch.isin(validation["ids"], train["pseudo_ids"]).any()
    ):
        raise ValueError("validation IDs appear in a training pool")


def _configure(seed, device):
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but unavailable")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        raw_threads = os.environ.get("OMP_NUM_THREADS", "1")
        try:
            threads = int(raw_threads)
        except ValueError as error:
            raise ValueError("OMP_NUM_THREADS must be a positive integer") from error
        if threads < 1:
            raise ValueError("OMP_NUM_THREADS must be a positive integer")
        torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    return device


def _cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _make_model(train, config, classes, device):
    return Model(
        train["graph"].shape[1],
        train["text"].shape[1],
        classes,
        config["family"],
        config["width"],
        config["dropout"],
    ).to(device)


def _forward_ids(model, train, ids, device):
    return model(
        train["graph"][ids].to(device),
        train["text"][ids].to(device),
        train["logp_g"][ids].to(device),
        train["logp_l"][ids].to(device),
    )


def _per_example_loss(output, targets, family):
    loss = F.nll_loss(output["log_probs"], targets, reduction="none")
    if family == "residual":
        branches = output["branch_log_probs"]
        gathered = branches.gather(
            2, targets[:, None, None].expand(-1, branches.shape[1], 1)
        ).squeeze(2)
        loss = loss + RESIDUAL_AUX_WEIGHT * (-gathered).mean(1)
    return loss


@torch.no_grad()
def _predict_model(model, train, ids, device):
    model.eval()
    parts = []
    for start in range(0, len(ids), PREDICTION_BATCH_SIZE):
        part = ids[start : start + PREDICTION_BATCH_SIZE]
        probability = _forward_ids(model, train, part, device)["log_probs"].exp()
        if not bool(torch.isfinite(probability).all()):
            raise ValueError("nonfinite prediction")
        if not torch.allclose(
            probability.sum(1),
            torch.ones(len(probability), device=probability.device),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("prediction probabilities are not normalized")
        parts.append(probability.cpu())
    classes = train["logp_g"].shape[1]
    return torch.cat(parts) if parts else torch.empty((0, classes), dtype=torch.float32)


def train_member(train, validation, config, seed, device):
    """Train one member with full-pool mean gradients and validation-only selection."""
    _require_slurm()
    _validate_config(config)
    rows, classes = _validate_train(train)
    _validate_validation(validation, rows, classes, train)
    device = _configure(seed, device)
    model = _make_model(train, config, classes, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
    )
    generators = {
        "gold": torch.Generator().manual_seed(seed + 1009),
        "pseudo": torch.Generator().manual_seed(seed + 2017),
    }
    history = []
    best = None
    best_state = None
    stale = 0

    def validate(epoch):
        probability = _predict_model(model, train, validation["ids"], device)
        targets = validation["targets"]
        accuracy = float((probability.argmax(1) == targets).double().mean().item())
        nll = float(
            -probability[torch.arange(len(targets)), targets]
            .double()
            .clamp_min(1e-12)
            .log()
            .mean()
            .item()
        )
        if not math.isfinite(accuracy) or not math.isfinite(nll):
            raise ValueError("nonfinite validation metric")
        return accuracy, nll

    accuracy, nll = validate(0)
    best = {"accuracy": accuracy, "nll": nll, "epoch": 0}
    best_state = _cpu_state(model)
    history.append(
        {
            "epoch": 0,
            "train_loss": None,
            "gold_loss": None,
            "pseudo_loss": None,
            "gold_seen": 0,
            "pseudo_seen": 0,
            "gold_unique": 0,
            "pseudo_unique": 0,
            "gold_batches": 0,
            "pseudo_batches": 0,
            "accuracy": accuracy,
            "nll": nll,
        }
    )

    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pool_stats = {}
        objective = 0.0
        for pool, coefficient in (("gold", 1.0), ("pseudo", float(config["pseudo_weight"]))):
            ids = train[f"{pool}_ids"]
            targets = train[f"{pool}_targets"]
            count = len(ids)
            order = torch.randperm(count, generator=generators[pool])
            seen = []
            loss_sum = 0.0
            batches = 0
            for start in range(0, count, config["batch_size"]):
                positions = order[start : start + config["batch_size"]]
                batch_ids = ids[positions]
                seen.append(batch_ids)
                batches += 1
                if coefficient == 0.0:
                    continue
                output = _forward_ids(model, train, batch_ids, device)
                losses = _per_example_loss(output, targets[positions].to(device), config["family"])
                if not bool(torch.isfinite(losses).all()):
                    raise ValueError("nonfinite training loss")
                scaled = coefficient * losses.sum() / count
                scaled.backward()
                loss_sum += float(losses.detach().double().sum().item())
            seen_ids = torch.cat(seen) if seen else torch.empty(0, dtype=torch.long)
            if seen_ids.numel() != count or seen_ids.unique().numel() != count or not torch.equal(
                seen_ids.sort().values, ids.sort().values
            ):
                raise RuntimeError(f"incomplete or duplicate {pool} epoch coverage")
            mean_loss = loss_sum / count if count and coefficient else 0.0
            objective += coefficient * mean_loss
            pool_stats[pool] = {
                "loss": mean_loss,
                "seen": int(seen_ids.numel()),
                "unique": int(seen_ids.unique().numel()),
                "batches": batches,
            }
        if not math.isfinite(objective):
            raise ValueError("nonfinite epoch objective")
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        if not bool(torch.isfinite(gradient_norm)):
            raise ValueError("nonfinite gradient")
        optimizer.step()

        row = {
            "epoch": epoch,
            "train_loss": objective,
            "gold_loss": pool_stats["gold"]["loss"],
            "pseudo_loss": pool_stats["pseudo"]["loss"],
            "gold_seen": pool_stats["gold"]["seen"],
            "pseudo_seen": pool_stats["pseudo"]["seen"],
            "gold_unique": pool_stats["gold"]["unique"],
            "pseudo_unique": pool_stats["pseudo"]["unique"],
            "gold_batches": pool_stats["gold"]["batches"],
            "pseudo_batches": pool_stats["pseudo"]["batches"],
            "accuracy": None,
            "nll": None,
        }
        if epoch % config["validate_every"] == 0:
            accuracy, nll = validate(epoch)
            row["accuracy"], row["nll"] = accuracy, nll
            candidate = (-accuracy, nll, epoch)
            incumbent = (-best["accuracy"], best["nll"], best["epoch"])
            if candidate < incumbent:
                best = {"accuracy": accuracy, "nll": nll, "epoch": epoch}
                best_state = _cpu_state(model)
                stale = 0
            else:
                stale += 1
        history.append(row)
        if epoch % config["validate_every"] == 0 and stale >= config["patience"]:
            break

    model.load_state_dict(best_state, strict=True)
    validation_probs = _predict_model(model, train, validation["ids"], device)
    return {
        "state": best_state,
        "history": history,
        "best": best,
        "validation_probs": validation_probs,
        "config": copy.deepcopy(config),
        "seed": seed,
    }


def predict(state, config, train, ids, device):
    """Restore one CPU checkpoint and return CPU probabilities in bounded chunks."""
    _require_slurm()
    _validate_config(config)
    _, classes = _validate_train(train)
    _validate_ids(ids, "prediction IDs", train["graph"].shape[0], allow_empty=True)
    if not isinstance(state, dict) or not state or any(not torch.is_tensor(value) for value in state.values()):
        raise ValueError("state must be a non-empty tensor dictionary")
    device = _configure(0, device)
    model = _make_model(train, config, classes, device)
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, KeyError) as error:
        raise ValueError("checkpoint does not match model configuration") from error
    return _predict_model(model, train, ids, device)
