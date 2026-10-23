"""Full-coverage warm-up and pseudo refinement for Joint-v6 Repair."""
from __future__ import annotations

import copy
import math
import os
import random

import torch
import torch.nn.functional as F

from .model import Head, _require_slurm


TRAIN_KEYS = {
    "graph", "text", "logp_g", "logp_l", "gold_ids", "gold_targets",
    "pseudo_ids", "pseudo_targets",
}
VALIDATION_KEYS = {"ids", "targets"}
CONFIG_KEYS = {
    "id", "width", "dropout", "pseudo_weight", "keep_weight", "lr",
    "weight_decay", "batch_size", "max_epochs", "validate_every", "patience",
}
BRANCHES = (("g", "logp_g", 101), ("l", "logp_l", 202))
CRITERIA = ("accuracy", "nll")
GRADIENT_CLIP = 5.0
PREDICTION_BATCH_SIZE = 4096
HISTORY_KEYS = {
    "epoch", "gold_seen", "gold_unique", "gold_batches", "pseudo_seen",
    "pseudo_unique", "pseudo_batches", "gold_loss", "pseudo_loss",
    "total_loss", "accuracy", "nll",
}


def _exact_schema(value, expected, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a plain dictionary")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} schema mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _validate_config(config, stage=None):
    _exact_schema(config, CONFIG_KEYS, "config")
    if not isinstance(config["id"], str) or not config["id"]:
        raise ValueError("config id must be a non-empty string")
    for key in ("width", "batch_size", "max_epochs", "validate_every", "patience"):
        value = config[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if stage == "refine" and config["max_epochs"] < config["validate_every"]:
        raise ValueError("refine requires a positive scheduled validation epoch")
    dropout = config["dropout"]
    if (not isinstance(dropout, (int, float)) or isinstance(dropout, bool)
            or not math.isfinite(float(dropout)) or not 0 <= float(dropout) < 1):
        raise ValueError("invalid dropout")
    for key in ("pseudo_weight", "keep_weight", "weight_decay"):
        value = config[key]
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(float(value)) or float(value) < 0):
            raise ValueError(f"invalid {key}")
    if (not isinstance(config["lr"], (int, float)) or isinstance(config["lr"], bool)
            or not math.isfinite(float(config["lr"])) or float(config["lr"]) <= 0):
        raise ValueError("invalid lr")
    if stage == "warm":
        if float(config["pseudo_weight"]) != 0:
            raise ValueError("warm pseudo_weight must be zero")
        if float(config["keep_weight"]) != 0:
            raise ValueError("warm keep_weight must be zero")
    if stage == "refine" and float(config["pseudo_weight"]) <= 0:
        raise ValueError("refine pseudo_weight must be strictly positive")


def _validate_ids(ids, name, rows, allow_empty):
    if (not torch.is_tensor(ids) or ids.device.type != "cpu"
            or ids.dtype != torch.long or ids.ndim != 1):
        raise ValueError(f"{name} must be a CPU rank-one torch.long tensor")
    if not allow_empty and ids.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= rows):
        raise ValueError(f"{name} are out of range")
    if ids.unique().numel() != ids.numel():
        raise ValueError(f"{name} contain duplicates")


def _validate_targets(targets, name, count, classes):
    if (not torch.is_tensor(targets) or targets.device.type != "cpu"
            or targets.dtype != torch.long or targets.ndim != 1):
        raise ValueError(f"{name} must be a CPU rank-one torch.long tensor")
    if targets.numel() != count:
        raise ValueError(f"{name} length mismatch")
    if targets.numel() and (int(targets.min()) < 0 or int(targets.max()) >= classes):
        raise ValueError(f"{name} are out of class range")


def _validate_train(train):
    _exact_schema(train, TRAIN_KEYS, "train")
    for key in ("graph", "text", "logp_g", "logp_l"):
        value = train[key]
        if (not torch.is_tensor(value) or value.device.type != "cpu"
                or not value.is_floating_point() or value.ndim != 2):
            raise ValueError(f"train {key} must be a CPU rank-two floating tensor")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"nonfinite train {key}")
    rows = train["graph"].shape[0]
    if (rows < 1 or train["text"].shape[0] != rows or train["logp_g"].shape[0] != rows
            or train["logp_l"].shape[0] != rows):
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
    _validate_targets(
        train["pseudo_targets"], "pseudo targets", len(train["pseudo_ids"]), classes
    )
    if bool(torch.isin(train["gold_ids"], train["pseudo_ids"]).any()):
        raise ValueError("gold and pseudo IDs overlap")
    return rows, classes


def _validate_validation(validation, rows, classes, train):
    _exact_schema(validation, VALIDATION_KEYS, "validation")
    _validate_ids(validation["ids"], "validation IDs", rows, allow_empty=False)
    _validate_targets(
        validation["targets"], "validation targets", len(validation["ids"]), classes
    )
    if (bool(torch.isin(validation["ids"], train["gold_ids"]).any())
            or bool(torch.isin(validation["ids"], train["pseudo_ids"]).any())):
        raise ValueError("validation IDs appear in a training pool")


def _configure(seed, device):
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    resolved = torch.device(device)
    if resolved.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but unavailable")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if resolved.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    else:
        try:
            threads = int(os.environ.get("OMP_NUM_THREADS", "2"))
        except ValueError as error:
            raise ValueError("OMP_NUM_THREADS must be a positive integer") from error
        if threads < 1:
            raise ValueError("OMP_NUM_THREADS must be a positive integer")
        torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    return resolved


def _cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _make_head(train, config, classes, device):
    return Head(
        train["graph"].shape[1], train["text"].shape[1], classes,
        config["width"], config["dropout"],
    ).to(device)


def _forward_delta(model, train, ids, device):
    return model(train["graph"][ids].to(device), train["text"][ids].to(device))


def _branch_log_probs(model, train, ids, branch, device):
    delta = _forward_delta(model, train, ids, device)
    base = train[f"logp_{branch}"][ids].detach().to(device=device, dtype=torch.float32)
    return (base + delta).log_softmax(1)


@torch.no_grad()
def _predict_delta_model(model, train, ids, device):
    model.eval()
    parts = []
    for start in range(0, len(ids), PREDICTION_BATCH_SIZE):
        part = _forward_delta(model, train, ids[start:start + PREDICTION_BATCH_SIZE], device)
        if not bool(torch.isfinite(part).all()):
            raise ValueError("nonfinite predicted delta")
        parts.append(part.detach().to(device="cpu", dtype=torch.float32))
    classes = train["logp_g"].shape[1]
    return torch.cat(parts) if parts else torch.empty((0, classes), dtype=torch.float32)


@torch.no_grad()
def _teacher_probabilities(model, train, ids, branch, device):
    model.eval()
    parts = []
    for start in range(0, len(ids), PREDICTION_BATCH_SIZE):
        part_ids = ids[start:start + PREDICTION_BATCH_SIZE]
        probabilities = _branch_log_probs(model, train, part_ids, branch, device).exp()
        if not bool(torch.isfinite(probabilities).all()):
            raise ValueError("nonfinite teacher probability")
        parts.append(probabilities.detach().cpu())
    classes = train["logp_g"].shape[1]
    return torch.cat(parts) if parts else torch.empty((0, classes), dtype=torch.float32)


def _pseudo_losses(student_log_probs, targets, teacher_probabilities, keep_weight):
    losses = F.nll_loss(student_log_probs, targets, reduction="none")
    if float(keep_weight) != 0:
        losses = losses + float(keep_weight) * F.kl_div(
            student_log_probs, teacher_probabilities, reduction="none"
        ).sum(1)
    return losses


@torch.no_grad()
def _validation_metrics(model, train, validation, branch, device):
    model.eval()
    ids = validation["ids"]
    targets = validation["targets"]
    pieces = []
    for start in range(0, len(ids), PREDICTION_BATCH_SIZE):
        part_ids = ids[start:start + PREDICTION_BATCH_SIZE]
        pieces.append(_branch_log_probs(model, train, part_ids, branch, device).cpu())
    probabilities = torch.cat(pieces).exp().double()
    accuracy = float((probabilities.argmax(1) == targets).double().mean().item())
    nll = float(
        -probabilities[torch.arange(len(targets)), targets]
        .clamp_min(1e-12)
        .log()
        .mean()
        .item()
    )
    if not math.isfinite(accuracy) or not math.isfinite(nll):
        raise ValueError("nonfinite validation metric")
    return accuracy, nll


def _metric_record(epoch, metrics):
    return {"epoch": int(epoch), "accuracy": float(metrics[0]), "nll": float(metrics[1])}


def _accuracy_rank(record):
    return (-record["accuracy"], record["nll"], record["epoch"])


def _nll_rank(record):
    return (record["nll"], -record["accuracy"], record["epoch"])


def _empty_history_row(epoch, accuracy=None, nll=None):
    row = {
        "epoch": int(epoch),
        "gold_seen": 0,
        "gold_unique": 0,
        "gold_batches": 0,
        "pseudo_seen": 0,
        "pseudo_unique": 0,
        "pseudo_batches": 0,
        "gold_loss": None,
        "pseudo_loss": None,
        "total_loss": None,
        "accuracy": accuracy,
        "nll": nll,
    }
    assert set(row) == HISTORY_KEYS
    return row


def _pool_epoch(model, train, ids, targets, branch, device, batch_size, generator,
                coefficient, teacher_probabilities=None, keep_weight=0.0):
    count = len(ids)
    order = torch.randperm(count, generator=generator)
    seen = []
    loss_sum = 0.0
    batches = 0
    for start in range(0, count, batch_size):
        positions = order[start:start + batch_size]
        batch_ids = ids[positions]
        seen.append(batch_ids)
        batches += 1
        log_probs = _branch_log_probs(model, train, batch_ids, branch, device)
        batch_targets = targets[positions].to(device)
        if teacher_probabilities is None:
            losses = F.nll_loss(log_probs, batch_targets, reduction="none")
        else:
            losses = _pseudo_losses(
                log_probs, batch_targets, teacher_probabilities[positions].to(device), keep_weight
            )
        if not bool(torch.isfinite(losses).all()):
            raise ValueError("nonfinite training loss")
        (float(coefficient) * losses.sum() / count).backward()
        loss_sum += float(losses.detach().double().sum().item())
    seen_ids = torch.cat(seen) if seen else torch.empty(0, dtype=torch.long)
    if (seen_ids.numel() != count or seen_ids.unique().numel() != count
            or not torch.equal(seen_ids.sort().values, ids.sort().values)):
        raise RuntimeError("incomplete or duplicate epoch coverage")
    return {
        "loss": loss_sum / count if count else 0.0,
        "seen": int(seen_ids.numel()),
        "unique": int(seen_ids.unique().numel()),
        "batches": batches,
    }


def _validate_state(state, name):
    if (not isinstance(state, dict) or not state
            or any(not isinstance(key, str) or not torch.is_tensor(value)
                   or value.device.type != "cpu" for key, value in state.items())):
        raise ValueError(f"{name} must be a non-empty CPU tensor state dictionary")


def _warm_initial_state(warm, branch, config, seed):
    _exact_schema(warm, {"stage", "config", "seed", "branches"}, "warm result")
    if warm["stage"] != "warm" or warm["seed"] != seed:
        raise ValueError("warm result stage or seed mismatch")
    _validate_config(warm["config"], stage="warm")
    if (warm["config"]["width"] != config["width"]
            or float(warm["config"]["dropout"]) != float(config["dropout"])):
        raise ValueError("warm result architecture differs from refine config")
    if not isinstance(warm["branches"], dict) or set(warm["branches"]) != {"g", "l"}:
        raise ValueError("warm result branch schema mismatch")
    record = warm["branches"][branch]
    _exact_schema(record, {"states", "best", "validation_delta", "history"}, "warm branch")
    if not isinstance(record["states"], dict) or set(record["states"]) != set(CRITERIA):
        raise ValueError("warm state criteria mismatch")
    state = record["states"]["accuracy"]
    _validate_state(state, "warm accuracy state")
    return state


def _train_branch(train, validation, config, seed, device, branch, stage, warm_state=None):
    classes = train["logp_g"].shape[1]
    device = _configure(seed, device)
    model = _make_head(train, config, classes, device)
    if warm_state is not None:
        try:
            model.load_state_dict(warm_state, strict=True)
        except (RuntimeError, KeyError) as error:
            raise ValueError("warm checkpoint does not match refine configuration") from error
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
    )
    generators = {
        "gold": torch.Generator().manual_seed(seed + 1009),
        "pseudo": torch.Generator().manual_seed(seed + 2017),
    }

    teacher_probabilities = None
    if (stage == "refine" and len(train["pseudo_ids"]) > 0
            and float(config["keep_weight"]) != 0):
        teacher = _make_head(train, config, classes, device)
        teacher.load_state_dict(warm_state, strict=True)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teacher_probabilities = _teacher_probabilities(
            teacher, train, train["pseudo_ids"], branch, device
        ).detach()

    initial_metrics = _validation_metrics(model, train, validation, branch, device)
    history = [_empty_history_row(0, *initial_metrics)]
    states = {criterion: None for criterion in CRITERIA}
    best = {criterion: None for criterion in CRITERIA}
    if stage == "warm":
        initial = _metric_record(0, initial_metrics)
        for criterion in CRITERIA:
            states[criterion] = _cpu_state(model)
            best[criterion] = copy.deepcopy(initial)
    stale = 0

    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        gold = _pool_epoch(
            model, train, train["gold_ids"], train["gold_targets"], branch, device,
            config["batch_size"], generators["gold"], 1.0,
        )
        pseudo = {"loss": 0.0, "seen": 0, "unique": 0, "batches": 0}
        if stage == "refine" and len(train["pseudo_ids"]) > 0:
            pseudo = _pool_epoch(
                model, train, train["pseudo_ids"], train["pseudo_targets"], branch, device,
                config["batch_size"], generators["pseudo"], config["pseudo_weight"],
                teacher_probabilities=teacher_probabilities,
                keep_weight=config["keep_weight"],
            )
        total_loss = gold["loss"] + float(config["pseudo_weight"]) * pseudo["loss"]
        if not math.isfinite(total_loss):
            raise ValueError("nonfinite epoch objective")
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        if not bool(torch.isfinite(gradient_norm)):
            raise ValueError("nonfinite gradient")
        optimizer.step()

        row = {
            "epoch": epoch,
            "gold_seen": gold["seen"],
            "gold_unique": gold["unique"],
            "gold_batches": gold["batches"],
            "pseudo_seen": pseudo["seen"],
            "pseudo_unique": pseudo["unique"],
            "pseudo_batches": pseudo["batches"],
            "gold_loss": gold["loss"],
            "pseudo_loss": pseudo["loss"],
            "total_loss": total_loss,
            "accuracy": None,
            "nll": None,
        }
        if epoch % config["validate_every"] == 0:
            metrics = _validation_metrics(model, train, validation, branch, device)
            row["accuracy"], row["nll"] = metrics
            candidate = _metric_record(epoch, metrics)
            accuracy_improved = (
                best["accuracy"] is None
                or _accuracy_rank(candidate) < _accuracy_rank(best["accuracy"])
            )
            if accuracy_improved:
                best["accuracy"] = copy.deepcopy(candidate)
                states["accuracy"] = _cpu_state(model)
                stale = 0
            else:
                stale += 1
            if best["nll"] is None or _nll_rank(candidate) < _nll_rank(best["nll"]):
                best["nll"] = copy.deepcopy(candidate)
                states["nll"] = _cpu_state(model)
        assert set(row) == HISTORY_KEYS
        history.append(row)
        if epoch % config["validate_every"] == 0 and stale >= config["patience"]:
            break

    if any(best[criterion] is None or states[criterion] is None for criterion in CRITERIA):
        raise RuntimeError("training finished without an eligible checkpoint")
    validation_delta = {}
    for criterion in CRITERIA:
        model.load_state_dict(states[criterion], strict=True)
        validation_delta[criterion] = _predict_delta_model(
            model, train, validation["ids"], device
        )
    return {
        "states": states,
        "best": best,
        "validation_delta": validation_delta,
        "history": history,
    }


def _train_member(train, validation, config, seed, device, stage, warm=None):
    _require_slurm()
    _validate_config(config, stage=stage)
    rows, classes = _validate_train(train)
    _validate_validation(validation, rows, classes, train)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    branches = {}
    for branch, _base_key, offset in BRANCHES:
        warm_state = None if stage == "warm" else _warm_initial_state(
            warm, branch, config, seed
        )
        branches[branch] = _train_branch(
            train, validation, config, seed + offset, device, branch, stage, warm_state
        )
    return {
        "stage": stage,
        "config": copy.deepcopy(config),
        "seed": seed,
        "branches": branches,
    }


def warm_member(train, val, config, seed, device):
    """Train two independent gold-only branch heads."""
    return _train_member(train, val, config, seed, device, stage="warm")


def refine_member(train, val, config, seed, device, warm):
    """Refine warm accuracy checkpoints with uniform pseudo supervision."""
    return _train_member(train, val, config, seed, device, stage="refine", warm=warm)


def predict_delta(state, config, train, ids, device):
    """Restore one Head state and return raw CPU float32 deltas in bounded batches."""
    _require_slurm()
    _validate_config(config)
    rows, classes = _validate_train(train)
    _validate_ids(ids, "prediction IDs", rows, allow_empty=True)
    _validate_state(state, "state")
    resolved = _configure(0, device)
    model = _make_head(train, config, classes, resolved)
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, KeyError) as error:
        raise ValueError("checkpoint does not match model configuration") from error
    return _predict_delta_model(model, train, ids, resolved)
