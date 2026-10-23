"""Unchanged PURF trainer except agreement CE replacing KL; coverage is observational."""
import math
import os
import time

import torch
from torch.nn import functional as F

from eargtc.joint_explore.joint_purf_v10_v1.model import UnifiedRepair
from eargtc.joint_explore.joint_purf_v10_v1.training import (
    checkpoint_key,
    make_model,
    predict,
    score,
)


DEFAULTS = dict(
    dim=64,
    lr=0.001,
    cross=1.0,
    gate_lr=1.0,
    warmup=0,
    gate_gold=False,
    aux_gold=0.0,
    aux_pseudo=0.0,
    gate_mode="learned",
    class_power=0.0,
    agreement_ce=1.0,
    max_steps=500,
    validate_every=50,
    patience=5,
    batch_size=64,
)

def class_weights(targets, num_classes, power):
    """Return smoothed inverse-frequency weights on CPU, or None at power zero."""
    if not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("Invalid num_classes")
    if not math.isfinite(power) or power < 0:
        raise ValueError("Invalid power")
    if power == 0:
        return None
    values = torch.as_tensor(targets).detach().to(device="cpu", dtype=torch.long).reshape(-1)
    if len(values) and (values.min().item() < 0 or values.max().item() >= num_classes):
        raise ValueError("Target outside class range")
    counts = torch.bincount(values, minlength=num_classes).to(torch.float32)
    smoothed_inverse = (len(values) + num_classes) / (num_classes * (counts + 1))
    return smoothed_inverse.pow(power).clamp(0.25, 4.0).to(torch.float32)


def train(f, s, p, h, config, seed):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Training requires Slurm")
    cfg = {**DEFAULTS, **config}
    if set(cfg) != set(DEFAULTS):
        raise ValueError("Unknown configuration fields")
    for key in ("dim", "max_steps", "validate_every", "patience", "batch_size"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError("Invalid " + key)
    for key in (
        "lr",
        "cross",
        "gate_lr",
        "aux_gold",
        "aux_pseudo",
        "class_power",
        "agreement_ce",
    ):
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError("Invalid " + key)
    if cfg["agreement_ce"] <= 0:
        raise ValueError("Agreement CE must be positive")
    if cfg["gate_mode"] not in ("learned", "fixed"):
        raise ValueError("Invalid gate_mode")
    if (
        cfg["lr"] == 0
        or cfg["gate_lr"] == 0
        or cfg["warmup"] < 0
        or cfg["warmup"] >= cfg["max_steps"]
    ):
        raise ValueError("Invalid optimizer schedule")
    if not len(s["gold_ids"]):
        raise ValueError("Gold support must be nonempty")

    num_classes = f["logp_g"].shape[1]
    training_targets = torch.cat(
        (
            s["gold_targets"].detach().to(device="cpu", dtype=torch.long),
            f["selected_pred"][p["teach_g"]].detach().to(device="cpu", dtype=torch.long),
            f["selected_pred"][p["teach_l"]].detach().to(device="cpu", dtype=torch.long),
        )
    )
    counts = torch.bincount(training_targets, minlength=num_classes)
    weights = class_weights(training_targets, num_classes, cfg["class_power"])

    torch.set_num_threads(2)
    torch.manual_seed(seed)
    model = make_model(f, h, cfg)
    gate = [model.raw_alpha, model.raw_beta]
    body = [
        value
        for name, value in model.named_parameters()
        if name not in ("raw_alpha", "raw_beta")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": body, "lr": cfg["lr"], "weight_decay": 1e-4},
            {
                "params": gate,
                "lr": cfg["lr"] * cfg["gate_lr"],
                "weight_decay": 0.0,
            },
        ]
    )
    rng = {
        name: torch.Generator().manual_seed(seed + offset)
        for name, offset in (
            ("gold", 101),
            ("teach_g", 1101),
            ("teach_l", 2101),
            ("agreement", 3101),
        )
    }
    pool_ids = {"gold": s["gold_ids"], **{k: p[k] for k in rng if k != "gold"}}
    seen = {k: torch.zeros(len(v), dtype=torch.bool) for k, v in pool_ids.items()}
    draws = {k: 0 for k in pool_ids}
    checkpoint_coverage = {}

    def coverage():
        return {k: dict(pool_size=len(v), draws=draws[k], unique=int(seen[k].sum()),
                        repeat_draws=draws[k]-int(seen[k].sum()),
                        fraction=float(seen[k].float().mean()) if len(v) else None)
                for k, v in pool_ids.items()}

    states = {}
    checkpoints = {}
    valps = {}
    history = []
    stale = 0
    start = time.monotonic()

    def validate(step, losses, gradients):
        nonlocal stale
        probs, repair_probs = predict(model, f, h, s["selection_ids"], repair=True)
        value = {**score(probs, s["selection_targets"]), "step": step}
        improve = "accuracy" not in checkpoints or checkpoint_key(value) < checkpoint_key(
            checkpoints["accuracy"]
        )
        for criterion in ("accuracy", "nll"):
            if criterion not in checkpoints or checkpoint_key(
                value, criterion
            ) < checkpoint_key(checkpoints[criterion], criterion):
                states[criterion] = {
                    name: tensor.detach().clone()
                    for name, tensor in model.state_dict().items()
                }
                checkpoints[criterion] = dict(value)
                checkpoint_coverage[criterion] = coverage()
                valps[criterion] = probs.clone()
        stale = 0 if improve else stale + 1
        alpha, beta = model.coefficients()
        history.append(
            {
                **value,
                "repair": score(repair_probs, s["selection_targets"]),
                "alpha": float(alpha.detach()),
                "beta": float(beta.detach()),
                "weights": model.weights().detach().tolist(),
                "losses": losses,
                "gradient_norms": gradients,
                "coverage": coverage(),
            }
        )

    def draw(name, ids):
        if len(ids):
            positions = torch.randint(
                len(ids), (cfg["batch_size"],), generator=rng[name]
            )
            seen[name][positions] = True
            draws[name] += len(positions)
            return ids[positions]
        return ids

    def forward(ids, detach=False):
        return model(
            f["gnn_embeddings"][ids],
            h[ids],
            f["logp_g"][ids],
            f["logp_l"][ids],
            detach_gate=detach or cfg["gate_mode"] == "fixed",
        )

    def nll_loss(log_probability, target):
        if weights is None:
            return F.nll_loss(log_probability, target)
        return F.nll_loss(log_probability, target, weight=weights)

    validate(0, {}, {})
    for step in range(1, cfg["max_steps"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        warm = step <= cfg["warmup"]
        positions = draw("gold", torch.arange(len(s["gold_ids"])))
        ids = s["gold_ids"][positions]
        targets = s["gold_targets"][positions]
        logp, logr = forward(ids, warm)
        gold = nll_loss(logp, targets)
        auxgold = (
            nll_loss(logr, targets) if cfg["aux_gold"] else gold * 0
        )
        loss = gold + cfg["aux_gold"] * auxgold
        losses = {"gold": float(gold.detach()), "aux_gold": float(auxgold.detach())}
        for name in ("teach_g", "teach_l"):
            ids = draw(name, p[name]) if cfg["cross"] else p[name][:0]
            if len(ids):
                logp, logr = forward(ids, warm or cfg["gate_gold"])
                target = f["selected_pred"][ids]
                ce = nll_loss(logp, target)
                aux = (
                    nll_loss(logr, target)
                    if cfg["aux_pseudo"]
                    else ce * 0
                )
                loss = loss + cfg["cross"] * 0.5 * (
                    ce + cfg["aux_pseudo"] * aux
                )
                losses[name] = float(ce.detach())
                losses["aux_" + name] = float(aux.detach())
            else:
                losses[name] = 0.0
                losses["aux_" + name] = 0.0
        ids = draw("agreement", p["agreement"])
        if len(ids):
            logp, logr = forward(ids, warm or cfg["gate_gold"])
            target = f["selected_pred"][ids]
            ce = nll_loss(logp, target)
            aux = nll_loss(logr, target) if cfg["aux_pseudo"] else ce * 0
            loss = loss + cfg["agreement_ce"] * (ce + cfg["aux_pseudo"] * aux)
            losses["agreement"] = float(ce.detach())
            losses["aux_agreement"] = float(aux.detach())
        else:
            losses["agreement"] = 0.0
            losses["aux_agreement"] = 0.0
        losses["total"] = float(loss.detach())
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        if any(
            value.grad is not None and not torch.isfinite(value.grad).all()
            for value in model.parameters()
        ):
            raise ValueError("Nonfinite gradient")
        gradients = {
            name: float(
                torch.linalg.vector_norm(
                    torch.stack(
                        [value.grad.norm() for value in group if value.grad is not None]
                    )
                )
            )
            if any(value.grad is not None for value in group)
            else 0.0
            for name, group in (("body", body), ("gate", gate))
        }
        optimizer.step()
        if step % cfg["validate_every"] == 0 or step == cfg["max_steps"]:
            validate(step, losses, gradients)
            if step > cfg["warmup"] and stale >= cfg["patience"]:
                break

    return dict(
        config=cfg,
        training_seed=seed,
        split_seed=seed,
        inner_ensemble=False,
        states=states,
        checkpoints=checkpoints,
        validation_probability=valps["accuracy"],
        validation_probabilities=valps,
        validation_ids=s["selection_ids"].clone(),
        history=history,
        parameter_count=sum(value.numel() for value in model.parameters()),
        train_seconds=time.monotonic() - start,
        test_truth_loaded=False,
        slurm_job_id=os.environ["SLURM_JOB_ID"],
        coverage=coverage(),
        checkpoint_coverage=checkpoint_coverage,
        class_power=cfg["class_power"],
        class_weights=None if weights is None else weights.tolist(),
        training_class_counts=counts.tolist(),
    )


__all__ = [
    "DEFAULTS",
    "UnifiedRepair",
    "class_weights",
    "train",
    "make_model",
    "predict",
    "score",
    "checkpoint_key",
]
