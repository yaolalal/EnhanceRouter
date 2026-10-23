"""Fixed-budget full-pool training without query truth access."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
import os
from typing import Mapping

import torch
import torch.nn.functional as F

from .model import ARCHITECTURES, DualResidualModel, Head


TRAIN_KEYS = {
    "node_ids", "graph", "text", "logp_g", "logp_l", "choose_g", "agreement",
    "selected_pred", "gold_ids", "gold_targets", "query_ids",
    "disagreement_choose_g_ids", "disagreement_choose_l_ids", "agreement_ids",
}
CHECKPOINT_KEYS = {"ids", "targets"}
SAMPLING = {"full_coverage", "full_replacement", "legacy64_replacement"}


def _slurm_only() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Joint-v6 systematic numerical work must run under Slurm")


@dataclass(frozen=True)
class TrainConfig:
    architecture: str
    width: int
    dropout: float
    lr: float
    weight_decay: float
    pseudo_weight: float
    keep_weight: float
    updates: int
    warm_updates: int
    validate_every: int
    chunk_size: int
    gold_sampling: str
    pseudo_sampling: str
    legacy_draws: int
    disagreement_all: bool
    agreement_ce: bool
    residual: bool
    clip_norm: float | None
    teacher: str
    gold_only_control: bool
    inference_batch: int
    threads: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | "TrainConfig") -> "TrainConfig":
        if isinstance(value, cls):
            result = value
        else:
            expected = {field.name for field in fields(cls)}
            if not isinstance(value, Mapping) or set(value) != expected:
                raise ValueError("invalid training config schema")
            result = cls(**value)
        result.validate()
        return result

    def validate(self) -> None:
        if self.architecture not in ARCHITECTURES:
            raise ValueError("unknown architecture")
        for name in ("width", "updates", "validate_every", "chunk_size", "legacy_draws", "inference_batch", "threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.warm_updates) is not int or not 0 <= self.warm_updates < self.updates:
            raise ValueError("warm_updates must leave at least one mixed update")
        for name in ("dropout", "lr", "weight_decay", "pseudo_weight", "keep_weight"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name}")
        if self.lr <= 0 or self.dropout >= 1:
            raise ValueError("lr must be positive and dropout below one")
        if self.clip_norm is not None and (
            not isinstance(self.clip_norm, (int, float))
            or not math.isfinite(self.clip_norm)
            or self.clip_norm <= 0
        ):
            raise ValueError("clip_norm must be positive or None")
        for name in ("disagreement_all", "agreement_ce", "residual", "gold_only_control"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be Boolean")
        if self.gold_sampling not in SAMPLING or self.pseudo_sampling not in SAMPLING:
            raise ValueError("unknown sampling mode")
        if self.teacher not in {"original", "warm"}:
            raise ValueError("teacher must be original or warm")
        if self.teacher == "warm" and self.warm_updates == 0:
            raise ValueError("warm teacher requires a positive warm prefix")
        zero_pseudo = self.pseudo_weight == 0 and self.keep_weight == 0
        if zero_pseudo != self.gold_only_control:
            raise ValueError("zero pseudo losses require the explicit gold_only_control")


def config_schema() -> tuple[str, ...]:
    return tuple(field.name for field in fields(TrainConfig))


def _ids(value: torch.Tensor, n: int, name: str, *, allow_empty: bool = True) -> None:
    if not torch.is_tensor(value) or value.dtype != torch.long or value.ndim != 1:
        raise ValueError(f"{name} must be an int64 vector")
    if not allow_empty and len(value) == 0:
        raise ValueError(f"{name} cannot be empty")
    if len(value) != len(value.unique()) or bool(((value < 0) | (value >= n)).any()):
        raise ValueError(f"{name} contains duplicate or out-of-range IDs")


def _finite_rows(value: torch.Tensor, name: str) -> None:
    for start in range(0, len(value), 4096):
        if not bool(torch.isfinite(value[start : start + 4096]).all()):
            raise ValueError(f"{name} contains nonfinite values")


def validate_inputs(train: Mapping[str, torch.Tensor], checkpoint: Mapping[str, torch.Tensor] | None = None) -> None:
    if not isinstance(train, Mapping) or set(train) != TRAIN_KEYS:
        raise ValueError("invalid training schema")
    n = len(train["node_ids"])
    if n == 0 or not torch.equal(train["node_ids"], torch.arange(n)):
        raise ValueError("node IDs must be canonical")
    for name in ("graph", "text", "logp_g", "logp_l"):
        value = train[name]
        if not torch.is_tensor(value) or value.ndim != 2 or value.shape[0] != n or value.shape[1] < 1:
            raise ValueError(f"invalid {name}")
        _finite_rows(value, name)
    if train["logp_g"].shape != train["logp_l"].shape:
        raise ValueError("expert class spaces differ")
    classes = train["logp_g"].shape[1]
    for name in ("logp_g", "logp_l"):
        if not torch.allclose(train[name].logsumexp(1), torch.zeros(n), atol=2e-5):
            raise ValueError("base log probabilities are not normalized")
    for name in ("choose_g", "agreement"):
        if train[name].dtype != torch.bool or train[name].shape != (n,):
            raise ValueError(f"invalid {name}")
    if train["selected_pred"].dtype != torch.long or train["selected_pred"].shape != (n,):
        raise ValueError("invalid selected predictions")
    g, l = train["logp_g"].argmax(1), train["logp_l"].argmax(1)
    if not torch.equal(train["agreement"], g == l):
        raise ValueError("agreement differs from expert predictions")
    if not torch.equal(train["selected_pred"], torch.where(train["choose_g"], g, l)):
        raise ValueError("selected predictions differ from frozen routing")
    for name in ("gold_ids", "query_ids", "disagreement_choose_g_ids", "disagreement_choose_l_ids", "agreement_ids"):
        _ids(train[name], n, name, allow_empty=name not in {"gold_ids", "query_ids"})
    if bool(torch.isin(train["gold_ids"], train["query_ids"]).any()):
        raise ValueError("gold and query IDs overlap")
    pool_names = ("disagreement_choose_g_ids", "disagreement_choose_l_ids", "agreement_ids")
    all_pools = torch.cat([train[name] for name in pool_names])
    if len(all_pools) != len(all_pools.unique()) or not bool(torch.isin(all_pools, train["query_ids"]).all()):
        raise ValueError("accepted pools must be disjoint query subsets")
    dg, dl, agree = (train[name] for name in pool_names)
    if not bool((~train["agreement"][dg] & train["choose_g"][dg]).all()):
        raise ValueError("invalid selected-GNN disagreement pool")
    if not bool((~train["agreement"][dl] & ~train["choose_g"][dl]).all()):
        raise ValueError("invalid selected-LLM disagreement pool")
    if not bool(train["agreement"][agree].all()):
        raise ValueError("invalid agreement pool")
    targets = train["gold_targets"]
    if targets.dtype != torch.long or targets.shape != train["gold_ids"].shape:
        raise ValueError("invalid gold targets")
    if bool(((targets < 0) | (targets >= classes)).any()):
        raise ValueError("gold target outside class space")
    if checkpoint is not None:
        if not isinstance(checkpoint, Mapping) or set(checkpoint) != CHECKPOINT_KEYS:
            raise ValueError("invalid checkpoint schema")
        _ids(checkpoint["ids"], n, "checkpoint ids", allow_empty=False)
        y = checkpoint["targets"]
        if y.dtype != torch.long or y.shape != checkpoint["ids"].shape:
            raise ValueError("invalid checkpoint targets")
        if bool(((y < 0) | (y >= classes)).any()):
            raise ValueError("checkpoint target outside class space")
        protected = torch.cat((train["gold_ids"], train["query_ids"]))
        if bool(torch.isin(checkpoint["ids"], protected).any()):
            raise ValueError("checkpoint IDs overlap train/query pools")


def _union(*parts: torch.Tensor) -> torch.Tensor:
    present = [part for part in parts if len(part)]
    if not present:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(present).unique().sort().values


def branch_pools(
    train: Mapping[str, torch.Tensor],
    branch: str,
    *,
    disagreement_all: bool,
    agreement_ce: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one merged CE pool and one merged KL pool for a branch."""

    _slurm_only()
    if branch not in {"g", "l"}:
        raise ValueError("branch must be g or l")
    dg = train["disagreement_choose_g_ids"]
    dl = train["disagreement_choose_l_ids"]
    agreement = train["agreement_ids"]
    disagreements = _union(dg, dl)
    ce_disagreement = disagreements if disagreement_all else (dl if branch == "g" else dg)
    kl_disagreement = disagreements if disagreement_all else (dg if branch == "g" else dl)
    ce = _union(ce_disagreement, agreement) if agreement_ce else _union(ce_disagreement)
    kl = _union(kl_disagreement, agreement)
    return ce, kl


def draw_pool(
    ids: torch.Tensor, mode: str, generator: torch.Generator, legacy_draws: int
) -> torch.Tensor:
    _slurm_only()
    if mode not in SAMPLING or legacy_draws < 1:
        raise ValueError("invalid pool sampling request")
    if len(ids) == 0:
        return ids
    if mode == "full_coverage":
        return ids[torch.randperm(len(ids), generator=generator)]
    draws = len(ids) if mode == "full_replacement" else int(legacy_draws)
    return ids[torch.randint(len(ids), (draws,), generator=generator)]


def _generators(seed: int, branch: str) -> dict[str, torch.Generator]:
    offset = 10_000 if branch == "g" else 20_000
    return {
        name: torch.Generator().manual_seed(int(seed) + offset + extra)
        for name, extra in (("gold", 11), ("ce", 22), ("kl", 33))
    }


def _pool_stats(ids: torch.Tensor) -> dict[str, int]:
    if len(ids) == 0:
        return {"seen": 0, "unique": 0, "max_count": 0}
    _, counts = ids.unique(return_counts=True)
    return {"seen": len(ids), "unique": len(counts), "max_count": int(counts.max())}


def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _head_logits(
    head: Head,
    train: Mapping[str, torch.Tensor],
    ids: torch.Tensor,
    branch: str,
    config: TrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    graph = train["graph"][ids].to(device=device, dtype=torch.float32)
    text = train["text"][ids].to(device=device, dtype=torch.float32)
    base = train[f"logp_{branch}"][ids].to(device=device, dtype=torch.float32)
    delta = head(graph, text)
    logits = base + delta if config.residual else delta
    return base, delta, logits


def _backward_ce(
    head: Head,
    train: Mapping[str, torch.Tensor],
    ids: torch.Tensor,
    targets: torch.Tensor,
    branch: str,
    config: TrainConfig,
    device: torch.device,
    weight: float,
) -> float:
    if len(ids) == 0 or weight == 0:
        return 0.0
    total = torch.zeros((), dtype=torch.float64)
    for start in range(0, len(ids), config.chunk_size):
        part = ids[start : start + config.chunk_size]
        y = targets[start : start + config.chunk_size].to(device)
        _, _, logits = _head_logits(head, train, part, branch, config, device)
        loss_sum = F.cross_entropy(logits, y, reduction="sum")
        (loss_sum * (weight / len(ids))).backward()
        total += loss_sum.detach().double().cpu()
    return float(total / len(ids))


def _backward_kl(
    head: Head,
    teacher: Head | None,
    train: Mapping[str, torch.Tensor],
    ids: torch.Tensor,
    branch: str,
    config: TrainConfig,
    device: torch.device,
) -> float:
    if len(ids) == 0 or config.keep_weight == 0:
        return 0.0
    total = torch.zeros((), dtype=torch.float64)
    for start in range(0, len(ids), config.chunk_size):
        part = ids[start : start + config.chunk_size]
        base, _, logits = _head_logits(head, train, part, branch, config, device)
        if teacher is None:
            teacher_probability = base.softmax(1).detach()
        else:
            with torch.no_grad():
                _, _, teacher_logits = _head_logits(teacher, train, part, branch, config, device)
                teacher_probability = teacher_logits.softmax(1).detach()
        loss_sum = F.kl_div(
            logits.log_softmax(1), teacher_probability, reduction="none"
        ).sum(1).sum()
        (loss_sum * (config.keep_weight / len(ids))).backward()
        total += loss_sum.detach().double().cpu()
    return float(total / len(ids))


def _score(probability: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    rows = torch.arange(len(targets))
    return {
        "accuracy": float((probability.argmax(1) == targets).double().mean()),
        "nll": float(-probability[rows, targets].double().clamp_min(1e-12).log().mean()),
    }


@torch.no_grad()
def _predict_head(
    head: Head,
    train: Mapping[str, torch.Tensor],
    ids: torch.Tensor,
    branch: str,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    head.eval()
    base_parts, delta_parts, logits_parts = [], [], []
    for start in range(0, len(ids), config.inference_batch):
        base, delta, logits = _head_logits(
            head, train, ids[start : start + config.inference_batch], branch, config, device
        )
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("nonfinite prediction logits")
        base_parts.append(base.cpu()); delta_parts.append(delta.cpu()); logits_parts.append(logits.cpu())
    classes = train["logp_g"].shape[1]
    empty = torch.empty((0, classes), dtype=torch.float32)
    base = torch.cat(base_parts) if base_parts else empty
    delta = torch.cat(delta_parts) if delta_parts else empty.clone()
    logits = torch.cat(logits_parts) if logits_parts else empty.clone()
    return {
        "base_logits": base,
        "delta_logits": delta,
        "logits": logits,
        "probabilities": logits.softmax(1),
    }


def predict_branch(
    state: Mapping[str, torch.Tensor],
    config: Mapping[str, object] | TrainConfig,
    train: Mapping[str, torch.Tensor],
    ids: torch.Tensor,
    branch: str,
    device: str,
) -> dict[str, torch.Tensor]:
    _slurm_only()
    cfg = TrainConfig.from_mapping(config)
    validate_inputs(train)
    _ids(ids, len(train["node_ids"]), "prediction ids")
    head = Head(
        train["graph"].shape[1], train["text"].shape[1], train["logp_g"].shape[1],
        cfg.architecture, cfg.width, cfg.dropout,
    ).to(device)
    head.load_state_dict(state, strict=True)
    return _predict_head(head, train, ids, branch, cfg, torch.device(device))


def predict(
    states: Mapping[str, Mapping[str, torch.Tensor]],
    config: Mapping[str, object] | TrainConfig,
    train: Mapping[str, torch.Tensor],
    ids: torch.Tensor,
    device: str,
) -> dict[str, dict[str, torch.Tensor]]:
    _slurm_only()
    if set(states) != {"g", "l"}:
        raise ValueError("prediction requires independent g/l states")
    return {
        branch: predict_branch(states[branch], config, train, ids, branch, device)
        for branch in ("g", "l")
    }


def train_member(
    train: Mapping[str, torch.Tensor],
    checkpoint: Mapping[str, torch.Tensor],
    config: Mapping[str, object] | TrainConfig,
    seed: int,
    device: str,
):
    """Train both branches for the full budget using checkpoint labels only."""

    _slurm_only()
    cfg = TrainConfig.from_mapping(config)
    validate_inputs(train, checkpoint)
    torch.set_num_threads(cfg.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(int(seed))
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    target_device = torch.device(device)
    model = DualResidualModel(
        train["graph"].shape[1], train["text"].shape[1], train["logp_g"].shape[1],
        cfg.architecture, cfg.width, cfg.dropout,
    ).to(target_device)
    heads = {"g": model.g, "l": model.l}
    optimizers = {
        branch: torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        for branch, head in heads.items()
    }
    generators = {branch: _generators(seed, branch) for branch in heads}
    teachers: dict[str, Head] = {}
    best: dict[str, dict[str, object]] = {}
    best_predictions: dict[str, torch.Tensor] = {}
    logs = []
    gold_lookup = torch.full((len(train["node_ids"]),), -1, dtype=torch.long)
    gold_lookup[train["gold_ids"]] = train["gold_targets"]

    for update in range(1, cfg.updates + 1):
        phase = "warm" if update <= cfg.warm_updates else "mixed"
        eligible = cfg.gold_only_control or phase == "mixed"
        branch_rows = {}
        for branch_index, branch in enumerate(("g", "l")):
            head = heads[branch]
            head.train()
            optimizer = optimizers[branch]
            optimizer.zero_grad(set_to_none=True)
            # A branch/update-specific dropout stream avoids cross-branch RNG coupling;
            # pool streams below remain continuous across the warm/mixed boundary.
            torch.manual_seed(int(seed) + (branch_index + 1) * 1_000_000 + update)
            rng = generators[branch]
            gold = draw_pool(train["gold_ids"], cfg.gold_sampling, rng["gold"], cfg.legacy_draws)
            ce_population, kl_population = branch_pools(
                train, branch,
                disagreement_all=cfg.disagreement_all,
                agreement_ce=cfg.agreement_ce,
            )
            ce = (
                draw_pool(ce_population, cfg.pseudo_sampling, rng["ce"], cfg.legacy_draws)
                if phase == "mixed" and cfg.pseudo_weight > 0 else ce_population[:0]
            )
            kl = (
                draw_pool(kl_population, cfg.pseudo_sampling, rng["kl"], cfg.legacy_draws)
                if phase == "mixed" and cfg.keep_weight > 0 else kl_population[:0]
            )
            gold_loss = _backward_ce(
                head, train, gold, gold_lookup[gold], branch, cfg, target_device, 1.0
            )
            ce_loss = _backward_ce(
                head, train, ce, train["selected_pred"][ce], branch, cfg, target_device,
                cfg.pseudo_weight,
            )
            kl_loss = _backward_kl(
                head, teachers.get(branch), train, kl, branch, cfg, target_device
            )
            if any(
                parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                for parameter in head.parameters()
            ):
                raise ValueError("nonfinite training gradient")
            if cfg.clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.clip_norm)
            optimizer.step()
            branch_rows[branch] = {
                "losses": {
                    "gold_ce": gold_loss,
                    "pseudo_ce": ce_loss,
                    "teacher_kl": kl_loss,
                    "total": gold_loss + cfg.pseudo_weight * ce_loss + cfg.keep_weight * kl_loss,
                },
                "pools": {"gold": _pool_stats(gold), "ce": _pool_stats(ce), "kl": _pool_stats(kl)},
            }

        if update == cfg.warm_updates and cfg.warm_updates > 0 and cfg.teacher == "warm":
            for branch in ("g", "l"):
                teacher = Head(
                    train["graph"].shape[1], train["text"].shape[1], train["logp_g"].shape[1],
                    cfg.architecture, cfg.width, cfg.dropout,
                ).to(target_device)
                teacher.load_state_dict(_cpu_state(heads[branch]), strict=True)
                teacher.eval()
                for parameter in teacher.parameters():
                    parameter.requires_grad_(False)
                teachers[branch] = teacher

        validation = None
        if update % cfg.validate_every == 0 or update == cfg.updates:
            validation = {}
            for branch in ("g", "l"):
                predicted = _predict_head(
                    heads[branch], train, checkpoint["ids"], branch, cfg, target_device
                )
                score = _score(predicted["probabilities"], checkpoint["targets"])
                validation[branch] = score
                key = (-score["accuracy"], score["nll"], update)
                if eligible and (branch not in best or key < best[branch]["key"]):
                    best[branch] = {
                        "key": key,
                        "update": update,
                        "accuracy": score["accuracy"],
                        "nll": score["nll"],
                        "state": _cpu_state(heads[branch]),
                    }
                    best_predictions[branch] = predicted["probabilities"].clone()
        logs.append({
            "update": update,
            "phase": phase,
            "eligible": eligible,
            "teacher": cfg.teacher if phase == "mixed" else None,
            "runtime": {
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            },
            "branches": branch_rows,
            "checkpoint": validation,
        })

    if set(best) != {"g", "l"}:
        raise RuntimeError("training produced no eligible checkpoint")
    checkpoints = {}
    for branch in ("g", "l"):
        value = dict(best[branch]); value.pop("key")
        checkpoints[branch] = value
    return {
        "config": asdict(cfg),
        "seed": int(seed),
        "checkpoints": checkpoints,
        "checkpoint_predictions": best_predictions,
        "checkpoint_ids": checkpoint["ids"].clone(),
        "logs": logs,
        "parameter_counts": {
            "g": sum(parameter.numel() for parameter in heads["g"].parameters()),
            "l": sum(parameter.numel() for parameter in heads["l"].parameters()),
            "total": sum(parameter.numel() for parameter in model.parameters()),
        },
        "test_labels_read": False,
    }

