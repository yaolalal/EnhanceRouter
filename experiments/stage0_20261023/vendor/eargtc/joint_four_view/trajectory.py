"""Fixed-step, label-isolated residual-repair trajectory training."""
from __future__ import annotations

from dataclasses import asdict
import time

import torch
import torch.nn.functional as F

from eargtc.joint_cross_repair.artifacts import validate_features
from eargtc.joint_cross_repair.model import DualResidualRepair, pool_loss
from eargtc.joint_cross_repair.training import TrainConfig, _cpu_state


DEFAULT_SNAPSHOTS = (50, 150, 300, 500)
_POOL_KEYS = {"teach_g", "teach_l", "keep_g", "keep_l", "agreement", "unselected"}


def _ids(ids: torch.Tensor, n: int, name: str, *, nonempty: bool = False) -> None:
    if not torch.is_tensor(ids) or ids.dtype != torch.long or ids.ndim != 1:
        raise ValueError(f"{name} must be an int64 vector")
    if (nonempty and ids.numel() == 0) or ids.unique().numel() != ids.numel():
        raise ValueError(f"{name} must be nonempty and unique")
    if bool(((ids < 0) | (ids >= n)).any()):
        raise ValueError(f"{name} contains out-of-range IDs")


def _validate_inputs(features, gold_ids, gold_targets, pools, hidden, recipe, config, snapshots):
    recipe.validate()
    config.validate()
    validate_features(features)
    n, classes = features["logp_g"].shape
    _ids(gold_ids, n, "gold_ids", nonempty=True)
    if (
        not torch.is_tensor(gold_targets)
        or gold_targets.dtype != torch.long
        or gold_targets.shape != gold_ids.shape
        or bool(((gold_targets < 0) | (gold_targets >= classes)).any())
    ):
        raise ValueError("gold targets must align with gold IDs and class space")
    if not torch.is_tensor(hidden) or hidden.ndim != 2 or hidden.shape[0] != n:
        raise ValueError("hidden rows must align with features")
    if not bool(torch.isfinite(hidden).all()):
        raise ValueError("hidden features must be finite")
    if set(pools) != _POOL_KEYS:
        raise ValueError("invalid pool schema")
    for name, ids in pools.items():
        _ids(ids, n, name)
        if bool(torch.isin(ids, gold_ids).any()):
            raise ValueError("training pools must exclude gold IDs")

    agreement, choose_g = features["agreement"], features["choose_g"]
    if not bool((~agreement[pools["teach_g"]] & ~choose_g[pools["teach_g"]]).all()):
        raise ValueError("GNN repair must be taught by selected LLM conflicts")
    if not bool((~agreement[pools["teach_l"]] & choose_g[pools["teach_l"]]).all()):
        raise ValueError("LLM repair must be taught by selected GNN conflicts")
    if not bool(agreement[pools["agreement"]].all()):
        raise ValueError("agreement pool contains conflicts")
    for branch, other in (("g", "l"), ("l", "g")):
        expected = torch.cat((pools["teach_" + other], pools["agreement"])).unique().sort().values
        if not torch.equal(pools["keep_" + branch].sort().values, expected):
            raise ValueError("retention must protect only selected teacher and agreement")
    selected = torch.cat((pools["teach_g"], pools["teach_l"], pools["agreement"])).unique()
    if bool(torch.isin(pools["unselected"], selected).any()):
        raise ValueError("unselected pool overlaps selected repair pools")

    try:
        requested = tuple(snapshots)
    except TypeError as error:
        raise ValueError("snapshots must be an iterable of training steps") from error
    if not requested or any(not isinstance(step, int) or isinstance(step, bool) for step in requested):
        raise ValueError("snapshots must contain integer training steps")
    if tuple(sorted(requested)) != requested or len(set(requested)) != len(requested):
        raise ValueError("snapshots must be sorted and unique")
    if requested[0] < 1 or requested[-1] > config.max_steps:
        raise ValueError("snapshots must lie within configured training steps")
    return requested


def _sample(ids: torch.Tensor, generator: torch.Generator, batch_size: int) -> torch.Tensor:
    if not len(ids):
        return ids
    return ids[torch.randint(len(ids), (batch_size,), generator=generator)]


def train_trajectory(
    features,
    gold_ids: torch.Tensor,
    gold_targets: torch.Tensor,
    pools,
    hidden: torch.Tensor,
    recipe,
    config: TrainConfig,
    snapshots=DEFAULT_SNAPSHOTS,
):
    """Train two residual heads for every requested fixed step.

    The only labels accepted by this interface are ``gold_targets``. Pool
    direction and separation from gold are checked here; full split ownership
    remains the export runner's responsibility.
    """
    snapshots = _validate_inputs(
        features, gold_ids, gold_targets, pools, hidden, recipe, config, snapshots
    )
    _, classes = features["logp_g"].shape
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    if config.device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)
    model = DualResidualRepair(
        features["gnn_embeddings"].shape[1], hidden.shape[1], classes, config.hidden_dim
    ).to(config.device)
    modules = {"g": model.gnn_head, "l": model.llm_head}
    optimizers = {
        name: torch.optim.AdamW(module.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        for name, module in modules.items()
    }
    rng = {
        name: {
            pool: torch.Generator().manual_seed(config.seed + offset + extra)
            for pool, extra in (("gold", 0), ("cross", 1000), ("keep", 2000))
        }
        for name, offset in (("g", 101), ("l", 202))
    }

    with torch.no_grad():
        initial = gold_ids[: min(len(gold_ids), 8)]
        for branch in ("g", "l"):
            repair = model.repair_g if branch == "g" else model.repair_l
            logits = repair(
                features["gnn_embeddings"][initial].float().to(config.device),
                hidden[initial].float().to(config.device),
                features["logp_" + branch][initial].to(config.device),
            )
            if not torch.equal(logits.cpu(), features["logp_" + branch][initial]):
                raise ValueError("zero-initialized branches do not reproduce experts")

    states = {"g": {}, "l": {}}
    start = time.monotonic()
    for step in range(1, config.max_steps + 1):
        model.train()
        for branch in ("g", "l"):
            optimizer = optimizers[branch]
            optimizer.zero_grad(set_to_none=True)
            positions = torch.randint(
                len(gold_ids), (config.batch_size,), generator=rng[branch]["gold"]
            )
            gold = gold_ids[positions]
            teach = (
                _sample(pools["teach_" + branch], rng[branch]["cross"], config.batch_size)
                if recipe.cross
                else gold[:0]
            )
            keep = (
                _sample(pools["keep_" + branch], rng[branch]["keep"], config.batch_size)
                if recipe.keep
                else gold[:0]
            )
            idx = torch.cat((gold, teach, keep))
            gold_count, teach_count = len(gold), len(teach)
            repair = model.repair_g if branch == "g" else model.repair_l
            logits = repair(
                features["gnn_embeddings"][idx].float().to(config.device),
                hidden[idx].float().to(config.device),
                features["logp_" + branch][idx].to(config.device),
            )
            gold_loss = F.cross_entropy(logits[:gold_count], gold_targets[positions].to(config.device))
            other = "l" if branch == "g" else "g"
            cross_loss = pool_loss(
                logits[gold_count : gold_count + teach_count],
                features["logp_" + other][teach].argmax(1).to(config.device),
                "ce",
            )
            keep_loss = pool_loss(
                logits[gold_count + teach_count :],
                features["logp_" + branch][keep].exp().to(config.device),
                "keep",
            )
            loss = gold_loss + recipe.cross * cross_loss + recipe.keep * keep_loss
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite training loss")
            loss.backward()
            if any(
                parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                for parameter in modules[branch].parameters()
            ):
                raise ValueError("nonfinite repair gradients")
            optimizer.step()
        if step in snapshots:
            for branch, module in modules.items():
                states[branch][str(step)] = _cpu_state(module)

    return {
        "states": states,
        "recipe": asdict(recipe),
        "config": asdict(config),
        "steps": list(snapshots),
        "train_seconds": time.monotonic() - start,
        "validation_labels_read": False,
        "test_labels_read": False,
    }


@torch.no_grad()
def trajectory_logits(member, branch, step, features, hidden, ids, config: TrainConfig):
    """Restore one saved branch state and produce its repaired log probabilities."""
    config.validate()
    validate_features(features)
    n, classes = features["logp_g"].shape
    _ids(ids, n, "ids")
    if not torch.is_tensor(hidden) or hidden.ndim != 2 or hidden.shape[0] != n:
        raise ValueError("hidden rows must align with features")
    if branch not in {"g", "l"}:
        raise ValueError("branch must be g or l")
    key = str(step)
    try:
        state = member["states"][branch][key]
    except (KeyError, TypeError) as error:
        raise ValueError("requested trajectory snapshot is unavailable") from error
    model = DualResidualRepair(
        features["gnn_embeddings"].shape[1], hidden.shape[1], classes, config.hidden_dim
    ).to(config.device)
    module = model.gnn_head if branch == "g" else model.llm_head
    module.load_state_dict(state)
    model.eval()
    output = []
    repair = model.repair_g if branch == "g" else model.repair_l
    for start in range(0, len(ids), config.inference_batch):
        part = ids[start : start + config.inference_batch]
        logits = repair(
            features["gnn_embeddings"][part].float().to(config.device),
            hidden[part].float().to(config.device),
            features["logp_" + branch][part].to(config.device),
        )
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("nonfinite repaired logits")
        output.append(logits.cpu())
    return torch.cat(output)
