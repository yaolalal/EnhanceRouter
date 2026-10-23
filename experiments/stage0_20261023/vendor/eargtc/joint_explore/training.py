"""Label-isolated trainer operating only on validated frozen caches."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from eargtc.joint_cross_repair.artifacts import validate_features, validate_pools, validate_supervision
from eargtc.joint_cross_repair.training import scheme_b_probability
from .model import ARCHITECTURES, JointClassifier


PRESETS = {
    "gold": dict(pseudo_weight=0., keep_weight=0., alignment_weight=0., width=128, dropout=.3, lr=.001),
    "low": dict(pseudo_weight=.25, keep_weight=.1, alignment_weight=0., width=128, dropout=.3, lr=.001),
    "medium": dict(pseudo_weight=1., keep_weight=.1, alignment_weight=0., width=128, dropout=.3, lr=.001),
    "high": dict(pseudo_weight=4., keep_weight=.1, alignment_weight=0., width=128, dropout=.3, lr=.001),
    "aligned": dict(pseudo_weight=1., keep_weight=.1, alignment_weight=.05, width=128, dropout=.3, lr=.001),
    "wide": dict(pseudo_weight=1., keep_weight=.5, alignment_weight=0., width=256, dropout=.5, lr=.0005),
}


@dataclass(frozen=True)
class TrainSettings:
    architecture: str = "balanced_residual"
    preset: str = "medium"
    width: int = 128
    dropout: float = .3
    pseudo_weight: float = 1.
    keep_weight: float = .1
    alignment_weight: float = 0.
    lr: float = .001
    weight_decay: float = .001
    seed: int = 42
    device: str = "cuda"
    max_steps: int = 1000
    batch_size: int = 128
    validate_every: int = 50
    patience: int = 6
    inference_batch: int = 8192
    threads: int = 2

    def validate(self):
        if self.architecture not in ARCHITECTURES or self.preset not in PRESETS:
            raise ValueError("unknown architecture or preset")
        for key in ("width", "max_steps", "batch_size", "validate_every", "patience", "inference_batch", "threads"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.architecture == "attention_direct" and self.width % 4:
            raise ValueError("attention width must be divisible by four")
        if not isinstance(self.seed, int) or not isinstance(self.device, str) or not self.device:
            raise ValueError("invalid seed or device")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("invalid dropout")
        for key in ("pseudo_weight", "keep_weight", "alignment_weight", "weight_decay"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {key}")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("invalid lr")
        return self


def make_settings(architecture, preset, seed=42, **overrides):
    if architecture not in ARCHITECTURES or preset not in PRESETS:
        raise ValueError("unknown architecture or preset")
    values = {**PRESETS[preset], **overrides}
    settings = replace(TrainSettings(), architecture=architecture, preset=preset, seed=seed, **values)
    return settings.validate()


def _configure(settings):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.set_num_threads(settings.threads)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _probability(logits, architecture):
    return logits.exp() if architecture == "gated_experts" else logits.softmax(1)


def _model_from_member(member, f, h, device):
    cfg = TrainSettings(**member["settings"]).validate()
    return JointClassifier(f["gnn_embeddings"].shape[1], h.shape[1], f["logp_g"].shape[1],
                           cfg.architecture, cfg.width, cfg.dropout).to(device), cfg


@torch.no_grad()
def predict_candidate(member, f, h, ids, criterion="accuracy", device="cuda", batch_size=8192):
    if criterion not in ("accuracy", "nll") or criterion not in member["states"]:
        raise ValueError("unknown checkpoint criterion")
    if ids.dtype != torch.long or ids.ndim != 1 or batch_size < 1:
        raise ValueError("invalid prediction IDs or batch size")
    cfg = TrainSettings(**member["settings"]).validate()
    _configure(replace(cfg, device=device))
    model, cfg = _model_from_member(member, f, h, device)
    model.load_state_dict(member["states"][criterion], strict=True)
    model.eval()
    pg0, pl0 = f["logp_g"].detach().exp(), f["logp_l"].detach().exp()
    pb0 = scheme_b_probability(pg0, pl0, f["choose_g"], f["agreement"])
    output = []
    for start in range(0, len(ids), batch_size):
        part = ids[start:start + batch_size]
        logits = model(f["gnn_embeddings"][part].to(device), h[part].to(device),
                       pg0[part].to(device), pl0[part].to(device), pb0[part].to(device))
        probability = _probability(logits, cfg.architecture)
        if not bool(torch.isfinite(probability).all()):
            raise ValueError("nonfinite prediction")
        output.append(probability.cpu())
    return torch.cat(output) if output else torch.empty((0, f["logp_g"].shape[1]))


def train_candidate(f, s, p, h, settings):
    settings.validate()
    validate_features(f)
    n, c = f["logp_g"].shape
    validate_supervision(s, n, c)
    validate_pools(p, f, s)
    if not torch.is_tensor(h) or h.ndim != 2 or h.shape[0] != n:
        raise ValueError("hidden rows must align with features")
    for start in range(0, n, 4096):
        if not bool(torch.isfinite(h[start:start + 4096]).all()):
            raise ValueError("nonfinite hidden input")
    accepted = torch.cat((p["teach_g"], p["teach_l"]))
    keep = p["unselected"]
    if accepted.unique().numel() != accepted.numel():
        raise ValueError("accepted pools overlap")
    training_ids = {"gold": s["gold_ids"].clone(), "pseudo": accepted.clone(), "keep": keep.clone()}
    if any(bool(torch.isin(s["selection_ids"], ids).any()) for ids in training_ids.values()):
        raise ValueError("validation IDs appear in training pools")
    _configure(settings)
    device = torch.device(settings.device)
    model = JointClassifier(f["gnn_embeddings"].shape[1], h.shape[1], c, settings.architecture,
                            settings.width, settings.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    pg0, pl0 = f["logp_g"].detach().exp(), f["logp_l"].detach().exp()
    pb0 = scheme_b_probability(pg0, pl0, f["choose_g"], f["agreement"])
    generators = {name: torch.Generator().manual_seed(settings.seed + offset)
                  for name, offset in (("gold", 101), ("pseudo", 202), ("keep", 303), ("align", 404))}
    best, states, stale, history = {}, {}, 0, []
    started = time.monotonic()

    def sample(ids, name):
        if len(ids) == 0:
            return ids
        return ids[torch.randint(len(ids), (settings.batch_size,), generator=generators[name])]

    def forward(ids):
        return model(f["gnn_embeddings"][ids].to(device), h[ids].to(device), pg0[ids].to(device),
                     pl0[ids].to(device), pb0[ids].to(device))

    # This is the sole function that accesses selection_targets, and gradients are disabled.
    @torch.no_grad()
    def validate(step, losses):
        nonlocal stale
        model.eval()
        probability_parts = []
        ids = s["selection_ids"]
        for start in range(0, len(ids), settings.inference_batch):
            probability_parts.append(_probability(forward(ids[start:start + settings.inference_batch]), settings.architecture).cpu())
        probability = torch.cat(probability_parts)
        targets = s["selection_targets"]
        accuracy = float((probability.argmax(1) == targets).double().mean())
        nll = float(-probability[torch.arange(len(targets)), targets].double().clamp_min(1e-12).log().mean())
        row = {"step": step, "accuracy": accuracy, "nll": nll, "losses": losses}
        history.append(row)
        changed_accuracy = "accuracy" not in best or (-accuracy, nll, step) < (-best["accuracy"][0], best["accuracy"][1], best["accuracy"][2])
        changed_nll = "nll" not in best or (nll, -accuracy, step) < (best["nll"][1], -best["nll"][0], best["nll"][2])
        if changed_accuracy:
            best["accuracy"] = (accuracy, nll, step); states["accuracy"] = _cpu_state(model); stale = 0
        else:
            stale += 1
        if changed_nll:
            best["nll"] = (accuracy, nll, step); states["nll"] = _cpu_state(model)

    validate(0, {})
    for step in range(1, settings.max_steps + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); pieces = {}
        gold_positions = torch.randint(len(s["gold_ids"]), (settings.batch_size,), generator=generators["gold"])
        gold = s["gold_ids"][gold_positions]
        gold_loss = F.cross_entropy(forward(gold), s["gold_targets"][gold_positions].to(device))
        pieces["gold"] = gold_loss
        loss = pieces["gold"]
        if settings.pseudo_weight and len(accepted):
            pseudo = sample(accepted, "pseudo")
            per_node = F.cross_entropy(forward(pseudo), f["selected_pred"][pseudo].to(device), reduction="none")
            trust = f["trust"][pseudo].detach().to(device).clamp(.05, 1.)
            pieces["pseudo"] = (per_node * trust).sum() / trust.sum()
            loss = loss + settings.pseudo_weight * pieces["pseudo"]
        if settings.keep_weight and len(keep):
            keep_ids = sample(keep, "keep")
            pieces["keep"] = F.kl_div(_probability(forward(keep_ids), settings.architecture).clamp_min(1e-12).log(),
                                      pb0[keep_ids].detach().to(device), reduction="batchmean")
            loss = loss + settings.keep_weight * pieces["keep"]
        if settings.alignment_weight:
            align_ids = sample(f["node_ids"], "align")
            pieces["alignment"] = model.alignment_loss(f["gnn_embeddings"][align_ids].to(device), h[align_ids].to(device))
            loss = loss + settings.alignment_weight * pieces["alignment"]
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite training loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
        if step % settings.validate_every == 0 or step == settings.max_steps:
            validate(step, {key: float(value.detach()) for key, value in pieces.items()} | {"total": float(loss.detach())})
            if stale >= settings.patience:
                break
    member = {"settings": asdict(settings), "states": states}
    validation_probability = {criterion: predict_candidate(member, f, h, s["selection_ids"], criterion,
                                                              settings.device, settings.inference_batch)
                              for criterion in ("accuracy", "nll")}
    return {**member, "validation_probability": validation_probability,
            "validation_ids": s["selection_ids"].clone(), "history": history,
            "train_seconds": float(time.monotonic() - started), "test_labels_read": False,
            "training_ids": training_ids, "alignment_population": "all_node_ids_label_free"}
