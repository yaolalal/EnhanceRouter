from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F


@dataclass
class TrustRecord:
    node_id: int
    region: str
    pseudo_label: int | None
    label_source: str
    gnn_pred: int
    llm_pred: int
    gnn_conf: float
    llm_conf: float


def entropy_confidence(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(logits.size(-1), dtype=logits.dtype, device=logits.device)).clamp_min(1e-12)
    return 1.0 - entropy / max_entropy


def max_probability(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1).max(dim=-1).values


def margin_confidence(logits: torch.Tensor) -> torch.Tensor:
    top2 = F.softmax(logits, dim=-1).topk(k=min(2, logits.size(-1)), dim=-1).values
    if top2.size(-1) == 1:
        return torch.ones(logits.size(0), device=logits.device)
    return top2[:, 0] - top2[:, 1]


def compute_confidence(logits: torch.Tensor, metric: str = "entropy") -> torch.Tensor:
    metric = metric.lower()
    if metric == "entropy":
        return entropy_confidence(logits)
    if metric == "max_prob":
        return max_probability(logits)
    if metric == "margin":
        return margin_confidence(logits)
    raise ValueError(f"Unknown confidence metric: {metric}")


def partition_trust_regions(
    gnn_logits: torch.Tensor,
    llm_logits: torch.Tensor,
    candidate_idx: Iterable[int],
    theta_g: float,
    theta_l: float,
    confidence_metric: str = "entropy",
    agreement_tie_delta: float = 0.05,
) -> List[TrustRecord]:
    gnn_pred = gnn_logits.argmax(dim=-1)
    llm_pred = llm_logits.argmax(dim=-1)
    gnn_conf = compute_confidence(gnn_logits, confidence_metric)
    llm_conf = compute_confidence(llm_logits, confidence_metric)

    records: List[TrustRecord] = []
    for node_id in [int(i) for i in candidate_idx]:
        gh = bool(gnn_conf[node_id] >= theta_g)
        lh = bool(llm_conf[node_id] >= theta_l)
        gp = int(gnn_pred[node_id].item())
        lp = int(llm_pred[node_id].item())
        gc = float(gnn_conf[node_id].item())
        lc = float(llm_conf[node_id].item())
        pseudo_label = None
        source = "none"

        if gh and not lh:
            region = "gnn_trust"
            pseudo_label = gp
            source = "gnn"
        elif lh and not gh:
            region = "llm_trust"
            pseudo_label = lp
            source = "llm"
        elif gh and lh:
            region = "agreement" if gp == lp else "high_conflict"
            if gp == lp:
                pseudo_label = gp
                source = "consensus"
            elif abs(gc - lc) >= agreement_tie_delta:
                pseudo_label = gp if gc > lc else lp
                source = "gnn_high_conflict" if gc > lc else "llm_high_conflict"
            else:
                source = "uncertain_high_conflict"
        else:
            region = "uncertain"

        records.append(
            TrustRecord(
                node_id=node_id,
                region=region,
                pseudo_label=pseudo_label,
                label_source=source,
                gnn_pred=gp,
                llm_pred=lp,
                gnn_conf=gc,
                llm_conf=lc,
            )
        )
    return records


def build_pseudo_label_tensor(data, trust_records: List[TrustRecord]):
    labels = torch.full_like(data.y, fill_value=-1)
    sources: Dict[int, str] = {}
    train_ids = data.train_mask.nonzero().view(-1)
    labels[train_ids] = data.y[train_ids]
    for node_id in train_ids.detach().cpu().tolist():
        sources[int(node_id)] = "true"

    for rec in trust_records:
        if rec.pseudo_label is None:
            continue
        if sources.get(rec.node_id) == "true":
            continue
        labels[rec.node_id] = int(rec.pseudo_label)
        sources[rec.node_id] = rec.label_source
    idx = (labels >= 0).nonzero().view(-1)
    return labels, idx, sources


def summarize_regions(records: List[TrustRecord]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for rec in records:
        counts[rec.region] = counts.get(rec.region, 0) + 1
    counts["pseudo_labeled"] = sum(1 for rec in records if rec.pseudo_label is not None)
    return counts
