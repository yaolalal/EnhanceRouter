from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ReliabilityDomain:
    dataset: str
    module: str
    node_ids: torch.Tensor
    evidence: torch.Tensor
    feature_names: list[str]
    target: torch.Tensor
    choose_g: torch.Tensor
    b_score: torch.Tensor
    state: list[str]

    def __post_init__(self) -> None:
        ids = torch.as_tensor(self.node_ids, dtype=torch.long).view(-1)
        evidence = torch.as_tensor(self.evidence, dtype=torch.float64)
        target = torch.as_tensor(self.target, dtype=torch.bool).view(-1)
        choose_g = torch.as_tensor(self.choose_g, dtype=torch.bool).view(-1)
        score = torch.as_tensor(self.b_score, dtype=torch.float64).view(-1)
        if self.module not in {"conflict", "agreement"}:
            raise ValueError(f"unknown reliability module: {self.module}")
        if evidence.ndim != 2 or evidence.size(1) != len(self.feature_names):
            raise ValueError("evidence shape and feature names differ")
        if not (
            ids.numel()
            == evidence.size(0)
            == target.numel()
            == choose_g.numel()
            == score.numel()
            == len(self.state)
        ):
            raise ValueError("reliability domain row counts differ")
        if ids.numel() != torch.unique(ids).numel():
            raise ValueError("duplicate node IDs in reliability domain")
        if not bool(torch.isfinite(evidence).all()) or not bool(
            torch.isfinite(score).all()
        ):
            raise ValueError("reliability domain contains non-finite values")
        object.__setattr__(self, "node_ids", ids)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "choose_g", choose_g)
        object.__setattr__(self, "b_score", score)


def selected_correctness(
    labels: torch.Tensor,
    gnn_pred: torch.Tensor,
    llm_pred: torch.Tensor,
    choose_g: torch.Tensor,
) -> torch.Tensor:
    y = torch.as_tensor(labels).view(-1)
    g = torch.as_tensor(gnn_pred).view(-1)
    l = torch.as_tensor(llm_pred).view(-1)
    route = torch.as_tensor(choose_g, dtype=torch.bool).view(-1)
    if not (y.numel() == g.numel() == l.numel() == route.numel()):
        raise ValueError("selected correctness inputs must have equal row counts")
    return torch.where(route, g, l) == y


def partition_population(
    *,
    dataset: str,
    node_ids: torch.Tensor,
    evidence: torch.Tensor,
    feature_names: list[str],
    labels: torch.Tensor,
    gnn_pred: torch.Tensor,
    llm_pred: torch.Tensor,
    choose_g: torch.Tensor,
    b_score: torch.Tensor,
) -> dict[str, ReliabilityDomain]:
    ids = torch.as_tensor(node_ids, dtype=torch.long).view(-1)
    x = torch.as_tensor(evidence, dtype=torch.float64)
    y = torch.as_tensor(labels).view(-1)
    g = torch.as_tensor(gnn_pred).view(-1)
    l = torch.as_tensor(llm_pred).view(-1)
    route = torch.as_tensor(choose_g, dtype=torch.bool).view(-1)
    score = torch.as_tensor(b_score, dtype=torch.float64).view(-1)
    if x.ndim != 2 or x.size(1) != len(feature_names):
        raise ValueError("evidence shape and feature names differ")
    if not (
        ids.numel()
        == x.size(0)
        == y.numel()
        == g.numel()
        == l.numel()
        == route.numel()
        == score.numel()
    ):
        raise ValueError("population inputs must have equal row counts")
    if ids.numel() != torch.unique(ids).numel():
        raise ValueError("duplicate node IDs")
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(score).all()):
        raise ValueError("population inputs contain non-finite values")

    conflict = g != l
    target = selected_correctness(y, g, l, route)
    states = []
    for index in range(ids.numel()):
        if bool(conflict[index]):
            if bool(g[index] == y[index]):
                states.append("gnn_only")
            elif bool(l[index] == y[index]):
                states.append("llm_only")
            else:
                states.append("neither_correct")
        else:
            states.append(
                "both_correct" if bool(g[index] == y[index]) else "both_wrong"
            )

    result = {}
    for module, mask in (("conflict", conflict), ("agreement", ~conflict)):
        positions = mask.nonzero().view(-1)
        result[module] = ReliabilityDomain(
            dataset=str(dataset),
            module=module,
            node_ids=ids[positions],
            evidence=x[positions],
            feature_names=list(feature_names),
            target=target[positions],
            choose_g=route[positions],
            b_score=score[positions],
            state=[states[int(position)] for position in positions.tolist()],
        )
    return result
