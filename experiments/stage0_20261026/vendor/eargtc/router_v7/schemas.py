from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class EnvironmentSpec:
    environment_id: str
    channel: str
    family: str
    severity: float
    seed: int
    role: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.environment_id).strip():
            raise ValueError("environment_id must be non-empty")
        if self.channel not in {"clean", "graph", "text", "mixed"}:
            raise ValueError(f"invalid environment channel: {self.channel}")
        if self.role not in {"train", "validation", "test"}:
            raise ValueError(f"invalid environment role: {self.role}")
        if not 0.0 <= float(self.severity) <= 1.0:
            raise ValueError("environment severity must be in [0, 1]")
        if int(self.seed) < 0:
            raise ValueError("environment seed must be non-negative")
        object.__setattr__(self, "environment_id", str(self.environment_id))
        object.__setattr__(self, "family", str(self.family))
        object.__setattr__(self, "severity", float(self.severity))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "parameters", dict(self.parameters))


@dataclass(frozen=True)
class CounterfactualRow:
    dataset: str
    node_id: int
    fold_id: int
    environment: EnvironmentSpec
    local_features: tuple[float, ...]
    context_features: tuple[float, ...]
    gnn_prob: tuple[float, ...]
    llm_prob: tuple[float, ...]
    true_label: int | None = None

    @property
    def key(self) -> tuple[str, int, str]:
        return self.dataset, int(self.node_id), self.environment.environment_id
