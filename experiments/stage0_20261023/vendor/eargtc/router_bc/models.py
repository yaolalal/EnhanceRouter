from __future__ import annotations

from dataclasses import dataclass

import torch

from .evaluation import factorized_probabilities
from .logistic import LogisticModel


@dataclass(frozen=True)
class DirectPreferenceRouter:
    preference: LogisticModel

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """P(GNN wins | exactly one expert correct, conflict); not correctness."""
        return self.preference.probabilities(x)


@dataclass(frozen=True)
class FactorizedOutcomeRouter:
    availability: LogisticModel
    preference: LogisticModel
    agreement_correctness: LogisticModel

    def conflict_probabilities(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return factorized_probabilities(self.availability.probabilities(x), self.preference.probabilities(x))

    def agreement_probability(self, x: torch.Tensor) -> torch.Tensor:
        return self.agreement_correctness.probabilities(x)

