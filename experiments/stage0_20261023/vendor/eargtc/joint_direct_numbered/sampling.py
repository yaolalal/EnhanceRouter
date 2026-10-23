from __future__ import annotations

import torch


class ReplacementPoolSampler:
    def __init__(self, pool: torch.Tensor, *, batch_size: int, seed: int) -> None:
        self.pool = torch.as_tensor(pool, dtype=torch.long).view(-1).clone()
        if not self.pool.numel():
            raise ValueError("replacement sampler pool cannot be empty")
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))

    def next_batch(self) -> torch.Tensor:
        positions = torch.randint(
            self.pool.numel(), (self.batch_size,), generator=self.generator
        )
        return self.pool[positions]


class StratifiedReplacementPoolSampler:
    def __init__(
        self,
        pool: torch.Tensor,
        groups: torch.Tensor,
        *,
        batch_size: int,
        seed: int,
    ) -> None:
        self.pool = torch.as_tensor(pool, dtype=torch.long).view(-1).clone()
        self.groups = torch.as_tensor(groups, dtype=torch.long).view(-1).clone()
        if not self.pool.numel() or self.groups.numel() != self.pool.numel():
            raise ValueError("stratified sampler requires a non-empty group per node")
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        self.unique_groups = self.groups.unique(sorted=True)
        self.members = {
            int(group): self.pool[self.groups == group]
            for group in self.unique_groups.tolist()
        }
        self._order = torch.empty(0, dtype=torch.long)
        self._cursor = 0

    def _next_group(self) -> int:
        if self._cursor >= self._order.numel():
            self._order = self.unique_groups[
                torch.randperm(self.unique_groups.numel(), generator=self.generator)
            ]
            self._cursor = 0
        group = int(self._order[self._cursor])
        self._cursor += 1
        return group

    def next_batch(self) -> torch.Tensor:
        result = []
        for _ in range(self.batch_size):
            members = self.members[self._next_group()]
            position = torch.randint(members.numel(), (1,), generator=self.generator)
            result.append(members[position])
        return torch.cat(result)

