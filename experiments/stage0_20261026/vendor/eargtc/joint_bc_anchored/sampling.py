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
            self.pool.numel(),
            (self.batch_size,),
            generator=self.generator,
        )
        return self.pool[positions]


class StratifiedReplacementPoolSampler:
    """Sample strata uniformly, then sample a node within each stratum.

    Strata are visited in independently shuffled complete cycles.  This keeps
    even batch-size-one training exactly balanced over every full cycle while
    retaining replacement sampling within each class/route group.
    """

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
        if not self.pool.numel():
            raise ValueError("stratified replacement sampler pool cannot be empty")
        if self.groups.numel() != self.pool.numel():
            raise ValueError("stratified sampler requires one group per pool row")
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
        self._group_order = torch.empty(0, dtype=torch.long)
        self._group_cursor = 0

    def _next_group(self) -> int:
        if self._group_cursor >= self._group_order.numel():
            permutation = torch.randperm(
                self.unique_groups.numel(), generator=self.generator
            )
            self._group_order = self.unique_groups[permutation]
            self._group_cursor = 0
        group = int(self._group_order[self._group_cursor])
        self._group_cursor += 1
        return group

    def next_batch(self) -> torch.Tensor:
        result = []
        for _ in range(self.batch_size):
            members = self.members[self._next_group()]
            position = torch.randint(
                members.numel(), (1,), generator=self.generator
            )
            result.append(members[position])
        return torch.cat(result)
