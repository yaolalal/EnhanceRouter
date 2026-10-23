from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch

from .audit import assert_fit_ids_exclude_test
from .cache import validate_hidden_shard
from .data import BaseDatasetBundle
from .projection import (
    FittedProjection,
    ProjectionSpec,
    fit_llm_anchored_projection,
    fit_projection,
    mixed_embedding_features,
)


@dataclass(frozen=True)
class DatasetFeatureBank:
    dataset: str
    full_logit_evidence: torch.Tensor
    support_logit_evidence: torch.Tensor
    support_node_ids: torch.Tensor
    projected: dict[str, tuple[torch.Tensor, torch.Tensor]]
    projections: dict[str, FittedProjection]
    projection_fit_ids: torch.Tensor
    train_ids: torch.Tensor
    val_ids: torch.Tensor
    test_ids: torch.Tensor

    def feature_keys(self) -> list[str]:
        keys = ["A"]
        for family, (zg, _) in self.projected.items():
            for dim in (16, 32, 64):
                if dim <= zg.shape[1]:
                    keys.extend([f"B_{family}_d{dim}", f"C_{family}_d{dim}"])
        return keys

    def _embedding_feature(
        self, family: str, dim: int, node_ids: torch.Tensor
    ) -> torch.Tensor:
        zg, zl = self.projected[family]
        ids = torch.as_tensor(node_ids, dtype=torch.long).view(-1)
        return mixed_embedding_features(zg[ids, :dim], zl[ids, :dim])

    def full_feature(self, key: str, node_ids: torch.Tensor) -> torch.Tensor:
        ids = torch.as_tensor(node_ids, dtype=torch.long).view(-1)
        if key == "A":
            return self.full_logit_evidence[ids]
        family_type, family, dim_text = key.split("_")
        dim = int(dim_text.removeprefix("d"))
        embedding = self._embedding_feature(family, dim, ids)
        if family_type == "B":
            return embedding
        if family_type == "C":
            return torch.cat([self.full_logit_evidence[ids], embedding], dim=1)
        raise ValueError(f"unknown feature key: {key}")

    def support_feature(
        self, key: str, support_local_ids: torch.Tensor
    ) -> torch.Tensor:
        local = torch.as_tensor(support_local_ids, dtype=torch.long).view(-1)
        if key == "A":
            return self.support_logit_evidence[local]
        family_type, family, dim_text = key.split("_")
        dim = int(dim_text.removeprefix("d"))
        global_ids = self.support_node_ids[local]
        embedding = self._embedding_feature(family, dim, global_ids)
        if family_type == "B":
            return embedding
        if family_type == "C":
            return torch.cat(
                [self.support_logit_evidence[local], embedding], dim=1
            )
        raise ValueError(f"unknown feature key: {key}")


def _projection_fit_ids(
    bundle: BaseDatasetBundle, *, maximum: int, seed: int
) -> torch.Tensor:
    eligible = torch.cat([bundle.train_ids, bundle.val_ids]).unique(sorted=True)
    assert_fit_ids_exclude_test(eligible, bundle.test_ids)
    if eligible.numel() <= int(maximum):
        return eligible
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(eligible.numel(), generator=generator)[: int(maximum)]
    return eligible[order].sort().values


def _transform_batched(
    projection: FittedProjection,
    gnn: torch.Tensor,
    llm: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    zg_rows, zl_rows = [], []
    for start in range(0, gnn.shape[0], int(batch_size)):
        end = min(start + int(batch_size), gnn.shape[0])
        zg, zl = projection.transform(gnn[start:end], llm[start:end])
        zg_rows.append(zg)
        zl_rows.append(zl)
    return torch.cat(zg_rows).contiguous(), torch.cat(zl_rows).contiguous()


def build_dataset_feature_bank(
    bundle: BaseDatasetBundle,
    *,
    hidden_path: str | Path,
    seed: int = 42,
    max_projection_rows: int = 512,
    transform_batch_size: int = 4096,
) -> DatasetFeatureBank:
    hidden_shard = validate_hidden_shard(hidden_path)
    if hidden_shard.manifest.dataset != bundle.dataset:
        raise ValueError("LLM hidden dataset does not match frozen expert bundle")
    expected_ids = torch.arange(bundle.labels.numel(), dtype=torch.long)
    if not torch.equal(hidden_shard.node_ids, expected_ids):
        raise ValueError("LLM hidden cache does not cover all node IDs in order")
    llm = hidden_shard.hidden
    fit_ids = _projection_fit_ids(
        bundle, maximum=max_projection_rows, seed=seed
    )
    local_ids = torch.arange(fit_ids.numel(), dtype=torch.long)
    empty_test = torch.empty(0, dtype=torch.long)
    fit_g = bundle.gnn_embeddings[fit_ids]
    fit_l = llm[fit_ids]
    projected = {}
    projections = {}
    for family in ("cca", "pca", "random"):
        fitted = fit_projection(
            ProjectionSpec(family=family, dim=64, pca_dim=64, ridge=1e-3),
            fit_g,
            fit_l,
            local_ids,
            empty_test,
            seed=seed,
        )
        fitted = replace(fitted, fit_ids=fit_ids)
        zg, zl = _transform_batched(
            fitted,
            bundle.gnn_embeddings,
            llm,
            batch_size=transform_batch_size,
        )
        projected[family] = (zg, zl)
        projections[family] = fitted
    anchored = fit_llm_anchored_projection(
        bundle.gnn_embeddings,
        llm,
        fit_ids,
        bundle.test_ids,
        dim=64,
        ridge=1.0,
        seed=seed,
    )
    zg, zl = _transform_batched(
        anchored,
        bundle.gnn_embeddings,
        llm,
        batch_size=transform_batch_size,
    )
    projected["anchor"] = (zg, zl)
    projections["anchor"] = anchored
    return DatasetFeatureBank(
        dataset=bundle.dataset,
        full_logit_evidence=bundle.logit_evidence,
        support_logit_evidence=bundle.support_logit_evidence,
        support_node_ids=bundle.support_node_ids,
        projected=projected,
        projections=projections,
        projection_fit_ids=fit_ids,
        train_ids=bundle.train_ids,
        val_ids=bundle.val_ids,
        test_ids=bundle.test_ids,
    )


__all__ = [
    "DatasetFeatureBank",
    "build_dataset_feature_bank",
]
