from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .audit import assert_fit_ids_exclude_test


@dataclass(frozen=True)
class ProjectionSpec:
    family: str
    dim: int
    pca_dim: int = 256
    ridge: float = 1e-3

    def __post_init__(self) -> None:
        if self.family not in {"cca", "pca", "random", "llm_anchor"}:
            raise ValueError(f"unknown projection family: {self.family}")
        if self.dim <= 0 or self.pca_dim <= 0 or self.ridge <= 0:
            raise ValueError("projection dimensions and ridge must be positive")


@dataclass(frozen=True)
class FittedProjection:
    spec: ProjectionSpec
    g_center: torch.Tensor
    g_scale: torch.Tensor
    l_center: torch.Tensor
    l_scale: torch.Tensor
    g_weight: torch.Tensor
    l_weight: torch.Tensor
    canonical_correlations: torch.Tensor
    effective_dim: int
    fit_ids: torch.Tensor
    seed: int
    g_bias: torch.Tensor | None = None

    def transform(
        self, h_g: torch.Tensor, h_l: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.as_tensor(h_g, dtype=torch.float32)
        l = torch.as_tensor(h_l, dtype=torch.float32)
        if g.ndim != 2 or l.ndim != 2 or g.shape[0] != l.shape[0]:
            raise ValueError("paired embeddings must be aligned matrices")
        if g.shape[1] != self.g_center.numel() or l.shape[1] != self.l_center.numel():
            raise ValueError("embedding widths differ from fitted projection")
        zg = ((g - self.g_center) / self.g_scale) @ self.g_weight
        zl = ((l - self.l_center) / self.l_scale) @ self.l_weight
        if self.g_bias is not None:
            zg = zg + self.g_bias
        if not bool(torch.isfinite(zg).all().item()) or not bool(
            torch.isfinite(zl).all().item()
        ):
            raise ValueError("projection produced non-finite values")
        return zg.contiguous(), zl.contiguous()


def _robust_location_scale(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    center = x.median(dim=0).values
    q25 = torch.quantile(x, 0.25, dim=0)
    q75 = torch.quantile(x, 0.75, dim=0)
    scale = (q75 - q25).clamp_min(1e-4)
    return center, scale


def _pca_basis(x: torch.Tensor, width: int) -> torch.Tensor:
    _, singular, vh = torch.linalg.svd(x, full_matrices=False)
    tolerance = torch.finfo(x.dtype).eps * max(x.shape) * singular.max().clamp_min(1.0)
    rank = max(int((singular > tolerance).sum().item()), 1)
    return vh[: min(int(width), rank)].T.contiguous()


def _inverse_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(matrix)
    values = values.clamp_min(torch.finfo(matrix.dtype).eps)
    return (vectors * values.rsqrt().view(1, -1)) @ vectors.T


def fit_projection(
    spec: ProjectionSpec,
    h_g: torch.Tensor,
    h_l: torch.Tensor,
    fit_ids: torch.Tensor,
    test_ids: torch.Tensor,
    seed: int,
) -> FittedProjection:
    g = torch.as_tensor(h_g, dtype=torch.float64).cpu()
    l = torch.as_tensor(h_l, dtype=torch.float64).cpu()
    ids = torch.as_tensor(fit_ids, dtype=torch.long).view(-1).cpu()
    tests = torch.as_tensor(test_ids, dtype=torch.long).view(-1).cpu()
    if g.ndim != 2 or l.ndim != 2 or g.shape[0] != l.shape[0]:
        raise ValueError("paired embeddings must be aligned matrices")
    assert_fit_ids_exclude_test(ids, tests)
    if ids.numel() < 3 or ids.numel() != torch.unique(ids).numel():
        raise ValueError("projection needs at least three unique fit IDs")
    if bool(((ids < 0) | (ids >= g.shape[0])).any().item()):
        raise ValueError("projection fit IDs out of range")
    if not bool(torch.isfinite(g).all().item()) or not bool(torch.isfinite(l).all().item()):
        raise ValueError("embeddings contain non-finite values")
    assert_fit_ids_exclude_test(ids, tests)

    g_center, g_scale = _robust_location_scale(g[ids])
    l_center, l_scale = _robust_location_scale(l[ids])
    gx = (g[ids] - g_center) / g_scale
    lx = (l[ids] - l_center) / l_scale
    maximum = min(int(spec.dim), ids.numel() - 1, g.shape[1], l.shape[1])
    if maximum < 1:
        raise ValueError("projection has no available components")

    correlations = torch.empty(0, dtype=torch.float64)
    if spec.family == "random":
        generator = torch.Generator().manual_seed(int(seed))
        g_raw = torch.randn(g.shape[1], maximum, generator=generator, dtype=torch.float64)
        l_raw = torch.randn(l.shape[1], maximum, generator=generator, dtype=torch.float64)
        g_weight = torch.linalg.qr(g_raw, mode="reduced").Q
        l_weight = torch.linalg.qr(l_raw, mode="reduced").Q
    else:
        pre_width = min(
            max(int(spec.pca_dim), maximum),
            ids.numel() - 1,
            g.shape[1],
            l.shape[1],
        )
        g_pca = _pca_basis(gx, pre_width)
        l_pca = _pca_basis(lx, pre_width)
        if spec.family == "pca":
            maximum = min(maximum, g_pca.shape[1], l_pca.shape[1])
            g_weight = g_pca[:, :maximum]
            l_weight = l_pca[:, :maximum]
        else:
            gr = gx @ g_pca
            lr = lx @ l_pca
            denominator = max(ids.numel() - 1, 1)
            cgg = gr.T @ gr / denominator
            cll = lr.T @ lr / denominator
            cgl = gr.T @ lr / denominator
            cgg = cgg + float(spec.ridge) * torch.eye(cgg.shape[0], dtype=cgg.dtype)
            cll = cll + float(spec.ridge) * torch.eye(cll.shape[0], dtype=cll.dtype)
            wg = _inverse_sqrt(cgg)
            wl = _inverse_sqrt(cll)
            left, singular, right_t = torch.linalg.svd(wg @ cgl @ wl)
            maximum = min(maximum, left.shape[1], right_t.shape[0])
            g_weight = g_pca @ wg @ left[:, :maximum]
            l_weight = l_pca @ wl @ right_t.T[:, :maximum]
            correlations = singular[:maximum]

    fit_zg = gx @ g_weight
    fit_zl = lx @ l_weight
    cross = (fit_zg * fit_zl).sum(dim=0)
    signs = torch.where(cross < 0, -torch.ones_like(cross), torch.ones_like(cross))
    l_weight = l_weight * signs.view(1, -1)
    return FittedProjection(
        spec=spec,
        g_center=g_center.float(),
        g_scale=g_scale.float(),
        l_center=l_center.float(),
        l_scale=l_scale.float(),
        g_weight=g_weight.float(),
        l_weight=l_weight.float(),
        canonical_correlations=correlations.float(),
        effective_dim=int(maximum),
        fit_ids=ids,
        seed=int(seed),
    )


def fit_llm_anchored_projection(
    h_g: torch.Tensor,
    h_l: torch.Tensor,
    fit_ids: torch.Tensor,
    test_ids: torch.Tensor,
    *,
    dim: int,
    ridge: float,
    seed: int,
) -> FittedProjection:
    """Map dataset-local GNN states into one fixed LLM semantic coordinate system."""
    g = torch.as_tensor(h_g, dtype=torch.float64).cpu()
    l = torch.as_tensor(h_l, dtype=torch.float64).cpu()
    ids = torch.as_tensor(fit_ids, dtype=torch.long).view(-1).cpu()
    tests = torch.as_tensor(test_ids, dtype=torch.long).view(-1).cpu()
    if g.ndim != 2 or l.ndim != 2 or g.shape[0] != l.shape[0]:
        raise ValueError("paired embeddings must be aligned matrices")
    assert_fit_ids_exclude_test(ids, tests)
    if ids.numel() < 3 or ids.numel() != torch.unique(ids).numel():
        raise ValueError("projection needs at least three unique fit IDs")
    if bool(((ids < 0) | (ids >= g.shape[0])).any().item()):
        raise ValueError("projection fit IDs out of range")
    if not bool(torch.isfinite(g).all().item()) or not bool(
        torch.isfinite(l).all().item()
    ):
        raise ValueError("embeddings contain non-finite values")
    width = min(int(dim), l.shape[1])
    if width < 1 or float(ridge) <= 0:
        raise ValueError("projection dimension and ridge must be positive")

    generator = torch.Generator().manual_seed(int(seed))
    raw = torch.randn(l.shape[1], width, generator=generator, dtype=torch.float64)
    l_weight = torch.linalg.qr(raw, mode="reduced").Q[:, :width]
    target = l[ids] @ l_weight

    g_center, g_scale = _robust_location_scale(g[ids])
    gx = (g[ids] - g_center) / g_scale
    gx_mean = gx.mean(dim=0, keepdim=True)
    target_mean = target.mean(dim=0, keepdim=True)
    centered_g = gx - gx_mean
    centered_target = target - target_mean
    gram = centered_g.T @ centered_g
    regularized = gram + float(ridge) * torch.eye(
        gram.shape[0], dtype=gram.dtype
    )
    g_weight = torch.linalg.solve(
        regularized, centered_g.T @ centered_target
    )
    g_bias = (
        target_mean - gx_mean @ g_weight
    ).view(-1)

    fit_zg = gx @ g_weight + g_bias
    fit_zl = target
    correlations = []
    for column in range(width):
        pair = torch.stack([fit_zg[:, column], fit_zl[:, column]])
        correlation = torch.corrcoef(pair)[0, 1]
        correlations.append(torch.nan_to_num(correlation))
    return FittedProjection(
        spec=ProjectionSpec(
            family="llm_anchor", dim=width, pca_dim=width, ridge=float(ridge)
        ),
        g_center=g_center.float(),
        g_scale=g_scale.float(),
        l_center=torch.zeros(l.shape[1], dtype=torch.float32),
        l_scale=torch.ones(l.shape[1], dtype=torch.float32),
        g_weight=g_weight.float(),
        l_weight=l_weight.float(),
        canonical_correlations=torch.stack(correlations).float(),
        effective_dim=width,
        fit_ids=ids,
        seed=int(seed),
        g_bias=g_bias.float(),
    )


def mixed_embedding_features(z_g: torch.Tensor, z_l: torch.Tensor) -> torch.Tensor:
    g = torch.as_tensor(z_g, dtype=torch.float32)
    l = torch.as_tensor(z_l, dtype=torch.float32)
    if g.shape != l.shape or g.ndim != 2:
        raise ValueError("projected embeddings must have identical matrix shapes")
    cosine = F.cosine_similarity(g, l, dim=1, eps=1e-8).view(-1, 1)
    result = torch.cat([g, l, g - l, g * l, cosine], dim=1)
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError("mixed embedding evidence contains non-finite values")
    return result.contiguous()


__all__ = [
    "FittedProjection",
    "ProjectionSpec",
    "fit_llm_anchored_projection",
    "fit_projection",
    "mixed_embedding_features",
]
