from __future__ import annotations

import torch


def _matrix(value: torch.Tensor, *, name: str) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float64).cpu()
    if result.ndim != 2 or min(result.shape) < 1:
        raise ValueError(f"{name} must be a nonempty matrix")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} contains non-finite values")
    return result


def source_reference(weights: list[torch.Tensor], *, dim: int) -> torch.Tensor:
    if not weights:
        raise ValueError("source reference requires at least one projection basis")
    matrices = [_matrix(value, name="source weight") for value in weights]
    rows = matrices[0].shape[0]
    width = int(dim)
    if width <= 0 or any(value.shape[0] != rows or value.shape[1] < width for value in matrices):
        raise ValueError("source projection bases do not support the requested dimension")
    concatenated = torch.cat([value[:, :width] for value in matrices], dim=1)
    left, _, _ = torch.linalg.svd(concatenated, full_matrices=False)
    reference = left[:, :width].contiguous()
    for column in range(reference.shape[1]):
        pivot = int(reference[:, column].abs().argmax())
        if reference[pivot, column] < 0:
            reference[:, column] *= -1
    return reference.to(torch.float32)


def procrustes_rotation(local_weight: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    local = _matrix(local_weight, name="local weight")
    target = _matrix(reference, name="reference")
    if local.shape != target.shape:
        raise ValueError("local and reference projection bases must align")
    left, _, right_t = torch.linalg.svd(local.T @ target, full_matrices=False)
    return (left @ right_t).to(torch.float32).contiguous()


def alignment_rotations(
    weights_by_dataset: dict[str, torch.Tensor],
    *,
    source_names: tuple[str, ...],
    family: str,
) -> tuple[dict[str, torch.Tensor], dict]:
    if family not in {"cca", "pca", "random", "anchor"}:
        raise ValueError(f"unknown projection family: {family}")
    if not weights_by_dataset or not source_names:
        raise ValueError("alignment requires datasets and source names")
    missing = set(source_names) - set(weights_by_dataset)
    if missing:
        raise ValueError(f"alignment sources are missing: {sorted(missing)}")
    widths = {tuple(torch.as_tensor(value).shape) for value in weights_by_dataset.values()}
    if len(widths) != 1:
        raise ValueError("projection weights must have equal shapes")
    _, dim = next(iter(widths))
    if family in {"random", "anchor"}:
        rotations = {
            name: torch.eye(dim, dtype=torch.float32) for name in weights_by_dataset
        }
        return rotations, {
            "family": family,
            "alignment": "shared_basis_identity",
            "source_names": list(source_names),
            "dimension": dim,
            "residuals": {name: 0.0 for name in weights_by_dataset},
        }
    reference = source_reference(
        [weights_by_dataset[name] for name in source_names], dim=dim
    )
    rotations = {}
    residuals = {}
    for name, weight in weights_by_dataset.items():
        rotation = procrustes_rotation(weight, reference)
        rotations[name] = rotation
        local = torch.as_tensor(weight, dtype=torch.float32)
        denominator = torch.linalg.norm(reference).clamp_min(1e-12)
        residuals[name] = float(
            (torch.linalg.norm(local @ rotation - reference) / denominator).item()
        )
    return rotations, {
        "family": family,
        "alignment": "source_reference_llm_weight_procrustes",
        "source_names": list(source_names),
        "dimension": dim,
        "reference": reference,
        "residuals": residuals,
    }


__all__ = ["alignment_rotations", "procrustes_rotation", "source_reference"]

