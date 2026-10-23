from __future__ import annotations

import hashlib
import json
import sys
import types
import importlib.machinery
import gzip
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from .utils import ensure_dir, set_seed


TEXT_ATTRS = ("raw_texts", "texts", "node_text", "text")
LABEL_ATTRS = ("label_name", "label_names")


def _install_pyg_optional_extension_stubs() -> None:
    """Avoid broken optional PyG CUDA extensions during Data unpickling.

    Some environments have torch_geometric installed but optional compiled
    packages such as torch_cluster are ABI-incompatible with the active torch.
    Loading a saved PyG Data object only needs torch_geometric.data, not those
    extensions. Stubbing them before torch.load lets unpickling proceed while
    our own pure-PyTorch GNN backend handles message passing.
    """
    specs = {
        "torch_cluster": [
            "knn", "knn_graph", "radius", "radius_graph", "nearest", "fps",
            "graclus_cluster", "grid_cluster", "random_walk",
        ],
        "torch_spline_conv": ["spline_basis", "spline_weighting"],
    }
    for module_name, attrs in specs.items():
        if module_name in sys.modules:
            continue
        module = types.ModuleType(module_name)
        module.__spec__ = importlib.machinery.ModuleSpec(module_name, None)
        for attr in attrs:
            def _missing(*args, _module_name=module_name, **kwargs):
                raise ImportError(f"{_module_name} is stubbed because the installed optional PyG extension is unusable.")
            setattr(module, attr, _missing)
        sys.modules[module_name] = module


def _to_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    rev = edge_index.flip(0)
    both = torch.cat([edge_index.long(), rev.long()], dim=1)
    num_nodes = int(both.max().item()) + 1 if both.numel() else 0
    keys = both[0] * max(num_nodes, 1) + both[1]
    uniq = torch.unique(keys)
    return torch.stack([uniq // max(num_nodes, 1), uniq % max(num_nodes, 1)], dim=0).long()


def _move_to_device(data, device: torch.device):
    if hasattr(data, "to"):
        return data.to(device)
    for key, value in list(vars(data).items()):
        if torch.is_tensor(value):
            setattr(data, key, value.to(device))
    return data


def _dataset_file(dataset_root: str | Path, dataset: str) -> Path:
    return Path(dataset_root) / "datasets" / f"{dataset}.pt"


def load_graph_dataset(dataset: str, dataset_root: str | Path, device: torch.device):
    path = _dataset_file(dataset_root, dataset)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    _install_pyg_optional_extension_stubs()
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not hasattr(data, "edge_index"):
        raise ValueError(f"{path} does not look like a PyG Data object: missing edge_index")
    data.edge_index = _to_undirected(data.edge_index)
    return _move_to_device(data, device)


def get_texts(data) -> List[str]:
    for attr in TEXT_ATTRS:
        if hasattr(data, attr):
            value = getattr(data, attr)
            if isinstance(value, (list, tuple)):
                return ["" if x is None else str(x) for x in value]
            if isinstance(value, np.ndarray):
                return ["" if x is None else str(x) for x in value.tolist()]
    return [""] * int(data.num_nodes)


def get_label_names(data, num_classes: int) -> List[str]:
    for attr in LABEL_ATTRS:
        if hasattr(data, attr):
            value = getattr(data, attr)
            if isinstance(value, (list, tuple)):
                return [str(x) for x in value]
    return [str(i) for i in range(num_classes)]


def masks_dict(data):
    masks = {}
    for name in ("train", "val", "test"):
        attr = f"{name}_mask"
        if hasattr(data, attr):
            masks[name] = getattr(data, attr).bool()
    return masks


def _empty_mask_like(mask: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(mask, dtype=torch.bool)


def _read_ogb_split_csv(split_root: Path, name: str, device: torch.device) -> torch.Tensor:
    path = split_root / f"{name}.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        values = [int(line.strip()) for line in f if line.strip()]
    return torch.tensor(values, dtype=torch.long, device=device)


def _reference_split_file(dataset_root: str | Path, dataset: str, shots: int, seed: int) -> Path:
    return Path(dataset_root) / "datasets" / "splits" / f"{dataset}_{shots}shot_seed{seed}.json"


def _apply_saved_few_shot_split(data, split_file: Path, device: torch.device):
    with open(split_file, "r", encoding="utf-8") as f:
        split = json.load(f)
    if "train_ids" not in split or "val_ids" not in split:
        raise ValueError(f"Saved split is missing train_ids/val_ids: {split_file}")

    train_ids = torch.tensor([int(i) for i in split["train_ids"]], dtype=torch.long, device=device)
    val_ids = torch.tensor([int(i) for i in split["val_ids"]], dtype=torch.long, device=device)
    num_nodes = int(data.num_nodes)
    for name, ids in (("train_ids", train_ids), ("val_ids", val_ids)):
        if ids.numel() and (int(ids.min().item()) < 0 or int(ids.max().item()) >= num_nodes):
            raise ValueError(f"{name} in {split_file} contains node ids outside [0, {num_nodes})")
        if torch.unique(ids.detach().cpu()).numel() != ids.numel():
            raise ValueError(f"{name} in {split_file} contains duplicate node ids")

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[train_ids] = True
    val_mask[val_ids] = True
    if bool((train_mask & val_mask).any().item()):
        raise ValueError(f"Saved split has train/val overlap: {split_file}")

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = ~(train_mask | val_mask)
    return data


def _create_arxiv_few_shot(data, shots: int, seed: int, dataset_root: str | Path, device: torch.device):
    set_seed(seed)
    split_root = Path(dataset_root) / "dataset" / "ogbn_arxiv" / "split" / "time"
    if split_root.exists():
        train_idx = _read_ogb_split_csv(split_root, "train", device)
        val_idx = _read_ogb_split_csv(split_root, "valid", device)
        test_idx = _read_ogb_split_csv(split_root, "test", device)
    else:
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
        except ImportError as exc:
            raise ImportError("Few-shot arxiv split requires ogb or local OGB split CSV files.") from exc

        ogb_root = Path(dataset_root) / "dataset"
        original_torch_load = torch.load

        def _trusted_torch_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        try:
            torch.load = _trusted_torch_load
            ogb_dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(ogb_root))
        finally:
            torch.load = original_torch_load
        split_idx = ogb_dataset.get_idx_split()
        train_idx = split_idx["train"].to(device)
        val_idx = split_idx["valid"].to(device)
        test_idx = split_idx["test"].to(device)

    official_train = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    official_val = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    official_test = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    official_train[train_idx] = True
    official_val[val_idx] = True
    official_test[test_idx] = True

    num_classes = int(data.y.max().item() + 1)
    train_mask = _empty_mask_like(official_train)
    for c in range(num_classes):
        ids = ((data.y == c) & official_train).nonzero().view(-1)
        if ids.numel() == 0:
            continue
        take = min(shots, ids.numel())
        train_mask[ids[torch.randperm(ids.numel(), device=device)[:take]]] = True

    data.train_mask = train_mask
    data.val_mask = official_val
    data.test_mask = official_test | (official_train & ~train_mask)
    return data


def create_few_shot_dataset(
    dataset: str,
    dataset_root: str | Path,
    shots: int,
    seed: int,
    device: torch.device,
    split_output_dir: str | Path | None = None,
):
    set_seed(seed)
    data = load_graph_dataset(dataset, dataset_root, device)
    if not hasattr(data, "train_mask"):
        raise ValueError("Few-shot sampling requires an existing train_mask.")

    reference_split = _reference_split_file(dataset_root, dataset, shots, seed)
    if reference_split.exists():
        data = _apply_saved_few_shot_split(data, reference_split, device)
    elif dataset.lower() == "arxiv":
        data = _create_arxiv_few_shot(data, shots, seed, dataset_root, device)
    else:
        num_classes = int(data.y.max().item() + 1)
        base_train = data.train_mask.bool()
        train_mask = _empty_mask_like(base_train)
        for c in range(num_classes):
            ids = ((data.y == c) & base_train).nonzero().view(-1)
            if ids.numel() == 0:
                continue
            take = min(shots, ids.numel())
            train_mask[ids[torch.randperm(ids.numel(), device=device)[:take]]] = True

        pool = (~base_train).nonzero().view(-1)
        val_mask = _empty_mask_like(base_train)
        if pool.numel() > 0:
            take_val = min(500, pool.numel())
            val_mask[pool[torch.randperm(pool.numel(), device=device)[:take_val]]] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = ~(train_mask | val_mask)

    if split_output_dir is not None:
        split_dir = ensure_dir(split_output_dir)
        train_ids = data.train_mask.nonzero().view(-1).detach().cpu().tolist()
        val_ids = data.val_mask.nonzero().view(-1).detach().cpu().tolist()
        test_ids = data.test_mask.nonzero().view(-1).detach().cpu().tolist()
        split_hash = hashlib.md5(str(train_ids + val_ids + test_ids).encode()).hexdigest()[:8]
        with open(split_dir / f"{dataset}_{shots}shot_seed{seed}.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "hash": split_hash,
                    "train_ids": train_ids,
                    "val_ids": val_ids,
                    "test_count": len(test_ids),
                },
                f,
                indent=2,
            )
    return data


def label_space(data) -> Tuple[int, List[str]]:
    num_classes = len(getattr(data, "label_name", [])) if hasattr(data, "label_name") else int(data.y.max().item() + 1)
    return num_classes, get_label_names(data, num_classes)
