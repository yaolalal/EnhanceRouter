from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch


CHECKPOINT_VALIDATION_NAMESPACE = "checkpoint-validation-v1"
CHECKPOINT_VALIDATION_MAX_NODES = 1000


@dataclass(frozen=True)
class DownstreamSplit:
    gold_support_ids: torch.Tensor
    official_validation_ids: torch.Tensor
    pseudo_training_candidate_ids: torch.Tensor
    heldout_evaluation_ids: torch.Tensor
    protocol: str
    fold: int | None = None


def recovered_eval_ids(dataset: str, root: Path) -> torch.Tensor:
    path = Path(root) / f"{dataset}_5shot_seed42_test_node_ids.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    ids = torch.tensor(value["node_ids"], dtype=torch.long)
    if ids.numel() != 1000 or ids.unique().numel() != ids.numel():
        raise ValueError(f"recovered evaluation ids are not 1000 unique nodes: {path}")
    return ids


def _set(value: torch.Tensor) -> set[int]:
    return set(torch.as_tensor(value, dtype=torch.long).view(-1).tolist())


def checkpoint_validation_selection(
    official_validation_ids: torch.Tensor,
    dataset: str,
    *,
    max_nodes: int = CHECKPOINT_VALIDATION_MAX_NODES,
) -> torch.Tensor:
    """Select the preregistered target-free checkpoint validation subset."""
    ids = torch.as_tensor(official_validation_ids, dtype=torch.long).view(-1)
    if ids.unique().numel() != ids.numel():
        raise ValueError("official validation ids contain duplicates")
    if not str(dataset):
        raise ValueError("dataset must be non-empty")
    if int(max_nodes) <= 0:
        raise ValueError("max_nodes must be positive")

    def rank(node_id: int) -> tuple[bytes, int]:
        payload = (
            f"{CHECKPOINT_VALIDATION_NAMESPACE}:{dataset}:{int(node_id)}"
        ).encode("utf-8")
        return hashlib.sha256(payload).digest(), int(node_id)

    selected = sorted(ids.tolist(), key=rank)[: min(int(max_nodes), ids.numel())]
    return torch.tensor(selected, dtype=torch.long, device=ids.device)


def node_id_list_sha256(node_ids: torch.Tensor) -> str:
    ids = torch.as_tensor(node_ids, dtype=torch.long).view(-1)
    payload = json.dumps(ids.tolist(), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def targets_for_node_ids(
    source_ids: torch.Tensor,
    source_targets: torch.Tensor,
    selected_ids: torch.Tensor,
) -> torch.Tensor:
    source = torch.as_tensor(source_ids, dtype=torch.long).view(-1)
    selected = torch.as_tensor(selected_ids, dtype=torch.long).view(-1)
    targets = torch.as_tensor(source_targets)
    if source.unique().numel() != source.numel():
        raise ValueError("source node ids contain duplicates")
    if selected.unique().numel() != selected.numel():
        raise ValueError("selected node ids contain duplicates")
    if targets.ndim == 0 or targets.size(0) != source.numel():
        raise ValueError("source ids and targets are not row-aligned")
    position = {int(node): index for index, node in enumerate(source.tolist())}
    missing = [int(node) for node in selected.tolist() if int(node) not in position]
    if missing:
        raise ValueError(f"selected node ids are missing from source ids: {missing[:10]}")
    indices = torch.tensor(
        [position[int(node)] for node in selected.tolist()],
        dtype=torch.long,
        device=targets.device,
    )
    return targets.index_select(0, indices)


def strict_holdout_split(
    *,
    gold_support_ids: torch.Tensor,
    official_validation_ids: torch.Tensor,
    pseudo_training_candidate_ids: torch.Tensor,
    heldout_evaluation_ids: torch.Tensor,
) -> DownstreamSplit:
    gold = torch.as_tensor(gold_support_ids, dtype=torch.long).view(-1)
    validation = torch.as_tensor(official_validation_ids, dtype=torch.long).view(-1)
    evaluation = torch.as_tensor(heldout_evaluation_ids, dtype=torch.long).view(-1)
    if _set(gold) & _set(validation) or _set(gold) & _set(evaluation) or _set(validation) & _set(evaluation):
        raise ValueError("gold, validation, and evaluation ids must be disjoint")
    candidates = torch.as_tensor(pseudo_training_candidate_ids, dtype=torch.long).view(-1)
    if _set(candidates) & (_set(gold) | _set(validation)):
        raise ValueError("pseudo candidates overlap gold or validation ids")
    keep = ~torch.isin(candidates, evaluation)
    candidates = candidates[keep]
    if _set(candidates) & _set(evaluation):
        raise AssertionError("heldout evaluation ids remain in pseudo candidates")
    return DownstreamSplit(
        gold,
        validation,
        candidates,
        evaluation,
        "strict_holdout_pseudo_label_supervision",
    )


def node_hash_fold(node_id: int, dataset: str, folds: int = 5) -> int:
    if int(folds) <= 1:
        raise ValueError("folds must exceed one")
    digest = hashlib.sha256(f"{dataset}:{int(node_id)}".encode("utf-8")).digest()
    return int.from_bytes(digest, byteorder="big") % int(folds)


def crossfit_splits(
    test_ids: torch.Tensor,
    dataset: str,
    pseudo_candidate_ids: torch.Tensor,
    *,
    folds: int = 5,
    gold_support_ids: torch.Tensor | None = None,
    official_validation_ids: torch.Tensor | None = None,
) -> list[DownstreamSplit]:
    test = torch.as_tensor(test_ids, dtype=torch.long).view(-1)
    if test.unique().numel() != test.numel():
        raise ValueError("test ids contain duplicates")
    gold = torch.empty(0, dtype=torch.long) if gold_support_ids is None else torch.as_tensor(gold_support_ids, dtype=torch.long).view(-1)
    validation = torch.empty(0, dtype=torch.long) if official_validation_ids is None else torch.as_tensor(official_validation_ids, dtype=torch.long).view(-1)
    candidates = torch.as_tensor(pseudo_candidate_ids, dtype=torch.long).view(-1)
    result = []
    assignments = torch.tensor(
        [node_hash_fold(int(node), dataset, folds) for node in test], dtype=torch.long
    )
    for fold in range(int(folds)):
        held = test[assignments == fold]
        keep = ~torch.isin(candidates, held)
        split = strict_holdout_split(
            gold_support_ids=gold,
            official_validation_ids=validation,
            pseudo_training_candidate_ids=candidates[keep],
            heldout_evaluation_ids=held,
        )
        result.append(
            DownstreamSplit(
                split.gold_support_ids,
                split.official_validation_ids,
                split.pseudo_training_candidate_ids,
                split.heldout_evaluation_ids,
                "transductive_graph_representation_cross_fitted_pseudo_label_supervision",
                fold,
            )
        )
    return result
