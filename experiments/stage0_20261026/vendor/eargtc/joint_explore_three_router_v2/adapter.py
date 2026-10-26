"""Guarded three-router view of the historical full-candidate export."""
from __future__ import annotations

from collections import OrderedDict

from eargtc.joint_five_router import adapter as _old_adapter

ROOT = _old_adapter.ROOT
DATASETS = _old_adapter.DATASETS
DEFAULT_EXPORT = _old_adapter.DEFAULT_EXPORT
ROUTERS = OrderedDict((
    ("r6_legacy_c", "R6-Legacy-C"),
    ("r6_split_c", "R6-Split-C"),
    ("a20ms_split_c", "A20-MS-Split-C"),
))
_old_load_cell = _old_adapter.load_cell


def load_cell(export_root, router_id: str, dataset: str, verify_sources: bool = True):
    """Load an allowed historical cell and describe Joint-Explore's pool use."""
    if router_id not in ROUTERS or dataset not in DATASETS:
        raise ValueError("unknown router or dataset")

    features, supervision, pools, hidden, old_meta = _old_load_cell(
        export_root, router_id, dataset, verify_sources=verify_sources
    )
    candidate_count = len(supervision["test_ids"])
    accepted_conflict_count = len(pools["teach_g"]) + len(pools["teach_l"])
    accepted_agreement_count = len(pools["agreement"])
    accepted_count = old_meta["accepted_count"]
    if accepted_count != accepted_conflict_count + accepted_agreement_count:
        raise ValueError("accepted count does not equal teaching plus agreement")
    if candidate_count != accepted_count + len(pools["unselected"]):
        raise ValueError("accepted and unselected pools do not cover full candidates")

    meta = dict(old_meta)
    meta.update({
        "candidate_count": candidate_count,
        "pseudo_teaching_count": accepted_count,
        "accepted_conflict_count": accepted_conflict_count,
        "accepted_agreement_count": accepted_agreement_count,
        "candidate_strategy": "full",
        "trust_note": (
            "Joint-Explore uses frozen confidence from the Router for pseudo-label weighting "
            "and retrieval across accepted conflict and agreement nodes; the "
            "historical note described only old Joint."
        ),
    })
    return features, supervision, pools, hidden, meta
