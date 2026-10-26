"""Five-Router B-specific Scheme-C and validation top-k experiment."""

from .policy import DEFAULT_GRID, TopKCurvePoint, TopKSelection

ROUTER_NAMES = {
    "r6_legacy_c": "R6-LegacyC",
    "r6_split_c": "R6-SplitC",
    "r6_expert_conf_c": "R6-EC",
    "a20ms_split_c": "A20-SplitC",
    "a20emb_v2_split_c": "A20Emb-SplitC",
}

__all__ = ["DEFAULT_GRID", "TopKCurvePoint", "TopKSelection", "ROUTER_NAMES"]
