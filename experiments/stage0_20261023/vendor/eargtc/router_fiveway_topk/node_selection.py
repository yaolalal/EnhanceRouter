"""Label-free candidate selection ahead of frozen five-way Router policies.

Only ``selection_metrics`` accepts evaluation truth. Selection and routing
operate on IDs, cached predictions/embeddings, and previously frozen scores.
"""

from __future__ import annotations

from fractions import Fraction
from math import ceil

import numpy as np

from .policy import rank_exact


STRATEGIES = {
    "full": "Full + Top-k",
    "random": "Random → Router",
    "reliability": "Reliability → Router",
    "reliability_cluster": "Reliability + cluster → Router",
}


def _aligned(node_ids, scores):
    ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if ids.size != values.size or np.unique(ids).size != ids.size:
        raise ValueError("node IDs and scores must align and IDs must be unique")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    return ids, values


def select_candidates(
    node_ids, reliability, *, strategy: str, fraction: float = .2,
    seed: int = 42, cluster_ids=None, reliable_pool_multiplier: float = 2.0,
) -> np.ndarray:
    """Return a mask in input order, with the same exact budget for partial schemes.

    Cluster allocation is round-robin among available clusters. At each level,
    order clusters by their next member's reliability, breaking ties by node ID.
    Exhausted clusters release their capacity to the others. This also gives a
    score-based allocation when the budget is smaller than the cluster count.
    """
    ids, score = _aligned(node_ids, reliability)
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if not np.isfinite(reliable_pool_multiplier) or reliable_pool_multiplier < 1:
        raise ValueError("reliable_pool_multiplier must be finite and >= 1")
    if strategy == "full":
        return np.ones(ids.size, dtype=bool)
    quota = min(ids.size, ceil(Fraction(str(fraction)) * ids.size))
    mask = np.zeros(ids.size, dtype=bool)
    if not quota:
        return mask
    order = np.lexsort((ids, -score))
    if strategy == "random":
        ordered = np.argsort(ids)
        chosen = np.random.default_rng(seed).permutation(ordered)[:quota]
    elif strategy == "reliability":
        chosen = order[:quota]
    else:
        clusters = np.asarray(cluster_ids).reshape(-1)
        if clusters.size != ids.size or not np.issubdtype(clusters.dtype, np.integer):
            raise ValueError("integer cluster IDs must align with node IDs")
        pool = order[:min(ids.size, ceil(reliable_pool_multiplier * quota))]
        # 'level' is within-cluster rank in the reliable pool. Lexicographic
        # ordering allocates one per cluster before a second, and so on.
        _, inverse = np.unique(clusters[pool], return_inverse=True)
        counts = np.zeros(int(inverse.max()) + 1, dtype=np.int64)
        levels = np.empty(pool.size, dtype=np.int64)
        for i, cluster in enumerate(inverse):
            levels[i] = counts[cluster]
            counts[cluster] += 1
        balanced = np.lexsort((ids[pool], -score[pool], levels))
        chosen = pool[balanced[:quota]]
    mask[chosen] = True
    return mask


def embedding_clusters(node_ids, embeddings, *, n_clusters=32, seed=42):
    """Cluster normalized cached embeddings without labels, in canonical ID order."""
    from sklearn.cluster import MiniBatchKMeans
    from threadpoolctl import threadpool_limits

    raw_ids = np.asarray(node_ids).reshape(-1)
    ids, _ = _aligned(raw_ids, np.zeros(raw_ids.size))
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] != ids.size or not ids.size or not np.isfinite(x).all():
        raise ValueError("finite non-empty embeddings must align with IDs")
    if int(n_clusters) < 1:
        raise ValueError("n_clusters must be positive")
    order = np.argsort(ids)
    x = np.array(x[order], copy=True)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    model = MiniBatchKMeans(
        n_clusters=min(int(n_clusters), ids.size), random_state=int(seed),
        n_init=3, batch_size=4096, max_iter=100, reassignment_ratio=0.0,
    )
    with threadpool_limits(limits=1):
        sorted_labels = model.fit_predict(x)
    labels = np.empty(ids.size, dtype=np.int64)
    labels[order] = sorted_labels
    return labels, model.cluster_centers_


def route_candidates(node_ids, scores, candidate_ids, k_fraction):
    """Apply original frozen k within this module's candidate population."""
    ids, score = _aligned(node_ids, scores)
    if not 0 < k_fraction <= 1:
        raise ValueError("k_fraction must be in (0, 1]")
    positions = np.flatnonzero(np.isin(ids, np.asarray(candidate_ids, dtype=np.int64)))
    accepted = np.zeros(ids.size, dtype=bool)
    if positions.size:
        accepted[positions] = rank_exact(ids[positions], score[positions], k_fraction)
    return accepted


def selection_metrics(candidates, accepted, correct):
    candidate = np.asarray(candidates, dtype=bool).reshape(-1)
    accept = np.asarray(accepted, dtype=bool).reshape(-1)
    truth = np.asarray(correct, dtype=bool).reshape(-1)
    if not candidate.size or not candidate.size == accept.size == truth.size:
        raise ValueError("non-empty masks and correctness must align")
    if (accept & ~candidate).any():
        raise ValueError("accepted nodes must be candidates")
    n, m, a = candidate.size, int(candidate.sum()), int(accept.sum())
    c = int(truth[accept].sum())
    return dict(total=n, candidates=m, prefiltered=n-m, router_rejected=m-a,
                accepted=a, rejected=n-a, correct=c, incorrect=a-c,
                accuracy=c/a if a else None, coverage=a/n, candidate_fraction=m/n,
                acceptance_within_candidates=a/m if m else None)


def coverage_diagnostics(selected, classes, clusters, n_classes):
    """Predicted-class and cluster coverage, not true-label coverage."""
    mask = np.asarray(selected, dtype=bool)
    pred = np.asarray(classes, dtype=np.int64)
    group = np.asarray(clusters, dtype=np.int64)
    if not mask.size == pred.size == group.size:
        raise ValueError("coverage diagnostic arrays must align")
    histogram = np.bincount(pred[mask], minlength=n_classes)
    _, counts = np.unique(group[mask], return_counts=True)
    p = counts / counts.sum() if counts.size else np.array([])
    return {
        "predicted_classes_covered": int((histogram > 0).sum()),
        "predicted_class_histogram": histogram.tolist(),
        "clusters_covered": int(counts.size),
        "cluster_entropy": float(-(p * np.log(p)).sum()),
    }
