from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import STATE_NAMES


DISPLAY_NAMES = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
    "arxiv": "Arxiv",
    "ogbn-products": "OGBN-Products",
}
COLORS = {
    "gnn_only": "#3274A1",
    "llm_only": "#E1812C",
    "correct": "#3A923A",
    "wrong": "#C44E52",
    "correct_agreement": "#55A868",
    "wrong_agreement": "#8172B2",
    "conflict_neither": "#8C8C8C",
}


def _save(figure, output_dir: Path, category: str, dataset: str):
    target = Path(output_dir) / category
    target.mkdir(parents=True, exist_ok=True)
    png = target / f"{dataset}.png"
    pdf = target / f"{dataset}.pdf"
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _density_layer(axis, density: dict, label: str, color: str):
    edges = np.asarray(density["bin_edges"])
    hist = np.asarray(density["hist_density"])
    axis.bar(
        edges[:-1],
        hist,
        width=np.diff(edges),
        align="edge",
        color=color,
        alpha=0.25,
        edgecolor=color,
        linewidth=0.35,
    )
    axis.plot(
        density["grid"],
        density["kde_density"],
        color=color,
        linewidth=2.1,
        label=f"{label} (n={density['count']:,}, median={density['median']:.3f})",
    )
    axis.axvline(density["median"], color=color, linestyle=":", linewidth=1.1)


def plot_b_density(analysis: dict, output_dir: Path):
    dataset = analysis["dataset"]
    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    _density_layer(axis, analysis["b_density"]["gnn_only"], "GNN-only", COLORS["gnn_only"])
    _density_layer(axis, analysis["b_density"]["llm_only"], "LLM-only", COLORS["llm_only"])
    axis.axvline(
        analysis["routing_threshold"], color="black", linestyle="--", linewidth=1.3,
        label=fr"Frozen $\tau_B$={analysis['routing_threshold']:.3f}",
    )
    axis.set(xlim=(0, 1), xlabel="Scheme B preference score c(x)", ylabel="Probability density")
    axis.set_title(f"{DISPLAY_NAMES.get(dataset, dataset)} — Scheme B decisive-score density")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8.5)
    return _save(figure, output_dir, "scheme_b_density", dataset)


def plot_c_density(analysis: dict, output_dir: Path):
    dataset = analysis["dataset"]
    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    _density_layer(axis, analysis["c_density"]["correct"], "Selected correct", COLORS["correct"])
    _density_layer(axis, analysis["c_density"]["wrong"], "Selected wrong", COLORS["wrong"])
    axis.axvline(
        analysis["top25_cutoff"], color="black", linestyle="--", linewidth=1.3,
        label=f"Top-25% cutoff={analysis['top25_cutoff']:.3f}",
    )
    axis.set(xlim=(0, 1), xlabel="Scheme C selected trust q(x)", ylabel="Probability density")
    axis.set_title(f"{DISPLAY_NAMES.get(dataset, dataset)} — Scheme C correctness density")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8.5)
    return _save(figure, output_dir, "scheme_c_density", dataset)


def plot_rank_composition(analysis: dict, output_dir: Path):
    dataset = analysis["dataset"]
    segments = analysis["segments"]
    figure, axis = plt.subplots(figsize=(8.6, 5.0))
    x = np.arange(len(segments))
    bottom = np.zeros(len(segments))
    labels = {
        "correct_agreement": "Correct agreement",
        "wrong_agreement": "Wrong agreement",
        "gnn_only": "GNN-only",
        "llm_only": "LLM-only",
        "conflict_neither": "Conflict neither",
    }
    for state in STATE_NAMES:
        values = np.asarray([row[f"{state}_proportion"] for row in segments])
        axis.bar(
            x, values, bottom=bottom, color=COLORS[state], label=labels[state],
            edgecolor="white", linewidth=0.5,
        )
        bottom += values
    for index, row in enumerate(segments):
        axis.text(index, 1.02, f"P={row['precision']:.3f}", ha="center", fontsize=8)
    axis.set_xticks(x, [row["segment"] for row in segments])
    axis.set_ylim(0, 1.10)
    axis.set_ylabel("Within-segment node proportion")
    axis.set_xlabel("Descending Scheme C trust rank")
    axis.set_title(f"{DISPLAY_NAMES.get(dataset, dataset)} — trust-ranked node composition")
    axis.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    axis.grid(axis="y", alpha=0.2)
    return _save(figure, output_dir, "scheme_c_rank_composition", dataset)


def plot_loss_effects(analysis: dict, output_dir: Path):
    dataset = analysis["dataset"]
    effects = sorted(
        [row for row in analysis["loss_effects"] if row["rank_biserial"] is not None],
        key=lambda row: abs(row["rank_biserial"]),
        reverse=True,
    )[:10]
    effects = list(reversed(effects))
    figure, axis = plt.subplots(figsize=(8.2, 5.5))
    values = np.asarray([row["rank_biserial"] for row in effects], dtype=float)
    colors = [COLORS["wrong"] if value > 0 else COLORS["gnn_only"] for value in values]
    axis.barh(np.arange(len(effects)), values, color=colors)
    axis.set_yticks(np.arange(len(effects)), [row["feature"] for row in effects])
    axis.axvline(0, color="black", linewidth=1)
    axis.set_xlim(-1, 1)
    axis.set_xlabel("Rank-biserial effect (loss − protected)")
    title_count = effects[-1]["loss_count"] if effects else 0
    reference_count = effects[-1]["reference_count"] if effects else 0
    axis.set_title(
        f"{DISPLAY_NAMES.get(dataset, dataset)} — loss-node evidence effects\n"
        f"loss n={title_count:,}, protected n={reference_count:,}"
    )
    axis.grid(axis="x", alpha=0.2)
    return _save(figure, output_dir, "loss_feature_effects", dataset)
