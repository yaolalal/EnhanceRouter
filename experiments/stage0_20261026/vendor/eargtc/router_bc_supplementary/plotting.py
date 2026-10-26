"""Deterministic publication-oriented figures for Router B/C diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "gnn": "#3274A1",
    "llm": "#E1812C",
    "router": "#3A923A",
    "trust": "#7A4EAB",
    "loss": "#C44E52",
    "neutral": "#777777",
    "both_correct": "#55A868",
    "both_wrong": "#8C8C8C",
    "agreement_wrong": "#CC79A7",
    "conflict_neither": "#999999",
}

DISPLAY_NAMES = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
    "arxiv": "Arxiv",
    "ogbn-products": "OGBN-Products",
}


def _name(analysis: dict) -> str:
    return DISPLAY_NAMES.get(analysis["dataset"], analysis["dataset"])


def _axes_grid(count: int = 5):
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 8.2))
    flat = list(axes.ravel())
    for axis in flat[count:]:
        axis.axis("off")
    return figure, flat[:count]


def _save(figure, output_dir: Path, basename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.subplots_adjust(left=0.08, right=0.94, bottom=0.12, top=0.86, wspace=0.36, hspace=0.50)
    png = output_dir / f"{basename}.png"
    pdf = output_dir / f"{basename}.pdf"
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(values, dtype=float))
    return ordered, np.arange(1, ordered.size + 1) / ordered.size


def normalized_histogram(
    values: np.ndarray | Sequence[float], bins: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-width [0,1] probability mass for one score group."""

    scores = np.asarray(values, dtype=float).reshape(-1)
    if bins <= 0:
        raise ValueError("histogram_bins_must_be_positive")
    if np.any(~np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("histogram_score_out_of_range")
    edges = np.linspace(0.0, 1.0, bins + 1)
    counts, _ = np.histogram(scores, bins=edges)
    mass = counts.astype(float)
    if scores.size:
        mass /= scores.size
    return mass, edges


def plot_preference_distribution(analyses: Sequence[dict], output_dir: Path) -> Path:
    figure, axes = _axes_grid(len(analyses))
    for axis, analysis in zip(axes, analyses):
        arrays = analysis["arrays"]
        for mask_name, label, color in (
            ("gnn_only", "GNN-only", COLORS["gnn"]),
            ("llm_only", "LLM-only", COLORS["llm"]),
        ):
            x, y = _ecdf(arrays["decision_margin"][arrays[mask_name]])
            axis.plot(x, y, color=color, linewidth=2.0, label=f"{label} (n={x.size:,})")
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1.1)
        decisive_margin = arrays["decision_margin"][
            arrays["gnn_only"] | arrays["llm_only"]
        ]
        if decisive_margin.size:
            low, high = np.quantile(decisive_margin, [0.01, 0.99])
            if high > low:
                axis.set_xlim(low, high)
        axis.set_title(
            f"{_name(analysis)}  AUROC={analysis['preference']['auroc']:.3f}"
        )
        axis.set_xlabel(r"Decision margin  $\mathrm{logit}(c)-\mathrm{logit}(\tau_B)$")
        axis.set_ylabel("Empirical CDF")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="lower right")
    figure.suptitle(
        "Scheme B preference separation on decisive rows\n"
        "(display range: decisive-score 1st–99th percentile; boundary fixed at 0)",
        fontsize=14,
    )
    return _save(figure, output_dir, "preference_distribution")


def _plot_reliability(
    analyses: Sequence[dict],
    output_dir: Path,
    basename: str,
    bins_key: str,
    metrics_key: str,
    color: str,
    title: str,
) -> Path:
    figure, axes = _axes_grid(len(analyses))
    for axis, analysis in zip(axes, analyses):
        bins = analysis[bins_key]
        x = np.asarray([item["mean_score"] for item in bins])
        y = np.asarray([item["empirical_rate"] for item in bins])
        lower = np.asarray([item["wilson_lower"] for item in bins])
        upper = np.asarray([item["wilson_upper"] for item in bins])
        counts = np.asarray([item["count"] for item in bins])
        axis.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1)
        axis.errorbar(
            x,
            y,
            yerr=np.vstack([y - lower, upper - y]),
            fmt="none",
            color=color,
            capsize=2.5,
            linewidth=1.5,
        )
        sizes = 28.0 + 50.0 * np.sqrt(counts / max(1, np.max(counts)))
        axis.scatter(
            x,
            y,
            s=sizes,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        for x_value, y_value, count in zip(x, y, counts):
            x_offset = -4 if x_value > 0.90 else 4
            y_offset = -7 if y_value > 0.92 else 5
            axis.annotate(
                f"n={count:,}",
                (x_value, y_value),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha="right" if x_value > 0.90 else "left",
                va="top" if y_value > 0.92 else "bottom",
                fontsize=6.5,
                color=color,
            )
        metrics = analysis[metrics_key]
        axis.set_title(
            f"{_name(analysis)}  ECE={metrics['ece']:.3f}, "
            f"AUROC={metrics['auroc']:.3f}"
        )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Mean predicted score")
        axis.set_ylabel("Empirical positive rate")
        axis.grid(alpha=0.2)
    figure.suptitle(title, fontsize=14)
    return _save(figure, output_dir, basename)


def plot_preference_calibration(analyses: Sequence[dict], output_dir: Path) -> Path:
    return _plot_reliability(
        analyses,
        output_dir,
        "preference_calibration",
        "preference_calibration_bins",
        "preference",
        COLORS["router"],
        "Scheme B conditional preference calibration\n"
        "(decisive rows only; positive event = GNN-only; Wilson 95% intervals)",
    )


def correction_loss_percentages(analysis: dict) -> tuple[float, float]:
    """Return corrections and introduced errors as percentages of test rows."""

    test_count = analysis["counts"]["all"]
    if test_count <= 0:
        raise ValueError("correction-loss decomposition requires nonempty test data")
    routing = analysis["routing"]
    return (
        100.0 * routing["wins"] / test_count,
        100.0 * routing["losses"] / test_count,
    )


def plot_correction_loss(analyses: Sequence[dict], output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(11.5, 5.8))
    y = np.arange(len(analyses))
    components = [correction_loss_percentages(item) for item in analyses]
    wins = np.asarray([item[0] for item in components], dtype=float)
    losses = np.asarray([item[1] for item in components], dtype=float)
    axis.barh(y, wins, color=COLORS["router"], label="Corrections (wins)")
    axis.barh(y, -losses, color=COLORS["loss"], label="Introduced errors (losses)")
    for index, analysis in enumerate(analyses):
        routing = analysis["routing"]
        axis.text(
            wins[index],
            index,
            f"  +{routing['wins']:,} ({wins[index]:.2f}%)",
            va="center",
            fontsize=9,
        )
        axis.text(
            -losses[index],
            index,
            f"−{routing['losses']:,} ({losses[index]:.2f}%)  ",
            va="center",
            ha="right",
            fontsize=9,
        )
        axis.text(
            0.99,
            index,
            f"net {routing['net_correct']:+,} "
            f"({routing['gain_over_best_single'] * 100:+.2f} pp)",
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    max_component = max(float(np.max(wins)), float(np.max(losses)), 1e-6)
    axis.set_xlim(-1.45 * max_component, 2.0 * max_component)
    axis.axvline(0, color="black", linewidth=0.9)
    axis.set_yticks(y, [_name(item) for item in analyses])
    axis.invert_yaxis()
    axis.set_xlabel("Contribution as percentage of test nodes")
    axis.set_title("Scheme B correction–loss decomposition")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    return _save(figure, output_dir, "correction_loss_decomposition")


def plot_oracle_recovery(analyses: Sequence[dict], output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(10.5, 5.5))
    values = np.asarray(
        [
            np.nan
            if item["routing"]["oracle_recovery"] is None
            else item["routing"]["oracle_recovery"] * 100
            for item in analyses
        ]
    )
    y = np.arange(len(analyses))
    axis.barh(y, values, color=COLORS["router"])
    for index, value in enumerate(values):
        axis.text(value + 0.5, index, f"{value:.1f}%", va="center", fontweight="bold")
    axis.set_yticks(y, [_name(item) for item in analyses])
    axis.invert_yaxis()
    axis.set_xlabel("Recovered fraction of best-single → Oracle headroom (%)")
    axis.set_title("How much expert complementarity does Scheme B recover?")
    axis.set_xlim(0, max(45, float(np.nanmax(values)) + 5))
    axis.grid(axis="x", alpha=0.2)
    return _save(figure, output_dir, "oracle_gap_recovery")


def plot_decisive_confusion(analyses: Sequence[dict], output_dir: Path) -> Path:
    figure, axes = plt.subplots(1, len(analyses), figsize=(16, 3.8))
    if len(analyses) == 1:
        axes = [axes]
    for axis, analysis in zip(axes, analyses):
        preference = analysis["preference"]
        g_recall = preference["gnn_only_recall"]
        l_recall = preference["llm_only_recall"]
        matrix = np.asarray([[g_recall, 1 - g_recall], [1 - l_recall, l_recall]])
        axis.imshow(matrix, vmin=0, vmax=1, cmap="YlGn")
        for row in range(2):
            for column in range(2):
                value = matrix[row, column]
                axis.text(
                    column,
                    row,
                    f"{value * 100:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if value > 0.6 else "black",
                    fontsize=10,
                    fontweight="bold",
                )
        axis.set_xticks([0, 1], ["Route G", "Route L"], rotation=25)
        axis.set_yticks([0, 1], ["G-only", "L-only"])
        axis.set_title(
            f"{_name(analysis)}\nBalanced={preference['balanced_accuracy']:.3f}"
        )
    figure.suptitle("Frozen-threshold routing on decisive rows", fontsize=14)
    return _save(figure, output_dir, "decisive_confusion")


def plot_threshold_sensitivity(analyses: Sequence[dict], output_dir: Path) -> Path:
    figure, axes = _axes_grid(len(analyses))
    for axis, analysis in zip(axes, analyses):
        curve = analysis["threshold_sensitivity"]
        thresholds = np.asarray([item["threshold"] for item in curve])
        gain = np.asarray([item["gain_over_best_single"] for item in curve]) * 100
        axis.plot(thresholds, gain, color=COLORS["router"], linewidth=1.8)
        frozen = [item for item in curve if item["is_frozen"]]
        if frozen:
            point = frozen[0]
            axis.scatter(
                [point["threshold"]],
                [point["gain_over_best_single"] * 100],
                color="black",
                marker="*",
                s=90,
                zorder=4,
                label="Frozen validation threshold",
            )
        axis.axhline(0.0, color=COLORS["neutral"], linestyle="--", linewidth=1)
        axis.set_xlim(0, 1)
        axis.set_title(_name(analysis))
        axis.set_xlabel(r"Retrospective test threshold $\tau$")
        axis.set_ylabel("Gain over best single (pp)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Retrospective test threshold sensitivity\n"
        "(diagnostic only; black star is the frozen validation-selected result)",
        fontsize=14,
    )
    return _save(figure, output_dir, "threshold_sensitivity")


def plot_scheme_c_reliability(analyses: Sequence[dict], output_dir: Path) -> Path:
    return _plot_reliability(
        analyses,
        output_dir,
        "scheme_c_reliability",
        "trust_calibration_bins",
        "trust",
        COLORS["trust"],
        "Scheme C selected-label reliability\n"
        "(all test rows; positive event = Scheme B selected label is correct)",
    )


def plot_scheme_c_risk_coverage(analyses: Sequence[dict], output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(10.5, 6.0))
    palette = plt.cm.viridis(np.linspace(0.08, 0.92, len(analyses)))
    for color, analysis in zip(palette, analyses):
        curve = analysis["risk_coverage"]
        coverage = np.asarray([item["coverage"] for item in curve])
        risk = np.asarray([item["risk"] for item in curve])
        if coverage.size > 2000:
            indices = np.unique(np.linspace(0, coverage.size - 1, 2000).astype(int))
            coverage = coverage[indices]
            risk = risk[indices]
        axis.plot(
            coverage,
            risk,
            color=color,
            linewidth=1.8,
            label=f"{_name(analysis)} (AURC={analysis['trust'].get('aurc', float('nan')):.3f})",
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Coverage")
    axis.set_ylabel("Risk = 1 − prefix precision")
    axis.set_title("Scheme C selected-label risk–coverage")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=9)
    return _save(figure, output_dir, "scheme_c_risk_coverage")


def plot_four_state_distribution(analyses: Sequence[dict], output_dir: Path) -> Path:
    figure, axes = _axes_grid(len(analyses))
    state_specs = (
        ("both_correct", "Both correct", COLORS["both_correct"]),
        ("gnn_only", "GNN-only", COLORS["gnn"]),
        ("llm_only", "LLM-only", COLORS["llm"]),
        ("both_wrong", "Both wrong", COLORS["both_wrong"]),
    )
    for axis, analysis in zip(axes, analyses):
        arrays = analysis["arrays"]
        values = [
            arrays["preference_score"][arrays[state]]
            for state, _, _ in state_specs
        ]
        box = axis.boxplot(
            values,
            patch_artist=True,
            widths=0.65,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
        )
        for patch, (_, _, color) in zip(box["boxes"], state_specs):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        axis.axhline(
            analysis["routing"]["threshold"],
            color="black",
            linestyle="--",
            linewidth=1,
            label=r"Frozen $\tau_B$",
        )
        axis.set_xticks(
            range(1, 5),
            [label for _, label, _ in state_specs],
            rotation=28,
            ha="right",
        )
        axis.set_ylim(0, 1)
        axis.set_title(_name(analysis))
        axis.set_ylabel("Preference score c(x)")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Four-state score extrapolation diagnostic\n"
        "(only GNN-only and LLM-only define Scheme B's preference target)",
        fontsize=14,
    )
    return _save(figure, output_dir, "four_state_distribution")


def _plot_mass(
    axis,
    values: np.ndarray,
    *,
    label: str,
    color: str,
    bins: int = 30,
) -> None:
    mass, edges = normalized_histogram(values, bins=bins)
    legend = f"{label} (n={values.size:,})"
    axis.stairs(mass, edges, fill=True, alpha=0.18, color=color)
    axis.stairs(mass, edges, linewidth=2.0, color=color, label=legend)


def plot_preference_score_distribution(
    analysis: dict, output_dir: Path
) -> Path:
    """Plot non-cumulative B score mass for decisive preference classes."""

    figure, axis = plt.subplots(figsize=(7.8, 5.2))
    arrays = analysis["arrays"]
    _plot_mass(
        axis,
        arrays["preference_score"][arrays["gnn_only"]],
        label="GNN-only",
        color=COLORS["gnn"],
    )
    _plot_mass(
        axis,
        arrays["preference_score"][arrays["llm_only"]],
        label="LLM-only",
        color=COLORS["llm"],
    )
    axis.axvline(
        analysis["routing"]["threshold"],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=r"Frozen $\tau_B$",
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Scheme B preference score  c(x)")
    axis.set_ylabel("Probability mass per fixed-width bin")
    axis.set_title(
        f"{_name(analysis)}: Scheme B non-cumulative score distribution"
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=9)
    return _save(
        figure,
        output_dir,
        f"preference_score_distribution_{analysis['dataset']}",
    )


def plot_selected_trust_distribution(
    analysis: dict, output_dir: Path
) -> Path:
    """Plot non-cumulative C trust mass for selected correctness groups."""

    figure, axis = plt.subplots(figsize=(7.8, 5.2))
    arrays = analysis["arrays"]
    correct = arrays["selected_correct"]
    _plot_mass(
        axis,
        arrays["selected_trust"][correct],
        label="B-selected correct",
        color=COLORS["trust"],
    )
    _plot_mass(
        axis,
        arrays["selected_trust"][~correct],
        label="B-selected wrong",
        color=COLORS["loss"],
    )
    trust_threshold = analysis["routing"].get("trust_threshold")
    if trust_threshold is not None:
        axis.axvline(
            trust_threshold,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Frozen validation C threshold",
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Scheme C selected trust  q(x)")
    axis.set_ylabel("Probability mass per fixed-width bin")
    axis.set_title(
        f"{_name(analysis)}: Scheme C non-cumulative trust distribution"
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=9)
    return _save(
        figure,
        output_dir,
        f"selected_trust_distribution_{analysis['dataset']}",
    )


def plot_trust_rank_composition(analysis: dict, output_dir: Path) -> Path:
    """Plot exclusive node-state composition across C trust-rank segments."""

    rows = analysis["trust_rank_composition"]
    x = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    bottom = np.zeros(len(rows), dtype=float)
    specifications = (
        ("agreement_correct", "Agreement-correct", COLORS["both_correct"]),
        ("agreement_wrong", "Agreement-wrong", COLORS["agreement_wrong"]),
        ("gnn_only", "GNN-only", COLORS["gnn"]),
        ("llm_only", "LLM-only", COLORS["llm"]),
        ("conflict_neither", "Conflict-neither", COLORS["conflict_neither"]),
    )
    for key, label, color in specifications:
        values = np.asarray([row[f"{key}_fraction"] for row in rows])
        axis.bar(x, values, bottom=bottom, color=color, label=label, width=0.78)
        bottom += values
    precision = np.asarray([row["precision"] for row in rows])
    precision_axis = axis.twinx()
    precision_axis.plot(
        x,
        precision,
        color="black",
        marker="o",
        linewidth=1.8,
        label="Segment precision",
    )
    precision_axis.set_ylim(0, 1.02)
    precision_axis.set_ylabel("B-selected label precision")
    axis.set_ylim(0, 1)
    axis.set_xticks(x, [f"D{row['segment']}" for row in rows])
    axis.set_xlabel("Scheme C trust rank segment (D1 highest → D10 lowest)")
    axis.set_ylabel("Node-type fraction")
    axis.set_title(f"{_name(analysis)}: composition by Scheme C trust rank")
    axis.grid(axis="y", alpha=0.2)
    handles, labels = axis.get_legend_handles_labels()
    p_handles, p_labels = precision_axis.get_legend_handles_labels()
    axis.legend(
        handles + p_handles,
        labels + p_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        fontsize=8,
    )
    return _save(
        figure,
        output_dir,
        f"trust_rank_composition_{analysis['dataset']}",
    )


def generate_all_figures(analyses: Sequence[dict], output_dir: Path | str) -> list[Path]:
    """Generate all required figures, returning the nine PNG paths."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    plotters: tuple[Callable[[Sequence[dict], Path], Path], ...] = (
        plot_preference_distribution,
        plot_preference_calibration,
        plot_correction_loss,
        plot_oracle_recovery,
        plot_decisive_confusion,
        plot_threshold_sensitivity,
        plot_scheme_c_reliability,
        plot_scheme_c_risk_coverage,
        plot_four_state_distribution,
    )
    return [plotter(analyses, target) for plotter in plotters]


def generate_extended_figures(
    analyses: Sequence[dict], output_dir: Path | str
) -> list[Path]:
    """Generate per-dataset non-cumulative and trust-composition figures."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    result: list[Path] = []
    for analysis in analyses:
        result.extend(
            [
                plot_preference_score_distribution(analysis, target),
                plot_selected_trust_distribution(analysis, target),
                plot_trust_rank_composition(analysis, target),
            ]
        )
    return result
