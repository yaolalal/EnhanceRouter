from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eargtc.router_bc_posthoc_diagnostics.density import bounded_density


DATASETS = ("cora", "citeseer", "pubmed", "arxiv", "ogbn-products")
DISPLAY_NAMES = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
    "arxiv": "Arxiv",
    "ogbn-products": "OGBN-Products",
}
MODULES = (
    ("conflict", "Disagreement / C_conflict"),
    ("agreement", "Agreement / C_agreement"),
)
SCORE_FIELD = "split_calibrated_reliability"
CORRECT_COLOR = "#2f8f3a"
WRONG_COLOR = "#d64f4f"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid_jsonl:{path}:{line_number}") from error
    if not rows:
        raise ValueError(f"empty_decisions:{path}")
    return rows


def _load_panel(input_root: Path, dataset: str, module: str) -> dict:
    path = input_root / dataset / f"decisions_{module}.jsonl"
    rows = _read_jsonl(path)
    scores = []
    correctness = []
    for line_number, row in enumerate(rows, start=1):
        if row.get("module") != module:
            raise ValueError(f"module_mismatch:{path}:{line_number}")
        if not isinstance(row.get("correct"), bool):
            raise ValueError(f"invalid_correctness:{path}:{line_number}")
        try:
            score = float(row[SCORE_FIELD])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid_score:{path}:{line_number}:{SCORE_FIELD}") from error
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"score_out_of_range:{path}:{line_number}")
        scores.append(score)
        correctness.append(row["correct"])
    score_array = np.asarray(scores, dtype=float)
    correct_array = np.asarray(correctness, dtype=bool)
    if not np.any(correct_array) or not np.any(~correct_array):
        raise ValueError(f"both_correctness_groups_required:{path}")
    return {
        "dataset": dataset,
        "module": module,
        "scores": score_array,
        "correct": correct_array,
        "source": str(path),
    }


def _deterministic_sample(values: np.ndarray, key: str, limit: int = 20_000) -> np.ndarray:
    if values.size <= limit:
        return values
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")
    indices = np.random.default_rng(seed).choice(values.size, size=limit, replace=False)
    return values[indices]


def _profile(values: np.ndarray, key: str, bin_edges: np.ndarray, grid: np.ndarray) -> dict:
    sample = _deterministic_sample(values, key)
    profile = bounded_density(sample, bin_edges=bin_edges, grid=grid)
    histogram, _ = np.histogram(values, bins=bin_edges, density=True)
    profile["hist_density"] = histogram
    profile["count"] = int(values.size)
    profile["median"] = float(np.median(values))
    return profile


def _summary(panel: dict) -> dict:
    scores = panel["scores"]
    correct = panel["correct"]
    return {
        "dataset": panel["dataset"],
        "module": panel["module"],
        "score_field": SCORE_FIELD,
        "count": int(scores.size),
        "correct_count": int(correct.sum()),
        "wrong_count": int((~correct).sum()),
        "correct_rate": float(correct.mean()),
        "correct_median": float(np.median(scores[correct])),
        "wrong_median": float(np.median(scores[~correct])),
        "top25_cutoff": float(np.quantile(scores, 0.75)),
        "source": panel["source"],
    }


def _write_stats(rows: Iterable[dict], path: Path) -> None:
    rows = list(rows)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_score_distribution_figure(input_root: Path, output_root: Path) -> dict:
    """Render calibrated split-C score densities for five datasets and two modules."""

    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    panels = {
        (dataset, module): _load_panel(input_root, dataset, module)
        for dataset in DATASETS
        for module, _ in MODULES
    }
    summaries = [
        _summary(panels[(dataset, module)])
        for dataset in DATASETS
        for module, _ in MODULES
    ]

    bin_edges = np.linspace(0.0, 1.0, 31)
    grid = np.linspace(0.0, 1.0, 401)
    figure, axes = plt.subplots(5, 2, figsize=(18, 23), sharex=True, squeeze=False)
    for row_index, dataset in enumerate(DATASETS):
        for column_index, (module, module_title) in enumerate(MODULES):
            axis = axes[row_index, column_index]
            panel = panels[(dataset, module)]
            summary = _summary(panel)
            scores = panel["scores"]
            correct = panel["correct"]
            correct_profile = _profile(
                scores[correct], f"{dataset}:{module}:correct", bin_edges, grid
            )
            wrong_profile = _profile(
                scores[~correct], f"{dataset}:{module}:wrong", bin_edges, grid
            )
            widths = np.diff(bin_edges)
            axis.bar(
                bin_edges[:-1],
                correct_profile["hist_density"],
                width=widths,
                align="edge",
                color=CORRECT_COLOR,
                alpha=0.16,
                linewidth=0,
            )
            axis.bar(
                bin_edges[:-1],
                wrong_profile["hist_density"],
                width=widths,
                align="edge",
                color=WRONG_COLOR,
                alpha=0.16,
                linewidth=0,
            )
            axis.plot(
                grid,
                correct_profile["kde_density"],
                color=CORRECT_COLOR,
                linewidth=2.0,
                label=(
                    f"Correct (n={summary['correct_count']:,}, "
                    f"median={summary['correct_median']:.3f})"
                ),
            )
            axis.plot(
                grid,
                wrong_profile["kde_density"],
                color=WRONG_COLOR,
                linewidth=2.0,
                label=(
                    f"Wrong (n={summary['wrong_count']:,}, "
                    f"median={summary['wrong_median']:.3f})"
                ),
            )
            axis.axvline(
                summary["correct_median"],
                color=CORRECT_COLOR,
                linestyle=":",
                linewidth=1.4,
            )
            axis.axvline(
                summary["wrong_median"],
                color=WRONG_COLOR,
                linestyle=":",
                linewidth=1.4,
            )
            axis.axvline(
                summary["top25_cutoff"],
                color="#202020",
                linestyle="--",
                linewidth=1.4,
                label=f"Within-panel top-25% cutoff={summary['top25_cutoff']:.3f}",
            )
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(bottom=0.0)
            axis.grid(alpha=0.18)
            axis.set_title(
                f"{DISPLAY_NAMES[dataset]} — {module_title}\n"
                f"accuracy={summary['correct_rate']:.1%}, n={summary['count']:,}",
                fontsize=13,
            )
            axis.set_ylabel("Probability density")
            axis.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
            if row_index == len(DATASETS) - 1:
                axis.set_xlabel("Calibrated Scheme C reliability score  q(x)")

    figure.suptitle(
        "Router-v6 Split Scheme C — Correct vs. Wrong Reliability Distributions\n"
        "C_conflict and C_agreement are ranked separately",
        fontsize=18,
        y=0.998,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.985), h_pad=2.0, w_pad=1.3)
    stem = "router_v6_split_c_correctness_density_10panel"
    png_path = output_root / f"{stem}.png"
    pdf_path = output_root / f"{stem}.pdf"
    csv_path = output_root / f"{stem}_stats.csv"
    figure.savefig(png_path, dpi=240, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    _write_stats(summaries, csv_path)
    return {
        "panel_count": len(summaries),
        "png": str(png_path),
        "pdf": str(pdf_path),
        "stats_csv": str(csv_path),
        "score_field": SCORE_FIELD,
    }

