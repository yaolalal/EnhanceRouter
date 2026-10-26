"""Guarded orchestration and reporting for frozen Router B/C diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Mapping, Sequence

import numpy as np

from .analysis import analyze_dataset, risk_coverage, threshold_sensitivity
from .plotting import (
    DISPLAY_NAMES,
    generate_all_figures,
    generate_extended_figures,
)


DATASET_ORDER = ["cora", "citeseer", "pubmed", "arxiv", "ogbn-products"]
TOLERANCE = 2e-6


def _display(dataset: str) -> str:
    return DISPLAY_NAMES.get(dataset, dataset)


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:invalid_jsonl_line:{line_number}") from error
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _dataset_dirs(input_root: Path) -> list[Path]:
    candidates = [
        path
        for path in input_root.iterdir()
        if path.is_dir()
        and (path / "metrics.json").is_file()
        and (path / "decisions.jsonl").is_file()
    ]
    order = {name: index for index, name in enumerate(DATASET_ORDER)}
    return sorted(candidates, key=lambda path: (order.get(path.name, 999), path.name))


def _assert_close(dataset: str, name: str, actual, expected) -> None:
    if expected is None:
        return
    if actual is None or not math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=TOLERANCE
    ):
        raise ValueError(
            f"{dataset}:metric_mismatch:{name}:"
            f"recomputed={actual}:stored={expected}"
        )


def _assert_equal(dataset: str, name: str, actual, expected) -> None:
    if expected is None:
        return
    if actual != expected:
        raise ValueError(
            f"{dataset}:metric_mismatch:{name}:"
            f"recomputed={actual}:stored={expected}"
        )


def _precision_at_coverages(
    correctness: np.ndarray, trust: np.ndarray
) -> dict[str, float]:
    order = np.lexsort((np.arange(trust.size), -trust))
    sorted_correct = correctness[order].astype(float)
    result = {}
    for coverage in (0.10, 0.25, 0.50, 0.75, 1.00):
        count = min(trust.size, max(1, int(math.ceil(coverage * trust.size))))
        result[f"{coverage:.2f}"] = float(np.mean(sorted_correct[:count]))
    return result


def _augment_analysis(analysis: dict) -> None:
    arrays = analysis["arrays"]
    analysis["threshold_sensitivity"] = threshold_sensitivity(
        arrays, analysis["routing"]["threshold"]
    )
    curve = risk_coverage(arrays["selected_correct"], arrays["selected_trust"])
    analysis["risk_coverage"] = curve
    analysis["trust"]["aurc"] = float(
        np.mean([point["risk"] for point in curve])
    )
    analysis["trust"]["precision_at_coverage"] = _precision_at_coverages(
        arrays["selected_correct"], arrays["selected_trust"]
    )


def _audit_against_metrics(analysis: dict, metrics: Mapping) -> dict:
    dataset = analysis["dataset"]
    checks: dict[str, bool] = {}
    saved_b = metrics.get("scheme_b", {}).get("test", {})
    routing = analysis["routing"]
    pairs = {
        "count": (analysis["counts"]["all"], saved_b.get("count")),
        "conflict_count": (
            analysis["counts"]["conflict"],
            saved_b.get("conflict_count"),
        ),
        "decisive_conflict_count": (
            analysis["counts"]["decisive"],
            saved_b.get("decisive_conflict_count"),
        ),
        "gnn_accuracy": (routing["gnn_accuracy"], saved_b.get("gnn_accuracy")),
        "llm_accuracy": (routing["llm_accuracy"], saved_b.get("llm_accuracy")),
        "best_single_accuracy": (
            routing["best_single_accuracy"],
            saved_b.get("best_single_accuracy"),
        ),
        "router_accuracy": (
            routing["router_accuracy"],
            saved_b.get("router_accuracy"),
        ),
        "gain_over_best_single": (
            routing["gain_over_best_single"],
            saved_b.get("gain_over_best_single"),
        ),
        "oracle_accuracy": (
            routing["oracle_accuracy"],
            saved_b.get("oracle_accuracy"),
        ),
        "oracle_gap": (routing["oracle_gap"], saved_b.get("oracle_gap")),
        "gnn_route_rate_on_conflict": (
            routing["gnn_route_rate_on_conflict"],
            saved_b.get("gnn_route_rate_on_conflict"),
        ),
        "preference_auroc": (
            analysis["preference"]["auroc"],
            saved_b.get("preference_auroc"),
        ),
    }
    integer_names = {"count", "conflict_count", "decisive_conflict_count"}
    for name, (actual, expected) in pairs.items():
        if expected is not None:
            if name in integer_names:
                _assert_equal(dataset, name, actual, expected)
            else:
                _assert_close(dataset, name, actual, expected)
        checks[name] = True

    saved_paired = saved_b.get("paired_vs_best_single", {})
    paired = {
        "comparator": routing["comparator"],
        "wins": routing["wins"],
        "losses": routing["losses"],
        "ties": routing["ties"],
        "net_correct": routing["net_correct"],
    }
    for name, actual in paired.items():
        expected = saved_paired.get(name)
        if expected is not None:
            _assert_equal(dataset, f"paired_{name}", actual, expected)
        checks[f"paired_{name}"] = True

    saved_c = (
        metrics.get("scheme_c", {}).get("metrics", {}).get("selected", {})
    )
    trust = analysis["trust"]
    for name in ("auroc", "nll", "brier", "ece", "aurc"):
        expected = saved_c.get(name)
        if expected is not None:
            _assert_close(dataset, f"scheme_c_{name}", trust[name], expected)
        checks[f"scheme_c_{name}"] = True
    saved_coverages = saved_c.get("precision_at_coverage", {})
    for coverage, actual in trust["precision_at_coverage"].items():
        expected = saved_coverages.get(coverage)
        if expected is not None:
            _assert_close(
                dataset,
                f"scheme_c_precision_at_{coverage}",
                actual,
                expected,
            )
        checks[f"scheme_c_precision_at_{coverage}"] = True

    _assert_equal(
        dataset,
        "loss_node_count",
        len(analysis["loss_nodes"]),
        routing["losses"],
    )
    checks["loss_node_count"] = True
    composition = analysis["trust_rank_composition"]
    composition_count = sum(row["count"] for row in composition)
    _assert_equal(
        dataset,
        "trust_rank_composition_count",
        composition_count,
        analysis["counts"]["all"],
    )
    checks["trust_rank_composition_count"] = True
    for row in composition:
        category_count = sum(
            row[f"{name}_count"]
            for name in (
                "agreement_correct",
                "agreement_wrong",
                "gnn_only",
                "llm_only",
                "conflict_neither",
            )
        )
        _assert_equal(
            dataset,
            f"trust_rank_segment_{row['segment']}_count",
            category_count,
            row["count"],
        )
        checks[f"trust_rank_segment_{row['segment']}_count"] = True
    return {"all_checks_pass": all(checks.values()), "checks": checks}


def _summary_row(analysis: dict) -> dict:
    counts = analysis["counts"]
    preference = analysis["preference"]
    routing = analysis["routing"]
    trust = analysis["trust"]
    return {
        "dataset": analysis["dataset"],
        "test_count": counts["all"],
        "both_correct": counts["both_correct"],
        "gnn_only": counts["gnn_only"],
        "llm_only": counts["llm_only"],
        "both_wrong": counts["both_wrong"],
        "decisive_count": counts["decisive"],
        "b_checkpoint": analysis["checkpoint"],
        "b_view": analysis["view"],
        "routing_threshold": routing["threshold"],
        "preference_auroc": preference["auroc"],
        "preference_average_precision": preference["average_precision"],
        "preference_ks": preference["ks_distance"],
        "preference_nll": preference["nll"],
        "preference_brier": preference["brier"],
        "preference_ece": preference["ece"],
        "preference_calibration_intercept": preference[
            "calibration_intercept"
        ],
        "preference_calibration_slope": preference["calibration_slope"],
        "gnn_only_median_margin": preference["gnn_only_median_margin"],
        "llm_only_median_margin": preference["llm_only_median_margin"],
        "gnn_only_recall": preference["gnn_only_recall"],
        "llm_only_recall": preference["llm_only_recall"],
        "decisive_accuracy": preference["decisive_accuracy"],
        "balanced_decisive_accuracy": preference["balanced_accuracy"],
        "gnn_accuracy": routing["gnn_accuracy"],
        "llm_accuracy": routing["llm_accuracy"],
        "best_single_accuracy": routing["best_single_accuracy"],
        "router_accuracy": routing["router_accuracy"],
        "gain_over_best_single": routing["gain_over_best_single"],
        "oracle_accuracy": routing["oracle_accuracy"],
        "oracle_gap": routing["oracle_gap"],
        "oracle_recovery": routing["oracle_recovery"],
        "comparator": routing["comparator"],
        "wins": routing["wins"],
        "losses": routing["losses"],
        "ties": routing["ties"],
        "net_correct": routing["net_correct"],
        "trust_checkpoint": analysis["trust_checkpoint"],
        "trust_threshold": routing["trust_threshold"],
        "trust_auroc": trust["auroc"],
        "trust_nll": trust["nll"],
        "trust_brier": trust["brier"],
        "trust_ece": trust["ece"],
        "trust_aurc": trust["aurc"],
        "trust_precision_at_10": trust["precision_at_coverage"]["0.10"],
        "trust_precision_at_25": trust["precision_at_coverage"]["0.25"],
        "trust_precision_at_50": trust["precision_at_coverage"]["0.50"],
        "trust_precision_at_75": trust["precision_at_coverage"]["0.75"],
        "trust_precision_at_100": trust["precision_at_coverage"]["1.00"],
    }


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_jsonable(list(rows)))


def _format(value, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _markdown_summary_table(analyses: Sequence[dict]) -> str:
    lines = [
        "| Dataset | GNN | LLM | Best single | Scheme B | Gain | Oracle | "
        "Oracle recovery | Wins/Losses |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in analyses:
        routing = item["routing"]
        recovery = (
            "NA"
            if routing["oracle_recovery"] is None
            else f"{routing['oracle_recovery'] * 100:.1f}%"
        )
        lines.append(
            f"| {_display(item['dataset'])} "
            f"| {_format(routing['gnn_accuracy'])} "
            f"| {_format(routing['llm_accuracy'])} "
            f"| {_format(routing['best_single_accuracy'])} "
            f"| **{_format(routing['router_accuracy'])}** "
            f"| {routing['gain_over_best_single'] * 100:+.2f} pp "
            f"| {_format(routing['oracle_accuracy'])} "
            f"| {recovery} "
            f"| {routing['wins']:,}/{routing['losses']:,} |"
        )
    return "\n".join(lines)


def _markdown_preference_table(analyses: Sequence[dict]) -> str:
    lines = [
        "| Dataset | Decisive | G-only/L-only | AUROC | AP | KS | "
        "Pref-NLL | Pref-Brier | Pref-ECE | Balanced acc | "
        "G-only recall | L-only recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in analyses:
        counts = item["counts"]
        pref = item["preference"]
        lines.append(
            f"| {_display(item['dataset'])} "
            f"| {counts['decisive']:,} "
            f"| {counts['gnn_only']:,}/{counts['llm_only']:,} "
            f"| {_format(pref['auroc'])} "
            f"| {_format(pref['average_precision'])} "
            f"| {_format(pref['ks_distance'])} "
            f"| {_format(pref['nll'])} "
            f"| {_format(pref['brier'])} "
            f"| {_format(pref['ece'])} "
            f"| {_format(pref['balanced_accuracy'])} "
            f"| {_format(pref['gnn_only_recall'])} "
            f"| {_format(pref['llm_only_recall'])} |"
        )
    return "\n".join(lines)


def _markdown_loss_table(analyses: Sequence[dict]) -> str:
    lines = [
        "| Dataset | Comparator | Losses | State of every loss | "
        "Median absolute margin | Median trust | C accepted | D1–D3 | D8–D10 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for item in analyses:
        summary = item["loss_summary"]
        early_count = sum(
            summary[f"trust_decile_{segment}_count"]
            for segment in range(1, 4)
        )
        late_count = sum(
            summary[f"trust_decile_{segment}_count"]
            for segment in range(8, 11)
        )
        loss_count = summary["loss_count"]
        lines.append(
            f"| {_display(item['dataset'])} "
            f"| {summary['comparator'].upper()} "
            f"| {loss_count:,} "
            f"| {summary['state'] or 'none'} "
            f"| {_format(summary['absolute_decision_margin_median'])} "
            f"| {_format(summary['selected_trust_median'])} "
            f"| {_format(summary['accepted_fraction'])} "
            f"| {_format(early_count / loss_count if loss_count else None)} "
            f"| {_format(late_count / loss_count if loss_count else None)} |"
        )
    return "\n".join(lines)


def _top_coverage_composition(
    analysis: dict, coverage: float = 0.25
) -> dict[str, float | int]:
    arrays = analysis["arrays"]
    count = arrays["selected_trust"].size
    selected_count = min(count, max(1, int(math.ceil(coverage * count))))
    order = np.lexsort((arrays["node_id"], -arrays["selected_trust"]))
    selected = order[:selected_count]
    agreement = arrays["agreement"][selected]
    return {
        "count": selected_count,
        "agreement_fraction": float(np.mean(agreement)),
        "precision": float(np.mean(arrays["selected_correct"][selected])),
    }


def _markdown_rank_table(analyses: Sequence[dict]) -> str:
    lines = [
        "| Dataset | D1 agreement | D1 precision | First conflict-majority "
        "decile | D10 conflict-neither | Top-25% agreement | "
        "Precision@25% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in analyses:
        composition = item["trust_rank_composition"]
        first = composition[0]
        last = composition[-1]
        first_conflict = next(
            (
                row["segment"]
                for row in composition
                if row["agreement_fraction"] < 0.5
            ),
            None,
        )
        top = _top_coverage_composition(item)
        lines.append(
            f"| {_display(item['dataset'])} "
            f"| {_format(first['agreement_fraction'])} "
            f"| {_format(first['precision'])} "
            f"| {'NA' if first_conflict is None else f'D{first_conflict}'} "
            f"| {_format(last['conflict_neither_fraction'])} "
            f"| {_format(top['agreement_fraction'])} "
            f"| {_format(top['precision'])} |"
        )
    return "\n".join(lines)


def _dataset_figure_block(
    analyses: Sequence[dict],
    figure_prefix: str,
    filename_prefix: str,
    title: str,
) -> str:
    lines: list[str] = []
    for item in analyses:
        display = _display(item["dataset"])
        lines.extend(
            [
                f"### {title}: {display}",
                "",
                f"![{title}: {display}]"
                f"({figure_prefix}/{filename_prefix}_{item['dataset']}.png)",
                "",
            ]
        )
    return "\n".join(lines)


def _markdown_trust_table(analyses: Sequence[dict]) -> str:
    lines = [
        "| Dataset | Selected C | AUROC | NLL | Brier | ECE | AURC | "
        "Precision@25% |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in analyses:
        trust = item["trust"]
        lines.append(
            f"| {_display(item['dataset'])} "
            f"| {item['trust_checkpoint']} "
            f"| {_format(trust['auroc'])} "
            f"| {_format(trust['nll'])} "
            f"| {_format(trust['brier'])} "
            f"| {_format(trust['ece'])} "
            f"| {_format(trust['aurc'])} "
            f"| {_format(trust['precision_at_coverage']['0.25'])} |"
        )
    return "\n".join(lines)


def _dataset_interpretations(analyses: Sequence[dict]) -> str:
    lines = []
    for item in analyses:
        counts = item["counts"]
        pref = item["preference"]
        routing = item["routing"]
        trust = item["trust"]
        calibration_phrase = (
            "relatively small"
            if pref["ece"] < 0.10
            else "material and should not be ignored"
        )
        lines.extend(
            [
                f"### {_display(item['dataset'])}",
                "",
                f"- Scheme B separates GNN-only from LLM-only with AUROC "
                f"`{pref['auroc']:.4f}` over `{counts['decisive']:,}` decisive "
                f"rows; frozen-threshold balanced decisive accuracy is "
                f"`{pref['balanced_accuracy']:.4f}`.",
                f"- Relative to the `{routing['comparator'].upper()}` best-single "
                f"comparator, it corrects `{routing['wins']:,}` nodes and "
                f"introduces `{routing['losses']:,}` errors, for a net gain of "
                f"`{routing['net_correct']:+,}` nodes "
                f"(`{routing['gain_over_best_single'] * 100:+.2f}` pp).",
                f"- Conditional preference ECE is `{pref['ece']:.4f}`; this is "
                f"{calibration_phrase}. It is a diagnostic of the conditional "
                f"probability reading, not the main routing claim.",
                f"- Scheme C `{item['trust_checkpoint']}` trust has AUROC "
                f"`{trust['auroc']:.4f}`, ECE `{trust['ece']:.4f}`, and "
                f"Precision@25% `{trust['precision_at_coverage']['0.25']:.4f}`.",
                "",
            ]
        )
    return "\n".join(lines)


def render_report(
    analyses: Sequence[dict],
    input_root: Path,
    output_root_name: str,
    figure_prefix: str = "figures",
    document_path_name: str = "docs/router_bc_supplementary_experiments_2026-07-28-v3.md",
    generated_date: str = "2026-07-28",
) -> str:
    """Render the complete human-readable supplementary report."""

    figures = [
        ("Preference distribution", "preference_distribution.png"),
        ("Conditional preference calibration", "preference_calibration.png"),
        ("Correction–loss decomposition", "correction_loss_decomposition.png"),
        ("Oracle-gap recovery", "oracle_gap_recovery.png"),
        ("Decisive routing confusion", "decisive_confusion.png"),
        ("Retrospective threshold sensitivity", "threshold_sensitivity.png"),
        ("Scheme C reliability", "scheme_c_reliability.png"),
        ("Scheme C risk–coverage", "scheme_c_risk_coverage.png"),
        ("Four-state score diagnostic", "four_state_distribution.png"),
    ]
    figure_lines = []
    for index, (title, filename) in enumerate(figures, start=1):
        figure_lines.extend(
            [
                f"### Figure {index}: {title}",
                "",
                f"![{title}]({figure_prefix}/{filename})",
                "",
            ]
        )
    figure_block = "\n".join(figure_lines)
    preference_distribution_block = _dataset_figure_block(
        analyses,
        figure_prefix,
        "preference_score_distribution",
        "Scheme B probability-mass distribution",
    )
    trust_distribution_block = _dataset_figure_block(
        analyses,
        figure_prefix,
        "selected_trust_distribution",
        "Scheme C probability-mass distribution",
    )
    rank_composition_block = _dataset_figure_block(
        analyses,
        figure_prefix,
        "trust_rank_composition",
        "Scheme C trust-rank composition",
    )
    all_gain = all(
        item["routing"]["gain_over_best_single"] > 0 for item in analyses
    )
    top_25_agreement = [
        _top_coverage_composition(item)["agreement_fraction"]
        for item in analyses
    ]
    top_25_description = (
        "almost entirely agreement rows"
        if min(top_25_agreement) >= 0.99
        else "a mixture of agreement and conflict rows"
    )
    return f"""# Router B/C Supplementary Diagnostics

> Generated: {generated_date}  
> Frozen input: `{input_root}`  
> Artifact root: `{output_root_name}`

## 1. Executive conclusion

This is a frozen post-hoc evaluation of the already selected Router B/C
checkpoints. It performs no retraining, checkpoint selection, calibration
fitting, or threshold selection. Test labels are used only to evaluate the
frozen decisions.

The central findings are:

- Scheme B's score is evaluated as a **decisive preference score**. Its
  distribution separation and its conditional preference calibration are shown
  in different figures because they answer different questions.
- Full-coverage Scheme B {'exceeds' if all_gain else 'does not uniformly exceed'}
  the test-set best single expert across the analyzed datasets. The
  correction–loss figure shows whether the net gain comes from more recovered
  complementary cases than introduced mistakes.
- Scheme C is evaluated separately as selected-label trust. Scheme C does not
  alter Scheme B's route or full-coverage accuracy.
- The threshold curves are **retrospective test sensitivity** diagnostics. The
  black-star point is the original official-validation-selected threshold; no
  test-optimal threshold is promoted as a result.

## 2. Exact semantics

For each row, let

$$
Y_G=\\mathbf 1(\\hat y_G=y),\\qquad
Y_L=\\mathbf 1(\\hat y_L=y).
$$

The four mutually exclusive states are both-correct, GNN-only, LLM-only, and
both-wrong. Scheme B is trained on the decisive set

$$
\\mathcal D=\\{{i:Y_{{G,i}}\\ne Y_{{L,i}}\\}},
$$

with GNN-only as the positive class and LLM-only as the negative class.
Consequently, `GNN-correct` and `LLM-correct` are not used as two score groups:
they overlap on both-correct rows.

The cross-dataset distribution plot uses

$$
m(x)=\\operatorname{{logit}}c(x)-\\operatorname{{logit}}\\tau_B,
$$

so zero is the frozen decision boundary in every dataset. The calibration plot
uses raw $c(x)$ only on decisive rows and asks whether it numerically matches
the empirical conditional GNN-only frequency. Because selected endpoints can
contain pairwise loss or legacy nonlinear scores, this **conditional preference
calibration** is diagnostic; it does not turn $c(x)$ into an unconditional
GNN-correctness probability.

Scheme C instead evaluates

$$
q(x)=P(\\text{{Scheme B selected label is correct}}\\mid x),
$$

which is the appropriate score for conventional selected-label calibration and
risk–coverage analysis.

## 3. Experimental protocol

- Inputs are the immutable `decisions.jsonl` and `metrics.json` files from the
  audited final run.
- The saved validation-selected Scheme B endpoint and $\\tau_B$ are retained.
- The saved validation-selected Scheme C checkpoint and trust scores are
  retained.
- All node-level predictions, correctness fields, routes, headline metrics,
  paired wins/losses, and Scheme C metrics are independently recomputed.
- Fixed-width 10-bin ECE is used for both calibration diagnostics.
- Binomial empirical-rate intervals are Wilson 95% intervals.
- Best single is the post-hoc maximum of test GNN and LLM accuracy.
- Oracle is the fraction of nodes where at least one expert is correct.

## 4. Primary result: full-coverage routing value

{_markdown_summary_table(analyses)}

Oracle-gap recovery is

$$
\\rho_{{oracle}}=
\\frac{{A_B-A_{{best}}}}{{A_{{oracle}}-A_{{best}}}},
$$

and measures the fraction of available best-single-to-Oracle headroom recovered
by the frozen Router.

## 5. Scheme B preference separation and calibration

{_markdown_preference_table(analyses)}

AUROC/AP/KS measure separation or ranking. Preference ECE/NLL/Brier and
calibration slope/intercept measure the optional conditional-probability
interpretation. A Router can therefore have useful routing separation while
remaining poorly calibrated as a probability estimator.

## 6. Scheme C selected-label trust

{_markdown_trust_table(analyses)}

Scheme C shares Scheme B's route. Its contribution is selective acceptance:
high-trust prefixes should have lower risk than full coverage.

## 7. Figures

{figure_block}
## 8. Non-cumulative Scheme B score distributions

Each panel below is a normalized fixed-bin histogram of the raw preference
score $c(x)$ on decisive test rows. The GNN-only and LLM-only histograms are
normalized independently, so each curve sums to one and is a probability-mass
distribution rather than a CDF. The vertical line is the frozen $\tau_B$.

{preference_distribution_block}
## 9. Non-cumulative Scheme C trust distributions

Each panel overlays independently normalized distributions of $q(x)$ for rows
where Scheme B's selected label is correct versus wrong. These plots test
separation directly; they do not change the selected Scheme B route.

{trust_distribution_block}
## 10. Per-dataset analysis

{_dataset_interpretations(analyses)}
## 11. Interpretation of the correction–loss result

Relative to best single:

- a **win** is a node where Scheme B is correct and best single is wrong;
- a **loss** is a node where Scheme B is wrong and best single is correct;
- a tie means their correctness is identical.

The exact identity

$$
A_B-A_{{best}}=\\frac{{W-L}}N
$$

is independently checked. This decomposition is more informative than accuracy
alone because it shows whether routing gains are produced by exploiting genuine
expert complementarity or by a small number of unstable switches.

## 12. Paired-loss node diagnostics

{_markdown_loss_table(analyses)}

Every paired loss is necessarily a decisive node: the best-single comparator
is correct, but Scheme B routes to the other, incorrect expert. Thus, when GNN
is the comparator every loss is GNN-only; when LLM is the comparator every loss
is LLM-only. These are not both-wrong nodes, because a both-wrong row cannot be
lost relative to either single expert.

The margin columns show whether the wrong switch was close to the frozen
Scheme B boundary; the trust and decile columns show whether Scheme C placed
these mistakes early or late in its ranking. `C accepted` uses the saved
validation-selected Scheme C trust threshold. Complete node-level fields are in
`loss_nodes.csv`, including node ID, labels, both predictions, route, $c(x)$,
$m(x)$, $q(x)$, trust acceptance, and trust-rank decile. The frozen artifacts do
not include node text or graph-neighborhood attributes, so no unsupported
semantic or structural characterization is claimed here.

## 13. Retrospective threshold sensitivity

The sensitivity figure evaluates frozen Scheme B scores at many hypothetical
test thresholds. It is included only to show whether behavior changes smoothly
around the original validation-selected threshold. The curve must not be used
to select a replacement threshold, quote a test-optimized result, or revise the
reported Scheme B accuracy.

## 14. Secondary four-state diagnostic

The four-state figure includes both-correct and both-wrong rows to show how the
preference model extrapolates outside its decisive training target. These states
do not enter preference ECE:

- routing is label-invariant on both-correct agreement rows;
- neither expert can recover a correct answer on both-wrong rows;
- only GNN-only versus LLM-only tests expert-choice preference.

## 15. Scheme C trust-rank composition

{_markdown_rank_table(analyses)}

{rank_composition_block}

The nodes are sorted by decreasing Scheme C trust and split into ten equal-size
rank segments (`D1` is highest trust). Bars partition every segment into five
exclusive groups: correct agreement, wrong agreement, GNN-only, LLM-only, and
conflict-neither. The black line is the empirical correctness of Scheme B in
that segment.

Across these frozen datasets, the top 25% consists of {top_25_description}.
Therefore, the reported Precision@25% is driven predominantly by easy agreement
rows and should not be read as evidence that Scheme C identifies the correct
expert on conflicts equally well. This is still a valid selected-label trust
result, but the composition table makes its source explicit.

There is no required order such as agreement, then GNN-only/LLM-only, then
neither. Scheme C predicts whether the already selected label is correct; it is
not trained as a four-state classifier. Conflict-neither tends to be enriched
in later, lower-trust segments when the trust score is informative, but its
presence is not required to be confined to the final segment. Likewise,
agreement-wrong rows may receive low trust even though they are agreement rows.

## 16. Limitations and publication-safe claims

1. These five test sets are development results because test outcomes were
   inspected during method development.
2. The test threshold sweep is retrospective and supplies no additional model
   selection evidence.
3. Node-level Wilson intervals do not capture graph dependence or retraining
   variance.
4. A single fixed support split cannot measure few-shot episode variance.
5. Best single is a strong post-hoc test comparator, not a deployable
   validation-selected expert policy.
6. OGBN-Products uses up-to-5-shot per available class rather than strict
   47-way 5-shot.
7. Rank-view target normalization uses all unlabeled target-node evidence and
   is transductive.
8. Conditional preference ECE must not be presented as calibration of
   unconditional GNN correctness.

The defensible claim is that the frozen Scheme B decisions recover more
complementary expert successes than the errors they introduce on these
development datasets, while Scheme C provides a separate selected-label trust
ranking. Generalization claims require newly frozen, untouched datasets or
pre-registered splits.

## 17. Reproduction

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
/home/wujingyao/miniconda3/envs/uncopy/bin/python \\
scripts/router_bc_supplementary/generate_supplementary_report.py \\
  --input-root runs/router_bc_improved_final_20260719_v1 \\
  --output-root runs/{output_root_name} \\
  --document-path {document_path_name} \\
  --generated-date {generated_date}
```

Machine-readable tables, the independent audit, and SHA256 provenance are
stored under `{output_root_name}`.
"""


def _write_manifest(root: Path) -> None:
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "manifest.sha256"
    )
    lines = [f"{_sha256(path)}  {path.relative_to(root)}" for path in paths]
    (root / "manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_supplementary(
    input_root: Path | str,
    output_root: Path | str,
    document_path: Path | str,
    generated_date: str = "2026-07-28",
) -> dict:
    """Run the complete guarded supplementary analysis."""

    source = Path(input_root).resolve()
    output = Path(output_root).resolve()
    document = Path(document_path).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"missing_input_root:{source}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"nonempty_output_root:{output}")
    if document.exists() and document.stat().st_size > 0:
        raise FileExistsError(f"existing_document_path:{document}")
    dataset_dirs = _dataset_dirs(source)
    if not dataset_dirs:
        raise ValueError(f"no_dataset_artifacts:{source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent)
    )
    analyses: list[dict] = []
    dataset_audits: list[dict] = []
    input_hashes: dict[str, str] = {}
    try:
        for dataset_dir in dataset_dirs:
            decisions_path = dataset_dir / "decisions.jsonl"
            metrics_path = dataset_dir / "metrics.json"
            rows = _read_jsonl(decisions_path)
            metrics = _read_json(metrics_path)
            analysis = analyze_dataset(rows, metrics)
            _augment_analysis(analysis)
            audit = _audit_against_metrics(analysis, metrics)
            analyses.append(analysis)
            dataset_audits.append(
                {
                    "dataset": analysis["dataset"],
                    **audit,
                    "recomputed": _summary_row(analysis),
                }
            )
            input_hashes[str(decisions_path.relative_to(source))] = _sha256(
                decisions_path
            )
            input_hashes[str(metrics_path.relative_to(source))] = _sha256(
                metrics_path
            )

        generate_all_figures(analyses, stage / "figures")
        generate_extended_figures(analyses, stage / "figures")
        summary_rows = [_summary_row(item) for item in analyses]
        _write_csv(stage / "summary_metrics.csv", summary_rows)
        (stage / "summary_metrics.json").write_text(
            json.dumps(_jsonable(summary_rows), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        preference_bins = []
        trust_bins = []
        threshold_rows = []
        count_rows = []
        loss_rows = []
        loss_summary_rows = []
        trust_rank_rows = []
        for analysis in analyses:
            dataset = analysis["dataset"]
            preference_bins.extend(
                {"dataset": dataset, "kind": "preference", **item}
                for item in analysis["preference_calibration_bins"]
            )
            trust_bins.extend(
                {"dataset": dataset, "kind": "selected_trust", **item}
                for item in analysis["trust_calibration_bins"]
            )
            threshold_rows.extend(
                {"dataset": dataset, **item}
                for item in analysis["threshold_sensitivity"]
            )
            count_rows.append({"dataset": dataset, **analysis["counts"]})
            loss_rows.extend(analysis["loss_nodes"])
            loss_summary_rows.append(analysis["loss_summary"])
            trust_rank_rows.extend(
                {"dataset": dataset, **item}
                for item in analysis["trust_rank_composition"]
            )
        _write_csv(stage / "preference_calibration_bins.csv", preference_bins)
        _write_csv(stage / "scheme_c_calibration_bins.csv", trust_bins)
        _write_csv(stage / "threshold_sensitivity.csv", threshold_rows)
        _write_csv(stage / "four_state_counts.csv", count_rows)
        _write_csv(stage / "loss_nodes.csv", loss_rows)
        _write_csv(stage / "loss_summary.csv", loss_summary_rows)
        _write_csv(stage / "trust_rank_composition.csv", trust_rank_rows)

        source_audit = source / "audit.json"
        if source_audit.is_file():
            source_audit_payload = _read_json(source_audit)
            if source_audit_payload.get("all_checks_pass") is not True:
                raise ValueError("source_audit_not_passing")
            input_hashes["audit.json"] = _sha256(source_audit)
        source_summary = source / "summary.json"
        if source_summary.is_file():
            input_hashes["summary.json"] = _sha256(source_summary)
        source_config = source / "config.json"
        if source_config.is_file():
            input_hashes["config.json"] = _sha256(source_config)

        audit_payload = {
            "all_checks_pass": all(
                item["all_checks_pass"] for item in dataset_audits
            ),
            "input_root": str(source),
            "input_hashes": input_hashes,
            "datasets": dataset_audits,
            "protocol": {
                "frozen_post_selection_evaluation": True,
                "retraining": False,
                "checkpoint_reselection": False,
                "threshold_reselection": False,
                "test_threshold_curve": "retrospective_sensitivity_only",
                "preference_calibration_condition": "gnn_only_vs_llm_only",
                "extended_distribution_plots": "normalized_probability_mass_not_cdf",
                "loss_definition": "best_single_correct_and_router_wrong",
                "trust_rank_segments": 10,
            },
        }
        (stage / "audit.json").write_text(
            json.dumps(_jsonable(audit_payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        document_arg = os.path.relpath(document, Path.cwd())
        report = render_report(
            analyses,
            source,
            output.name,
            "figures",
            document_arg,
            generated_date,
        )
        (stage / "REPORT.md").write_text(report, encoding="utf-8")
        _write_manifest(stage)

        if output.exists():
            output.rmdir()
        os.replace(stage, output)
        document.parent.mkdir(parents=True, exist_ok=True)
        doc_report = render_report(
            analyses,
            source,
            output.name,
            f"../runs/{output.name}/figures",
            document_arg,
            generated_date,
        )
        document.write_text(doc_report, encoding="utf-8")
        return _jsonable(audit_payload)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
