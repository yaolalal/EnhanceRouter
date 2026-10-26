"""Verify frozen v5 results and publish bounded comparison artifacts."""
import csv
import itertools
import json
import os
import statistics
from pathlib import Path

from eargtc.joint_cross_repair.artifacts import read_json, sha256, verify_files, write_json

ROOT = Path(__file__).resolve().parents[2]


def load_performance(path, historical_old_five=False):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for raw in rows:
        row = dict(raw)
        for key, value in tuple(row.items()):
            if key not in {"router", "dataset", "source"} and value not in {"", None}:
                row[key] = float(value)
        if historical_old_five:
            row["router_accuracy"] = row["accuracy"] - row["router_delta"]
        result.append(row)
    return result


def verify_preserved_snapshot(snapshot_path, complete_path):
    snapshot_path, complete_path = Path(snapshot_path).resolve(), Path(complete_path).resolve()
    snapshot, complete = read_json(snapshot_path), read_json(complete_path)
    files = snapshot.get("files", {})
    if (complete.get("all_checks_pass") is not True
            or complete.get("file_count") != len(files)
            or complete.get("snapshot_sha256") != sha256(snapshot_path)):
        raise ValueError("preserved snapshot completion receipt differs")
    records = {}
    for relative, digest in files.items():
        local = ROOT / relative
        if not local.exists():
            local = snapshot_path.parent / relative
        records[str(local)] = digest
    verify_files(records)
    return len(files)


def _write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path, details, datasets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = list(range(len(datasets)))
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.bar([value - 0.19 for value in x], [100 * row["delta_v4"] for row in details], width=0.38,
             label="v5 Fixed − v4 Fixed", color="#377eb8")
    axis.bar([value + 0.19 for value in x], [100 * row["delta_router"] for row in details], width=0.38,
             label="v5 Fixed − Router-B", color="#e41a1c")
    axis.set_xticks(x, datasets)
    axis.set_ylabel("Accuracy change (percentage points)")
    axis.set_title("Validation-selected Joint-v5 Fixed accuracy changes")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def publish(run_root):
    from .experiment import verify_frozen

    run_root = Path(run_root).resolve()
    config = verify_frozen(run_root)
    docs = ROOT / "docs"
    report = docs / "joint_v5_fixed_final_report_2026-09-09.md"
    detail_csv = report.with_suffix(".csv")
    comparison_csv = docs / "joint_v5_fixed_comparison_2026-09-09.csv"
    visualization = run_root / "visualization"
    run_report = run_root / "REPORT.md"
    final_audit = run_root / "FINAL_AUDIT.json"
    if any(path.exists() for path in [report, detail_csv, comparison_csv, run_report, visualization, final_audit]):
        raise FileExistsError("publication outputs are immutable")
    evaluation = run_root / "evaluation"
    complete = read_json(evaluation / "complete.json")
    if complete.get("all_checks_pass") is not True or complete.get("evaluated_cells") != 5:
        raise ValueError("incomplete final evaluation")
    verify_files({str(evaluation / name): digest for name, digest in complete["artifacts"].items()})
    if (complete["frozen_sha256"] != sha256(run_root / "FROZEN.json")
            or complete["pretest_audit_sha256"] != sha256(run_root / "PRETEST_AUDIT.json")):
        raise ValueError("evaluation provenance differs")

    snapshot = Path(config["preserved_snapshot"])
    preserved_count = verify_preserved_snapshot(snapshot, snapshot.parent / "COMPLETE.json")
    if preserved_count != config["preserved_file_count"] or preserved_count != 787:
        raise ValueError("preserved file population differs")

    rows = load_performance(evaluation / "performance.csv")
    datasets = config["datasets"]
    keys = set(itertools.product(config["routers"], datasets))
    if len(rows) != 5 or {(row["router"], row["dataset"]) for row in rows} != keys:
        raise ValueError("wrong v5 result population")
    payload = read_json(evaluation / "results.json")
    current = {(row["router"], row["dataset"]): row for row in rows}
    history_root = Path(config["history_root"])
    history = load_performance(history_root / "evaluation/performance.csv")
    historical = {(row["router"], row["dataset"]): row for row in history}
    old_five_path = Path(config["old_five_root"]) / "evaluation/performance.csv"
    old_five = load_performance(old_five_path, historical_old_five=True)

    details = []
    for dataset in datasets:
        new, old = current["v8_v5", dataset], historical["v8_v4", dataset]
        if new["router_accuracy"] != old["router_accuracy"]:
            raise ValueError("v5 full Router-B differs from frozen v4")
        cell = payload["cells"]["v8_v5/" + dataset]
        if new["accuracy"] != cell["metrics"]["optimized"]["accuracy"]:
            raise ValueError("evaluation CSV/result metric differs")
        details.append({
            "dataset": dataset, "v5_router_b_accuracy": new["router_accuracy"],
            "v5_fixed_accuracy": new["accuracy"], "delta_router": new["delta_router"],
            "v4_fixed_accuracy": old["accuracy"], "delta_v4": new["accuracy"] - old["accuracy"],
            "macro_f1": new["macro_f1"], "nll": new["nll"],
            "ensemble_accuracy": new["accuracy"], "seed_accuracy_mean": new["seed_accuracy_mean"],
            "seed_accuracy_std": new["seed_accuracy_std"], "accepted_count": int(new["accepted_count"]),
            "conflict_count": int(new["conflict_count"]), "agreement_count": int(new["agreement_count"]),
            "selected_family": cell["selection"]["optimized"]["family"],
            "g_id": cell["selection"]["optimized"].get("g_id", ""),
            "l_id": cell["selection"]["optimized"].get("l_id", ""),
            "alpha": cell["selection"]["optimized"].get("alpha", ""),
            "beta": cell["selection"]["optimized"].get("beta", ""),
        })

    comparison = []
    for router in ("v8_v3", "v8_v4", "v8_v5"):
        source = current if router == "v8_v5" else historical
        item = {"router": router, **{dataset: source[router, dataset]["accuracy"] for dataset in datasets}}
        item["mean"] = statistics.fmean(item[dataset] for dataset in datasets)
        comparison.append(item)
    historical_eight = []
    for router in ["r6_legacy_c", "r6_split_c", "r6_expert_conf_c", "a20ms_split_c", "a20emb_v2_split_c"]:
        by_key = {(row["router"], row["dataset"]): row for row in old_five}
        item = {"router": router, **{dataset: by_key[router, dataset]["accuracy"] for dataset in datasets}}
        item["mean"] = statistics.fmean(item[dataset] for dataset in datasets)
        historical_eight.append(item)
    historical_eight.extend(comparison)

    docs = ROOT / "docs"
    report = docs / "joint_v5_fixed_final_report_2026-09-09.md"
    _write_csv(detail_csv, details)
    _write_csv(comparison_csv, historical_eight)
    visualization.mkdir(exist_ok=False)
    plot = visualization / "accuracy_change.png"
    _plot(plot, details, datasets)

    names = {"v8_v3": "Router-v8 v3 + Fixed", "v8_v4": "Router-v8 v4 + Fixed", "v8_v5": "Router-v8 v5 + Fixed"}
    lines = ["# Joint-v5 Fixed 最终报告", "",
        "v5 保持历史 Fixed 模型不变。接受的冲突节点沿用原交叉 CE；新增的接受一致节点只进入两个专家各自的 keep-KL 池，不增加 agreement CE，也不使用 v5 分数加权。完整 Router-B 概率与 v4 完全一致。", "",
        "## 主表：Fixed Joint 准确率（%）", "", "| 方法 | " + " | ".join(datasets) + " | 五数据集等权平均 |", "|---|" + "---:|" * 6]
    for item in comparison:
        lines.append("| " + names[item["router"]] + " | " + " | ".join(f"{100*item[key]:.2f}" for key in [*datasets, "mean"]) + " |")
    lines += ["", "## v5 逐数据集明细", "", "| 数据集 | Router-B % | v5 Fixed % | Δ B pp | Δ v4 pp | Macro-F1 % | NLL | 接受数（冲突 + 一致） | 单种子准确率均值±标准差 % | 验证集所选家族 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for item in details:
        lines.append(f"| {item['dataset']} | {100*item['v5_router_b_accuracy']:.2f} | {100*item['v5_fixed_accuracy']:.2f} | {100*item['delta_router']:+.2f} | {100*item['delta_v4']:+.2f} | {100*item['macro_f1']:.2f} | {item['nll']:.6f} | {item['accepted_count']} ({item['conflict_count']} + {item['agreement_count']}) | {100*item['seed_accuracy_mean']:.2f}±{100*item['seed_accuracy_std']:.2f} | {item['selected_family']} |")
    lines += ["", "准确率取三种子平均概率的 argmax。单种子均值和标准差描述同一划分上的三个独立初始化，因此可能与集成准确率不同，也不是跨划分置信区间。", "",
        "optimized 候选及全部超参数只在冻结的 model-selection validation half 上选择，并在读取 test truth 之前完成五格统一审计。历史 validation/test 结果此前已经被观察，因此本结果不能称为 test-best 模型或全新留出基准。接受集合精度也不等于完整 test 分类精度。", "",
        f"保护审计重新计算了历史 SNAPSHOT 所列全部 {preserved_count} 个文件的哈希。完整指标、逐种子结果、池和选模记录保存在 run evaluation artifacts。", ""]
    lines += ["- [逐格结果 CSV](joint_v5_fixed_final_report_2026-09-09.csv)", "- [八方案比较 CSV](joint_v5_fixed_comparison_2026-09-09.csv)", "- [准确率变化图](../runs/joint_v5_fixed_20260909_v1/visualization/accuracy_change.png)", ""]
    report.write_text("\n".join(lines))
    run_lines = lines[:-4] + [
        f"- [逐格结果 CSV]({os.path.relpath(detail_csv, run_root)})",
        f"- [八方案比较 CSV]({os.path.relpath(comparison_csv, run_root)})",
        f"- [准确率变化图]({os.path.relpath(plot, run_root)})", ""]
    run_report.write_text("\n".join(run_lines) + "\n")
    artifacts = [report, detail_csv, comparison_csv, run_report, plot]
    final = {"all_checks_pass": True, "cell_count": 5, "members": {"total": 135, "fresh": 120, "reused": 15},
        "preservation": {"snapshot_sha256": sha256(snapshot), "verified_file_count": preserved_count},
        "evaluation_complete_sha256": sha256(evaluation / "complete.json"),
        "artifacts": {str(path): sha256(path) for path in artifacts}, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    write_json(final_audit, final)
    print(json.dumps(final, indent=2), flush=True)
    return final
