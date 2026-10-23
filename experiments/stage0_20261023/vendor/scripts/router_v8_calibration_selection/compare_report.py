#!/usr/bin/env python3
"""Post-evaluation comparison against already frozen historic results, never selection."""
import argparse,hashlib,json,os,sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from scripts.router_v8_calibration_selection.locked_protocol import verify_report_ready
DATASETS=('cora','citeseer','pubmed','arxiv','ogbn-products')

def fmt(x,pct=False): return '—' if x is None else (f'{100*x:.2f}%' if pct else f'{x:.4f}')


def run(root):
    verify_report_ready(root)
    current=json.loads((root/'test/metrics.json').read_text())
    previous=json.loads((REPO/'runs/router_v8_trust_preference_20260906_v1/test/metrics.json').read_text())
    reference=REPO/'runs/router_v8_posthoc_review_20260907/comparison.json'
    v6=json.loads(reference.read_text())['datasets']
    lines=['## 与 Structural-OOF Router-v6 和 V8 首版对比','',
        '下表均为同一冻结专家与 disagreement 人口。v6 原协议允许目标 validation 选择端点/路由；本轮 V8 主协议从 source 派生 shortlist，通过目标 calibration 交叉验证选择端点，source-only 第二轮仅为预先声明的参考。新旧 label budget 与协议差异须同时考虑。','',
        '| 数据集 | v6 Preference AUC | 首版 V8 Preference AUC | 新 V8 Preference AUC | v6 全 disagreement accuracy | 首版 V8 accuracy | 新 V8 accuracy |',
        '|---|---:|---:|---:|---:|---:|---:|']
    rows={}
    for d in DATASETS:
        n=current[d]['metrics']; o=previous[d]['metrics']; b=v6[d]['v6']
        if n['count']!=v6[d]['count'] or n['count']!=o['count']: raise ValueError('population count differs')
        lines.append(f"| {d} | {fmt(b['preference_auc'])} | {fmt(o['preference_auroc'])} | {fmt(n['preference_auroc'])} | {fmt(b['conflict_accuracy'],True)} | {fmt(o['full_conflict_pseudo_accuracy'],True)} | {fmt(n['full_conflict_pseudo_accuracy'],True)} |")
        rows[d]=dict(trust_auc_delta_vs_v8=n['trust_auroc']-o['trust_auroc'],pref_auc_delta_vs_v8=n['preference_auroc']-o['preference_auroc'],
            pref_auc_delta_vs_v6=n['preference_auroc']-b['preference_auc'],
            conflict_accuracy_delta_vs_v8=n['full_conflict_pseudo_accuracy']-o['full_conflict_pseudo_accuracy'],
            conflict_accuracy_delta_vs_v6=n['full_conflict_pseudo_accuracy']-b['conflict_accuracy'])
    macro={k:sum(r[k] for r in rows.values())/len(rows) for k in next(iter(rows.values()))}
    lines+=['',f"等权五图平均变化：相对首版 V8，Trust AUROC {macro['trust_auc_delta_vs_v8']:+.4f}、Preference AUROC {macro['pref_auc_delta_vs_v8']:+.4f}、全 disagreement accuracy {100*macro['conflict_accuracy_delta_vs_v8']:+.2f} 个百分点；相对 v6，Preference AUROC {macro['pref_auc_delta_vs_v6']:+.4f}、全 disagreement accuracy {100*macro['conflict_accuracy_delta_vs_v6']:+.2f} 个百分点。",'',
        '变化只描述已冻结系统；不以正负变化决定重新搜索或选择不同 test 端点。','']
    payload=dict(reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),datasets=rows,macro=macro)
    (root/'test/historical_comparison.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    comparison='\n'.join(lines); (root/'COMPARISON.md').write_text(comparison)
    report=(root/'REPORT.md').read_text()+'\n'+comparison+'\n'+(root/'DEVELOPMENT_REPORT.md').read_text().replace('# Router-v8：冻结前开发结果','## 冻结前开发数值',1)
    (REPO/'docs/router_v8_calibration_selection_results_2026-09-07.md').write_text(report)
    print(json.dumps(payload,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Run through Slurm')
    run(a.root)
