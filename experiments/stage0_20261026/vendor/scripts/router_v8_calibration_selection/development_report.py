#!/usr/bin/env python3
"""Report target-calibration selection and disjoint policy evaluation; no test labels."""
import argparse,json,os
from pathlib import Path
DATASETS=('cora','citeseer','pubmed','arxiv','ogbn-products')
def show(x,pct=False):return '—' if x is None else (f'{100*x:.2f}%' if pct else f'{x:.4f}')
def run(root):
    lines=['# Router-v8：冻结前开发结果','',
        '模型候选由其他四图source结果产生；目标calibration half通过LOO/五折概率评价与固定logit AUROC选头，policy half仅供评价和接受策略。开发集在连续探索中已观察，不能称作全新未见holdout。以下没有test标签。','',
        '| 目标 | Head | 选定候选 source AUC | Calibration AUC | Crossfit BCE | 候选数 | Target policy AUC | Target policy BCE |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    summary={}
    for d in DATASETS:
        src=json.loads((root/'targets'/d/'source_selection.json').read_text());sel=json.loads((root/'targets'/d/'selection.json').read_text());summary[d]={}
        for h in ('trust','preference'):
            cfg=sel['configs'][h];source=next(t for t in src[h]['trials'] if t['config']['id']==cfg['id'])
            cal=sel['calibration_selection_summary'][h];pol=sel['validation_metrics'][h]['after_calibration']
            lines.append(f"| {d} | {h} | {show(source['macro_auc'])} | {show(cal['calibration_auc'])} | {show(cal['crossfit_bce'])} | {cal['shortlist_count']} | {show(pol['auc'])} | {show(pol['bce'])} |")
            summary[d][h]=dict(config=cfg,source_macro_auc=source['macro_auc'],source_macro_bce=source['macro_bce'],calibration_selection=cal,target_policy=pol)
    lines+=['','| 目标 | 接受比例 | policy接受数 | policy pseudo accuracy | Wilson10%接受数 |','|---|---:|---:|---:|---:|']
    for d in DATASETS:
        sel=json.loads((root/'targets'/d/'selection.json').read_text());p=sel['top_fraction_policy'];w=sel['policies']['0.1']
        lines.append(f"| {d} | {show(p['fraction'],True)} | {p['validation_accepted']} | {show(p['validation_accuracy'],True)} | {w['validation_accepted']} |")
        summary[d]['top_fraction_policy']=p
    lines+=['','policy precision本身在该half上选择，不是无偏precision估计；最终性能在五图一起冻结后评价。','']
    (root/'DEVELOPMENT_REPORT.md').write_text('\n'.join(lines));(root/'DEVELOPMENT_SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run through Slurm')
    run(a.root)
