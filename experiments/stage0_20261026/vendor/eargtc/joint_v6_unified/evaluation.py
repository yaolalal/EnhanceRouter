"""The only module allowed to parse query truth, after the all-five barrier."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import statistics
import os
import torch
from .experiment import (DATASETS,FAMILIES,DEFAULT_RUN,LABEL_ROOT,require_slurm,read_json,
                         write_json,sha256,verify_files,verify_frozen,probability)


def metrics(p,y):
    require_slurm()
    probability(p)
    if y.dtype!=torch.long or y.shape!=(len(p),) or bool(((y<0)|(y>=p.shape[1])).any()):
        raise ValueError('Invalid evaluation target')
    p=p.double();c=p.shape[1];pred=p.argmax(1)
    matrix=torch.bincount(y*c+pred,minlength=c*c).reshape(c,c).double()
    denom=matrix.sum(0)+matrix.sum(1)
    f1=torch.where(denom>0,2*matrix.diag()/denom,torch.zeros_like(denom))
    return dict(count=len(y),accuracy=float((pred==y).double().mean()),macro_f1=float(f1.mean()),
                nll=float(-p[torch.arange(len(y)),y].clamp_min(1e-12).log().mean()))


def evaluate(run_root=DEFAULT_RUN):
    require_slurm();root=Path(run_root).resolve()
    # Nothing below this line executes unless every cell has been audited and sealed.
    config=verify_frozen(root)
    output=root/'evaluation';output.mkdir(exist_ok=False)
    write_json(output/'STARTED.json',dict(frozen_sha256=sha256(root/'FROZEN.json'),
                                        slurm_job_id=os.environ['SLURM_JOB_ID']))
    verify_files(config['label_file_hashes'])
    from .data import load_data
    results={};table=[];family_table=[]
    for ds in DATASETS:
        train,val,inf,meta=load_data(ds)
        ids=inf['query_ids'];raw=torch.load(root/ds/'raw.pt',weights_only=True,map_location='cpu',mmap=True)
        label_path=LABEL_ROOT/ds/'decisions.jsonl'
        rows=[json.loads(line) for line in label_path.read_text().splitlines() if line.strip()]
        by_id={int(r['node_id']):r for r in rows}
        if len(by_id)!=len(rows) or set(by_id)!=set(ids.tolist()):raise ValueError('Truth/node alignment mismatch')
        y=torch.tensor([int(by_id[int(i)]['true_label']) for i in ids],dtype=torch.long)
        for branch in ('gnn','llm'):
            expected=torch.tensor([int(by_id[int(i)][branch+'_pred']) for i in ids])
            if not torch.equal(raw[branch][ids].argmax(1),expected):raise ValueError('Historical expert identity mismatch')
        reports={k:metrics(raw[k][ids],y) for k in ('gnn','llm','average')}
        seeds={};all_probs={k:raw[k][ids] for k in reports}
        for family in FAMILIES:
            obj=torch.load(root/ds/(family+'.pt'),weights_only=True,map_location='cpu',mmap=True)
            p=obj['seeds'][:,ids];avg=p.mean(0);all_probs[family]=avg
            reports[family]=metrics(avg,y)
            single=[metrics(v,y) for v in p]
            seeds[family]={'individual':single,'accuracy_mean':statistics.mean(r['accuracy'] for r in single),
                           'accuracy_std':statistics.stdev(r['accuracy'] for r in single)}
            family_table.append(dict(dataset=ds,family=family,**reports[family]))
        selection=read_json(root/ds/'selection.json');winner=selection['name'];chosen=reports[winner]
        if winner not in seeds:
            seeds[winner]=dict(individual=[chosen]*3,accuracy_mean=chosen['accuracy'],accuracy_std=0.)
        base=all_probs['average'].argmax(1)==y;new=all_probs[winner].argmax(1)==y
        transitions=dict(corrections=int((~base&new).sum()),damages=int((base&~new).sum()))
        results[ds]=dict(selected=winner,selection=selection,metrics=reports,seed_metrics=seeds,
                         selected_metrics=chosen,changes_vs_average=transitions,
                         gold_count=len(train['gold_ids']),pseudo_count=len(train['pseudo_ids']),
                         query_count=len(ids),test_truth_sha256=sha256(label_path))
        table.append(dict(dataset=ds,selected=winner,accuracy=chosen['accuracy'],macro_f1=chosen['macro_f1'],
                          nll=chosen['nll'],gnn_accuracy=reports['gnn']['accuracy'],llm_accuracy=reports['llm']['accuracy'],
                          average_accuracy=reports['average']['accuracy'],delta_average=chosen['accuracy']-reports['average']['accuracy'],
                          seed_std=seeds[winner]['accuracy_std'],pseudo_count=len(train['pseudo_ids'])))
        print(json.dumps({'dataset':ds,'selected':winner,'test_accuracy':chosen['accuracy']}),flush=True)
    write_json(output/'results.json',results)
    for name,data in [('performance.csv',table),('families.csv',family_table)]:
        with (output/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    mean=statistics.mean(r['accuracy'] for r in table)
    lines=['# Router-v8(v6) + Joint Unified 实验报告','',
           '三种结构、每结构四个固定配置；每图 12 个 seed42 初筛成员加 6 个确认成员，五图共 90 个。最终方案完全按验证集选择，以下测试成绩未用于追加搜索。',
           '', '| 数据集 | GNN % | LLM % | 原专家均值 % | Joint % | Δ均值 pp | 选中结构 |',
           '|---|---:|---:|---:|---:|---:|---|']
    for r in table:
        lines.append(f"| {r['dataset']} | {100*r['gnn_accuracy']:.2f} | {100*r['llm_accuracy']:.2f} | {100*r['average_accuracy']:.2f} | {100*r['accuracy']:.2f} | {100*r['delta_average']:+.2f} | {r['selected']} |")
    lines += ['',f'五图等权平均准确率：{100*mean:.2f}%。', '',
              '每个完整 epoch 无放回覆盖全部 gold 和全部已接受伪标签；所有伪标签系数相同。GNN/LLM 独立归一化并投影到相同宽度，未使用 Router-B 融合、来源权重、一致/冲突专用损失或历史 Joint checkpoint。', '',
              '训练真值仅来自历史 gold support；validation 仅用于 checkpoint 和有限模型选择；query 真值仅在五图封存与预评估审计后读取。当前是 full-query 传导式协议，接受的 query 伪标签可以训练，真实 query 标签不可以。上游冻结特征包含传导式 SSL；历史 query 成绩已被观察，不能宣称全新未触碰测试集或全流程归纳式实验。', '',
              '旧 Joint-Fixed 使用不同上游 Router，因此不能将旧表与本表的差值全部归因于 Joint 架构。本次直接对照为同一冻结专家的 GNN、LLM 及其 50/50 概率均值。测试家族表仅用于冻结后消融报告；不会据此改换已选模型。', '',
              '单种子标准差仅描述同一历史划分上的初始化差异，不是跨划分置信区间。搜索是预先固定的有限预算，结果不代表全局最优。', '',
              '来源与重放证据：../CONFIG.json、../PRETEST_AUDIT.json、../FROZEN.json；逐图选择及 checkpoint 在各数据集目录。']
    with (output/'REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    files={str(output/name):sha256(output/name) for name in ('STARTED.json','results.json','performance.csv','families.csv','REPORT.md')}
    write_json(output/'COMPLETE.json',dict(all_checks_pass=True,files=files,slurm_job_id=os.environ['SLURM_JOB_ID'],
                                         frozen_sha256=sha256(root/'FROZEN.json')))
    return table
