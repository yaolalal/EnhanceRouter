"""Only post-seal evaluation may parse true query labels; never retunes a model."""
import csv
import json
import os
import statistics
from pathlib import Path
import torch
from eargtc.joint_v6_unified.evaluation import metrics
from eargtc.joint_v6_unified.experiment import require_slurm,read_json,write_json,sha256,verify_files,probability
from .experiment import ROOT,DATASETS,DEFAULT_RUN,INCUMBENT,LABEL_ROOT,verify_frozen


def evaluate(root=DEFAULT_RUN):
    require_slurm();root=Path(root).resolve();config=verify_frozen(root)
    output=root/'evaluation';output.mkdir(exist_ok=False)
    write_json(output/'STARTED.json',dict(frozen_sha256=sha256(root/'FROZEN.json'),slurm_job_id=os.environ['SLURM_JOB_ID']))
    verify_files(config['label_file_hashes'])
    with Path(config['fixed_comparator']).open() as f:
        fixed={r['dataset']:r for r in csv.DictReader(f) if r['split_seed']=='42' and r['population']=='full_query'}
    from eargtc.joint_v6_unified.data import load_data
    rows=[];results={}
    for ds in DATASETS:
        train,val,inf,meta=load_data(ds)
        if meta!=config['metadata'][ds]:raise ValueError('Evaluation inputs changed')
        ids=inf['query_ids'];label_path=LABEL_ROOT/ds/'decisions.jsonl'
        source=[json.loads(l) for l in label_path.read_text().splitlines() if l.strip()]
        byid={int(r['node_id']):r for r in source}
        if len(byid)!=len(source) or set(byid)!=set(ids.tolist()):raise ValueError('Query truth alignment mismatch')
        y=torch.tensor([int(byid[int(i)]['true_label']) for i in ids],dtype=torch.long)
        raw_saved=torch.load(root/ds/'raw.pt',weights_only=True,map_location='cpu',mmap=True)
        raw={b:raw_saved[b][ids] for b in ('g','l')}
        for b,name in (('g','gnn'),('l','llm')):
            if not torch.equal(raw[b].argmax(1),torch.tensor([int(byid[int(i)][name+'_pred']) for i in ids])):raise ValueError('Expert identity changed')
        saved=torch.load(root/ds/'predictions.pt',weights_only=True,map_location='cpu',mmap=True)
        selection=read_json(root/ds/'selection.json');oldsel=read_json(INCUMBENT/ds/'selection.json')
        previous=torch.load(INCUMBENT/ds/(oldsel['name']+'.pt'),weights_only=True,map_location='cpu',mmap=True)['seeds'][:,ids]
        p=saved['seeds'][:,ids];repair=saved['repair_seeds'][:,ids]
        chosen=metrics(p.mean(0),y);old=metrics(previous.mean(0),y);alternative=metrics(repair.mean(0),y)
        singles=[metrics(v,y) for v in p]
        delta=chosen['accuracy']-old['accuracy'];df=chosen['accuracy']-float(fixed[ds]['accuracy'])
        result=dict(selection=selection,selected_metrics=chosen,previous_unified=old,new_repair=alternative,
            fixed_published=fixed[ds],single_seeds=singles,seed_std=statistics.stdev(m['accuracy'] for m in singles),
            raw_gnn=metrics(raw['g'],y),raw_llm=metrics(raw['l'],y),raw_average=metrics(raw_saved['average'][ids],y),
            pseudo_count=len(train['pseudo_ids']),query_count=len(ids))
        results[ds]=result
        row=dict(dataset=ds,winner=selection['winner'],accuracy=chosen['accuracy'],macro_f1=chosen['macro_f1'],nll=chosen['nll'],
            previous_unified_accuracy=old['accuracy'],fixed_accuracy=float(fixed[ds]['accuracy']),
            delta_unified_pp=100*delta,delta_fixed_pp=100*df,seed_std=result['seed_std'],pseudo_count=len(train['pseudo_ids']))
        rows.append(row);print(json.dumps(row),flush=True)
    write_json(output/'results.json',results)
    with (output/'performance.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    means={k:statistics.mean(r[k] for r in rows) for k in ('accuracy','previous_unified_accuracy','fixed_accuracy','delta_unified_pp','delta_fixed_pp')}
    write_json(output/'summary.json',dict(means=means,slurm_job_id=os.environ['SLURM_JOB_ID']))
    text=['# Router-v8(v6) + Joint Repair：共同 seed42/full-query 对照','',
        '全部模型与融合选择在读取本轮 query 真值前封存。每图6个新 gold 初始化成员、36个全伪标签微调成员，共210个双头成员。每配置均使用内部42/43/44三种子。',
        '', '| 数据集 | v6 + Fixed % | 上轮 Unified % | 本轮 % | ΔFixed pp | ΔUnified pp | 选择 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for r in rows:text.append(f"| {r['dataset']} | {100*r['fixed_accuracy']:.2f} | {100*r['previous_unified_accuracy']:.2f} | {100*r['accuracy']:.2f} | {r['delta_fixed_pp']:+.2f} | {r['delta_unified_pp']:+.2f} | {r['winner']} |")
    text += ['',f"五图等权均值：本轮{100*means['accuracy']:.2f}%，上轮{100*means['previous_unified_accuracy']:.2f}%，同划分Fixed{100*means['fixed_accuracy']:.2f}%。",'',
        '先用 gold 初始化独立、同宽双模态残差头；微调阶段每个接受伪标签每轮恰好出现一次，所有来源/一致或冲突标签共用CE与可选冻结初始化教师KL。两个头独立选择checkpoint、损失配方和残差幅度；融合仅为两个专家修正分布的凸组合，没有pB。', '',
        '与先前已封存Unified选择一起按验证集准确率、NLL比较，精确平局保留旧模型。只有验证集不退步的保证，测试集不具备单调提升保证。测试表中的repair消融不能用于改变本轮已选结果。', '',
        '限制：本轮仍为历史seed42、full-query传导式协议；接受query伪标签可以训练，其真值只用于封存后评分。gold初始化阶段不使用伪标签，微调阶段完整覆盖。已观察过历史测试分数，此次属于适应性benchmark开发，不能声称新测试集或全局最优。没有训练额外外层划分，不能直接与五外层seed均值比较。架构、监督、优化和搜索共同变化，不作单因素因果归因。', '',
        '代码/输入/旧模型/选择/输出哈希见../CONFIG.json、../PRETEST_AUDIT.json、../FROZEN.json。已有历史源代码与结果均未覆盖。']
    with (output/'REPORT.md').open('x') as f:f.write('\n'.join(text)+'\n')
    paths=['STARTED.json','results.json','performance.csv','summary.json','REPORT.md']
    write_json(output/'COMPLETE.json',dict(all_checks_pass=True,files={str(output/p):sha256(output/p) for p in paths},
        frozen_sha256=sha256(root/'FROZEN.json'),slurm_job_id=os.environ['SLURM_JOB_ID']))
    return rows
