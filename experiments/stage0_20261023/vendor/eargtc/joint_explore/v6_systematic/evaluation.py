"""One post-freeze query evaluation. Never imported by the training controller."""
from __future__ import annotations
import csv
import io
import itertools
import json
import math
import os
from pathlib import Path
import statistics
from .manifest import (OLD_ROOT,sha256,read_json,immutable_json,verify_files,write_bundle,
    verify_bundle,bundle_records,tensor_load,bootstrap,slurm_only)
from .plans import DATASETS,OUTERS,PHASES
from .scoring import score


def classification_report(prob,truth):
    result=score(prob,truth)
    import torch
    pred=prob.argmax(1)
    labels=torch.unique(torch.cat([truth,pred]))
    fs=[]
    for label in labels:
        tp=((pred==label)&(truth==label)).sum().item()
        denom=(pred==label).sum().item()+(truth==label).sum().item()
        fs.append(2*tp/denom if denom else 0.)
    return dict(result,macro_f1=sum(fs)/len(fs))


def csv_text(rows):
    if not rows:return ''
    f=io.StringIO(); fields=list(dict.fromkeys(k for r in rows for k in r))
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for r in rows: w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    return f.getvalue()


def aggregate(rows):
    out=[]
    keys=sorted({(r['dataset'],r['candidate'],r['readout']) for r in rows})
    for ds,candidate,readout in keys:
        group=[r for r in rows if (r['dataset'],r['candidate'],r['readout'])==(ds,candidate,readout)]
        row={'dataset':ds,'candidate':candidate,'readout':readout,'outer_count':len(group),'outers':[r['outer'] for r in group]}
        for metric in ('accuracy','macro_f1','nll'):
            values=[r[metric] for r in group]
            row[metric+'_mean']=statistics.mean(values)
            row[metric+'_std_ddof0']=statistics.pstdev(values)
            row[metric+'_std_ddof1']=statistics.stdev(values) if len(values)>1 else None
        out.append(row)
    return out


def factorial_effects(result):
    output=[]
    for ds,outer in itertools.product(DATASETS,OUTERS):
        rows={r['id']:r for r in result['rows'] if r['dataset']==ds and r['outer']==outer}
        # Each effect is a mean difference (or a mean difference of differences)
        # over the complementary factorial settings; no inference across datasets.
        for factors in (('D',),('A',),('S',),('D','A'),('D','S'),('A','S'),('D','A','S')):
            for metric in ('accuracy','nll'):
                total=0.
                for d,a,s in itertools.product((0,1),repeat=3):
                    switches={'D':d,'A':a,'S':s}
                    sign=math.prod(1 if switches[f] else -1 for f in factors)
                    total+=sign*rows[f'D{d}A{a}S{s}']['select'][metric]
                output.append({'dataset':ds,'outer':outer,'effect':':'.join(factors),'metric':metric,'delta':total/(2**(3-len(factors)))})
        for left,right,label in (('goldfull_pseudo64','legacy64','gold_exposure'),('D0A0S0','goldfull_pseudo64','pseudo_exposure'),('D0A0S1','D0A0S0','coverage_D0A0'),('D1A1S1','gold_only160','total_pseudo_D1A1S1')):
            for metric in ('accuracy','nll'):
                output.append({'dataset':ds,'outer':outer,'effect':label,'metric':metric,'delta':rows[left]['select'][metric]-rows[right]['select'][metric]})
    return output


def _report(rows,summary,effects,choices):
    selected=[r for r in summary if r['candidate']=='selected' and r['readout']=='calibrated_noPB']
    lines=['# Joint-v6 systematic study','',
        '所有最终选择、三成员概率、输入与代码在读取查询真值前已封存。选中模型无 pB；最终校准只用本 outer 的 FIT 标签。',
        '历史查询结果此前已被观察，因此这是自适应基准研究，不是新的盲测。FIT/SELECT 分离只约束 Joint 阶段，不能追溯消除上游 Router 使用验证集的影响。',
        '三个 inner seed 平均的是概率，不算三个独立 outer 重复。Outer 共享同一图且节点重叠；以下方差描述支持划分稳定性。','',
        '| Dataset | Accuracy mean ± std(ddof=0) | Macro-F1 | NLL |','|---|---:|---:|---:|']
    for r in selected:
        lines.append(f"| {r['dataset']} | {100*r['accuracy_mean']:.3f} ± {100*r['accuracy_std_ddof0']:.3f} | {100*r['macro_f1_mean']:.3f} | {r['nll_mean']:.5f} |")
    if selected: lines+=['',f"五数据集 accuracy 宏平均：{100*statistics.mean(r['accuracy_mean'] for r in selected):.3f}%。"]
    lines+=['','主表使用各数据集五个 outer 的完整 query；邻域消融使用全部五个匹配 outer，各 outer 独立调优。query_metrics.csv 保留每个设置、每个 split 的改善或下降，matching_fixed.csv 是同 IDs 的冻结 v6+Fixed 配对差。',
        'A/B/C 与 HPO 的选择依据固定 50:50、scale=1 的本 outer SELECT accuracy/NLL，绝不跨 outer 传递选择。A 的 D/A/S 全因子主效应和交互在 stage_a_effects.csv；每个 dataset/outer 从 A 到 HPO 独立选择框架、参数和 G/L 配对；跨 outer 汇总仅在全部选择锁定后报告。完整阶段表及成员 loss/exposure 历史可从 stage_tables.json 和成员工件追溯。',
        '先锁架构与训练机制，再进行 12 点粗搜索、固定顺序邻坐标细化、120/240/480 预算比较、相同预算 anchor 和 top3 三 seed 的九个 G/L 配对。这个有限搜索只给出注册范围内的最佳验证选择，不保证全局最优。',
        'HPO 校准诊断 residual scales 为 {.25,.5,1}；FINAL FIT 校准为 {0,.25,.5,1}，direct classifier 只用 1。scale=0 表示保留原始专家、关闭修正，不能算训练增益；具体零尺度分支见 calibration.json。校准贡献与固定读出训练贡献分开；pB 只作为同 alpha/scales 下的诊断。',
        'raw_concat_block 是投影/融合整体移除；projected_concat_fusion 对 interaction/sum/gate 是融合控制，对 projected_concat_wide 是容量控制。residual_toggle 的 on/off 由每分支 config 明示。warm teacher 的 warm_off 同时换 original teacher，是耦合移除。',
        '历史 Fixed 已有独立归一化、残差头、独立 checkpoint、分支选择和残差校准；其 500-update/bank 搜索与本轮预算/曝光/验证划分不同，不能把整体差当单因素效应。邻域贡献是本 outer 调优后模型附近的条件效应，负效应照实报告，不按邻域消融结果重调。',
        '封存 archive 含代码、结果及引用 gold checkpoints；上游原始数据、专家和 hidden caches 仍为外部哈希绑定依赖，不是独立数据包。','']
    return '\n'.join(lines)


def evaluate(root):
    slurm_only()
    from .experiment import verify_globalfreeze,_cell_path
    root,cfg,lock=verify_globalfreeze(root)
    if (root/'EVALUATION_STARTED.json').exists() or (root/'evaluation').exists():
        raise FileExistsError('evaluation is immutable; partial evaluation requires explicit inspection')
    immutable_json(root/'EVALUATION_STARTED.json',{'frozen_sha256':sha256(root/'FROZEN.json'),'slurm_job_id':os.environ['SLURM_JOB_ID']})
    # This is deliberately the first query-truth accessor import in this module.
    from eargtc.data import _install_pyg_optional_extension_stubs
    _install_pyg_optional_extension_stubs()
    import torch
    bootstrap(root)
    rows=[]; truth_hashes={}; fixed_comparison=[]
    for ds in DATASETS:
        split0=read_json(root/'inputs'/ds/'42/split.json')
        graph_path=Path(split0['graph_path']); verify_files({str(graph_path):split0['graph_sha256']})
        graph=torch.load(graph_path,weights_only=False,map_location='cpu')
        truth=graph.y.reshape(-1).long(); del graph
        if len(truth)!=split0['num_nodes']: raise ValueError('truth node count differs')
        truth_hashes[str(graph_path)]=split0['graph_sha256']
        for outer in OUTERS:
            split=read_json(root/'inputs'/ds/str(outer)/'split.json')
            if any(split[k]!=split0[k] for k in ('graph_path','graph_sha256','num_nodes')): raise ValueError('outer graph identity differs')
            p=_cell_path(root,'FINAL',ds,outer); pred=tensor_load(p/'predictions.pt')
            ids=pred['query_ids']; expected=torch.tensor(split['evaluation_full_ids'],dtype=torch.long)
            if not torch.equal(ids,expected): raise ValueError('query population differs')
            y=truth[ids]; aliases=read_json(p/'pair_aliases.json')
            for candidate,canonical in aliases.items():
                for readout,prob in pred['candidates'][canonical].items():
                    rows.append({'dataset':ds,'outer':outer,'candidate':candidate,'canonical':canonical,'readout':readout,'population':'full_query','query_count':len(ids),**classification_report(prob,y)})
            historical_path=OLD_ROOT/f'seed{outer}/joint/r8_v6/{ds}/predictions.pt'
            seal=read_json(root/'INPUT_SEAL.json')['files']
            if str(historical_path) not in seal: raise ValueError('matching Fixed probabilities not historically sealed')
            verify_files({str(historical_path):seal[str(historical_path)]})
            old=tensor_load(historical_path)
            if not torch.equal(old['node_ids'][ids],ids): raise ValueError('Fixed query ID alignment differs')
            oldscore=classification_report(old['families']['optimized']['ensemble'][ids],y)
            rows.append({'dataset':ds,'outer':outer,'candidate':'historical_v6_Fixed','canonical':'historical_v6_Fixed','readout':'frozen','population':'full_query','query_count':len(ids),**oldscore})
            selected=next(r for r in reversed(rows) if r['dataset']==ds and r['outer']==outer and r['candidate']=='selected' and r['readout']=='calibrated_noPB')
            fixed_comparison.append({'dataset':ds,'outer':outer,**{m+'_selected_minus_fixed':selected[m]-oldscore[m] for m in ('accuracy','macro_f1','nll')}})
        del truth
    summary=aggregate(rows)
    stages={stage:read_json(root/'stages'/stage/'lock/result.json') for stage in PHASES}
    factorial=factorial_effects(stages['A']); paired=[]
    for ds,outer in itertools.product(DATASETS,OUTERS):
        subset=[r for r in rows if r['dataset']==ds and r['outer']==outer]
        for readout in ('fixed','calibrated_noPB','pB_diagnostic'):
            selected=next(r for r in subset if r['candidate']=='selected' and r['readout']==readout)
            for r in subset:
                if r['readout']==readout:
                    paired.append({'dataset':ds,'outer':outer,'control':r['candidate'],'readout':readout,**{m+'_selected_minus_control':selected[m]-r[m] for m in ('accuracy','macro_f1','nll')}})
    bootstrap(root); verify_files(lock['files']); verify_files(read_json(root/'INPUT_SEAL.json')['files'])
    payloads={'query_metrics.csv':csv_text(rows),'mean_std.csv':csv_text(summary),'matching_fixed.csv':csv_text(fixed_comparison),'paired_query_effects.csv':csv_text(paired),'stage_a_effects.csv':csv_text(factorial),'stage_tables.json':stages,'results.json':{'rows':rows,'summary':summary,'truth_hashes':truth_hashes,'fixed_comparison':fixed_comparison,'paired_outer_effects':paired},'report.md':_report(rows,summary,paired,stages['H11_replicate']['tuned'])}
    for stage,result in stages.items():
        payloads[stage+'_validation.csv']=csv_text(result['rows'])
    write_bundle(root/'evaluation',payloads,{'frozen_sha256':sha256(root/'FROZEN.json'),'test_labels_read':True}, {str(root/'FROZEN.json'):sha256(root/'FROZEN.json'),str(root/'EVALUATION_STARTED.json'):sha256(root/'EVALUATION_STARTED.json')})
    return receipt(root)


def receipt(root):
    slurm_only()
    from .experiment import verify_globalfreeze
    root,cfg,lock=verify_globalfreeze(root)
    started=read_json(root/'EVALUATION_STARTED.json')
    if started.get('frozen_sha256')!=sha256(root/'FROZEN.json'): raise ValueError('evaluation STARTED identity differs')
    marker=verify_bundle(root/'evaluation',{'frozen_sha256':sha256(root/'FROZEN.json'),'test_labels_read':True})
    result=read_json(root/'evaluation/results.json')
    selected=[r for r in result['rows'] if r['candidate']=='selected' and r['readout']=='calibrated_noPB']
    if {(r['dataset'],r['outer']) for r in selected}!=set(itertools.product(DATASETS,OUTERS)) or len(selected)!=len(DATASETS)*len(OUTERS): raise ValueError('evaluation selected coverage differs')
    if aggregate(result['rows'])!=result['summary']: raise ValueError('saved outer summaries do not reproduce')
    # Query scores are reproduced from the frozen saved probabilities, with no
    # fit/selection operation and no changes to choices or training artifacts.
    from eargtc.data import _install_pyg_optional_extension_stubs
    _install_pyg_optional_extension_stubs()
    import torch
    from .experiment import _cell_path
    for ds in DATASETS:
        split=read_json(root/'inputs'/ds/'42/split.json'); p=Path(split['graph_path'])
        verify_files({str(p):split['graph_sha256']})
        graph=torch.load(p,weights_only=False,map_location='cpu'); y=graph.y.reshape(-1).long(); del graph
        for outer in OUTERS:
            path=_cell_path(root,'FINAL',ds,outer); saved=tensor_load(path/'predictions.pt'); aliases=read_json(path/'pair_aliases.json')
            for row in [r for r in result['rows'] if r['dataset']==ds and r['outer']==outer and r['candidate']!='historical_v6_Fixed']:
                metrics=classification_report(saved['candidates'][aliases[row['candidate']]][row['readout']],y[saved['query_ids']])
                if any(metrics[k]!=row[k] for k in metrics): raise ValueError('saved prediction metrics fail reproduction')
        del y
    bootstrap(root)
    receipt={'status':'COMPLETE','cell_count':len(DATASETS)*len(OUTERS),'all_saved_prediction_metrics_reproduced':True,'all_input_and_preserved_hashes_verified':True,'all_phase_grids_verified':True,'all_query_ids_verified':True,'frozen_sha256':sha256(root/'FROZEN.json'),'evaluation_sha256':sha256(root/'evaluation/complete.json'),'files':bundle_records(root/'evaluation')}
    dest=root/'FINAL_RECEIPT.json'
    if dest.exists():
        if read_json(dest)!=receipt: raise ValueError('immutable final receipt differs')
    else: immutable_json(dest,receipt)
    return receipt
