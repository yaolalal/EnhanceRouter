"""Reconstruct all ten validation selections before a single full-test evaluation."""
import csv
import itertools
import math
import statistics
from pathlib import Path
import torch
from eargtc.joint_cross_repair.artifacts import read_json,write_json,sha256
from eargtc.joint_cross_repair.training import scheme_b_probability,blend
from eargtc.joint_five_router.experiment import select_candidates
from eargtc.joint_five_router.evaluation import (
    _assert_close,_positions,_label_rows,_bound_label_root,_verify_historical_expert_identities,
    classification_report,correction_damage,labels_for_ids,verify_selection_reconstruction,_seed_summary)
from eargtc.joint_selector.evaluation import verify_seed_ensemble


def _load_outputs(path):
    return (torch.load(path/'predictions.pt',weights_only=True,map_location='cpu'),
            read_json(path/'selection.json'),torch.load(path/'validation.pt',weights_only=True,map_location='cpu'))


def verify_full_family(value,name,shape):
    if value['ensemble'].shape!=shape or value['seeds'].shape!=(3,*shape):
        raise ValueError(name+' full node coverage differs')
    for probability in [value['ensemble'],value['seeds']]:
        if (not torch.isfinite(probability).all() or (probability<0).any()
            or not torch.allclose(probability.sum(-1),torch.ones_like(probability.sum(-1)),atol=2e-5,rtol=0)):
            raise ValueError(name+' invalid probabilities')
    verify_seed_ensemble(value,name)


def audit_all(run_root):
    from .adapter import load_cell
    from .experiment import preflight,ROUTERS
    root=Path(run_root).resolve();config=preflight(root);torch.set_num_threads(config['train']['threads'])
    expected=list(itertools.product(config['routers'],config['datasets']))
    if len(expected)!=10 or config['routers']!=list(ROUTERS) or config['seeds']!=[42,43,44]:raise ValueError('all10 formal cells required')
    records={}
    for rid,ds in expected:
        f,s,p,h,meta=load_cell(config['export_root'],rid,ds,verify_sources=True)
        pred,selection,validation=_load_outputs(root/rid/ds)
        if not torch.equal(pred['node_ids'],f['node_ids']):raise ValueError('full node identity differs')
        pg,pl=f['logp_g'].exp(),f['logp_l'].exp();pb=scheme_b_probability(pg,pl,f['choose_g'],f['agreement'])
        for name,value in [('gnn',pg),('llm',pl),('router',pb)]:_assert_close(pred['baselines'][name],value,'baseline '+name+' differs')
        _assert_close(validation['router'],pb[s['selection_ids']],'validation Router differs')
        rebuilt=select_candidates(validation['candidates'],validation['router'],s['selection_targets'])
        if rebuilt!=selection:raise ValueError('validation selection does not reproduce')
        verify_selection_reconstruction(pred,selection,validation,s['selection_ids'],s['selection_targets'])
        b=selection['base_fusion']
        _assert_close(pred['baselines']['base_fusion'],blend(pg,pl,pb,b['alpha'],b['beta']),'full base fusion differs')
        if set(pred['families'])!={'gold_only','weighted','scaled','joint','optimized'}:
            raise ValueError('missing candidate family')
        for name,value in pred['families'].items():verify_full_family(value,name,pg.shape)
        accepted=torch.cat([p['teach_g'],p['teach_l'],p['agreement']]).unique()
        if len(p['agreement']):raise ValueError('v8 supplies disagreement pseudo labels only')
        records[f'{rid}/{ds}']={'accepted':len(accepted),'test':len(s['test_ids']),'gold':len(s['gold_ids']),
            'selection':len(s['selection_ids']),'pool_counts':{k:len(v) for k,v in p.items()},'validation_reproduced':True}
        print('Pretest validated',rid,ds,flush=True)
        del f,s,p,h,pred,validation,pg,pl,pb
    audit={'all_checks_pass':True,'cell_count':10,'test_labels_read':False,'frozen_sha256':sha256(root/'FROZEN.json'),'cells':records}
    path=root/'PRETEST_AUDIT.json'
    if path.exists():
        if read_json(path)!=audit:raise ValueError('immutable pretest audit differs')
    else:write_json(path,audit)
    return config,audit


def _csv(path,rows):
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def evaluate_all(run_root):
    root=Path(run_root).resolve();output=root/'evaluation'
    if output.exists():raise FileExistsError('evaluation is immutable')
    stage=root/'evaluation.staging'
    if stage.exists():raise FileExistsError('evaluation staging exists; inspect previous attempt')
    config,audit=audit_all(root)
    stage.mkdir()  # Claim one-shot evaluation before any test labels are opened.
    from .adapter import load_cell
    labelroot=_bound_label_root(config,config['label_root'])
    labels={ds:_label_rows(labelroot/ds/'decisions.jsonl') for ds in config['datasets']}
    rows=[];cells={}
    for rid,ds in itertools.product(config['routers'],config['datasets']):
        f,s,p,h,meta=load_cell(config['export_root'],rid,ds,verify_sources=True)
        pred,selection,validation=_load_outputs(root/rid/ds)
        ids=s['test_ids'];pos=_positions(pred['node_ids'],ids);y=labels_for_ids(labels[ds],ids)
        baseline=pred['baselines'];optimized=pred['families']['optimized'];q=optimized['ensemble'][pos]
        _verify_historical_expert_identities(labels[ds],ids,baseline['gnn'][pos],baseline['llm'][pos])
        metrics={name:classification_report(value[pos],y) for name,value in baseline.items()}
        metrics.update({name:classification_report(value['ensemble'][pos],y) for name,value in pred['families'].items()})
        seed=_seed_summary(optimized['seeds'],pos,y)
        accepted_ids=torch.cat([p['teach_g'],p['teach_l'],p['agreement']]).unique();accepted=torch.isin(ids,accepted_ids)
        groups={name:classification_report(q[mask],y[mask]) for name,mask in [('accepted',accepted),('unaccepted',~accepted)]}
        cd=correction_damage(baseline['router'][pos].argmax(1),q.argmax(1),y)
        delta=metrics['optimized']['accuracy']-metrics['router']['accuracy']
        if not math.isclose(cd['gain'],delta,abs_tol=1e-10):raise ValueError('correction/damage mismatch')
        cells[f'{rid}/{ds}']={'router_id':rid,'dataset':ds,'metrics':metrics,'selection':selection,'seed_metrics':seed,
            'accepted_count':len(accepted_ids),'test_count':len(ids),'groups':groups,'pool_counts':{k:len(v) for k,v in p.items()},
            'optimized_vs_router':cd,'test_label_sha256':sha256(labelroot/ds/'decisions.jsonl')}
        rows.append({'router':rid,'dataset':ds,**{k:metrics['optimized'][k] for k in ['accuracy','macro_f1','nll']},
            'router_accuracy':metrics['router']['accuracy'],'delta_router':delta,
            'gold_only_accuracy':metrics['gold_only']['accuracy'],'base_fusion_accuracy':metrics['base_fusion']['accuracy'],
            'seed_accuracy_mean':seed['accuracy']['mean'],'seed_accuracy_std':seed['accuracy']['std'],
            'accepted_count':len(accepted_ids),'test_count':len(ids)})
        print('Evaluated',rid,ds,flush=True)
        del f,s,p,h,pred,validation,baseline,optimized,q
    payload={'config':{k:config[k] for k in ['routers','datasets','architecture','train','seeds','recipes']},'cells':cells}
    write_json(stage/'results.json',payload);_csv(stage/'performance.csv',rows)
    bykey={(r['router'],r['dataset']):r for r in rows};matrices={}
    for metric in ['accuracy','macro_f1','delta_router','router_accuracy','gold_only_accuracy','base_fusion_accuracy']:
        matrix=[]
        for rid in config['routers']:
            row={'router':rid,**{ds:bykey[rid,ds][metric] for ds in config['datasets']}}
            row['mean']=statistics.fmean(row[ds] for ds in config['datasets']);matrix.append(row)
        _csv(stage/(metric+'_matrix.csv'),matrix);matrices[metric]=matrix
    write_report(stage/'REPORT.md',payload,matrices)
    write_json(stage/'complete.json',{'all_checks_pass':True,'preflight_cells':10,'evaluated_cells':10,
        'frozen_sha256':sha256(root/'FROZEN.json'),'pretest_audit_sha256':sha256(root/'PRETEST_AUDIT.json'),
        'artifacts':{p.name:sha256(p) for p in stage.iterdir() if p.is_file()}})
    stage.rename(output);return payload


def write_report(path,payload,matrices):
    cfg=payload['config'];datasets=cfg['datasets'];names={'v8_v3':'Router-v8 v3','v8_v4':'Router-v8 v4'}
    lines=['# Router-v8 v3/v4 接入固定 V2 Joint：最终结果','',
        '两版各五数据集，共10格。固定原始专家、原始embedding、双独立64维残差修复器、gold/交叉教学/保持损失、训练采样和早停方法。9配方×3种子，相同搜索范围，分别在原model-selection validation half选模；全部10格冻结并复算选模后才读取测试标签。',
        '两版仅提供disagreement伪标签，不额外引入agreement训练节点；golden support始终保留。拒绝前route决定所有验证/test分歧节点的原始B预测，agreement仍按原方法平均原专家概率。伪标签接受比例沿用各版已冻结策略，不重新选择。',
        '准确率为三种子概率平均后argmax，平均列为五数据集等权平均；全部方法在原始完整test集合上评估。', '']
    for metric,title,delta in [('accuracy','Joint准确率（%）',False),('router_accuracy','对应原始Router-B完整test准确率（%）',False),
        ('delta_router','Joint相对自身Router-B变化（pp）',True),('macro_f1','Joint Macro-F1（%）',False),
        ('gold_only_accuracy','Gold-only候选家族最终融合准确率（%）',False),('base_fusion_accuracy','无修复专家融合准确率（%）',False)]:
        lines += ['## '+title,'','| Router | '+' | '.join(datasets)+' | 平均 |','|'+'---|'*(len(datasets)+2)]
        for row in matrices[metric]:
            fmt=[f'{100*row[k]:+.2f}' if delta else f'{100*row[k]:.2f}' for k in [*datasets,'mean']]
            lines.append('| '+names[row['router']]+' | '+' | '.join(fmt)+' |')
        lines.append('')
    lines += ['## 解读边界','',
        'v4上游Router扩大了validation监督，并可能使用无标签query一致性；两版Router已使用目标validation进行校准/选择。本轮保持Joint原验证划分，因此是既有开发协议下的下游比较，不能称为等标签预算的纯架构比较或全流程独立验证。历史test结果也已观察。',
        '当前Joint候选包括原Router与无修复融合，因此optimized结果不保证每格启用两个修复器。接受集合精度不等于完整test分类精度。',
        '本轮数值计算通过Slurm运行；源码、输入、配置、全部checkpoint、预测与审计记录均保留。详细指标、种子波动、伪标签数量和最终配置见performance.csv和results.json。','']
    Path(path).write_text('\n'.join(lines))
