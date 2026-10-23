"""All-100 validation barrier, followed by a single descriptive test report."""
from __future__ import annotations
import csv
import itertools
import math
import statistics
from pathlib import Path
import torch
from eargtc.joint_cross_repair.artifacts import read_json,write_json,sha256,verify_files
from eargtc.joint_cross_repair.training import blend,scheme_b_probability
from eargtc.joint_five_router.evaluation import (
    _assert_close,_positions,_label_rows,_bound_label_root,_verify_historical_expert_identities,
    classification_report,correction_damage,labels_for_ids,verify_selection_reconstruction,_seed_summary)
from eargtc.joint_five_router.experiment import select_candidates


def verify_fixed_control(control,validation,source_selection,node_ids,selection_ids,*,full_baselines=None):
    if not torch.equal(control['node_ids'],node_ids) or not torch.equal(validation['node_ids'],selection_ids):
        raise ValueError('fixed control node IDs differ')
    if control['source_selection']!=source_selection:
        raise ValueError('fixed control configuration differs from frozen full selection')
    pb=validation['router']; pick=source_selection
    if pick['family']=='router':expected=pb
    else:
        by_id={c['id']:c for branch in validation['candidates'].values() for c in branch}
        expected=blend(by_id[pick['g_id']]['probability'],by_id[pick['l_id']]['probability'],pb,pick['alpha'],pick['beta'])
    pos=_positions(node_ids,selection_ids)
    _assert_close(control['ensemble'][pos],expected,'fixed control validation reconstruction differs')
    seeds=control['seeds']
    if seeds.ndim!=3 or seeds.shape[0]!=3:
        raise ValueError('fixed control requires three seeds')
    _assert_close(seeds.mean(0),control['ensemble'],'fixed control seed average differs')
    if pick['family'] in ('router','base_fusion'):
        if full_baselines is None:
            raise ValueError('deterministic fixed control requires full baselines')
        expected_full=full_baselines['router'] if pick['family']=='router' else blend(
            full_baselines['gnn'],full_baselines['llm'],full_baselines['router'],pick['alpha'],pick['beta'])
        _assert_close(control['ensemble'],expected_full,'deterministic fixed control full prediction differs')
        for seed in seeds:
            _assert_close(seed,expected_full,'deterministic fixed control seed differs')


def verify_seed_ensemble(value,name):
    seeds=value['seeds']
    if not isinstance(seeds,torch.Tensor) or seeds.ndim!=3 or seeds.shape[0]!=3:
        raise ValueError(name+' requires exactly three seeds')
    _assert_close(seeds.mean(0),value['ensemble'],name+' full seed average differs')


def verify_full_identity(predictions,control,historical):
    if not torch.equal(predictions['node_ids'],historical['node_ids']) or not torch.equal(control['node_ids'],historical['node_ids']):
        raise ValueError('full historical node identity differs')
    reference=historical['families']['optimized']
    for name,value in [('optimized',predictions['families']['optimized']),('fixed control',control)]:
        for field in ('ensemble','seeds'):
            if not torch.equal(value[field],reference[field]):
                raise ValueError('full historical '+name+' '+field+' differs')


def _load_outputs(path):
    return (torch.load(path/'predictions.pt',weights_only=True,map_location='cpu'),
            read_json(path/'selection.json'),
            torch.load(path/'validation.pt',weights_only=True,map_location='cpu'),
            torch.load(path/'fixed_control.pt',weights_only=True,map_location='cpu'))


def audit_all(run_root):
    """Stream cells; do not keep 100 full-graph prediction tensors in memory."""
    from .adapter import load_cell
    from .experiment import preflight
    root=Path(run_root).resolve();config=preflight(root)
    torch.set_num_threads(config['train']['threads'])
    if config.get('smoke'):raise ValueError('smoke cannot read final test truth')
    expected=list(itertools.product(config['strategies'],config['routers'],config['datasets']))
    if len(expected)!=100:raise ValueError('all100 cells required')
    oldroot=Path(config['old_five_root']);records={}
    for strategy,rid,ds in expected:
        f,s,p,h,meta=load_cell(config['export_root'],strategy,rid,ds,verify_sources=True)
        path=root/strategy/rid/ds
        predictions,selection,validation,control=_load_outputs(path)
        if not torch.equal(predictions['node_ids'],f['node_ids']):raise ValueError('full node identity differs')
        pg,pl=f['logp_g'].exp(),f['logp_l'].exp()
        pb=scheme_b_probability(pg,pl,f['choose_g'],f['agreement'])
        for name,value in [('gnn',pg),('llm',pl),('router',pb)]:
            _assert_close(predictions['baselines'][name],value,'unchanged '+name+' differs')
        _assert_close(validation['router'],pb[s['selection_ids']],'validation Router differs')
        rebuilt=select_candidates(validation['candidates'],validation['router'],s['selection_targets'])
        if rebuilt!=selection:raise ValueError('validation selection does not reproduce')
        verify_selection_reconstruction(predictions,selection,validation,s['selection_ids'],s['selection_targets'])
        for name,value in predictions['families'].items():
            verify_seed_ensemble(value,name)
        full_pick=read_json(oldroot/rid/ds/'selection.json')['optimized']
        verify_fixed_control(control,validation,full_pick,f['node_ids'],s['selection_ids'],
            full_baselines={'gnn':pg,'llm':pl,'router':pb})
        if strategy=='full':
            old=torch.load(oldroot/rid/ds/'predictions.pt',weights_only=True,map_location='cpu')
            if selection!=read_json(oldroot/rid/ds/'selection.json'):raise ValueError('full selection differs from historical control')
            verify_full_identity(predictions,control,old)
            del old
        accepted=torch.cat([p['teach_g'],p['teach_l'],p['agreement']]).unique()
        records[f'{strategy}/{rid}/{ds}']={'accepted':int(accepted.numel()),'test':int(s['test_ids'].numel()),
            'gold':int(s['gold_ids'].numel()),'selection':int(s['selection_ids'].numel()),
            'pool_counts':{k:int(v.numel()) for k,v in p.items()},'validation_reproduced':True,'fixed_control_reproduced':True}
        print('Pretest validated',strategy,rid,ds,flush=True)
        del f,s,p,h,predictions,validation,control,pg,pl,pb
    audit={'all_checks_pass':True,'cell_count':len(records),'test_labels_read':False,
        'frozen_sha256':sha256(root/'FROZEN.json'),'cells':records}
    path=root/'PRETEST_AUDIT.json'
    if path.exists() and read_json(path)!=audit:raise ValueError('immutable pretest audit differs')
    if not path.exists():write_json(path,audit)
    return config,audit


def _write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def evaluate_all(run_root):
    root=Path(run_root).resolve();output=root/'evaluation'
    if output.exists():raise FileExistsError('evaluation is immutable')
    config,audit=audit_all(root)  # must complete all100 before _label_rows
    from .adapter import load_cell
    label_root=_bound_label_root(config,config['label_root'])
    label_data={ds:_label_rows(label_root/ds/'decisions.jsonl') for ds in config['datasets']}
    cells={};rows=[]
    for strategy,rid,ds in itertools.product(config['strategies'],config['routers'],config['datasets']):
        f,s,p,h,meta=load_cell(config['export_root'],strategy,rid,ds,verify_sources=True)
        path=root/strategy/rid/ds;pred,selection,validation,control=_load_outputs(path)
        ids=s['test_ids'];pos=_positions(pred['node_ids'],ids);y=labels_for_ids(label_data[ds],ids)
        baseline=pred['baselines'];optimized=pred['families']['optimized'];q=optimized['ensemble'][pos]
        _verify_historical_expert_identities(label_data[ds],ids,baseline['gnn'][pos],baseline['llm'][pos])
        metrics={name:classification_report(v[pos],y) for name,v in baseline.items()}
        metrics.update({name:classification_report(v['ensemble'][pos],y) for name,v in pred['families'].items()})
        metrics['fixed_full_config']=classification_report(control['ensemble'][pos],y)
        seed=_seed_summary(optimized['seeds'],pos,y)
        accepted_ids=torch.cat([p['teach_g'],p['teach_l'],p['agreement']]).unique();accepted=torch.isin(ids,accepted_ids)
        groups={name:classification_report(q[mask],y[mask]) for name,mask in [('accepted',accepted),('unaccepted',~accepted)]}
        cd=correction_damage(baseline['router'][pos].argmax(1),q.argmax(1),y)
        if not math.isclose(cd['gain'],metrics['optimized']['accuracy']-metrics['router']['accuracy'],abs_tol=1e-10):raise ValueError('correction-damage inconsistency')
        key=f'{strategy}/{rid}/{ds}'
        cells[key]={'strategy':strategy,'router_id':rid,'dataset':ds,'metrics':metrics,'selection':selection,
            'fixed_full_selection':control['source_selection'],'seed_metrics':seed,'accepted_count':int(accepted.sum()),
            'test_count':int(ids.numel()),'groups':groups,'pool_counts':{k:int(v.numel()) for k,v in p.items()},
            'optimized_vs_router':cd,'test_label_sha256':sha256(label_root/ds/'decisions.jsonl')}
        rows.append({'strategy':strategy,'router':rid,'dataset':ds,**{k:metrics['optimized'][k] for k in ['accuracy','macro_f1','nll']},
            'fixed_accuracy':metrics['fixed_full_config']['accuracy'],'fixed_macro_f1':metrics['fixed_full_config']['macro_f1'],
            'fixed_nll':metrics['fixed_full_config']['nll'],'router_accuracy':metrics['router']['accuracy'],
            'seed_accuracy_mean':seed['accuracy']['mean'],'seed_accuracy_std':seed['accuracy']['std'],
            'accepted_count':int(accepted.sum()),'test_count':int(ids.numel())})
        print('Evaluated',key,flush=True)
        del f,s,p,h,pred,validation,control,baseline,optimized,q
    by_key={(r['strategy'],r['router'],r['dataset']):r for r in rows}
    for row in rows:
        full=by_key['full',row['router'],row['dataset']]
        row['delta_full']=row['accuracy']-full['accuracy'];row['fixed_delta_full']=row['fixed_accuracy']-full['accuracy']
        row['delta_router']=row['accuracy']-row['router_accuracy']
        cells[f"{row['strategy']}/{row['router']}/{row['dataset']}"]['delta_full']=row['delta_full']
    payload={'config':{k:config[k] for k in ['strategies','routers','datasets','architecture','train','seeds','recipes']},'cells':cells}
    stage=root/'evaluation.staging'
    if stage.exists():raise FileExistsError('evaluation staging exists; inspect failed prior run')
    stage.mkdir();write_json(stage/'results.json',payload);_write_csv(stage/'performance.csv',rows)
    matrices={}
    for metric in ['accuracy','macro_f1','fixed_accuracy','delta_full','fixed_delta_full']:
        matrix=[]
        for strategy,rid in itertools.product(config['strategies'],config['routers']):
            row={'strategy':strategy,'router':rid,**{ds:by_key[strategy,rid,ds][metric] for ds in config['datasets']}}
            row['mean']=statistics.fmean(row[ds] for ds in config['datasets']);matrix.append(row)
        _write_csv(stage/(metric+'_matrix.csv'),matrix);matrices[metric]=matrix
    write_report(stage/'REPORT.md',payload,matrices)
    write_json(stage/'complete.json',{'all_checks_pass':True,'preflight_cells':100,'evaluated_cells':len(cells),
        'frozen_sha256':sha256(root/'FROZEN.json'),'pretest_audit_sha256':sha256(root/'PRETEST_AUDIT.json'),
        'artifacts':{p.name:sha256(p) for p in stage.iterdir() if p.is_file()}})
    stage.rename(output)
    return payload


def write_report(path,payload,matrices):
    cfg=payload['config'];ds=cfg['datasets']
    names={'r6_legacy_c':'R6-LegacyC','r6_split_c':'R6-SplitC','r6_expert_conf_c':'R6-EC','a20ms_split_c':'A20-SplitC','a20emb_v2_split_c':'A20Emb-SplitC'}
    strategy_names={'full':'全量','random':'随机20%','reliability':'可靠性20%','reliability_cluster':'可靠性+聚类20%'}
    lines=['# 四筛选方案 × 五 Router：固定 V2 Joint 最终报告','',
        '全部100格在原有model-selection validation half选模并统一冻结，复算全部选模与固定配置对照后才进行本轮test评估。全部方法在相同完整test节点上报告。',
        '', '## 控制变量','',
        '固定原始专家、128维GNN/4096维原始LLM表示、双独立64维残差MLP、gold CE权重1、交叉教学与保持损失、采样/优化器/早停方法及数据划分。9种配方×3种子，所有方案使用相同搜索范围。B路由、C分数及既定k冻结，仅接入的伪标签节点池变化；未重新推理或降维。',
        '全量训练成员经输入、训练池、配置和validation输出核验后复用，gold-only对照也复用；其余重新训练。每种方案分别调优，因此主表包含筛选对调优选择的影响。固定配置对照沿用同Router/数据集全量方案的配方、checkpoint选择准则、残差强度和融合权重；早停步骤仍按同一训练方法产生，并非强制固定历史epoch。',
        '三种子概率平均后argmax。原始gold/selection/threshold/test划分不变；筛选的20%是Router候选比例，最终伪标签还经过原k筛选。','']
    for metric,title,is_delta in [('accuracy','分别调优的Joint准确率（%）',False),('delta_full','相对同Router全量Joint变化（pp）',True),('fixed_accuracy','沿用全量超参数的Joint准确率（%）',False),('fixed_delta_full','固定超参数对照相对全量变化（pp）',True),('macro_f1','分别调优的Macro-F1（%）',False)]:
        lines+=['## '+title,'','| 筛选方案 | Router | '+' | '.join(ds)+' | 平均 |','|'+'---|'*(len(ds)+3)]
        for row in matrices[metric]:
            values=[f"{100*row[d]:+.2f}" if is_delta else f"{100*row[d]:.2f}" for d in [*ds,'mean']]
            lines.append('| '+strategy_names[row['strategy']]+' | '+names[row['router']]+' | '+' | '.join(values)+' |')
        lines+=['']
    lines+=['## 解释边界','',
        '主表与固定配置表均在test评估前冻结，没有按test选择筛选方案或拼接逐数据集赢家。该结果属于既有项目的后续探索：历史上游使用过完整validation，历史test结果也已观察；不声称全链条独立benchmark、理论最优或必然提升。',
        '固定配置对照保持的是超参数和checkpoint选择方法，训练数据变化仍可能改变早停步骤。独立调优结果衡量同等搜索规则下的完整系统表现。',
        '', '100格详细准确率、Macro-F1、NLL、种子波动、伪标签数量及差值见performance.csv；全部分支选模细节见results.json。']
    Path(path).write_text('\n'.join(lines)+'\n',encoding='utf-8')
