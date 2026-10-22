#!/usr/bin/env python3
"""Locked preparation, registration, evaluation and reporting for learned V8.

Only evaluate_locked may deserialize graph labels. Other stages consume safe
validation labels or label-free features. New models never read old test state.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from eargtc.router_v8.artifacts import sha256,freeze_paths,verify_frozen,begin_once
from scripts.router_v8_calibration_selection.locked_protocol import (
    dataset_result,validate_scores,arr,route_metrics,
)
DATASETS=('cora','citeseer','pubmed','arxiv','ogbn-products')
OLD=REPO/'runs/router_v8_trust_preference_20260906_v1'
REFERENCE=REPO/'runs/router_v8_calibration_selection_20260907_v3'
WORK=REPO/'docs/router_v8_learned_work'
TARGET_FILES=('model.pt','scores.pt','validation_scores.pt','source_selection.json','target_selection.json','selection.json')
COVERAGE=[.01,.025,.05,.1,.15,.2,.25,.35,.5,.75,1.]


def read(path):return json.loads(Path(path).read_text())

def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    os.link(temp,path);temp.unlink()


def check_hashes(values,kind='input'):
    for name,digest in values.items():
        if not Path(name).is_file() or sha256(name)!=digest:raise ValueError(f'{kind} hash mismatch: {name}')


def code_paths():
    # Includes immutable imported legacy metric/policy/graph compatibility closure.
    paths={Path(p) for p in read(REFERENCE/'PREREGISTERED.json')['code_hashes']}
    for directory in ('eargtc/router_v8_learned','scripts/router_v8_learned'):
        paths.update((REPO/directory).rglob('*.py'))
    return sorted(paths)


def verify_preregistration(root,stage='full'):
    root=Path(root);pre=read(root/'PREREGISTERED.json')
    current={str(p.resolve()) for p in code_paths()}
    if current!=set(pre['code_hashes']):raise ValueError('registered code inventory changed')
    check_hashes(pre['code_hashes'],'code');check_hashes(pre.get('protocol_hashes',{}),'protocol')
    if stage in ('source','target'):
        # Bank loader checks bytes on actual reads; gate binds their inventories.
        check_hashes({p:h for p,h in pre['input_hashes'].items() if Path(p).name=='manifest.json'},'bank manifest')
    else:check_hashes(pre['input_hashes'])
    return pre


def query_for_metrics(bank):
    if {'state','labels','y','y_true','target','targets','correctness'} & set(bank):
        raise ValueError('query label fields forbidden')
    tab=bank['tab']
    return dict(node_ids=bank['ids'],candidate_g=tab[:,:27],candidate_l=tab[:,27:54],
                symmetric_global=tab[:,54:],gnn_pred=bank['gnn_pred'],llm_pred=bank['llm_pred'])


def preregister(root):
    from eargtc.router_v8_learned.model import neural_grid
    from eargtc.router_v8_learned.experiment import tabular_grid
    root=Path(root);paths=set(code_paths())
    inputs={REPO/'runs/router_v8_posthoc_review_20260907/comparison.json'}
    reference_audit=read(REFERENCE/'FINAL_AUDIT.json')
    if not reference_audit['all_checks_pass']:raise ValueError('reference failed audit')
    if sha256(REFERENCE/'selection/FROZEN.json')!=reference_audit['frozen_sha256']:raise ValueError('reference freeze changed')
    check_hashes(reference_audit['result_hashes'],'reference result')
    for name in ('selection/FROZEN.json','FINAL_AUDIT.json','PRESERVED_VERSION.json','test/metrics.json'):
        inputs.add(REFERENCE/name)
    for d in DATASETS:
        inputs.add(REFERENCE/'test'/f'{d}_decisions.pt')
        manifest=root/'banks'/d/'manifest.json';m=read(manifest);inputs.add(manifest)
        # Earlier safe Joint export records the raw hidden bytes, unlike AUDIT alone.
        binding=REPO/'runs/joint_cross_repair_export_20260906_v2'/d
        binding_complete=read(binding/'complete.json');binding_meta=read(binding/'metadata.json')
        if binding_complete.get('dataset')!=d or binding_complete.get('test_labels_exported') is not False:
            raise ValueError('historical hidden binding identity mismatch')
        if sha256(binding/'metadata.json')!=binding_complete['artifacts'].get('metadata.json'):
            raise ValueError('historical hidden binding metadata changed')
        hidden_path=binding_meta['llm_hidden_path']
        if m['input_hashes'].get(hidden_path)!=binding_meta['source_hashes'].get(hidden_path) or hidden_path not in m['input_hashes']:
            raise ValueError('raw hidden differs from historical byte binding')
        inputs.update((binding/'complete.json',binding/'metadata.json'))
        check_hashes(m['input_hashes'],'raw provenance')
        inputs.update(Path(p) for p in m['input_hashes'])
        for name,h in m['bank_hashes'].items():
            p=root/'banks'/d/name
            if sha256(p)!=h:raise ValueError('prepared bank changed')
            inputs.add(p)
        old_manifest=OLD/'banks'/d/'manifest.json';om=read(old_manifest)
        inputs.update((old_manifest,OLD/'banks'/d/'query.pt',OLD/'banks'/d/'validation.pt'))
        graph=Path(om['graph_path'])
        expected=om['input_hashes'].get(str(graph))
        safe_meta=read(REPO/'runs/joint_direct_numbered_router_export_20260813_v1'/d/'safe_input_metadata.json')
        if Path(safe_meta['graph_path']).resolve()!=graph.resolve() or safe_meta['graph_path_sha256']!=expected:
            raise ValueError('raw GNN graph provenance differs from historical graph')
        if expected is None or sha256(graph)!=expected:raise ValueError('historical graph input changed')
        inputs.add(graph)
    pre=dict(schema='router_v8_learned_v4',seed=42,datasets=list(DATASETS),neural_grid=neural_grid(),tabular_grid=tabular_grid(),
        banks_root=str((root/'banks').resolve()),reference_root=str(REFERENCE),coverage=COVERAGE,
        top_k=[20,50,100,200,500,1000],primary_protocol='expanded_source_validation_and_target_calibration_crossfit',
        target_folds=5,source_protocol='leave_target_and_held_source_out; held halves reciprocal adaptation/evaluation',
        test_selection=False,transductive_query_consistency=True,
        code_hashes={str(p.resolve()):sha256(p) for p in paths},
        input_hashes={str(p.resolve()):sha256(p) for p in inputs},
        protocol_hashes={str(p.resolve()):sha256(p) for p in (WORK/'SPEC.md',WORK/'PLAN.md')})
    write(root/'PREREGISTERED.json',pre)
    print(json.dumps(dict(preregistered=str(root),neural_configs=len(pre['neural_grid']),heads=2)),flush=True)


def audit_inputs(root):
    from eargtc.router_v8_learned.data import load_bank
    from eargtc.router_v8.deployment import validation_halves
    root=Path(root);pre=verify_preregistration(root);checks={}
    for d in DATASETS:
        v=load_bank(root,d,'validation');q=load_bank(root,d,'query');u=load_bank(root,d,'unlabeled')
        oldv=torch.load(OLD/'banks'/d/'validation.pt',weights_only=False,map_location='cpu')
        oldq=torch.load(OLD/'banks'/d/'query.pt',weights_only=False,map_location='cpu')
        for a,b in ((v,oldv),(q,oldq)):
            for newkey,oldkey in (('ids','node_ids'),('gnn_pred','gnn_pred'),('llm_pred','llm_pred')):
                if not np.array_equal(arr(a[newkey]),arr(b[oldkey])):raise ValueError('historical population/expert changed')
        if not np.array_equal(arr(v['state']),arr(oldv['state'])):raise ValueError('validation labels changed')
        ca,po=validation_halves(arr(v['ids']),d)
        if not np.array_equal(ca,arr(v['cal_mask'])) or not np.array_equal(po,arr(v['policy_mask'])):raise ValueError('historical split changed')
        query_for_metrics(q);query_for_metrics(u)
        if not set(arr(u['ids']).tolist())<=set(arr(q['ids']).tolist()):raise ValueError('unlabeled pool not query subset')
        checks[d]=dict(historical_identity=True,validation_count=len(v['ids']),calibration_count=int(ca.sum()),
                       policy_count=int(po.sum()),test_conflict_count=len(q['ids']),unlabeled_count=len(u['ids']))
    write(root/'PREFLIGHT_AUDIT.json',dict(all_checks_pass=True,preregistration_sha256=fingerprint(pre),datasets=checks))
    print(json.dumps(checks),flush=True)


def verify_target_complete(folder,pre):
    folder=Path(folder);complete=read(folder/'COMPLETE.json');expected=fingerprint(pre)
    if complete.get('preregistration_sha256')!=expected:raise ValueError('target preregistration fingerprint mismatch')
    if set(complete.get('files',{}))!=set(TARGET_FILES):raise ValueError('target COMPLETE inventory mismatch')
    check_hashes({str(folder/n):h for n,h in complete['files'].items()},'target')
    if read(folder/'selection.json').get('preregistration_sha256')!=expected:raise ValueError('selection fingerprint mismatch')
    return complete


def verify_model_replay(root,dataset,pre,device='cpu'):
    """Recompute exported logits from checkpoints before any test label read."""
    from eargtc.router_v8_learned.data import load_bank
    from eargtc.router_v8_learned.experiment import _predict
    from eargtc.router_v8_exploration.core import apply_affine,score_payload
    root=Path(root);folder=root/'targets'/dataset
    model=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
    selected=read(folder/'selection.json')
    if model.get('preregistration_sha256')!=fingerprint(pre):raise ValueError('model replay fingerprint mismatch')
    if model['route_boundary']!=selected['route_boundary']:raise ValueError('model replay boundary mismatch')
    for head in ('trust','preference'):
        fit=model['heads'][head]
        if fit['head']!=head or fit['selected_config']!=selected['configs'][head] or fit['calibration']!=selected['calibration'][head]:
            raise ValueError('model replay selected head mismatch')
    checks={}
    for split,filename in (('query','scores.pt'),('validation','validation_scores.pt')):
        bank=load_bank(root,dataset,split);saved=torch.load(folder/filename,map_location='cpu',weights_only=False)
        raw={h:_predict(model['heads'][h],bank,device) for h in ('trust','preference')}
        calibrated={h:apply_affine(raw[h],selected['calibration'][h]) for h in raw}
        compatible=dict(node_ids=bank['ids'],gnn_pred=bank['gnn_pred'],llm_pred=bank['llm_pred'])
        expected=score_payload(compatible,calibrated['trust'],calibrated['preference'],selected['route_boundary'])
        for head in raw:expected['adapted_'+head+'_logit']=torch.as_tensor(raw[head])
        errors={}
        for key,value in expected.items():
            a,b=arr(value),arr(saved[key])
            if a.shape!=b.shape:raise ValueError('model replay shape mismatch: '+key)
            if key in ('node_ids','gnn_pred','llm_pred','route'):
                if not np.array_equal(a,b):raise ValueError('model replay discrete mismatch: '+key)
            else:
                if not np.isfinite(b).all() or not np.allclose(a,b,rtol=1e-5,atol=5e-5):raise ValueError('model replay float mismatch: '+key)
                errors[key]=float(np.max(np.abs(a-b))) if a.size else 0.
        if split=='validation':
            for saved_key,bank_key in (('state','state'),('calibration_mask','cal_mask'),('policy_mask','policy_mask')):
                if not np.array_equal(arr(saved[saved_key]),arr(bank[bank_key])):raise ValueError('model replay validation identity mismatch')
        checks[split]=dict(route_exact=True,node_ids_expert_predictions_exact=True,max_absolute_errors=errors,
                           replay_device=device,float_rtol=1e-5,float_atol=5e-5)
    return checks


def freeze_all(root):
    from eargtc.router_v8_learned.data import load_bank
    from eargtc.router_v8_learned.experiment import validate_source_artifacts
    root=Path(root);pre=verify_preregistration(root);replay={}
    audit=read(root/'PREFLIGHT_AUDIT.json')
    if not audit['all_checks_pass'] or audit['preregistration_sha256']!=fingerprint(pre):raise ValueError('preflight missing/stale')
    paths=[root/'PREREGISTERED.json',root/'PREFLIGHT_AUDIT.json']
    for group in ('code_hashes','input_hashes','protocol_hashes'):paths.extend(Path(p) for p in pre[group])
    for d in DATASETS:
        folder=root/'targets'/d;verify_target_complete(folder,pre)
        paths.extend([folder/'COMPLETE.json']+[folder/n for n in TARGET_FILES])
        bank=load_bank(root,d,'validation');sel=read(folder/'selection.json')
        for key,mask in (('calibration_ids','cal_mask'),('policy_ids','policy_mask')):
            if set(sel[key])!=set(arr(bank['ids'])[arr(bank[mask]).astype(bool)].tolist()):raise ValueError('target label-role IDs mismatch')
        query=load_bank(root,d,'query');scores=torch.load(folder/'scores.pt',weights_only=False,map_location='cpu')
        validate_scores(query_for_metrics(query),scores)
        replay[d]=verify_model_replay(root,d,pre)
        for held in DATASETS:
            if held==d:continue
            source=root/'source'/f'{d}__{held}';comp=validate_source_artifacts(root,d,held,pre)
            paths.append(source/'COMPLETE.json');paths.extend(source/n for n in comp['files'])
    write(root/'MODEL_REPLAY_AUDIT.json',dict(all_checks_pass=True,preregistration_sha256=fingerprint(pre),datasets=replay))
    paths.append(root/'MODEL_REPLAY_AUDIT.json')
    freeze_paths(paths,root/'selection/FROZEN.json',dict(datasets=list(DATASETS),seed=42,test_labels_accessed=False,primary_protocol=pre['primary_protocol']))
    print('All five learned Router models and policies frozen before test label access.',flush=True)


def verify_run_frozen(root):
    root=Path(root);frozen=verify_frozen(root/'selection/FROZEN.json')
    verify_preregistration(root,stage='source')
    return frozen


def evaluate_locked(root):
    root=Path(root);verify_run_frozen(root)
    pre=read(root/'PREREGISTERED.json');out=root/'test';out.mkdir(exist_ok=True)
    begin_once(out/'EVALUATION_STARTED')
    from eargtc.data import _install_pyg_optional_extension_stubs
    _install_pyg_optional_extension_stubs()
    results={}
    for d in pre['datasets']:
        man=read(OLD/'banks'/d/'manifest.json');folder=root/'targets'/d
        scores=torch.load(folder/'scores.pt',map_location='cpu',weights_only=False)
        query=torch.load(OLD/'banks'/d/'query.pt',map_location='cpu',weights_only=False)
        validate_scores(query,scores)
        graph=torch.load(man['graph_path'],map_location='cpu',weights_only=False)
        ids=arr(scores['node_ids']);g=arr(scores['gnn_pred']);l=arr(scores['llm_pred']);y=arr(graph.y).reshape(-1)[ids]
        state=np.where(y==g,0,np.where(y==l,1,2));del graph
        result,accepted=dataset_result(scores,state,read(folder/'selection.json'),pre,query)
        json.dumps(result,allow_nan=False)
        route=arr(scores['route']).astype(int)
        decisions=dict(**scores,state=torch.from_numpy(state),selected_action=torch.from_numpy(np.where(accepted,route,2)),
                       pseudo_label=torch.from_numpy(np.where(accepted,np.where(route==0,g,l),-1)))
        torch.save(decisions,out/f'{d}_decisions.pt');write(out/f'{d}_metrics.json',result);results[d]=result
        print(json.dumps(dict(dataset=d,trust_auc=result['metrics']['trust_auroc'],preference_auc=result['metrics']['preference_auroc'],
                              full_conflict_accuracy=result['metrics']['full_conflict_pseudo_accuracy'],chosen_top_fraction=result['chosen_top_fraction'])),flush=True)
    write(out/'metrics.json',results)
    write(out/'COMPLETE.json',dict(complete=True,datasets=pre['datasets'],frozen_sha256=sha256(root/'selection/FROZEN.json'),metrics_sha256=sha256(out/'metrics.json')))


def final_audit(root):
    root=Path(root);verify_run_frozen(root);pre=read(root/'PREREGISTERED.json');complete=read(root/'test/COMPLETE.json')
    if not complete['complete'] or complete['frozen_sha256']!=sha256(root/'selection/FROZEN.json') or complete['metrics_sha256']!=sha256(root/'test/metrics.json'):
        raise ValueError('evaluation completion binding changed')
    results=read(root/'test/metrics.json');checks={};paths=[root/'test/COMPLETE.json',root/'test/metrics.json']
    for d in DATASETS:
        s=torch.load(root/'test'/f'{d}_decisions.pt',weights_only=False,map_location='cpu')
        score=torch.load(root/'targets'/d/'scores.pt',weights_only=False,map_location='cpu')
        q=torch.load(OLD/'banks'/d/'query.pt',weights_only=False,map_location='cpu');validate_scores(q,score)
        old=torch.load(REFERENCE/'test'/f'{d}_decisions.pt',weights_only=False,map_location='cpu')
        for key in ('node_ids','gnn_pred','llm_pred','state'):
            if not np.array_equal(arr(s[key]),arr(old[key])):raise ValueError('historical test identity mismatch')
        for key in score:
            if not np.array_equal(arr(s[key]),arr(score[key])):raise ValueError('exported frozen score changed')
        expected,accepted=dataset_result(score,arr(s['state']),read(root/'targets'/d/'selection.json'),pre,q)
        if expected!=results[d] or expected!=read(root/'test'/f'{d}_metrics.json'):raise ValueError('metric recomputation mismatch')
        route=arr(score['route'])
        if not np.array_equal(arr(s['selected_action']),np.where(accepted,route,2)):raise ValueError('action mismatch')
        pseudo=np.where(accepted,np.where(route==0,arr(score['gnn_pred']),arr(score['llm_pred'])),-1)
        if not np.array_equal(arr(s['pseudo_label']),pseudo):raise ValueError('pseudo label mismatch')
        paths.extend((root/'test'/f'{d}_decisions.pt',root/'test'/f'{d}_metrics.json'))
        checks[d]=dict(old_population_expert_state_identity=True,all_metrics_recomputed=True,accepted_actions_pseudo_labels_verified=True)
    write(root/'FINAL_AUDIT.json',dict(all_checks_pass=True,checks=checks,frozen_sha256=sha256(root/'selection/FROZEN.json'),
                                     result_hashes={str(p.resolve()):sha256(p) for p in paths}))
    print('Final learned Router score/label/provenance audit passed.',flush=True)


def development_report(root):
    from eargtc.router_v8_learned.data import load_bank
    root=Path(root);rows={};lines=['# Learned Router-v8 developer results','',
        'Policy labels only select acceptance and supply development diagnostics; no test labels. Target calibration selected candidates by internal crossfit, so its metrics are selection diagnostics.','',
        '| Dataset | Policy Trust AUC | Policy Preference AUC | Policy conflict accuracy | Selected fraction | Policy accepted | Policy precision |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for d in DATASETS:
        bank=load_bank(root,d,'validation');s=torch.load(root/'targets'/d/'validation_scores.pt',weights_only=False,map_location='cpu')
        if not np.array_equal(arr(s['node_ids']),arr(bank['ids'])):raise ValueError('validation score ID order changed')
        po=arr(bank['policy_mask']).astype(bool);sel=read(root/'targets'/d/'selection.json')
        metrics=route_metrics(arr(s['q'])[po],arr(bank['state'])[po],arr(s['route'])[po],arr(s['node_ids'])[po])
        policy=sel['top_fraction_policy'];rows[d]=dict(policy_metrics=metrics,top_fraction_policy=policy,configs=sel['configs'])
        lines.append(f"| {d} | {fmt(metrics['trust_auroc'])} | {fmt(metrics['preference_auroc'])} | {fmt(metrics['full_conflict_pseudo_accuracy'],True)} | {fmt(policy['fraction'],True)} | {policy['validation_accepted']} | {fmt(policy['validation_accuracy'],True)} |")
    (root/'DEVELOPMENT_REPORT.md').write_text('\n'.join(lines)+'\n')
    (root/'DEVELOPMENT_SUMMARY.json').write_text(json.dumps(rows,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print('\n'.join(lines),flush=True)


def fmt(x,pct=False):return '—' if x is None else (f'{100*x:.2f}%' if pct else f'{x:.4f}')


def report(root):
    root=Path(root);audit=read(root/'FINAL_AUDIT.json')
    if not audit['all_checks_pass'] or audit['frozen_sha256']!=sha256(root/'selection/FROZEN.json'):raise ValueError('final audit missing/stale')
    check_hashes(audit['result_hashes'],'result');verify_preregistration(root,stage='source')
    results=read(root/'test/metrics.json');ref=read(REFERENCE/'test/metrics.json');v6=read(REPO/'runs/router_v8_posthoc_review_20260907/comparison.json')['datasets']
    lines=['# Router-v8：特征选择、可学习原始双专家表示与扩充监督','',
        '固定五图与历史seed42、专家及split。该版本沿用disagreement候选集合。先进行source-LODO，源validation进入Router监督训练；目标calibration参数训练通过五折产生honest选择分数。最终端点、校准、路由、policy和分数在测试前全五图冻结。test未用于本轮选模。历史benchmark结果已知，validation在连续开发中复用，不能声称全新测试或全局最优。','',
        '原始GNN128与LLM4096输入可学习投影；gold-trained GNN的gold行不用于新增监督。两个任务分别拟合，允许概率/全量/稀疏门控特征，以及无embedding/GNN/LLM/双embedding配置。增强配方包含mixup、排序与无标签query一致性；后者明确使用transductive输入，不把伪监督当真值。CPU监督特征选择和浅层树/逻辑回归作为有限候选。新增标签权限/特征可见性不同于归档v3，不能将收益全部归于架构。','',
        '## 最终测试双头与伪标签输出','',
        '| 数据集 | Trust AUC | Preference AUC | Preference G/L accuracy | 全disagreement accuracy | 接受比例 | 接受数 | Pseudo accuracy | Wilson10%接受数 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d in DATASETS:
        r=results[d];m=r['metrics'];p=r['chosen_top_fraction']
        lines.append(f"| {d} | {fmt(m['trust_auroc'])} | {fmt(m['preference_auroc'])} | {fmt(m['preference_accuracy'],True)} | {fmt(m['full_conflict_pseudo_accuracy'],True)} | {fmt(p['validation_selected_fraction'],True)} | {p['accepted_count']} | {fmt(p['pseudo_accuracy'],True)} | {r['policies']['0.1']['accepted_count']} |")
    lines+=['','接受比例由policy validation提前决定；这是比例策略，不享有Wilson的同样风险条件。拒绝全部时accuracy未定义。','',
        '## 与归档v3及v6的同人口比较','',
        '| 数据集 | v3 Trust AUC | 新 Trust AUC | v3 Preference AUC | 新 Preference AUC | v6 Preference AUC | v3路由accuracy | 新路由accuracy | v6路由accuracy |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    comparisons={}
    for d in DATASETS:
        a=ref[d]['metrics'];b=results[d]['metrics'];c=v6[d]['v6']
        comparisons[d]=dict(trust_delta_v3=b['trust_auroc']-a['trust_auroc'],preference_delta_v3=b['preference_auroc']-a['preference_auroc'],
                            route_delta_v3=b['full_conflict_pseudo_accuracy']-a['full_conflict_pseudo_accuracy'])
        cells=[fmt(a['trust_auroc']),fmt(b['trust_auroc']),fmt(a['preference_auroc']),fmt(b['preference_auroc']),fmt(c['preference_auc']),fmt(a['full_conflict_pseudo_accuracy'],True),fmt(b['full_conflict_pseudo_accuracy'],True),fmt(c['conflict_accuracy'],True)]
        lines.append('| '+d+' | '+' | '.join(cells)+' |')
    lines+=['','## 同覆盖率伪标签精度','',
        '| 数据集 | v3 Top5% | 新 Top5% | v3 Top10% | 新 Top10% | 新 Top25% | 新 Top50% |','|---|---:|---:|---:|---:|---:|---:|']
    for d in DATASETS:
        a=ref[d]['operating_points'];b=results[d]['operating_points']
        cells=[a['top_fraction_0.05'],b['top_fraction_0.05'],a['top_fraction_0.1'],b['top_fraction_0.1'],b['top_fraction_0.25'],b['top_fraction_0.5']]
        lines.append('| '+d+' | '+' | '.join(f"{fmt(x['pseudo_accuracy'],True)} / {x['accepted_count']}" for x in cells)+' |')
    lines+=['','## 实际选定配置','']
    for d in DATASETS:
        s=read(root/'targets'/d/'selection.json')
        lines.extend([f'### {d}','',f"`{json.dumps(s['configs'],ensure_ascii=False,sort_keys=True)}`",'',
                      f'完整候选、每折训练/评价ID与OOF结果：targets/{d}/target_selection.json。源域筛选：source_selection.json；最终模型和特征变换：model.pt。',''])
    lines+=['## 可比性和交付边界','',
        'OOF分数来自每折约80% calibration行训练的模型，最终分数来自全部calibration重拟合模型；在OOF分数上拟合的affine与路由阈值被迁移到该重拟合模型，存在明确的校准分布变化。内部交叉拟合不能消除这种迁移误差。','',
        '源域validation训练和目标calibration参数训练增加了Router监督标签预算；专家本身仍是原5-shot。原始GNN坐标在不同图之间可能不对齐，原始LLM Arxiv缓存继承2/173审计argmax不一致的限制，原专家logits保持冻结。共享域偏移、内部CV选模偏差和校准/测试分布差异仍可能造成退步；没有将单seed观察包装成统计稳定性。','',
        '没有执行完整baselines、机制消融和多seed稳定性。两头可选择不同配置；若某类embedding或增强候选没有被选择，仍保留其源域和目标CV结果，不用测试替换选择。更多指标、错误分解、轻量confidence参照见test/metrics.json。','',
        'Pipeline使用test/<dataset>_decisions.pt的node_ids和pseudo_label（-1拒绝），不能把审计真值state输入训练。所有新成果独立于已经归档的v3；任务结束前进行哈希和结果复算审计。','']
    if (root/'DEVELOPMENT_REPORT.md').exists():lines.append((root/'DEVELOPMENT_REPORT.md').read_text().replace('# Learned Router-v8 developer results','## 开发过程指标',1))
    text='\n'.join(lines)
    (root/'REPORT.md').write_text(text)
    (REPO/'docs/router_v8_learned_results_2026-09-07.md').write_text(text)
    (root/'test/comparison.json').write_text(json.dumps(comparisons,indent=2,sort_keys=True)+'\n')
    print(str(root/'REPORT.md'),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preregister','audit','freeze','evaluate','final-audit','report','development']);parser.add_argument('--root',type=Path,required=True);a=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run numerical stages through Slurm')
    {'preregister':preregister,'audit':audit_inputs,'freeze':freeze_all,'evaluate':evaluate_locked,'final-audit':final_audit,'report':report,'development':development_report}[a.stage](a.root.resolve())

if __name__=='__main__':main()
