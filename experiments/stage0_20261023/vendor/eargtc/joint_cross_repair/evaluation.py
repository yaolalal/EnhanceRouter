"""Evaluation-only access to test gold; run after all five selections are frozen."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from .artifacts import DATASETS,read_json,write_json,verify_files,sha256,verify_training_link


def labels_for_ids(rows,ids):
    mapping={int(r['node_id']):int(r['true_label']) for r in rows}
    if len(mapping)!=len(rows):raise ValueError('duplicate evaluation labels')
    if len(ids)!=len(ids.unique()) or set(mapping)!=set(ids.tolist()):raise ValueError('evaluation labels do not exactly cover test IDs')
    return torch.tensor([mapping[int(i)] for i in ids],dtype=torch.long)


def classification_report(probability,y):
    n=len(y)
    if n==0:return {'count':0,'accuracy':None,'macro_f1':None,'nll':None}
    p=probability.double()
    if p.ndim!=2 or len(p)!=n or not bool(torch.isfinite(p).all()) or bool((p<0).any()):raise ValueError('invalid prediction probabilities')
    if not torch.allclose(p.sum(1),torch.ones(n,dtype=torch.float64),atol=1e-5):raise ValueError('probabilities not normalized')
    c=p.shape[1];pred=p.argmax(1)
    if bool(((y<0)|(y>=c)).any()):raise ValueError('evaluation class outside class space')
    matrix=torch.bincount(y*c+pred,minlength=c*c).reshape(c,c).double()
    tp=matrix.diag();denom=matrix.sum(0)+matrix.sum(1)
    f1=torch.where(denom>0,2*tp/denom,torch.zeros_like(tp))
    return {'count':n,'accuracy':float((pred==y).double().mean()),'macro_f1':float(f1.mean()),
            'nll':float(-p[torch.arange(n),y].clamp_min(1e-12).log().mean())}


def correction_damage(base,new,y):
    b=base==y;j=new==y
    corrections=int((~b&j).sum());damages=int((b&~j).sum())
    return {'corrections':corrections,'damages':damages,'stable_correct':int((b&j).sum()),
            'stable_wrong':int((~b&~j).sum()),'gain':(corrections-damages)/len(y) if len(y) else None}


def evaluate_all(run_root,export_root,frozen_root):
    run_root,export_root,frozen_root=map(Path,[run_root,export_root,frozen_root])
    output=run_root/'evaluation'
    if output.exists():raise FileExistsError('evaluation already exists; do not overwrite locked results')
    # The barrier is checked before opening any dataset's test decisions.
    locked=read_json(run_root/'FROZEN.json')
    if set(locked['datasets'])!=set(DATASETS):raise ValueError('all five datasets must be frozen before test evaluation')
    verify_files(locked['files'])
    preflight={}
    for ds in DATASETS:
        config=verify_training_link(run_root/ds,export_root/ds)
        export=export_root/ds
        ec=read_json(export/'complete.json')
        verify_files({str(export/k):v for k,v in ec['artifacts'].items()})
        meta=read_json(export/'metadata.json')
        verify_files(meta['source_hashes'])
        preflight[ds]=(config,meta)
    results={}
    for ds in DATASETS:
        export=export_root/ds;run=run_root/ds
        config,meta=preflight[ds]
        sup=torch.load(export/'supervision.pt',weights_only=True,map_location='cpu')
        features=torch.load(export/'features.pt',weights_only=True,map_location='cpu',mmap=True)
        pools=torch.load(export/'pools.pt',weights_only=True,map_location='cpu')
        pred=torch.load(run/'predictions.pt',weights_only=True,map_location='cpu')
        if not torch.equal(pred['node_ids'],features['node_ids']):raise ValueError('prediction IDs differ from export')
        test=sup['test_ids']
        target_path=frozen_root/ds/'decisions.jsonl'
        rows=[json.loads(line) for line in target_path.open() if line.strip()]
        y=labels_for_ids(rows,test)
        by_id={int(r['node_id']):r for r in rows}
        for i in test.tolist():
            row=by_id[i]
            if int(pred['p_b'][i].argmax())!=row['selected_pred'] or int(pred['p_g'][i].argmax())!=row['gnn_pred'] or int(pred['p_l'][i].argmax())!=row['llm_pred']:
                raise ValueError('frozen Router/expert prediction identity failed')
            if abs(float(features['trust'][i])-row['selected_trust'])>1e-6:raise ValueError('frozen trust identity failed')
        metrics={key:classification_report(value[test],y) for key,value in pred.items() if key!='node_ids'}
        delta=correction_damage(pred['p_b'][test].argmax(1),pred['p_joint'][test].argmax(1),y)
        if abs(delta['gain']-(metrics['p_joint']['accuracy']-metrics['p_b']['accuracy']))>1e-10:raise ValueError('correction-damage identity failed')
        selected=torch.cat([pools['teach_g'],pools['teach_l'],pools['agreement']]).unique()
        groups={'selected_for_any_loss':torch.isin(test,selected),'unselected_for_any_loss':~torch.isin(test,selected),
                'agreement':features['agreement'][test],'conflict':~features['agreement'][test]}
        group_metrics={name:{key:classification_report(pred[key][test[mask]],y[mask]) for key in ['p_b','p_joint']} for name,mask in groups.items()}
        pool_audit={}
        for name,ids in pools.items():
            mask=torch.isin(test,ids)
            pool_audit[name]={'count':len(ids),'teacher_accuracy':float((features['selected_pred'][test[mask]]==y[mask]).double().mean()) if len(ids) else None}
        branch_transitions={name:correction_damage(pred[base][test].argmax(1),pred[new][test].argmax(1),y)
                            for name,base,new in [('g','p_g','p_g_repaired'),('l','p_l','p_l_repaired')]}
        g0,l0=pred['p_g'][test].argmax(1)==y,pred['p_l'][test].argmax(1)==y
        g1,l1=pred['p_g_repaired'][test].argmax(1)==y,pred['p_l_repaired'][test].argmax(1)==y
        def states(g,l):return {'both_correct':int((g&l).sum()),'g_only':int((g&~l).sum()),'l_only':int((~g&l).sum()),'neither':int((~g&~l).sum()),'oracle_accuracy':float((g|l).double().mean())}
        results[ds]={'dataset':ds,'protocol':meta['protocol'],'metrics':metrics,'joint_vs_b':delta,
                     'branch_transitions':branch_transitions,'states_before':states(g0,l0),'states_after':states(g1,l1),
                     'groups':group_metrics,'pool_audit_evaluation_only':pool_audit,'test_target_sha256':sha256(target_path),
                     'selection':read_json(run/'selection.json'),'policies':read_json(export/'policies.json'),
                     'training_config':config['train'],'candidate_count':meta['pseudo_candidate_count'],
                     'split_counts':{k:meta[k] for k in ['gold_count','model_selection_count','threshold_count','test_count']}}
    output.mkdir()
    write_json(output/'results.json',results)
    lines=['# Joint Cross Repair: five-dataset transductive evaluation','','Test-node pseudo supervision is transductive, not strict-holdout evaluation. Test labels were opened only after all five selections, export links and source manifests were verified.','','| Dataset | G0 | L0 | B | G+ | L+ | Joint | Base fusion | Corrections / damages |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ds,r in results.items():
        m=r['metrics'];vals=' | '.join(f"{m[k]['accuracy']:.4f}" for k in ['p_g','p_l','p_b','p_g_repaired','p_l_repaired','p_joint','p_base_fusion'])
        d=r['joint_vs_b'];lines.append(f"| {ds} | {vals} | {d['corrections']} / {d['damages']} |")
    lines+=['','Raw embeddings, zero-initialized independent residual heads, gold CE + directed pseudo CE + retention KL. One frozen teacher generation; no feedback.','Validation-threshold halves select top-k percentages per source/conflict and agreement; historical calibration halves select checkpoints and fusion. The upstream frozen Router already used official validation.','Optimization seeds and candidate counts are listed below; no statistical robustness claim or test-driven retuning. Arxiv raw cache retains documented numerical-parity limitations.','Detailed policies, pool membership, split counts, selected/unselected metrics and provenance are saved beside this report or in the associated exports.']
    lines+=['','| Dataset | Seed | Pseudo candidates / test |','|---|---:|---:|']
    for ds,r in results.items():
        lines.append(f"| {ds} | {r['training_config']['seed']} | {r['candidate_count']} / {r['split_counts']['test_count']} |")
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(output/'complete.json',{'all_checks_pass':True,'artifacts':{k:sha256(output/k) for k in ['results.json','REPORT.md']}})
    return results
