"""Separate post-freeze evaluator. No model fitting or selection is allowed here."""
import csv
import json
from pathlib import Path
import statistics
import torch
from eargtc.joint_cross_repair.artifacts import DATASETS,read_json,write_json,sha256
from eargtc.joint_cross_repair.evaluation import classification_report,correction_damage,labels_for_ids
from eargtc.joint_cross_repair.training import _score,scheme_b_probability
from .experiment import preflight
from .policy import apply_policy,choose_from_statistics,make_folds

METHODS=('gnn','llm','router','v1','v2','base','static','confidence','agreement_guard','primary','direct')
def _audit_pretest_dataset(run,export):
    """Verify all label-free selected outputs before opening test decisions."""
    run=Path(run);export=Path(export)
    selection=read_json(run/'selection.json')
    stats=torch.load(run/'validation.pt',weights_only=True,map_location='cpu')
    pred=torch.load(run/'predictions.pt',weights_only=True,map_location='cpu',mmap=True)
    features=torch.load(export/'features.pt',weights_only=True,map_location='cpu',mmap=True)
    supervision=torch.load(export/'supervision.pt',weights_only=True,map_location='cpu')
    ids=supervision['selection_ids']
    if not torch.equal(stats['node_ids'],ids):raise ValueError('validation node IDs differ from safe supervision')
    if not torch.equal(stats['folds'],make_folds(ids)):raise ValueError('validation folds differ from deterministic selection folds')
    if not torch.equal(pred['node_ids'],features['node_ids']):raise ValueError('prediction node IDs differ from safe features')
    router=scheme_b_probability(features['logp_g'].exp(),features['logp_l'].exp(),
        features['choose_g'],features['agreement'])[ids]
    if not torch.allclose(stats['router'],router,atol=2e-6,rtol=0):raise ValueError('saved router differs from safe scheme B')
    observed=choose_from_statistics(read_json(run/'candidate_catalog.json'),stats['correct'],stats['nll_sum'],stats['counts'])
    if observed!=selection:raise ValueError('frozen selection cannot be reproduced')
    picks=dict(selection['families']);picks.update(primary=selection['primary'],direct=selection['direct'])
    for name,pick in picks.items():
        expected=apply_policy(pick['candidate'],stats['branches'],stats['router'],features['agreement'][ids])
        actual=pred['methods'].get(name)
        if actual is None:raise ValueError(f'missing saved {name} output')
        actual=actual[ids]
        if not torch.allclose(actual,expected,atol=2e-6,rtol=0) or not torch.equal(actual.argmax(1),expected.argmax(1)):
            raise ValueError(f'saved {name} output differs from reconstructed policy')
        score=_score(actual,supervision['selection_targets'])
        if abs(score['accuracy']-pick['accuracy'])>1e-10 or abs(score['nll']-pick['nll'])>2e-6:
            raise ValueError(f'saved {name} validation score differs from frozen selection')
    family=selection['primary']['family']
    if family not in pred['methods'] or not torch.equal(pred['methods']['primary'],pred['methods'][family]):
        raise ValueError('primary output differs from frozen family')
    direct_family=selection['direct']['family']
    if direct_family not in pred['methods'] or not torch.equal(pred['methods']['direct'],pred['methods'][direct_family]):
        raise ValueError('direct output differs from its frozen family')
    return selection




def evaluate(run_root):
    root=Path(run_root).resolve();dest=root/'evaluation'
    if dest.exists():raise FileExistsError('evaluation output is immutable')
    config=preflight(root);torch.set_num_threads(2)
    # Reproduce every label-free frozen choice before any test target is opened.
    for ds in DATASETS:
        _audit_pretest_dataset(root/ds,Path(config['export_root'])/ds)
    results={}
    for ds in DATASETS:
        run=root/ds;export=Path(config['export_root'])/ds
        pred=torch.load(run/'predictions.pt',weights_only=True,map_location='cpu',mmap=True)
        f=torch.load(export/'features.pt',weights_only=True,map_location='cpu',mmap=True)
        sup=torch.load(export/'supervision.pt',weights_only=True,map_location='cpu')
        pools=torch.load(export/'pools.pt',weights_only=True,map_location='cpu')
        if not torch.equal(pred['node_ids'],f['node_ids']):raise ValueError('node alignment differs')
        target=Path(config['router_root'])/ds/'decisions.jsonl'
        rows=[json.loads(line) for line in target.open() if line.strip()]
        test=sup['test_ids'];y=labels_for_ids(rows,test);by_id={r['node_id']:r for r in rows}
        for key,field in [('gnn','gnn_pred'),('llm','llm_pred'),('router','selected_pred')]:
            expected=torch.tensor([by_id[int(i)][field] for i in test])
            if not torch.equal(pred['baselines'][key][test].argmax(1),expected):raise ValueError('July29 expert or Router identity failed')
        trust=torch.tensor([by_id[int(i)]['selected_trust'] for i in test],dtype=torch.float64)
        if not torch.allclose(f['trust'][test],trust,atol=1e-6,rtol=0):raise ValueError('frozen trust differs')
        methods={**pred['baselines'],**pred['methods']}
        metrics={k:classification_report(p[test],y) for k,p in methods.items()}
        transitions={k:correction_damage(methods[k][test].argmax(1),methods['primary'][test].argmax(1),y) for k in ('router','v1','v2')}
        for key,t in transitions.items():
            if abs(t['gain']-(metrics['primary']['accuracy']-metrics[key]['accuracy']))>1e-10:raise ValueError('correction/damage identity failed')
        seeds={}
        for key,ps in pred['seeds'].items():
            reports=[classification_report(p[test],y) for p in ps];acc=[r['accuracy'] for r in reports]
            seeds[key]={'metrics':reports,'accuracy_mean':statistics.mean(acc),'accuracy_sd':statistics.stdev(acc) if len(acc)>1 else 0.,
                'note':'same selected policy per seed; primary applies gate AFTER branch probability averaging, not average of nonlinear seed gates'}
        selected=torch.cat([pools['teach_g'],pools['teach_l'],pools['agreement']]).unique()
        masks={'selected':torch.isin(test,selected),'unselected':~torch.isin(test,selected),
               'agreement':f['agreement'][test],'conflict':~f['agreement'][test]}
        groups={g:{k:classification_report(methods[k][test[m]],y[m]) for k in ('router','v1','v2','primary')} for g,m in masks.items()}
        results[ds]={'metrics':metrics,'transitions':transitions,'seed_results':seeds,'groups':groups,
            'selection':read_json(run/'selection.json'),'metadata':read_json(run/'metadata.json'),'test_target_sha256':sha256(target)}
    dest.mkdir();write_json(dest/'results.json',results)
    csvrows=[]
    for ds,r in results.items():
        row={'dataset':ds,'primary_family':r['selection']['primary']['family'],'test_count':r['metadata']['split_counts']['test_count']}
        for method in METHODS:
            for metric in ('accuracy','macro_f1','nll'):row[f'{method}_{metric}']=r['metrics'][method][metric]
        for base in ('router','v1','v2'):row[f'delta_{base}_pp']=100*r['transitions'][base]['gain']
        csvrows.append(row)
    with (dest/'performance.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    lines=['# Joint four-view V3 — frozen validation-selected results','',
        'Fixed July29 Structural-OOF Router-v6, exact original five-dataset partitions and pseudo pools. 45 fresh trajectories, seeds42/43/44, fixed50/150/300/500 snapshots; gold anchors in both heads. No selection gradients or test targets in training.','',
        'Primary: family selected using five-fold CV within historical model-selection half, then configuration selected on that whole half. Direct is a prespecified full-validation-selection diagnostic. Neither is selected using test performance. All five were frozen before this evaluator opened test truth.','',
        '| Dataset | G0 | L0 | Router | V1 | V2 | Base | Static4 | Confidence4 | Agreement guard | V3 primary | Direct diagnostic | ΔV2 pp |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ds,r in results.items():
        lines.append('| '+ds+' | '+' | '.join(f"{100*r['metrics'][m]['accuracy']:.2f}" for m in METHODS)+f" | {100*r['transitions']['v2']['gain']:+.3f} |")
    mean={m:statistics.mean(r['metrics'][m]['accuracy'] for r in results.values()) for m in METHODS}
    lines.append('| Macro mean | '+' | '.join(f'{100*mean[m]:.2f}' for m in METHODS)+f" | {100*(mean['primary']-mean['v2']):+.3f} |")
    lines+=['','| Dataset | Chosen family | Selector CV accuracy | Refit validation accuracy | Primary test macro-F1 | NLL | Single-seed mean ± SD | Corrections / damages vs V2 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for ds,r in results.items():
        p=r['selection']['primary'];m=r['metrics']['primary'];ss=r['seed_results']['primary'];t=r['transitions']['v2']
        lines.append(f"| {ds} | {p['family']} | {100*p['oof_accuracy']:.2f} | {100*p['accuracy']:.2f} | {100*m['macro_f1']:.2f} | {m['nll']:.4f} | {100*ss['accuracy_mean']:.2f} ± {100*ss['accuracy_sd']:.2f} | {t['corrections']} / {t['damages']} |")
    lines+=['','## Limits of the evidence','',
        '- Confidence gates are deterministic prediction rules, not trained estimates of repair benefit. Full-support upstream dependencies prevent claiming a clean gold-OOF utility gate.','- Selector CV is used to choose the family; its winning score is not an independent test estimate. Upstream frozen Router historically used full official validation. Previous V1/V2 test outcomes were already observed, so this is exploratory follow-up.','- V1 and V2 are comparison baselines, excluded from new selector CV because their checkpoints already used the whole selection half. Thus the procedure cannot guarantee no regression to historical models.','- Three-seed branch probabilities are averaged before applying the selected gate. For nonlinear gates this differs from averaging three separately gated outputs; the single-seed statistics quantify those separate runs.','- Best means selected within this finite policy search. It does not mean a true five-dataset performance ceiling. No post-test retuning or per-dataset cherry-picking occurred.','']
    (dest/'REPORT.md').write_text('\n'.join(lines))
    write_json(dest/'complete.json',{'all_checks_pass':True,'frozen_sha256':sha256(root/'FROZEN.json'),
        'artifacts':{k:sha256(dest/k) for k in ('results.json','performance.csv','REPORT.md')}})
    return results
