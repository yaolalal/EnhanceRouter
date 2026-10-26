"""One final label-consuming evaluator; no fitting or model selection occurs here."""
import csv
import json
from pathlib import Path
import statistics
import torch
from eargtc.joint_cross_repair.artifacts import DATASETS,read_json,write_json,sha256
from eargtc.joint_cross_repair.evaluation import classification_report,correction_damage,labels_for_ids
from .experiment import preflight


def evaluate(run_root):
    root=Path(run_root).resolve();output=root/'evaluation'
    if output.exists():raise FileExistsError('final evaluation is immutable')
    # This verifies every dataset and every source before the first test-label open.
    config=preflight(root);torch.set_num_threads(2);results={}
    for ds in DATASETS:
        export=Path(config['export_root'])/ds;run=root/ds
        f=torch.load(export/'features.pt',weights_only=True,map_location='cpu',mmap=True)
        s=torch.load(export/'supervision.pt',weights_only=True,map_location='cpu')
        pools=torch.load(export/'pools.pt',weights_only=True,map_location='cpu')
        pred=torch.load(run/'predictions.pt',weights_only=True,map_location='cpu')
        selection=read_json(run/'selection.json');meta=read_json(run/'dataset_metadata.json')
        if not torch.equal(pred['node_ids'],f['node_ids']):raise ValueError('prediction node IDs differ')
        chosen=selection['optimized']['family']
        expected=pred['families'][chosen]['ensemble'] if chosen in pred['families'] else pred['baselines'][chosen]
        if not torch.equal(pred['families']['optimized']['ensemble'],expected):raise ValueError('locked winner differs from exported predictions')
        target=Path(config['router_root'])/ds/'decisions.jsonl'
        rows=[json.loads(line) for line in target.open() if line.strip()]
        test=s['test_ids'];y=labels_for_ids(rows,test);by_id={r['node_id']:r for r in rows}
        for key,field in [('gnn','gnn_pred'),('llm','llm_pred'),('router','selected_pred')]:
            reference=torch.tensor([by_id[int(i)][field] for i in test])
            if not torch.equal(pred['baselines'][key][test].argmax(1),reference):raise ValueError('July29 frozen expert/Router identity failed')
        reference_trust=torch.tensor([by_id[int(i)]['selected_trust'] for i in test],dtype=torch.float64)
        if not torch.allclose(f['trust'][test],reference_trust,atol=1e-6,rtol=0):raise ValueError('frozen trust differs')
        methods={**pred['baselines'],**{k:v['ensemble'] for k,v in pred['families'].items()}}
        metrics={k:classification_report(v[test],y) for k,v in methods.items()}
        transitions={base:correction_damage(methods[base][test].argmax(1),methods['optimized'][test].argmax(1),y) for base in ['router','v1']}
        for base,value in transitions.items():
            if abs(value['gain']-(metrics['optimized']['accuracy']-metrics[base]['accuracy']))>1e-10:raise ValueError('correction identity failed')
        seed_results={}
        for family,value in pred['families'].items():
            scores=[classification_report(p[test],y) for p in value['seeds']]
            acc=[v['accuracy'] for v in scores]
            seed_results[family]={'seeds':config['seeds'],'metrics':scores,'accuracy_mean':statistics.mean(acc),
                                  'accuracy_sample_sd':statistics.stdev(acc) if len(acc)>1 else 0.,
                                  'mode':'frozen_baseline' if family=='optimized' and chosen in pred['baselines'] else 'shared_validation_selected_recipe_across_training_seeds'}
        selected=torch.cat([pools['teach_g'],pools['teach_l'],pools['agreement']]).unique()
        masks={'selected':torch.isin(test,selected),'unselected':~torch.isin(test,selected),
               'agreement':f['agreement'][test],'conflict':~f['agreement'][test]}
        groups={name:{k:classification_report(methods[k][test[mask]],y[mask]) for k in ['router','v1','optimized']} for name,mask in masks.items()}
        def state(g,l):
            a=g[test].argmax(1)==y;b=l[test].argmax(1)==y
            return {'both_correct':int((a&b).sum()),'g_only':int((a&~b).sum()),'l_only':int((~a&b).sum()),
                    'neither':int((~a&~b).sum()),'oracle_accuracy':float((a|b).double().mean())}
        states={'original':state(methods['gnn'],methods['llm'])}
        states.update({k:state(v['g'],v['l']) for k,v in pred['repaired'].items()})
        results[ds]={'metrics':metrics,'seed_results':seed_results,'transitions':transitions,'groups':groups,
                     'states':states,'selection':selection,'metadata':meta,'test_target_sha256':sha256(target)}
    output.mkdir()
    write_json(output/'results.json',results)
    methods=['gnn','llm','router','v1','base_fusion','gold_only','weighted','scaled','joint','optimized']
    csvrows=[]
    for ds,r in results.items():
        row={'dataset':ds,'validation_selected_family':r['selection']['optimized']['family'],'test_count':r['metadata']['split_counts']['test_count']}
        for method in methods:
            for metric in ['accuracy','macro_f1','nll']:row[method+'_'+metric]=r['metrics'][method][metric]
        row.update({'delta_router_pp':100*r['transitions']['router']['gain'],'delta_v1_pp':100*r['transitions']['v1']['gain'],
                    'optimized_seed_mean_accuracy':r['seed_results']['optimized']['accuracy_mean'],
                    'optimized_seed_sd_accuracy':r['seed_results']['optimized']['accuracy_sample_sd']})
        csvrows.append(row)
    with (output/'performance.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    lines=['# Joint Cross Repair V2 — validation-selected search results','',
           'Fixed July29 Router-v6, original full-test pseudo candidates, exact historical splits, nine loss recipes and seeds 42/43/44. Test labels opened only after all five selections and provenance manifests passed. Prior V1 test results were already observed: this is follow-up exploratory research.','',
           'Primary optimized output is selected on validation, not by test accuracy. Repaired families use mean probability across all three seeds. The best refers only to the prespecified finite search.','',
           '| Dataset | G0 | L0 | Router B | V1 | Base fusion | Gold only | Weights | + scale | Joint selection | Optimized | ΔV1 pp |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ds,r in results.items():
        values=' | '.join(f"{100*r['metrics'][m]['accuracy']:.2f}" for m in methods)
        lines.append(f"| {ds} | {values} | {100*r['transitions']['v1']['gain']:+.3f} |")
    means={m:statistics.mean(r['metrics'][m]['accuracy'] for r in results.values()) for m in methods}
    lines.append('| Macro mean | '+' | '.join(f'{100*means[m]:.2f}' for m in methods)+f" | {100*(means['optimized']-means['v1']):+.3f} |")
    lines+=['','| Dataset | Selected family | Validation accuracy | Single-seed mean ± SD | Ensemble accuracy | Macro F1 | NLL | Corrections / damages vs V1 |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for ds,r in results.items():
        pick=r['selection']['optimized'];m=r['metrics']['optimized'];seed=r['seed_results']['optimized'];d=r['transitions']['v1']
        lines.append(f"| {ds} | {pick['family']} | {100*pick['accuracy']:.2f} | {100*seed['accuracy_mean']:.2f} ± {100*seed['accuracy_sample_sd']:.2f} | {100*m['accuracy']:.2f} | {100*m['macro_f1']:.2f} | {m['nll']:.4f} | {d['corrections']} / {d['damages']} |")
    lines+=['','All values except NLL/counts are percentages. Single-seed metrics use the same validation-selected recipe, checkpoint criterion, residual scale and fusion across seeds; this is not best-seed reporting. A frozen V1/B fallback has no new seed variation. Family comparisons are exploratory and have different selection budgets, not matched-budget causal ablations.','',
            'Full candidate metadata, seed-specific checkpoint steps, training artifacts, frozen selections, selected/unselected metrics and oracle diagnostics accompany this report. No post-test retuning was performed.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(output/'complete.json',{'all_checks_pass':True,'frozen_sha256':sha256(root/'FROZEN.json'),
             'artifacts':{k:sha256(output/k) for k in ['results.json','performance.csv','REPORT.md']}})
    return results
