"""Report completed D3 final accuracy after all75 models pass their hash gate."""
from d3_recovery import *
setup()
import torch,csv,statistics
from report import accuracy,boot
checked=read(RUN/'D3_ALL75_RECOVERED_VERIFIED.json')
assert checked['all_checks_pass'] and checked['cells']==75
baseline_root=RUN/'completed-l2-20261027'
for rel,h in read(baseline_root/'COMPLETE.json')['files'].items():assert digest(baseline_root/rel)==h
reference=read(baseline_root/'RESULTS.json')
assert reference['seeds']==list(DEV)
base={r['key']:r for r in reference['per_cell'] if r['variant']=='b0'}
assert len(base)==75
out=RUN/'completed-d3-l2-20261027';out.mkdir()
rows=[];truths={}
for sh,se,ds in CELLS:
 p=folder(sh,se,ds)/'d3'
 assert digest(p/'COMPLETE.json')==checked['files'][str(p/'COMPLETE.json')]
 if ds not in truths:truths[ds]=graph(ds).y.reshape(-1).long()
 ids=torch.tensor(evaluation_ids(se,ds));assert len(ids)==1000
 result=accuracy(p,ids,truths[ds]);ref=base[key(sh,se,ds)]
 head_path=folder(sh,se,ds)/'small_native/HEAD_COMPLETE.json';head=read(head_path)
 rows.append(dict(key=key(sh,se,ds),dataset=ds,shot=sh,seed=se,accuracy=result['accuracy'],macro_f1=result['macro_f1'],B0_accuracy=ref['accuracy'],delta_pp=100*(result['accuracy']-ref['accuracy']),validation_count=head['validation_count'],actual_per_class=head['actual_per_class'],head_receipt_sha256=digest(head_path),D3_seal_sha256=digest(p/'COMPLETE.json')))
summary=[]
for sh in SHOTS:
 for ds in DS:
  group=[r for r in rows if (r['shot'],r['dataset'])==(sh,ds)]
  summary.append(dict(dataset=ds,shot=sh,seeds=5,B0_accuracy_mean=statistics.mean(r['B0_accuracy'] for r in group),B0_accuracy_std_ddof0=statistics.pstdev(r['B0_accuracy'] for r in group),D3_accuracy_mean=statistics.mean(r['accuracy'] for r in group),D3_accuracy_std_ddof0=statistics.pstdev(r['accuracy'] for r in group),delta_mean_pp=statistics.mean(r['delta_pp'] for r in group)))
comparison=boot([r['delta_pp'] for r in rows]);comparison['per_dataset']={ds:boot([r['delta_pp'] for r in rows if r['dataset']==ds]) for ds in DS}
write(out/'RESULTS.json',dict(scope='D3 completed final-accuracy evaluation only; no Stage0 acceptance decision',seeds=list(DEV),per_cell=rows,summary=summary,paired_delta=comparison,L1_evaluated=False,L1_budget='deferred',Stage0_decision_complete=False,D3_gate_sha256=digest(RUN/'D3_ALL75_RECOVERED_VERIFIED.json'),B0_report_sha256=digest(baseline_root/'COMPLETE.json'),report_code_sha256=digest(__file__),source=metadata()))
for filename,data in [('per_cell.csv',rows),('summary.csv',summary)]:
 with (out/filename).open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
lines=['# D3 small-validation final-accuracy results','', 'All75 D3 models passed the completion/hash gate before evaluation. Split seeds42–46, own-seed fixed eval1000 shared across shots.','', 'D3 caps each class at k labels from the original validation pool. Classes with fewer than k available labels retain their actual counts; all per-class counts are recorded in RESULTS.json. L1 and Stage0 acceptance are not assessed.','', '| Dataset | Shot | B0 accuracy (%) | D3 accuracy (%) | Paired delta (pp) |','|---|---:|---:|---:|---:|']
for r in summary:
 lines.append(f"| {r['dataset']} | {r['shot']} | {100*r['B0_accuracy_mean']:.2f} ± {100*r['B0_accuracy_std_ddof0']:.2f} | {100*r['D3_accuracy_mean']:.2f} ± {100*r['D3_accuracy_std_ddof0']:.2f} | {r['delta_mean_pp']:+.2f} |")
lines+=['',f"Overall paired accuracy delta={comparison['mean']:+.4f}pp,95%CI=[{comparison['lower']:.4f},{comparison['upper']:.4f}],n={comparison['n']}."]
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
seal(out,dict(all_checks_pass=True,cells=75,scope='D3 final accuracy only',L1_evaluated=False,Stage0_complete=False))
print('\n'.join(lines),flush=True)
