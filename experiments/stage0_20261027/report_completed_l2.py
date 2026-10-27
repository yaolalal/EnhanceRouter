"""Report only completed B0/D4 final accuracy; never resolves the postponed L1 issue."""
from d3_recovery import *
setup()
import torch,csv,statistics
from report import accuracy,compare
out=RUN/'completed-l2-20261027';assert not out.exists()
variants=['b0','d4/confidence_router','d4/confidence_preference'];seals={}
for sh,se,ds in CELLS:
 for v in variants:
  path=folder(sh,se,ds)/v
  for rel,h in read(path/'COMPLETE.json')['files'].items():assert digest(path/rel)==h,(path,rel)
  seals[str(path/'COMPLETE.json')]=digest(path/'COMPLETE.json')
for p,e in all_sources().items():assert digest(p,'md5')==e['md5'],p
out.mkdir()
write(out/'MODELS_FROZEN.json',dict(variants=variants,cells=75,seals=seals,L1_not_evaluated=True,release='Completed B0/D4 L2 only. No Stage0 conclusion or new model selection.',source=metadata(),report_code_sha256=digest(__file__)))
rows=[];truths={};base={}
for sh,se,ds in CELLS:
 if ds not in truths:truths[ds]=graph(ds).y.reshape(-1).long()
 ids=torch.tensor(evaluation_ids(se,ds));truth=truths[ds]
 for v in variants:
  row=dict(key=key(sh,se,ds),dataset=ds,shot=sh,seed=se,variant=v,**accuracy(folder(sh,se,ds)/v,ids,truth));rows.append(row)
  if v=='b0':base[row['key']]=row
summary=[]
for sh in SHOTS:
 for ds in DS:
  for v in variants:
   a=[r['accuracy'] for r in rows if (r['shot'],r['dataset'],r['variant'])==(sh,ds,v)]
   summary.append(dict(dataset=ds,shot=sh,variant=v,seeds=5,accuracy_mean=statistics.mean(a),accuracy_std_ddof0=statistics.pstdev(a)))
comparison={v:compare([r for r in rows if r['variant']==v],base,'accuracy') for v in variants[1:]}
write(out/'RESULTS.json',dict(scope='B0/D4 final accuracy only; not a Stage0 acceptance decision',seeds=list(DEV),evaluation='own-seed fixed eval1000 shared across shots',per_cell=rows,summary=summary,paired_comparisons=comparison,L1_metric_issue_untouched=True))
for filename,data in [('per_cell.csv',rows),('summary.csv',summary)]:
 with (out/filename).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
lines=['# Completed B0 / D4 final-accuracy results','', 'Five split seeds42–46; each cell evaluates the fixed1000. L1 and Stage0 gates are not assessed.','', '| Dataset | Shot | FairRouter B0 | Confidence router | Confidence preference |','|---|---:|---:|---:|---:|']
for sh in SHOTS:
 for ds in DS:
  vals=[]
  for v in variants:
   r=next(x for x in summary if (x['shot'],x['dataset'],x['variant'])==(sh,ds,v));vals.append(f"{100*r['accuracy_mean']:.2f} ± {100*r['accuracy_std_ddof0']:.2f}")
  lines.append('| '+ds+' | '+str(sh)+' | '+' | '.join(vals)+' |')
for v,r in comparison.items():lines+=['',f"{v}: paired accuracy delta={r['mean']:+.4f}pp,95%CI=[{r['lower']:.4f},{r['upper']:.4f}],n={r['n']}"]
(out/'REPORT.md').write_text('\n'.join(lines)+'\n');seal(out,dict(all_checks_pass=True,scope='completed B0/D4 L2 only',L1_evaluated=False,Stage0_complete=False))
print('\n'.join(lines),flush=True)
