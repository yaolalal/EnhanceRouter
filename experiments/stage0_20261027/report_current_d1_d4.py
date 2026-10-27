"""Snapshot completed artifacts; evaluate only budget-independent diagnostics."""
from d3_recovery import *
setup()
import torch, numpy as np, csv, statistics
from datetime import datetime, timezone, timedelta
from report import auc, accuracy

METRICS=('trust_auroc','preference_auroc','D_Prec_01','D_Prec_05','D_Prec_20','D_Prec_100','accuracy','macro_f1')
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out=RUN/'current-results'/stamp
out.mkdir(parents=True)

def sealed(path):
 if not (path/'COMPLETE.json').exists():return None
 try:r=read(path/'COMPLETE.json')
 except ValueError:return None # An active writer may still be finishing its seal.
 assert isinstance(r.get('files'),dict)
 return r

def verify(path,record,names=None):
 for rel,h in record['files'].items():
  if names is None or rel in names:assert digest(path/rel)==h,(path,rel)

def stats(values):
 a=[float(v) for v in values if v is not None]
 return dict(n=len(a),mean=statistics.mean(a) if a else None,std_ddof0=statistics.pstdev(a) if a else None)

variants=['b0',*[f'd1_frozen/{f}' for f in (.25,.5,.75,1.)],*[f'd1_numeric/{f}' for f in (.25,.5,.75,1.)],*[f'd2{a}/{n}' for a in ('a','b') for n in (50,200,1000)],'d4/confidence_router','d4/confidence_preference']
snapshot=[];counts={v:0 for v in variants};by_dataset={v:{d:0 for d in DS} for v in variants}
for sh,se,ds in CELLS:
 for variant in variants:
  path=variant_folder(sh,se,ds,variant.replace('d1_frozen/','d1/'))
  record=sealed(path)
  if record is None:continue
  snapshot.append(dict(key=key(sh,se,ds),shot=sh,seed=se,dataset=ds,variant=variant,path=str(path),seal_sha256=digest(path/'COMPLETE.json')))
  counts[variant]+=1;by_dataset[variant][ds]+=1
write(out/'FROZEN_SNAPSHOT.json',dict(timestamp_shanghai=datetime.now(timezone(timedelta(hours=8))).isoformat(),counts=counts,by_dataset=by_dataset,artifacts=snapshot,test_outcomes_read=False,L1_B0_budget='deferred',report_code_sha256=digest(__file__),source=metadata()))

# The already published final-accuracy reports are immutable inputs here.
published={};finals=[]
for name in ('completed-l2-20261027','completed-d3-l2-20261027'):
 p=RUN/name;r=read(p/'COMPLETE.json');verify(p,r)
 published[name]=digest(p/'COMPLETE.json')
 data=read(p/'RESULTS.json')
 if name=='completed-l2-20261027':finals.extend(data['per_cell'])
 else:
  finals.extend(dict(x,variant='d3') for x in data['per_cell'])
final_index={(r['key'],r['variant']):r for r in finals}

def budget_free(path,up,ids,truth):
 gp=up['gnn_logits'].argmax(1);lp=up['llm_logits'].argmax(1)
 data=load(path/'disagreement-scores.pt');mask=torch.isin(data['ids'],ids);nodes=data['ids'][mask].long()
 expected=ids[gp[ids]!=lp[ids]]
 assert set(nodes.tolist())==set(expected.tolist())
 score=data['score'][mask].numpy();route=data['route'][mask].numpy()
 assert set(np.unique(route))<=set((0,1))
 assert bool(torch.isfinite(data['trust'][mask]).all()) and bool(torch.isfinite(data['preference'][mask]).all())
 order=np.lexsort((nodes.numpy(),-score))
 g=gp[nodes];l=lp[nodes];y=truth[nodes]
 t=(g==y)|(l==y);p=(g==y)
 chosen=np.where(route==0,g.numpy(),l.numpy());correct=chosen==y.numpy()
 result=dict(eval_count=len(ids),disagreement_count=len(nodes),trust_positive_count=int(t.sum()),trust_auroc=auc(t.numpy(),data['trust'][mask].numpy()),preference_auroc=auc(p[t].numpy(),data['preference'][mask].numpy()[t.numpy()]))
 for pct in (1,5,20,100):
  n=min(len(nodes),int(np.ceil(pct*len(nodes)/100)))
  result[f'D_Prec_{pct:02d}' if pct<100 else 'D_Prec_100']=float(correct[order[:n]].mean()) if n else None
 return result

rows=[];truths={};grouped={}
for r in snapshot:grouped.setdefault((r['shot'],r['seed'],r['dataset']),[]).append(r)
for (sh,se,ds),items in grouped.items():
 if ds not in truths:truths[ds]=graph(ds).y.reshape(-1).long()
 truth=truths[ds];ids=torch.tensor(evaluation_ids(se,ds));assert len(ids)==1000
 up=upstream_bank(sh,se,ds)
 for item in items:
  path=Path(item['path']);record=read(path/'COMPLETE.json')
  assert digest(path/'COMPLETE.json')==item['seal_sha256']
  verify(path,record,{'disagreement-scores.pt','agreement-scores.pt','config.json','diagnostics.json','controlled-audit.json','predictions.pt','predictions-reference.json'})
  row=dict(item,**budget_free(path,up,ids,truth))
  variant=item['variant']
  if variant.startswith('d1_numeric/'):
   cfg=read(path/'config.json');audit=read(path/'controlled-audit.json')
   assert audit['all_checks_pass'] and cfg['new_code_sha256']==digest(Path(__file__).with_name('d1_numeric.py'))
   row.update(accuracy(path,ids,truth))
  elif (item['key'],variant) in final_index:
   prior=final_index[(item['key'],variant)]
   row.update({m:prior[m] for m in ('accuracy','macro_f1')})
  if variant.startswith('d2'):
   d=read(path/'diagnostics.json')
   assert d['oracle'] and d['fixed_eval_overlap']==0
   row.update({k:d[k] for k in ('requested_N','actual_N','extra_T_rows','extra_P_rows')})
  rows.append(row)
 print('SNAPSHOT_EVALUATED',sh,se,ds,len(items),flush=True)
 del up

baseline={r['key']:r for r in rows if r['variant']=='b0'}
assert len(baseline)==75
for r in rows:
 if r['variant']=='d1_frozen/1.0':
  for m in METRICS[:6]:
   a=r.get(m);b=baseline[r['key']].get(m)
   assert (a is None and b is None) or (a is not None and b is not None and abs(a-b)<1e-12),(r['key'],m,a,b)

summaries=[]
for v in variants:
 for ds in ('ALL',)+DS:
  group=[r for r in rows if r['variant']==v and (ds=='ALL' or r['dataset']==ds)]
  entry=dict(variant=v,dataset=ds,cells=len(group),metrics={m:stats([r.get(m) for r in group]) for m in METRICS},matched_B0={m:stats([baseline[r['key']].get(m) for r in group]) for m in METRICS})
  entry['paired_delta_pp']={m:stats([100*(r[m]-baseline[r['key']][m]) for r in group if r.get(m) is not None and baseline[r['key']].get(m) is not None]) for m in METRICS}
  if v.startswith('d2'):
   a=[r['actual_N'] for r in group];entry['actual_N']=dict(min=min(a) if a else None,max=max(a) if a else None,mean=statistics.mean(a) if a else None)
  summaries.append(entry)

# Use exactly the same completed cells at all four D1 fractions for its partial curve.
numeric_keys=set.intersection(*[{r['key'] for r in rows if r['variant']==f'd1_numeric/{f}'} for f in (.25,.5,.75,1.)])
numeric=[]
for ds in ('ALL',)+DS:
 for f in (.25,.5,.75,1.):
  group=[r for r in rows if r['variant']==f'd1_numeric/{f}' and r['key'] in numeric_keys and (ds=='ALL' or r['dataset']==ds)]
  numeric.append(dict(dataset=ds,fraction=f,cells=len(group),keys=[r['key'] for r in group],metrics={m:stats([r.get(m) for r in group]) for m in METRICS}))
write(out/'RESULTS.json',dict(scope='Completed artifacts at a fixed snapshot; budget-independent descriptive metrics only',snapshot=read(out/'FROZEN_SNAPSHOT.json'),per_cell=rows,summary=summaries,numeric_complete_cohort=numeric,completed_L2=finals,published_report_seals=published,primary_D_Prec_at_B0=None,A_Prec_at_B0=None,L1_budget='deferred at user request',acceptance_assessed=False,Stage0_complete=False,report_code_sha256=digest(__file__),source=metadata()))
for filename,data in [('per_cell.csv',rows),('completion_by_variant.csv',[dict(variant=v,cells=counts[v],**by_dataset[v]) for v in variants])]:
 fields=sorted(set().union(*(r.keys() for r in data)))
 with (out/filename).open('x',newline='') as fp:
  writer=csv.DictWriter(fp,fieldnames=fields);writer.writeheader();writer.writerows(data)

def fmt(v,auc_value=False):return '—' if v is None else f'{v:.3f}' if auc_value else f'{100*v:.2f}'
def line_metrics(row):return [fmt(row['metrics'][m]['mean'],m.endswith('auroc')) for m in METRICS[:6]]
lines=['# D1–D4 current results','', 'Snapshot (Asia/Shanghai): '+read(out/'FROZEN_SNAPSHOT.json')['timestamp_shanghai'],'','AUROC uses scale0–1; precision and final accuracy use %. Means average cells, not pooled nodes. Undefined AUROC cells are excluded and their defined counts are retained in RESULTS.json. D-Prec@B0/A-Prec@B0 remain deferred; fixed-coverage precision here does not select that budget. No acceptance or Stage1 decision.','','## Completion','','| Variant | Complete cells /75 | Cora | CiteSeer | PubMed | Arxiv | Products |','|---|---:|---:|---:|---:|---:|---:|']
for v in variants:lines.append('| '+v+' | '+str(counts[v])+' | '+' | '.join(str(by_dataset[v][d]) for d in DS)+' |')
lines+=['','## Old fully frozen D1: all75 cells per fraction','','| Dataset | Fraction | n | Trust AUROC | Preference AUROC | P@1% | P@5% | P@20% | P@100% |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in summaries:
 if r['variant'].startswith('d1_frozen/'):
  lines.append('| '+r['dataset']+' | '+r['variant'].split('/')[1]+' | '+str(r['cells'])+' | '+' | '.join(line_metrics(r))+' |')
lines+=['','## New numeric D1: partial, common completed cohort only','','Only cells with all four completed fractions enter these curves; partial individual outputs remain in per_cell.csv. These are not the complete75-cell results.','','| Dataset | Fraction | n | Trust AUROC | Preference AUROC | P@1% | P@5% | P@20% | P@100% | Accuracy |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in numeric:lines.append('| '+r['dataset']+' | '+str(r['fraction'])+' | '+str(r['cells'])+' | '+' | '.join(line_metrics(r)+[fmt(r['metrics']['accuracy']['mean'])])+' |')
lines+=['','## D2: completed cells only','','Actual N is the number of available added parent nodes; requested N is an upper bound. Overall comparisons use the matched B0 cells recorded below; completion counts show whether all75 cells are available.','','| Dataset | Variant | n | Actual N range | Trust AUROC | Preference AUROC | P@1% | P@5% | P@20% | P@100% |','|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
for r in summaries:
 if r['variant'].startswith('d2'):
  if r['variant']=='d2a/50':lines.append('| '+r['dataset']+' | matched B0 | '+str(r['cells'])+' | — | '+' | '.join(line_metrics(dict(metrics=r['matched_B0'])))+' |')
  a=r['actual_N'];span='—' if not r['cells'] else str(a['min'])+'–'+str(a['max']);lines.append('| '+r['dataset']+' | '+r['variant']+' | '+str(r['cells'])+' | '+span+' | '+' | '.join(line_metrics(r))+' |')
lines+=['','## B0 and D4: budget-independent router metrics','','| Dataset | Variant | n | Trust AUROC | Preference AUROC | P@1% | P@5% | P@20% | P@100% |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in summaries:
 if r['variant']=='b0' or r['variant'].startswith('d4/'):
  lines.append('| '+r['dataset']+' | '+r['variant']+' | '+str(r['cells'])+' | '+' | '.join(line_metrics(r))+' |')
lines+=['','## Completed D3 / D4 final accuracy','','| Dataset | Shot | B0 | D3 | Confidence router | Confidence preference |','|---|---:|---:|---:|---:|---:|']
for sh in SHOTS:
 for ds in DS:
  vals=[]
  for v in ('b0','d3','d4/confidence_router','d4/confidence_preference'):
   a=[r['accuracy'] for r in finals if (r['shot'],r['dataset'],r['variant'])==(sh,ds,v)];assert len(a)==5
   vals.append(f'{100*statistics.mean(a):.2f} ± {100*statistics.pstdev(a):.2f}')
  lines.append('| '+ds+' | '+str(sh)+' | '+' | '.join(vals)+' |')
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
seal(out,dict(all_checks_pass=True,scope='budget-independent current-result snapshot',test_outcomes_used_for_selection=False,L1_B0_budget_evaluated=False,Stage0_complete=False))
print('CURRENT_RESULTS_COMPLETE',out,counts,'numeric_common_cells',len(numeric_keys),flush=True)
