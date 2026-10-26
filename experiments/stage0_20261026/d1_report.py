"""Report preregistered D1 after all 75 B0/D1 cells are sealed; no D2 dependency."""
from common import *
from report import level1,compare

def main():
 setup();import torch,csv,joblib
 registration=metric_budget_registration()
 out=RUN/'d1-report'
 assert not out.exists(),out
 variants=['b0',*[f'd1/{f}' for f in [.25,.5,.75,1.]]]
 seals={}
 for shot,seed,ds in CELLS:
  for variant in variants:
   p=variant_folder(shot,seed,ds,variant)/'COMPLETE.json'
   assert p.is_file(),p
   # Authenticate each sealed artifact before exposing any evaluation outcomes.
   data=read(p)
   for rel,expected in data['files'].items():
    assert digest(p.parent/rel)==expected,(p,rel)
   seals[str(p.relative_to(RUN))]=digest(p)
 checked=all_sources() | read(RUN/'SOURCE_EXTENSION_FILES_V2.json') | read(HERE/'source-code.json')
 for receipt in (RUN/'shared/text').glob('*.json'):
  checked.update(read(receipt).get('source_extra',{}))
 for p,entry in checked.items():assert digest(p,'md5')==entry['md5'],p
 out.mkdir()
 write(out/'MODELS_FROZEN.json',dict(cells=75,variants=variants,seals=seals,held_out_evaluations=0,source=metadata(),
  evaluation_release='D1 only; all D1/B0 cells sealed. D2/D3/D4 recipes remain preregistered and cannot change in response to these outcomes.'))
 rows=[];truths={};baseline={}
 for shot,seed,ds in CELLS:
  if ds not in truths:truths[ds]=graph(ds).y.reshape(-1).long()
  ids=torch.tensor(evaluation_ids(seed,ds));base=folder(shot,seed,ds)/'b0';budget={}
  for branch in ['agreement','disagreement']:
   b=load(base/f'{branch}-scores.pt');budget[branch]=level1_budget(b,ids,joblib.load(base/'models.joblib')[branch]['policy']['fraction'],registration['mode'])['primary']
  for variant in variants:
   row=dict(key=key(shot,seed,ds),dataset=ds,shot=shot,seed=seed,variant=variant)
   row.update(level1(variant_folder(shot,seed,ds,variant),shot,seed,ds,truths[ds],budget));rows.append(row)
   if variant=='b0':baseline[row['key']]=row
 comparisons={v:{m:compare([r for r in rows if r['variant']==v],baseline,m) for m in ['D-Prec@B0','A-Prec@B0','trust_auroc','preference_auroc']} for v in variants[1:]}
 summary=[]
 for ds in DS:
  for v in variants:
   group=[r for r in rows if r['dataset']==ds and r['variant']==v]
   row=dict(dataset=ds,variant=v,cells=len(group))
   for m in ['D-Prec@B0','A-Prec@B0','trust_auroc','preference_auroc','D-Prec@0.01','D-Prec@0.05','D-Prec@0.2','D-Prec@1.0']:
    values=[r[m] for r in group if r[m] is not None];row[m]=sum(values)/len(values) if values else None;row[m+'_defined']=len(values)
   summary.append(row)
 write(out/'RESULTS.json',dict(dev_cells=75,held_out_evaluations=0,scope='D1 router supervision only; not end-to-end label-budget accuracy',
  protocol=registered_protocol(),metric_budget=registration,amendments=[read(RUN/'EXACT_FIVE_SHOT_REGISTRATION.json')],
  dev_per_cell=rows,summary=summary,paired_comparisons=comparisons,source=metadata()))
 for name,data in [('per_cell.csv',rows),('summary.csv',summary)]:
  with (out/name).open('x',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=sorted({k for r in data for k in r}));writer.writeheader();writer.writerows(data)
 lines=['D1: router training sample learning curve','75 paired cells; seeds42-46 only, no reserved seeds.',
  'All expert outputs, prototypes, source data, selected recipes, calibration and acceptance fractions stay fixed.',
  'Primary measure: D-Prec at the user-registered B0 budget on each seed fixed evaluation1000.','']
 for v in variants[1:]:
  r=comparisons[v]['D-Prec@B0']
  if r['mean'] is None:lines.append(f'{v}: no defined paired values');continue
  lines.append(f"{v}: delta vs100%={r['mean']:+.4f}pp;95%CI=[{r['lower']:.4f},{r['upper']:.4f}];n={r['n']}")
  lines.append(', '.join(f"{d}: {a['mean']:+.4f}pp" for d,a in r['per_dataset'].items()))
 (out/'REPORT.txt').write_text('\n'.join(lines)+'\n')
 write(out/'SOURCE_MD5_RECHECK.json',dict(unchanged=True,checked_files=len(checked),files={p:e['md5'] for p,e in checked.items()}))
 seal(out,dict(all_checks_pass=True,dev_cells=75,held_out_evaluations=0,scope='D1 only; Phase0 not declared complete'))
 print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
