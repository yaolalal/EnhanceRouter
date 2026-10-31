"""Independently recalculate published CSV counts/accuracies; no training."""
from pathlib import Path
import csv,hashlib,json,os,math,collections,datetime
ROOT=Path(__file__).resolve().parents[2]
PUB=ROOT/'reports/synthetic_pseudo_20261031'
RUN=ROOT/'runs/synthetic_pseudo_20261031-v1'
OUT=ROOT/'reports/synthetic_pseudo_audit_20261031'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def close(a,b):
 if a is None or b is None:assert a is b,(a,b)
 else:assert abs(float(a)-float(b))<1e-12,(a,b)
def fraction(a):return sum(a)/len(a) if a else None
def main():
 assert os.environ.get('SLURM_JOB_ID')
 assert not OUT.exists()
 complete=read(PUB/'COMPLETE.json');assert complete['all_checks_pass'] and complete['models']==1850
 assert sha(PUB/'PUBLICATION_SHA256.json')==complete['manifest_sha256']
 manifest=read(PUB/'PUBLICATION_SHA256.json')
 for name,h in manifest.items():assert sha(PUB/name)==h,name
 results=read(PUB/'RESULTS.json');runs=results['per_run'];assert len(runs)==1850
 assert len({r['run_id'] for r in runs})==1850
 base={(r['dataset'],r['seed']):r for r in read(RUN/'REGISTRATION.json')['B0_rows']}
 seen={};counts=collections.Counter();evidence=[]
 for r in runs:
  ds,se=r['dataset'],r['split_seed'];src=PUB/'evidence'/f'{r["index"]:02d}'/r['run_id'].split('/')[1]
  assert read(src/'RESULTS.json')==r
  cfg=read(src/'CONFIG.json');expected=dict(base[ds,se]['config']);expected['keep']=0.
  expected['cross']=0. if r['rule']=='A0' else (expected['cross'] or 1.)
  if r['selection_mode']=='last500':expected['max_steps']=500
  assert cfg['config']==expected==r['config']
  assert r['effective_rng_seed']==se+1000003*r['training_seed']
  assert r['selected_step0']==(r['selected_step']==0)
  if r['selection_mode']=='last500':assert r['selected_step']==r['stop_step']==500
  p=rows(src/'pool.csv');e=rows(src/'eval1000.csv');assert len(p)==r['N'] and len(e)==1000
  split=read(ROOT/f'runs/stage0-20261026-v2/splits/5/{se}/{ds}.json')
  ids=[int(x['node_id']) for x in e];assert ids==split['evaluation_1000_ids']
  assert [int(x['eval_order']) for x in e]==list(range(1000))
  protected=set(ids)|set(split['support_ids'])|set(split['valid_ids'])
  selected=[int(x['node_id']) for x in p];assert len(set(selected))==len(selected)
  assert not set(selected)&protected and set(selected)<=set(split['unlabeled_ids'])
  key=(ds,se)
  if r['rule']!='A0':
   if key in seen:assert seen[key]==selected
   else:seen[key]=selected
  correct=[]
  for x in p:
   g,l,t,z=map(int,(x['gnn_pred'],x['llm_pred'],x['truth'],x['pseudo_label']))
   assert g!=l and ((g==t) != (l==t))
   assert z==(g if x['chosen_expert']=='GNN' else l)
   assert x['pool']==('teach_g' if x['chosen_expert']=='GNN' else 'teach_l')
   assert x['correct']==str(z==t)
   correct.append(z==t)
  k=len(p) if r['rule']=='A100' else 0 if r['rule']=='A0' else round(.8*len(p))
  assert sum(correct)==r['K']==k
  close(fraction(correct),r['actual_precision'])
  for name in ('teach_g','teach_l'):
   group=[int(x['pseudo_label'])==int(x['truth']) for x in p if x['pool']==name]
   assert len(group)==r[name+'_count'];close(fraction(group),r[name+'_precision'])
  if r['rule'] in ('R2','R3','R4','R5','R7','R8','R9','R10'):
   field={'R2':'degree','R3':'degree','R4':'margin','R5':'margin','R7':'pL_confidence','R8':'pL_confidence','R9':'pG_confidence','R10':'pG_confidence'}[r['rule']]
   good=[float(x[field]) for x,c in zip(p,correct) if c];bad=[float(x[field]) for x,c in zip(p,correct) if not c]
   if good and bad:
    if r['rule'] in ('R2','R4','R7','R9'):assert min(good)>=max(bad)
    else:assert max(good)<=min(bad)
  if r['rule']=='R6':
   info=r['pool_summary']['R6'];assert info['actual_wrong']==len(p)-k
   full=set(info['fully_wrong_classes']);partial=info['partially_wrong_class']
   for x,c in zip(p,correct):
    if int(x['truth']) in full:assert not c
    elif int(x['truth'])!=partial:assert c
  truth=[int(x['truth']) for x in e]
  for col,metric in [('pJ','full_accuracy'),('pR','pR_accuracy'),('pGL_own','pGL_own_accuracy'),('pGL_B0','pGL_B0_accuracy')]:close(fraction([int(x[col])==t for x,t in zip(e,truth)]),r[metric])
  subset=[x for x in e if x['pR']!=x['pGL_B0']]
  assert len(subset)==r['S_count'];close(len(subset)/1000,r['S_fraction'])
  for col,metric in [('pJ','S_full_accuracy'),('pR','S_pR_accuracy'),('pGL_B0','S_pGL_accuracy')]:close(fraction([x[col]==x['truth'] for x in subset]),r[metric])
  close(r['full_accuracy']-r['pGL_B0_accuracy'],r['pR_contribution'])
  wrong=[x for x in e if x['pGL_B0']!=x['truth']];right=[x for x in e if x['pGL_B0']==x['truth']]
  assert len(wrong)==r['reference_wrong_count'] and len(right)==r['reference_correct_count']
  close(fraction([x['pR']==x['truth'] for x in wrong]),r['rescue_rate'])
  close(fraction([x['pR']!=x['truth'] for x in right]),r['harm_rate'])
  hist=collections.Counter(int(x['pR']) for x in e);entropy=-sum((n/1000)*math.log(n/1000) for n in hist.values())
  close(entropy,r['prediction_entropy']);assert len(hist)==r['prediction_class_count'];close(max(hist.values())/1000,r['prediction_max_class_fraction'])
  context=read(RUN/'prepared'/ds/str(se)/'CONTEXT.json')
  for part in ('head','tail'):
   classes=set(context[part+'_classes']);group=[x for x in e if int(x['truth']) in classes]
   assert len(group)==r[part+'_eval_count'];close(fraction([x['pR']==x['truth'] for x in group]),r[part+'_pR_accuracy'])
  counts[r['selection_mode'],ds,r['rule']]+=1
  evidence.append(dict(run_id=r['run_id'],pool_csv_sha256=sha(src/'pool.csv'),eval_csv_sha256=sha(src/'eval1000.csv'),all_checks_pass=True))
 assert len(seen)==25
 for (mode,ds,rule),n in counts.items():assert n==(10 if rule.startswith('A') else 30)
 assert len(counts)==75
 OUT.mkdir(parents=True)
 receipt=dict(all_checks_pass=True,models=1850,pool_and_eval_evidence_verified=1850,selected_node_sets=25,metric_tolerance=1e-12,
  source_manifest_sha256=sha(PUB/'PUBLICATION_SHA256.json'),source_complete_sha256=sha(PUB/'COMPLETE.json'),
  script_sha256=sha(Path(__file__)),slurm_job=os.environ['SLURM_JOB_ID'],checked_at=datetime.datetime.now().astimezone().isoformat(),
  checks=['publication hashes','frozen configuration except prescribed changes','seed mapping','fixed500 last selection','original eval1000 ordered ids','protected node exclusion','same selected nodes all rules','expert labels XOR correctness','exact K/N','per-pool counts and precision','rank rule ordering','R6 class errors','all classification and S metrics from exported per-node labels','rescue/harm','prediction entropy','head/tail'],evidence=evidence)
 (OUT/'RESULTS.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
 (OUT/'COMPLETE.json').write_text(json.dumps(dict(all_checks_pass=True,models=1850,results_sha256=sha(OUT/'RESULTS.json'),script_sha256=receipt['script_sha256']),indent=2)+'\n')
 print('SYNTHETIC_INDEPENDENT_AUDIT_PASS',1850,flush=True)
if __name__=='__main__':main()
