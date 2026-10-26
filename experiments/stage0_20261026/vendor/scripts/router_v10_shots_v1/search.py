"""One dataset/seed search; consumes safe gold and validation only."""
from common import *
from policy import folds,crossfit_head,outfold_metric,key,crossfit_pair,fit_route,affine,select_acceptance,improves
from catalog import initial,refine
import gc

def head_metric(z,vb,h,fold,fraction):
 ids=np.asarray(vb['ids']);state=np.asarray(vb['target'] if h=='agreement' else vb['state'])
 eligible=state!=2 if h=='preference' else np.ones(len(state),bool)
 y=state.astype(bool) if h=='agreement' else state!=2 if h=='trust' else state==0
 oof=crossfit_head(z,y,ids,fold,eligible);p=oof['score']
 score=np.maximum(p,1-p) if h=='preference' else p
 correct=((p>=.5)==y) if h=='preference' else y
 metric=outfold_metric(score[eligible],correct[eligible],ids[eligible],fold[eligible],fraction)
 pp=np.clip(p[eligible],1e-6,1-1e-6);yy=y[eligible]
 metric['bce']=float(-np.mean(yy*np.log(pp)+(~yy)*np.log1p(-pp))) if len(pp) else None
 return metric

def ranking(row):
 m=row['metric'];return (-key(m)[0],-key(m)[1],m['bce'] if m.get('bce') is not None else 1e9,row['id']!='incumbent',row['id'])

def fit_head(seed,d,h,banks,vb,fold,fraction,old):
 branch='agreement' if h=='agreement' else 'disagreement';dest=RUN/f'seed{seed}/search'/d/h;rows=[];seen=set()
 def record(cid,w,z,stage):
  metric=head_metric(z,vb,h,fold,fraction);p=dest/(cid+'.joblib');dump(p,w);save(dest/(cid+'.pt'),dict(raw=torch.as_tensor(z),validation_ids=vb['ids']))
  row=dict(id=cid,stage=stage,config=w['model']['config'],engine=w['engine'],model_path=str(p),model_sha256=sha(p),raw_path=str(dest/(cid+'.pt')),raw_sha256=sha(dest/(cid+'.pt')),metric=metric)
  write(dest/(cid+'.json'),row);rows.append(row);return row
 if h in old['models']:
  w=copy.deepcopy(old['models'][h]);record('incumbent',w,raw(w,vb),'fresh_refit_incumbent')
 def run_configs(configs,stage):
  skipped=[]
  for i,cfg in enumerate(configs):
   if cfg['id'] in seen:continue
   seen.add(cfg['id']);w=dict(engine='tune',model=dict(config=cfg))
   if not eligible(w,h,banks,d):skipped.append(dict(config=cfg,reason='no_eligible_gold_rows_in_allowed_scope'));continue
   model=engine('tune').fit_candidate(cfg,h,banks,d);w=dict(engine='tune',model=model);z=raw(w,vb)
   record(cfg['id'],w,z,stage)
   if (i+1)%5==0:print('CANDIDATES',seed,d,h,stage,i+1,'/',len(configs),flush=True)
   del model,w;gc.collect()
  return skipped
 first=initial(h);write(dest/'INITIAL_CATALOG.json',first);skip1=run_configs(first,'initial')
 if not rows:
  write(dest/'RESULT.json',dict(rows=[],shortlist=[],unavailable=skip1,query_access=False));return [],[]
 leader=sorted(rows,key=ranking)[0];w=joblib.load(leader['model_path']);second=refine(w,h)
 write(dest/'REFINEMENT_CATALOG.json',dict(leader=leader['id'],configs=second,selection_data='validation-only OOF head score'))
 skip2=run_configs(second,'refinement')
 shortlist=sorted(rows,key=ranking)[:3]
 if any(r['id']=='incumbent' for r in rows) and not any(r['id']=='incumbent' for r in shortlist):shortlist.append(next(r for r in rows if r['id']=='incumbent'))
 write(dest/'RESULT.json',dict(rows=rows,shortlist=[r['id'] for r in shortlist],unavailable=skip1+skip2,query_access=False))
 return shortlist,rows

def audit_oof(oof,ids,fold):
 for a in oof['audit']:
  assert set(a['fit_ids']).isdisjoint(a['held_ids'])
  assert set(a['held_ids'])==set(ids[fold==a['fold']].tolist())
  assert set(a['fit_ids'])<=set(ids[fold!=a['fold']].tolist())

def refit_incumbent(seed,d,branch,banks):
 """Transfer recipe only, refit all weights/statistics on this shot/seed."""
 from policy import folds,crossfit_head,crossfit_pair,select_acceptance,affine,fit_route
 choice=source_choice(seed,d,branch);path=checked(choice['model_source'])
 assert sha(path)==choice['model_sha256']
 original=joblib.load(path);vb=eval_bank(seed,d,branch,'validation')
 ids=np.asarray(vb['ids']);fold=folds(ids,d);fraction=float(original['policy']['fraction'])
 models={};audits={}
 for h,w in original['models'].items():
  if not eligible(w,h,banks,d):continue
  fitted=fit(w,h,banks,d)
  if fitted['engine']=='v6' and h!='agreement':
   a,oof=norm(fitted,h,banks,d);fitted['gold_normalization']=a
   save(RUN/f'seed{seed}/incumbents'/d/branch/(h+'_gold_oof.pt'),tensor_tree(oof))
  models[h]=fitted
  audits[h]=fitted['model'].get('fit_audit',fitted['model'].get('training_audit'))
 bundle=dict(branch=branch,dataset=d,models=models,tuning_schema='crossfit_v1',split_seed=seed,shots=SHOT,version='Router-v10-MS-Original-shot-refit')
 complete=len(models)==len(original['models']) and len(ids)>0
 if complete:
  z={h:raw(w,vb) for h,w in models.items()}
  if branch=='agreement':
   y=np.asarray(vb['target'],bool);oof=crossfit_head(z['agreement'],y,ids,fold)
   bundle['calibration']=dict(agreement=affine(z['agreement'],y));correct=y
  else:
   state=np.asarray(vb['state']);oof=crossfit_pair(z,state,ids,fold,fraction)
   cal,recipe,trials=fit_route(z,state,ids,fraction);bundle.update(calibration=cal,route_recipe=recipe)
   correct=oof['route']==state
  bundle['policy']=select_acceptance(oof['score'],correct,ids,fold)
 else:
  bundle.update(unavailable=True,policy=dict(fraction=0.,reject_all=True,reason='incumbent_scope_unavailable'))
 # Fixed-K comparisons retain source reference only if fresh baseline rejects all.
 if bundle['policy']['fraction']==0:bundle['comparison_fraction']=fraction
 dest=RUN/f'seed{seed}/incumbents'/d/branch
 dump(dest/'bundle.joblib',bundle)
 write(dest/'REFIT.json',dict(source_model=str(path),source_sha256=sha(path),recipe_only=True,all_weights_refit=True,shot=SHOT,seed=seed,fit_audits=audits,query_access=False))
 return bundle

def select_branch(seed,d,branch,banks):
 root=RUN/f'seed{seed}'
 vb=eval_bank(seed,d,branch,'validation');ids=np.asarray(vb['ids']);fold=folds(ids,d)
 dest=root/'models'/d/branch;save(dest/'validation_fold_ids.pt',dict(ids=vb['ids'],fold=torch.as_tensor(fold)))
 if not len(ids):
  bundle=dict(branch=branch,dataset=d,models={},unavailable=True,policy=dict(fraction=0.,reject_all=True,reason='empty_validation_population'))
  dump(dest/'bundle.joblib',bundle);dump(dest/'fixed_bundle.joblib',bundle)
  write(dest/'SELECTION.json',dict(unavailable=True,policy=bundle['policy'],query_fit_rows=0));return
 old=refit_incumbent(seed,d,branch,banks);fraction=old.get('comparison_fraction',old['policy']['fraction'])
 heads=['agreement'] if branch=='agreement' else ['trust','preference'];lists={};allrows={}
 for h in heads:lists[h],allrows[h]=fit_head(seed,d,h,banks,vb,fold,fraction,old)
 if not len(ids) or any(not lists[h] for h in heads):
  bundle=dict(branch=branch,dataset=d,models={},unavailable=True,policy=dict(fraction=0.,reject_all=True,reason='no_eligible_model_or_validation_rows'))
  dump(dest/'bundle.joblib',bundle);dump(dest/'fixed_bundle.joblib',bundle)
  write(dest/'SELECTION.json',dict(unavailable=True,policy=bundle['policy'],query_fit_rows=0));return
 rows=[];payloads={}
 if branch=='agreement':
  # Head shortlist already evaluated on all validation folds with gold-only fitted models.
  for r in lists['agreement']:
   z=load(r['raw_path'])['raw'].numpy();oof=crossfit_head(z,np.asarray(vb['target'],bool),ids,fold);audit_oof(oof,ids,fold)
   m=outfold_metric(oof['score'],np.asarray(vb['target'],bool),ids,fold,fraction)
   row=dict(id=r['id'],heads=dict(agreement=r['id']),metric=m);rows.append(row);payloads[r['id']]=dict(oof=oof,raw=z)
 else:
  import itertools
  for t,p in itertools.product(lists['trust'],lists['preference']):
   z={h:load(r['raw_path'])['raw'].numpy() for h,r in (('trust',t),('preference',p))}
   oof=crossfit_pair(z,np.asarray(vb['state']),ids,fold,fraction);audit_oof(oof,ids,fold)
   cid=t['id']+'|'+p['id'];m=outfold_metric(oof['score'],oof['route']==np.asarray(vb['state']),ids,fold,fraction)
   rows.append(dict(id=cid,heads=dict(trust=t['id'],preference=p['id']),metric=m));payloads[cid]=dict(oof=oof,raw=z)
   print('PAIR',seed,d,cid,'OOF',m['precision'],flush=True)
 baseid='incumbent' if branch=='agreement' else 'incumbent|incumbent'
 baseline=next((r for r in rows if r['id']==baseid),None)
 winner=sorted(rows,key=lambda r:(-key(r['metric'])[0],-key(r['metric'])[1],-key(r['metric'])[2],r['id']!=baseid,r['id']))[0]
 adopt=baseline is None or (winner['id']!=baseid and improves(winner['metric'],baseline['metric']))
 selected=winner if adopt else baseline;chosen=payloads[selected['id']];oof=chosen['oof'];z=chosen['raw']
 if not adopt:
  bundle=copy.deepcopy(old);fixed=copy.deepcopy(old);policy_source='fresh_refit_validation_calibrated_incumbent_retained'
  fixed['policy']=dict(fraction=fraction,reject_all=fraction==0.,reason='fixed_comparison_fraction_diagnostic_only')
 else:
  models={h:joblib.load(next(r for r in allrows[h] if r['id']==cid)['model_path']) for h,cid in selected['heads'].items()}
  bundle=dict(branch=branch,dataset=d,models=models,tuning_schema='crossfit_v1',version='Router-v10-multiseed-tuned',split_seed=seed)
  if branch=='agreement':
   bundle['calibration']=dict(agreement=affine(z,np.asarray(vb['target'],bool)));correct=np.asarray(vb['target'],bool)
  else:
   cal,recipe,route_trials=fit_route(z,np.asarray(vb['state']),ids,fraction);bundle.update(calibration=cal,route_recipe=recipe)
   write(dest/'FINAL_ROUTE_TRIALS.json',route_trials);correct=oof['route']==np.asarray(vb['state'])
  bundle['policy']=select_acceptance(oof['score'],correct,ids,fold)
  fixed=copy.deepcopy(bundle);fixed['policy']=dict(fraction=fraction,reject_all=fraction==0.,reason='fixed_baseline_coverage_comparison')
  policy_source='OOF-selected fraction; full-validation calibration after selection'
 dump(dest/'bundle.joblib',bundle);dump(dest/'fixed_bundle.joblib',fixed)
 save(dest/'SELECTED_OOF.pt',tensor_tree(oof));save(dest/'BASELINE_OOF.pt',tensor_tree(payloads[baseid]['oof'] if baseline else {}))
 write(dest/'SELECTION.json',dict(seed=seed,dataset=d,branch=branch,adopted=adopt,selected=selected,best_candidate=winner,baseline=baseline,pair_trials=rows,policy=bundle['policy'],policy_source=policy_source,baseline_fraction=fraction,validation_router_weight_fit_rows=0,calibration_uses_validation=True,OOF_is_selection_evidence_not_unbiased_bound=True,query_fit_rows=0,slurm_job_id=os.environ['SLURM_JOB_ID']))
 a=predict_bundle(bundle,vb);b=predict_bundle(joblib.load(dest/'bundle.joblib'),vb)
 assert exact(a,b);print('BRANCH SELECTED',seed,d,branch,'adopted',adopt,'K',bundle['policy']['fraction'],flush=True)

def main(seed,d):
 require();torch.set_num_threads(1);assert_open(seed);check_code();assert seed in SEEDS and d in DATASETS
 for branch in BRANCHES:
  banks=training(seed,branch);select_branch(seed,d,branch,banks);del banks;gc.collect()
 write(RUN/f'seed{seed}/search'/d/'COMPLETE.json',dict(all_checks_pass=True,query_access=False,slurm_job_id=os.environ['SLURM_JOB_ID']))
if __name__=='__main__':main(int(sys.argv[1]),sys.argv[2])
