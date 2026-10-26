from common import *
import argparse

def fitted_models(shot,seed,ds,small=False):
 import joblib,numpy as np
 from enhancerouter.router import engine,predict_bundle
 idx=read(ROOT/'configs/inputs.json');template=idx['cells'][f'{shot}/{seed if seed<=44 else 42}/{ds}'];models={}
 for branch in ['agreement','disagreement']:
  old=joblib.load(template['branches'][branch]['bundle']);new=copy.deepcopy(old)
  training={d:bank(shot,seed,d,branch,'train',small) for d in DS}
  for h,w in new.get('models',{}).items():
   w['model']=engine(w['engine']).fit_candidate(copy.deepcopy(w['model']['config']),h,training,ds)
  models[branch]=new
 return models

def calibrated(models,shot,seed,ds,small=False):
 import numpy as np
 from enhancerouter.router import raw,engine,predict
 from enhancerouter import policy
 from eargtc.router_aug_search.common import fit_affine_grouped
 from native_calibration import calibrate
 anchor=read(SOURCE/'runs/router_v10_multiseed_20260912_v1/ANCHOR.json')
 for branch,m in models.items():
  b=bank(shot,seed,ds,branch,'validation',small);ids=np.asarray(b['ids'])
  training={d:bank(shot,seed,d,branch,'train',small) for d in DS};norms={}
  if branch=='disagreement':
   for h,w in m['models'].items():
    if w['engine']=='v6':
     def fit(bs):return dict(engine=w['engine'],model=engine(w['engine']).fit_candidate(copy.deepcopy(w['model']['config']),h,bs,ds))
     r=fit_affine_grouped(training,ds,h,fit,predict);norms[h]={k:float(r['affine'][k]) for k in ('slope','bias')}
     if shot!=5:w['gold_normalization']=copy.deepcopy(norms[h])
    else:
     model=w['model'];norms[h]=dict(slope=1/model['scale'],bias=-model['center']/model['scale'])
  reference_fraction=float(m['policy']['fraction'])
  if shot==5 and m.get('tuning_schema')!='crossfit_v1':
   m.pop('tuning_schema',None)
   m,selection,_=calibrate(ds,branch,m,b,norms,anchor[ds][branch],anchor[ds][branch]['reference_fraction'])
   m['native_selection_audit']=selection
  else:
   fold=policy.folds(ids,ds) if len(ids) else np.array([],dtype=int)
   if not len(ids):
    m.update(unavailable=True,policy=dict(fraction=0.,reject_all=True,reason='empty_validation_population'));continue
   z={h:raw(w,b) for h,w in m['models'].items()}
   if branch=='agreement':
    y=np.asarray(b['target']);oof=policy.crossfit_head(z['agreement'],y,ids,fold)
    m.update(tuning_schema='crossfit_v1',calibration=dict(agreement=policy.affine(z['agreement'],y)))
    chosen=policy.select_acceptance(oof['score'],y,ids,fold)
   else:
    state=np.asarray(b['state']);oof=policy.crossfit_pair(z,state,ids,fold,reference_fraction)
    cal,recipe,_=policy.fit_route(z,state,ids,reference_fraction)
    m.update(tuning_schema='crossfit_v1',calibration=cal,route_recipe=recipe)
    chosen=policy.select_acceptance(oof['score'],oof['route']==state,ids,fold)
   m['policy']=chosen;m['inherited_reference_fraction']=reference_fraction
  models[branch]=m
 return models


def route(models,shot,seed,ds,small=False,variant='b0',reference=None):
 import numpy as np,torch
 from enhancerouter.router import predict_bundle,top_mask
 mode='small_stratified' if small else 'standard';base=folder(shot,seed,ds)/mode;up=upstream_bank(shot,seed,ds,small);n=len(up['gnn_logits'])
 gp=up['gnn_logits'].argmax(1);lp=up['llm_logits'].argmax(1);agreement=gp==lp
 choose=torch.ones(n,dtype=torch.bool);accepted=torch.zeros(n,dtype=torch.bool);scores=torch.zeros(n,dtype=torch.float64)
 results={}
 for branch in ['agreement','disagreement']:
  for role in ['validation','query']:
   b=bank(shot,seed,ds,branch,role,small)
   if role=='validation' and branch=='agreement':continue
   clean={k:b[k] for k in ('dataset','ids','tab','g','l','gnn_pred','llm_pred')}
   pred=predict_bundle(models[branch],clean);ids=b['ids']
   if branch=='disagreement':
    q=pred['q'];trust=q[:,0]+q[:,1];pref=np.divide(q[:,0],trust,out=np.full(len(ids),.5),where=trust>0)
    if variant in ('confidence_router','confidence_preference'):
     pg=up['gnn_logits'][ids].softmax(1).max(1).values.numpy();pl=up['llm_logits'][ids].softmax(1).max(1).values.numpy()
     pred['route']=(pg<pl).astype(np.int64);pref=pg-pl
     if variant=='confidence_router':trust=np.maximum(pg,pl);pred['score']=trust
     else:pred['score']=trust
    choose[ids]=torch.as_tensor(pred['route'])==0
   else:
    if variant=='confidence_router':
     pg=up['gnn_logits'][ids].softmax(1).max(1).values.numpy();pl=up['llm_logits'][ids].softmax(1).max(1).values.numpy();pred['score']=np.sqrt(pg*pl)
    trust=np.asarray(pred['score']);pref=np.full(len(ids),np.nan)
   scores[ids]=torch.as_tensor(pred['score'],dtype=torch.float64)
   if role=='query':
    if reference is None:keep=top_mask(pred['score'],ids,models[branch]['policy']['fraction'])
    else:
     k=int(reference['accepted'][ids].sum());order=np.lexsort((np.asarray(ids),-np.asarray(pred['score'])));keep=np.zeros(len(ids),bool);keep[order[:k]]=True
    accepted[ids]=torch.as_tensor(keep)
    results[branch]=dict(ids=ids,score=torch.as_tensor(pred['score']).double(),trust=torch.as_tensor(trust).double(),preference=torch.as_tensor(pref).double(),
       route=torch.as_tensor(pred.get('route',np.zeros(len(ids),dtype=np.int64))),accepted=torch.as_tensor(keep))
 return dict(node_ids=torch.arange(n),choose_g=choose,trust=torch.where(agreement,torch.zeros_like(scores),scores).float(),
  selected_score=scores,accepted=accepted,selected_pred=torch.where(choose,gp,lp),agreement=agreement),results

def joint(shot,seed,ds,router,out,small=False,fixed=None):
 import torch
 from enhancerouter.adapter import prepare_arrays
 from eargtc.joint_explore.joint_purf_v10_v1 import training
 mode='small_stratified' if small else 'standard';sp=read(folder(shot,seed,ds)/mode/'split.json');cl=canonical(ds)
 f,s,p=prepare_arrays(sp,upstream_bank(shot,seed,ds,small),dict(node_ids=torch.arange(sp['num_nodes']),embeddings=cl['g']),
   supervision_bank(shot,seed,ds,small),router,ds)
 idx=read(ROOT/'configs/inputs.json');candidates=[];seen=set()
 for oldseed in [42,43,44]:
  cfg=idx['cells'][f'{shot}/{oldseed}/{ds}']['config'];tag=json.dumps(cfg,sort_keys=True)
  if tag not in seen:candidates.append(cfg);seen.add(tag)
 if fixed is not None:candidates=[fixed]
 results=[];best=None
 for i,cfg in enumerate(candidates):
  member=training.train(f,s,p,cl['l'],cfg,seed);v=member['checkpoints']['accuracy'];rank=(-v['accuracy'],v['nll'],i)
  results.append(dict(config=cfg,validation=v))
  if best is None or rank<best[0]:best=(rank,cfg,member)
 _,cfg,member=best;model=training.make_model(f,cl['l'],cfg);model.load_state_dict(member['states']['accuracy'])
 pred=training.predict(model,f,cl['l'],f['node_ids'])
 save(out/'checkpoint.pt',member);save(out/'predictions.pt',dict(probability=pred,node_ids=f['node_ids']))
 write(out/'joint-selection.json',dict(config=cfg,candidates=results,validation=member['checkpoints']['accuracy'],query_truth_used=False))
 return cfg

def baseline(index,only=None):
 setup();import joblib
 from enhancerouter.cli import router_recipe
 shot,seed,ds=CELLS[index]
 for small in ([False] if only=='b0' else [True] if only=='d3' else [False,True]):
  out=folder(shot,seed,ds)/('d3' if small else 'b0')
  if (out/'COMPLETE.json').exists():continue
  models=calibrated(fitted_models(shot,seed,ds,small),shot,seed,ds,small)
  router,results=route(models,shot,seed,ds,small)
  save(out/'router.pt',router)
  for b,r in results.items():save(out/f'{b}-scores.pt',r)
  out.mkdir(parents=True,exist_ok=True)
  with (out/'models.joblib').open('xb') as f:joblib.dump(models,f,compress=3)
  chosen=joint(shot,seed,ds,router,out,small)
  write(out/'config.json',dict(dataset=ds,shot=shot,seed=seed,joint=chosen,router={b:router_recipe(m) for b,m in models.items()},
     raw_normalization={b:{h:{k:w['model'][k] for k in ('center','scale') if k in w['model']} for h,w in m.get('models',{}).items()} for b,m in models.items()},
     validation_protocol='k labels/class' if small else 'original validation size',metadata=metadata()))
  seal(out,dict(query_truth_used=False,models_frozen=True))
  print('B0/D3',shot,seed,ds,'small',small,flush=True)

def oracle_requests():
 setup();import torch,random
 requests={ds:set(read(RUN/f'shared/text-request-v2-{ds}.json')['node_ids']) for ds in DS}
 for shot,seed,ds in CELLS:
  out=folder(shot,seed,ds)/'oracle'
  if (out/'SELECTION.json').exists():
   selection=read(out/'SELECTION.json')
  else:
   sp=split(shot,seed,ds);test=set(evaluation_ids(seed,ds));query=[i for i in sp['unlabeled_ids'] if i not in test]
   a=read(RUN/f'shared/random-oracle-v2-{ds}-{seed}.json')['node_ids']
   assert set(a)<=set(query)
   scores=load(folder(shot,seed,ds)/'b0/agreement-scores.pt');eligible=[(float(s),int(i)) for i,s in zip(scores['ids'],scores['score']) if int(i) not in test]
   eligible.sort(key=lambda v:(-v[0],v[1]));m=min(2000,(len(eligible)+4)//5);pool=[i for _,i in eligible[:m]]
   rng=random.Random(73019+seed);rng.shuffle(pool);b=pool[:1000]
   selection=dict(dataset=ds,shot=shot,seed=seed,a=a,b=b,high_confidence_pool_count=m,requested=[50,200,1000],
                  excludes_all_fixed_evaluation_nodes=not (set(a+b)&test),labels_used_for_selection=False)
   write(out/'SELECTION.json',selection)
  requests[ds].update(selection['a']);requests[ds].update(selection['b'])
 for ds in DS:
  write(RUN/f'shared/text-extra-request-{ds}.json',dict(node_ids=sorted(requests[ds]),dataset=ds))
 write(RUN/'ORACLE_SELECTED.json',dict(cells=75,query_truth_used=False,oracle_training_truth_deferred=True))

def oracle_banks(index):
 setup();import torch
 from eargtc.models.gnn import MLPHead
 from frontend import view_rows
 shot,seed,ds=CELLS[index];sp=split(shot,seed,ds);cl=canonical(ds);up=upstream_bank(shot,seed,ds)
 graphdata=graph(ds);truth=graphdata.y.reshape(-1).long();edge=graphdata.edge_index
 state=head_state(shot,seed,ds);head=MLPHead(128,state['net.0.weight'].shape[0],state['net.3.weight'].shape[0],dropout=0.);head.load_state_dict(state);head.eval()
 selection=read(folder(shot,seed,ds)/'oracle/SELECTION.json');test=set(evaluation_ids(seed,ds))
 for arm in ['a','b']:
  out=folder(shot,seed,ds)/f'oracle/{arm}'
  if (out/'COMPLETE.json').exists():continue
  ids=torch.tensor(selection[arm],dtype=torch.long);assert not set(ids.tolist())&test
  target=truth[ids].clone()
  assembled,_=view_rows(ds,ids,target,sp,up,head,cl,edge,extra=True)
  save(out/'disagreement.pt',assembled['disagreement'])
  write(out/'metadata.json',dict(arm=arm,parents=ids.tolist(),actual_max_N=len(ids),oracle=True,
   agreement_scorer_unchanged=True,unused_llm_hidden='target hL not consumed by this shot selected T/P modes; probabilities retained' if ds=='arxiv' or (ds=='pubmed' and shot!=10) else None,
   selected_targets=target.tolist(),clean_agreement_count=int((up['gnn_logits'][ids].argmax(1)==up['llm_logits'][ids].argmax(1)).sum()),
   environments=list(ENVIRONMENTS),fixed_eval_overlap=0,metadata=metadata()))
  seal(out,dict(oracle=True,query_evaluation_labels_used=False))
 print('ORACLE_BANKS',shot,seed,ds,flush=True)

def changed_models(original,shot,seed,ds,fraction=None,oracle=None):
 import torch
 from frozen_fit import refit,input_hash
 models=copy.deepcopy(original);diagnostics={}
 for branch,m in models.items():
  if oracle is not None and branch=='agreement':continue
  bs={d:bank(shot,seed,d,branch,'train') for d in DS};target=bs[ds]
  if fraction is not None:
   parents=torch.tensor(split(shot,seed,ds)['support_ids'],dtype=torch.long)
   gen=torch.Generator().manual_seed(seed+913)
   parents=parents[torch.randperm(len(parents),generator=gen)[:max(1,int(len(parents)*fraction+.5))]]
   target=subset(target,torch.isin(target['node_ids'],parents))
  if oracle is not None:target=merge_banks(target,oracle)
  bs[ds]=target
  for h,w in m.get('models',{}).items():
   m['models'][h],diag=refit(w,h,bs,ds)
   diag.update(parents=len(set(target['node_ids'].tolist())),rows=len(target['ids']))
   diagnostics[branch+'/'+h]=diag
 return models,diagnostics


def variant_receipt(out,shot,seed,ds,variant,parameters,models=None):
 import joblib
 base=folder(shot,seed,ds)/'b0/config.json'
 write(out/'config.json',dict(dataset=ds,shot=shot,seed=seed,variant=variant,parameters=parameters,
  inherited_B0_config=str(base),inherited_B0_config_sha256=digest(base),
  frozen='scorer family/mode/scope, full fitted input transform, raw normalization, affine, route recipe, acceptance fractions, Joint hyperparameters unless named as the measured factor',metadata=metadata()))
 if models is not None:
  with (out/'models.joblib').open('xb') as f:joblib.dump(models,f,compress=3)

def d1(index):
 setup();import torch,joblib
 from enhancerouter import router as runtime
 shot,seed,ds=CELLS[index];base=folder(shot,seed,ds)/'b0';original=joblib.load(base/'models.joblib')
 for fraction in [.25,.5,.75,1.]:
  out=folder(shot,seed,ds)/f'd1/{fraction}'
  if (out/'COMPLETE.json').exists():continue
  if fraction==1.:models,diag=original,dict(reference=True)
  else:models,diag=changed_models(original,shot,seed,ds,fraction=fraction)
  routed,scores=route(models,shot,seed,ds)
  for b,r in scores.items():save(out/f'{b}-scores.pt',r)
  write(out/'diagnostics.json',diag)
  variant_receipt(out,shot,seed,ds,'d1',dict(fraction=fraction,parent_selection_seed=seed+913,unit='support parent, all six views together'),models)
  seal(out,dict(fraction=fraction,models_frozen=True,query_truth_used=False))
 print('D1',shot,seed,ds,flush=True)


def d2(index):
 setup();import torch,joblib
 shot,seed,ds=CELLS[index];base=folder(shot,seed,ds)/'b0';original=joblib.load(base/'models.joblib')
 for arm in ['a','b']:
  oracle=load(folder(shot,seed,ds)/f'oracle/{arm}/disagreement.pt');selection=read(folder(shot,seed,ds)/'oracle/SELECTION.json')
  for requested in [50,200,1000]:
   out=folder(shot,seed,ds)/f'd2{arm}/{requested}'
   if (out/'COMPLETE.json').exists():continue
   ids=torch.tensor(selection[arm][:requested],dtype=torch.long);extra=subset(oracle,torch.isin(oracle['node_ids'],ids))
   models,diag=changed_models(original,shot,seed,ds,oracle=extra)
   routed,scores=route(models,shot,seed,ds)
   for b,r in scores.items():save(out/f'{b}-scores.pt',r)
   write(out/'diagnostics.json',dict(heads=diag,requested_N=requested,actual_N=len(ids),oracle=True,
     extra_T_rows=len(extra['ids']),extra_P_rows=int((extra['state']!=2).sum()),fixed_eval_overlap=0))
   variant_receipt(out,shot,seed,ds,'d2'+arm,dict(requested_N=requested,actual_N=len(ids),parents=ids.tolist(),oracle=True,bank_path=str(folder(shot,seed,ds)/f'oracle/{arm}/disagreement.pt'),bank_md5=digest(folder(shot,seed,ds)/f'oracle/{arm}/disagreement.pt','md5')),models)
   seal(out,dict(models_frozen=True,oracle=True,query_evaluation_labels_used=False))
 print('D2',shot,seed,ds,flush=True)

def d4(index):
 setup();import torch,joblib
 shot,seed,ds=CELLS[index];base=folder(shot,seed,ds)/'b0';original=joblib.load(base/'models.joblib')
 for variant in ['confidence_router','confidence_preference']:
  out=folder(shot,seed,ds)/f'd4/{variant}'
  if (out/'COMPLETE.json').exists():continue
  routed,scores=route(original,shot,seed,ds,variant=variant,reference=load(base/'router.pt'))
  save(out/'router.pt',routed)
  for b,r in scores.items():save(out/f'{b}-scores.pt',r)
  fixed=read(base/'config.json')['joint'];joint(shot,seed,ds,routed,out,fixed=fixed)
  variant_receipt(out,shot,seed,ds,'d4/'+variant,dict(joint_config=fixed,counts='exact B0 per-branch accepted counts',agreement_score='sqrt(cG*cL)' if variant=='confidence_router' else 'unchanged',disagreement_score='max(cG,cL)' if variant=='confidence_router' else 'learned trust',preference='cG>=cL'))
  seal(out,dict(models_frozen=True,query_truth_used=False,matched_B0_all_query_accepted_counts=True))
 print('D4',shot,seed,ds,flush=True)

def diagnostics(index):
 d1(index);d2(index);d4(index)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['baseline','b0','d3','oracle-select','oracle-banks','diagnostics','d1','d2','d4']);p.add_argument('--index',type=lambda v:int(os.environ['SLURM_ARRAY_TASK_ID']) if v.startswith('$') else int(v));a=p.parse_args()
 if a.stage in ('baseline','b0','d3'):baseline(a.index,a.stage if a.stage!='baseline' else None)
 elif a.stage=='d1':d1(a.index)
 elif a.stage=='d2':d2(a.index)
 elif a.stage=='d4':d4(a.index)
 elif a.stage=='oracle-select':oracle_requests()
 elif a.stage=='oracle-banks':oracle_banks(a.index)
 else:diagnostics(a.index)
