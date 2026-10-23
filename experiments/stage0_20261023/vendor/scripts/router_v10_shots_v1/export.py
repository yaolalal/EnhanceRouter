"""Label-free export only after every model/choice within a seed is frozen."""
from common import *
from eargtc.agreement_increment.policy import top_mask
from eargtc.router_v8_io.pipeline import merge_branches,load_pipeline,load_joint_pseudo_labels
import gc

def payload(bundle,b):
 s=predict_bundle(bundle,b);keep=torch.from_numpy(top_mask(s['score'],b['ids'],bundle['policy']['fraction']))
 p=dict(node_ids=b['ids'],gnn_pred=b['gnn_pred'],llm_pred=b['llm_pred'],selected_score=torch.as_tensor(s['score'],dtype=torch.float64),accepted_mask=keep)
 if bundle['branch']=='agreement':p.update(schema='agreement_increment_pipeline_v1',dataset=b['dataset'],pseudo_label=torch.where(keep,b['gnn_pred'],-1))
 else:
  route=torch.as_tensor(s['route'],dtype=torch.long);p.update(q=torch.as_tensor(s['q'],dtype=torch.float64),route=route,selected_action=torch.where(keep,route,2),pseudo_label=torch.where(keep,torch.where(route==0,b['gnn_pred'],b['llm_pred']),-1))
 return p

def main(seed):
 require();torch.set_num_threads(1);check_code();root=RUN/f'seed{seed}';freeze=read(root/'MODELS_FROZEN.json');check_files(freeze['files']);check_files(freeze['search_files'])
 items={v:{} for v in ('adaptive','fixed')}
 for d in DATASETS:
  parts={v:{} for v in items}
  for branch in BRANCHES:
   b=eval_bank(seed,d,branch,'query')
   for variant in items:
    name='bundle.joblib' if variant=='adaptive' else 'fixed_bundle.joblib';bundle=joblib.load(root/'models'/d/branch/name)
    p=payload(bundle,b);assert exact(p,payload(joblib.load(root/'models'/d/branch/name),b))
    dest=root if variant=='adaptive' else root/'fixed';save(dest/'branches'/d/(branch+'.pt'),p);parts[variant][branch]=p
  s=split(seed,d);splitids=dict(train=s['support_ids'],validation=s['valid_ids'],query=s['unlabeled_ids']);classes=read(checked(DATA/f'seed{seed}/banks'/d/'manifest.json'))['num_classes']
  for variant in items:
   dest=root if variant=='adaptive' else root/'fixed';ps=parts[variant];p=merge_branches(d,ps['disagreement'],ps['agreement'],splitids,s['num_nodes'],classes)
   target=dest/'pipeline'/(d+'.pt');save(target,p);items[variant][d]=dict(path='pipeline/'+d+'.pt',sha256=sha(target))
   save(dest/'pseudo_labels'/(d+'.pt'),{k:p[k][p['accepted_mask']].clone() for k in ('node_ids','pseudo_label','population')})
  # Replay validation without targets, then canonicalize for the downstream classifier.
  vb=validation_features(seed,d,'disagreement');bundle=joblib.load(root/'models'/d/'disagreement/bundle.joblib')
  vp=predict_bundle(bundle,vb)
  validation=dict(node_ids=vb['ids'],route=torch.as_tensor(vp['route'],dtype=torch.long),score=torch.as_tensor(vp['score'],dtype=torch.float64))
  save(root/'validation'/(d+'.pt'),validation)
  from canonical import canonical_router
  up=load(checked(DATA/f'seed{seed}/upstream'/d/'pipeline/predictions.pt'))
  canonical=canonical_router(load(root/'pipeline'/(d+'.pt')),parts['adaptive']['disagreement'],up,s,validation)
  save(root/'joint_inputs'/(d+'.pt'),canonical)
  del b,p,parts,vb,up,canonical;gc.collect();print('EXPORTED',seed,d,flush=True)
 for variant in ('fixed','adaptive'):
  dest=root if variant=='adaptive' else root/'fixed';write(dest/'pipeline/MANIFEST.json',dict(schema='router_v8_v5_manifest_v1',datasets=items[variant]))
  graphs={d:{k:split(seed,d)[k] for k in ('graph_path','graph_sha256')} for d in DATASETS}
  files=freeze_files(dest) if variant=='fixed' else {**freeze_files(root/'models'),**freeze_files(root/'branches'),**freeze_files(root/'pipeline'),**freeze_files(root/'pseudo_labels'),**freeze_files(root/'validation'),**freeze_files(root/'joint_inputs'),**freeze_files(root/'incumbents'),**freeze_files(root/'fixed'),str(root/'MODELS_FROZEN.json'):sha(root/'MODELS_FROZEN.json')}
  write(dest/'FROZEN.json',dict(files=files,graphs=graphs,models_frozen_sha256=sha(root/'MODELS_FROZEN.json'),query_truth_used=False,slurm_job_id=os.environ['SLURM_JOB_ID']))
  write(dest/'COMPLETE.json',dict(all_checks_pass=True,frozen_sha256=sha(dest/'FROZEN.json'),stage='label-free delivery, metric audit separate'))
  for d in DATASETS:assert exact(load_joint_pseudo_labels(dest,d),load(dest/'pseudo_labels'/(d+'.pt')))
 print('DELIVERY COMPLETE',seed,flush=True)
if __name__=='__main__':main(int(sys.argv[1]))
