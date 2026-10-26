from common import *
setup()
import torch
from eargtc.models.gnn import MLPHead
idx=read(RUN/'HISTORICAL_INPUTS_V2.json');files={};refs={};seeds={};checks=[]
for shot,seed,ds in CELLS:
 if seed>44 and shot!=5:continue
 c=idx['cells'][key(shot,seed,ds)];directory=Path(c['inputs']['embeddings']).parent
 provenance=directory/'PROVENANCE.json';meta=read(provenance);model_seed=int(meta['model_seed'])
 assert ds not in seeds or seeds[ds]==model_seed;seeds[ds]=model_seed
 files[str(provenance)]=dict(md5=digest(provenance,'md5'),sha256=digest(provenance))
 p=directory/'gnn_head.pt'
 if shot==5 and seed==42:p=SOURCE/f'runs/gcn_ssl_v2_selected_20260623_v1/{ds}/mlp_head.pt'
 assert p.is_file(),str(p)
 obj=load(p);container=next((k for k in ('state_dict','model_state_dict','head_state') if k in obj),None)
 state=obj[container] if container else obj
 assert 'net.0.weight' in state and 'net.3.weight' in state
 files[str(p)]=dict(md5=digest(p,'md5'),sha256=digest(p));refs[key(shot,seed,ds)]=dict(path=str(p),sha256=digest(p),container=container,model_seed=model_seed)
 if shot==5 and seed==42:
  head=MLPHead(128,state['net.0.weight'].shape[0],state['net.3.weight'].shape[0],dropout=0.);head.load_state_dict(state);head.eval()
  g=load(c['inputs']['embeddings'])['embeddings'].float();expected=load(c['inputs']['upstream'])['gnn_logits'];parts=[]
  with torch.inference_mode():
   for start in range(0,len(g),4096):parts.append(head(g[start:start+4096]))
  actual=torch.cat(parts);torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-4)
  assert torch.equal(actual.argmax(1),expected.argmax(1))
  checks.append(dict(dataset=ds,model_seed=model_seed,logits_match=True,top1_exact=True,max_abs=float((actual-expected).abs().max())))
for ds in DS:
 c=idx['cells'][f'5/42/{ds}'];p=Path(c['inputs']['embeddings']).parent/'graph.pt';files[str(p)]=dict(md5=digest(p,'md5'),sha256=digest(p))
write(RUN/'HEAD_SOURCE_FILES.json',files);write(RUN/'HEAD_REFERENCES.json',refs)
write(RUN/'GNN_TRAINING_SEEDS.json',dict(by_dataset=seeds,experiment_split_seeds=list(DEV),held_out_seeds=[],source='Exact historical model_seed from all55 PROVENANCE records',seed42_anchor_checks=checks))
write(RUN/'NATIVE_FRONTEND_CORRECTION.json',dict(reason='Historical expert RNG is dataset-specific and independent of split_seed. Preserve it for new20 B0 and all75 D3 heads.',old_standard='standard for20 new cells',old_small='small_stratified',new_standard='standard_native',new_small='small_native',old_outputs_retained=True,no_method_results_from_old_frontends=True,model_seeds=seeds))
print('HEAD_SOURCES_VALIDATED',len(refs),seeds,flush=True)
