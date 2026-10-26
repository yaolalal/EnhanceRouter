from common import *
import argparse

def text(ds,extra=False):
 setup();import torch,gc
 from eargtc.data import get_texts,_install_pyg_optional_extension_stubs
 _install_pyg_optional_extension_stubs()
 from eargtc.router_v7_decision.hidden import answer_hidden_and_logits
 from eargtc.router_v7.text_pipeline import align_text_replay
 from eargtc.numbered_prompts import build_numbered_prompt
 from eargtc.prompts import DIRECT_PROMPTS
 from transformers import AutoModelForCausalLM,AutoTokenizer
 request=read(RUN/f'shared/{"text-extra-request" if extra else "text-request-v2"}-{ds}.json')
 ids=torch.tensor(request['node_ids'],dtype=torch.long);mapping={int(i):j for j,i in enumerate(ids)}
 dest=RUN/f'shared/{"text-extra" if extra else "text"}/{ds}.pt'
 if dest.exists():
  receipt=read(dest.with_suffix('.json'));assert digest(dest,'md5')==receipt['output_md5'];return
 clean=canonical(ds);frozen=clean['llm_logits'][ids].softmax(1);model=None;tokenizer=None;outputs={};meta={};started=time.monotonic()
 cache_roots=[SOURCE/'runs/router_v10_multiseed_data_20260912_v1/shared/text',SOURCE/'runs/router_v10_shots_v1_20260914/shared/text'] if not extra else []
 base=load(textcache(ds)) if extra else None
 legacy=load(SOURCE/f'runs/router_v7_text_environments_20260815_v2/{ds}/text_environments.pt')['environments']
 refpath=next((SOURCE/f'runs/router_v7_text_logits_20260815_v2/text_truncate_050/{ds}/chunks').glob('*.jsonl.meta.json'))
 ref=read(refpath);assert ref['use_chat_template'] is True
 template=Path(ref['prompt_template_path']).read_text().rstrip() if ref.get('prompt_template_path') else None
 additional={str(refpath):dict(md5=digest(refpath,'md5'))}
 if ref.get('prompt_template_path'):additional[ref['prompt_template_path']]=dict(md5=digest(ref['prompt_template_path'],'md5'))
 pclean=SOURCE/f'runs/router_v7_text_inputs_20260815_v3/text_clean_replay/datasets/{ds}.pt'
 dclean=torch.load(pclean,map_location='cpu',weights_only=False);original=get_texts(dclean);del dclean
 additional[str(pclean)]=dict(md5=digest(pclean,'md5'))
 runtime_hash=hashlib.sha256(json.dumps(dict(torch=torch.__version__,gpu=[torch.cuda.get_device_name(j) for j in range(torch.cuda.device_count())],fp16=True,batch=int(os.environ.get('STAGE0_LLM_BATCH',1))),sort_keys=True).encode()).hexdigest()[:24]
 clean_cache=torch.full_like(frozen,float('nan'));clean_dir=RUN/f'shared/runtime-clean/{ds}/{runtime_hash}';clean_dir.mkdir(parents=True,exist_ok=True)
 for cp in sorted(clean_dir.glob('batch-*.pt')):
  r=load(cp);pairs=[(mapping[int(n)],j) for j,n in enumerate(r['node_ids']) if int(n) in mapping]
  if pairs:
   a,b=map(torch.tensor,zip(*pairs));clean_cache[a]=r['probability'][b]
 def pairs_for(nodes):return [(mapping[int(n)],j) for j,n in enumerate(nodes) if int(n) in mapping]
 def infer(texts):
  prompts=[]
  for t in texts:
   human=f'{t}\n{template}' if template is not None else build_numbered_prompt(text=t,base_prompt=DIRECT_PROMPTS[ds],label_names=ref['label_names'])
   prompts.append(tokenizer.apply_chat_template([{'role':'user','content':human}],tokenize=False,add_generation_prompt=True))
  enc=tokenizer(prompts,return_tensors='pt',padding=True,truncation=True,max_length=int(ref['max_input_length']))
  enc={k:v.to('cuda:0') for k,v in enc.items()};enc['position_ids']=(enc['attention_mask'].long().cumsum(-1)-1).clamp_min(0)
  h,num=answer_hidden_and_logits(model,enc,ref['number_token_ids'])
  logits=torch.empty_like(num,dtype=torch.float32);logits[:,ref['number_to_label_ids']]=num.float()
  return h.cpu().half(),logits.softmax(1).cpu()
 for name in ENVIRONMENTS[1:4]:
  probability=torch.full_like(frozen,float('nan'));hidden=clean['l'][ids].clone();available=torch.zeros(len(ids),dtype=torch.bool)
  for q in legacy[name].values():
   if not isinstance(q,dict) or 'node_ids' not in q or 'llm_prob' not in q:continue
   pairs=pairs_for(q['node_ids'])
   if pairs:
    a,b=map(torch.tensor,zip(*pairs));probability[a]=q['llm_prob'][b].float()
  for root in cache_roots:
   p=root/ds/(name+'.pt')
   if not p.exists():continue
   old=load(p);pairs=pairs_for(old['node_ids'])
   if pairs:
    a,b=map(torch.tensor,zip(*pairs));hidden[a]=old['l'][b];available[a]=True
    if 'llm_prob' in old:probability[a]=old['llm_prob'][b].float()
  # Previously cached frozen-expert forwards are label-blind and reusable by node ID.
  prior=ROOT/f'runs/stage0-20261023-v1/shared/text/{ds}.pt'
  if prior.exists():
   old=load(prior);pairs=[(a,b) for a,b in pairs_for(old['node_ids']) if not bool(available[a]) or not bool(torch.isfinite(probability[a]).all())]
   if pairs:
    a,b=map(torch.tensor,zip(*pairs));hidden[a]=old['environments'][name]['l'][b]
    probability[a]=old['environments'][name]['llm_prob'][b];available[a]=True
   additional[str(prior)]=dict(md5=digest(prior,'md5'))
  if base is not None:
   pairs=pairs_for(base['node_ids'])
   if pairs:
    a,b=map(torch.tensor,zip(*pairs));hidden[a]=base['environments'][name]['l'][b];available[a]=True;probability[a]=base['environments'][name]['llm_prob'][b]
  p=SOURCE/f'runs/router_v7_text_inputs_20260815_v3/{name}/datasets/{ds}.pt'
  shifted_data=torch.load(p,map_location='cpu',weights_only=False);shifted=get_texts(shifted_data);del shifted_data
  changed=torch.tensor([shifted[int(i)]!=original[int(i)] for i in ids]);available|=~changed;probability[~changed]=frozen[~changed];hidden[~changed]=clean['l'][ids][~changed]
  chunkdir=RUN/f'shared/text-progress-v2/{ds}/{"extra" if extra else "initial"}/{name}';chunkdir.mkdir(parents=True,exist_ok=True)
  for chunk in sorted(chunkdir.glob('batch-*.pt')):
   r=load(chunk);pairs=pairs_for(r['node_ids'])
   if pairs:
    a,b=map(torch.tensor,zip(*pairs));hidden[a]=r['l'][b];available[a]=True;probability[a]=r['llm_prob'][b]
  needed=torch.where(~available | ~torch.isfinite(probability).all(1))[0].tolist()
  if needed and model is None:
   if not torch.cuda.is_available():raise RuntimeError(f'CACHE_MISS_REQUIRES_GPU {ds} {name} {len(needed)}')
   modelpath='/data1/wujingyao-20354/wujingyao/codes/models/llama-3-8b-instruct'
   tokenizer=AutoTokenizer.from_pretrained(modelpath,local_files_only=True,use_fast=True)
   if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
   tokenizer.padding_side='left';tokenizer.truncation_side='left';kwargs=dict(device_map={'':'cuda:0'})
   if torch.cuda.get_device_properties(0).total_memory<20*1024**3:
    if torch.cuda.device_count()<2:raise RuntimeError('11GB GPUs require2 allocated GPUs to keep frozen fp16 model entirely on GPUs')
    kwargs=dict(device_map='balanced',max_memory={j:'9GiB' for j in range(torch.cuda.device_count())}|{'cpu':'0GiB'})
   model=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.float16,**kwargs);model.eval()
  batch=int(os.environ.get('STAGE0_LLM_BATCH',1))
  print('TEXT_START',ds,name,'requested',len(ids),'missing',len(needed),flush=True)
  with torch.inference_mode():
   for start in range(0,len(needed),batch):
    rows=needed[start:start+batch];h,prob=infer([shifted[int(ids[j])] for j in rows]);hidden[rows]=h
    missing=~torch.isfinite(probability[rows]).all(1)
    if bool(missing.any()):
     unknown=[j for j in rows if not bool(torch.isfinite(clean_cache[j]).all())]
     if unknown:
      _,runtime_clean=infer([original[int(ids[j])] for j in unknown]);clean_cache[unknown]=runtime_clean
      clean_tag=hashlib.sha256(ids[unknown].numpy().tobytes()).hexdigest()[:24]
      save(clean_dir/f'batch-{clean_tag}.pt',dict(node_ids=ids[unknown],probability=runtime_clean))
     aligned=align_text_replay(frozen_clean=frozen[rows],runtime_clean=clean_cache[rows],runtime_shifted=prob,changed_mask=changed[rows])
     probability[torch.tensor(rows)[missing]]=aligned[missing]
    tag=hashlib.sha256(ids[rows].numpy().tobytes()).hexdigest()[:24]
    save(chunkdir/f'batch-{tag}.pt',dict(node_ids=ids[rows],l=hidden[rows],llm_prob=probability[rows]))
    if start%100==0:print('LLM',ds,name,start,len(needed),flush=True)
    if time.monotonic()-started>5400:
     print('CHECKPOINT_REQUEUE',ds,name,start,flush=True)
     if int(os.environ.get('SLURM_RESTART_COUNT',0))>=12:raise RuntimeError('Text cache exceeded bounded requeue count')
     subprocess.run(['scontrol','requeue',os.environ['SLURM_JOB_ID']],check=True);raise SystemExit(0)
  assert bool(torch.isfinite(probability).all()) and torch.allclose(probability.sum(1),torch.ones(len(ids)),atol=1e-5)
  assert torch.equal(probability[~changed],frozen[~changed])
  outputs[name]=dict(l=hidden,llm_prob=probability,changed_mask=changed)
  meta[name]=dict(rows=len(ids),cached=int(available.sum()),inferred=len(needed));del shifted;gc.collect()
 save(dest,dict(node_ids=ids,environments=outputs))
 write(dest.with_suffix('.json'),dict(dataset=ds,environments=meta,extra=extra,source_extra=additional,
   source_probabilities='Historical sparse caches reused; missing outputs use exact original label permutation and align_text_replay with paired clean forward.',
   runtime_gpu=[torch.cuda.get_device_name(j) for j in range(torch.cuda.device_count())],
   full_config=dict(model='/data1/wujingyao-20354/wujingyao/codes/models/llama-3-8b-instruct',dtype='float16',quantized=False,batch=batch,padding_side='left',truncation_side='left',max_input_length=ref['max_input_length'],number_token_ids=ref['number_token_ids'],number_to_label_ids=ref['number_to_label_ids'],prompt_metadata=str(refpath),device_map=getattr(model,'hf_device_map',None) if model is not None else None),
   metadata=metadata(),output_md5=digest(dest,'md5')))

def heads(index):
 setup();import torch
 from tune_gcn_ssl_1000_eval import train_head,HeadTrial
 from eargtc.utils import set_seed
 shot,seed,ds=CELLS[index];sp=split(shot,seed,ds);dest=folder(shot,seed,ds)
 g=graph(ds);truth=g.y.reshape(-1).long();cl=canonical(ds);device=torch.device('cuda:0')
 emb=cl['g'].to(device);classes=cl['llm_logits'].shape[1]
 trial=HeadTrial(**read(SOURCE/f'runs/gcn_ssl_v2_selected_20260623_v1/{ds}/mlp_metrics.json')['trial'])
 support=torch.tensor(sp['support_ids'],dtype=torch.long);valid=torch.tensor(sp['valid_ids'],dtype=torch.long)
 gen=torch.Generator().manual_seed(seed+1181);small=[]
 for c in range(classes):
  nodes=valid[truth[valid]==c];small.extend(nodes[torch.randperm(len(nodes),generator=gen)[:shot]].tolist())
 small=torch.tensor(sorted(small),dtype=torch.long)
 for name,vids in [('standard_native',valid)]:
  out=dest/name
  if (out/'HEAD_COMPLETE.json').exists():continue
  target=torch.full_like(truth,-1);target[support]=truth[support];target[vids]=truth[vids]
  set_seed(gnn_training_seed(ds))
  logits,stats,artifact=train_head(emb,target.to(device),support.to(device),vids.to(device),classes,trial)
  save(out/'upstream.pt',dict(gnn_logits=logits.cpu().float(),llm_logits=cl['llm_logits']))
  save(out/'head.pt',{k:v.cpu() for k,v in artifact['state_dict'].items()})
  save(out/'supervision.pt',dict(support_ids=support,support_targets=truth[support],valid_ids=vids,valid_targets=truth[vids]))
  write(out/'head-training.json',dict(trial=trial.__dict__,stats=stats,history=artifact['history'],training_seed=gnn_training_seed(ds),split_seed=seed,valid_ids=vids.tolist(),query_truth_used=False))
  write(out/'HEAD_COMPLETE.json',dict(all_checks_pass=True,shot=shot,seed=seed,dataset=ds,validation_count=len(vids),metadata=metadata()))
  print('HEAD',shot,seed,ds,name,stats,flush=True)

def view_rows(ds,ids,targets,sp,up,head,cl,edge,*,extra=False):
 import torch
 from eargtc.router_augmentation_v8.data import organize
 from eargtc.router_v8.prototypes import fit_class_prototypes
 from eargtc.router_v7.perturbations import mixed_assignment
 from scripts.router_augmentation_v8.build import build70
 g,l=cl['g'],cl['l'];pg=up['gnn_logits'].float().softmax(1);pl=cl['llm_logits'].softmax(1);n=len(pg);classes=pg.shape[1]
 support=torch.tensor(sp['support_ids'],dtype=torch.long);safe=supervision_bank(sp['shots'],sp['split_seed'],ds)
 fitted=fit_class_prototypes(g,support,safe['support_targets'],num_classes=classes)
 clean=build70(pg[ids],pl[ids],ids,edge,pg.argmax(1),g,fitted,n,classes)
 cache=load(RUN/f'shared/{"text-extra" if extra and (RUN/f"shared/text-extra/{ds}.pt").exists() else "text"}/{ds}.pt')
 lookup={int(i):j for j,i in enumerate(cache['node_ids'])};need_hidden=True
 legacy=load(SOURCE/f'runs/router_v7_text_environments_20260815_v2/{ds}/text_environments.pt')['environments']
 assigned=mixed_assignment(range(n),seed=303);envs=[]
 for name in ENVIRONMENTS:
  eg=g[ids].clone();el=l[ids].clone();epg=pg[ids].clone();epl=pl[ids].clone();tab=clean.clone()
  if name.startswith('text'):
   if need_hidden:
    assert all(int(i) in lookup for i in ids),(ds,'text cache misses required nodes')
    pos=torch.tensor([lookup[int(i)] for i in ids]);r=cache['environments'][name]
    el=r['l'][pos];epl=r['llm_prob'][pos];changed=r['changed_mask'][pos]
   else:
    r=legacy[name]['query'];mp={int(i):j for j,i in enumerate(r['node_ids'])};epl=r['llm_prob'][torch.tensor([mp[int(i)] for i in ids])].float()
    changed=(epl!=pl[ids]).any(1)
   tab=build70(epg,epl,ids,edge,pg.argmax(1),g,fitted,n,classes);tab[~changed]=clean[~changed]
  elif name.startswith('graph'):
   r=load(SOURCE/f'runs/router_v10_multiseed_data_20260912_v1/shared/graph/{ds}/{name}.pt');shifted=r['g'];shiftedge=r['edge_index']
   changed=torch.tensor([assigned[int(i)]=='graph' for i in ids]) if 'mixed' in name else torch.ones(len(ids),dtype=torch.bool)
   with torch.inference_mode():prob=head(shifted).softmax(1)
   eg=shifted[ids].clone();eg[~changed]=g[ids][~changed];epg=prob[ids].clone();epg[~changed]=pg[ids][~changed]
   proto=g.clone();proto[ids]=eg;tab=build70(epg,epl,ids,shiftedge,prob.argmax(1),proto,fitted,n,classes);tab[~changed]=clean[~changed]
  envs.append(dict(environment=name,node_ids=ids,gnn_prob=epg,llm_prob=epl,tab70=tab,g=eg,l=el))
 return organize(ds,envs,gold_ids=ids,gold_targets=targets,num_nodes=n),fitted

def banks(index):
 setup();import torch
 assert read(RUN/'PREREQUISITES_FROZEN.json')['all_checks_pass']
 from eargtc.models.gnn import MLPHead
 from scripts.router_augmentation_v8.build import build70
 from scripts.router_v10_shots_v1.views import evaluation_bank
 shot,seed,ds=CELLS[index];sp=split(shot,seed,ds);dest=folder(shot,seed,ds);cl=canonical(ds);g=graph(ds);edge=canonical_edge(ds);n=len(cl['g'])
 for mode in (['small_native'] if seed<=44 or shot==5 else ['standard_native','small_native']):
  out=dest/mode
  if (out/'BANK_COMPLETE.json').exists():continue
  up=load(out/'upstream.pt');safe=load(out/'supervision.pt');ids=safe['support_ids'];state=load(out/'head.pt')
  head=MLPHead(128,state['net.0.weight'].shape[0],state['net.3.weight'].shape[0],dropout=0.);head.load_state_dict(state);head.eval()
  localsp=dict(sp,valid_ids=safe['valid_ids'].tolist())
  localsp['unlabeled_ids']=sorted(set(range(n))-set(localsp['support_ids'])-set(localsp['valid_ids']))
  localsp['evaluation_full_ids']=localsp['unlabeled_ids'];write(out/'split.json',localsp)
  assembled,fitted=view_rows(ds,ids,safe['support_targets'],sp,up,head,cl,edge)
  for b in ['agreement','disagreement']:save(out/f'{b}_train.pt',assembled[b])
  pg=up['gnn_logits'].softmax(1);pl=up['llm_logits'].softmax(1)
  for role,rows in [('validation',safe['valid_ids']),('query',torch.tensor(localsp['unlabeled_ids'],dtype=torch.long))]:
   tab=build70(pg[rows],pl[rows],rows,edge,pg.argmax(1),cl['g'],fitted,n,pg.shape[1])
   for b in ['agreement','disagreement']:
    value=evaluation_bank(ds,rows,tab,cl['g'][rows],cl['l'][rows],pg.argmax(1)[rows],pl.argmax(1)[rows],branch=b,targets=safe['valid_targets'] if role=='validation' else None)
    save(out/f'{b}_{role}.pt',value)
  write(out/'BANK_COMPLETE.json',dict(dataset=ds,shot=shot,seed=seed,metadata=metadata(),query_truth_used=False,files={p.name:dict(md5=digest(p,'md5'),sha256=digest(p)) for p in out.glob('*.pt')}))
  print('BANK',shot,seed,ds,mode,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['text','heads','banks']);p.add_argument('--dataset');p.add_argument('--index',type=lambda v:int(os.environ['SLURM_ARRAY_TASK_ID']) if v.startswith('$') else int(v));p.add_argument('--extra',action='store_true');a=p.parse_args()
 if a.stage=='text':text(a.dataset,a.extra)
 elif a.stage=='heads':heads(a.index)
 else:banks(a.index)
