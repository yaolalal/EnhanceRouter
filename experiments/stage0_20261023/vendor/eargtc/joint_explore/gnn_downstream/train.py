"""Independent single-seed continuation; validation-only checkpoint selection."""
import time,os
import torch.nn.functional as F
from .io import *
from .core import seed_all,assert_open,encode,adjacency,loss,optimizer

CONFIG={'datasets':list(DATASETS),'split_seeds':list(range(42,47)),'variants':list(VARIANTS),
        'model_seed':'equals split_seed','inner_ensemble':False,'head_lr':.001,'encoder_lr':.0001,
        'weight_decay':.0001,'max_epochs':500,'validation_interval':5,'patience_checks':20,
        'pseudo_weight':1.,'selection':'validation accuracy, then lower NLL, then earlier epoch',
        'initial_checkpoint_eligible':True,'precision':'float32, TF32 disabled','gradient_clip':None,'device':'cpu','torch_threads':2,
        'loss':'independently averaged support CE + accepted pseudo CE, unweighted',
        'protocol':'transductive, frozen Router-v9-new, no new query-truth selection'}

def initialize():
    require();assert_open(RUN)
    audit=read(RUN/'INPUT_AUDIT.json');assert audit['all_pass'] and len(audit['cells'])==25
    reviewed=read(RUN/'INPUT_REVIEW_VERIFY.json');assert reviewed['all_pass'] and reviewed['audit_sha256']==sha(RUN/'INPUT_AUDIT.json')
    verify(audit['files'])
    pilot=read(RUN/'PILOT_GATE.json');assert pilot['all_pass'] and len(pilot['pilots'])==12
    verify(pilot['files'])
    write(RUN/'CONFIG.json',CONFIG)
    write(RUN/'PREREGISTERED.json',{'config_sha256':sha(RUN/'CONFIG.json'),
         'input_audit_sha256':sha(RUN/'INPUT_AUDIT.json'),'code_hashes':code_hashes(),
         'gate_hashes':{str(p):sha(p) for p in (RUN/'PILOT_GATE.json',RUN/'INPUT_REVIEW_VERIFY.json')},
         'reference_hashes':{str(p):sha(p) for p in [JOINT/'joint/evaluation/per_split.csv']+[JOINT/f'seed{s}/joint/r9_new/{d}/predictions.pt' for s in range(42,47) for d in DATASETS]},
         'query_targets_accessed':False,'slurm_job_id':os.environ['SLURM_JOB_ID']})

def gate():
    pre=read(RUN/'PREREGISTERED.json');verify(pre['code_hashes']);verify(pre['reference_hashes']);verify(pre['gate_hashes'])
    verify({RUN/'CONFIG.json':pre['config_sha256'],RUN/'INPUT_AUDIT.json':pre['input_audit_sha256']})
    if read(RUN/'CONFIG.json')!=CONFIG:raise ValueError('Training configuration mismatch')

def run_cell(ds,seed,variant,pilot=False):
    require();assert_open(RUN)
    if ds not in DATASETS or seed not in range(42,47) or variant not in VARIANTS:raise ValueError('Invalid experiment cell')
    if not pilot:gate()
    audit=read(RUN/'INPUT_AUDIT.json')
    gp=RUN/f'inputs/{ds}/graph.pt';ip=RUN/f'inputs/{ds}/seed{seed}.pt'
    verify({str(p):audit['files'][str(p)] for p in (gp,ip)})
    dest=RUN/('pilot' if pilot else 'cells')/f'seed{seed}'/ds/variant
    if (dest/'COMPLETE.json').exists():
        done=read(dest/'COMPLETE.json');verify(done['files']);print('Already complete',dest,flush=True);return
    if dest.exists() and any(dest.iterdir()):raise RuntimeError('Incomplete nonempty cell; preserve and use explicit recovery')
    graph=load(gp);item=load(ip)
    assert item['split_seed']==item['model_seed']==seed
    device=torch.device(CONFIG['device']);torch.set_num_threads(2);seed_all(seed)
    e,h=models({**graph,**item},device);tune=variant.endswith('finetune');use_pseudo=variant.startswith('router')
    initial_e=cpu_state(e);initial_h=cpu_state(h)
    opt=optimizer(e,h,tune,CONFIG['head_lr'],CONFIG['encoder_lr'],CONFIG['weight_decay'])
    support=item['support_ids'].to(device);targets=item['support_targets'].to(device)
    valid=item['valid_ids'].to(device);vy=item['valid_targets'].to(device)
    pseudo=item['pseudo_ids'].to(device) if use_pseudo else torch.empty(0,dtype=torch.long,device=device)
    py=item['pseudo_targets'].to(device) if use_pseudo else torch.empty(0,dtype=torch.long,device=device)
    x=graph['x'].to(device) if tune else None
    adj=adjacency(graph['edge_index'].to(device),len(graph['x'])) if tune else None
    cached=graph['embeddings'].to(device) if not tune else None
    def forward():return h(encode(e,x,adj) if tune else cached)
    def validation():
        e.eval();h.eval()
        with torch.no_grad():
            logits=forward();vl=F.cross_entropy(logits[valid],vy).item()
            correct=int((logits[valid].argmax(1)==vy).sum())
        if not torch.isfinite(logits).all() or not torch.isfinite(torch.tensor(vl)):raise RuntimeError('Nonfinite output')
        return correct,vl
    seed_all(seed) # Continuation RNG is the split seed, independent of constructor RNG consumption.
    best_correct,best_nll=validation();initial_val={'accuracy':best_correct/len(valid),'nll':best_nll}
    best_e=initial_e;best_h=initial_h;best_epoch=0;wait=0;history=[];started=time.monotonic()
    max_epochs=10 if pilot else CONFIG['max_epochs'];updated_e=False;updated_h=False
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,max_epochs+1):
        e.train(tune);h.train();opt.zero_grad(set_to_none=True)
        logits=forward();objective=loss(logits,support,targets,pseudo,py)
        if not torch.isfinite(objective):raise RuntimeError('Nonfinite loss')
        objective.backward();opt.step()
        if epoch==1:
            updated_e=any(not torch.equal(v.detach().cpu(),initial_e[k]) for k,v in e.state_dict().items())
            updated_h=any(not torch.equal(v.detach().cpu(),initial_h[k]) for k,v in h.state_dict().items())
            if updated_e!=tune or not updated_h:raise RuntimeError('Wrong parameter update scope')
        if epoch%CONFIG['validation_interval']==0 or epoch==max_epochs:
            correct,nll=validation()
            history.append({'epoch':epoch,'loss':float(objective.detach()),'validation_accuracy':correct/len(valid),'validation_nll':nll})
            if correct>best_correct or (correct==best_correct and nll<best_nll):
                best_correct,best_nll=correct,nll;best_epoch=epoch;best_e=cpu_state(e);best_h=cpu_state(h);wait=0
            else:wait+=1
            if epoch%50==0 or pilot:print(ds,seed,variant,history[-1],'best',best_epoch,flush=True)
            if wait>=CONFIG['patience_checks']:break
    e.load_state_dict(best_e);h.load_state_dict(best_h);e.eval();h.eval()
    with torch.no_grad():probability=forward().softmax(1).cpu()
    if not tune:assert all(torch.equal(v,initial_e[k]) for k,v in best_e.items())
    metadata={'dataset':ds,'split_seed':seed,'model_seed':seed,'variant':variant,'pilot':pilot,
              'initial_validation':initial_val,'selected_epoch':best_epoch,
              'selected_validation':{'accuracy':best_correct/len(valid),'nll':best_nll},
              'ran_epochs':epoch,'accepted_training_count':len(pseudo),'first_step_encoder_updated':updated_e,
              'first_step_head_updated':updated_h,'seconds':time.monotonic()-started,
              'cuda_peak_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,'device':str(device),
              'torch_version':torch.__version__,'slurm_job_id':os.environ['SLURM_JOB_ID'],
              'query_targets_accessed':False,'input_hashes':{str(gp):sha(gp),str(ip):sha(ip)},
              'preregistration_sha256':None if pilot else sha(RUN/'PREREGISTERED.json')}
    assert_open(RUN)
    save(dest/'checkpoint.pt',{'encoder_state':best_e,'head_state':best_h})
    save(dest/'probabilities.pt',{'node_ids':torch.arange(len(probability)),'probability':probability})
    write(dest/'history.json',history);write(dest/'metadata.json',metadata)
    write(dest/'COMPLETE.json',{'files':{str(dest/f):sha(dest/f) for f in ('checkpoint.pt','probabilities.pt','history.json','metadata.json')},
                               'model_seed':seed,'split_seed':seed,'variant':variant,'dataset':ds,'pilot':pilot})
    print('Completed',ds,seed,variant,metadata,flush=True)

def freeze():
    require();gate();audit=read(RUN/'INPUT_AUDIT.json');verify(audit['files']);files={};cells=[]
    for seed in range(42,47):
        for ds in DATASETS:
            for variant in VARIANTS:
                dest=RUN/f'cells/seed{seed}/{ds}/{variant}'
                done=read(dest/'COMPLETE.json');verify(done['files'])
                assert (done['model_seed'],done['split_seed'],done['dataset'],done['variant'],done['pilot'])==(seed,seed,ds,variant,False)
                meta=read(dest/'metadata.json');assert meta['query_targets_accessed'] is False
                assert meta['preregistration_sha256']==sha(RUN/'PREREGISTERED.json')
                item=load(RUN/f'inputs/{ds}/seed{seed}.pt')
                saved=load(dest/'probabilities.pt');p=saved['probability'];valid=item['valid_ids'];y=item['valid_targets']
                assert torch.equal(saved['node_ids'],torch.arange(len(p))) and torch.isfinite(p).all()
                assert abs(float((p[valid].argmax(1)==y).double().mean())-meta['selected_validation']['accuracy'])<1e-12
                choices=[(meta['initial_validation']['accuracy'],-meta['initial_validation']['nll'],0)]+[(v['validation_accuracy'],-v['validation_nll'],-v['epoch']) for v in read(dest/'history.json')]
                assert -max(choices)[2]==meta['selected_epoch']
                assert meta['first_step_encoder_updated']==variant.endswith('finetune') and meta['first_step_head_updated']
                assert meta['device']==CONFIG['device']
                files.update(done['files']);files[str(dest/'COMPLETE.json')]=sha(dest/'COMPLETE.json')
                cells.append({'dataset':ds,'seed':seed,'variant':variant})
    files[str(RUN/'PREREGISTERED.json')]=sha(RUN/'PREREGISTERED.json')
    write(RUN/'FROZEN.json',{'cells':cells,'cell_count':len(cells),'files':files,'query_targets_accessed':False})
