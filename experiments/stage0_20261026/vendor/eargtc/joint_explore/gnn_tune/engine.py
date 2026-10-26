"""Independent continuation; no query labels or query metrics are consumed."""
import os,time
import torch
import torch.nn.functional as F
from eargtc.joint_explore.gnn_downstream.core import seed_all,adjacency
from eargtc.joint_explore.gnn_downstream.io import models,cpu_state,load,read,write,save,sha,verify,require
from . import protocol as p

def lambda_at(weight,ramp,epoch):return weight*min(1.,epoch/ramp) if ramp else weight

def update_mode(mode,epoch):return ('frozen' if epoch<=20 else 'all') if mode=='warm20' else mode

def set_trainable(encoder,mode):
    encoder.requires_grad_(mode=='all')
    if mode=='last':encoder.convs[-1].requires_grad_(True)

def embeddings(encoder,ax,adj,mode):
    x=ax
    for i,conv in enumerate(encoder.convs):
        if i:x=torch.sparse.mm(adj,x)
        x=conv.linear(x)
        if i<len(encoder.convs)-1:
            x=F.dropout(F.relu(encoder.norms[i](x)),p=encoder.dropout,training=encoder.training and mode=='all')
    return x

def objective(logits,targets,n_gold,weight):
    gold=F.cross_entropy(logits[:n_gold],targets[:n_gold])
    pseudo=F.cross_entropy(logits[n_gold:],targets[n_gold:]) if len(targets)>n_gold else gold.new_zeros(())
    return gold+weight*pseudo,gold,pseudo

class Context:
    def __init__(self,graph,item,seed,device='cuda'):
        require();p.check_item(item,len(graph['x']),seed)
        self.graph,self.item,self.seed,self.device=graph,item,seed,torch.device(device)
        self.cached=graph['embeddings'].to(device)
        self.adj=adjacency(graph['edge_index'].to(device),len(graph['x']))
        self.ax=torch.sparse.mm(self.adj,graph['x'].to(device))
        self.valid=item['valid_ids'].to(device);self.vy=item['valid_targets'].to(device)
    def create(self,c):
        seed_all(self.seed);e,h=models({**self.graph,**self.item},self.device)
        h.net[2].p=c['dropout']
        return e,h
    def validate(self,e,h,c):
        e.eval();h.eval()
        with torch.no_grad():
            z=self.cached if c['mode']=='frozen' else embeddings(e,self.ax,self.adj,'frozen')
            logits=h(z[self.valid]);nll=F.cross_entropy(logits,self.vy).item()
            correct=int((logits.argmax(1)==self.vy).sum())
        if not torch.isfinite(logits).all() or not torch.isfinite(torch.tensor(nll)):raise RuntimeError('Nonfinite validation')
        return {'correct':correct,'count':len(self.valid),'accuracy':correct/len(self.valid),'nll':nll}

def train(ctx,c,arm,dest,pilot=False):
    require();p.open_training()
    if (dest/'COMPLETE.json').exists():
        verify(read(dest/'COMPLETE.json')['files']);return read(dest/'metadata.json')
    if dest.exists() and any(dest.iterdir()):raise RuntimeError('Incomplete candidate; preserve and explicitly recover: '+str(dest))
    e,h=ctx.create(c);initial_e=cpu_state(e);initial_h=cpu_state(h)
    initial_device={k:v.detach().clone() for k,v in e.named_parameters()}
    optimizer=torch.optim.AdamW([{'params':h.parameters(),'lr':c['head_lr']},{'params':e.parameters(),'lr':c['encoder_lr']}],weight_decay=c['wd'])
    s=ctx.item['support_ids'];sy=ctx.item['support_targets']
    pseudo=ctx.item['pseudo_ids'] if arm=='router' else torch.empty(0,dtype=torch.long)
    py=ctx.item['pseudo_targets'] if arm=='router' else torch.empty(0,dtype=torch.long)
    ids=torch.cat((s,pseudo)).to(ctx.device);targets=torch.cat((sy,py)).to(ctx.device)
    seed_all(ctx.seed)
    v=ctx.validate(e,h,c);best={'validation':v,'selected_epoch':0,'id':p.config_id(c)}
    best_e,best_h=initial_e,initial_h;initial_val=v;history=[];last_improve=0;start=time.monotonic()
    update_checks=[];previous_mode=None
    for epoch in range(1,c['max_epochs']+1):
        mode=update_mode(c['mode'],epoch);set_trainable(e,mode);e.train(mode!='frozen');h.train()
        optimizer.zero_grad(set_to_none=True)
        z=ctx.cached if mode=='frozen' else embeddings(e,ctx.ax,ctx.adj,mode)
        logits=h(z[ids]);weight=lambda_at(c['lambda'],c['ramp'],epoch)
        value,gold,pseudo_loss=objective(logits,targets,len(s),weight)
        anchor=gold.new_zeros(())
        if c['anchor'] and mode!='frozen':
            terms=[(v-initial_device[k]).square().sum() for k,v in e.named_parameters() if v.requires_grad]
            n=sum(v.numel() for v in e.parameters() if v.requires_grad)
            anchor=torch.stack(terms).sum()/n
        total=value+c['anchor']*anchor
        if not torch.isfinite(total):raise RuntimeError('Nonfinite training loss')
        total.backward()
        if mode!=previous_mode:
            before={k:v.detach().clone() for k,v in e.named_parameters()}
            hb={k:v.detach().clone() for k,v in h.named_parameters()}
        optimizer.step()
        if mode!=previous_mode:
            changed=[k for k,v in e.named_parameters() if not torch.equal(v,before[k])]
            head_changed=any(not torch.equal(v,hb[k]) for k,v in h.named_parameters())
            if mode=='frozen':assert not changed
            elif mode=='last':assert changed and all(k.startswith(f'convs.{len(e.convs)-1}.') for k in changed)
            else:assert changed
            assert head_changed
            update_checks.append({'epoch':epoch,'mode':mode,'encoder_changed':changed,'head_changed':head_changed});previous_mode=mode
        if epoch<=p.SPEC['early_dense_epochs'] or epoch%5==0 or epoch==c['max_epochs']:
            v=ctx.validate(e,h,c)
            row={'epoch':epoch,'validation':v,'gold_loss':gold.item(),'pseudo_loss':pseudo_loss.item(),'lambda':weight,'anchor':anchor.item(),'loss':total.item()}
            history.append(row);candidate={'validation':v,'selected_epoch':epoch,'id':best['id']}
            if p.rank(candidate)<p.rank(best):
                best=candidate;best_e=cpu_state(e);best_h=cpu_state(h);last_improve=epoch
            if epoch-last_improve>=100 and (c['mode']!='warm20' or epoch>20):break
    p.open_training()
    meta={**best,'config':c,'dataset':ctx.item.get('dataset'),'seed':ctx.seed,'arm':arm,
          'initial_validation':initial_val,'ran_epochs':epoch,'seconds':time.monotonic()-start,
          'device':str(ctx.device),'torch_version':torch.__version__,'gpu':torch.cuda.get_device_name(ctx.device) if ctx.device.type=='cuda' else None,
          'accepted_training_count':len(pseudo),'query_targets_accessed':False,'parameter_update_checks':update_checks,
          'slurm_job_id':os.environ['SLURM_JOB_ID'],'legacy':False,'pilot':pilot}
    save(dest/'checkpoint.pt',{'encoder_state':best_e,'head_state':best_h})
    write(dest/'metadata.json',meta);write(dest/'history.json',history)
    write(dest/'COMPLETE.json',{'files':{str(dest/f):sha(dest/f) for f in ('checkpoint.pt','metadata.json','history.json')}})
    print('TRIAL',ctx.item.get('dataset'),ctx.seed,arm,c['mode'],meta['id'],v,'best',best['selected_epoch'],'seconds',round(meta['seconds'],2),flush=True)
    return meta
