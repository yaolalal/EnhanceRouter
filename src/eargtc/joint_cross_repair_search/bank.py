"""Independent training bank. No graph datasets or test-label files are opened."""
from dataclasses import asdict, dataclass
import math
import time
import torch
import torch.nn.functional as F
from eargtc.joint_cross_repair.artifacts import validate_features, validate_supervision, validate_pools
from eargtc.joint_cross_repair.model import DualResidualRepair, pool_loss
from eargtc.joint_cross_repair.training import TrainConfig, _score, _cpu_state


@dataclass(frozen=True)
class Recipe:
    name: str
    cross: float
    keep: float

    def validate(self):
        if not self.name or any(not math.isfinite(w) or w < 0 for w in [self.cross,self.keep]):
            raise ValueError('invalid loss recipe')


RECIPES = (
    Recipe('gold_only',0,0), Recipe('gold_keep',0,.1),
    Recipe('cross025',.25,.1), Recipe('cross050',.5,.1), Recipe('cross100',1,.1),
    Recipe('preserve025',.25,.5), Recipe('preserve050',.5,.5),
    Recipe('preserve100',1,.5), Recipe('preserve_strong',.5,1),
)
SEEDS = (42,43,44)


@torch.no_grad()
def _logits(model, f, hidden, ids, config, branch=None):
    model.eval();output={'g':[], 'l':[]}
    for start in range(0,len(ids),config.inference_batch):
        idx=ids[start:start+config.inference_batch]
        g=f['gnn_embeddings'][idx].float().to(config.device)
        l=hidden[idx].float().to(config.device)
        names=['g','l'] if branch is None else [branch]
        for name in names:
            fn=model.repair_g if name=='g' else model.repair_l
            z=fn(g,l,f['logp_'+name][idx].to(config.device))
            if not bool(torch.isfinite(z).all()):raise ValueError('nonfinite repaired logits')
            output[name].append(z.cpu())
    return {k:torch.cat(v) for k,v in output.items() if v}


def member_logits(member, branch, criterion, features, hidden, ids, config):
    model=DualResidualRepair(features['gnn_embeddings'].shape[1],hidden.shape[1],features['logp_g'].shape[1],config.hidden_dim).to(config.device)
    module=model.gnn_head if branch=='g' else model.llm_head
    module.load_state_dict(member['states'][branch][criterion])
    return _logits(model,features,hidden,ids,config,branch)[branch]


def train_member(features, supervision, pools, hidden, recipe, config, progress=None):
    recipe.validate();config.validate()
    validate_features(features)
    n,c=features['logp_g'].shape
    validate_supervision(supervision,n,c);validate_pools(pools,features,supervision)
    if hidden.ndim!=2 or hidden.shape[0]!=n:raise ValueError('hidden alignment differs')
    torch.set_num_threads(config.threads);torch.manual_seed(config.seed)
    model=DualResidualRepair(features['gnn_embeddings'].shape[1],hidden.shape[1],c,config.hidden_dim).to(config.device)
    modules={'g':model.gnn_head,'l':model.llm_head}
    optim={k:torch.optim.AdamW(m.parameters(),lr=config.lr,weight_decay=config.weight_decay) for k,m in modules.items()}
    # Dedicated streams keep gold draws identical across loss recipes and empty pools.
    rng={name:{pool:torch.Generator().manual_seed(config.seed+offset+extra)
               for pool,extra in [('gold',0),('cross',1000),('keep',2000)]}
         for name,offset in [('g',101),('l',202)]}
    states={name:{} for name in modules};checkpoints={name:{} for name in modules};val_logits={name:{} for name in modules}
    active={name:True for name in modules};stale={name:0 for name in modules};history=[]
    ids=supervision['selection_ids'];y=supervision['selection_targets'];start=time.monotonic()

    def validate(step,losses):
        logits=_logits(model,features,hidden,ids,config)
        row={'step':step,'losses':losses}
        for name,z in logits.items():
            if step==0 and not torch.equal(z,features['logp_'+name][ids]):raise ValueError('nonzero initialization')
            score={**_score(z.softmax(1),y),'step':step};row[name]=score
            improved=False
            for criterion in ['accuracy','nll']:
                def key(v):
                    return (-v['accuracy'],v['nll'],v['step']) if criterion=='accuracy' else (v['nll'],-v['accuracy'],v['step'])
                if criterion not in checkpoints[name] or key(score)<key(checkpoints[name][criterion]):
                    checkpoints[name][criterion]=dict(score);states[name][criterion]=_cpu_state(modules[name])
                    val_logits[name][criterion]=z.clone()
                    if criterion=='accuracy':improved=True
            if active[name]:
                stale[name]=0 if improved else stale[name]+1
                if stale[name]>=config.patience:active[name]=False
        history.append(row)
        if progress:progress(row)

    def sample(population,generator):
        if not len(population):return population
        return population[torch.randint(len(population),(config.batch_size,),generator=generator)]

    validate(0,{})
    for step in range(1,config.max_steps+1):
        model.train();losses={}
        for name in ['g','l']:
            if not active[name]:continue
            optim[name].zero_grad(set_to_none=True)
            positions=torch.randint(len(supervision['gold_ids']),(config.batch_size,),generator=rng[name]['gold'])
            gold=supervision['gold_ids'][positions]
            teach=sample(pools['teach_'+name],rng[name]['cross']) if recipe.cross else gold[:0]
            keep=sample(pools['keep_'+name],rng[name]['keep']) if recipe.keep else gold[:0]
            idx=torch.cat([gold,teach,keep]);ng,nt=len(gold),len(teach)
            fn=model.repair_g if name=='g' else model.repair_l
            z=fn(features['gnn_embeddings'][idx].float().to(config.device),hidden[idx].float().to(config.device),features['logp_'+name][idx].to(config.device))
            gold_loss=F.cross_entropy(z[:ng],supervision['gold_targets'][positions].to(config.device))
            other='l' if name=='g' else 'g'
            cross=pool_loss(z[ng:ng+nt],features['logp_'+other][teach].argmax(1).to(config.device),'ce')
            keep_loss=pool_loss(z[ng+nt:],features['logp_'+name][keep].exp().to(config.device),'keep')
            loss=gold_loss+recipe.cross*cross+recipe.keep*keep_loss
            if not bool(torch.isfinite(loss)):raise ValueError('nonfinite training loss')
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in modules[name].parameters()):raise ValueError('nonfinite gradient')
            optim[name].step()
            losses[name]={k:float(v.detach()) for k,v in [('gold',gold_loss),('cross',cross),('keep',keep_loss),('total',loss)]}
        if step%config.validate_every==0 or step==config.max_steps:
            validate(step,losses)
            if not any(active.values()):break
    return {'recipe':asdict(recipe),'config':asdict(config),'states':states,'checkpoints':checkpoints,
            'validation_logits':val_logits,'history':history,'train_seconds':time.monotonic()-start,
            'validation_ids':ids.clone(),'test_labels_read':False}
