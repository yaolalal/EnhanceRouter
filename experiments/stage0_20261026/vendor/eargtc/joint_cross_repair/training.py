"""Frozen-cache training; this module never opens graph or test-label artifacts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import time
import torch
import torch.nn.functional as F

from .artifacts import (load_export, validate_features, validate_supervision, validate_pools, sha256,
                        read_json, write_json, save_tensor, verify_files)
from .model import DualResidualRepair, pool_loss


@dataclass(frozen=True)
class TrainConfig:
    hidden_dim: int = 64
    batch_size: int = 64
    max_steps: int = 500
    validate_every: int = 25
    patience: int = 5
    lr: float = .001
    weight_decay: float = .0001
    cross_weight: float = 1.
    keep_weight: float = .1
    seed: int = 42
    device: str = 'cpu'
    inference_batch: int = 1024
    threads: int = 4

    def validate(self):
        for key in ['hidden_dim','batch_size','max_steps','validate_every','patience','inference_batch','threads']:
            if getattr(self,key)<1: raise ValueError(f'{key} must be positive')
        for key in ['lr','weight_decay','cross_weight','keep_weight']:
            if not math.isfinite(getattr(self,key)) or getattr(self,key)<0:raise ValueError(f'invalid {key}')
        if self.lr==0:raise ValueError('learning rate must be positive')


def scheme_b_probability(pg, pl, choose_g, agreement):
    return torch.where(agreement[:,None], (pg+pl)*.5, torch.where(choose_g[:,None],pg,pl))


def blend(pg, pl, pb, alpha, beta):
    return (1-float(beta))*pb+float(beta)*(float(alpha)*pg+(1-float(alpha))*pl)


def _score(p,y):
    if len(y)==0 or p.shape[0]!=len(y) or not torch.isfinite(p).all():raise ValueError('invalid validation predictions')
    return {'accuracy':float((p.argmax(1)==y).double().mean()),
            'nll':float(-p[torch.arange(len(y)),y].double().clamp_min(1e-12).log().mean())}


def select_fusion(pg,pl,pb,y):
    trials=[]
    for alpha in [0.,.5,1.]:
        for beta in [0.,.25,.5,1.]:
            trials.append({'alpha':alpha,'beta':beta,**_score(blend(pg,pl,pb,alpha,beta),y)})
    best=min(trials,key=lambda r:(-r['accuracy'],r['nll'],r['beta'],abs(r['alpha']-.5),r['alpha']))
    return {**best,'trials':trials,'selection_population':'historical_validation_calibration_half'}


def _cpu_state(module):
    return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}


@torch.no_grad()
def predict(model,features,hidden,ids,config):
    model.eval()
    device=torch.device(config.device)
    g_out,l_out=[],[]
    for start in range(0,len(ids),config.inference_batch):
        part=ids[start:start+config.inference_batch]
        g=features['gnn_embeddings'][part].to(device=device,dtype=torch.float32)
        l=hidden[part].to(device=device,dtype=torch.float32)
        bg,bl=[features[k][part].to(device) for k in ['logp_g','logp_l']]
        zg,zl=model(g,l,bg,bl)
        if not bool(torch.isfinite(zg).all() & torch.isfinite(zl).all()):raise ValueError('nonfinite repaired logits')
        g_out.append(zg.softmax(1).cpu());l_out.append(zl.softmax(1).cpu())
    return torch.cat(g_out),torch.cat(l_out)


def train_arrays(features,supervision,pools,hidden,config,progress=None):
    config.validate()
    validate_features(features)
    n,c=features['logp_g'].shape
    validate_supervision(supervision,n,c)
    if hidden.ndim!=2 or hidden.shape[0]!=n:raise ValueError('hidden rows must align')
    if set(pools)!={'teach_g','teach_l','keep_g','keep_l','agreement','unselected'}:raise ValueError('invalid pools')
    for ids in pools.values():
        if ids.dtype!=torch.long or ids.ndim!=1 or len(ids)!=len(ids.unique()) or not bool(torch.isin(ids,supervision['test_ids']).all()):
            raise ValueError('training pool must contain unique test candidate IDs only')
    validate_pools(pools,features,supervision)
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    if config.device.startswith('cuda'):torch.cuda.manual_seed_all(config.seed)
    model=DualResidualRepair(features['gnn_embeddings'].shape[1],hidden.shape[1],c,config.hidden_dim).to(config.device)
    modules={'g':model.gnn_head,'l':model.llm_head}
    optimizers={k:torch.optim.AdamW(m.parameters(),lr=config.lr,weight_decay=config.weight_decay) for k,m in modules.items()}
    generators={k:torch.Generator().manual_seed(config.seed+offset) for k,offset in [('g',101),('l',202)]}
    step0=_cpu_state(model)
    check_ids=supervision['selection_ids']
    with torch.no_grad():
        part=check_ids[:min(len(check_ids),8)]
        zg,zl=model(features['gnn_embeddings'][part].to(config.device),hidden[part].float().to(config.device),
                    features['logp_g'][part].to(config.device),features['logp_l'][part].to(config.device))
        if not torch.equal(zg.cpu(),features['logp_g'][part]) or not torch.equal(zl.cpu(),features['logp_l'][part]):
            raise ValueError('zero-initialized branches do not reproduce experts')
    best={};best_states={};stale={'g':0,'l':0};active={'g':True,'l':True};history=[]
    start_time=time.monotonic()

    def validate(step,losses):
        pg,pl=predict(model,features,hidden,check_ids,config)
        row={'step':step,'losses':losses,'elapsed_seconds':time.monotonic()-start_time}
        for name,p in [('g',pg),('l',pl)]:
            score={**_score(p,supervision['selection_targets']),'step':step}
            row[name]=score
            key=(-score['accuracy'],score['nll'],step)
            if name not in best or key<(-best[name]['accuracy'],best[name]['nll'],best[name]['step']):
                best[name]=score;best_states[name]=_cpu_state(modules[name]);stale[name]=0
            elif active[name]:stale[name]+=1
            if stale[name]>=config.patience:active[name]=False
        history.append(row)
        if progress:progress(row)

    def sampled(ids,generator):
        if len(ids)==0:return ids
        return ids[torch.randint(len(ids),(config.batch_size,),generator=generator)]

    def one_loss(name,ids,kind,target=None):
        if len(ids)==0:return next(modules[name].parameters()).sum()*0
        g=features['gnn_embeddings'][ids].float().to(config.device)
        l=hidden[ids].float().to(config.device)
        base=features['logp_'+name][ids].to(config.device)
        logits=(model.repair_g if name=='g' else model.repair_l)(g,l,base)
        if kind=='keep':target=base.exp()
        return pool_loss(logits,target.to(config.device).detach(),kind)

    validate(0,{})
    for step in range(1,config.max_steps+1):
        model.train();losses={}
        for name in ['g','l']:
            if not active[name]:continue
            opt=optimizers[name];opt.zero_grad(set_to_none=True);gen=generators[name]
            # Gold labels come only from support-aligned targets, never full-node labels.
            positions=torch.randint(len(supervision['gold_ids']),(config.batch_size,),generator=gen)
            gold=supervision['gold_ids'][positions]
            gold_loss=one_loss(name,gold,'ce',supervision['gold_targets'][positions])
            teach=sampled(pools['teach_'+name],gen)
            other='l' if name=='g' else 'g'
            pseudo=features['logp_'+other][teach].argmax(1)
            cross=one_loss(name,teach,'ce',pseudo)
            keep=one_loss(name,sampled(pools['keep_'+name],gen),'keep')
            loss=gold_loss+config.cross_weight*cross+config.keep_weight*keep
            if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in modules[name].parameters()):
                raise ValueError('nonfinite repair gradients')
            opt.step()
            losses[name]={'gold':float(gold_loss.detach()),'cross':float(cross.detach()),'keep':float(keep.detach()),'total':float(loss.detach())}
        if step%config.validate_every==0 or step==config.max_steps:
            validate(step,losses)
            if not any(active.values()):break
    last=_cpu_state(model)
    for name,module in modules.items():module.load_state_dict(best_states[name])
    best_combined=_cpu_state(model)
    pg,pl=predict(model,features,hidden,features['node_ids'],config)
    pg0,pl0=features['logp_g'].exp(),features['logp_l'].exp()
    pb=scheme_b_probability(pg0,pl0,features['choose_g'],features['agreement'])
    if not torch.equal(pb.argmax(1),features['selected_pred']):raise ValueError('B anchor argmax mismatch')
    y=supervision['selection_targets'];idx=check_ids
    fusion=select_fusion(pg[idx],pl[idx],pb[idx],y)
    control=select_fusion(pg0[idx],pl0[idx],pb[idx],y)
    predictions={'node_ids':features['node_ids'].clone(),'p_g':pg0,'p_l':pl0,'p_b':pb,
                 'p_g_repaired':pg,'p_l_repaired':pl,
                 'p_joint':blend(pg,pl,pb,fusion['alpha'],fusion['beta']),
                 'p_base_fusion':blend(pg0,pl0,pb,control['alpha'],control['beta'])}
    return {'predictions':predictions,'selection':{**best,'fusion':fusion,'base_fusion':control},
            'history':history,'states':{'step0':step0,'best':best_combined,'last':last},
            'train_seconds':time.monotonic()-start_time,'parameter_count':sum(p.numel() for p in model.parameters())}


def train_dataset(export_dir,output_dir,config):
    export_dir,output_dir=Path(export_dir).resolve(),Path(output_dir).resolve()
    if output_dir.exists():raise FileExistsError(f'refusing to overwrite run: {output_dir}')
    root=Path(__file__).resolve().parents[2]
    code_paths=sorted(Path(__file__).parent.glob('*.py'))+sorted((root/'scripts/joint_cross_repair').glob('*.py'))
    source_code={str(path):sha256(path) for path in code_paths}
    f,s,p,h,meta=load_export(export_dir)
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    stage=output_dir.with_name(output_dir.name+f'.staging.{os.getpid()}');stage.mkdir()
    write_json(stage/'config.json',{'train':asdict(config),'dataset':meta['dataset'],'export_dir':str(export_dir),
                                 'export_complete_sha256':sha256(export_dir/'complete.json'),'protocol':meta['protocol']})
    history_path=stage/'progress.jsonl'
    def progress(row):
        import json
        with history_path.open('a') as handle:handle.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps({'dataset':meta['dataset'],**row},allow_nan=False),flush=True)
    result=train_arrays(f,s,p,h,config,progress)
    for name,state in result.pop('states').items():save_tensor(stage/f'{name}.pt',state)
    save_tensor(stage/'predictions.pt',result.pop('predictions'))
    write_json(stage/'selection.json',result['selection']);write_json(stage/'training_summary.json',result)
    verify_files(meta['source_hashes'])
    verify_files(source_code)
    files=['config.json','step0.pt','best.pt','last.pt','predictions.pt','selection.json','training_summary.json','progress.jsonl']
    write_json(stage/'complete.json',{'schema_version':1,'dataset':meta['dataset'],'all_checks_pass':True,
                                   'test_labels_read':False,'source_code':source_code,
                                   'artifacts':{name:sha256(stage/name) for name in files}})
    os.rename(stage,output_dir)
    return result
