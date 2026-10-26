"""Finite validation-only calibration of independent expert residuals."""
import math
import torch
from eargtc.joint_v6_unified.experiment import require_slurm,probability,score
SCALES=(.25,.5,1.)
ALPHAS=(0.,.25,.5,.75,1.)


def scaled_ensemble(base,deltas,scale):
    require_slurm()
    if (base.ndim!=2 or deltas.shape!=(3,*base.shape) or not math.isfinite(scale) or scale not in SCALES
            or not bool(torch.isfinite(base).all()) or not bool(torch.isfinite(deltas).all())):
        raise ValueError('Invalid three-seed residual inputs')
    return probability(torch.stack([(base+scale*d).softmax(1) for d in deltas]).mean(0))


def build_candidates(members,train,val):
    require_slurm();out={}
    for branch in ('g','l'):
        base=train['logp_'+branch][val['ids']]
        values=[dict(id=branch+'/raw',probability=base.exp(),metadata=dict(raw=True,branch=branch))]
        for cid,group in sorted(members.items()):
            if [m['seed'] for m in group]!=[42,43,44]:raise ValueError('Wrong seed population')
            for criterion in ('accuracy','nll'):
                delta=torch.stack([m['branches'][branch]['validation_delta'][criterion] for m in group])
                for scale in SCALES:
                    values.append(dict(id=f'{branch}/{cid}/{criterion}/{scale:g}',
                        probability=scaled_ensemble(base,delta,scale),metadata=dict(raw=False,branch=branch,
                        config_id=cid,criterion=criterion,scale=scale)))
        out[branch]=values
    return out


def select(candidates,targets,incumbent):
    require_slurm()
    if set(candidates)!={'g','l'}:raise ValueError('Only GNN and LLM candidates are permitted')
    kept={}
    for branch,values in candidates.items():
        if not values or len({c['id'] for c in values})!=len(values):raise ValueError('Invalid candidate identities')
        raw=[c for c in values if c['metadata']['raw']]
        if len(raw)!=1:raise ValueError('Exactly one raw expert required')
        for c in values:score(c['probability'],targets)
        nonraw=[c for c in values if not c['metadata']['raw']]
        nonraw.sort(key=lambda c:(-score(c['probability'],targets)['accuracy'],score(c['probability'],targets)['nll'],c['id']))
        kept[branch]=raw+nonraw[:3]
    trials=[]
    for g in kept['g']:
        for l in kept['l']:
            for alpha in ALPHAS:
                p=alpha*g['probability']+(1-alpha)*l['probability']
                active=int(alpha>0 and not g['metadata']['raw'])+int(alpha<1 and not l['metadata']['raw'])
                trials.append(dict(g_id=g['id'],l_id=l['id'],alpha=alpha,active_heads=active,**score(p,targets)))
    best=min(trials,key=lambda t:(-t['accuracy'],t['nll'],t['active_heads'],t['g_id'],t['l_id'],t['alpha']))
    baseline=score(incumbent,targets)
    winner='incumbent' if (-baseline['accuracy'],baseline['nll'])<=(-best['accuracy'],best['nll']) else 'repair'
    selected=baseline if winner=='incumbent' else {k:best[k] for k in ('accuracy','nll')}
    return dict(winner=winner,**selected,g_id=best['g_id'],l_id=best['l_id'],alpha=best['alpha'],
                active_heads=0 if winner=='incumbent' else best['active_heads'],
                best_repair=best,incumbent_score=baseline,
                shortlist={b:[c['id'] for c in v] for b,v in kept.items()},trials=trials)
