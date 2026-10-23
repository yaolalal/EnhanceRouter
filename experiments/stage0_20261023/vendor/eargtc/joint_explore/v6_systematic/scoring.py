"""Validation-only readouts. Raw graph labels are never loaded here."""
from itertools import product
from .manifest import slurm_only


def score(prob,targets):
    slurm_only()
    import torch
    if prob.ndim!=2 or len(prob)!=len(targets) or not len(targets) or not torch.isfinite(prob).all() or (prob<0).any():
        raise ValueError('invalid probability/target alignment')
    if not torch.allclose(prob.sum(1),torch.ones(len(prob)),atol=2e-5):
        raise ValueError('probabilities are not normalized')
    return {'accuracy':float((prob.argmax(1)==targets).double().mean()),'nll':float(-prob[torch.arange(len(targets)),targets].double().clamp_min(1e-12).log().mean())}


def mean_probability(members):
    slurm_only()
    import torch
    return torch.stack(members).mean(0)


def fixed_score(g,l,targets):
    return score(.5*(mean_probability(g)+mean_probability(l)),targets)


def branch_probability(members,scale,residual=True):
    if not residual:
        if scale!=1.: raise ValueError('direct classifiers require scale1')
        return mean_probability([x['delta_logits'].softmax(-1) for x in members])
    return mean_probability([(x['base_logits']+scale*x['delta_logits']).softmax(-1) for x in members])


def fused_probability(g,l,choice):
    return choice['alpha']*branch_probability(g,choice['scale_g'],choice.get('residual_g',True))+(1-choice['alpha'])*branch_probability(l,choice['scale_l'],choice.get('residual_l',True))


def calibrate(g,l,targets,residual_g=True,residual_l=True,include_zero=False):
    candidates=[]
    scales=(0.,.25,.5,1.) if include_zero else (.25,.5,1.)
    for sg,sl,alpha in product(scales if residual_g else (1.,),scales if residual_l else (1.,),(0.,.25,.5,.75,1.)):
        choice={'scale_g':sg,'scale_l':sl,'alpha':alpha,'residual_g':residual_g,'residual_l':residual_l}
        candidates.append(dict(choice,**score(fused_probability(g,l,choice),targets)))
    return min(candidates,key=lambda x:(-x['accuracy'],x['nll'],x['scale_g'],x['scale_l'],x['alpha']))


def scheme_b(train,ids):
    slurm_only()
    import torch
    pg,pl=train['logp_g'][ids].exp(),train['logp_l'][ids].exp()
    return torch.where(train['agreement'][ids,None],.5*(pg+pl),torch.where(train['choose_g'][ids,None],pg,pl))


def pb_calibrate(calibrated,pb,targets):
    rows=[{'beta':beta,**score((1-beta)*calibrated+beta*pb,targets)} for beta in (0.,.5,1.)]
    return min(rows,key=lambda r:(-r['accuracy'],r['nll'],r['beta']))
