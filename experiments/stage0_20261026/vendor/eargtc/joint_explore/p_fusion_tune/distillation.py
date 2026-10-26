"""Detached frozen-teacher distributions; no truth or inference blending."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def teacher_probability(logp_g, logp_l, choose_g, agreement, mode, temperature):
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    # Temper each expert before averaging. Log probabilities differ from logits
    # only by row constants, so softmax(logp / T) is exactly logit tempering.
    pg=(logp_g.detach().float()/temperature).softmax(1)
    pl=(logp_l.detach().float()/temperature).softmax(1)
    mean=(pg+pl)*.5
    if mode=='mean':
        return mean
    if mode!='routed':
        raise ValueError('Unknown teacher mode')
    chosen=torch.where(choose_g[:,None],pg,pl)
    return torch.where(agreement[:,None],mean,chosen)


def distillation_loss(logits, teacher, temperature, agreement, mode):
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    if not len(logits):
        return logits.sum()*0
    value=F.kl_div(F.log_softmax(logits/temperature,1),teacher.detach(),reduction='none').sum(1)*temperature**2
    if mode=='pooled':
        return value.mean()
    if mode!='balanced':
        raise ValueError('Unknown pseudo pool mode')
    return torch.stack([value[m].mean() for m in (agreement,~agreement) if bool(m.any())]).mean()
