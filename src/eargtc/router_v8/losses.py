"""Proper domain-balanced G/L/N likelihood; N never supervises preference."""
import torch
from torch.nn import functional as F


def loss_weights(state, domain, treatment='proper'):
    weight = torch.zeros_like(state,dtype=torch.float32)
    for d in torch.unique(domain):
        mask = domain == d
        z = state[mask]
        w = torch.ones_like(z,dtype=torch.float32)
        if treatment == 'sqrt-balanced':
            counts = torch.bincount(z,minlength=3).float()
            class_weight = (len(z)/(3*counts.clamp_min(1))).sqrt().clamp(.5,4.)
            w = class_weight[z]
        elif treatment != 'proper':
            raise ValueError(treatment)
        weight[mask] = w / w.sum()
    return weight / torch.unique(domain).numel()


def factorized_loss(c, r, state, domain=None, treatment='proper', weights=None):
    if weights is None:
        if domain is None:
            domain = torch.zeros_like(state)
        weights = loss_weights(state, domain, treatment)
    resolvable = state != 2
    trust = F.binary_cross_entropy_with_logits(c,resolvable.to(c.dtype),reduction='none')
    preference = F.binary_cross_entropy_with_logits(r,(state==0).to(r.dtype),reduction='none')
    return ((trust+preference*resolvable)*weights).sum()
