"""One preregistered finite search over the union of historical PURF options."""
import hashlib
import itertools
import json
from .training import DEFAULTS


def cid(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]


def dedup(configs):
    return list({cid(c): c for c in configs}.values())


def from_old(config):
    c = dict(config)
    c.pop('keep')
    if 'aux' in c:
        a = c.pop('aux'); c.update(aux_gold=a, aux_pseudo=a)
    return dict(DEFAULTS, **c, agreement_ce=1.0)


def rank(rows):
    return sorted(rows, key=lambda r: (-r['accuracy'], r['nll'], r['id']))


def catalog(stage, anchor=None, eligible=False):
    if stage == 1:
        return [dict(DEFAULTS, dim=d, cross=c, lr=lr, agreement_ce=a)
                for d, c, lr, a in itertools.product((16,32,64), (0.,.25,1.), (.0003,.001,.003), (.25,.5,1.))]
    out = [dict(anchor)]
    if stage == 2:
        out += [dict(anchor, gate_lr=g, warmup=w, gate_gold=s)
                for g,w,s in itertools.product((1.,.1), (0,50), (False,True))]
        out += [dict(anchor, aux_gold=a, aux_pseudo=a) for a in (.1,.3)]
    elif stage == 3:
        for k, values in (('dim',(16,32,64)), ('cross',(0.,.25,1.)), ('lr',(.0003,.001,.003)), ('agreement_ce',(.25,.5,1.))):
            out += [dict(anchor, **{k:v}) for v in values]
        out += [dict(anchor, aux_gold=.1, aux_pseudo=.1)]
    elif stage == 4:
        for k,v in (('gate_lr',1.), ('warmup',0), ('gate_gold',False), ('cross',0.)):
            out += [dict(anchor, **{k:v})]
        out += [dict(anchor, aux_gold=a, aux_pseudo=a) for a in (0.,.1)]
    elif stage == 5:
        out += [dict(anchor, max_steps=n, lr=lr) for n,lr in itertools.product((500,1000,2000), (anchor['lr']/3,anchor['lr']))]
    elif stage == 6:
        out += [dict(anchor, max_steps=n, gate_mode='learned', gate_lr=g)
                for n,g in itertools.product((500,1000,2000),(1.,3.,10.))]
        out += [dict(anchor, max_steps=n, gate_mode='fixed', gate_lr=1.) for n in (500,1000,2000)]
    elif stage == 7:
        out += [dict(anchor, aux_gold=g, aux_pseudo=p) for g,p in itertools.product((0.,.1,.3), repeat=2)]
        out += [dict(anchor, aux_gold=a, aux_pseudo=a) for a in (.05,.5)]
    elif stage == 8:
        out += [dict(anchor, aux_gold=0.), dict(anchor, aux_pseudo=0.), dict(anchor, aux_gold=0.,aux_pseudo=0.),
                dict(anchor, gate_mode='fixed', gate_lr=1.), dict(anchor,max_steps=500)]
        if eligible:
            out += [dict(anchor,class_power=p) for p in (.25,.5)]
    elif stage == 9:
        # Revisit every structural/fusion choice after extended optimization/auxiliary selection.
        for k,values in (('dim',(16,32,64)), ('cross',(0.,.25,1.)), ('lr',(.0003,.001,.003)),
                         ('agreement_ce',(.25,.5,1.)), ('gate_lr',(.1,1.,3.,10.)), ('warmup',(0,50)), ('gate_gold',(False,True))):
            out += [dict(anchor, **{k:v}) for v in values]
    else:
        raise ValueError('Invalid stage')
    return dedup(out)
