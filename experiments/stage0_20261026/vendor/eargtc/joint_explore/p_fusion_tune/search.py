"""Fixed finite search rules; each outer uses only its own validation."""
import hashlib
import json
from itertools import product

STAGES=('training','representation','regularization','optimization')
DEFAULT=dict(graph_width=64,text_width=64,hidden=64,dropout=.1,lr=.001,weight_decay=.0001,
    batch_size=64,max_steps=500,validate_every=50,patience=5,pseudo_weight=.5,
    pool_mode='pooled',ramp=0,warmup=0,smoothing=0.,kd_weight=0.,temperature=2.,teacher='routed',projection_lr=1.)


def adapt_previous(config):
    config=dict(config)
    width=config.pop('width')
    return {**DEFAULT,**config,'graph_width':width,'text_width':width}


def initial_configs(seed,previous=None):
    base={**DEFAULT,'seed':seed}
    anchor=dict(previous) if previous is not None else base
    if anchor['seed']!=seed:
        raise ValueError('Previous configuration seed differs')
    changes=[{'kd_weight':k,'temperature':t,'teacher':'routed'} for k,t in product((.1,.5,1.),(1.,2.,4.))]
    changes += [{'kd_weight':k,'temperature':2.,'teacher':'mean'} for k in (.1,.5,1.)]
    return [base,anchor]+[{**anchor,**change} for change in changes]


def stage_configs(stage,incumbent):
    changes={
        'representation':[{'graph_width':g,'text_width':l} for g,l in product((32,64),(64,128,256))]+[{'hidden':128}],
        'regularization':[{'dropout':0.},{'dropout':.3},{'weight_decay':.00001},{'weight_decay':.001},
                          {'smoothing':.05},{'pool_mode':'balanced'}],
        'optimization':[{'lr':.0003},{'lr':.003},{'projection_lr':.3},{'projection_lr':3.},
                        {'batch_size':128},{'max_steps':1000}]
            +[{'pseudo_weight':v} for v in (0.,.25,1.,2.)]
            +[{'kd_weight':v} for v in (0.,.1,.5,1.)]
            +[{'temperature':v} for v in (1.,2.,4.)]
            +[{'teacher':v} for v in ('routed','mean')],
    }
    return [{**incumbent,**change} for change in changes[stage]]


def identity(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]


def choose(rows):
    if not rows:
        raise ValueError('Cannot select from empty candidates')
    return min(rows,key=lambda r:(-r['accuracy'],r['nll'],r['id']))
