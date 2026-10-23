"""Finite, preregistered sequential exploration using validation alone."""
import hashlib
import json

STAGES = ('training', 'representation', 'regularization', 'optimization')
DEFAULT = dict(width=64, hidden=64, dropout=.1, lr=.001, weight_decay=.0001,
               batch_size=64, max_steps=500, validate_every=50, patience=5,
               pseudo_weight=.5, pool_mode='pooled', ramp=0, warmup=0, smoothing=0.)


def initial_configs(seed):
    base = {**DEFAULT, 'seed': seed}
    return [{**base, **change} for change in (
        {'pseudo_weight': 0.}, {'pseudo_weight': .25}, {}, {'pseudo_weight': 1.},
        {'pseudo_weight': 2.}, {'pool_mode': 'balanced'}, {'ramp': 100},
        {'warmup': 100}, {'smoothing': .05})]


def stage_configs(stage, incumbent):
    changes = {
        'representation': [{'width':16}, {'width':32}, {'hidden':128}],
        'regularization': [{'dropout':0.}, {'dropout':.3}, {'weight_decay':.00001}, {'weight_decay':.001}],
        'optimization': [{'lr':.0003}, {'lr':.003}, {'batch_size':128}, {'max_steps':1000}],
    }
    return [{**incumbent, **change} for change in changes[stage]]


def identity(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def choose(rows):
    if not rows:
        raise ValueError('Cannot select from empty candidates')
    return min(rows, key=lambda r: (-r['accuracy'], r['nll'], r['id']))
