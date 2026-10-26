"""Preregistered study matrices and deterministic bounded phase construction."""
from __future__ import annotations
import itertools
from .manifest import digest

DATASETS=('cora','citeseer','pubmed','arxiv','ogbn-products')
OUTERS=(42,43,44,45,46)
INNERS=(42,43,44)
ARCHS=('raw_concat','projected_concat','projected_interaction','projected_sum','projected_gate','raw_depth','projected_concat_wide')
COARSE=((64,1,.1,.001,.0001),(32,.2,0,.001,.0001),(128,.2,.1,.001,.0001),(64,.05,.1,.001,.0001),(64,2,.1,.001,.0001),(64,.2,.5,.001,.0001),(64,1,0,.001,.0001),(64,.2,.1,.0003,.0001),(64,.2,.1,.003,.0001),(64,1,.1,.001,0),(64,1,.1,.001,.001),(128,.05,0,.0003,.001))
GRIDS={'pseudo_weight':[.02,.05,.2,.5,1,2],'keep_weight':[0,.03,.1,.3,.5],'lr':[.0001,.0003,.001,.003],'width':[32,64,128],'weight_decay':[0,.0001,.001],'warm_fraction':[.125,.25,.5],'dropout':[.05,.1,.2],'clip_norm':[1,5,10]}
COORDINATES=tuple(GRIDS)
PHASES=('A','B','C','H00_coarse')+tuple(f'H{i:02d}_{key}' for i,key in enumerate(COORDINATES,1))+('H09_budget','H10_anchor','H11_replicate','FINAL')


def anchor(**changes):
    cfg=dict(architecture='raw_concat',width=64,dropout=0.,lr=.001,weight_decay=.0001,pseudo_weight=1.,keep_weight=.1,updates=160,warm_updates=0,validate_every=10,chunk_size=512,gold_sampling='full_coverage',pseudo_sampling='full_coverage',legacy_draws=64,disagreement_all=True,agreement_ce=True,residual=True,clip_norm=None,teacher='original',gold_only_control=False,inference_batch=4096,threads=2)
    cfg.update(changes)
    return cfg


def config_digest(cfg):
    if set(cfg)!=set(anchor()):
        raise ValueError('configuration must contain exactly training fields, without trial IDs')
    return digest(cfg)


def trial(name,cfg,eligible=True,**extra):
    return dict(id=name,config=cfg,eligible=eligible,**extra)


def stage_a():
    out=[trial(f'D{d}A{a}S{s}',anchor(disagreement_all=bool(d),agreement_ce=bool(a),pseudo_sampling='full_coverage' if s else 'full_replacement')) for d,a,s in itertools.product(range(2),repeat=3)]
    out += [trial('legacy64',anchor(disagreement_all=False,agreement_ce=False,gold_sampling='legacy64_replacement',pseudo_sampling='legacy64_replacement')),trial('goldfull_pseudo64',anchor(disagreement_all=False,agreement_ce=False,pseudo_sampling='legacy64_replacement')),trial('gold_only160',anchor(pseudo_weight=0.,keep_weight=0.,gold_only_control=True))]
    return out


def stage_b():
    return [trial(a,anchor(architecture=a)) for a in ARCHS]


def stage_c(architecture):
    cfg=anchor(architecture=architecture)
    settings=(('cold160',{}),('warm40',{'warm_updates':40}),('warm80',{'warm_updates':80}),('residual_off',{'residual':False}),('clip5',{'clip_norm':5.}),('dropout01',{'dropout':.1}),('warm40_warmteacher',{'warm_updates':40,'teacher':'warm'}),('cold120_diagnostic',{'updates':120}),('warm40_dropout01',{'warm_updates':40,'dropout':.1}),('warm40_clip5',{'warm_updates':40,'clip_norm':5.}))
    return [trial(name,dict(cfg,**values),name!='cold120_diagnostic') for name,values in settings]


def with_budget(cfg,budget):
    warm=round(cfg['warm_updates']/cfg['updates']*budget)
    return dict(cfg,updates=budget,warm_updates=warm)


def coarse_trials(cfg):
    return [trial(f'coarse{i:02d}',dict(with_budget(cfg,240),**dict(zip(('width','pseudo_weight','keep_weight','lr','weight_decay'),values)))) for i,values in enumerate(COARSE)]


def adjacent_trials(cfg,coordinate):
    if (coordinate=='warm_fraction' and not cfg['warm_updates']) or (coordinate=='dropout' and not cfg['dropout']) or (coordinate=='clip_norm' and cfg['clip_norm'] is None):
        return []
    grid=GRIDS[coordinate]
    current=cfg['warm_updates']/cfg['updates'] if coordinate=='warm_fraction' else cfg[coordinate]
    if current not in grid:
        raise ValueError('incumbent is not on preregistered coordinate grid')
    i=grid.index(current)
    values=[grid[j] for j in (i-1,i+1) if 0<=j<len(grid)]
    return [trial(f'{coordinate}_{value}',dict(cfg,**({'warm_updates':round(value*cfg['updates'])} if coordinate=='warm_fraction' else {coordinate:value}))) for value in values]


def neighborhood(pair):
    transforms={
        'selected':{},'raw_concat_block':{'architecture':'raw_concat'},'projected_concat_fusion':{'architecture':'projected_concat'},
        'warm_off':{'warm_updates':0,'teacher':'original'},'teacher_original':{'teacher':'original'},'residual_toggle':{},
        'KL_off':{'keep_weight':0.},'replacement':{'pseudo_sampling':'full_replacement'},'D_off':{'disagreement_all':False},'A_off':{'agreement_ce':False},
    }
    out={}
    for name,change in transforms.items():
        if name=='projected_concat_fusion' and not any(c['architecture'] in ('projected_interaction','projected_sum','projected_gate','projected_concat_wide') for c in pair.values()):
            continue
        out[name]={branch:dict(cfg,**({'residual':not cfg['residual']} if name=='residual_toggle' else {} if name=='projected_concat_fusion' and cfg['architecture'] not in ('projected_interaction','projected_sum','projected_gate','projected_concat_wide') else change)) for branch,cfg in pair.items()}
    return out


def make_plan(stage,trials,parents,inners=None):
    plan={'stage':stage,'trials':trials,'parents':parents,'outers':list(OUTERS),'inners':list(INNERS if stage in ('H11_replicate','FINAL') else ([42] if inners is None else inners))}
    if stage=='FINAL':
        plan['outers']=list(OUTERS)
    return plan


def expected_grid(plan):
    return [(ds,outer,t['id'],inner) for ds,ts in plan['trials'].items() for t in ts for outer in t.get('outers',plan['outers']) for inner in plan['inners']]


def validate_request(plan,dataset,outer):
    if dataset not in plan['trials'] or outer not in plan['outers']:
        raise ValueError('dataset/outer is outside the registered phase grid')


def rank_rows(rows,order):
    ranks={name:i for i,name in enumerate(order)}
    if any(r['id'] not in ranks for r in rows):
        raise ValueError('unknown candidate ID')
    return sorted(rows,key=lambda r:(-r['accuracy'],r['nll'],ranks[r['id']]))


def cell_key(dataset,outer):
    return f'{dataset}/{outer}'


def cell_trials(plan,dataset,outer):
    return [t for t in plan['trials'][dataset] if outer in t.get('outers',plan['outers'])]


def independent_trials(builder):
    return {ds:[dict(t,outers=[outer]) for outer in OUTERS for t in builder(ds,outer)] for ds in DATASETS}
