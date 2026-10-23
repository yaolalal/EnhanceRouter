"""Single-branch absolute-threshold selection: no budget or other-branch inputs."""
import numpy as np
from scipy.special import expit
QUANTILES=tuple([i/20 for i in range(19)]+[.925,.95,.975])
Z=1.6448536269514722

def accept(score,threshold):
    x=np.asarray(score,float)
    if x.ndim!=1 or not np.isfinite(x).all() or not np.isfinite(threshold):raise ValueError('nonfinite or non-vector threshold input')
    return x>=threshold

def lower(good,n):
    if not n:return 0.
    p=good/n
    return float((p+Z*Z/(2*n)-Z*np.sqrt(p*(1-p)/n+Z*Z/(4*n*n)))/(1+Z*Z/n))

def profile(score,correct,half,threshold):
    keep=accept(score,threshold);y=np.asarray(correct,bool);half=np.asarray(half,bool)
    if keep.shape!=y.shape or y.shape!=half.shape:raise ValueError('unaligned branch validation')
    parts=[]
    for mask in (np.ones(len(y),bool),half,~half):
        m=keep&mask;n=int(m.sum());good=int(y[m].sum())
        parts.append(dict(accepted=n,correct=good,precision=good/n if n else 0.,lower=lower(good,n)))
    valid=parts[0]['accepted']>=20 and all(p['accepted']>=10 for p in parts[1:])
    return dict(valid=valid,parts=parts,objective=.5*parts[0]['lower']+.5*min(p['lower'] for p in parts[1:]),worst_precision=min(p['precision'] for p in parts[1:]))

def threshold_trials(recipe,score,correct,half,complexity):
    x=np.asarray(score,float);accept(x,0.)
    if not len(x):raise ValueError('empty validation branch')
    thresholds=sorted(set(float(t) for t in np.quantile(x,QUANTILES)))
    return [dict(recipe=recipe,threshold=t,complexity=float(complexity),**profile(x,correct,half,t)) for t in thresholds]

def select(rows):
    eligible=[r for r in rows if r['valid']]
    if not eligible:raise ValueError('insufficient validation support; never disable branch')
    # Rounded objective removes irrelevant machine epsilon differences at exact ties.
    return min(eligible,key=lambda r:(-round(r['objective'],12),-r['parts'][0]['precision'],-r['parts'][0]['accepted'],r['complexity'],r['recipe'],r['threshold']))

def dis_scores(trust,preference,cfg):
    t=np.asarray(trust,float);p=np.asarray(preference,float)
    if t.shape!=p.shape:raise ValueError('unaligned heads')
    tt=expit(cfg['ts']*t+cfg['tb']);pp=expit(cfg['ps']*p+cfg['pb']);route=(pp<.5).astype(np.int64)
    q=np.column_stack((tt*pp,tt*(1-pp),1-tt))
    return dict(score=q[np.arange(len(t)),route],route=route,q=q)

def strict_shortlist(rows,available,n):
    valid=[r for r in rows if r['id']!='incumbent' and r['id'] in available]
    ordered=sorted(valid,key=lambda r:(-r['ranking']['robust'],-(r['ranking']['auc'] if r['ranking']['auc'] is not None else .5),r['id']))
    if not ordered:raise ValueError('no strict v7 checkpoint available')
    return [r['id'] for r in ordered[:n]]

def dis_grid():
    return [dict(ts=a,tb=b,ps=c,pb=d) for a in (.5,1.,2.) for b in (-1.,0.,1.) for c in (.5,1.,2.) for d in (-1.,-.5,0.,.5,1.)]
