"""Label-free rank policies and validation-only conservative budget selection."""
import numpy as np
from scipy.stats import rankdata

Z=1.6448536269514722


def percentile(score):
    s=np.asarray(score,float)
    if s.ndim!=1 or not np.isfinite(s).all():raise ValueError('invalid finite score vector')
    return (rankdata(s,method='average')-.5)/len(s) if len(s) else np.empty(0)


def exact_mask(score,ids,k):
    s=np.asarray(score,float);ids=np.asarray(ids)
    if s.ndim!=1 or ids.shape!=s.shape or not np.isfinite(s).all() or len(np.unique(ids))!=len(ids):raise ValueError('invalid ranking input')
    if int(k)!=k or not 0<=k<=len(s):raise ValueError('invalid K')
    m=np.zeros(len(s),bool);m[np.lexsort((ids,-s))[:int(k)]]=True
    return m


def quota_route(preference,baseline_route,ids):
    r=np.asarray(baseline_route)
    if not np.isin(r,[0,1]).all():raise ValueError('invalid route')
    return (~exact_mask(preference,ids,int((r==0).sum()))).astype(np.int64)


def allocations(kd,ka,nd,na):
    if min(kd,ka,nd,na)<0 or kd>nd or ka>na:raise ValueError('invalid original counts')
    total=kd+ka
    if total<=0 or total>nd+na:raise ValueError('invalid budget')
    lo=max(0,total-na);hi=min(nd,total)
    raw=[kd]+[int(round(total*f)) for f in (0.,.1,.25,.5,.75,1.)]+[kd-int(round(.1*total)),kd+int(round(.1*total))]
    return [(v,total-v) for v in sorted({min(hi,max(lo,v)) for v in raw})]


def lower_bound(good,n):
    if n<=0:return 0.
    p=good/n
    return float((p+Z*Z/(2*n)-Z*np.sqrt(p*(1-p)/n+Z*Z/(4*n*n)))/(1+Z*Z/n))


def branch_profile(score,correct,ids,half,k,population):
    s=np.asarray(score,float);y=np.asarray(correct,bool);ids=np.asarray(ids);half=np.asarray(half,bool)
    if s.shape!=y.shape or s.shape!=half.shape or population<=0 or not 0<=k<=population:raise ValueError('invalid branch profile')
    parts=[];valid=True
    for i,mask in enumerate((np.ones(len(s),bool),half,~half)):
        # Integer arithmetic avoids floating-point ceil errors for exact ratios.
        n=int(mask.sum());count=(int(k)*n+int(population)-1)//int(population)
        keep=exact_mask(s[mask],ids[mask],count);good=int(y[mask][keep].sum())
        supported=k==0 or count>=(10 if i==0 else 5)
        valid=valid and supported
        parts.append(dict(accepted=count,correct=good,precision=good/count if count else 0.,lower=lower_bound(good,count),supported=supported))
    return dict(k=int(k),population=int(population),valid=bool(valid),parts=parts)


def combine(dis,agr):
    total=dis['k']+agr['k']
    if total<=0:raise ValueError('empty budget')
    weights=[dis['k']/total,agr['k']/total];parts=[]
    for i in range(3):
        parts.append({key:sum(w*b['parts'][i][key] for w,b in zip(weights,(dis,agr))) for key in ('lower','precision')})
    return dict(valid=bool(dis['valid'] and agr['valid']),objective=.5*parts[0]['lower']+.5*min(p['lower'] for p in parts[1:]),full_lcb=parts[0]['lower'],half_lcb=[p['lower'] for p in parts[1:]],full_precision=parts[0]['precision'],half_precision=[p['precision'] for p in parts[1:]])


def choose(rows,baseline):
    eligible=[r for r in rows if r.get('valid',True) and all(a>=b-.005 for a,b in zip(r['half_lcb'],baseline['half_lcb']))]
    if not eligible:return baseline
    best=min(eligible,key=lambda r:(-r['objective'],-r['full_lcb'],r['complexity'],r['id']))
    return best if best['objective']>=baseline['objective']+.002 else baseline


def disagreement_candidates(base,new,winners,ids):
    # Only model outputs are accepted here. There is no label argument.
    ids=np.asarray(ids);route=np.asarray(base['route'],dtype=np.int64)
    bs=np.asarray(base['selected_score'],float);ns=np.asarray(new['selected_score'],float)
    out={'v6':dict(score=bs,route=route,complexity=0.,description='exact v6'),
         'v7':dict(score=ns,route=np.asarray(new['route'],dtype=np.int64),complexity=2.,description='exact v7 deployment')}
    br=percentile(bs);t0=percentile(base['trust_logit']);p0=percentile(base['preference_logit'])
    t1=percentile(winners['trust']);p1=percentile(winners['preference'])
    for a in (0.,.25,.5):
        for b in (0.,.25,.5):
            t=(1-a)*t0+a*t1;p=(1-b)*p0+b*p1
            rr=route.copy() if b==0 else quota_route(p,route,ids)
            confidence=t*np.where(rr==0,p,1-p)
            for residual in (.25,.5):
                cid=f'rank_t{a}_p{b}_r{residual}'
                out[cid]=dict(score=(1-residual)*br+residual*percentile(confidence),route=rr,complexity=a+b+residual,description='v6-anchored label-free cohort rank fusion; fixed GNN-route count')
    return out


def agreement_candidates(base,new):
    bs=np.asarray(base,float);ns=np.asarray(new,float)
    if bs.shape!=ns.shape:raise ValueError('unaligned agreement scores')
    out={'v6':dict(score=bs,complexity=0.),'v7':dict(score=ns,complexity=1.)}
    for alpha in (.25,.5,.75):out[f'rank_{alpha}']=dict(score=(1-alpha)*percentile(bs)+alpha*percentile(ns),complexity=alpha)
    return out


def policy_rows(dis,agr,ds,ay,dids,aids,dh,ah,counts,populations):
    kd,ka=counts;nd,na=populations;grid=allocations(kd,ka,nd,na)
    dp={};ap={}
    for count_d,count_a in grid:
        for cid,c in dis.items():dp[cid,count_d]=branch_profile(c['score'],c['route']==ds,dids,dh,count_d,nd)
        for cid,c in agr.items():ap[cid,count_a]=branch_profile(c['score'],ay,aids,ah,count_a,na)
    rows=[]
    for count_d,count_a in grid:
        for did,d in dis.items():
            for aid,a in agr.items():
                row=dict(id=f'{did}|{aid}|{count_d}|{count_a}',disagreement=did,agreement=aid,k_dis=count_d,k_agr=count_a,
                         complexity=d['complexity']+a['complexity']+abs(count_d-kd)/(kd+ka),**combine(dp[did,count_d],ap[aid,count_a]))
                rows.append(row)
    baseline=next(r for r in rows if r['disagreement']==r['agreement']=='v6' and r['k_dis']==kd)
    selected={
        'fixed_branch':choose([r for r in rows if r['k_dis']==kd],baseline),
        'allocation_only':choose([r for r in rows if r['disagreement']==r['agreement']=='v6'],baseline),
        'joint':choose(rows,baseline)}
    profiles=dict(disagreement={f'{cid}|{k}':p for (cid,k),p in dp.items()},agreement={f'{cid}|{k}':p for (cid,k),p in ap.items()})
    return rows,baseline,selected,profiles
