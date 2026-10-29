"""Fig10 disagreement-scope supplement; frozen outcomes only, Slurm-only."""
from own_pgl import *

def main():
    registered();import numpy as np
    from scipy.stats import rankdata
    report=OUT/'report';check_seal(report);data=read(report/'RESULTS.json');dest=OUT/'scope-supplement';assert not dest.exists()
    dest.mkdir();write(dest/'INPUTS.json',dict(report_seal_sha256=digest(report/'COMPLETE.json'),code_sha256=digest(__file__),
        reason='Fig10 concerns disagreement precision; retain both disagreement-only and all-accepted precision, no posthoc replacement',
        cp_reference='each selected model own pGL is the primary reference',bootstrap='dataset x seed clusters, 4000 draws',model_selection_performed=False))
    rows=data['rows'];pairs=[]
    for r in rows:
        if r['treatment']=='B0':continue
        c=r['corrective'];bc=r['matched_B0_corrective'];m=r['router_metrics'];bm=r['matched_B0_router']
        assert r['corrective']['pseudo_precision']==m['deployed_precision']
        pairs.append(dict(key=r['key'],dataset=r['dataset'],shot=r['shot'],seed=r['seed'],treatment=r['treatment'],regime=r['regime'],
            accuracy_delta_pp=next(p['accuracy_delta_pp'] for p in data['pairs'] if p['key']==r['key'] and p['variant']==r['variant']),
            corrective_precision_delta_pp=100*(c['corrective_precision']-bc['corrective_precision']) if c['corrective_precision'] is not None and bc['corrective_precision'] is not None else None,
            D_accepted_precision_delta_pp=100*(m['D_deployed_precision']-bm['D_deployed_precision']) if m['D_deployed_precision'] is not None and bm['D_deployed_precision'] is not None else None,
            all_accepted_precision_delta_pp=100*(c['pseudo_precision']-bc['pseudo_precision']) if c['pseudo_precision'] is not None and bc['pseudo_precision'] is not None else None))
    def corr(x,y):
        x=x-np.nanmean(x,axis=-1,keepdims=True);y=y-np.nanmean(y,axis=-1,keepdims=True);den=np.sqrt(np.nansum(x*x,axis=-1)*np.nansum(y*y,axis=-1))
        return np.divide(np.nansum(x*y,axis=-1),den,out=np.full(den.shape,np.nan),where=den>0)
    rng=np.random.default_rng(402931);associations=[]
    for ds in ('ALL',*DS):
        for t in TREATMENTS:
            for regime in ('fixed','retuned'):
                for comparison in ('D_accepted_precision_delta_pp','all_accepted_precision_delta_pp'):
                    group=[p for p in pairs if p['treatment']==t and p['regime']==regime and (ds=='ALL' or p['dataset']==ds) and p['corrective_precision_delta_pp'] is not None and p[comparison] is not None]
                    if len(group)<3:continue
                    clusters={}
                    for i,p in enumerate(group):clusters.setdefault((p['dataset'],p['seed']),[]).append(i)
                    cs=list(clusters.values());pad=np.full((len(cs),max(map(len,cs))),-1,int)
                    for j,ids in enumerate(cs):pad[j,:len(ids)]=ids
                    draws=pad[rng.integers(len(cs),size=(4000,len(cs)))].reshape(4000,-1);bad=draws<0;draws=np.maximum(draws,0)
                    a=np.asarray([p['corrective_precision_delta_pp'] for p in group]);b=np.asarray([p[comparison] for p in group]);y=np.asarray([p['accuracy_delta_pp'] for p in group]);ax=a[draws];bx=b[draws];yy=y[draws];ax[bad]=np.nan;bx[bad]=np.nan;yy[bad]=np.nan
                    for method in ('Pearson','Spearman'):
                        aa,bb,yt,xx,zz,yb=a,b,y,ax,bx,yy
                        if method=='Spearman':aa,bb,yt=map(rankdata,(a,b,y));xx=rankdata(ax,axis=1,nan_policy='omit');zz=rankdata(bx,axis=1,nan_policy='omit');yb=rankdata(yy,axis=1,nan_policy='omit')
                        ca=corr(xx,yb);cb=corr(zz,yb);difference=ca-cb
                        def finite(x):return float(x) if np.isfinite(x) else None
                        def ci(x):
                            x=x[np.isfinite(x)];return np.quantile(x,[.025,.975]).tolist() if len(x) else None
                        associations.append(dict(dataset=ds,treatment=t,regime=regime,comparison=comparison,method=method,n=len(group),clusters=len(cs),
                            corrective_r=finite(corr(aa,yt)),precision_r=finite(corr(bb,yt)),corrective_ci95=ci(ca),precision_ci95=ci(cb),difference_ci95=ci(difference)))
    summary=[]
    for cohort in ('all75','fig10_5shot_seed42_44'):
        for ds in (('ALL',*DS) if cohort=='all75' else ('pubmed','ogbn-products')):
            for v in LABELS:
                g=[r for r in rows if r['variant']==v and (ds=='ALL' or r['dataset']==ds) and (cohort=='all75' or (r['shot']==5 and r['seed']<=44))]
                summary.append(dict(cohort=cohort,dataset=ds,variant=v,cells=len(g),accuracy=stat([r['accuracy'] for r in g]),
                    corrective_precision=stat([r['corrective']['corrective_precision'] for r in g]),own_pGL_corrective_precision=stat([r['corrective_own_pGL']['corrective_precision'] for r in g]),
                    D_accepted_precision=stat([r['router_metrics']['D_deployed_precision'] for r in g]),all_accepted_precision=stat([r['router_metrics']['deployed_precision'] for r in g]),
                    corrective_count=stat([r['corrective']['corrective_count'] for r in g]),net=stat([r['corrective']['net_corrections'] for r in g]),
                    signed_labels=stat([r['corrective']['corrective_right_minus_wrong'] for r in g])))
    effects=[]
    for ds in ('ALL',*DS):
        for t in TREATMENTS:
            g=[r for r in data['interactions'] if r['treatment']==t and (ds=='ALL' or r['dataset']==ds)]
            clusters={}
            for r in g:clusters.setdefault((r['dataset'],r['seed']),[]).append(r['tuning_accuracy_gain_pp'])
            arr=np.asarray(list(clusters.values()));means=arr[rng.integers(len(arr),size=(4000,len(arr)))].mean((1,2))
            effects.append(dict(dataset=ds,treatment=t,n=len(g),mean_pp=stat([r['tuning_accuracy_gain_pp'] for r in g])['mean'],ci95_pp=np.quantile(means,[.025,.975]).tolist()))
    historical=[r for r in rows if r['variant']=='B0' and r['seed']<=44]
    historical_cross={str(c):sum(r['cross']==c for r in historical) for c in (0.,.25,1.)}
    write(dest/'RESULTS.json',dict(summary=summary,associations=associations,tuning_effects=effects,paired_deltas=pairs,
        historical45_cross_counts=historical_cross,source_report_seal_sha256=digest(report/'COMPLETE.json'),code_sha256=digest(__file__)))
    csv_write(dest/'D_scope_paired_deltas.csv',pairs)
    csv_write(dest/'scope_correlations.csv',[{**{k:v for k,v in a.items() if not isinstance(v,list)},**{name+'_'+edge:(a[name][i] if a[name] else None) for name in ('corrective_ci95','precision_ci95','difference_ci95') for i,edge in enumerate(('lower','upper'))}} for a in associations])
    def fmt(x,scale=1):return 'NA' if x is None else f'{x*scale:.3f}'
    lines=['# 纠正性指标：分歧区、定义分母及调参效应补充','',
        '原论文Fig10是分歧区指标，因此同时展示“实际接受分歧伪标签精度”和“A+D全部接受精度”；原报告和原相关性保持封存。纠正性精度的主参照为各模型自身pGL。',
        '空纠正集合不能定义精度，均值只统计有定义配置；n列是有定义配置数/全部配置数。小样本均值不能等同节点池化精度。',
        '','| Cohort | Dataset | Setting | Accuracy% | D accepted precision% | All accepted precision% | Corrective precision% | Defined n | Corrective nodes | Net gain | Right-minus-wrong |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        cp=s['corrective_precision'];lines.append('| '+' | '.join([s['cohort'],s['dataset'],s['variant'],fmt(s['accuracy']['mean'],100),fmt(s['D_accepted_precision']['mean'],100),fmt(s['all_accepted_precision']['mean'],100),fmt(cp['mean'],100),str(cp['n'])+'/'+str(s['cells']),fmt(s['corrective_count']['mean']),fmt(s['net']['mean']),fmt(s['signed_labels']['mean'])])+' |')
    lines+=['','## 同样本相关比较','', '比较使用相同有定义配置的相对B0差值，D2的B0已匹配剩余Query。按dataset×seed簇重采样4000次，保持shot依赖；数据集内仅5簇，区间是探索性的。不显著不代表不相关。',
        '','| Dataset | Treatment | Joint | Precision comparison | Method | n | Corrective r | Precision r | Difference 95% CI |','|---|---|---|---|---|---:|---:|---:|---|']
    for a in associations:
        ci=a['difference_ci95'];ct='NA' if ci is None else '['+', '.join(fmt(x) for x in ci)+']'
        lines.append('| '+' | '.join([a['dataset'],a['treatment'],a['regime'],a['comparison'],a['method'],str(a['n']),fmt(a['corrective_r']),fmt(a['precision_r']),ct])+' |')
    lines+=['','## 调参效果（同Router，retuned−fixed）','','| Dataset | Treatment | n | Accuracy gain pp | Cluster 95% CI pp |','|---|---|---:|---:|---|']
    for e in effects:lines.append('| '+' | '.join([e['dataset'],e['treatment'],str(e['n']),fmt(e['mean_pp']),'['+', '.join(fmt(x) for x in e['ci95_pp'])+']'])+' |')
    lines+=['','历史45组B0 cross计数：'+str(historical_cross),'','重调同时开放多个Joint数值项，因此不能把全部调参收益归因cross。描述性相关也不能证明因果机制。']
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n');seal(dest,dict(all_checks_pass=True,scope_analysis_only=True));print('SCOPE_SUPPLEMENT_COMPLETE',flush=True)
if __name__=='__main__':main()
