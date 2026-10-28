"""Offline descriptive association/uncertainty; requires sealed full 3x2 report."""
from factorial_joint import *

def main():
    registered();import numpy as np
    from scipy.stats import rankdata
    report=OUT/'report';check_seal(report);data=read(report/'RESULTS.json')
    dest=OUT/'associations';assert not dest.exists();dest.mkdir()
    write(dest/'INPUTS.json',dict(report_seal_sha256=digest(report/'COMPLETE.json'),
        analysis_code_sha256=digest(__file__),replicates=4000,rng_seed=902917,
        resampling='dataset x split-seed clusters; keep shots from a cluster together',
        same_population_for_competing_correlations=True,model_selection_performed=False))
    def corr(x,y):
        x=x-np.nanmean(x,axis=-1,keepdims=True);y=y-np.nanmean(y,axis=-1,keepdims=True)
        den=np.sqrt(np.nansum(x*x,axis=-1)*np.nansum(y*y,axis=-1))
        return np.divide(np.nansum(x*y,axis=-1),den,out=np.full(den.shape,np.nan),where=den>0)
    fields=('corrective_precision_delta_pp','net_per_1000_query_delta')
    results=[];rng=np.random.default_rng(902917)
    for ds in ('ALL',*DS):
        for treatment in TREATMENTS:
            for regime in ('fixed','retuned'):
                for field in fields:
                    rows=[r for r in data['pairs'] if r['treatment']==treatment and r['regime']==regime and (ds=='ALL' or r['dataset']==ds)
                          and r[field] is not None and r['pseudo_precision_delta_pp'] is not None]
                    cluster={}
                    for j,r in enumerate(rows):cluster.setdefault((r['dataset'],r['seed']),[]).append(j)
                    if not rows:continue
                    clusters=list(cluster.values());padded=np.full((len(clusters),max(map(len,clusters))),-1,dtype=int)
                    for j,idx in enumerate(clusters):padded[j,:len(idx)]=idx
                    draws=padded[rng.integers(0,len(clusters),size=(4000,len(clusters)))].reshape(4000,-1)
                    bad=draws<0;draws=np.maximum(draws,0)
                    a=np.asarray([r[field] for r in rows]);b=np.asarray([r['pseudo_precision_delta_pp'] for r in rows]);y=np.asarray([r['accuracy_delta_pp'] for r in rows])
                    ax=a[draws];bx=b[draws];yy=y[draws];ax[bad]=np.nan;bx[bad]=np.nan;yy[bad]=np.nan
                    for method in ('Pearson','Spearman'):
                        if method=='Spearman':
                            aa=rankdata(a);bb=rankdata(b);yt=rankdata(y)
                            xx=rankdata(ax,axis=1,nan_policy='omit');zz=rankdata(bx,axis=1,nan_policy='omit');ytb=rankdata(yy,axis=1,nan_policy='omit')
                        else:aa,bb,yt,xx,zz,ytb=a,b,y,ax,bx,yy
                        ca=corr(xx,ytb);cb=corr(zz,ytb);diff=ca-cb
                        def ci(x):
                            x=x[np.isfinite(x)]
                            return dict(defined_bootstraps=len(x),ci95=np.quantile(x,[.025,.975]).tolist() if len(x) else None)
                        def finite(x):return float(x) if np.isfinite(x) else None
                        results.append(dict(dataset=ds,treatment=treatment,regime=regime,predictor=field,method=method,n=len(rows),clusters=len(clusters),
                            correlation=finite(corr(aa,yt)),overall_precision_correlation=finite(corr(bb,yt)),
                            correlation_difference=finite(corr(aa,yt)-corr(bb,yt)),
                            predictor_uncertainty=ci(ca),overall_precision_uncertainty=ci(cb),difference_uncertainty=ci(diff),
                            caveat='descriptive; dataset-specific CIs have only5 seed clusters; non-significance does not establish no relation'))
    write(dest/'RESULTS.json',dict(results=results,report_seal_sha256=digest(report/'COMPLETE.json'),analysis_code_sha256=digest(__file__)))
    csv_write(dest/'correlations_with_uncertainty.csv',[{**{k:v for k,v in r.items() if not isinstance(v,dict)},
        **{prefix+k:v for prefix,part in [('predictor_',r['predictor_uncertainty']),('overall_',r['overall_precision_uncertainty']),('difference_',r['difference_uncertainty'])] for k,v in dict(defined_bootstraps=part['defined_bootstraps'],ci95_lower=part['ci95'][0] if part['ci95'] else None,ci95_upper=part['ci95'][1] if part['ci95'] else None).items()}} for r in results])
    lines=['# 纠正性指标与最终准确率：配对相关及不确定性','',
        '全部使用相对B0的配对差值，D2匹配剔除新增训练节点后的Query。同一比较两种相关系数使用相同完整样本。按dataset×seed簇重采样4000次，保持同seed三种shot相依；数据集内只有5个seed簇，区间须谨慎解释。',
        '不显著不能证明不相关；相关更强也不等于机制已被因果识别。Joint重调同时改变多项数值参数。',
        '', '| Dataset | Treatment | Joint | Predictor | Method | n | r | Overall-precision r | Difference | Difference 95% CI |',
        '|---|---|---|---|---|---:|---:|---:|---:|---|']
    def f(x):return 'NA' if x is None else f'{x:+.3f}'
    for r in results:
        ci=r['difference_uncertainty']['ci95'];ct='NA' if ci is None else '['+', '.join(f(x) for x in ci)+']'
        lines.append('| '+' | '.join([r['dataset'],r['treatment'],r['regime'],r['predictor'],r['method'],str(r['n']),f(r['correlation']),f(r['overall_precision_correlation']),f(r['correlation_difference']),ct])+' |')
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    colors=dict(zip(DS,plt.get_cmap('tab10').colors[:5]))
    for field,caption in [('corrective_precision_delta_pp','Corrective precision delta (pp)'),('pseudo_precision_delta_pp','Overall accepted precision delta (pp)'),('net_per_1000_query_delta','Net corrections /1000 Query delta')]:
        fig,axes=plt.subplots(2,3,figsize=(15,9),squeeze=False)
        for row,regime in enumerate(('fixed','retuned')):
            for col,t in enumerate(TREATMENTS):
                ax=axes[row,col]
                for ds in DS:
                    g=[r for r in data['pairs'] if r['treatment']==t and r['regime']==regime and r['dataset']==ds and r[field] is not None]
                    ax.scatter([r[field] for r in g],[r['accuracy_delta_pp'] for r in g],s=20,alpha=.75,color=colors[ds],label=ds)
                ax.axhline(0,color='grey',lw=.7);ax.axvline(0,color='grey',lw=.7);ax.grid(alpha=.15)
                ax.set_title(t+' | Joint '+regime);ax.set_xlabel(caption);ax.set_ylabel('Final accuracy delta vs B0 (pp)')
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=5,bbox_to_anchor=(.5,.995))
        fig.subplots_adjust(top=.90,hspace=.38,wspace=.30);fig.savefig(dest/(field+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)
        lines+=['','!['+caption+']('+field+'.png)']
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n');seal(dest,dict(all_checks_pass=True,offline_association_only=True))
    print('ASSOCIATIONS_COMPLETE',flush=True)
if __name__=='__main__':main()
