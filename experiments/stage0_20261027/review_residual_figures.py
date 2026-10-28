"""Layout-only replacement for two Arxiv panels; frozen numerical data unchanged."""
from analyze_residual import *

def main(source,out):
    preflight()
    record=read(source/'COMPLETE.json')
    for name in ('RESULTS.json','coefficient_history.csv'):
        assert digest(source/name)==record['files'][name]
    data=read(source/'RESULTS.json');rows=data['per_cell']
    index={(r['shot'],r['seed'],r['dataset'],r['variant']):r for r in rows}
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    out.mkdir(parents=True);series=[]
    for name,control in [('Arxiv_D1_25_vs_100','d1_numeric/1.0'),('Arxiv_D1_25_vs_B0','b0')]:
        fig,axes=plt.subplots(3,5,figsize=(20,11.6))
        fig.subplots_adjust(left=.05,right=.995,bottom=.06,top=.83,hspace=.45,wspace=.29)
        for i,sh in enumerate(SHOTS):
            for j,se in enumerate(DEV):
                ax=axes[i,j];q=index[(sh,se,'arxiv','d1_numeric/0.25')];c=index[(sh,se,'arxiv',control)]
                for r,style,prefix in [(c,'--','control '),(q,'-','25% ')]:
                    for field,color in [('alpha','#D55E00'),('beta','#0072B2')]:
                        steps=[x['step'] for x in r['history']];values=[x[field] for x in r['history']]
                        selected=next(x for x in r['history'] if x['step']==r['selected_step'])
                        assert abs(selected[field]-r[field])<2e-7
                        ax.plot(steps,values,style,color=color,marker='.',ms=3,lw=1.4,label=prefix+field)
                        ax.scatter([r['selected_step']],[r[field]],edgecolors=color,
                                   facecolors=color if style=='-' else 'none',s=32,
                                   marker='o' if style=='-' else 's',zorder=4)
                        series.append(dict(figure=name,key=r['key'],variant=r['variant'],coefficient=field,steps=steps,values=values,selected_step=r['selected_step'],selected_value=r[field]))
                ax.set_title(f'{sh}-shot | seed {se}\nselected: 25%={q["selected_step"]}, control={c["selected_step"]}',fontsize=9)
                ax.set_xlabel('Training step');ax.set_ylabel('Coefficient');ax.grid(alpha=.22)
            
        label='100% internal control' if control!='b0' else 'FairRouter / B0'
        fig.suptitle(f'Arxiv numeric-retuned D1: 25% vs {label}',y=.985,fontsize=14)
        fig.text(.5,.946,'Solid =25%; dashed =control; points =actual logged steps; large circles/squares =selected checkpoint',ha='center',fontsize=10)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.926),ncol=4,frameon=False)
        fig.savefig(out/(name+'.png'),dpi=170);fig.savefig(out/(name+'.pdf'));plt.close(fig)
    write(out/'PLOTTED_SERIES.json',dict(series=series,source_results_sha256=digest(source/'RESULTS.json')))
    write(out/'REVIEW.json',dict(layout_only=True,numerical_series_unchanged=True,reason='Reserve dedicated top band so shared legend cannot overlap first-row titles',
          original_results_sha256=digest(source/'RESULTS.json'),original_history_csv_sha256=digest(source/'coefficient_history.csv'),
          original_figures={name:digest(source/'figures'/name) for name in ('Arxiv_D1_25_vs_100.png','Arxiv_D1_25_vs_100.pdf','Arxiv_D1_25_vs_B0.png','Arxiv_D1_25_vs_B0.pdf')},
          code_sha256=digest(__file__),slurm_job=os.environ['SLURM_JOB_ID']))
    seal(out,dict(all_checks_pass=True,layout_only=True,numerical_series_unchanged=True))
    print('RESIDUAL_FIGURE_REVIEW_COMPLETE',out,flush=True)
if __name__=='__main__':main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
