"""Prespecified summaries, split-cluster paired CIs and empirical variance."""
from experiment import *
import statistics
METRICS=('full_accuracy','pR_accuracy','pGL_own_accuracy','pGL_B0_accuracy','pR_contribution','rescue_rate','harm_rate',
 'S_fraction','S_pR_accuracy','S_pGL_accuracy','S_full_accuracy','head_pR_accuracy','tail_pR_accuracy',
 'prediction_entropy','prediction_normalized_entropy','prediction_class_count','prediction_max_class_fraction',
 'selected_step','selected_step0','stop_step','alpha','beta','N','actual_precision','teach_g_count','teach_l_count',
 'teach_g_precision','teach_l_precision','correct_mean_degree','wrong_mean_degree','correct_mean_margin','wrong_mean_margin','equal_pool_precision')
RATE_METRICS=set(METRICS[:13])|{'prediction_max_class_fraction','prediction_normalized_entropy','actual_precision','teach_g_precision','teach_l_precision','equal_pool_precision','selected_step0'}
FILTERS=('all','no_step0','no_cross_zero_cell','no_step0_no_cross_zero_cell')
def eligible(r,fl):
 return ('no_step0' not in fl or not r['selected_step0']) and ('no_cross_zero_cell' not in fl or not r['cell_B0_cross_zero'])
def stats(a):
 a=[float(x) for x in a if x is not None]
 return dict(n=len(a),mean=statistics.mean(a) if a else None,std=statistics.pstdev(a) if a else None)
def summarize(rows,ds,rule,mode,fl,seed=None):
 return dict(dataset=ds,rule=rule,selection_mode=mode,filter=fl,split_seed=seed,runs=len(rows),splits=len({r['split_seed'] for r in rows}),
  **{m+'_'+k:v for m in METRICS for k,v in stats([r[m] for r in rows]).items()})
def pairs(rows):
 import numpy as np
 by={(r['dataset'],r['split_seed'],r['selection_mode'],r['rule'],r['noise_seed'],r['training_seed']):r for r in rows}
 effects=[];detail=[]
 for ds in DS:
  for mode in ('best','last500'):
   targets=(*RULES[1:],'A0','A100') if mode=='best' else ('A0','A100')
   for rule in targets:
    for fl in FILTERS:
     pp=[]
     for se in SEEDS:
      for no in (0,1,2):
       for tr in (0,1):
        ref=by[ds,se,mode,'R1',no,tr];cur=by[ds,se,mode,rule,None if rule.startswith('A') else no,tr]
        if not eligible(ref,fl) or not eligible(cur,fl):continue
        d=dict(dataset=ds,split_seed=se,selection_mode=mode,rule=rule,noise_seed=no,training_seed=tr,filter=fl,reference_run=ref['run_id'],current_run=cur['run_id'])
        for m in METRICS:
         a,b=cur[m],ref[m];d[m]=None if a is None or b is None else float(a)-float(b)
        pp.append(d);detail.append(d)
     for m in METRICS:
      clusters=[]
      for se in SEEDS:
       values=[p[m] for p in pp if p['split_seed']==se and p[m] is not None]
       if values:clusters.append(float(np.mean(values)))
      n=len(clusters);avg=float(np.mean(clusters)) if n else None
      if n>=2:
       rng=np.random.default_rng(20261031);ind=rng.integers(0,n,size=(10000,n));boot=np.asarray(clusters)[ind].mean(1);lo,hi=(float(x) for x in np.quantile(boot,[.025,.975]))
      else:lo=hi=None
      effects.append(dict(dataset=ds,selection_mode=mode,rule=rule,reference='R1',filter=fl,metric=m,paired_runs=sum(p[m] is not None for p in pp),split_clusters=n,mean_difference=avg,CI95_low=lo,CI95_high=hi))
 return effects,detail
def variance(rows):
 import numpy as np
 out=[]
 for ds in DS:
  for mode in ('best','last500'):
   g=[r for r in rows if r['dataset']==ds and r['rule']=='R1' and r['selection_mode']==mode];assert len(g)==30
   y=np.empty((5,3,2))
   for r in g:y[SEEDS.index(r['split_seed']),r['noise_seed'],r['training_seed']]=100*r['full_accuracy']
   total=float(np.var(y));split=float(np.var(y.mean((1,2))));noise=float(np.var(y.mean(2),axis=1).mean());train=float(np.var(y,axis=2).mean())
   assert abs(total-split-noise-train)<1e-9
   mu=y.mean();s=y.mean((1,2))-mu;n=y.mean((0,2))-mu;t=y.mean((0,1))-mu
   sn=y.mean(2)-mu-s[:,None]-n[None,:];st=y.mean(1)-mu-s[:,None]-t[None,:];nt=y.mean(0)-mu-n[:,None]-t[None,:]
   snt=y-mu-s[:,None,None]-n[None,:,None]-t[None,None,:]-sn[:,:,None]-st[:,None,:]-nt[None,:,:]
   crossed={k:float(np.mean(v*v)) for k,v in [('split_main',s),('noise_main',n),('train_main',t),('split_noise',sn),('split_train',st),('noise_train',nt),('split_noise_train',snt)]}
   assert abs(total-sum(crossed.values()))<1e-9
   out.append(dict(dataset=ds,selection_mode=mode,runs=30,total_variance_pp2=total,split_variance_pp2=split,noise_within_split_variance_pp2=noise,train_within_split_noise_variance_pp2=train,
    split_share=split/total if total else None,noise_share=noise/total if total else None,train_share=train/total if total else None,
    **{k+'_pp2':v for k,v in crossed.items()},point_variance='not identifiable: one point seed',method='finite balanced law of total variance split->noise->train; crossed ANOVA interactions separately'))
 return out

def aggregate():
 registered();assert not PUB.exists();rows=[];proof=[];baseline=[];pre=read(RUN/'PREFLIGHT_PASS.json')['rows']
 for i,(seed,ds) in enumerate(CELLS):
  p=RUN/'models'/f'{i:02d}';c=read(p/'CELL_COMPLETE.json');assert c['runs']==len(PLAN) and c['files']==files()
  for plan in PLAN:
   trial=p/run_id(plan);check(trial);r=read(trial/'RESULTS.json');assert (r['dataset'],r['split_seed'])==(ds,seed);rows.append(r)
  proof.append(dict(index=i,seal_sha256=sha(p/'CELL_COMPLETE.json')))
  ctx=read(cellpath(seed,ds)/'CONTEXT.json');baseline.append(dict(dataset=ds,split_seed=seed,**ctx['B0_metrics'],selected_step=ctx['B0_replay']['selected']['step']))
 assert len(rows)==1850 and len({r['run_id'] for r in rows})==1850
 summaries=[];split_summaries=[]
 for ds in DS:
  for mode in ('best','last500'):
   rules=(*RULES,'A0','A100') if mode=='best' else ('R1','A0','A100')
   for rule in rules:
    for fl in FILTERS:
     g=[r for r in rows if r['dataset']==ds and r['selection_mode']==mode and r['rule']==rule and eligible(r,fl)]
     summaries.append(summarize(g,ds,rule,mode,fl))
     for se in SEEDS:split_summaries.append(summarize([r for r in g if r['split_seed']==se],ds,rule,mode,fl,se))
 effects,pair_detail=pairs(rows);var=variance(rows)
 lookup={(r['dataset'],r['split_seed'],r['rule'],r['noise_seed'],r['training_seed'],r['selection_mode']):r for r in rows};fixed_comparison=[]
 for r in rows:
  if r['selection_mode']!='last500':continue
  b=lookup[r['dataset'],r['split_seed'],r['rule'],r['noise_seed'],r['training_seed'],'best']
  fixed_comparison.append(dict(dataset=r['dataset'],split_seed=r['split_seed'],rule=r['rule'],noise_seed=r['noise_seed'],training_seed=r['training_seed'],best_run=b['run_id'],fixed_run=r['run_id'],best_selected_step=b['selected_step'],best_stop_step=b['stop_step'],best_step0=b['selected_step0'],best_accuracy=b['full_accuracy'],last500_accuracy=r['full_accuracy'],difference_pp=100*(r['full_accuracy']-b['full_accuracy'])))
 def table(h,rr):return '\n'.join(['| '+' | '.join(h)+' |','|'+'|'.join('---' for _ in h)+'|',*['| '+' | '.join(map(str,r))+' |' for r in rr]])
 def val(r,m,scale=100):
  a,b=r.get(m+'_mean'),r.get(m+'_std');return 'NA' if a is None else f'{scale*a:.2f} ± {scale*b:.2f}'
 lines=['# 精度可控合成伪标签：Joint/PURF诊断实验','',
  '**这是使用真值构造标签的诊断实验，不是方法，不用于模型选择或方法接纳。**',
  '**eval1000全部排除在合成池之外，gold/validation也排除；最终分类仍用原eval1000及原顺序。**',
  '**Joint超参冻结对应B0，keep强制0；非A0遇B0 cross=0强制为1并标注，A0始终cross=keep=0。**','',
  '## 观察','',f'全部1850个训练运行完成：主版本1600、固定500步版本250；另有25格B0逐节点复现门槛全部通过。主版本step0数量：{sum(r["selected_step0"] for r in rows if r["selection_mode"]=="best")}；B0 cross为0的格数：{sum(r["B0_cross"]==0 for r in pre)}。',
  '', '### 候选池preflight（不足1000正常取全部）','',
  table(['dataset','split','候选数','GNN正确','LLM正确','N','K','n_capped','B0 cross'],[[r[k] for k in ('dataset','split_seed','candidate_count','candidate_GNN_correct','candidate_LLM_correct','N','K','n_capped','B0_cross')] for r in pre]),
  '', '### 按数据集集中展示','']
 for ds in DS:
  bg=[r for r in baseline if r['dataset']==ds]
  lines+=['','## '+ds,'',f'引用已封存B0：完整accuracy {100*statistics.mean(r["full_accuracy"] for r in bg):.2f} ± {100*statistics.pstdev(r["full_accuracy"] for r in bg):.2f}%；pR {100*statistics.mean(r["pR_accuracy"] for r in bg):.2f}%；固定B0 pGL {100*statistics.mean(r["pGL_B0_accuracy"] for r in bg):.2f}%。','']
  for mode,title in [('best','原validation checkpoint选择'),('last500','固定500步、最后一步')]:
   ss=[r for r in summaries if r['dataset']==ds and r['selection_mode']==mode and r['filter']=='all']
   lines+=['','### '+title,'',table(['规则','n','完整accuracy%','pR%','自身pGL%','固定B0 pGL%','pR贡献pp','救援率%','误伤率%','step0比例%'],[[r['rule'],r['runs'],*[val(r,m) for m in ('full_accuracy','pR_accuracy','pGL_own_accuracy','pGL_B0_accuracy','pR_contribution','rescue_rate','harm_rate','selected_step0')]] for r in ss]),
    '',table(['规则','S比例%','S内pR%','S内pGL%','S内完整%','head pR%','tail pR%','硬预测熵nat','出现类别数','最大类比例%'],[[r['rule'],*[val(r,m) for m in ('S_fraction','S_pR_accuracy','S_pGL_accuracy','S_full_accuracy','head_pR_accuracy','tail_pR_accuracy')],val(r,'prediction_entropy',1),val(r,'prediction_class_count',1),val(r,'prediction_max_class_fraction')] for r in ss])]
   pp=[r for r in effects if r['dataset']==ds and r['selection_mode']==mode and r['filter']=='all' and r['metric']=='full_accuracy']
   lines+=['','与R1配对的完整分类差值（按split簇重采样95%CI）：','',table(['规则−R1','配对次数','split簇数','差值pp','95% CI pp'],[[r['rule'],r['paired_runs'],r['split_clusters'],'NA' if r['mean_difference'] is None else f'{100*r["mean_difference"]:+.2f}','NA' if r['CI95_low'] is None else f'[{100*r["CI95_low"]:+.2f}, {100*r["CI95_high"]:+.2f}]'] for r in pp])]
  lines+=['','### 剔除第0步、剔除强制cross格后的主版本','',table(['筛选','规则','n','split数','完整accuracy%','pR%','pR贡献pp'],[[r['filter'],r['rule'],r['runs'],r['splits'],val(r,'full_accuracy'),val(r,'pR_accuracy'),val(r,'pR_contribution')] for r in summaries if r['dataset']==ds and r['selection_mode']=='best' and r['filter']!='all'])]
 lines+=['','## R1方差分解','',
  '单位pp²。主分解按split→noise→train的条件总方差公式，三项精确相加为总体经验方差。noise项含split×noise交互，train项含涉及训练seed的交互；不是三个独立随机效应的无偏估计。variance.csv另给七项正交有限ANOVA分解用于核查交互。选点seed只有0，不能估计选点方差。','',
  table(['dataset','版本','总方差','split方差','split内noise方差','split/noise内train方差'],[[r['dataset'],r['selection_mode'],*[f'{r[k]:.5f}' for k in ('total_variance_pp2','split_variance_pp2','noise_within_split_variance_pp2','train_within_split_noise_variance_pp2')]] for r in var]),
  '', '## 解读与计算口径','',
  '- 十规则精度均为K/N，约80%，且相同dataset/split/选点seed的节点集合完全相同。R2/R3度降/升；R4/R5固定B0 pGL margin降/升；R7/R8 LLM置信度降/升；R9/R10 GNN置信度降/升。噪声seed只用于R1、R6和并列打破，不改变选点。',
  '- 均值±标准差跨split×noise×train的唯一运行；R规则每dataset主表n30，锚点n10，不把同一锚点复制成独立试验。split层均值在split_summary.csv。',
  '- 配对先匹配同split/noise/train，再每split平均；锚点与各R1噪声比较但不增加独立锚点数量。CI对split簇10000次bootstrap。仅5个split簇，区间是描述性诊断。剔除step0的配对要求双方均非step0。',
  '- pR贡献=完整accuracy−固定B0系数pGL accuracy。S、救援率、误伤率同样以固定B0 pGL为参照，另列自身pGL准确率。救援率分母为pGL答错节点，误伤率分母为pGL答对节点。',
  '- head/tail用完整图真实类别频率中位数；head>中位数，tail<=中位数，集合只用于离线评估。预测熵为eval1000硬预测类别分布的自然对数熵，不是平均softmax熵。空集合指标为NA，汇总保留有效n。',
  '- 训练seed是独立控制id0/1，RNG基数=split+1000003×id；id0逐节点重放B0通过。split本身不变。',
  '- 用户指定的synthetic teach_g表示选择GNN，teach_l表示选择LLM。原B0历史pool名表示被修复对象，命名相反，B0复现保留原命名以保持RNG行为。synthetic按用户定义分池；损失仍两个独立等大小batch，各cross/2，不按池大小重新加权。',
  '- 因两池比例随标签分配改变，固定总精度80%不保证等池batch精度相同。per_run.csv记录两池数量、精度、equal_pool_precision和pseudo_loss_mass_multiplier；一个池为空时保留原半权重，不补选或加倍。',
  '- 原B0训练池可能包含eval1000节点的专家伪标签；本诊断明确排除。B0作为已封存锚点引用，不把其与合成池差值当作单因素纯因果效应。',
  '- 不据单个seed反例接纳任何规则；第0步、cross强制、候选capped全部分列。stage1及历史预算争议保持暂缓。','',
  '## 文件与代码','',
  '- per_run.csv：1850次运行一行，含所有数值指标、训练步数、alpha/beta、池大小/精度、源路径及哈希。summary.csv和split_summary.csv含全部过滤版本。',
  '- paired_effects.csv / paired_run_differences.csv：各指标配对差值和簇CI。variance.csv：层级方差及完整交互分解。fixed500_vs_best.csv：250份固定步数与对应best配对。',
  '- diagnostics.csv：第0步、cross强制或capped运行；preflight.csv为全部25格候选池与B0门槛。B0_reference.csv为已封存B0引用指标。',
  '- evidence/{cell}/{run}/pool.csv和eval1000.csv：每次运行的池名单/标签/专家/度/margin/confidence，以及eval1000的pR/自身pGL/固定pGL/pJ标签。',
  '- RESULTS.json：全部结果、源哈希、封存证明；checkpoint二进制和完整图缓存仍在服务器，路径/哈希逐运行可查。',
  '- 代码：experiments/synthetic_pseudo_20261031/experiment.py、training.py、report.py；训练数学来源与变化见TRAINING_PROVENANCE.json。','']
 PUB.mkdir(parents=True)
 flat=[]
 for r in rows:flat.append({k:v for k,v in r.items() if not isinstance(v,(dict,list))})
 csvout(PUB/'per_run.csv',flat);csvout(PUB/'summary.csv',summaries);csvout(PUB/'split_summary.csv',split_summaries)
 csvout(PUB/'paired_effects.csv',effects);csvout(PUB/'paired_run_differences.csv',pair_detail);csvout(PUB/'variance.csv',var);csvout(PUB/'fixed500_vs_best.csv',fixed_comparison)
 csvout(PUB/'preflight.csv',pre);csvout(PUB/'B0_reference.csv',[{k:v for k,v in r.items() if not isinstance(v,list)} for r in baseline])
 diagnostic=[r for r in flat if r['selected_step0'] or r['cross_forced'] or r['n_capped']];csvout(PUB/'diagnostics.csv',diagnostic,list(flat[0]))
 for i in range(25):
  src=RUN/'models'/f'{i:02d}'
  for plan in PLAN:
   target=PUB/'evidence'/f'{i:02d}'/run_id(plan);target.mkdir(parents=True)
   for name in ('CONFIG.json','RESULTS.json','pool.csv','eval1000.csv'):shutil.copy2(src/run_id(plan)/name,target/name)
 write(PUB/'RESULTS.json',dict(all_checks_pass=True,diagnostic_uses_truth=True,models=1850,B0_replays=25,per_run=rows,summary=summaries,split_summary=split_summaries,paired_effects=effects,variance=var,B0_reference=baseline,preflight=pre,proof=proof,registration=read(RUN/'REGISTRATION.json'),environment_sha256=sha(RUN/'environment.json'),files=files()))
 (PUB/'REPORT.md').write_text('\n'.join(lines));shutil.copy2(RUN/'REGISTRATION.json',PUB/'REGISTRATION.json');shutil.copy2(RUN/'environment.json',PUB/'environment.json')
 write(PUB/'PUBLICATION_SHA256.json',{str(p.relative_to(PUB)):sha(p) for p in sorted(PUB.rglob('*')) if p.is_file()})
 write(PUB/'COMPLETE.json',dict(all_checks_pass=True,models=1850,B0_replays=25,report_sha256=sha(PUB/'REPORT.md'),manifest_sha256=sha(PUB/'PUBLICATION_SHA256.json'),source_files=files(),slurm_job=os.environ['SLURM_JOB_ID']))
 import tarfile
 asset=ROOT/'runs/synthetic_pseudo_20261031-results.tar.gz'
 assert not asset.exists()
 with tarfile.open(asset,'w:gz') as t:t.add(PUB,arcname=str(PUB.relative_to(ROOT)));t.add(HERE,arcname=str(HERE.relative_to(ROOT)))
 write(RUN/'PUBLICATION.json',dict(all_checks_pass=True,models=1850,archive=str(asset),archive_bytes=asset.stat().st_size,archive_sha256=sha(asset),report_sha256=sha(PUB/'REPORT.md')))
 print('SYNTHETIC_REPORT_PASS',1850,flush=True)
