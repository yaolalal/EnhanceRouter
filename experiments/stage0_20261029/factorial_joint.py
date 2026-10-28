"""Preregistered 3x2 Router treatment x Joint tuning; all entrypoints Slurm-only."""
from pathlib import Path
import sys,os,json,hashlib,copy,csv,gc,statistics,math,subprocess
THIS=Path(__file__).resolve().parent
sys.path.insert(0,str(THIS.parent/'stage0_20261027'))
from report_full_query import preflight,router_metrics,validate_bank,score_files,id_hash
from d3_recovery import *
from analyze_residual import Sources,inputs,csv_write
from d1_numeric import native_joint,joint_locked,JOINT_NUMERIC
OUT=ROOT/'runs/joint-factorial-3x2-20261029-v1'
TREATMENTS={'D1_25':'d1_numeric/0.25','D2a_1000':'d2a/1000','D4_confidence':'d4/confidence_router'}
ARMS=(('D1_25','fixed'),('D2a_1000','fixed'),('D2a_1000','retuned'),('D4_confidence','retuned'))
LABELS=('B0',*[t+'__'+r for t in TREATMENTS for r in ('fixed','retuned')])
PILOT=(0,3,4,18,75,78,79,93,150,153,154,168,225,228,229,243)

def check_seal(path,names=None):
    record=read(path/'COMPLETE.json')
    for name,h in record['files'].items():
        if names is None or name in names:assert digest(path/name)==h,(str(path),name)
    return record

def code_files():
    paths=[THIS/'AGENTS.md',THIS/'job.sh',THIS/'associations.py',Path(__file__),
           THIS.parent/'stage0_20261027/d1_numeric.py',THIS.parent/'stage0_20261027/analyze_residual.py',
           THIS.parent/'stage0_20261027/report_full_query.py']
    return {str(p.resolve().relative_to(ROOT)):digest(p) for p in paths}

def register():
    preflight();assert not (OUT/'REGISTRATION.json').exists()
    rows=[]
    for i,(sh,se,ds) in enumerate(CELLS):
        base=folder(sh,se,ds);sources={}
        for name in ('b0',*TREATMENTS.values()):
            p=base/name;check_seal(p)
            sources[name]=dict(path=str(p),seal_sha256=digest(p/'COMPLETE.json'))
        cfg=read(base/'b0/config.json')['joint'];num=read(base/'d1_numeric/0.25/config.json')
        assert joint_locked(num['joint'])==joint_locked(cfg)
        d4=read(base/'d4/confidence_router/config.json')['parameters']['joint_config'];assert d4==cfg
        diag=read(base/'d2a/1000/diagnostics.json');actual=diag['actual_N']
        parents=read(base/'d2a/1000/config.json')['parameters']['parents']
        assert actual==len(parents)==len(set(parents)) and actual<=1000
        assert not set(parents)&set(evaluation_ids(se,ds))
        rows.append(dict(index=i,key=key(sh,se,ds),sources=sources,B0_joint=cfg,
                         d2_actual_N=actual,d2_added_parent_ids=parents))
        print('REGISTER_CELL',i,key(sh,se,ds),actual,flush=True)
    write(OUT/'REGISTRATION.json',dict(schema=1,rows=rows,cells=75,seeds=list(DEV),shots=list(SHOTS),
        files=code_files(),treatments=TREATMENTS,new_arms=ARMS,labels=LABELS,
        Joint_numeric_fields=sorted(JOINT_NUMERIC),reuse=['B0','D1_25__retuned','D4_confidence__fixed'],
        pseudo_pool_pairing='identical Router outputs within treatment; only Joint numerical search differs',
        gold='original full support only, including D2; no extra gold sent to Joint',
        pGL_primary='B0 final selected checkpoint alpha/beta, fixed for all treatments and Joint regimes',
        pGL_secondary='each final selected Joint checkpoint alpha/beta',
        corrective='accepted pseudo != argmax(pGL); precision; rescued/harmed/wrong-to-wrong; net=rescued-harmed',
        router_scope='complete original Query; D2 excludes added-training parents; B0 paired on same remaining Query',
        classification='original ordered eval1000',evaluation_labels_used_for_selection=False,
        original_L1_budget='deferred',stage1='not started',metadata=metadata()))
    print('REGISTRATION_PASS',flush=True)

def registered():
    preflight();r=read(OUT/'REGISTRATION.json');assert r['files']==code_files()
    return r

def source(index,treatment):
    r=read(OUT/'REGISTRATION.json')['rows'][index];p=Path(r['sources'][TREATMENTS[treatment]]['path'])
    assert digest(p/'COMPLETE.json')==r['sources'][TREATMENTS[treatment]]['seal_sha256']
    return p

def reconstruct(sh,se,ds,path,up):
    import torch
    scores,_=score_files(path);sp=split(sh,se,ds);n=sp['num_nodes'];g=up['gnn_logits'].argmax(1);l=up['llm_logits'].argmax(1)
    import numpy as np
    q=np.sort(np.asarray(sp['unlabeled_ids'],dtype=np.int64));validate_bank(scores,q,g.numpy(),l.numpy())
    choose=torch.ones(n,dtype=torch.bool);trust=torch.zeros(n);accepted=torch.zeros(n,dtype=torch.bool);sel_score=torch.zeros(n)
    for branch,data in scores.items():
        ids=data['ids'].long();choose[ids]=data['route']==0;accepted[ids]=data['accepted'];sel_score[ids]=data['score'].float()
        if branch=='disagreement':trust[ids]=data['score'].float()
    router=dict(node_ids=torch.arange(n),choose_g=choose,trust=trust,accepted=accepted,
                selected_pred=torch.where(choose,g,l),agreement=g==l,selected_score=sel_score)
    if (path/'router.pt').exists():
        check_seal(path,['router.pt']);old=load(path/'router.pt');ids=torch.tensor(sp['unlabeled_ids'])
        for name in ('choose_g','trust','accepted','selected_pred','agreement'):
            torch.testing.assert_close(router[name][ids],old[name][ids],rtol=0,atol=0)
        # Preserve original full-node validation auxiliaries when available.
        router=old
    return router

def newpath(index,treatment,regime):return OUT/'models'/f'{index:02d}'/treatment/regime

def modelpath(index,treatment,regime):
    sh,se,ds=CELLS[index]
    if treatment=='B0':return folder(sh,se,ds)/'b0'
    if (treatment,regime)==('D1_25','retuned'):return source(index,treatment)
    if (treatment,regime)==('D4_confidence','fixed'):return source(index,treatment)
    return newpath(index,treatment,regime)

def train_task(task):
    registered()
    if task not in PILOT:assert read(OUT/'PILOT_PASS.json')['all_checks_pass']
    arm,i=divmod(task,75);treatment,regime=ARMS[arm];sh,se,ds=CELLS[i]
    dest=newpath(i,treatment,regime)
    if (dest/'COMPLETE.json').exists():check_seal(dest);return
    if dest.exists():
        archive=dest.with_name(dest.name+'.interrupted-'+str(os.environ['SLURM_JOB_ID']))
        dest.rename(archive)
    src=source(i,treatment);base=folder(sh,se,ds)/'b0';r=read(OUT/'REGISTRATION.json')['rows'][i]
    assert digest(base/'COMPLETE.json')==r['sources']['b0']['seal_sha256']
    check_seal(src,['router.pt','agreement-scores.pt','disagreement-scores.pt','config.json','diagnostics.json'])
    checker=Sources();cfg,cp,e,h,up,pred,provenance=inputs(sh,se,ds,'b0',checker)
    # Common Joint entrypoint must load the exact same expert tensors as B0.
    c=read(ROOT/'configs/inputs.json')['cells'][f'5/42/{ds}']['inputs']
    assert digest(c['embeddings'])==provenance['embeddings']['hash']
    assert digest(c['hidden'])==provenance['hidden']['hash']
    router=reconstruct(sh,se,ds,src,up)
    save(dest/'router.pt',router)
    write(dest/'INPUTS_BEFORE_TRAINING.json',dict(treatment=treatment,regime=regime,index=i,key=key(sh,se,ds),
        router_source=str(src),router_seal_sha256=digest(src/'COMPLETE.json'),
        router_sha256=digest(dest/'router.pt'),B0_inputs=provenance,
        gold_ids=split(sh,se,ds)['support_ids'],validation_ids=split(sh,se,ds)['valid_ids'],
        registration_sha256=digest(OUT/'REGISTRATION.json'),code=code_files(),
        training_labels='only original gold support and validation targets; extra D2 labels stay in frozen Router',
        evaluation_labels_read=False))
    del cp,e,h,up,pred,checker;gc.collect()
    if regime=='fixed':
        from compute import joint
        chosen=joint(sh,se,ds,router,dest,fixed=cfg);assert chosen==cfg
    else:
        chosen=native_joint(sh,se,ds,router,dest,cfg);assert joint_locked(chosen)==joint_locked(cfg)
    assert digest(dest/'router.pt')==read(dest/'INPUTS_BEFORE_TRAINING.json')['router_sha256']
    write(dest/'config.json',dict(joint=chosen,base_joint=cfg,treatment=treatment,regime=regime,
        registration_sha256=digest(OUT/'REGISTRATION.json'),evaluation_labels_used_for_selection=False,metadata=metadata()))
    seal(dest,dict(all_checks_pass=True,training_finished=True,query_truth_used_for_selection=False,
                   protocol='3x2 Joint numerical tuning confound audit'))
    print('FACTORIAL_TRAINED',task,key(sh,se,ds),treatment,regime,flush=True)

def gate(tasks,name):
    registered();proof=[]
    for task in tasks:
        arm,i=divmod(task,75);t,r=ARMS[arm];p=newpath(i,t,r);check_seal(p)
        proof.append(dict(task=task,path=str(p),seal_sha256=digest(p/'COMPLETE.json')))
    write(OUT/name,dict(all_checks_pass=True,tasks=proof,code=code_files(),registration_sha256=digest(OUT/'REGISTRATION.json')))
    print('GATE_PASS',name,len(tasks),flush=True)

def weights(member):
    import torch
    state=member['states']['accuracy']
    return tuple(float(torch.nn.functional.softplus(state[k])) for k in ('raw_alpha','raw_beta'))

def correction(nodes,accepted,pseudo,reference,truth,query_count):
    import numpy as np
    y=truth[nodes];c=pseudo[nodes]==y;b=reference[nodes]==y;change=accepted[nodes]&(pseudo[nodes]!=reference[nodes])
    selected=accepted[nodes];n=int(change.sum());rescued=int((change&c&~b).sum());harmed=int((change&b&~c).sum())
    wrong=int((change&~b&~c).sum());correct=int((change&c).sum());net=rescued-harmed
    assert n==rescued+harmed+wrong and correct==rescued
    assert int(c[selected].sum())-int(b[selected].sum())==net
    count=int(selected.sum())
    return dict(population_count=len(nodes),accepted_count=count,accepted_correct=int(c[selected].sum()),
        pseudo_precision=float(c[selected].mean()) if count else None,
        corrective_count=n,corrective_precision=correct/n if n else None,
        rescued_count=rescued,harmed_count=harmed,wrong_to_wrong_count=wrong,net_corrections=net,
        corrective_correct_count=correct,corrective_incorrect_count=harmed+wrong,
        corrective_right_minus_wrong=correct-harmed-wrong,
        corrective_fraction_of_accepted=n/count if count else None,
        net_per_1000_query=1000*net/query_count if query_count else None,
        pGL_accuracy_on_accepted=float(b[selected].mean()) if count else None,
        signed_gain_per_accepted=net/count if count else None)

def prefixed(prefix,d):return {prefix+k:v for k,v in d.items()}

def report_cell(i):
    registered();import torch,numpy as np
    check=OUT/'ALL_TRAINING_PASS.json';assert read(check)['all_checks_pass']
    sh,se,ds=CELLS[i];dest=OUT/'metrics'/f'{i:02d}'
    if (dest/'COMPLETE.json').exists():check_seal(dest);return
    checker=Sources();basecfg,bcp,e,h,up,bpred,base_prov=inputs(sh,se,ds,'b0',checker)
    a0,b0=weights(bcp);g=up['gnn_logits'].argmax(1).numpy();l=up['llm_logits'].argmax(1).numpy();n=len(g)
    sp=split(sh,se,ds);q=np.sort(np.asarray(sp['unlabeled_ids'],dtype=np.int64));ev=np.asarray(evaluation_ids(se,ds),dtype=np.int64)
    assert len(ev)==1000 and set(ev)<=set(q) and not set(q)&set(sp['support_ids']+sp['valid_ids'])
    # Float64 probability mixture to match the previous frozen residual analysis.
    def pgl(a,b):
        pred=np.empty(n,dtype=np.int64)
        for start in range(0,n,32768):
            pg=up['gnn_logits'][start:start+32768].float().log_softmax(1).exp().double()
            pl=up['llm_logits'][start:start+32768].float().log_softmax(1).exp().double()
            pred[start:start+len(pg)]=(a*pg+b*pl).argmax(1).numpy()
        return pred
    primary=pgl(a0,b0);scorecache={};routercache={};rows=[];node_records=[]
    d2cfg=read(source(i,'D2a_1000')/'config.json');excluded=np.asarray(d2cfg['parameters']['parents'],dtype=np.int64)
    qd2=np.setdiff1d(q,excluded,assume_unique=True);assert not set(excluded)&set(ev)
    assert len(q)-len(qd2)==len(excluded)
    # Evaluation truth is loaded only after all model selection was sealed.
    truth=graph(ds).y.reshape(-1).long().numpy()
    for t in ('B0',*TREATMENTS):
        scorepath=folder(sh,se,ds)/'b0' if t=='B0' else source(i,t)
        scores,scoreprov=score_files(scorepath);validate_bank(scores,q,g,l)
        router=reconstruct(sh,se,ds,scorepath,up);accepted=router['accepted'].numpy();pseudo=router['selected_pred'].numpy()
        pop=qd2 if t=='D2a_1000' else q
        rm=router_metrics(scores,pop,g,l,truth)
        if t=='B0':
            matched_d2=router_metrics(scores,qd2,g,l,truth)
            base_correction_d2=correction(qd2,accepted,pseudo,primary,truth,len(qd2))
        for regime in (('baseline',) if t=='B0' else ('fixed','retuned')):
            path=modelpath(i,t,regime);check_seal(path,['checkpoint.pt','config.json','predictions.pt','predictions-reference.json','INPUTS_BEFORE_TRAINING.json'])
            if t=='B0':cfg,cp,pred=basecfg,bcp,bpred
            elif (t,regime)==('D1_25','retuned'):
                cfg=read(path/'config.json')['joint'];cp=load(path/'checkpoint.pt');pred=load(path/'predictions.pt')
            else:
                cfg=read(path/'config.json').get('joint') or read(path/'config.json')['parameters']['joint_config']
                cp=load(path/'checkpoint.pt');pred=load(path/'predictions.pt')
            assert cp['config']==cfg and cp['training_seed']==se
            if regime=='fixed':assert cfg==basecfg
            if regime=='retuned':assert joint_locked(cfg)==joint_locked(basecfg)
            a,b=weights(cp);own=pgl(a,b);pj=pred['probability'][torch.as_tensor(ev)].argmax(1).numpy();acc=float((pj==truth[ev]).mean())
            primaryc=correction(pop,accepted,pseudo,primary,truth,len(pop))
            secondary=correction(pop,accepted,pseudo,own,truth,len(pop))
            # 'train_active' excludes agreement unless keep>0 and conflict unless cross>0.
            active=accepted.copy();same=g==l
            active[same]&=cfg['keep']>0;active[~same]&=cfg['cross']>0
            row=dict(index=i,key=key(sh,se,ds),dataset=ds,shot=sh,seed=se,treatment=t,regime=regime,
                variant='B0' if t=='B0' else t+'__'+regime,accuracy=acc,alpha=a,beta=b,
                B0_alpha=a0,B0_beta=b0,cross=cfg['cross'],aux=cfg['aux'],keep=cfg['keep'],
                selected_step=cp['checkpoints']['accuracy']['step'],stop_step=cp['history'][-1]['step'],
                config=cfg,full_query_original_count=len(q),excluded_training_count=len(excluded) if t=='D2a_1000' else 0,
                query_ids_sha256=id_hash(pop),eval_ordered_ids_sha256=id_hash(ev),router_metrics=rm,
                corrective=primaryc,corrective_own_pGL=secondary,
                conflict_corrective=correction(pop[~same[pop]],accepted,pseudo,primary,truth,len(pop)),
                train_active_corrective=correction(pop,active,pseudo,primary,truth,len(pop)),
                actual_training_pool=correction(q,accepted,pseudo,primary,truth,len(q)),
                matched_B0_router=matched_d2 if t=='D2a_1000' else rows[0]['router_metrics'] if rows else rm,
                matched_B0_corrective=base_correction_d2 if t=='D2a_1000' else rows[0]['corrective'] if rows else primaryc,
                sources=dict(model_path=str(path),model_seal_sha256=digest(path/'COMPLETE.json'),router=scoreprov,
                             B0_inputs=base_prov),evaluation_labels_used_for_selection=False)
            rows.append(row)
            # Per-node exported evidence on eval1000 only; complete Query tensors stay on server.
            node_records.extend(dict(variant=row['variant'],eval_order=k,node_id=int(node),truth=int(truth[node]),
                pGL_B0=int(primary[node]),pGL_own=int(own[node]),pseudo=int(pseudo[node]),accepted=bool(accepted[node]),
                joint_prediction=int(pj[k]),corrective=bool(accepted[node] and pseudo[node]!=primary[node])) for k,node in enumerate(ev))
            print('FACTORIAL_METRIC',i,row['variant'],acc,primaryc['corrective_precision'],primaryc['net_corrections'],flush=True)
            if t!='B0':del cp,pred
        del router,scores;gc.collect()
    by={r['variant']:r for r in rows}
    for t in TREATMENTS:
        x,y=(by[t+'__'+r] for r in ('fixed','retuned'))
        for field in ('router_metrics','corrective','actual_training_pool','query_ids_sha256','eval_ordered_ids_sha256'):
            assert x[field]==y[field],(t,field)
    # Existing published classification values must be unchanged for reused models.
    prior=read(ROOT/'reports/full-query-20261027/RESULTS.json')
    pi={(r['key'],r['variant']):r for r in prior['per_cell']}
    for label,v in [('B0','b0'),('D1_25__retuned','d1_numeric/0.25'),('D4_confidence__fixed','d4/confidence_router')]:
        assert abs(by[label]['accuracy']-pi[(key(sh,se,ds),v)]['accuracy'])<1e-12
    write(dest/'RESULTS.json',dict(all_checks_pass=True,index=i,rows=rows,code=code_files(),
        registration_sha256=digest(OUT/'REGISTRATION.json'),slurm_job=os.environ['SLURM_JOB_ID']))
    csv_write(dest/'eval1000_evidence.csv',node_records)
    seal(dest,dict(all_checks_pass=True,rows=7,router_pair_identity_pass=True))

def flat(row):
    return {**{k:v for k,v in row.items() if not isinstance(v,(dict,list))},
            **prefixed('router_',row['router_metrics']),**prefixed('corrective_',row['corrective']),
            **prefixed('ownGL_',row['corrective_own_pGL']),**prefixed('train_active_',row['train_active_corrective'])}

def stat(v):
    v=[x for x in v if x is not None]
    return dict(n=len(v),mean=statistics.mean(v) if v else None,std=statistics.pstdev(v) if v else None)

def aggregate():
    registered();import numpy as np
    from scipy.stats import pearsonr,spearmanr
    rows=[];proof=[]
    for i in range(75):
        p=OUT/'metrics'/f'{i:02d}';check_seal(p);r=read(p/'RESULTS.json');assert len(r['rows'])==7
        rows.extend(r['rows']);proof.append(dict(index=i,sha256=digest(p/'COMPLETE.json')))
    assert len(rows)==525
    dest=OUT/'report';assert not dest.exists();dest.mkdir()
    metrics=['accuracy','router_trust_auroc','router_preference_auroc','router_D_Prec_01','router_D_Prec_05','router_D_Prec_20',
        'router_deployed_precision','corrective_corrective_precision','corrective_net_corrections','corrective_net_per_1000_query',
        'corrective_corrective_count','corrective_accepted_count','corrective_corrective_right_minus_wrong','cross','aux','keep']
    flats=[flat(r) for r in rows];summary=[]
    for ds in ('ALL',*DS):
        for shot in ('ALL',*SHOTS):
            for v in LABELS:
                group=[r for r in flats if r['variant']==v and (ds=='ALL' or r['dataset']==ds) and (shot=='ALL' or r['shot']==shot)]
                summary.append(dict(dataset=ds,shot=shot,variant=v,cells=len(group),metrics={m:stat([r.get(m) for r in group]) for m in metrics},
                    cross_counts={str(c):sum(r['cross']==c for r in group) for c in (0.,.25,1.)}))
    pairs=[];by={(r['key'],r['variant']):r for r in rows}
    for r in rows:
        if r['treatment']=='B0':continue
        base=by[(r['key'],'B0')];c=r['corrective'];bc=r['matched_B0_corrective']
        pairs.append(dict(key=r['key'],dataset=r['dataset'],shot=r['shot'],seed=r['seed'],treatment=r['treatment'],regime=r['regime'],variant=r['variant'],
            accuracy_delta_pp=100*(r['accuracy']-base['accuracy']),
            corrective_precision_delta_pp=100*(c['corrective_precision']-bc['corrective_precision']) if c['corrective_precision'] is not None and bc['corrective_precision'] is not None else None,
            pseudo_precision_delta_pp=100*(c['pseudo_precision']-bc['pseudo_precision']) if c['pseudo_precision'] is not None and bc['pseudo_precision'] is not None else None,
            net_per_1000_query_delta=c['net_per_1000_query']-bc['net_per_1000_query'],
            corrective_count_delta=c['corrective_count']-bc['corrective_count'],cross=r['cross'],cross_B0=base['cross']))
    interactions=[]
    for i,(sh,se,ds) in enumerate(CELLS):
        for t in TREATMENTS:
            x,y=(by[(key(sh,se,ds),t+'__'+r)] for r in ('fixed','retuned'))
            interactions.append(dict(key=x['key'],dataset=ds,shot=sh,seed=se,treatment=t,
                tuning_accuracy_gain_pp=100*(y['accuracy']-x['accuracy']),cross_fixed=x['cross'],cross_retuned=y['cross'],
                fixed_validation=x['selected_step'],retuned_validation=y['selected_step']))
    correlations=[]
    for ds in ('ALL',*DS):
        for t in TREATMENTS:
            for regime in ('fixed','retuned'):
                g=[p for p in pairs if p['treatment']==t and p['regime']==regime and (ds=='ALL' or p['dataset']==ds)]
                for m in ('corrective_precision_delta_pp','pseudo_precision_delta_pp','net_per_1000_query_delta'):
                    h=[p for p in g if p[m] is not None];x=np.asarray([p[m] for p in h]);y=np.asarray([p['accuracy_delta_pp'] for p in h])
                    defined=len(x)>=3 and np.ptp(x)>0 and np.ptp(y)>0
                    correlations.append(dict(dataset=ds,treatment=t,regime=regime,predictor=m,n=len(h),
                        pearson=float(pearsonr(x,y).statistic) if defined else None,spearman=float(spearmanr(x,y).statistic) if defined else None,
                        interpretation='descriptive paired deltas; no causal or independence claim'))
    data=dict(all_checks_pass=True,rows=rows,summary=summary,pairs=pairs,interactions=interactions,correlations=correlations,
        proof=proof,code=code_files(),registration_sha256=digest(OUT/'REGISTRATION.json'),
        monitor='PAUSED',L1_budget='deferred',stage1='not started',metadata=metadata())
    write(dest/'RESULTS.json',data);csv_write(dest/'per_cell.csv',flats);csv_write(dest/'paired_deltas.csv',pairs)
    csv_write(dest/'tuning_effects.csv',interactions);csv_write(dest/'correlations.csv',correlations)
    csv_write(dest/'summary.csv',[dict(dataset=s['dataset'],shot=s['shot'],variant=s['variant'],cells=s['cells'],**{m:v['mean'] for m,v in s['metrics'].items()}) for s in summary])
    lines=['# 3×2：Router处理 × Joint数值超参选择','',
        'D1使用numeric25% Router，D2使用随机a1000，D4使用confidence_router；每格5datasets×3shots×5seeds=75。B0另作参照。',
        '固定指固定B0超参数并重新训练权重；重调沿用原生四阶段验证集数值搜索，非搜索设置固定B0。每行两设置的Router输出、已接受伪标签池逐项相同。',
        '纠正性精度：实际已接受伪标签与B0固定pGL预测不同的节点中，伪标签正确比例。净纠正=原错变对−原对变错；另报纠正标签正误差=正确伪标签数−错误伪标签数（该错误数包含错转错）。两口径均保留，避免把错转错算作准确率损失。空集合为null；各模型自身pGL另存。',
        'Router/伪标签主指标用完整Query；D2排除新增训练节点，B0同名单配对。最终分类只用原顺序eval1000。D2额外真标签只进入原Router训练，Joint gold仍为原support。',
        '完整Query节点量不同，净纠正原始数与每1000 Query归一化值同时给出。数值均为配置均值，非节点池化。',
        '', '| Dataset | Setting | n | Accuracy% | Trust AUROC | Pref AUROC | D P@1% | Accepted prec% | Corrective prec% | Corrective n | Net | Net/1000Q | Corrective right-minus-wrong |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x,scale=1):return 'NA' if x is None else f'{x*scale:.3f}'
    for s in summary:
        if s['shot']!='ALL':continue
        m=s['metrics'];vals=[fmt(m[k]['mean'],scale) for k,scale in [('accuracy',100),('router_trust_auroc',1),('router_preference_auroc',1),('router_D_Prec_01',100),('router_deployed_precision',100),('corrective_corrective_precision',100),('corrective_corrective_count',1),('corrective_net_corrections',1),('corrective_net_per_1000_query',1),('corrective_corrective_right_minus_wrong',1)]]
        lines.append('| '+s['dataset']+' | '+s['variant']+' | '+str(s['cells'])+' | '+' | '.join(vals)+' |')
    lines+=['','## Fig.10相关范围：5-shot、seeds42–44、PubMed/Products','','| Dataset | Setting | n | Accuracy% | Accepted precision% | Corrective precision% | Net | Corrective right-minus-wrong |','|---|---|---:|---:|---:|---:|---:|---:|']
    for ds in ('pubmed','ogbn-products'):
        for v in LABELS:
            g=[r for r in flats if r['dataset']==ds and r['shot']==5 and r['seed']<=44 and r['variant']==v]
            keys=[('accuracy',100),('router_deployed_precision',100),('corrective_corrective_precision',100),('corrective_net_corrections',1),('corrective_corrective_right_minus_wrong',1)]
            lines.append('| '+ds+' | '+v+' | '+str(len(g))+' | '+' | '.join(fmt(stat([r.get(k) for r in g])['mean'],scale) for k,scale in keys)+' |')
    lines+=['','## 重调带来的分类变化与cross选择','','| Dataset | Treatment | Tuning gain pp | Fixed cross 0/.25/1 counts | Retuned cross 0/.25/1 counts |','|---|---|---:|---|---|']
    for ds in ('ALL',*DS):
        for t in TREATMENTS:
            g=[r for r in interactions if r['treatment']==t and (ds=='ALL' or r['dataset']==ds)]
            counts=lambda field:'/'.join(str(sum(r[field]==c for r in g)) for c in (0.,.25,1.))
            lines.append(f"| {ds} | {t} | {statistics.mean(r['tuning_accuracy_gain_pp'] for r in g):+.3f} | {counts('cross_fixed')} | {counts('cross_retuned')} |")
    lines+=['','## 离线相关性','','下表用相对B0的配对差值；D2的伪标签参照已匹配剩余Query。完整数据集/shot/seed明细在CSV/JSON。相关性不能替代干预证据；不以测试结果选择方法。',
        '','| Dataset | Treatment | Joint | Predictor delta | n | Pearson | Spearman |','|---|---|---|---|---:|---:|---:|']
    for c in correlations:
        lines.append('| '+' | '.join([c['dataset'],c['treatment'],c['regime'],c['predictor'],str(c['n']),fmt(c['pearson']),fmt(c['spearman'])])+' |')
    lines+=['','这次重调同时开放原Joint数值搜索项，包括cross/aux/keep/学习率等；调参收益不能单独归因cross。原L1预算争议和阶段1未推进。',
        '','文件：per_cell.csv逐模型；paired_deltas.csv相对B0；tuning_effects.csv同Router冻结/重调差；correlations.csv离线相关；RESULTS.json包含完整配置/来源/分母。']
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n');seal(dest,dict(all_checks_pass=True,models=525,cells=75))
    print('FACTORIAL_REPORT_COMPLETE',flush=True)

if __name__=='__main__':
    action=sys.argv[1]
    if action=='register':register()
    elif action=='train':train_task(int(os.environ['SLURM_ARRAY_TASK_ID']))
    elif action=='pilot-gate':gate(PILOT,'PILOT_PASS.json')
    elif action=='all-gate':gate(range(300),'ALL_TRAINING_PASS.json')
    elif action=='metrics':report_cell(int(os.environ['SLURM_ARRAY_TASK_ID']))
    elif action=='aggregate':aggregate()
    elif action=='associations':
        from associations import main
        main()
    else:raise ValueError(action)
