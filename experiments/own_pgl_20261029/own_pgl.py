"""Promote sealed own-pGL diagnostics; preserve all trained models and pools."""
from pathlib import Path
import sys
EXPERIMENT=Path(__file__).resolve().parent
sys.path.insert(0,str(EXPERIMENT.parent/'stage0_20261029'))
import factorial_joint as original
from factorial_joint import *
sys.path.insert(0,str(EXPERIMENT))
import shutil,tarfile
OLD_OUT=original.OUT
OUT=ROOT/'runs/joint-factorial-own-pgl-20261029-v2'
COMPONENTS=ROOT/'runs/joint-factorial-components-20261029-v1/report'
PUBLICATION=ROOT/'reports/joint-factorial-own-pgl-20261029'
COUNTS=('population_count','accepted_count','D_accepted_count','training_active_accepted_count','corrective_count','rescued_count','harmed_count','wrong_to_wrong_count','net_corrections','corrective_right_minus_wrong')
METRICS=('accuracy','repair_only_accuracy','without_repair_accuracy','pseudo_precision','D_pseudo_precision','corrective_precision','net_per_1000_query')

def analysis_files():
    paths=[EXPERIMENT/p for p in ('AGENTS.md','own_pgl.py','associations.py','scope_analysis.py','DERIVATION.json')]
    paths += sorted((EXPERIMENT/'recovery').glob('*'))
    paths += [EXPERIMENT.parent/'stage0_20261029'/p for p in ('associations.py','scope_analysis.py')]
    return {str(p.resolve().relative_to(ROOT)):digest(p) for p in paths}

def register():
    original.registered();check_seal(OLD_OUT/'report');check_seal(COMPONENTS)
    audit=read(OLD_OUT/'INDEPENDENT_AUDIT_PASS.json')
    assert audit['all_checks_pass'] and audit['models']==525
    assert audit['report_seal_sha256']==digest(OLD_OUT/'report/COMPLETE.json')
    write(OUT/'REGISTRATION.json',dict(user_requested=True,only_change='primary corrective reference: own final selected Joint alpha/beta instead of B0 alpha/beta',
        source_own_field='corrective_own_pGL',models=525,training=False,model_selection=False,
        query_scope='same original full Query; D2 excludes added training nodes; paired B0 same remaining Query',
        classification='unchanged original ordered eval1000',counts='node-configuration occurrences; no deduplication across shots/seeds',
        files=analysis_files(),original_report_seal_sha256=digest(OLD_OUT/'report/COMPLETE.json'),
        component_report_seal_sha256=digest(COMPONENTS/'COMPLETE.json'),original_audit_sha256=digest(OLD_OUT/'INDEPENDENT_AUDIT_PASS.json'),metadata=metadata()))

def registered():
    original.registered();r=read(OUT/'REGISTRATION.json');assert r['files']==analysis_files()
    assert r['original_report_seal_sha256']==digest(OLD_OUT/'report/COMPLETE.json')
    assert r['component_report_seal_sha256']==digest(COMPONENTS/'COMPLETE.json')
    return r

def validate(c):
    C,R,H,W=(c[k] for k in ('corrective_count','rescued_count','harmed_count','wrong_to_wrong_count'))
    assert C==R+H+W and 0<=C<=c['accepted_count']<=c['population_count']
    assert c['corrective_correct_count']==R and c['corrective_incorrect_count']==H+W
    assert c['net_corrections']==R-H and c['corrective_right_minus_wrong']==R-H-W
    assert c['corrective_precision']==(R/C if C else None)
    assert c['pseudo_precision']==(c['accepted_correct']/c['accepted_count'] if c['accepted_count'] else None)
    assert abs(c['net_per_1000_query']-1000*(R-H)/c['population_count'])<1e-12
    if c['accepted_count']:
        assert abs(c['pseudo_precision']-c['pGL_accuracy_on_accepted']-c['signed_gain_per_accepted'])<1e-12

def normalize():
    registered();data=read(OLD_OUT/'report/RESULTS.json');components=read(COMPONENTS/'RESULTS.json')
    assert components['all_checks_pass'] and components['models']==525
    bycomponent={(r['key'],r['variant']):r for r in components['rows']}
    byoriginal={(r['key'],r['variant']):r for r in data['rows']}
    keep=('index','key','dataset','shot','seed','treatment','regime','variant','accuracy','alpha','beta','B0_alpha','B0_beta','cross','aux','keep','selected_step','stop_step','config','full_query_original_count','excluded_training_count','query_ids_sha256','eval_ordered_ids_sha256','router_metrics','matched_B0_router','sources','evaluation_labels_used_for_selection')
    rows=[];matched=[]
    for old in data['rows']:
        row={k:copy.deepcopy(old[k]) for k in keep};c=copy.deepcopy(old['corrective_own_pGL']);validate(c)
        assert c['population_count']==old['full_query_original_count']-old['excluded_training_count']
        for k in ('population_count','accepted_count','accepted_correct','pseudo_precision'):
            assert c[k]==old['corrective'][k]
        assert c['accepted_count']==old['router_metrics']['deployed_count']
        assert c['pseudo_precision']==old['router_metrics']['deployed_precision']
        if old['variant']=='B0':assert c==old['corrective']
        row.update(corrective=c,corrective_own_pGL=copy.deepcopy(c),matched_B0_corrective=copy.deepcopy(old['matched_B0_corrective']),
            primary_reference='own final accuracy checkpoint alpha/beta',training_active_accepted_count=old['train_active_corrective']['accepted_count'])
        validate(row['matched_B0_corrective'])
        comp=bycomponent[(row['key'],row['variant'])]
        assert (comp['alpha'],comp['beta'],comp['config'],comp['source'])==(row['alpha'],row['beta'],row['config'],row['sources'])
        assert comp['joint_accuracy']==row['accuracy'] and comp['eval_ordered_ids_sha256']==row['eval_ordered_ids_sha256']
        row.update(repair_only_accuracy=comp['repair_only_accuracy'],without_repair_accuracy=comp['without_repair_accuracy'])
        for k in keep:assert row[k]==old[k]
        rows.append(row)
        if row['variant']=='D2a_1000__fixed':
            b=copy.deepcopy(byoriginal[(row['key'],'B0')]);bc=copy.deepcopy(row['matched_B0_corrective']);validate(bc)
            b.update(variant='B0_matched_D2',corrective=bc,router_metrics=copy.deepcopy(row['matched_B0_router']),query_ids_sha256=row['query_ids_sha256'],excluded_training_count=row['excluded_training_count'])
            cb=bycomponent[(row['key'],'B0')]
            b.update(repair_only_accuracy=cb['repair_only_accuracy'],without_repair_accuracy=cb['without_repair_accuracy'],training_active_accepted_count=None)
            matched.append(b)
    assert len(rows)==525 and len(matched)==75
    by={(r['key'],r['variant']):r for r in rows};pairs=[]
    oldpairs={(r['key'],r['variant']):r for r in data['pairs']}
    for r in rows:
        if r['variant']=='B0':continue
        b=by[(r['key'],'B0')];c=r['corrective'];bc=r['matched_B0_corrective']
        p={k:r[k] for k in ('key','dataset','shot','seed','treatment','regime','variant')}
        p.update(accuracy_delta_pp=100*(r['accuracy']-b['accuracy']),
            corrective_precision_delta_pp=100*(c['corrective_precision']-bc['corrective_precision']) if c['corrective_precision'] is not None and bc['corrective_precision'] is not None else None,
            pseudo_precision_delta_pp=100*(c['pseudo_precision']-bc['pseudo_precision']) if c['pseudo_precision'] is not None and bc['pseudo_precision'] is not None else None,
            net_per_1000_query_delta=c['net_per_1000_query']-bc['net_per_1000_query'],corrective_count_delta=c['corrective_count']-bc['corrective_count'],cross=r['cross'],cross_B0=b['cross'])
        for k in ('accuracy_delta_pp','pseudo_precision_delta_pp','cross','cross_B0'):assert p[k]==oldpairs[(r['key'],r['variant'])][k]
        pairs.append(p)
    for t in TREATMENTS:
        for sh,se,ds in CELLS:
            a,b=(by[(key(sh,se,ds),t+'__'+v)] for v in ('fixed','retuned'))
            for f in ('router_metrics','query_ids_sha256','eval_ordered_ids_sha256'):assert a[f]==b[f]
            for f in ('accepted_count','accepted_correct','pseudo_precision'):assert a['corrective'][f]==b['corrective'][f]
    return rows,matched,pairs,data['interactions']

def summarize(group,dataset,shot,variant,cohort='all75'):
    g=[]
    for r in group:
        c=r['corrective'];g.append(dict(**{k:c[k] for k in ('population_count','accepted_count','corrective_count','rescued_count','harmed_count','wrong_to_wrong_count','net_corrections','corrective_right_minus_wrong','corrective_precision','pseudo_precision','net_per_1000_query')},
            accuracy=r['accuracy'],repair_only_accuracy=r['repair_only_accuracy'],without_repair_accuracy=r['without_repair_accuracy'],D_accepted_count=r['router_metrics']['D_deployed_count'],D_pseudo_precision=r['router_metrics']['D_deployed_precision'],training_active_accepted_count=r['training_active_accepted_count']))
    counts={k:dict(total=sum(r[k] for r in g),**stat([r[k] for r in g])) if all(r[k] is not None for r in g) else dict(total=None,**stat([r[k] for r in g])) for k in COUNTS}
    return dict(dataset=dataset,shot=shot,variant=variant,cohort=cohort,cells=len(g),metrics={k:stat([r[k] for r in g]) for k in METRICS},counts=counts,
        pooled_corrective_precision=counts['rescued_count']['total']/counts['corrective_count']['total'] if counts['corrective_count']['total'] else None)

def report():
    rows,matched,pairs,interactions=normalize();summaries=[]
    for ds in ('ALL',*DS):
        for sh in ('ALL',*SHOTS):
            for v in (*LABELS,'B0_matched_D2'):
                g=[r for r in (matched if v=='B0_matched_D2' else rows) if r['variant']==v and (ds=='ALL' or r['dataset']==ds) and (sh=='ALL' or r['shot']==sh)]
                summaries.append(summarize(g,ds,sh,v))
    for ds in ('pubmed','ogbn-products'):
        for v in LABELS:
            g=[r for r in rows if r['dataset']==ds and r['shot']==5 and r['seed']<=44 and r['variant']==v]
            summaries.append(summarize(g,ds,5,v,'fig10_5shot_seed42_44'))
    p=OUT/'report';write(p/'RESULTS.json',dict(all_checks_pass=True,rows=rows,matched_B0_rows=matched,pairs=pairs,interactions=interactions,summary=summaries,registration=read(OUT/'REGISTRATION.json'),only_reference_changed=True))
    flat=[]
    for r in (*rows,*matched):
        flat.append({**{k:v for k,v in r.items() if not isinstance(v,(dict,list))},**prefixed('corrective_',r['corrective']),**prefixed('router_',r['router_metrics'])})
    csv_write(p/'per_cell.csv',flat);csv_write(p/'paired_deltas.csv',pairs)
    csv_write(p/'summary.csv',[dict(dataset=s['dataset'],shot=s['shot'],variant=s['variant'],cohort=s['cohort'],cells=s['cells'],corrective_defined=s['metrics']['corrective_precision']['n'],**{k:v['mean'] for k,v in s['metrics'].items()},**{k+'_total':v['total'] for k,v in s['counts'].items()},**{k+'_mean':v['mean'] for k,v in s['counts'].items()},pooled_corrective_precision=s['pooled_corrective_precision']) for s in summaries])
    seal(p,dict(all_checks_pass=True,models=525,unchanged_models_router_labels_configs_query_and_classification=True))
    print('OWN_PGL_REPORT_PASS',525,flush=True)

def publish():
    registered();check_seal(OUT/'report');check_seal(OUT/'scope-supplement');check_seal(OUT/'associations')
    assert not PUBLICATION.exists();PUBLICATION.mkdir()
    for name in ('report','scope-supplement','associations'):shutil.copytree(OUT/name,PUBLICATION/name)
    shutil.copy2(OUT/'REGISTRATION.json',PUBLICATION/'REGISTRATION.json');shutil.copy2(OUT/'JOB_GRAPH.json',PUBLICATION/'JOB_GRAPH.json')
    data=read(OUT/'report/RESULTS.json');scope=read(OUT/'scope-supplement/RESULTS.json')
    def fmt(v,scale=1):return 'NA' if v is None else f'{scale*v:.3f}'
    lines=['# 3×2纠正性指标修订：各模型自身pGL','',
        '仅将主纠正性参照从固定B0系数切换为各模型最终accuracy checkpoint自身alpha/beta；复用已封存的corrective_own_pGL。模型、Router、接受节点、伪标签、训练与超参、Query和eval1000均未改变。',
        '纠正性指标在完整Query上计算；D2剔除其新增真标签训练节点，配对B0使用相同剩余Query及B0自身pGL。完整模型/pR单独/去pR准确率仍为原eval1000结果。',
        'C={实际接受且pseudo != argmax(pGL_own)}；R=原错变对，H=原对变错，W=错换另一个错标签。C=R+H+W；纠正性精度=R/C，空C为NA；净纠正=R-H；正确−错误标签数=R-H-W。',
        '以下精度是配置等权均值，纠正性精度仅平均C非空配置；有效n是配置数。计数总数是跨配置的节点—配置次数，节点可跨shot/seed重复，不是去重图节点数；每组均值含空C的零计数。原B0固定参照报告保留归档，相关性解读以本次自身参照版本为准。',
        '', '## 七设置精度（%）','', '| Setting | n | 完整模型 | pR单独 | 去pR | 全部接受伪标签精度 | 分歧接受精度 | 自身pGL纠正性精度 | 有效n |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    main=[s for s in data['summary'] if s['dataset']=='ALL' and s['shot']=='ALL' and s['cohort']=='all75']
    for s in main:
        if s['variant']=='B0_matched_D2':continue
        m=s['metrics'];lines.append('| '+s['variant']+' | '+str(s['cells'])+' | '+' | '.join(fmt(m[k]['mean'],100) for k in ('accuracy','repair_only_accuracy','without_repair_accuracy','pseudo_precision','D_pseudo_precision','corrective_precision'))+' | '+str(m['corrective_precision']['n'])+'/75 |')
    def count_table(group,title,which):
        lines.extend(['','## '+title,'','| Setting | 接受伪标签A+D | 接受分歧D | C | R | H | W | R−H | R−H−W |','|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
        for s in group:
            c=s['counts'];names=('accepted_count','D_accepted_count','corrective_count','rescued_count','harmed_count','wrong_to_wrong_count','net_corrections','corrective_right_minus_wrong')
            lines.append('| '+s['variant']+' | '+' | '.join(str(c[k][which]) if which=='total' else fmt(c[k][which]) for k in names)+' |')
    count_table(main,'全部75组的计数总和','total');count_table(main,'每组配置的平均计数','mean')
    lines+=['','## 自身pGL配对相关性更新','', '沿用原分析：相对匹配B0的差值，竞争指标使用同一有效配置，dataset×seed簇重采样4000次。以下比较纠正性精度与接受分歧精度的Pearson相关；A+D、Spearman、净纠正和各数据集结果见补充报告。',
        '', '| Treatment | Joint | 共同有效n | 纠正性精度r | 分歧伪标签精度r | 差值95%CI |','|---|---|---:|---:|---:|---|']
    for a in scope['associations']:
        if a['dataset']=='ALL' and a['method']=='Pearson' and a['comparison']=='D_accepted_precision_delta_pp':
            ci=a['difference_ci95'];lines.append('| '+a['treatment']+' | '+a['regime']+' | '+str(a['n'])+' | '+fmt(a['corrective_r'])+' | '+fmt(a['precision_r'])+' | '+('['+', '.join(fmt(v) for v in ci)+']' if ci else 'NA')+' |')
    lines+=['','## 分数据集计数总和','']
    for ds in DS:
        group=[s for s in data['summary'] if s['dataset']==ds and s['shot']=='ALL' and s['cohort']=='all75']
        count_table(group,ds,'total')
    lines+=['','## 定位','', '[全部数据集/shot/seed的精度、C/R/H/W及系数](report/per_cell.csv)；[各层级总数与均值](report/summary.csv)；[完整JSON](report/RESULTS.json)。',
        '[Fig10范围、分数据集相关与区间](scope-supplement/REPORT.md)；[完整配对相关及散点图](associations/REPORT.md)。',
        'Router接受数量不等于实际损失启用数量：cross/keep可能为0，逐组CSV另列training_active_accepted_count；本次仍只改变纠正性参照，不改变伪标签权重。',
        '[新指令与分析代码](../../experiments/own_pgl_20261029/AGENTS.md)；[旧固定B0参照归档](../joint-factorial-3x2-20261029/INTERPRETATION.md)。']
    (PUBLICATION/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(PUBLICATION/'PUBLICATION_SHA256.json',{str(p.relative_to(PUBLICATION)):digest(p) for p in sorted(PUBLICATION.rglob('*')) if p.is_file()})
    archive=ROOT/'runs/joint-factorial-own-pgl-20261029-results.tar.gz';assert not archive.exists()
    with tarfile.open(archive,'w:gz') as t:
        t.add(PUBLICATION,arcname=str(PUBLICATION.relative_to(ROOT)));t.add(EXPERIMENT,arcname=str(EXPERIMENT.relative_to(ROOT)))
    receipt=dict(all_checks_pass=True,models=525,publication=str(PUBLICATION),archive=str(archive),archive_sha256=digest(archive),archive_bytes=archive.stat().st_size,files=len(read(PUBLICATION/'PUBLICATION_SHA256.json')),only_reference_changed=True)
    write(OUT/'PUBLICATION.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    register();report()
    import associations,scope_analysis
    assert Path(associations.__file__).resolve().parent==EXPERIMENT
    assert Path(scope_analysis.__file__).resolve().parent==EXPERIMENT
    associations.main();scope_analysis.main();publish()
