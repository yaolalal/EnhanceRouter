"""Frozen 3x2 checkpoints: full pJ, pR-only, and pGL without pR; Slurm-only."""
import factorial_joint as factorial
from factorial_joint import *
import shutil,tarfile
BASE_OUT=factorial.OUT
OUT=ROOT/'runs/joint-factorial-components-20261029-v1'
METRICS=('joint_accuracy','repair_only_accuracy','without_repair_accuracy','label_change_fraction','rescued_fraction','harmed_fraction')

def register():
    factorial.registered();check_seal(BASE_OUT/'report')
    audit=read(BASE_OUT/'INDEPENDENT_AUDIT_PASS.json');assert audit['all_checks_pass'] and audit['models']==525
    assert audit['report_seal_sha256']==digest(BASE_OUT/'report/COMPLETE.json')
    write(OUT/'REGISTRATION.json',dict(models=525,variants=LABELS,seeds=list(DEV),shots=list(SHOTS),datasets=list(DS),
        analysis_code_sha256=digest(__file__),factorial_registration_sha256=digest(BASE_OUT/'REGISTRATION.json'),
        factorial_report_seal_sha256=digest(BASE_OUT/'report/COMPLETE.json'),
        independent_checkpoint_audit_sha256=digest(BASE_OUT/'INDEPENDENT_AUDIT_PASS.json'),
        evaluation='original ordered eval1000; same IDs across settings',
        full='(alpha*pG+beta*pL+pR)/(alpha+beta+1)',repair_only='pR from same frozen Joint',
        without_repair='(alpha*pG+beta*pL)/(alpha+beta); each model own trained alpha/beta, no refit',
        training=False,selection=False,router_metrics_changed=False,metadata=metadata()))
    print('COMPONENT_REGISTRATION_PASS',flush=True)

def setup_analysis():
    factorial.registered();r=read(OUT/'REGISTRATION.json')
    assert digest(__file__)==r['analysis_code_sha256']
    assert digest(BASE_OUT/'report/COMPLETE.json')==r['factorial_report_seal_sha256']
    assert digest(BASE_OUT/'INDEPENDENT_AUDIT_PASS.json')==r['independent_checkpoint_audit_sha256']
    return r

def dataset(index):
    setup_analysis();import numpy as np,torch
    from eargtc.joint_explore.joint_purf_v10_v1 import training
    ds=DS[index];dest=OUT/'datasets'/ds
    if (dest/'COMPLETE.json').exists():check_seal(dest);return
    assert not dest.exists(),'Preserve interrupted output and audit recovery before retry'
    data=read(BASE_OUT/'report/RESULTS.json');by={(r['key'],r['variant']):r for r in data['rows']}
    old=read(ROOT/'reports/residual-audit-20261028/RESULTS.json');oldindex={(r['key'],r['variant']):r for r in old['per_cell']}
    truth=graph(ds).y.reshape(-1).long();checker=Sources();rows=[];nodes=[];prior_verified=0
    for sh in SHOTS:
        for se in DEV:
            i=CELLS.index((sh,se,ds));sp=split(sh,se,ds);ids=torch.tensor(evaluation_ids(se,ds),dtype=torch.long)
            assert len(ids)==1000 and len(ids.unique())==1000 and not set(ids.tolist())&set(sp['support_ids']+sp['valid_ids'])
            basecfg,bcp,e,h,up,bpred,provenance=inputs(sh,se,ds,'b0',checker)
            f=dict(gnn_embeddings=e[ids].float(),logp_g=up['gnn_logits'][ids].float().log_softmax(1),logp_l=up['llm_logits'][ids].float().log_softmax(1));hs=h[ids]
            pg=f['logp_g'].exp().double();pl=f['logp_l'].exp().double();y=truth[ids]
            evidence={}
            path=BASE_OUT/'metrics'/f'{i:02d}';check_seal(path,['eval1000_evidence.csv'])
            with (path/'eval1000_evidence.csv').open() as fp:
                for row in csv.DictReader(fp):evidence.setdefault(row['variant'],[]).append(row)
            for label in LABELS:
                reference=by[(key(sh,se,ds),label)];modelpath=Path(reference['sources']['model_path'])
                assert digest(modelpath/'COMPLETE.json')==reference['sources']['model_seal_sha256']
                if label=='B0':cp,pred=bcp,bpred
                else:
                    check_seal(modelpath,['checkpoint.pt','predictions.pt','config.json'])
                    cp=load(modelpath/'checkpoint.pt');pred=load(modelpath/'predictions.pt')
                cfg=reference['config'];assert cp['config']==cfg and cp['training_seed']==se
                model=training.make_model(f,hs,cfg);model.load_state_dict(cp['states']['accuracy'])
                with torch.inference_mode():pj,pr=training.predict(model,f,hs,torch.arange(1000),repair=True)
                a,b=weights(cp);assert (a,b)==(reference['alpha'],reference['beta']) and a>0 and b>0
                pgl=(a*pg+b*pl)/(a+b)
                reconstructed=(a*pg+b*pl+pr.double())/(a+b+1)
                torch.testing.assert_close(pj.double(),reconstructed,rtol=2e-6,atol=2e-7)
                expected=pred['probability'][ids];torch.testing.assert_close(pj,expected,rtol=0,atol=2e-6)
                assert torch.equal(pj.argmax(1),expected.argmax(1))
                for p in (pj,pr,pgl):
                    assert bool(torch.isfinite(p).all()) and bool((p>=0).all())
                    torch.testing.assert_close(p.double().sum(1),torch.ones(1000,dtype=torch.double),rtol=0,atol=2e-6)
                j,r,n=pj.argmax(1),pr.argmax(1),pgl.argmax(1)
                prev=evidence[label];assert [int(x['node_id']) for x in prev]==ids.tolist()
                assert [int(x['pGL_own']) for x in prev]==n.tolist(),'Own-pGL labels differ from original 3x2 evidence'
                changed=j!=n;rescued=(n!=y)&(j==y);harmed=(n==y)&(j!=y);wrong=changed&(n!=y)&(j!=y)
                assert torch.equal(changed,rescued|harmed|wrong)
                jc,rc,nc=(int((v==y).sum()) for v in (j,r,n))
                assert jc-nc==int(rescued.sum())-int(harmed.sum())
                assert abs(jc/1000-reference['accuracy'])<1e-12
                row=dict(index=i,key=reference['key'],dataset=ds,shot=sh,seed=se,variant=label,
                    eval_count=1000,eval_ordered_ids_sha256=id_hash(ids),joint_accuracy=jc/1000,
                    repair_only_accuracy=rc/1000,without_repair_accuracy=nc/1000,
                    joint_correct=jc,repair_only_correct=rc,without_repair_correct=nc,
                    label_change_count=int(changed.sum()),label_change_fraction=float(changed.double().mean()),
                    rescued_count=int(rescued.sum()),harmed_count=int(harmed.sum()),wrong_to_wrong_count=int(wrong.sum()),
                    rescued_fraction=float(rescued.double().mean()),harmed_fraction=float(harmed.double().mean()),
                    alpha=a,beta=b,weight_R=1/(a+b+1),cross=cfg['cross'],aux=cfg['aux'],keep=cfg['keep'],
                    selected_step=cp['checkpoints']['accuracy']['step'],config=cfg,source=reference['sources'],
                    input_sources=provenance,prediction_verified=True,equation_verified=True,own_pGL_original_labels_verified=True)
                oldvariant='b0' if label=='B0' else 'd1_numeric/0.25' if label=='D1_25__retuned' else None
                prior=oldindex.get((row['key'],oldvariant))
                if prior is not None:
                    for m in ('joint_accuracy','repair_only_accuracy','without_repair_accuracy'):assert abs(row[m]-prior[m])<1e-12,(row['key'],m)
                    prior_verified+=1
                rows.append(row)
                nodes.extend(dict(variant=label,eval_order=k,node_id=int(node),truth=int(y[k]),joint_pred=int(j[k]),repair_only_pred=int(r[k]),without_repair_pred=int(n[k]),changed_by_repair=bool(changed[k]),rescued=bool(rescued[k]),harmed=bool(harmed[k])) for k,node in enumerate(ids))
                print('COMPONENT_CELL',row['key'],label,jc,rc,nc,flush=True)
                del model,pj,pr,pgl,reconstructed
                if label!='B0':del cp,pred
            del bcp,bpred,e,h,up;gc.collect()
    assert len(rows)==105
    write(dest/'RESULTS.json',dict(all_checks_pass=True,rows=rows,models=105,prior_components_verified=prior_verified,
        analysis_code_sha256=digest(__file__),registration_sha256=digest(OUT/'REGISTRATION.json'),slurm_job=os.environ['SLURM_JOB_ID']))
    csv_write(dest/'eval1000_components.csv',nodes)
    seal(dest,dict(all_checks_pass=True,models=105,classification_original_labels_exact=True,original_own_pGL_exact=True))
    print('COMPONENT_DATASET_PASS',ds,105,flush=True)

def aggregate():
    setup_analysis();rows=[];proof=[];old_count=0
    for ds in DS:
        p=OUT/'datasets'/ds;check_seal(p);data=read(p/'RESULTS.json')
        assert data['models']==105 and data['analysis_code_sha256']==digest(__file__)
        rows.extend(data['rows']);old_count+=data['prior_components_verified'];proof.append(dict(dataset=ds,seal_sha256=digest(p/'COMPLETE.json'),slurm_job=data['slurm_job']))
    assert len(rows)==525 and old_count==90
    summary=[]
    for ds in ('ALL',*DS):
        for shot in ('ALL',*SHOTS):
            for v in LABELS:
                g=[r for r in rows if r['variant']==v and (ds=='ALL' or r['dataset']==ds) and (shot=='ALL' or r['shot']==shot)]
                summary.append(dict(dataset=ds,shot=shot,variant=v,cells=len(g),metrics={m:stat([r[m] for r in g]) for m in METRICS}))
    out=OUT/'report';write(out/'RESULTS.json',dict(all_checks_pass=True,models=525,rows=rows,summary=summary,dataset_proofs=proof,
        prior90_components_exact=True,registration=read(OUT/'REGISTRATION.json'),source_sha256=digest(__file__),metadata=metadata()))
    csv_write(out/'per_cell.csv',[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows])
    csv_write(out/'summary.csv',[dict(dataset=s['dataset'],shot=s['shot'],variant=s['variant'],cells=s['cells'],**{m:v['mean'] for m,v in s['metrics'].items()},**{m+'_std':v['std'] for m,v in s['metrics'].items()}) for s in summary])
    lines=['# 3×2及B0：完整模型、pR单独、去pR准确率','',
        '覆盖五数据集、3/5/10-shot、seeds42–46、7种设置，共525份冻结模型。全部计算使用各seed原顺序eval1000，不重新训练、不重调超参。',
        '完整模型pJ=(alpha*pG+beta*pL+pR)/(alpha+beta+1)；pR单独为同一已训练checkpoint的argmax(pR)；去pR为(alpha*pG+beta*pL)/(alpha+beta)，保留该模型自身已训练alpha/beta。',
        '这衡量冻结模型的推理拆分，不等于独立重训pR-only或无pR模型。完整模型预测与3×2保存结果逐节点一致，去pR标签与此前3×2的own-pGL证据逐项一致；90份既有组件结果再次逐值一致。',
        '', '| Dataset | Setting | n | 完整模型% | pR单独% | 去pR% | pR改变标签比例% |','|---|---|---:|---:|---:|---:|---:|']
    for s in summary:
        if s['shot']!='ALL':continue
        m=s['metrics'];lines.append('| '+s['dataset']+' | '+s['variant']+' | '+str(s['cells'])+' | '+' | '.join(f"{100*m[k]['mean']:.2f}" for k in ('joint_accuracy','repair_only_accuracy','without_repair_accuracy','label_change_fraction'))+' |')
    lines+=['','## 各数据集、各shot（五seed均值）','','| Dataset | Shot | Setting | n | 完整模型% | pR单独% | 去pR% |','|---|---:|---|---:|---:|---:|---:|']
    for s in summary:
        if s['dataset']=='ALL' or s['shot']=='ALL':continue
        lines.append('| '+s['dataset']+' | '+str(s['shot'])+' | '+s['variant']+' | '+str(s['cells'])+' | '+' | '.join(f"{100*s['metrics'][k]['mean']:.2f}" for k in ('joint_accuracy','repair_only_accuracy','without_repair_accuracy'))+' |')
    lines+=['','逐seed结果及alpha/beta见per_cell.csv；525份完整输入路径、哈希及配置见RESULTS.json。datasets内CSV包含每节点三种预测及改变/改对/改错。',
        '', '[原3×2完整报告](../joint-factorial-3x2-20261029/INTERPRETATION.md)；[分析代码](../../experiments/stage0_20261029/components_factorial.py)。原模型、数据划分、Router及已有报告均未修改。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');seal(out,dict(all_checks_pass=True,models=525,prior90_verified=True))
    publish(out)

def publish(report):
    dest=ROOT/'reports/joint-factorial-components-20261029';assert not dest.exists();shutil.copytree(report,dest)
    shutil.copytree(OUT/'datasets',dest/'datasets');p=dest/'proofs';p.mkdir()
    for name in ('REGISTRATION.json','JOB_GRAPH.json'):
        shutil.copy2(OUT/name,p/name)
    shutil.copytree(OUT/'logs',dest/'logs')
    write(dest/'PUBLICATION_SHA256.json',{str(p.relative_to(dest)):digest(p) for p in sorted(dest.rglob('*')) if p.is_file()})
    asset=ROOT/'runs/joint-factorial-components-20261029-results.tar.gz';assert not asset.exists()
    with tarfile.open(asset,'w:gz') as t:
        t.add(dest,arcname=str(dest.relative_to(ROOT)));t.add(Path(__file__).resolve(),arcname=str(Path(__file__).resolve().relative_to(ROOT)))
    receipt=dict(all_checks_pass=True,models=525,publication=str(dest),archive=str(asset),archive_bytes=asset.stat().st_size,archive_sha256=digest(asset),source_sha256=digest(__file__),report_seal_sha256=digest(report/'COMPLETE.json'))
    write(OUT/'PUBLICATION.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='register':register()
    elif sys.argv[1]=='dataset':dataset(int(os.environ['SLURM_ARRAY_TASK_ID']))
    elif sys.argv[1]=='aggregate':aggregate()
    else:raise ValueError(sys.argv[1])
