"""Frozen B0/3x2: raw AUROCs, deployed pseudo quality and residual subsets."""
from pathlib import Path
import sys
EXPERIMENT=Path(__file__).resolve().parent
sys.path.insert(0,str(EXPERIMENT.parent/'stage0_20261029'))
import factorial_joint as original
from factorial_joint import *
sys.path.insert(0,str(EXPERIMENT))
import itertools,shutil,tarfile
OUT=ROOT/'runs/comprehensive-b0-3x2-20261029-v1'
OLD=original.OUT
OWN=ROOT/'runs/joint-factorial-own-pgl-20261029-v2/report'
COMP=ROOT/'reports/joint-factorial-components-20261029'
PUB=ROOT/'reports/comprehensive-b0-3x2-20261029'
CONTROL='B0_matched_D2'
ROUTERS=('B0',*TREATMENTS)
PILOT=(0,1,2,3,4,15,25,50)

def files():
    paths=[EXPERIMENT/p for p in ('AGENTS.md','comprehensive.py','render_report.py')]
    paths += [ROOT/p for p in ('src/enhancerouter/router.py','src/enhancerouter/policy.py','experiments/stage0_20261026/compute.py','experiments/stage0_20261027/d1_numeric.py','experiments/stage0_20261029/components_factorial.py')]
    return {str(p.resolve().relative_to(ROOT)):digest(p) for p in paths}

def register():
    original.registered();check_seal(OLD/'report');check_seal(OWN)
    c=read(COMP/'PUBLICATION_SHA256.json')
    for ds in DS:
        for name in ('RESULTS.json','eval1000_components.csv','COMPLETE.json'):
            rel=f'datasets/{ds}/{name}';assert digest(COMP/rel)==c[rel]
    audit=read(OLD/'INDEPENDENT_AUDIT_PASS.json');assert audit['all_checks_pass'] and audit['models']==525
    write(OUT/'REGISTRATION.json',dict(models=525,cells=75,pilot=list(PILOT),variants=LABELS,paired_control=CONTROL,seeds=list(DEV),shots=list(SHOTS),datasets=list(DS),files=files(),
        training=False,selection=False,original_report_seal_sha256=digest(OLD/'report/COMPLETE.json'),own_report_seal_sha256=digest(OWN/'COMPLETE.json'),component_manifest_sha256=digest(COMP/'PUBLICATION_SHA256.json'),
        raw_AUROC='runtime.predict wrapper / engine.predict_candidate, before raw() gold_normalization/center-scale and all outer affine/pair calibration; D4 original confidence scores, no learned calibration',
        post_P_at_k='sealed calibrated branch score and route; ceil(percent*remaining disagreement Query), tie node_id ascending',
        pseudo_scope='accepted original full Query; D2 removes its 1000 extra training parents and paired B0 uses identical remaining Query',
        corrective_scope='accepted disagreement only; each model own final accuracy checkpoint alpha/beta',
        classification='original ordered eval1000; S=argmax(pR)!=argmax(pGL_own); pR/no-pR/full accuracy on eval1000 and on S',
        aggregation='configuration-equal mean and population std; empty sets null; counts sum and mean, node-configuration occurrences',metadata=metadata()))
    print('COMPREHENSIVE_REGISTRATION_PASS',flush=True)

def registered():
    original.registered();r=read(OUT/'REGISTRATION.json');assert files()==r['files']
    assert digest(OLD/'report/COMPLETE.json')==r['original_report_seal_sha256']
    assert digest(OWN/'COMPLETE.json')==r['own_report_seal_sha256']
    assert digest(COMP/'PUBLICATION_SHA256.json')==r['component_manifest_sha256']
    return r

def close(a,b,tol=1e-12):
    if a is None or b is None:assert a is b,(a,b)
    else:assert abs(a-b)<=tol,(a,b)

def component_predictions(index,truth):
    import numpy as np
    sh,se,ds=CELLS[index];directory=COMP/'datasets'/ds;manifest=read(COMP/'PUBLICATION_SHA256.json')
    for name in ('RESULTS.json','eval1000_components.csv'):assert digest(directory/name)==manifest[f'datasets/{ds}/{name}']
    expected=np.asarray(evaluation_ids(se,ds),dtype=np.int64);result={};metadata_rows=read(directory/'RESULTS.json')['rows']
    evidence=OLD/'metrics'/f'{index:02d}';check_seal(evidence,['eval1000_evidence.csv'])
    with (evidence/'eval1000_evidence.csv').open() as f:
        previous={}
        for r in csv.DictReader(f):previous.setdefault(r['variant'],[]).append(r)
    with (directory/'eval1000_components.csv').open() as f:
        reader=csv.DictReader(f)
        for meta in metadata_rows:
            block=list(itertools.islice(reader,1000));assert len(block)==1000
            if meta['index']!=index:continue
            v=meta['variant'];assert [int(x['eval_order']) for x in block]==list(range(1000))
            assert all(x['variant']==v for x in block)
            ids=np.asarray([int(x['node_id']) for x in block]);assert np.array_equal(ids,expected)
            y=np.asarray([int(x['truth']) for x in block]);assert np.array_equal(y,truth[ids])
            j,r,n=(np.asarray([int(x[k]) for x in block]) for k in ('joint_pred','repair_only_pred','without_repair_pred'))
            prior=previous[v];assert [int(x['node_id']) for x in prior]==ids.tolist()
            assert [int(x['joint_prediction']) for x in prior]==j.tolist() and [int(x['pGL_own']) for x in prior]==n.tolist()
            assert [int(x['truth']) for x in prior]==y.tolist()
            for pred,field in ((j,'joint_accuracy'),(r,'repair_only_accuracy'),(n,'without_repair_accuracy')):close(float((pred==y).mean()),meta[field])
            subset=r!=n;m=int(subset.sum());assert not bool(((j!=n)&~subset).any()),'Joint changed consensus label outside requested subset'
            result[v]=dict(meta=meta,ids=ids,y=y,j=j,r=r,n=n,subset=subset,
                metrics=dict(eval_count=1000,pR_accuracy=float((r==y).mean()),without_pR_accuracy=float((n==y).mean()),joint_accuracy=float((j==y).mean()),
                    eval_pR_correct=int((r==y).sum()),eval_without_pR_correct=int((n==y).sum()),eval_joint_correct=int((j==y).sum()),
                    subset_count=m,subset_fraction=m/1000,subset_pR_correct=int(((r==y)&subset).sum()),subset_without_pR_correct=int(((n==y)&subset).sum()),subset_joint_correct=int(((j==y)&subset).sum()),
                    subset_pR_accuracy=float((r[subset]==y[subset]).mean()) if m else None,subset_without_pR_accuracy=float((n[subset]==y[subset]).mean()) if m else None,subset_joint_accuracy=float((j[subset]==y[subset]).mean()) if m else None,
                    old_joint_changed_count=int((j!=n).sum()),old_joint_changed_fraction=float((j!=n).mean())))
        assert next(reader,None) is None
    assert set(result)==set(LABELS);return result

def query_bank(sh,se,ds,checker):
    if se<=44 or sh==5:
        path=Path(read(RUN/'HISTORICAL_INPUTS_V2.json')['cells'][key(sh,se,ds)]['branches']['disagreement']['query'])
        obj,receipt=checker.tensor(path)
    else:
        path=folder(sh,se,ds)/standard_mode(sh,se)/'disagreement_query.pt'
        expected=read(path.parent/'BANK_COMPLETE.json')['files'][path.name]
        obj,receipt=checker.tensor(path,expected)
    return {k:obj[k] for k in ('dataset','ids','tab','g','l','gnn_pred','llm_pred')},receipt

def raw_router(path,clean,stored,checker,dest):
    import numpy as np,joblib
    from unittest.mock import patch
    import enhancerouter.router as runtime
    from frozen_fit import input_hash
    check_seal(path,['models.joblib']);modelpath=path/'models.joblib';assert modelpath.exists()
    models=joblib.load(modelpath);bundle=models['disagreement'];ids=np.asarray(clean['ids'],dtype=np.int64)
    assert np.array_equal(ids,stored['ids'].numpy())
    if bundle.get('unavailable'):
        return None,None,dict(unavailable=bundle['unavailable'],path=str(modelpath),sha256=digest(modelpath)),None
    outputs={};head_receipts={}
    for head in ('trust','preference'):
        wrapped=bundle['models'][head];before=input_hash(wrapped)
        z=np.asarray(runtime.predict(wrapped,clean),dtype=np.float64)
        assert z.shape==(len(ids),) and np.isfinite(z).all()
        assert input_hash(wrapped)==before
        outputs[id(wrapped)]=z
        head_receipts[head]=dict(engine=wrapped['engine'],config=wrapped['model']['config'],fitted_input_hash=before,
            output_sha256=hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest(),outer_calibration_applied=False)
    with patch.object(runtime,'predict',side_effect=lambda wrapped,b:outputs[id(wrapped)]):post=runtime.predict_bundle(bundle,clean)
    q=post['q'];trust=q[:,0]+q[:,1];pref=np.divide(q[:,0],trust,out=np.full(len(ids),.5),where=trust>0)
    for field,z in (('score',post['score']),('trust',trust),('preference',pref)):
        np.testing.assert_allclose(z,stored[field].numpy(),rtol=0,atol=1e-8,err_msg='Restored scorer no longer reproduces deployed '+field)
    assert np.array_equal(post['route'],stored['route'].numpy())
    zt=outputs[id(bundle['models']['trust'])];zp=outputs[id(bundle['models']['preference'])]
    binary=dest/(path.parent.name+'-'+path.name+'-raw-heads.npz')
    np.savez_compressed(binary,node_ids=ids,trust_raw=zt,preference_raw=zp)
    return zt,zp,dict(path=str(modelpath),sha256=digest(modelpath),heads=head_receipts,
        outer_calibration={k:bundle[k] for k in ('tuning_schema','affine','calibration','route_recipe','boundary','policy') if k in bundle},
        post_calibration_deployment_reproduced=True),dict(path=str(binary),sha256=digest(binary),bytes=binary.stat().st_size)

def worker(index):
    registered();import numpy as np,torch
    if index not in PILOT:assert read(OUT/'PILOT_PASS.json')['all_checks_pass']
    from report import auc
    sh,se,ds=CELLS[index];dest=OUT/'cells'/f'{index:02d}'
    if (dest/'COMPLETE.json').exists():check_seal(dest);return
    assert not dest.exists(),'Preserve interrupted cell before retry';dest.mkdir(parents=True)
    originals=read(OLD/'report/RESULTS.json')['rows'];by={r['variant']:r for r in originals if r['index']==index};assert len(by)==7
    checker=Sources();gpath=read(ROOT/'configs/inputs.json')['cells'][f'5/42/{ds}']['graph'];graph_receipt=checker.check(gpath)
    truth=graph(ds).y.reshape(-1).long().numpy();uprec=by['B0']['sources']['B0_inputs']['upstream']
    up,up_receipt=checker.tensor(uprec['path'],{uprec['hash_algorithm']:uprec['hash']})
    g=up['gnn_logits'].argmax(1).numpy();l=up['llm_logits'].argmax(1).numpy();sp=split(sh,se,ds)
    q=np.sort(np.asarray(sp['unlabeled_ids'],dtype=np.int64));ev=np.asarray(evaluation_ids(se,ds),dtype=np.int64)
    assert not set(q)&set(sp['support_ids']+sp['valid_ids']) and set(ev)<=set(q)
    added=np.asarray(read(original.source(index,'D2a_1000')/'config.json')['parameters']['parents'],dtype=np.int64)
    assert len(added)==1000 and not set(added)&set(ev);qd2=np.setdiff1d(q,added,assume_unique=True)
    clean,bank_receipt=query_bank(sh,se,ds,checker);bid=np.asarray(clean['ids'],dtype=np.int64)
    assert np.array_equal(np.asarray(clean['gnn_pred']),g[bid]) and np.array_equal(np.asarray(clean['llm_pred']),l[bid])
    assert bool((g[bid]!=l[bid]).all());components=component_predictions(index,truth)
    cache={};raw_evidence=[]
    for treatment in ROUTERS:
        variant='B0' if treatment=='B0' else treatment+'__fixed'
        path=Path(by[variant]['sources']['router']['path'])
        assert digest(path/'COMPLETE.json')==by[variant]['sources']['router']['seal_sha256']
        scores,provenance=score_files(path);validate_bank(scores,q,g,l)
        if treatment=='D4_confidence':
            zt=scores['disagreement']['trust'].numpy();zp=scores['disagreement']['preference'].numpy()
            receipt=dict(method='native confidence rule: max(cG,cL) and cG-cL; no learned calibration',source=provenance);binary=None
        else:zt,zp,receipt,binary=raw_router(path,clean,scores['disagreement'],checker,dest)
        cache[treatment]=(scores,zt,zp,provenance,receipt)
        if binary:raw_evidence.append(binary)
    nodes_export=[];eval_export=[];rows=[]
    def pgl(nodes,a,b):
        result=np.empty(len(nodes),dtype=np.int64)
        for start in range(0,len(nodes),32768):
            ids=torch.as_tensor(nodes[start:start+32768]);pg=up['gnn_logits'][ids].float().log_softmax(1).exp().double();pl=up['llm_logits'][ids].float().log_softmax(1).exp().double()
            result[start:start+len(ids)]=(a*pg+b*pl).argmax(1).numpy()
        return result
    for label in (*LABELS,CONTROL):
        prior=by['B0' if label==CONTROL else label];t='B0' if label==CONTROL else prior['treatment'];scores,zt,zp,scoreprov,rawprov=cache[t]
        pop=qd2 if t=='D2a_1000' or label==CONTROL else q;rm=router_metrics(scores,pop,g,l,truth)
        previous=by['D2a_1000__fixed']['matched_B0_router'] if label==CONTROL else prior['router_metrics']
        assert rm==previous,'Frozen calibrated Router metrics changed'
        dd=scores['disagreement'];ids=dd['ids'].numpy().astype(np.int64);mask=np.isin(ids,pop);dn=ids[mask]
        trusted=(g[dn]==truth[dn])|(l[dn]==truth[dn]);gcorrect=g[dn]==truth[dn]
        rawt=auc(trusted,zt[mask]) if zt is not None else None
        rawp=auc(gcorrect[trusted],zp[mask][trusted]) if zp is not None else None
        accepted=dd['accepted'].numpy()[mask].astype(bool);route=dd['route'].numpy()[mask];pseudo=np.where(route==0,g[dn],l[dn])
        reference=pgl(dn,prior['alpha'],prior['beta']);changed=accepted&(pseudo!=reference);correct=pseudo==truth[dn];basecorrect=reference==truth[dn]
        R=changed&correct&~basecorrect;H=changed&~correct&basecorrect;W=changed&~correct&~basecorrect
        C=int(changed.sum());nr,nh,nw=(int(x.sum()) for x in (R,H,W));assert C==nr+nh+nw
        oldc=by['D2a_1000__fixed']['matched_B0_corrective'] if label==CONTROL else prior['corrective_own_pGL']
        for f,value in (('corrective_count',C),('rescued_count',nr),('harmed_count',nh),('wrong_to_wrong_count',nw),('net_corrections',nr-nh),('corrective_right_minus_wrong',nr-nh-nw)):assert oldc[f]==value,(prior['key'],label,f)
        close(oldc['corrective_precision'],nr/C if C else None)
        # Agreement consensus under positive alpha/beta must agree with its pGL.
        aa=scores['agreement'];aid=aa['ids'].numpy().astype(np.int64);apop=aid[np.isin(aid,pop)];amask=np.isin(aid,pop)&aa['accepted'].numpy().astype(bool);an=aid[amask]
        assert np.array_equal(pgl(an,prior['alpha'],prior['beta']),g[an])
        ac=int((g[an]==truth[an]).sum());dc=int(correct[accepted].sum());na=len(an);nd=int(accepted.sum())
        assert na==rm['A_deployed_count'] and nd==rm['D_deployed_count'] and na+nd==rm['deployed_count']
        close(ac/na if na else None,rm['A_deployed_precision']);close(dc/nd if nd else None,rm['D_deployed_precision']);close((ac+dc)/(na+nd) if na+nd else None,rm['deployed_precision'])
        comp=components['B0' if label==CONTROL else label];assert (comp['meta']['alpha'],comp['meta']['beta'])==(prior['alpha'],prior['beta'])
        row=dict(index=index,key=prior['key'],dataset=ds,shot=sh,seed=se,variant=label,query_count=len(pop),query_ids_sha256=id_hash(pop),eval_ordered_ids_sha256=id_hash(ev),
            alpha=prior['alpha'],beta=prior['beta'],config=prior['config'],source=prior['sources'],router_raw=dict(trust_auroc=rawt,preference_auroc=rawp,disagreement_count=len(dn),trust_positive_count=int(trusted.sum()),preference_population_count=int(trusted.sum()),source=rawprov),
            router_calibrated=rm,pseudo=dict(A_count=na,A_correct=ac,A_precision=ac/na if na else None,D_count=nd,D_correct=dc,D_precision=dc/nd if nd else None,all_count=na+nd,all_correct=ac+dc,all_precision=(ac+dc)/(na+nd) if na+nd else None,
                full_A_count=len(apop),full_A_precision=float((g[apop]==truth[apop]).mean()) if len(apop) else None,full_D_count=len(dn),full_D_precision=float(correct.mean()) if len(dn) else None,
                full_count=len(pop),full_precision=(int((g[apop]==truth[apop]).sum())+int(correct.sum()))/len(pop)),
            corrective_D=dict(C=C,R=nr,H=nh,W=nw,precision=nr/C if C else None,net=nr-nh,right_minus_wrong=nr-nh-nw,population_count=len(dn),accepted_count=nd,reference='own selected Joint alpha/beta'),classification=comp['metrics'])
        rows.append(row)
        for k in np.flatnonzero(changed):nodes_export.append(dict(key=row['key'],variant=label,node_id=int(dn[k]),truth=int(truth[dn[k]]),pseudo=int(pseudo[k]),pGL_own=int(reference[k]),outcome='R' if R[k] else 'H' if H[k] else 'W'))
        for k in range(1000):eval_export.append(dict(key=row['key'],variant=label,eval_order=k,node_id=int(comp['ids'][k]),truth=int(comp['y'][k]),pR=int(comp['r'][k]),pGL_own=int(comp['n'][k]),pJ=int(comp['j'][k]),in_requested_subset=bool(comp['subset'][k])))
    for t in TREATMENTS:
        a,b=(next(r for r in rows if r['variant']==t+'__'+v) for v in ('fixed','retuned'))
        for field in ('router_raw','router_calibrated','pseudo','query_ids_sha256'):assert a[field]==b[field]
    write(dest/'RESULTS.json',dict(all_checks_pass=True,index=index,rows=rows,raw_binary_evidence=raw_evidence,inputs=dict(graph=graph_receipt,upstream=up_receipt,query_bank=bank_receipt),files=files(),registration_sha256=digest(OUT/'REGISTRATION.json'),slurm_job=os.environ['SLURM_JOB_ID']))
    csv_write(dest/'corrective_D_nodes.csv',nodes_export);csv_write(dest/'eval1000_labels_and_subset.csv',eval_export)
    seal(dest,dict(all_checks_pass=True,rows=8,old_metrics_exact=True,own_C_R_H_W_exact=True));print('COMPREHENSIVE_CELL_PASS',index,key(sh,se,ds),flush=True)

def pilot_gate():
    registered();proof=[]
    for i in PILOT:
        p=OUT/'cells'/f'{i:02d}';check_seal(p)
        d=read(p/'RESULTS.json');assert d['all_checks_pass'] and d['files']==files() and len(d['rows'])==8
        proof.append(dict(index=i,seal_sha256=digest(p/'COMPLETE.json')))
    write(OUT/'PILOT_PASS.json',dict(all_checks_pass=True,cells=proof,files=files()))
    print('COMPREHENSIVE_PILOT_GATE_PASS',len(proof),flush=True)

def aggregate():
    registered();rows=[];proof=[]
    for i in range(75):
        p=OUT/'cells'/f'{i:02d}';check_seal(p);d=read(p/'RESULTS.json');assert d['all_checks_pass'] and d['files']==files();assert len(d['rows'])==8
        rows.extend(d['rows']);proof.append(dict(index=i,sha256=digest(p/'COMPLETE.json'),slurm_job=d['slurm_job']))
    assert len(rows)==600
    from render_report import summarize,render
    summaries=[]
    for ds,shot in [('ALL','ALL'),*[(d,s) for d in DS for s in SHOTS]]:
        for label in (*LABELS,CONTROL):
            g=[r for r in rows if r['variant']==label and (ds=='ALL' or r['dataset']==ds) and (shot=='ALL' or r['shot']==shot)]
            summaries.append(summarize(g,ds,shot,label))
    p=OUT/'report';write(p/'RESULTS.json',dict(all_checks_pass=True,rows=rows,summary=summaries,proof=proof,registration=read(OUT/'REGISTRATION.json')))
    flat=[]
    for r in rows:
        item={k:v for k,v in r.items() if not isinstance(v,(dict,list))}
        for prefix,part in (('raw_',r['router_raw']),('calibrated_',r['router_calibrated']),('pseudo_',r['pseudo']),('corrective_D_',r['corrective_D']),('classification_',r['classification'])):
            item.update({prefix+k:v for k,v in part.items() if not isinstance(v,(dict,list))})
        flat.append(item)
    csv_write(p/'per_cell.csv',flat)
    csv_write(p/'summary.csv',[dict(dataset=s['dataset'],shot=s['shot'],variant=s['variant'],cells=s['cells'],**{k+'_mean':v['mean'] for k,v in s['metrics'].items()},**{k+'_std':v['std'] for k,v in s['metrics'].items()},**{k+'_defined':v['n'] for k,v in s['metrics'].items()},**{k+'_total':v['total'] for k,v in s['counts'].items()},**{k+'_mean':v['mean'] for k,v in s['counts'].items()},**{'pooled_'+k:v for k,v in s['pooled'].items()}) for s in summaries])
    text=render(summaries,rows);(p/'REPORT.md').write_text(text)
    seal(p,dict(all_checks_pass=True,models=525,paired_controls=75,groups=16))
    assert not PUB.exists();shutil.copytree(p,PUB)
    for name in ('REGISTRATION.json','JOB_GRAPH.json'):shutil.copy2(OUT/name,PUB/name)
    dest=PUB/'cells';dest.mkdir()
    for i in range(75):
        src=OUT/'cells'/f'{i:02d}';dst=dest/f'{i:02d}';dst.mkdir()
        for name in ('RESULTS.json','COMPLETE.json','corrective_D_nodes.csv','eval1000_labels_and_subset.csv'):shutil.copy2(src/name,dst/name)
    write(PUB/'PUBLICATION_SHA256.json',{str(f.relative_to(PUB)):digest(f) for f in sorted(PUB.rglob('*')) if f.is_file()})
    archive=ROOT/'runs/comprehensive-b0-3x2-20261029-results.tar.gz';assert not archive.exists()
    with tarfile.open(archive,'w:gz') as t:t.add(PUB,arcname=str(PUB.relative_to(ROOT)));t.add(EXPERIMENT,arcname=str(EXPERIMENT.relative_to(ROOT)))
    receipt=dict(all_checks_pass=True,models=525,paired_controls=75,groups=16,report=str(PUB/'REPORT.md'),report_sha256=digest(PUB/'REPORT.md'),archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=digest(archive),manifest_files=len(read(PUB/'PUBLICATION_SHA256.json')))
    write(OUT/'PUBLICATION.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='register':register()
    elif mode=='cell':worker(int(os.environ['SLURM_ARRAY_TASK_ID']))
    elif mode=='pilot-gate':pilot_gate()
    elif mode=='aggregate':aggregate()
    else:raise ValueError(mode)
