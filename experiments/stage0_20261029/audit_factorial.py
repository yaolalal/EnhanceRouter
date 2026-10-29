"""Independent checkpoint and report audit; Slurm-only, no training."""
from factorial_joint import *

def dataset(index):
    registered();import torch,numpy as np
    from eargtc.joint_explore.joint_purf_v10_v1 import training
    ds=DS[index];report=OUT/'report';check_seal(report);data=read(report/'RESULTS.json');rows=[r for r in data['rows'] if r['dataset']==ds]
    by={(r['key'],r['variant']):r for r in rows};truth=graph(ds).y.reshape(-1).long();checker=Sources();proof=[]
    for sh in SHOTS:
        for se in DEV:
            cellindex=CELLS.index((sh,se,ds));sp=split(sh,se,ds);ev=torch.tensor(evaluation_ids(se,ds))
            basecfg,bcp,e,h,up,bpred,provenance=inputs(sh,se,ds,'b0',checker)
            f=dict(gnn_embeddings=e[ev].float(),logp_g=up['gnn_logits'][ev].float().log_softmax(1),logp_l=up['llm_logits'][ev].float().log_softmax(1));hs=h[ev];loc=torch.arange(1000)
            for t in ('B0',*TREATMENTS):
                routed=[]
                for regime in (('baseline',) if t=='B0' else ('fixed','retuned')):
                    label='B0' if t=='B0' else t+'__'+regime;r=by[(key(sh,se,ds),label)];path=Path(r['sources']['model_path']);assert digest(path/'COMPLETE.json')==r['sources']['model_seal_sha256']
                    if t=='B0':cp,reference=bcp,bpred
                    else:check_seal(path,['checkpoint.pt','predictions.pt','config.json','router.pt','INPUTS_BEFORE_TRAINING.json']);cp=load(path/'checkpoint.pt');reference=load(path/'predictions.pt')
                    cfg=r['config'];assert cp['config']==cfg and cp['training_seed']==se and cp['test_truth_loaded'] is False
                    if regime=='fixed':assert cfg==basecfg
                    if regime=='retuned':assert joint_locked(cfg)==joint_locked(basecfg)
                    model=training.make_model(f,hs,cfg);model.load_state_dict(cp['states']['accuracy'])
                    with torch.inference_mode():probs=training.predict(model,f,hs,loc)
                    expected=reference['probability'][ev];torch.testing.assert_close(probs,expected,rtol=0,atol=2e-6)
                    assert torch.equal(probs.argmax(1),expected.argmax(1))
                    acc=float((probs.argmax(1)==truth[ev]).double().mean());assert abs(acc-r['accuracy'])<1e-12
                    a,b=weights(cp);assert a==r['alpha'] and b==r['beta']
                    for field in ('corrective','corrective_own_pGL','conflict_corrective','train_active_corrective','actual_training_pool'):
                        c=r[field];assert c['corrective_count']==c['rescued_count']+c['harmed_count']+c['wrong_to_wrong_count']
                        assert c['net_corrections']==c['rescued_count']-c['harmed_count']
                        assert c['corrective_right_minus_wrong']==c['corrective_correct_count']-c['corrective_incorrect_count']
                        assert c['corrective_correct_count']==c['rescued_count']
                        assert c['corrective_precision']==(c['corrective_correct_count']/c['corrective_count'] if c['corrective_count'] else None)
                        assert c['accepted_correct']-round(c['pGL_accuracy_on_accepted']*c['accepted_count'])==c['net_corrections'] if c['accepted_count'] else c['net_corrections']==0
                    if (path/'INPUTS_BEFORE_TRAINING.json').exists():
                        receipt=read(path/'INPUTS_BEFORE_TRAINING.json');assert receipt['gold_ids']==sp['support_ids'] and receipt['validation_ids']==sp['valid_ids']
                        assert receipt['code']==code_files() and receipt['evaluation_labels_read'] is False
                    if t!='B0':
                        rr=load(path/'router.pt');routed.append(rr)
                    dest=OUT/'training-records'/f'{cellindex:02d}'/(label+'.json')
                    write(dest,dict(key=r['key'],variant=label,config=cfg,checkpoints=cp['checkpoints'],history=cp['history'],model_source=r['sources'],training_seed=cp['training_seed']))
                    proof.append(dict(key=r['key'],variant=label,checkpoint_probability_verified=True,labels_exact_match=True,max_probability_abs_error=float((probs-expected).abs().max()),accuracy=acc))
                    del model,probs
                    if t!='B0':del cp,reference
                if len(routed)==2:
                    assert routed[0].keys()==routed[1].keys()
                    for k in routed[0]:torch.testing.assert_close(routed[0][k],routed[1][k],rtol=0,atol=0)
                    x,y=(by[(key(sh,se,ds),t+'__'+r)] for r in ('fixed','retuned'))
                    for field in ('router_metrics','corrective','query_ids_sha256','eval_ordered_ids_sha256'):assert x[field]==y[field]
                del routed
            del e,h,up,bcp,bpred;gc.collect()
    assert len(proof)==105
    dest=OUT/'independent-audit'/ds;write(dest/'RESULTS.json',dict(all_checks_pass=True,rows=proof,models=105,analysis_code_sha256=digest(__file__),source_registration_sha256=digest(OUT/'REGISTRATION.json'),slurm_job=os.environ['SLURM_JOB_ID']))
    seal(dest,dict(all_checks_pass=True,models=105));print('INDEPENDENT_DATASET_PASS',ds,105,flush=True)

def aggregate():
    registered();proof=[]
    for ds in DS:
        p=OUT/'independent-audit'/ds;check_seal(p);r=read(p/'RESULTS.json');assert r['all_checks_pass'] and r['models']==105
        proof.append(dict(dataset=ds,seal_sha256=digest(p/'COMPLETE.json'),slurm_job=r['slurm_job']))
    report=OUT/'report';check_seal(report);data=read(report/'RESULTS.json')
    for s in data['summary']:
        group=[flat(r) for r in data['rows'] if r['variant']==s['variant'] and (s['dataset']=='ALL' or r['dataset']==s['dataset']) and (s['shot']=='ALL' or r['shot']==s['shot'])]
        assert len(group)==s['cells']
        for m,v in s['metrics'].items():
            a=[r[m] for r in group if r.get(m) is not None];assert len(a)==v['n']
            if a:assert abs(sum(a)/len(a)-v['mean'])<1e-9
            else:assert v['mean'] is None
    check_seal(OUT/'associations')
    states=subprocess.check_output(['sacct','-X','-j','121420,121421,121422,121423,121424,121425,121426,121427','-n','-P','--format=JobID,State,ExitCode,Elapsed'],text=True)
    for line in states.splitlines():
        p=line.split('|')
        if len(p)>=3:assert p[1]=='COMPLETED' and p[2]=='0:0',p
    (OUT/'independent-audit/SLURM_STATES.txt').write_text(states)
    write(OUT/'INDEPENDENT_AUDIT_PASS.json',dict(all_checks_pass=True,models=525,datasets=proof,
        report_seal_sha256=digest(report/'COMPLETE.json'),associations_seal_sha256=digest(OUT/'associations/COMPLETE.json'),
        audit_source_sha256=digest(__file__),training_source_sha256=digest(THIS/'factorial_joint.py'),
        immutable_source_registration_sha256=digest(OUT/'REGISTRATION.json'),all_original_jobs_completed_exit0=True,
        slurm_job=os.environ['SLURM_JOB_ID']))
    print('INDEPENDENT_ALL525_PASS',flush=True)

if __name__=='__main__':
    if sys.argv[1]=='dataset':dataset(int(os.environ['SLURM_ARRAY_TASK_ID']))
    else:aggregate()
