"""Hash-bound reuse of sealed V1 inputs; no query truth loader in training."""
import itertools,os,sys
from pathlib import Path
import torch
from eargtc.joint_explore.joint_purf_v10_v1.protocol import read,write,save,sha,check,slurm
from eargtc.multiseed_joint.experiment import prepare_arrays
from eargtc.joint_cross_repair.artifacts import validate_hidden

REPO=Path(__file__).resolve().parents[3]
RUN=REPO/'runs/joint_purf_v10_v2_20260914'
PREVIOUS=REPO/'runs/joint_purf_v10_v1_20260914'
DATASETS=('cora','citeseer','pubmed','arxiv','ogbn-products')
SEEDS=(42,43,44,45,46)
CELLS=tuple(itertools.product(SEEDS,DATASETS))

def cell(seed,ds):return RUN/f'seed{seed}'/ds

def merge_records(*groups):
    out={}
    for group in groups:
        for path,digest in group.items():
            if path in out and out[path]!=digest:raise ValueError('Conflicting provenance: '+path)
            out[path]=digest
    return out

def training_open():
    slurm()
    if any((RUN/f).exists() for f in ('FROZEN.json','EVALUATION_STARTED.json','COMPLETE.json')):
        raise RuntimeError('Fitting and selection closed after global freeze')

def verify_code():
    slurm();r=read(RUN/'REGISTRATION.json');check(r['code_hashes'])
    sources=merge_records(r['protected'],r['dependencies']);check({p:h for p,h in sources.items() if p.endswith('.py')})
    return r

def verify_all():
    r=verify_code();check(merge_records(r['protected'],r['dependencies']))
    return r

def imbalance(counts):
    positives=[n for n in counts if n>0]
    return bool(positives) and (len(positives)<len(counts) or max(positives)>=2*min(positives))

def register():
    training_open()
    complete=read(PREVIOUS/'COMPLETE.json');audit=read(PREVIOUS/'FINAL_AUDIT.json')
    if not complete['all_checks_pass'] or not audit['all_checks_pass'] or audit['inner_seeds']!=[]:raise ValueError('V1 not complete')
    seal={str(PREVIOUS/k):v for k,v in complete['files'].items()};check(seal)
    if sha(PREVIOUS/'FROZEN.json')!=audit['frozen_sha256']:raise ValueError('V1 seal differs')
    frozen=read(PREVIOUS/'FROZEN.json')
    original=PREVIOUS/'original';regpath=original/'REGISTRATION.json'
    check({str(regpath):frozen['files'][str(regpath)]});old=read(regpath)
    protected=merge_records(old['protected'],old['code_hashes'],frozen['files'],seal,
                            {str(PREVIOUS/'COMPLETE.json'):sha(PREVIOUS/'COMPLETE.json')})
    dependencies=old['dependencies'];check(merge_records(protected,dependencies))
    records={}
    for seed,ds in CELLS:
        record=old['inputs'][f'{seed}/{ds}'];check(record['hashes']);split=read(record['paths']['split'])
        historical=REPO/f'runs/router_v10_multiseed_data_20260912_v1/seed{seed}/splits/{ds}_5shot_seed{seed}.json'
        if Path(record['paths']['split']).read_bytes()!=historical.read_bytes():raise ValueError('Historical split differs')
        if split['split_seed']!=seed or split['dataset']!=ds or split['test_truth_exported'] is not False:raise ValueError('Split contract differs')
        selection_path=original/f'seed{seed}/{ds}/final/selection.json';selected=read(selection_path)
        cp=original/f'seed{seed}/{ds}/candidates'/selected['candidate_id']/'checkpoint.pt'
        if selected['training_seed']!=seed or sha(cp)!=selected['candidate_sha256']:raise ValueError('Inherited checkpoint identity differs')
        if str(cp) not in protected or protected[str(cp)]!=selected['candidate_sha256']:raise ValueError('Unsealed inherited checkpoint')
        records[f'{seed}/{ds}']=dict(record,baseline_checkpoint=str(cp),baseline_selection=str(selection_path),
            baseline_predictions=str(original/f'seed{seed}/{ds}/final/predictions.pt'),
            fixed_predictions=str(original/f'seed{seed}/{ds}/fixed/predictions.pt'))
    paths=list((REPO/'eargtc/joint_explore/joint_purf_v10_v2').glob('*.py'))
    paths+=list((REPO/'scripts/joint_explore/joint_purf_v10_v2').glob('*'))
    paths+=[REPO/'docs/superpowers/plans/2026-09-14-joint-purf-v10-v2.md']
    code={str(p):sha(p) for p in paths if p.is_file()}
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and Path(path).is_absolute() and Path(path).is_file() and Path(path).is_relative_to(REPO) and str(path).endswith('.py'):
            if str(path) not in protected and str(path) not in dependencies:code[str(path)]=sha(path)
    write(RUN/'REGISTRATION.json',dict(code_hashes=code,protected=protected,dependencies=dependencies,inputs=records,
        seeds=list(SEEDS),datasets=list(DATASETS),inner_seeds=[],training_seed_equals_split_seed=True,
        router='Router-v10-MS-Original',pseudo_labels_unchanged=True,cross_split_selection=False,test_truth_loaded=False,
        previous_complete_sha256=sha(PREVIOUS/'COMPLETE.json'),slurm_job_id=os.environ['SLURM_JOB_ID']))
    print('REGISTERED25 fixed-input cells',flush=True)

def load(seed,ds):
    reg=verify_code();record=reg['inputs'][f'{seed}/{ds}'];check(record['hashes']);paths=record['paths'];split=read(paths['split'])
    arrays={k:torch.load(paths[k],map_location='cpu',weights_only=True,mmap=True) for k in ('upstream','embeddings','supervision','router')}
    f,s,p=prepare_arrays(split,arrays['upstream'],arrays['embeddings'],arrays['supervision'],arrays['router'],ds)
    hidden=torch.load(paths['hidden'],map_location='cpu',weights_only=True,mmap=True)
    validate_hidden(hidden,split['num_nodes'])
    if hidden['metadata']['dataset']!=ds:raise ValueError('Hidden dataset differs')
    return f,s,p,hidden['hidden'],split,arrays['router']

def diagnostic(f,s,p,prob):
    c=f['logp_g'].shape[1]
    labels=torch.cat([s['gold_targets'],f['selected_pred'][p['teach_g']],f['selected_pred'][p['teach_l']]])
    counts=torch.bincount(labels,minlength=c).tolist();pred=prob.argmax(1);y=s['selection_targets']
    recall=[];valid_counts=[]
    for k in range(c):
        mask=y==k;valid_counts.append(int(mask.sum()));recall.append(float((pred[mask]==k).float().mean()) if mask.any() else None)
    return dict(training_class_counts=counts,class_weight_eligible=imbalance(counts),
        validation_class_counts=valid_counts,baseline_validation_class_recall=recall,
        training_label_sources=['gold_support','accepted_teach_g','accepted_teach_l'],
        validation_diagnostic_used_for_class_weights=False,test_truth_loaded=False)
