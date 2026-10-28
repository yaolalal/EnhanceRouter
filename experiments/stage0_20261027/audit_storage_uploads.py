"""Slurm-only, read-only audit of migration, instructions and published results."""
from pathlib import Path
import os,json,hashlib,tarfile,collections,sys
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/stage0-20261026-v2'
OLD='/home/wujingyao/codes/TAG_Fewshot_Learning/GNN-as-Judge'
NEW='/data1/wujingyao-20354/wujingyao/codes/GNN-as-judge'
def read(p):return json.loads(Path(p).read_text())
def sha(p,kind='sha256'):
    h=hashlib.new(kind)
    with Path(p).open('rb') as fp:
        for b in iter(lambda:fp.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def require(condition,message):
    if not condition:raise AssertionError(message)
def main(out):
    require(bool(os.environ.get('SLURM_JOB_ID')),'Slurm required')
    require(out.resolve().is_relative_to(ROOT),'output outside project')
    amendment=read(ROOT/'provenance/full-query-amendment-20261027/AMENDMENT.json')
    frozen=read(RUN/'CODE_FREEZE.json')['files']
    for rel,expected in frozen.items():
        if rel in amendment['files']:
            r=amendment['files'][rel]
            require(r['original_sha256']==expected and sha(ROOT/r['archived_original'])==expected,rel)
            require(sha(ROOT/rel)==r['current_sha256'],rel)
        else:require(sha(ROOT/rel)==expected,rel)
    revision=amendment['current_document_revision']
    previous=ROOT/revision['previous_registration_archive']
    require(sha(previous)==revision['previous_registration_sha256'],'prior amendment archive')
    prior=read(previous)
    for rel in ('Agent.md','AGENTS.md'):
        p=ROOT/revision['previous_current_instructions_archive']/rel
        require(sha(p)==prior['files'][rel]['current_sha256'],'prior instruction archive '+rel)
    require((ROOT/'Agent.md').is_symlink() and os.readlink(ROOT/'Agent.md')=='AGENTS.md','instruction alias')
    require(os.path.samefile(ROOT/'Agent.md',ROOT/'AGENTS.md'),'single instruction source')
    require(Path(OLD).is_symlink() and Path(OLD).resolve()==Path(NEW).resolve(),'migration alias')
    sources={}
    for name in ('SOURCE_FILES.json','SOURCE_EXTENSION_FILES_V2.json','HEAD_SOURCE_FILES.json'):
        if (RUN/name).exists():sources.update(read(RUN/name))
    migrated=[]
    for path,expected in sources.items():
        if not path.startswith(OLD+'/'):continue
        source=Path(path);target=Path(NEW)/source.relative_to(OLD)
        require(source.is_file() and target.is_file() and os.path.samefile(source,target),'migrated source '+path)
        for kind in ('md5','sha256'):
            if kind in expected:require(sha(target,kind)==expected[kind],kind+' '+path)
        migrated.append(dict(original=path,current=str(target),resolved=str(target.resolve()),
                             bytes=target.stat().st_size,sha256=sha(target),unchanged=True))
    require(len(migrated)>=8,'missing graph/split dependencies')
    config=read(ROOT/'configs/inputs.json');graphs=[]
    for key,c in config['cells'].items():
        p=Path(c['graph']);require(p.is_file() and p.is_relative_to(OLD),'graph reference '+key)
        graphs.append(dict(key=key,path=str(p),resolved=str(p.resolve())))
    require(len(graphs)==45,'historical graph reference count')
    published=[]
    for directory in ('all-results-20261027','full-query-20261027'):
        base=ROOT/'reports'/directory;manifest=read(base/'PUBLICATION_SHA256.json')
        for rel,h in manifest.items():require(sha(base/rel)==h,'publication '+directory+'/'+rel)
        published.append(dict(directory=directory,files_checked=len(manifest),manifest_sha256=sha(base/'PUBLICATION_SHA256.json')))
    index=read(ROOT/'reports/all-results-20261027/raw-outputs/INDEX.json')
    full=read(ROOT/'reports/full-query-20261027/RESULTS.json')
    require(len(index['entries'])==len(full['per_cell'])==1350,'all outputs present')
    require({(r['key'],r['variant']) for r in index['entries']}=={(r['key'],r['variant']) for r in full['per_cell']},'publication cohorts differ')
    for row in index['entries']:
        require(sha(Path(row['source_path'])/'COMPLETE.json')==row['seal_sha256'],'model seal '+row['source_path'])
    assets=[dict(name='experiment-results.tar.gz',path=index['archive'],size=117102810,
                 digest='5bc29714a3c1baefa0bbd0767436396ff303e8369a9889eb99b178440e7592ff'),
            dict(name='full-query-metrics-20261028.tar.gz',path=str(RUN/'full-query-metrics-20261028.tar.gz'),size=2349753,
                 digest='a4a27964234da42423e7af350dfd2e4147f6039e74a327bb113ed676ef677c1e')]
    for a in assets:
        require(Path(a['path']).stat().st_size==a['size'] and sha(a['path'])==a['digest'],'release bytes '+a['name'])
    # Stream once; compare every archived config/diagnostic/export to its manifest.
    hashes={};manifest=None
    with tarfile.open(index['archive'],'r|gz') as tar:
        for member in tar:
            if not member.isfile():continue
            data=tar.extractfile(member).read()
            if member.name=='MANIFEST.json':manifest=json.loads(data)
            hashes[member.name]=hashlib.sha256(data).hexdigest()
    require(manifest is not None and len(manifest['artifacts'])==1350,'archive manifest')
    router_exports=final_exports=0;text_files=0
    for item in manifest['artifacts']:
        sh,se,ds=item['key'].split('/');prefix=f"shot{sh}/seed{se}/{ds}/{item['variant']}/"
        require(hashes[prefix+'COMPLETE.json']==item['seal_sha256'],'archived completion seal')
        for name,h in (item['exported_text_files']|item['exports']).items():
            require(hashes[prefix+name]==h,'archive member '+prefix+name);text_files+=1
        router_exports+=int('eval1000-router-scores.csv' in item['exports'])
        final_exports+=int('eval1000-final-predictions.csv' in item['exports'])
    require(router_exports==1350 and final_exports==600,'per-node export counts')
    counts=collections.Counter(r['variant'].split('/')[0] for r in full['per_cell'])
    result=dict(all_checks_pass=True,slurm_job=os.environ['SLURM_JOB_ID'],audit_code_sha256=sha(__file__),
                instructions=dict(canonical='AGENTS.md',compatibility_symlink='Agent.md -> AGENTS.md',old_freeze_files=len(frozen),old_freeze_unchanged=True),
                migration=dict(old=OLD,current=NEW,compatibility_symlink_valid=True,source_files=migrated,historical_graph_references=graphs),
                publication_manifests=published,models=1350,variant_counts=dict(counts),model_seals_checked=1350,
                release_assets=assets,archive_members=len(hashes),archived_text_and_export_files_checked=text_files,
                eval1000_router_exports=router_exports,eval1000_final_exports=final_exports,
                full_query_router_metric_records=1350,final_classification_population='unchanged ordered eval1000',
                uploaded='All published reports, per-cell metrics, model configs/diagnostics/selection text, existing eval1000 per-node exports and full-Query metric correction',
                server_only='Binary model weights, full graph caches and original full-Query binary scores; paths/hashes retained in published manifests',
                numerical_results_changed=False,training_performed=False,monitor='PAUSED')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as fp:json.dump(result,fp,indent=2,ensure_ascii=False);fp.write('\n')
    print('STORAGE_UPLOAD_AUDIT_PASS',out,'models',1350,'router_csv',router_exports,'final_csv',final_exports,flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))
