"""Verify retained model bytes and export a cloud-reviewable location/hash index."""
from pathlib import Path
import hashlib,json,os,datetime,tarfile
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/synthetic_pseudo_20261031-v1'
PUB=ROOT/'reports/synthetic_pseudo_20261031'
AUDIT=ROOT/'reports/synthetic_pseudo_audit_20261031'
OUT=ROOT/'reports/synthetic_pseudo_weights_index_20261031'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 assert os.environ.get('SLURM_JOB_ID') and not OUT.exists()
 assert read(AUDIT/'COMPLETE.json')['all_checks_pass']
 assert read(PUB/'COMPLETE.json')['all_checks_pass']
 data=read(PUB/'RESULTS.json');index=[]
 for r in data['per_run']:
  p=Path(r['checkpoint_path']);seal=p.parent/'COMPLETE.json';sealed=read(seal)
  assert sealed['all_checks_pass'] and sealed['source_files']==data['files']
  expected=sealed['files']['checkpoint.pt'];assert sha(p)==expected
  assert sha(p.parent/'RESULTS.json')==sealed['files']['RESULTS.json']
  assert read(p.parent/'RESULTS.json')==r
  index.append(dict(run_id=r['run_id'],dataset=r['dataset'],split_seed=r['split_seed'],rule=r['rule'],selection_mode=r['selection_mode'],noise_seed=r['noise_seed'],training_seed=r['training_seed'],checkpoint_path=str(p),bytes=p.stat().st_size,sha256=expected,seal_path=str(seal),seal_sha256=sha(seal)))
 assert len(index)==1850
 OUT.mkdir(parents=True)
 value=dict(all_checks_pass=True,models=1850,binaries_uploaded=False,checkpoint_files=index,source_manifest_sha256=sha(PUB/'PUBLICATION_SHA256.json'),audit_results_sha256=sha(AUDIT/'RESULTS.json'),script_sha256=sha(Path(__file__)),slurm_job=os.environ['SLURM_JOB_ID'],checked_at=datetime.datetime.now().astimezone().isoformat())
 (OUT/'CHECKPOINT_INDEX.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
 (OUT/'COMPLETE.json').write_text(json.dumps(dict(all_checks_pass=True,models=1850,index_sha256=sha(OUT/'CHECKPOINT_INDEX.json'),script_sha256=value['script_sha256']),indent=2)+'\n')
 asset=ROOT/'runs/synthetic_pseudo_20261031-audit.tar.gz';assert not asset.exists()
 with tarfile.open(asset,'w:gz') as t:
  for p in (AUDIT,OUT,Path(__file__).parent):t.add(p,arcname=str(p.relative_to(ROOT)))
 receipt=dict(all_checks_pass=True,models=1850,archive=str(asset),archive_bytes=asset.stat().st_size,archive_sha256=sha(asset),index_sha256=sha(OUT/'CHECKPOINT_INDEX.json'))
 (RUN/'AUDIT_PUBLICATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print('SYNTHETIC_CHECKPOINT_INDEX_PASS',1850,json.dumps(receipt),flush=True)
if __name__=='__main__':main()
