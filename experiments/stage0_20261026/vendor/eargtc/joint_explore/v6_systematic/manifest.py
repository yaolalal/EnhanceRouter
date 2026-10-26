"""Standard-library-only immutable artifacts and source bootstrap."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO/'runs/joint_v6_systematic_20260911_v1'
OLD_ROOT = REPO/'runs/router_v8_v6_joint_multiseed_20260910_v1'


def slurm_only():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('numerical pipeline work requires Slurm')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def immutable_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def merge_hashes(*mappings):
    out={}
    for mapping in mappings:
        for p,h in mapping.items():
            if p in out and out[p]!=h:
                raise ValueError('conflicting provenance hash: '+p)
            out[p]=h
    return out


def verify_files(records):
    for path,h in records.items():
        if not Path(path).is_file() or sha256(path)!=h:
            raise ValueError('artifact hash mismatch: '+str(path))


def root_path(root):
    root=Path(root).resolve()
    if root!=DEFAULT_ROOT and not root.is_relative_to(Path('/tmp')):
        raise ValueError('use dedicated systematic root or isolated /tmp diagnostic root')
    return root


def training_open(root):
    for name in ('FROZEN.json','EVALUATION_STARTED.json','evaluation'):
        if (Path(root)/name).exists():
            raise ValueError('training is closed after freeze/evaluation')


def loaded_sources():
    out={}
    for module in list(sys.modules.values()):
        raw=getattr(module,'__file__',None)
        if raw:
            p=Path(raw).resolve()
            if p.suffix=='.py' and p.is_relative_to(REPO) and p.is_file():
                out[str(p)]=sha256(p)
    return out


def bootstrap(root,check_loaded=True):
    """Verify code bytes before importing the numeric or legacy adapters.

    This detects audited source changes; it is not a general Python sandbox.
    """
    root=root_path(root)
    seal=read_json(root/'CODE_SEAL.json')
    verify_files(seal['files'])
    if check_loaded:
        for p,h in loaded_sources().items():
            if seal['files'].get(p)!=h:
                raise ValueError('unregistered or changed repository import: '+p)
    return seal


def write_bundle(path,payloads,identity,refs=None):
    """Reserve an exclusive directory. A missing last marker is never consumed."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.mkdir()
    files={}
    for name,value in payloads.items():
        if Path(name).name!=name or name=='complete.json':
            raise ValueError('invalid artifact name')
        dest=path/name
        if name.endswith('.json'):
            immutable_json(dest,value)
        elif name.endswith('.pt'):
            slurm_only()
            import torch
            with dest.open('xb') as f:
                torch.save(value,f); f.flush(); os.fsync(f.fileno())
        else:
            with dest.open('x') as f:
                f.write(value); f.flush(); os.fsync(f.fileno())
        files[str(dest.resolve())]=sha256(dest)
    complete={**identity,'files':files,'refs':dict(refs or {}),'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
    immutable_json(path/'complete.json',complete)
    return complete


def verify_bundle(path,expected=None):
    path=Path(path); marker=read_json(path/'complete.json')
    if expected and any(marker.get(k)!=v for k,v in expected.items()):
        raise ValueError('bundle identity differs')
    actual={str(p.resolve()) for p in path.iterdir() if p.is_file() and p.name!='complete.json'}
    if actual!=set(marker['files']):
        raise ValueError('bundle file coverage differs')
    verify_files(marker['files']); verify_files(marker.get('refs',{}))
    return marker


def bundle_records(path):
    marker=verify_bundle(path)
    return merge_hashes(marker['files'],marker.get('refs',{}),{str(Path(path).resolve()/'complete.json'):sha256(Path(path)/'complete.json')})


def tensor_load(path):
    slurm_only()
    import torch
    return torch.load(path,map_location='cpu',weights_only=True)
