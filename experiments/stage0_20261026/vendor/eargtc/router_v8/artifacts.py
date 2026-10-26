"""Atomic artifacts and fail-closed all-dataset freeze."""
import hashlib
import json
import os
from pathlib import Path


def write_json(value,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    os.replace(temporary,path)


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def freeze_paths(paths,destination,metadata=None):
    destination=Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    hashes={str(Path(p).resolve()):sha256(p) for p in sorted(set(map(str,paths)))}
    write_json(dict(schema='router_v8_freeze_v1',hashes=hashes,metadata=metadata or {}),destination)


def verify_frozen(path):
    frozen=json.loads(Path(path).read_text())
    for file,expected in frozen['hashes'].items():
        if not Path(file).is_file() or sha256(file)!=expected:
            raise ValueError(f'frozen hash mismatch: {file}')
    return frozen


def begin_once(path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:
        handle.write('Single locked evaluation started. Never tune using these test labels.\n')
