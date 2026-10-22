"""Run-local writes, content identities, and reproducibility receipts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ('cora', 'citeseer', 'pubmed', 'arxiv', 'ogbn-products')
SHOTS = (3, 5, 10)
SEEDS = (42, 43, 44)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def local(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Output outside EnhanceRouter: {path}')
    return path


def write(path, value):
    path = local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def save(path, value):
    import torch
    path = local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        torch.save(value, f)


def command(args):
    p = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    return dict(returncode=p.returncode, stdout=p.stdout.strip(), stderr=p.stderr.strip())


def require_slurm():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use ./reproduce.sh; numerical work requires Slurm')


def guard_writes():
    """Block Python writes outside this project, including frozen input paths."""
    def check(path):
        if isinstance(path, (str, bytes, os.PathLike)):
            p = Path(os.fsdecode(path)).resolve()
            if p != Path('/dev/null') and not p.is_relative_to(ROOT):
                raise PermissionError(f'EnhanceRouter write boundary: {p}')

    def audit(event, args):
        if event == 'open':
            path, mode, flags = args
            if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            ):
                check(path)
        elif event in ('os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.truncate'):
            check(args[0])
        elif event in ('os.rename', 'os.link', 'os.symlink'):
            check(args[0]); check(args[1])
    sys.addaudithook(audit)


def source_state():
    files = sorted((ROOT / 'src').rglob('*.py')) + sorted((ROOT / 'scripts').glob('*'))
    files += [ROOT / 'reproduce.sh', ROOT / 'configs/experiment.json', ROOT / 'configs/inputs.json']
    return {str(p.relative_to(ROOT)): digest(p) for p in files if p.is_file()}


def environment():
    import platform
    import torch
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                hostname=platform.node(), torch=str(torch.__version__),
                torch_threads=torch.get_num_threads(), torch_config=torch.__config__.show(),
                pip_freeze=command([sys.executable, '-m', 'pip', 'freeze']),
                conda_list=command(['/home/wujingyao/miniconda3/bin/conda', 'list', '--prefix', sys.prefix, '--json']),
                git_commit=command(['git', 'rev-parse', 'HEAD']),
                git_status=command(['git', 'status', '--porcelain']),
                slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_')},
                settings={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS','PYTHONHASHSEED','PYTHONPATH','PYTHONDONTWRITEBYTECODE','TMPDIR')})


class Inputs:
    def __init__(self, run):
        self.run = local(run)
        self.index = read(ROOT / 'configs/inputs.json')
        self.lock = read(self.run / 'frozen-inputs.json')
        self.checked = set()

    def path(self, path):
        path = str(path)
        record = self.lock['files'][path]
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (record['size'], record['mtime_ns']):
            raise ValueError(f'Frozen input metadata changed: {path}')
        if path not in self.checked:
            if digest(path, 'md5') != record['md5']:
                raise ValueError(f'Frozen input MD5 changed: {path}')
            self.checked.add(path)
        return Path(path)

    def tensor(self, path):
        import torch
        return torch.load(self.path(path), map_location='cpu', weights_only=True, mmap=True)

    def split(self, key):
        c = self.index['cells'][key]
        split = read(self.path(c['inputs']['split']))
        shot, seed, ds = key.split('/')
        assert (split['shots'], split['split_seed'], split['dataset']) == (int(shot), int(seed), ds)
        groups = [set(split[k]) for k in ('support_ids','valid_ids','unlabeled_ids')]
        assert not any(groups[i] & groups[j] for i in range(3) for j in range(i))
        assert set.union(*groups) == set(range(split['num_nodes']))
        standard = self.index['standard_eval_ids'][f'{seed}/{ds}']
        ten = read(self.path(self.index['cells'][f'10/{seed}/{ds}']['inputs']['split']))
        assert standard == ten['evaluation_1000_ids']
        assert len(standard) == len(set(standard)) == 1000 and set(standard) <= groups[2]
        assert not set(standard) & (groups[0] | groups[1])
        return split


def assert_imports_isolated():
    source = Path(read(ROOT / 'provenance/source.json')['source_root'])
    for name, module in sys.modules.copy().items():
        p = getattr(module, '__file__', None)
        if p and Path(p).resolve().is_relative_to(source):
            raise RuntimeError(f'Executed code from EALR-GTC: {name}: {p}')
