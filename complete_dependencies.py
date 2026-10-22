"""Complete static imports of copied historical packages, including script helpers."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

R=Path(__file__).resolve().parent
S=Path('/home/wujingyao/codes/TAG_Fewshot_Learning/EALR-GTC')
manifest=json.loads((R/'provenance/copied-sources.json').read_text())
records=manifest['files']


def path(name):
    if name.split('.')[0] not in ('eargtc','scripts'):return None
    p=S/Path(*name.split('.'))
    if p.with_suffix('.py').is_file():return p.with_suffix('.py')
    if (p/'__init__.py').is_file():return p/'__init__.py'
    return None


todo=['.'.join(p.relative_to(R/'src').with_suffix('').parts) for p in (R/'src/eargtc').rglob('*.py')]
seen=set();added=[]
while todo:
    name=todo.pop()
    if name.endswith('.__init__'):name=name[:-9]
    if name in seen:continue
    seen.add(name);p=path(name)
    if p is None:continue
    dest=R/'src'/p.relative_to(S)
    if not dest.exists():
        dest.parent.mkdir(parents=True,exist_ok=True);source=p.read_bytes()
        data=source.replace(str(S).encode(),str(R).encode());dest.write_bytes(data)
        records[str(p)]=dict(destination=str(dest.relative_to(R)),source_md5=hashlib.md5(source).hexdigest(),
              source_sha256=hashlib.sha256(source).hexdigest(),copied_md5=hashlib.md5(data).hexdigest(),
              copied_sha256=hashlib.sha256(data).hexdigest(),transformation='copy and relocate absolute project constants')
        added.append(str(dest.relative_to(R)))
    pkg=name if p.name=='__init__.py' else name.rpartition('.')[0]
    for node in ast.walk(ast.parse(p.read_text())):
        names=[]
        if isinstance(node,ast.Import):names=[a.name for a in node.names]
        elif isinstance(node,ast.ImportFrom):
            base=node.module or ''
            if node.level:
                try:base=importlib.util.resolve_name('.'*node.level+base,pkg)
                except (ValueError,ImportError):continue
            names=[base]+[base+'.'+a.name for a in node.names if a.name!='*']
        todo.extend(n for n in names if path(n))
    parts=name.split('.')
    for i in range(1,len(parts)):
        parent='.'.join(parts[:i])
        if path(parent):todo.append(parent)
        else:
            init=R/'src'/Path(*parts[:i])/'__init__.py'
            init.parent.mkdir(parents=True,exist_ok=True)
            if not init.exists():init.write_text('')
manifest['files']=records
manifest['dependency_completion']=added
(R/'provenance/copied-sources.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
source=json.loads((R/'provenance/source.json').read_text())
source['copied_sources_manifest_sha256']=hashlib.sha256((R/'provenance/copied-sources.json').read_bytes()).hexdigest()
source['source_tree_fingerprint_sha256']=hashlib.sha256(json.dumps({p:r['source_sha256'] for p,r in sorted(records.items())},sort_keys=True).encode()).hexdigest()
(R/'provenance/source.json').write_text(json.dumps(source,indent=2,ensure_ascii=False)+'\n')
print('Added',len(added),'dependency modules:',added)
