from common import *
setup()
idx=read(RUN/'HISTORICAL_INPUTS_V2.json');files={}
for shot,seed,ds in CELLS:
 if seed>44 and shot!=5:continue
 c=idx['cells'][key(shot,seed,ds)];p=Path(c['inputs']['embeddings']).parent/'gnn_head.pt';assert p.is_file()
 files[str(p)]=dict(md5=digest(p,'md5'),sha256=digest(p))
write(RUN/'HEAD_SOURCE_FILES.json',files)
print('HEAD_SOURCES',len(files),flush=True)
