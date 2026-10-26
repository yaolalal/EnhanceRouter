from common import *
setup()
import joblib
idx=read(ROOT/'configs/inputs.json')
for shot in SHOTS:
 for ds in DS:
  c=idx['cells'][f'{shot}/42/{ds}']
  for branch in ('agreement','disagreement'):
   b=joblib.load(c['branches'][branch]['bundle'])
   print('BUNDLE',shot,ds,branch,json.dumps({k:v for k,v in b.items() if k not in ('models',)},default=str),flush=True)
   for h,w in b.get('models',{}).items():
    m=w['model'];f=m.get('fitted',{})
    print('MODEL',shot,ds,h,'engine',w['engine'],'wrapper_keys',list(w),'keys',list(m),'config',m['config'],'fitted_keys',list(f),flush=True)
    for k,v in {**m,**f}.items():
     if any(t in k.lower() for t in ('transform','scaler','proj','prep')):print('TRANSFORM',k,type(v).__name__,repr(v)[:1200],flush=True)
print('AUDIT_COMPLETE',flush=True)
