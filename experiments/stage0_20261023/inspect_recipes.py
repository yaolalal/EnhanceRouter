from common import *
setup()
import joblib
idx=read(ROOT/'configs/inputs.json')
for shot in SHOTS:
 for ds in DS:
  c=idx['cells'][f'{shot}/42/{ds}']
  for branch in ('agreement','disagreement'):
   b=joblib.load(c['branches'][branch]['bundle'])
   print(shot,ds,branch,json.dumps({h:dict(engine=w['engine'],config=w['model']['config']) for h,w in b.get('models',{}).items()}),flush=True)
