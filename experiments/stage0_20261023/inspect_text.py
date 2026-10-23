from common import *
setup()
for ds in DS:
 x=load(SOURCE/f'runs/router_v7_text_environments_20260815_v2/{ds}/text_environments.pt')
 print(ds,x.keys(),flush=True)
 for env,v in x['environments'].items():
  print(env,[(role,list(b),len(b.get('node_ids',[]))) for role,b in v.items() if isinstance(b,dict)],flush=True)
