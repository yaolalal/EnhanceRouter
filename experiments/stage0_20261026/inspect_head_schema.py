from common import *
setup()
for ds in DS:
 p=SOURCE/f'runs/gcn_ssl_v2_selected_20260623_v1/{ds}/mlp_head.pt';x=load(p)
 print('HEAD_SCHEMA',ds,list(x),{k:list(v)[:8] for k,v in x.items() if isinstance(v,dict)},flush=True)
