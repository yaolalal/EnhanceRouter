from common import *
setup()
from datetime import datetime,timezone
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
state=dict(stage0_complete=(RUN/'report/COMPLETE.json').exists(),historical_exact_cells=45,
 standard_heads_complete=len(list((RUN/'cells').glob('shot*/seed*/*/standard/HEAD_COMPLETE.json'))),
 D3_stratified_heads_complete=len(list((RUN/'cells').glob('shot*/seed*/*/small_stratified/HEAD_COMPLETE.json'))),
 initial_text_datasets_complete=len(list((RUN/'shared/text').glob('*.json'))),
 dev_B0_complete=len(list((RUN/'cells').glob('shot*/seed*/*/b0/COMPLETE.json'))),
 dev_D3_complete=len(list((RUN/'cells').glob('shot*/seed*/*/d3/COMPLETE.json'))),
 dev_diagnostics_complete=len(list((RUN/'cells').glob('shot*/seed*/*/d4/confidence_preference/COMPLETE.json'))),
 heldout_evaluations=0,stage1_decision='pending full D2b results',timestamp_utc=stamp,
 current_jobs=dict(heads=118690,D3_stratified_heads=118801,text=118790,extra_text=118791,banks=118737,baseline=118738,oracle_selection=118739,oracle_banks=118743,diagnostics=118744,report=118745),
 queue=subprocess.check_output(['squeue','-u','wujingyao','-h','-o','%i|%j|%T|%M|%R'],text=True),metadata=metadata())
write(RUN/f'STATUS-{stamp}.json',state)
print(json.dumps({k:v for k,v in state.items() if k not in ('metadata','queue')},indent=2),flush=True)
