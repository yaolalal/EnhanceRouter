"""Current registered protocol status; counts of artifacts are not quality claims."""
from common import *
setup()
from datetime import datetime,timezone
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def complete(path):return (Path(path)/'COMPLETE.json').is_file()
def count_variant(name):return sum(complete(variant_folder(sh,se,d,name)) for sh,se,d in CELLS)
report=RUN/'report/COMPLETE.json'
state=dict(stage0_complete=report.is_file() and read(report).get('all_checks_pass') is True,seeds=list(DEV),reserved_seeds=[],timestamp_utc=stamp,
 split_audit_pass=read(RUN/'SPLIT_AUDIT.json')['all_checks_pass'],B0_sealed_cells=count_variant('b0'),D1_sealed_cells_by_fraction={str(f):count_variant(f'd1/{f}') for f in (.25,.5,.75,1.)},
 new_standard_native_heads=sum((folder(sh,se,d)/'standard_native/HEAD_COMPLETE.json').exists() for sh,se,d in CELLS),D3_small_native_heads=sum((folder(sh,se,d)/'small_native/HEAD_COMPLETE.json').exists() for sh,se,d in CELLS),
 D3_sealed_cells=count_variant('d3'),D4_confidence_router_cells=count_variant('d4/confidence_router'),D4_confidence_preference_cells=count_variant('d4/confidence_preference'),D2_sealed_cells_by_arm_N={f'{a}/{n}':count_variant(f'd2{a}/{n}') for a in ('a','b') for n in (50,200,1000)},
 metric_protocol_registered=(RUN/'METRIC_BUDGET_REGISTRATION.json').exists(),native_search_test_pass=(ROOT/'reports/audit-20261026/NATIVE_SEARCH_TEST.json').exists(),stage1_decision='pending completed Stage0 diagnostics',query_evaluation_outcomes_read=False,source=metadata())
for name in ('REMAINDER_JOB_GRAPH.json','GAMMA_JOB_GRAPH_AMENDMENT.json','NATIVE_SEARCH_JOB_AMENDMENT.json'):
 if (RUN/name).exists():state[name]=read(RUN/name)
state['queue']=subprocess.check_output(['squeue','-u','wujingyao','-h','-o','%i|%j|%T|%M|%R'],text=True)
write(RUN/f'STATUS-{stamp}.json',state)
print(json.dumps({k:v for k,v in state.items() if k not in ('source','queue')},indent=2),flush=True)
