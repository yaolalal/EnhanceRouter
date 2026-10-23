"""Report-only correction; never alters sealed model or prediction artifacts."""
from copy import deepcopy

GOLD_METHOD={'joint_fixed':None,'router_frozen':'gold_frozen','router_finetune':'gold_finetune',
             'gold_frozen':'gold_frozen','gold_finetune':'gold_finetune'}

def correct_rows(rows):
    out=deepcopy(rows)
    key=lambda r:(r['dataset'],r['split_seed'],r['population'],r['method'])
    lookup={key(r):r for r in out}
    if len(lookup)!=len(out):raise ValueError('Duplicate result identity')
    for row in out:
        baseline=GOLD_METHOD[row['method']]
        ref=lookup[(*key(row)[:3],baseline)] if baseline is not None else None
        for metric in ('accuracy','macro_f1'):
            row['delta_gold_'+metric]=row[metric]-ref[metric] if ref else None
    return out
