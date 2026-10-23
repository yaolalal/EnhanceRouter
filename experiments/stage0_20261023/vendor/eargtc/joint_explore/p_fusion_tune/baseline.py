"""Singleton-seed execution of unchanged historical Fixed V2 routines."""
from dataclasses import replace
import torch
from .protocol import SOURCE, PREVIOUS, cell, load, save, read, write, sha, verify


def run_fixed(seed, ds, f, s, pools, hidden, replay=False):
    from eargtc.joint_cross_repair.training import TrainConfig, scheme_b_probability, blend, _score
    from eargtc.joint_cross_repair_search.bank import RECIPES, train_member, member_logits
    from eargtc.joint_cross_repair_search.experiment import candidate_bank
    from eargtc.joint_cross_repair_search.selection import residual_probability
    from eargtc.joint_five_router.experiment import select_candidates, _check_member
    cfg = replace(TrainConfig(**read(SOURCE/'joint/CONFIG.json')['train']), seed=seed)
    torch.set_num_threads(cfg.threads)
    dest = cell(seed, ds) / 'fixed'
    source = SOURCE / f'seed{seed}/joint/r9_new/{ds}'
    receipt = read(source/'complete.json')
    members, origins = {}, {}
    for recipe in RECIPES:
        target = dest/'bank'/f'{recipe.name}.pt'
        if seed <= 44:
            path = source/f'bank/inner{seed}/{recipe.name}.pt'
            verify({str(path):receipt['artifacts'][str(path.relative_to(source))]})
            member = load(path)
            origins[recipe.name] = {'source':str(path), 'sha256':sha(path), 'reused':True}
        elif (PREVIOUS/f'seed{seed}/{ds}/fixed/bank/{recipe.name}.pt').exists():
            path=PREVIOUS/f'seed{seed}/{ds}/fixed/bank/{recipe.name}.pt'
            old_receipt=read(PREVIOUS/f'seed{seed}/{ds}/COMPLETE.json')
            verify({str(path):old_receipt['files'][str(path)]})
            member=load(path)
            origins[recipe.name]={'source':str(path),'sha256':sha(path),'reused':True}
        elif target.exists():
            marker = read(target.with_suffix('.json'))
            verify({str(target):marker['sha256']})
            member = load(target)
            origins[recipe.name] = {'source':str(target), 'sha256':sha(target), 'reused':False}
        else:
            raise ValueError('Expected authenticated prior Fixed member; no new baseline fits permitted')
        _check_member(member, cfg, recipe, s)
        if member['config']['seed'] != seed:
            raise ValueError('Fixed member seed differs from outer')
        members[(seed, recipe.name)] = member
    candidates = candidate_bank(members, f, s, [seed])
    pg, pl = f['logp_g'].exp(), f['logp_l'].exp()
    pb = scheme_b_probability(pg, pl, f['choose_g'], f['agreement'])
    selection = select_candidates(candidates, pb[s['selection_ids']], s['selection_targets'])
    catalog = {r['id']:r for rows in candidates.values() for r in rows}
    cache = {}
    def full(cid):
        if cid not in cache:
            m = catalog[cid]['metadata']
            base = f['logp_'+m['branch']]
            value = base.exp() if m['base'] else residual_probability(base, member_logits(
                members[(seed,m['recipe'])], m['branch'], m['criterion'], f, hidden, f['node_ids'], cfg), m['scale'])
            ref = catalog[cid]['probability']
            if not torch.allclose(value[s['selection_ids']], ref, atol=2e-6, rtol=0):
                raise ValueError('Fixed singleton candidate replay differs')
            cache[cid] = value
        return cache[cid]
    winner = selection['optimized']['family']
    if winner in selection['families']:
        pick = selection['families'][winner]
        prediction = blend(full(pick['g_id']), full(pick['l_id']), pb, pick['alpha'], pick['beta'])
    elif winner == 'router':
        prediction = pb
    elif winner == 'base_fusion':
        pick = selection['base_fusion']
        prediction = blend(pg, pl, pb, pick['alpha'], pick['beta'])
    else:
        raise ValueError('Unexpected Fixed winner: '+winner)
    validation = _score(prediction[s['selection_ids']], s['selection_targets'])
    if abs(validation['accuracy']-selection['optimized']['accuracy']) > 1e-10:
        raise ValueError('Fixed singleton selection fails replay')
    if not replay:
        save(dest/'validation.pt', dict(candidates=candidates, router=pb[s['selection_ids']],
                                       ids=s['selection_ids'], targets=s['selection_targets']))
        write(dest/'selection.json', selection)
        write(dest/'origins.json', origins)
    return prediction, dict(seed=seed, inner_seeds=[], origins=origins, validation=validation,
                            selected_family=winner)
