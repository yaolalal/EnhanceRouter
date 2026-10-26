from common import *
setup()
idx=read(RUN/'HISTORICAL_INPUTS.json');original=read(ROOT/'configs/inputs.json')
p=SOURCE/'runs/router_v10_observed_best_five_seeds_20260913_v1/CHOICES.json';choices=read(p);lookup={(x['seed'],x['dataset'],x['branch']):x for x in choices['rows']}
for seed in (42,43,44):
 for ds in DS:
  for b in ('agreement','disagreement'):
   assert original['cells'][f'5/{seed}/{ds}']['branches'][b]['bundle']==lookup[(seed,ds,b)]['model_source']
extra=read(RUN/'SOURCE_EXTENSION_FILES.json');changes=[]
for seed in (45,46):
 for ds in DS:
  c=idx['cells'][f'5/{seed}/{ds}']
  for b in ('agreement','disagreement'):
   row=lookup[(seed,ds,b)];src=row['model_source'];assert digest(src)==row['model_sha256']
   before=c['branches'][b]['bundle'];c['branches'][b]['bundle']=src
   extra[src]=dict(md5=digest(src,'md5'),sha256=digest(src))
   changes.append(dict(seed=seed,dataset=ds,branch=b,source=src,previous_derived_guess=before))
extra[str(p)]=dict(md5=digest(p,'md5'),sha256=digest(p))
write(RUN/'HISTORICAL_INPUTS_V2.json',idx);write(RUN/'SOURCE_EXTENSION_FILES_V2.json',extra)
write(RUN/'EXACT_FIVE_SHOT_REGISTRATION.json',dict(all_checks_pass=True,known42_44_paths_match=True,changes=changes,source_choices=str(p),source_choices_sha256=digest(p),no_new_selection=True,
 historical_selection_uses_query_truth=choices['selection_uses_query_truth'],scope='Replay exact deployed historical baseline, including its recorded selection provenance. No new selection from evaluation labels.',supersedes='HISTORICAL_INPUTS.json guessed45/46 model paths; no model fitted with those guesses'))
print('EXACT5_REGISTERED',len(changes),flush=True)
