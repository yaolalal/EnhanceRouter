"""Build gold-only v6 banks and v7 raw feature views; never fit Router heads."""
import csv,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from eargtc.router_augmentation_v8.sources import *
from eargtc.router_augmentation_v8.data import ENVIRONMENTS,organize,head_view
from eargtc.router_v8.evidence import probability_evidence,structural_evidence
from eargtc.router_v8.prototypes import fit_class_prototypes,prototype_evidence
from eargtc.router_v8_v7_explore.core_v2 import raw_features


def build70(pg,pl,ids,edge,full_pred,embedding,fitted,n,c):
    g,l=pg.argmax(1),pl.argmax(1)
    p=probability_evidence(pg,pl,g,l)
    s=structural_evidence(edge_index=edge,full_gnn_pred=full_pred,row_ids=ids,candidate_g=g,candidate_l=l,num_nodes=n,num_classes=c)
    r=prototype_evidence(embeddings=embedding,row_ids=ids,candidate_g=g,candidate_l=l,fitted=fitted)
    return torch.cat([torch.cat((p[i],s[i],r[i]),1) for i in range(3)],1)


def build_dataset(dataset):
    source=gold_source(dataset);ids,y=source['ids'],source['y'];raw=source['raw']
    n,c=source['meta']['node_count'],source['meta']['class_count']
    oldmeta=read(GOLD/dataset/'manifest.json');assert ids.tolist()==oldmeta['gold_ids']
    paths=[SAFE/dataset/'router_tensors.pt',SAFE/dataset/'training_supervision.pt',SAFE/dataset/'split_ids.json',GOLD/dataset/'manifest.json',HIDDEN/dataset/'hidden.pt',SSL/dataset/'encoder_and_embeddings.pt',GRAPH/dataset/'graph_environments.pt',TEXT/dataset/'text_environments.pt']
    edge_path=ROOT/'runs/multiseed_reproduction_20260909_v1/seed42/upstream'/dataset/'graph.pt';paths.append(edge_path)
    edge=load(edge_path)['edge_index'];emb=load(SSL/dataset/'encoder_and_embeddings.pt')['embeddings']
    clean_g=raw['gnn_embeddings'][ids].clone();clean_l=load(HIDDEN/dataset/'hidden.pt')['hidden'][ids].clone()
    pg=raw['gnn_logits'][ids].float().softmax(1);pl=raw['llm_logits'][ids].float().softmax(1)
    full_pred=raw['gnn_logits'].argmax(1)
    fitted=fit_class_prototypes(emb,ids,y,num_classes=c)
    clean_tab=build70(pg,pl,ids,edge,full_pred,emb,fitted,n,c)
    parity={};clean_banks={}
    for branch in ('agreement','disagreement'):
        p=GOLD/dataset/(branch+'_train.pt');paths.append(p)
        assert sha(p)==oldmeta['bank_hashes'][p.name]
        b=load(p);clean_banks[branch]=b;pos=positions(ids,b['ids'])
        generated=clean_tab[pos]
        if branch=='agreement':generated=torch.cat((generated[:,:27],generated[:,54:]),1)
        err=float((generated-b['tab']).abs().max()) if len(pos) else 0.
        torch.testing.assert_close(generated,b['tab'],atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(clean_g[pos],b['g'],atol=0,rtol=0)
        torch.testing.assert_close(clean_l[pos],b['l'],atol=0,rtol=0)
        parity[branch]=dict(reconstructed_tab_max_error=err,rows=len(pos))
        if branch=='agreement':clean_tab[pos]=torch.cat((b['tab'][:,:27],b['tab'][:,:27],b['tab'][:,27:]),1)
        else:clean_tab[pos]=b['tab']
    envs=[];input_audits=[]
    for name in ENVIRONMENTS:
        g=clean_g.clone();l=clean_l.clone();epg=pg.clone();epl=pl.clone();tab=clean_tab.clone()
        if name!='clean':
            cached=cached_environment(dataset,name);pos=positions(cached['node_ids'],ids)
            cached_pg,cached_pl=cached['gnn_prob'][pos],cached['llm_prob'][pos]
            replay_path=RUN/'replay'/dataset/(name+'.pt');paths.append(replay_path)
            replay=load(replay_path);assert torch.equal(replay['node_ids'],ids)
            changed=replay['changed_mask']
            if name.startswith('graph'):
                torch.testing.assert_close(cached_pl,pl,atol=3e-7,rtol=3e-6)
                epg=cached_pg.clone();epg[~changed]=pg[~changed]
                g=replay['g'];shift_emb=emb.clone();shift_emb[ids]=g
                tab=build70(epg,epl,ids,replay['edge_index'],replay['full_gnn_pred'],shift_emb,fitted,n,c)
                untouched_embedding_exact=torch.equal(l,clean_l)
            else:
                torch.testing.assert_close(cached_pg,pg,atol=3e-7,rtol=3e-6)
                epl=cached_pl.clone();epl[~changed]=pl[~changed]
                l=replay['l']
                tab=build70(epg,epl,ids,edge,full_pred,emb,fitted,n,c)
                untouched_embedding_exact=torch.equal(g,clean_g)
            tab[~changed]=clean_tab[~changed]
            assert untouched_embedding_exact
            assert torch.equal(epg.argmax(1),cached['gnn_pred'][pos])
            assert torch.equal(epl.argmax(1),cached['llm_pred'][pos])
            input_audits.append(dict(environment=name,changed_gold=int(changed.sum()),untouched_embedding_exact=untouched_embedding_exact,unchanged_tab_exact=torch.equal(tab[~changed],clean_tab[~changed]),historical_full_head_predictions_exact=True))
        envs.append(dict(environment=name,node_ids=ids,gnn_prob=epg,llm_prob=epl,tab70=tab,g=g,l=l))
    organized=organize(dataset,envs,gold_ids=ids,gold_targets=y,num_nodes=n)
    dest=RUN/'banks'/dataset;dest.mkdir(parents=True,exist_ok=False)
    rows=organized['rows'];save(dest/'rows.pt',rows)
    save(dest/'expert_probabilities.pt',dict(sample_id=rows['sample_id'],gnn_prob=torch.cat([e['gnn_prob'] for e in envs]),llm_prob=torch.cat([e['llm_prob'] for e in envs])))
    counts=[]
    for ei,name in enumerate(ENVIRONMENTS):
        m=rows['environment_index']==ei;a=m & rows['agreement_mask'];d=m & rows['disagreement_mask'];p=m & rows['preference_mask']
        counts.append(dict(dataset=dataset,environment=name,total=int(m.sum()),agreement=int(a.sum()),agreement_correct=int((a & (rows['agreement_target']==1)).sum()),agreement_wrong=int((a & (rows['agreement_target']==0)).sum()),disagreement=int(d.sum()),trust_positive=int((d & (rows['trust_target']==1)).sum()),trust_negative=int((d & (rows['trust_target']==0)).sum()),preference_gnn=int((p & (rows['preference_target']==1)).sum()),preference_llm=int((p & (rows['preference_target']==0)).sum())))
    widths={}
    for branch in ('agreement','disagreement'):
        b=organized[branch];save(dest/(branch+'_train.pt'),b)
        # Exact archived clean bank replay, including target/state and raw representations.
        original=clean_banks[branch];m=b['environment_index']==0;pos=positions(b['node_ids'][m],original['ids'])
        for key in ('tab','g','l','gnn_pred','llm_pred','target' if branch=='agreement' else 'state'):
            assert torch.equal(b[key][m][pos],original[key]),(dataset,branch,key)
        views={mode:torch.from_numpy(raw_features(b,mode).astype('float32')) for mode in ('prob','full','compact','context','embed')}
        widths[branch]={k:v.shape[1] for k,v in views.items()}
        save(dest/(branch+'_v7_features.pt'),dict(sample_id=b['ids'],features=views))
        for head in (('agreement',) if branch=='agreement' else ('trust','preference')):
            v6=head_view(b,head,'v6');save(dest/(head+'_v6.pt'),v6)
            # Head-specific masks are shared by every v7 mode.
            m=torch.ones(len(b['ids']),dtype=torch.bool) if head!='preference' else b['state']!=2
            v7={k:v for k,v in v6.items() if k not in ('tab','g','l')}
            v7['features']={mode:value[m] for mode,value in views.items()}
            save(dest/(head+'_v7.pt'),v7)
    with (dest/'rows.csv').open('x',newline='') as f:
        keys=['sample_id','node_id','environment','gold_target','gnn_pred','llm_pred','branch','agreement_target','trust_target','preference_target','preference_mask','weight']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for i in range(len(rows['sample_id'])):
            writer.writerow(dict(sample_id=int(rows['sample_id'][i]),node_id=int(rows['node_ids'][i]),environment=ENVIRONMENTS[int(rows['environment_index'][i])],gold_target=int(rows['gold_target'][i]),gnn_pred=int(rows['gnn_pred'][i]),llm_pred=int(rows['llm_pred'][i]),branch='agreement' if rows['agreement_mask'][i] else 'disagreement',agreement_target=int(rows['agreement_target'][i]),trust_target=int(rows['trust_target'][i]),preference_target=int(rows['preference_target'][i]),preference_mask=bool(rows['preference_mask'][i]),weight=float(rows['weight'][i])))
    for p in (RUN/'replay'/dataset).glob('*audit.json'):paths.append(p)
    manifest=dict(schema='router_augmentation_v8_gold_v1',dataset=dataset,num_nodes=n,num_classes=c,gold_ids=ids.tolist(),environments=list(ENVIRONMENTS),counts=counts,widths=widths,clean_parity=parity,input_audits=input_audits,input_hashes={str(p):sha(p) for p in paths},output_hashes={p.name:sha(p) for p in dest.iterdir() if p.is_file()},
                  supervision='safe gold targets only; legacy query probabilities restricted to original gold IDs; OOF support ignored',weight='one per observation including unchanged/duplicate-valued environments',prototype_reference='original clean gold SSL prototypes, shared by every environment; augmented graph rows use shifted GNN/SSL embeddings',legacy_container_note='legacy environment containers include whole-graph truth; loader whitelists probabilities/predictions/IDs and does not use legacy truth',slurm_job_id=os.environ['SLURM_JOB_ID'])
    write(dest/'manifest.json',manifest)
    print(json.dumps(dict(dataset=dataset,counts=counts,widths=widths,clean_parity=parity)),flush=True)
    return counts


if __name__=='__main__':
    require_slurm();torch.set_num_threads(2)
    counts=[]
    for dataset in DATASETS:counts.extend(build_dataset(dataset))
    with (RUN/'counts.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(counts[0]));w.writeheader();w.writerows(counts)
    write(RUN/'BUILD_COMPLETE.json',dict(datasets=list(DATASETS),environment_names=list(ENVIRONMENTS),counts=counts,slurm_job_id=os.environ['SLURM_JOB_ID']))
