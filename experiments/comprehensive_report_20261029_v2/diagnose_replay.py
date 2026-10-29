"""Read-only Slurm diagnosis of native scorer replay precision, cell31."""
import comprehensive as c
from comprehensive import *
import numpy as np,torch,joblib
from unittest.mock import patch
import enhancerouter.router as runtime

def main():
    registered();i=31;sh,se,ds=CELLS[i];checker=Sources();clean,bankrec=query_bank(sh,se,ds,checker)
    rows={r['variant']:r for r in read(OLD/'report/RESULTS.json')['rows'] if r['index']==i}
    results=[]
    for label in ('B0','D1_25__fixed','D2a_1000__fixed'):
        path=Path(rows[label]['sources']['router']['path']);scores,_=score_files(path);stored=scores['disagreement']
        paths=[('run_model',path/'models.joblib')]
        if label=='B0':paths.append(('historical_bundle',Path(read(RUN/'HISTORICAL_INPUTS_V2.json')['cells'][key(sh,se,ds)]['branches']['disagreement']['bundle'])))
        for origin,modelpath in paths:
            model=joblib.load(modelpath);bundle=model['disagreement'] if origin=='run_model' else model
            for threads in (1,2):
                torch.set_num_threads(threads);cache={};heads={}
                for head,w in bundle['models'].items():
                    z=np.asarray(runtime.predict(w,clean));cache[id(w)]=z
                    heads[head]=dict(engine=w['engine'],dtype=str(z.dtype),config=w['model']['config'],sha256=hashlib.sha256(z.tobytes()).hexdigest())
                direct=runtime.predict_bundle(bundle,clean)
                with patch.object(runtime,'predict',side_effect=lambda w,b:cache[id(w)]):replay=runtime.predict_bundle(bundle,clean)
                q=replay['q'];t=q[:,0]+q[:,1];p=np.divide(q[:,0],t,out=np.full(len(t),.5),where=t>0)
                fields={'score':replay['score'],'trust':t,'preference':p};errors={}
                for name,z in fields.items():
                    expected=stored[name].numpy();diff=np.abs(z-expected)
                    errors[name]=dict(dtype=str(z.dtype),stored_dtype=str(expected.dtype),max_abs=float(diff.max()),count_above_1e8=int((diff>1e-8).sum()),direct_vs_cached=float(np.max(np.abs(direct['score']-replay['score']))),
                        rank_equal=bool(np.array_equal(np.lexsort((stored['ids'].numpy(),-z)),np.lexsort((stored['ids'].numpy(),-expected)))),examples=[dict(index=int(j),actual=float(z[j]),expected=float(expected[j])) for j in np.argsort(diff)[-3:]])
                results.append(dict(variant=label,origin=origin,path=str(modelpath),model_sha256=digest(modelpath),threads=threads,heads=heads,outer_keys={k:bundle[k] for k in ('tuning_schema','affine','calibration','route_recipe','boundary') if k in bundle},errors=errors,route_equal=bool(np.array_equal(replay['route'],stored['route'].numpy()))))
    dest=OUT/'DIAGNOSE_REPLAY_31.json';write(dest,dict(job=os.environ['SLURM_JOB_ID'],bank=bankrec,results=results,code_sha256=digest(Path(__file__))))
    print(json.dumps(dict(path=str(dest),sha256=digest(dest),results=results)),flush=True)

if __name__=='__main__':main()
