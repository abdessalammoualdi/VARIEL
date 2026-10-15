#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, math, os, shutil, sys, urllib.request
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
PROJECT=Path(os.environ.get('DTRAR_PROJECT', str(Path.home()/'Documents'/'DTRA_Start'/'DTRA'))).resolve()
OUT=PROJECT/'artifacts'/'variel_highinfo_external_confirmation'
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(HERE))
import frozen_variel_highinfo_development as hi

FROZEN_DRAWER_SHA='1b2fbbff9e47fb78174109ae48968fc7245e66ec6e226549afb9cd54e0708b62'
SEEDS=list(range(10))
SOURCE_FRACTION=.30
TARGET_WINDOWS=14

LOCKS={
 'rialto':{
   'sha256':'12c16b6f473bebd1be79006399e06fbd825e163ca6e6a6695b072fe82bcc2183',
   'rows':82250,'features':27,'classes':10,
   'url':'https://raw.githubusercontent.com/ogozuacik/concept-drift-datasets-scikit-multiflow/master/real-world/rialto.csv'
 },
 'spam':{
   'sha256':'8880ced878c81d25564bd6c8ee1ba805a5bb1a8c0d98d12f4de62f799ac910a5',
   'rows':6213,'features':499,'classes':2,
   'url':'https://raw.githubusercontent.com/ogozuacik/concept-drift-datasets-scikit-multiflow/master/real-world/spam.csv'
 },
 'insects':{
   'sha256':'e4819251b250a6fc1bf2a3798bbb7a1cbd2aff81d2ea34252a904c5266272fff',
   'rows':52848,'features':33,'classes':6,
 },
 'ble_rssi':{
   'sha256':'d6b4bda20a54987e74f25e777bfe04d1a84cdc941c3c79a55264163c6eabae7f',
   'rows':13584,'features':2,'classes':3,
 },
}

def must(c,m):
    if not c: raise RuntimeError(m)

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def download(url,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size>1000: return
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req,timeout=600) as r,open(path,'wb') as f:
        shutil.copyfileobj(r,f)

def data_path(ds):
    if ds=='rialto':
        cands=[PROJECT/'artifacts'/'dtrar_final_frozen_rialto_spam'/'raw'/'rialto.csv', PROJECT/'datasets'/'rialto'/'rialto.csv']
        p=next((x for x in cands if x.exists()),cands[0]); download(LOCKS[ds]['url'],p); return p
    if ds=='spam':
        cands=[PROJECT/'artifacts'/'dtrar_final_frozen_rialto_spam'/'raw'/'spam.csv', PROJECT/'datasets'/'spam'/'spam.csv']
        p=next((x for x in cands if x.exists()),cands[0]); download(LOCKS[ds]['url'],p); return p
    if ds=='insects': return PROJECT/'datasets'/'insects'/'INSECTS.csv'
    if ds=='ble_rssi': return PROJECT/'datasets'/'ble_rssi'/'official'/'raw_rssi.csv'
    raise ValueError(ds)

def load_dataset(ds):
    p=data_path(ds); must(p.exists(),f'missing dataset: {p}')
    actual=sha256(p); must(actual==LOCKS[ds]['sha256'],f'{ds}: SHA mismatch {actual}')
    if ds in ('rialto','spam'):
        df=pd.read_csv(p,low_memory=False)
        df=df.loc[:,[c for c in df.columns if not str(c).lower().startswith('unnamed:')]]
        must(abs(len(df)-LOCKS[ds]['rows'])<=3,f'{ds}: row count {len(df)}')
        must(df.shape[1]==LOCKS[ds]['features']+1,f'{ds}: columns {df.shape[1]}')
        X=df.iloc[:,:-1].apply(pd.to_numeric,errors='raise').to_numpy(np.float32)
        ys=df.iloc[:,-1].astype(str).str.strip()
    elif ds=='insects':
        df=pd.read_csv(p,low_memory=False)
        must(df.shape==(52848,34),f'INSECTS shape {df.shape}')
        X=df.iloc[:,:-1].apply(pd.to_numeric,errors='raise').to_numpy(np.float32)
        ys=df['Class'].astype(str).str.strip()
    else:
        df=pd.read_csv(p,low_memory=False)
        must(df.shape==(13584,5),f'BLE shape {df.shape}')
        df['_orig_order']=np.arange(len(df))
        df['_timestamp']=pd.to_datetime(df['timestamp'],errors='raise')
        df=df.sort_values(['_timestamp','_orig_order'],kind='mergesort').reset_index(drop=True)
        must(df['_timestamp'].is_monotonic_increasing,'BLE chronology failed')
        X=df[['rssiOne','rssiTwo']].apply(pd.to_numeric,errors='raise').to_numpy(np.float32)
        ys=df['locationStatus'].astype(str).str.strip()
    labels=sorted(ys.unique().tolist()); must(len(labels)==LOCKS[ds]['classes'],f'{ds}: classes {labels}')
    mp={v:i for i,v in enumerate(labels)}; y=np.asarray([mp[v] for v in ys],int)
    n=len(y); source_n=int(math.floor(SOURCE_FRACTION*n))
    must(set(np.unique(y[:source_n]))==set(np.unique(y)),f'{ds}: source lacks target class')
    source=np.arange(source_n,dtype=int); target=np.arange(source_n,n,dtype=int)
    windows=[np.asarray(w,int) for w in np.array_split(target,TARGET_WINDOWS)]
    must(len(windows)==TARGET_WINDOWS and all(len(w)>0 for w in windows),f'{ds}: bad windows')
    return p,X,y,source,target,windows,[str(x) for x in labels],actual

def rounds_for(ds,X,y,source,seed,k):
    d=OUT/ds/f'seed_{seed}'; d.mkdir(parents=True,exist_ok=True)
    p=d/'ROUNDS.json'
    if p.exists(): return int(json.loads(p.read_text())['rounds'])
    r=hi.choose_rounds(X,y,source,seed,k)
    p.write_text(json.dumps({'rounds':int(r)},indent=2))
    return int(r)

def evaluate_seed(ds,seed):
    d=OUT/ds/f'seed_{seed}'; d.mkdir(parents=True,exist_ok=True)
    summary_path=d/'SUMMARY.json'; trace_path=d/'WINDOW_TRACE.csv'
    if summary_path.exists() and trace_path.exists():
        rec=json.loads(summary_path.read_text())
        print(f'EXTERNAL_REUSE dataset={ds} seed={seed} improvement={rec["improvement_vs_retrain"]:.9f}',flush=True)
        return rec
    p,X,y,source,target,windows,labels,file_sha=load_dataset(ds)
    k=len(labels); rounds=rounds_for(ds,X,y,source,seed,k)
    cache=d/'rolling_anchor_cache'
    issued,yy,row_id=hi.rolling_anchor(X,y,source,windows,seed,rounds,k,cache)
    must(np.array_equal(row_id,target),f'{ds}/{seed}: row id mismatch')
    must(np.array_equal(yy,y[target]),f'{ds}/{seed}: label mismatch')
    pos={int(r):i for i,r in enumerate(row_id)}
    winpos=[np.asarray([pos[int(r)] for r in ix],int) for ix in windows]
    pred=issued.copy(); predS=issued.copy(); traces=[]
    for t,posix in enumerate(winpos):
        anchor=issued[posix]; local_y=yy[posix]
        hist,w=hi.history_positions(winpos,t)
        if len(hist):
            full=hi.ProbabilityCalibrator().fit(issued[hist],yy[hist],w).predict_proba(anchor)
            tw=w/w.sum(); ess=float(1.0/np.sum(tw**2))
            residual=hi.ResidualCalibrator(anchor.shape[1],ess).fit(issued[hist],yy[hist],w).predict_proba(anchor)
            cands={'fullcal':full,'residual':residual}
        else:
            cands={'fullcal':anchor.copy(),'residual':anchor.copy()}
        m=max(1,int(math.floor(hi.CANARY_FRACTION*len(posix))))
        rng=np.random.default_rng(hi.stable_seed('variel_highinfo',seed,t,file_sha))
        strata,can,bulk=hi.choose_canaries(anchor,cands,m,rng)
        ests={}; best='raw_retrain'; best_gain=0.; best_eta=0.
        for name,q in cands.items():
            D,C,eta,gain=hi.estimate(anchor,q,local_y,strata,bulk); ests[name]=(D,C,eta,gain)
            if gain>best_gain: best=name; best_gain=gain; best_eta=eta
        out=anchor.copy()
        if best!='raw_retrain' and len(bulk): out[bulk]=hi.probs(anchor[bulk]+best_eta*(cands[best][bulk]-anchor[bulk]))
        out[can]=anchor[can]; pred[posix]=out
        bestS='raw_retrain'; bestSgain=0.; bestSeta=0.
        for name,q in cands.items():
            D,C,eta,_=ests[name]; etaS,_=hi.clip_H(anchor,q,bulk,can,eta,hi.EPS_SAFE); score=2*D*etaS-C*etaS*etaS
            if score>bestSgain: bestS=name; bestSgain=score; bestSeta=etaS
        outS=anchor.copy()
        if bestS!='raw_retrain' and len(bulk): outS[bulk]=hi.probs(anchor[bulk]+bestSeta*(cands[bestS][bulk]-anchor[bulk]))
        outS[can]=anchor[can]; predS[posix]=outS
        H=hi.prospective_H(anchor,outS); must(H<=hi.EPS_SAFE+2e-9,f'{ds}/{seed}/W{t+1}: H={H}')
        traces.append({'dataset':ds,'seed':seed,'window':t+1,'window_n':len(posix),'canary_n':len(can),
            'utility_choice':best,'utility_eta':best_eta,'safety_choice':bestS,'safety_eta':bestSeta,
            'raw_retrain_brier':float(np.mean(hi.brier_rows(anchor,local_y))),
            'utility_brier':float(np.mean(hi.brier_rows(out,local_y))),
            'safety02_brier':float(np.mean(hi.brier_rows(outS,local_y))),
            'safety_H':float(H)})
    raw=float(np.mean(hi.brier_rows(issued,yy))); util=float(np.mean(hi.brier_rows(pred,yy))); safe=float(np.mean(hi.brier_rows(predS,yy)))
    rec={'dataset':ds,'seed':int(seed),'raw_retrain_brier':raw,'utility_brier':util,'safety02_brier':safe,
         'improvement_vs_retrain':raw-util,'safety_improvement_vs_retrain':raw-safe,
         'beats_raw_retrain':bool(util<raw),'safety_beats_raw_retrain':bool(safe<raw),
         'max_safety_H':float(max(x['safety_H'] for x in traces)),'rounds':int(rounds),
         'full_feedback_label_fraction':1.0,'highinfo_total_label_fraction':1.0,
         'current_canary_fraction_within_full_regime':hi.CANARY_FRACTION,'dataset_sha256':file_sha,
         'frozen_drawer_sha256':FROZEN_DRAWER_SHA}
    pd.DataFrame(traces).to_csv(trace_path,index=False); summary_path.write_text(json.dumps(rec,indent=2,sort_keys=True))
    print(f'EXTERNAL_DONE dataset={ds} seed={seed} raw={raw:.9f} utility={util:.9f} safe={safe:.9f} delta={raw-util:.9f} safe_delta={raw-safe:.9f} H={rec["max_safety_H"]:.9f}',flush=True)
    return rec

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--datasets',nargs='+',required=True); args=ap.parse_args()
    must(sha256(HERE/'frozen_variel_highinfo_development.py')==FROZEN_DRAWER_SHA,'frozen HIGH-INFO drawer source hash mismatch')
    print('FROZEN_HIGHINFO_DRAWER_INTEGRITY_PASS',flush=True)
    with threadpool_limits(limits=1):
        for ds in args.datasets:
            must(ds in LOCKS,f'unknown dataset {ds}')
            print(f'EXTERNAL_DATASET_BEGIN dataset={ds}',flush=True)
            p,*_=load_dataset(ds); print(f'EXTERNAL_DATASET_LOCK dataset={ds} sha256={sha256(p)}',flush=True)
            rows=[evaluate_seed(ds,s) for s in SEEDS]
            df=pd.DataFrame(rows); dsdir=OUT/ds
            df.to_csv(dsdir/'EXTERNAL_SEEDWISE.csv',index=False)
            sm={'dataset':ds,'mean_raw_retrain_brier':float(df.raw_retrain_brier.mean()),
                'mean_highinfo_utility_brier':float(df.utility_brier.mean()),
                'mean_highinfo_safety_brier':float(df.safety02_brier.mean()),
                'mean_improvement_vs_raw_retrain':float(df.improvement_vs_retrain.mean()),
                'mean_safety_improvement_vs_raw_retrain':float(df.safety_improvement_vs_retrain.mean()),
                'utility_wins_vs_raw_retrain':int((df.improvement_vs_retrain>0).sum()),
                'safety_wins_vs_raw_retrain':int((df.safety_improvement_vs_retrain>0).sum()),
                'utility_mean_beats_raw_full_retrain':bool(df.utility_brier.mean()<df.raw_retrain_brier.mean()),
                'safety_mean_beats_raw_full_retrain':bool(df.safety02_brier.mean()<df.raw_retrain_brier.mean()),
                'max_safety_H':float(df.max_safety_H.max()),'total_label_fraction':1.0}
            (dsdir/'EXTERNAL_SUMMARY.json').write_text(json.dumps(sm,indent=2,sort_keys=True))
            print('EXTERNAL_DATASET_COMPLETE '+json.dumps(sm,sort_keys=True),flush=True)

if __name__=='__main__': main()
