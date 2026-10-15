#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os, math, hashlib, urllib.request, shutil, time, json
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import xgboost as xgb

PROJECT=Path(os.environ.get('DTRAR_PROJECT',str(Path.home()/'Documents'/'DTRA_Start'/'DTRA'))).resolve()
SEEDS=list(range(10)); WINDOWS=14; SOURCE_FRAC=.30

@dataclass
class Stream:
    name:str; X:np.ndarray; y:np.ndarray; source:np.ndarray; windows:list; classes:list
    @property
    def target(self): return np.concatenate(self.windows)

def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'little')%(2**32-1)

def probs(p):
    p=np.asarray(p,float); p=np.clip(p,1e-12,np.inf); return p/p.sum(axis=1,keepdims=True)

def brier(p,y):
    p=probs(p); y=np.asarray(y,int); return float(np.mean(np.sum((p-np.eye(p.shape[1])[y])**2,axis=1)))

def _to_numeric(X):
    if isinstance(X,pd.DataFrame):
        out=X.copy()
        for c in out.columns:
            if not pd.api.types.is_numeric_dtype(out[c]):
                out[c]=pd.factorize(out[c].astype(str),sort=True)[0]
        return out.apply(pd.to_numeric,errors='coerce').to_numpy(np.float32)
    return np.asarray(X,np.float32)

def load_builtin(name):
    from dtra.data import load_dataset
    d=load_dataset(name,PROJECT/'data'/'raw')
    X=_to_numeric(d.X); y=np.asarray(d.y,int)
    windows=[np.asarray(w,int) for w in d.windows]
    target=np.concatenate(windows)
    if hasattr(d,'source'):
        source=np.asarray(d.source,int)
    elif hasattr(d,'train'):
        source=np.asarray(d.train,int)
    else:
        source=np.arange(int(target.min()),dtype=int)
    classes=sorted(np.unique(y).tolist())
    return Stream(name,X,y,source,windows,classes)

def load_airlines():
    candidates=[PROJECT/'artifacts'/'dtrar_final_airlines_oneshot'/'raw'/'airlines.csv',PROJECT/'datasets'/'airlines'/'airlines.csv']
    p=next((x for x in candidates if x.exists()),None)
    if p is None: raise RuntimeError('Airlines CSV not found in expected project paths')
    df=pd.read_csv(p); df=df.loc[:,[c for c in df.columns if not str(c).lower().startswith('unnamed:')]]
    req=['Airline','Flight','AirportFrom','AirportTo','DayOfWeek','Time','Length','Delay']
    if list(df.columns)!=req: raise RuntimeError(f'Airlines schema mismatch {list(df.columns)}')
    n=len(df); sn=int(math.floor(.30*n)); yraw=pd.to_numeric(df.Delay).astype(int).to_numpy(); labels=sorted(np.unique(yraw).tolist()); mp={v:i for i,v in enumerate(labels)}; y=np.asarray([mp[v] for v in yraw],int)
    X=np.empty((n,7),np.float32)
    for j,c in enumerate(req[:-1]):
        if c in ['Airline','AirportFrom','AirportTo']:
            levels=sorted(df[c].iloc[:sn].astype(str).unique().tolist()); mp2={v:i for i,v in enumerate(levels)}; X[:,j]=df[c].astype(str).map(mp2).fillna(-1).astype(np.float32)
        else: X[:,j]=pd.to_numeric(df[c],errors='raise').astype(np.float32)
    source=np.arange(sn); target=np.arange(sn,n); windows=[np.asarray(w,int) for w in np.array_split(target,14)]
    return Stream('airlines',X,y,source,windows,list(range(len(labels))))

def _download(url,p):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and p.stat().st_size>1000:return
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req,timeout=300) as r,open(p,'wb') as f: shutil.copyfileobj(r,f)

def load_rs(name):
    specs={
      'rialto':('https://raw.githubusercontent.com/ogozuacik/concept-drift-datasets-scikit-multiflow/master/real-world/rialto.csv',82250,27),
      'spam':('https://raw.githubusercontent.com/ogozuacik/concept-drift-datasets-scikit-multiflow/master/real-world/spam.csv',6213,499)}
    url,rows,nf=specs[name]; p=PROJECT/'artifacts'/'dtrar_final_frozen_rialto_spam'/'raw'/f'{name}.csv'; _download(url,p)
    df=pd.read_csv(p); df=df.loc[:,[c for c in df.columns if not str(c).lower().startswith('unnamed:')]]
    X=df.iloc[:,:-1].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32); yr=df.iloc[:,-1].astype(str); labels=sorted(yr.unique()); mp={v:i for i,v in enumerate(labels)}; y=np.asarray([mp[v] for v in yr],int)
    sn=int(math.floor(.30*len(df))); source=np.arange(sn); target=np.arange(sn,len(df)); windows=[np.asarray(w,int) for w in np.array_split(target,14)]
    return Stream(name,X,y,source,windows,list(range(len(labels))))

def load_insects():
    p=PROJECT/'datasets'/'insects'/'INSECTS.csv'; df=pd.read_csv(p,low_memory=False); X=df.iloc[:,:-1].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32); yr=df.iloc[:,-1].astype(str).str.strip(); labels=sorted(yr.unique()); mp={v:i for i,v in enumerate(labels)}; y=np.asarray([mp[v] for v in yr],int)
    sn=int(math.floor(.30*len(df))); source=np.arange(sn); target=np.arange(sn,len(df)); windows=[np.asarray(w,int) for w in np.array_split(target,14)]
    return Stream('insects',X,y,source,windows,list(range(len(labels))))

def load_ble():
    p=PROJECT/'datasets'/'ble_rssi'/'official'/'raw_rssi.csv'; df=pd.read_csv(p,low_memory=False); df['_o']=np.arange(len(df)); df['_t']=pd.to_datetime(df.timestamp,errors='raise'); df=df.sort_values(['_t','_o'],kind='mergesort').reset_index(drop=True)
    X=df[['rssiOne','rssiTwo']].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32); yr=df.locationStatus.astype(str).str.strip(); labels=sorted(yr.unique()); mp={v:i for i,v in enumerate(labels)}; y=np.asarray([mp[v] for v in yr],int)
    sn=int(math.floor(.30*len(df))); source=np.arange(sn); target=np.arange(sn,len(df)); windows=[np.asarray(w,int) for w in np.array_split(target,14)]
    return Stream('ble_rssi',X,y,source,windows,list(range(len(labels))))

def load_stream(name):
    if name in ['bank','electricity','gas']: return load_builtin(name)
    if name=='airlines': return load_airlines()
    if name in ['rialto','spam']: return load_rs(name)
    if name=='insects': return load_insects()
    if name=='ble_rssi': return load_ble()
    raise ValueError(name)

def xgb_args(seed,k,rounds,lr=.05):
    return dict(n_estimators=int(rounds),max_depth=6,learning_rate=float(lr),subsample=.9,colsample_bytree=.9,min_child_weight=2,reg_lambda=1.,tree_method='hist',device=os.environ.get('COMPASS_XGB_DEVICE','cuda'),n_jobs=int(os.environ.get('COMPASS_XGB_NJOBS','2')),random_state=int(seed),objective='binary:logistic' if k==2 else 'multi:softprob',eval_metric='logloss' if k==2 else 'mlogloss')

def choose_rounds(st,seed):
    src=np.asarray(st.source,int); cut=max(1,int(.8*len(src))); tr,va=src[:cut],src[cut:]; imp=SimpleImputer(strategy='median').fit(st.X[tr]); k=len(st.classes)
    m=XGBClassifier(**xgb_args(seed,k,700),early_stopping_rounds=50); m.fit(imp.transform(st.X[tr]).astype(np.float32),st.y[tr],eval_set=[(imp.transform(st.X[va]).astype(np.float32),st.y[va])],verbose=False)
    return max(1,int(getattr(m,'best_iteration',699))+1)

def fit_model(st,train_idx,seed,rounds,lr=.05,xgb_model=None,extra_rounds=None):
    t0=time.perf_counter(); imp=SimpleImputer(strategy='median').fit(st.X[train_idx]); rr=rounds if extra_rounds is None else extra_rounds; m=XGBClassifier(**xgb_args(seed,len(st.classes),rr,lr=lr)); m.fit(imp.transform(st.X[train_idx]).astype(np.float32),st.y[train_idx],verbose=False,xgb_model=xgb_model); return m,imp,time.perf_counter()-t0

def pred(m,imp,X):
    """Stable prediction for GPU-trained XGBoost models.
    Training remains on the configured device; prediction uses a CPU booster copy
    to avoid CUDA device-mismatch / multiclass inplace-predict failures.
    """
    Xt=imp.transform(X).astype(np.float32)
    bst=m.get_booster().copy()
    bst.set_param({"device":"cpu"})
    z=np.asarray(bst.predict(xgb.DMatrix(Xt)))
    if z.ndim==1:
        z=np.c_[1-z,z]
    return probs(z)

def zshift(Xa,Xb):
    a=np.nanmean(Xa,axis=0); b=np.nanmean(Xb,axis=0); scale=np.nanstd(np.vstack([Xa,Xb]),axis=0)+1e-8; return float(np.mean(np.abs(a-b)/scale))
