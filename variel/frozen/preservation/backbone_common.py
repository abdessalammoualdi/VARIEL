#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os, sys, math, hashlib, time, copy, warnings, json, urllib.request, shutil, gc
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, ElasticNetCV
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, norm

PROJECT=Path(os.environ.get('DTRAR_PROJECT',str(Path.home()/'Documents'/'DTRA_Start'/'DTRA'))).resolve()
SEEDS=list(range(10)); AUDIT4=.04; AUDIT5=.05; EPS=.02
WHOLE_DATASETS=['bank','electricity','gas','airlines','rialto','spam','insects','ble_rssi']
SPECIALIST_DATASETS=['bank','electricity','occupancy_detection_v2','eeg_eye_state','airlines','spam']
ALL_DATASETS=['bank','electricity','occupancy_detection_v2','eeg_eye_state','gas','airlines','rialto','spam','insects','ble_rssi']
COMPASS_TARGETS=['electricity','gas','airlines']

@dataclass
class Stream:
    name:str; X:np.ndarray; y:np.ndarray; source:np.ndarray; windows:list; classes:list
    @property
    def target(self): return np.concatenate(self.windows)

def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'little')%(2**32-1)

def probs(p):
    p=np.asarray(p,float); p=np.clip(p,1e-12,np.inf); return p/p.sum(axis=1,keepdims=True)

def brier_rows(p,y):
    p=probs(p); y=np.asarray(y,int); return np.sum((p-np.eye(p.shape[1])[y])**2,axis=1)

def brier(p,y): return float(np.mean(brier_rows(p,y)))

def _to_numeric(X):
    if isinstance(X,pd.DataFrame):
        out=X.copy()
        for c in out.columns:
            if not pd.api.types.is_numeric_dtype(out[c]): out[c]=pd.factorize(out[c].astype(str),sort=True)[0]
        return out.apply(pd.to_numeric,errors='coerce').to_numpy(np.float32)
    return np.asarray(X,np.float32)

def _from_obj(name,d):
    X=_to_numeric(d.X); y=np.asarray(d.y,int); windows=[np.asarray(w,int) for w in d.windows]
    if hasattr(d,'source'): source=np.asarray(d.source,int)
    elif hasattr(d,'train'): source=np.asarray(d.train,int)
    else: source=np.arange(int(np.concatenate(windows).min()),dtype=int)
    labs=sorted(np.unique(y).tolist()); mp={v:i for i,v in enumerate(labs)}
    if labs != list(range(len(labs))): y=np.asarray([mp[v] for v in y],int)
    return Stream(name,X,y,source,windows,list(range(len(labs))))

def _enable_project_dtra():
    """Make the original DTRA src-layout package importable without reinstalling it."""
    for p in (PROJECT/'src', PROJECT):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0,str(p))

def _find_file(names, roots=None):
    roots = roots or [PROJECT/'data'/'raw', PROJECT/'data', PROJECT/'datasets', PROJECT/'artifacts']
    if isinstance(names,str): names=[names]
    for root in roots:
        if not root.exists(): continue
        for name in names:
            p=root/name
            if p.exists(): return p
        for name in names:
            hits=list(root.rglob(name))
            if hits: return hits[0]
    return None

def _direct_bank():
    p=_find_file('bank-full.csv')
    if p is None: raise RuntimeError('bank-full.csv not found under project data paths')
    df=pd.read_csv(p,sep=';')
    if 'y' not in df.columns: raise RuntimeError(f'Bank target column y missing in {p}')
    yraw=df['y'].astype(str).str.strip().str.lower()
    y=np.asarray((yraw=='yes').astype(int),int)
    Xdf=df.drop(columns=['y'])
    if 'duration' in Xdf.columns: Xdf=Xdf.drop(columns=['duration'])
    X=_to_numeric(Xdf)
    n=len(df); tr_end=int(math.floor(.50*n)); src_end=int(math.floor(.65*n))
    # This reproduces dtra.data's SOURCE-CALIBRATION block used by final VARIEL:
    # train=first 50%, source=next 15%, target=final 35%.
    source=np.arange(tr_end,src_end,dtype=int)
    target=np.arange(src_end,n,dtype=int)
    windows=[np.asarray(w,int) for w in np.array_split(target,12)]
    return Stream('bank',X,y,source,windows,[0,1])

def _direct_electricity():
    p=_find_file(['electricity-normalized.arff','electricity.arff'])
    if p is None: raise RuntimeError('Electricity ARFF not found under project data paths')
    from scipy.io import arff
    data,_=arff.loadarff(p)
    df=pd.DataFrame(data)
    # Decode byte/string columns.
    for c in df.columns:
        if df[c].dtype==object:
            df[c]=df[c].map(lambda v:v.decode() if isinstance(v,(bytes,bytearray)) else v)
    target_col='class' if 'class' in df.columns else df.columns[-1]
    yr=df[target_col].astype(str).str.strip().str.upper()
    labs=sorted(yr.unique()); mp={v:i for i,v in enumerate(labs)}
    y=np.asarray([mp[v] for v in yr],int)
    predictors=[c for c in df.columns if c not in {target_col,'date'}]
    # Authoritative audited protocol has exactly 7 predictors.
    X=_to_numeric(df[predictors])
    n=len(df); tr_end=int(math.floor(.50*n)); src_end=int(math.floor(.65*n))
    source=np.arange(tr_end,src_end,dtype=int)
    target=np.arange(src_end,n,dtype=int)
    windows=[np.asarray(w,int) for w in np.array_split(target,12)]
    return Stream('electricity',X,y,source,windows,list(range(len(labs))))

def _read_gas_batch(p):
    rows=[]; labels=[]
    with open(p,'r',encoding='utf-8',errors='replace') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            # UCI files can separate the class token from feature 1 with ';'.
            toks=line.replace(';',' ').split()
            lab=int(float(toks[0])); x=np.zeros(128,dtype=np.float32)
            for tok in toks[1:]:
                if ':' not in tok: continue
                a,b=tok.split(':',1); j=int(a)-1
                if 0<=j<128: x[j]=float(b)
            labels.append(lab); rows.append(x)
    return np.asarray(rows,np.float32),np.asarray(labels,int)

def _direct_gas():
    batch_paths=[]
    for i in range(1,11):
        p=_find_file(f'batch{i}.dat')
        if p is None: raise RuntimeError(f'Gas batch{i}.dat not found under project data paths')
        batch_paths.append(p)
    Xs=[];ys=[];offsets=[];off=0
    for p in batch_paths:
        X,y=_read_gas_batch(p);Xs.append(X);ys.append(y);offsets.append((off,off+len(y)));off+=len(y)
    X=np.vstack(Xs); yr=np.concatenate(ys); labs=sorted(np.unique(yr)); mp={v:i for i,v in enumerate(labs)}
    y=np.asarray([mp[int(v)] for v in yr],int)
    # Prefer the exact historical split indices if an old dtra.data package cannot be imported.
    # Deterministic stratified 70/30 fallback on batches 1-3.
    from sklearn.model_selection import train_test_split
    source_period=np.arange(offsets[0][0],offsets[2][1],dtype=int)
    tr,source=train_test_split(source_period,test_size=.30,random_state=42,stratify=y[source_period])
    source=np.sort(np.asarray(source,int))
    windows=[np.arange(a,b,dtype=int) for a,b in offsets[3:]]
    return Stream('gas',X,y,source,windows,list(range(len(labs))))

def load_builtin(name):
    # First choice: use the exact original project loader.  The previous v1 package
    # failed only because PROJECT/src was not added to sys.path.
    _enable_project_dtra()
    try:
        from dtra.data import load_dataset
        return _from_obj(name,load_dataset(name,PROJECT/'data'/'raw'))
    except ModuleNotFoundError as e:
        if getattr(e,'name',None) not in ('dtra','dtra.data'):
            raise
    except ImportError:
        pass
    # Self-contained fallback so the closure remains runnable even if the old
    # editable package has been removed.
    if name=='bank': return _direct_bank()
    if name=='electricity': return _direct_electricity()
    if name=='gas': return _direct_gas()
    raise ValueError(name)

def load_unseen(name):
    import sys
    sys.path.insert(0,str(PROJECT))
    import run_dtrar_unseen_confirmation_xgb as uc
    if getattr(uc,'MANIFEST',None) is None and hasattr(uc,'load_manifest'): uc.MANIFEST=uc.load_manifest()
    return _from_obj(name,uc.load_unseen(name))

def load_airlines():
    candidates=[PROJECT/'artifacts'/'dtrar_final_airlines_oneshot'/'raw'/'airlines.csv',PROJECT/'datasets'/'airlines'/'airlines.csv']
    p=next((x for x in candidates if x.exists()),None)
    if p is None: raise RuntimeError('Airlines CSV not found')
    df=pd.read_csv(p); df=df.loc[:,[c for c in df.columns if not str(c).lower().startswith('unnamed:')]]
    req=['Airline','Flight','AirportFrom','AirportTo','DayOfWeek','Time','Length','Delay']
    n=len(df); sn=int(math.floor(.30*n)); yraw=pd.to_numeric(df.Delay).astype(int).to_numpy(); labs=sorted(np.unique(yraw)); mp={v:i for i,v in enumerate(labs)}; y=np.asarray([mp[v] for v in yraw],int)
    X=np.empty((n,7),np.float32)
    for j,c in enumerate(req[:-1]):
        if c in ['Airline','AirportFrom','AirportTo']:
            levels=sorted(df[c].iloc[:sn].astype(str).unique()); mp2={v:i for i,v in enumerate(levels)}; X[:,j]=df[c].astype(str).map(mp2).fillna(-1).astype(np.float32)
        else: X[:,j]=pd.to_numeric(df[c],errors='coerce').astype(np.float32)
    source=np.arange(sn); target=np.arange(sn,n); windows=[np.asarray(w,int) for w in np.array_split(target,14)]
    return Stream('airlines',X,y,source,windows,list(range(len(labs))))

def _download(url,p):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and p.stat().st_size>1000:return
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req,timeout=300) as r,open(p,'wb') as f: shutil.copyfileobj(r,f)

def load_rs(name):
    specs={'rialto':('https://raw.githubusercontent.com/ogozuacik/concept-drift-datasets-scikit-multiflow/master/real-world/rialto.csv',),
           'spam':('https://raw.githubusercontent.com/ogozuacik/concept-drift-datasets-scikit-multiflow/master/real-world/spam.csv',)}
    url=specs[name][0]; p=PROJECT/'artifacts'/'dtrar_final_frozen_rialto_spam'/'raw'/f'{name}.csv'; _download(url,p)
    df=pd.read_csv(p); df=df.loc[:,[c for c in df.columns if not str(c).lower().startswith('unnamed:')]]
    X=df.iloc[:,:-1].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32); yr=df.iloc[:,-1].astype(str); labs=sorted(yr.unique()); mp={v:i for i,v in enumerate(labs)}; y=np.asarray([mp[v] for v in yr],int)
    sn=int(math.floor(.30*len(df))); source=np.arange(sn); target=np.arange(sn,len(df)); windows=[np.asarray(w,int) for w in np.array_split(target,14)]
    return Stream(name,X,y,source,windows,list(range(len(labs))))

def load_insects():
    p=PROJECT/'datasets'/'insects'/'INSECTS.csv'; df=pd.read_csv(p,low_memory=False)
    X=df.iloc[:,:-1].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32); yr=df.iloc[:,-1].astype(str).str.strip(); labs=sorted(yr.unique()); mp={v:i for i,v in enumerate(labs)}; y=np.asarray([mp[v] for v in yr],int)
    sn=int(math.floor(.30*len(df))); source=np.arange(sn); target=np.arange(sn,len(df)); windows=[np.asarray(w,int) for w in np.array_split(target,14)]
    return Stream('insects',X,y,source,windows,list(range(len(labs))))

def load_ble():
    p=PROJECT/'datasets'/'ble_rssi'/'official'/'raw_rssi.csv'; df=pd.read_csv(p,low_memory=False); df['_o']=np.arange(len(df)); df['_t']=pd.to_datetime(df.timestamp,errors='raise'); df=df.sort_values(['_t','_o'],kind='mergesort').reset_index(drop=True)
    X=df[['rssiOne','rssiTwo']].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32); yr=df.locationStatus.astype(str).str.strip(); labs=sorted(yr.unique()); mp={v:i for i,v in enumerate(labs)}; y=np.asarray([mp[v] for v in yr],int)
    sn=int(math.floor(.30*len(df))); source=np.arange(sn); target=np.arange(sn,len(df)); windows=[np.asarray(w,int) for w in np.array_split(target,14)]
    return Stream('ble_rssi',X,y,source,windows,list(range(len(labs))))

def load_stream(name):
    if name in ['bank','electricity','gas']: return load_builtin(name)
    if name in ['occupancy_detection_v2','eeg_eye_state']: return load_unseen(name)
    if name=='airlines': return load_airlines()
    if name in ['rialto','spam']: return load_rs(name)
    if name=='insects': return load_insects()
    if name=='ble_rssi': return load_ble()
    raise ValueError(name)

def class_complete(st,idx):
    idx=np.asarray(idx,int); present=set(map(int,np.unique(st.y[idx]))); extra=[]; src=np.asarray(st.source,int)
    for c in range(len(st.classes)):
        if c not in present:
            h=src[st.y[src]==c]
            if len(h): extra.append(int(h[0]))
    return np.r_[idx,np.asarray(extra,int)] if extra else idx

def largest_counts(sizes,rate):
    sizes=np.asarray(sizes,int); B=int(math.floor(rate*sizes.sum())); raw=B*sizes/max(1,sizes.sum()); a=np.floor(raw).astype(int)
    for j in np.argsort(-(raw-a))[:B-int(a.sum())]: a[j]+=1
    return a,B

def geometry_canary(p,m,seed):
    p=probs(p); n=len(p); m=min(max(1,int(m)),max(1,n-1)); ent=-np.sum(p*np.log(np.clip(p,1e-12,1)),axis=1); s=np.sort(p,axis=1); margin=s[:,-1]-s[:,-2] if p.shape[1]>1 else np.zeros(n); score=ent+1-margin
    order=np.argsort(score,kind='mergesort'); H=min(8,m,n); chunks=[np.asarray(x,int) for x in np.array_split(order,H) if len(x)]; masses=np.asarray([len(x)*max(float(np.ptp(score[x])),1e-6) for x in chunks]); caps=np.asarray([len(x) for x in chunks]); alloc=np.zeros(len(chunks),int)
    for _ in range(m):
        z=np.where(alloc<caps,masses/(alloc+1),-1.); alloc[int(np.argmax(z))]+=1
    rng=np.random.default_rng(seed); out=[]
    for pop,a in zip(chunks,alloc): q=pop.copy(); rng.shuffle(q); out.extend(q[:int(a)])
    return np.sort(np.asarray(out,int))

def zshift(Xa,Xb):
    a=np.nanmean(Xa,axis=0); b=np.nanmean(Xb,axis=0); scale=np.nanstd(np.vstack([Xa,Xb]),axis=0)+1e-8; return float(np.mean(np.abs(a-b)/scale))

def severity_score(Xshort,Xlong,bins=20):
    vals=[]
    for j in range(Xshort.shape[1]):
        a=np.asarray(Xshort[:,j],float); b=np.asarray(Xlong[:,j],float); a=a[np.isfinite(a)]; b=b[np.isfinite(b)]
        if not len(a) or not len(b): continue
        lo=min(a.min(),b.min()); hi=max(a.max(),b.max()); rg=max(hi-lo,1e-8); ks=ks_2samp(a,b).statistic; wd=wasserstein_distance(a,b)/rg; edges=np.linspace(lo,hi,bins+1); pa=np.histogram(a,bins=edges)[0].astype(float)+1e-12; pb=np.histogram(b,bins=edges)[0].astype(float)+1e-12; pa/=pa.sum();pb/=pb.sum(); js=float(jensenshannon(pa,pb,base=2.0)**2); vals.append(.4*ks+.3*min(1.,wd)+.3*min(1.,js))
    return float(np.mean(vals)) if vals else 0.

# ---------- backbone wrappers ----------
class Bundle:
    def __init__(self,backbone,model,prep,k,meta=None): self.backbone=backbone;self.model=model;self.prep=prep;self.k=k;self.meta=meta or {}

def _prep_fit(X):
    imp=SimpleImputer(strategy='median').fit(X); Z=imp.transform(X).astype(np.float32); sc=StandardScaler().fit(Z); return imp,sc

def _prep_apply(prep,X,scale=False):
    imp,sc=prep; Z=imp.transform(X).astype(np.float32); return sc.transform(Z).astype(np.float32) if scale else Z

def fit_catboost(st,idx,seed,gpu='0',iterations=None,valid_idx=None):
    from catboost import CatBoostClassifier
    idx=class_complete(st,idx); t0=time.perf_counter(); imp,sc=_prep_fit(st.X[idx]); Xtr=_prep_apply((imp,sc),st.X[idx],False); y=st.y[idx]; k=len(st.classes)
    it=int(iterations or int(os.environ.get('XB_CAT_ITERS','250')))
    args=dict(iterations=it,depth=int(os.environ.get('XB_CAT_DEPTH','6')),learning_rate=float(os.environ.get('XB_CAT_LR','.05')),random_seed=int(seed),verbose=False,allow_writing_files=False,loss_function='Logloss' if k==2 else 'MultiClass',thread_count=int(os.environ.get('XB_CPU_THREADS','4')))
    if os.environ.get('XB_FORCE_CPU','0')!='1': args.update(task_type='GPU',devices='0')
    m=CatBoostClassifier(**args)
    try:
        if valid_idx is not None and len(valid_idx): m.fit(Xtr,y,eval_set=(_prep_apply((imp,sc),st.X[valid_idx],False),st.y[valid_idx]),early_stopping_rounds=30,use_best_model=True)
        else: m.fit(Xtr,y)
    except Exception:
        args.pop('task_type',None);args.pop('devices',None);m=CatBoostClassifier(**args);m.fit(Xtr,y)
    meta={'iterations':int(getattr(m,'tree_count_',it))};return Bundle('catboost',m,(imp,sc),k,meta),time.perf_counter()-t0

def _torch_modules():
    import torch, torch.nn as nn
    class FTNet(nn.Module):
        def __init__(self,n_features,k,d=64,heads=8,layers=2,drop=.1):
            super().__init__(); self.w=nn.Parameter(torch.randn(n_features,d)*.02); self.b=nn.Parameter(torch.zeros(n_features,d)); self.cls=nn.Parameter(torch.zeros(1,1,d));
            enc=nn.TransformerEncoderLayer(d_model=d,nhead=heads,dim_feedforward=d*2,dropout=drop,activation='gelu',batch_first=True,norm_first=True); self.tr=nn.TransformerEncoder(enc,num_layers=layers); self.norm=nn.LayerNorm(d);self.head=nn.Linear(d,k)
        def forward(self,x):
            z=x.unsqueeze(-1)*self.w.unsqueeze(0)+self.b.unsqueeze(0); c=self.cls.expand(x.shape[0],-1,-1); z=torch.cat([c,z],1); z=self.tr(z)[:,0]; return self.head(self.norm(z))
    return torch,nn,FTNet

def _cuda_cleanup(*objs):
    try:
        import torch
        for o in objs:
            try:
                if hasattr(o,'model') and hasattr(o.model,'to'): o.model.to('cpu')
                elif hasattr(o,'to'): o.to('cpu')
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try: torch.cuda.ipc_collect()
            except Exception: pass
    except Exception:
        gc.collect()

def fit_ft(st,idx,seed,gpu='0',epochs=None,valid_idx=None):
    """FT-Transformer with automatic CUDA-OOM batch backoff.

    Each call is deterministic for a fixed seed.  On OOM only the batch size changes;
    architecture, data, optimizer, epochs and labels stay unchanged.
    """
    torch,nn,FTNet=_torch_modules()
    idx=class_complete(st,idx)
    imp,sc=_prep_fit(st.X[idx]); X=_prep_apply((imp,sc),st.X[idx],True); y=st.y[idx]; k=len(st.classes)
    use_cuda=torch.cuda.is_available() and os.environ.get('XB_FORCE_CPU','0')!='1'
    device=torch.device('cuda:0' if use_cuda else 'cpu')
    requested_bs=int(os.environ.get('XB_FT_BATCH','256'))
    min_bs=int(os.environ.get('XB_FT_MIN_BATCH','32'))
    maxep=int(epochs or int(os.environ.get('XB_FT_EPOCHS','8')))
    Xv=yv=None
    if valid_idx is not None and len(valid_idx):
        Xv=_prep_apply((imp,sc),st.X[valid_idx],True); yv=st.y[valid_idx]

    bs=max(min_bs,requested_bs)
    last_oom=None
    while True:
        model=opt=lossf=None
        try:
            t0=time.perf_counter()
            if use_cuda:
                print(f"FT_DEVICE physical_visible={os.environ.get('CUDA_VISIBLE_DEVICES','')} "
                      f"torch_device=0 name={torch.cuda.get_device_name(0)} "
                      f"batch={bs} n={len(X)} features={X.shape[1]}",flush=True)
                torch.cuda.empty_cache()
            torch.manual_seed(int(seed)); np.random.seed(int(seed))
            model=FTNet(
                X.shape[1],k,
                d=int(os.environ.get('XB_FT_D','48')),
                heads=int(os.environ.get('XB_FT_HEADS','4')),
                layers=int(os.environ.get('XB_FT_LAYERS','2')),
                drop=float(os.environ.get('XB_FT_DROP','.1'))
            ).to(device)
            opt=torch.optim.AdamW(model.parameters(),lr=float(os.environ.get('XB_FT_LR','8e-4')),weight_decay=1e-5)
            lossf=nn.CrossEntropyLoss()
            rng=np.random.default_rng(seed); best=None; bestloss=float('inf'); pat=0

            for ep in range(maxep):
                model.train(); order=rng.permutation(len(X))
                for a in range(0,len(order),bs):
                    ii=order[a:a+bs]
                    xb=torch.as_tensor(X[ii],device=device)
                    yb=torch.as_tensor(y[ii],device=device,dtype=torch.long)
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast(device_type='cuda',enabled=(device.type=='cuda')):
                        loss=lossf(model(xb),yb)
                    loss.backward(); opt.step()
                    del xb,yb,loss
                if Xv is not None:
                    model.eval(); ls=[]
                    with torch.no_grad():
                        for a in range(0,len(Xv),bs):
                            xb=torch.as_tensor(Xv[a:a+bs],device=device)
                            yb=torch.as_tensor(yv[a:a+bs],device=device,dtype=torch.long)
                            lv=lossf(model(xb),yb)
                            ls.append(float(lv.item())*len(xb))
                            del xb,yb,lv
                    vl=sum(ls)/len(Xv)
                    if vl<bestloss-1e-5:
                        bestloss=vl
                        # Store the validation-best state on CPU to avoid pinning another GPU-sized copy.
                        best={kk:vv.detach().cpu().clone() for kk,vv in model.state_dict().items()}
                        pat=0
                    else:
                        pat+=1
                    if pat>=2: break
            if best is not None:
                model.load_state_dict(best)
                del best
            meta={'epochs':ep+1,'device':str(device),'batch_size':bs}
            return Bundle('fttransformer',model,(imp,sc),k,meta),time.perf_counter()-t0

        except torch.OutOfMemoryError as e:
            last_oom=e
            print(f"FT_OOM_BACKOFF seed={seed} batch={bs} -> {bs//2 if bs>min_bs else 'CPU'}",flush=True)
            try:
                del opt,lossf,model
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try: torch.cuda.ipc_collect()
                except Exception: pass
            if bs>min_bs:
                bs=max(min_bs,bs//2)
                continue
            # Last-resort deterministic CPU completion for this one fit.
            if os.environ.get('XB_FT_ALLOW_CPU_FALLBACK','1')=='1':
                print(f"FT_CPU_FALLBACK seed={seed} after_cuda_oom=1",flush=True)
                old=os.environ.get('XB_FORCE_CPU')
                os.environ['XB_FORCE_CPU']='1'
                try:
                    return fit_ft(st,idx,seed,gpu=gpu,epochs=epochs,valid_idx=valid_idx)
                finally:
                    if old is None: os.environ.pop('XB_FORCE_CPU',None)
                    else: os.environ['XB_FORCE_CPU']=old
            raise last_oom

def fit_backbone(backbone,st,idx,seed,gpu='0',budget=None,source_tune=False):
    idx=np.asarray(idx,int)
    if source_tune and len(idx)>10:
        cut=max(1,int(.8*len(idx))); tr=idx[:cut];va=idx[cut:]
        if backbone=='catboost': probe,sec1=fit_catboost(st,tr,seed,gpu=gpu,iterations=budget,valid_idx=va);chosen=probe.meta.get('iterations',budget);del probe; final,sec2=fit_catboost(st,idx,seed,gpu=gpu,iterations=chosen,valid_idx=None);return final,sec1+sec2
        if backbone=='fttransformer':
            probe,sec1=fit_ft(st,tr,seed,gpu=gpu,epochs=budget,valid_idx=va)
            chosen=probe.meta.get('epochs',budget)
            _cuda_cleanup(probe); del probe
            final,sec2=fit_ft(st,idx,seed,gpu=gpu,epochs=chosen,valid_idx=None)
            return final,sec1+sec2
        raise ValueError(backbone)
    if backbone=='catboost': return fit_catboost(st,idx,seed,gpu=gpu,iterations=budget,valid_idx=None)
    if backbone=='fttransformer': return fit_ft(st,idx,seed,gpu=gpu,epochs=budget,valid_idx=None)
    raise ValueError(backbone)

def predict_bundle(b,X):
    Z=_prep_apply(b.prep,X,b.backbone=='fttransformer')
    if b.backbone=='catboost':
        p=np.asarray(b.model.predict_proba(Z),float)
        if p.ndim==1:p=np.c_[1-p,p]
        # CatBoost classes are normally 0..K-1 after class-complete replay.
        if p.shape[1]!=b.k:
            out=np.zeros((len(p),b.k)); cls=np.asarray(getattr(b.model,'classes_',np.arange(p.shape[1])),int); out[:,cls]=p;p=out
        return probs(p)
    torch,_,_=_torch_modules();device=next(b.model.parameters()).device;bs=int(os.environ.get('XB_FT_BATCH','1024'));out=[];b.model.eval()
    with torch.no_grad():
        for a in range(0,len(Z),bs):
            x=torch.as_tensor(Z[a:a+bs],device=device); out.append(torch.softmax(b.model(x),1).detach().cpu().numpy())
    return probs(np.vstack(out))

# ---------- exact-H probability repair ----------
def twofold(y,seed):
    y=np.asarray(y,int);rng=np.random.default_rng(seed);f=[[],[]]
    for c in np.unique(y):
        ii=np.where(y==c)[0].copy();rng.shuffle(ii)
        for j,v in enumerate(ii):f[j%2].append(int(v))
    return [np.asarray(sorted(x),int) for x in f]

def logp(p): return np.log(np.clip(probs(p),1e-8,1))
CAL=[f'temp:{x}' for x in (.5,.67,.8,1.,1.25,1.5,2.)]+[f'prior:{x}' for x in (.25,.5,.75,1.)]+[f'dir:{x}' for x in (.1,1.,10.)]
def fit_cal(kind,p,y,k):
    p=probs(p);y=np.asarray(y,int)
    if len(y)<max(6,2*k) or len(np.unique(y))<2:return None
    if kind.startswith('temp:'):return ('temp',float(kind.split(':')[1]))
    if kind.startswith('prior:'):
        g=float(kind.split(':')[1]);emp=np.bincount(y,minlength=k).astype(float)+1.;emp/=emp.sum();return ('prior',(emp/(p.mean(0)+1e-8))**g)
    if kind.startswith('dir:'):
        if len(np.unique(y))<k:return None
        try:return ('dir',LogisticRegression(C=float(kind.split(':')[1]),max_iter=300,solver='lbfgs').fit(logp(p),y))
        except Exception:return None

def apply_cal(m,p):
    p=probs(p)
    if m[0]=='temp':return probs(np.exp(logp(p)/m[1]))
    if m[0]=='prior':return probs(p*m[1][None,:])
    return probs(m[1].predict_proba(logp(p)))

def qstats(a,q,y):
    a=probs(a);q=probs(q);e=a-np.eye(a.shape[1])[np.asarray(y,int)];d=q-a;return float(np.mean(np.sum(e*e,1))),float(np.mean(np.sum(e*d,1))),float(np.mean(np.sum(d*d,1)))
def eta_star(st):
    _,B,A=st;return float(np.clip(-B/A,0,1)) if A>1e-18 else 0.
def qb(st,e):C,B,A=st;return C+2*B*e+A*e*e
def hcoeff(a,q,n):
    a=probs(a);q=probs(q);d=q-a;A=float(np.sum(np.sum(d*d,1))/n);B=float(2*np.sum(np.sum(a*d,1)-np.min(d,1))/n);return A,max(0.,B)
def hcap(h,eps=EPS):
    A,B=h
    if A<=1e-18:return float('inf') if B<=1e-18 else eps/B
    return max(0.,(-B+math.sqrt(max(0.,B*B+4*A*eps)))/(2*A))
def hval(h,e):A,B=h;return float(A*e*e+B*e)

def repair_endpoint(base,y,can,hist,tag,eps=EPS):
    base=probs(base);can=np.asarray(can,int);k=base.shape[1];mask=np.ones(len(base),bool);mask[can]=False;bulk=np.flatnonzero(mask);best={'can_brier':brier(base[can],y[can]),'target':base.copy(),'H':0.,'detail':'none'}
    for lb in (1,2,4):
        hp=[];hy=[]
        for h in hist[-lb:]:hp.append(h[0]);hy.append(h[1])
        hp=np.vstack(hp) if hp else None;hy=np.concatenate(hy) if hy else None
        for kind in CAL:
            yc=y[can]
            if len(can)<max(6,2*k) or len(np.unique(yc))<2:continue
            fs=twofold(yc,stable_seed('xb-cf',*tag,lb,kind));oof=np.full((len(can),k),np.nan);ok=True
            for fd in fs:
                tr=np.setdiff1d(np.arange(len(can)),fd);pp=base[can[tr]];yy=yc[tr]
                if hp is not None:pp=np.vstack([hp,pp]);yy=np.concatenate([hy,yy])
                m=fit_cal(kind,pp,yy,k)
                if m is None:ok=False;break
                oof[fd]=apply_cal(m,base[can[fd]])
            if not ok or not np.all(np.isfinite(oof)):continue
            pp=base[can];yy=yc
            if hp is not None:pp=np.vstack([hp,pp]);yy=np.concatenate([hy,yy])
            m=fit_cal(kind,pp,yy,k)
            if m is None:continue
            st=qstats(base[can],oof,yc);e=eta_star(st)
            if e<=0:continue
            q=apply_cal(m,base[bulk]);hc=hcoeff(base[bulk],q,len(base));e=min(e,hcap(hc,eps))
            if e<=0:continue
            score=qb(st,e)
            if score<best['can_brier']-1e-15:
                out=base.copy();out[bulk]=probs(base[bulk]+e*(q-base[bulk]));out[can]=base[can];best={'can_brier':score,'target':out,'H':hval(hc,e),'detail':f'{kind}|lb{lb}|eta{e:.5f}'}
    return best
