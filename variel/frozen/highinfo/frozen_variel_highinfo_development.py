#!/usr/bin/env python3
"""
VARIEL High-Information Cabinet — Development Closure
=====================================================

Development/challenge datasets are fixed as:
    Bank + Electricity + Gas + Airlines.

Bank/Electricity/Gas high-information results are FROZEN and reused from the
pre-existing DTRAR_HighInfoRetrainRepair development experiment. They are not
rerun and are not altered here.

This script runs ONLY the missing Airlines development cell.

Scientific question
-------------------
Can a HIGH-INFORMATION repair drawer strictly improve the raw rolling
full-feedback retraining anchor at the SAME 100% target-label cost?

If yes, then raw rolling retraining is dominated as an action:
    same label fraction = 1.0
    lower Brier = better

The frozen sparse VARIEL/DTRAR core is NOT changed.

Airlines protocol
-----------------
Canonical scikit-multiflow Airlines stream:
    539,383 rows
    file order preserved
    first 30% source
    remaining 70% target
    14 contiguous target windows
    10 seeds

For each target window W_t:
  1. train/retrain XGBoost using source + ALL completed target windows;
  2. issue raw rolling-retraining probabilities for W_t;
  3. use only PREVIOUS-window issued probabilities + labels to construct:
        - full probability calibrator
        - residual calibrator
  4. reveal ~1% of current W_t as geometry canaries;
     those labels are NOT an extra total cost because the FULL regime acquires
     100% of W_t after prediction anyway;
  5. estimate current Brier utility and choose risk-optimal eta;
  6. utility profile: apply best positive repair;
  7. safety profile: additionally enforce exact prospective H <= .02;
  8. canary predictions remain the already-issued retraining probabilities.

No Rialto, Spam, INSECTS or BLE outcomes are read.
"""
from pathlib import Path
from dataclasses import dataclass
import os, math, json, hashlib, urllib.request, shutil, zipfile
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
PROJECT=Path(os.environ.get(
    "DTRAR_PROJECT",
    str(Path.home()/"Documents"/"DTRA_Start"/"DTRA")
)).resolve()
OUT=PROJECT/"artifacts"/"variel_highinfo_cabinet_development"
OUT.mkdir(parents=True,exist_ok=True)

URLS=[
    "https://github.com/scikit-multiflow/streaming-datasets/raw/refs/heads/master/airlines.csv",
    "https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/master/airlines.csv",
]
EXPECTED_ROWS=539383
EXPECTED_COLUMNS=['Airline','Flight','AirportFrom','AirportTo','DayOfWeek','Time','Length','Delay']
CAT_COLS=['Airline','AirportFrom','AirportTo']
SOURCE_FRACTION=.30
TARGET_WINDOWS=14
SEEDS=list(range(10))
BUFFER=3
DECAY=.9
PENALTY=.001
CANARY_FRACTION=.01
EPS_SAFE=.02

def must(c,m):
    if not c: raise RuntimeError(m)

def stable_seed(*parts):
    return int.from_bytes(
        hashlib.sha256("|".join(map(str,parts)).encode()).digest()[:8],
        "little"
    )%(2**32-1)

def probs(p):
    p=np.asarray(p,float)
    p=np.clip(p,1e-12,np.inf)
    return p/p.sum(axis=1,keepdims=True)

def brier_rows(p,y):
    p=probs(p); y=np.asarray(y,int)
    return np.sum((p-np.eye(p.shape[1])[y])**2,axis=1)

def prospective_H(anchor,out):
    anchor=probs(anchor); out=probs(out)
    d=out-anchor
    return float(np.mean(
        np.sum(d*d,axis=1)
        +2*(np.sum(anchor*d,axis=1)-np.min(d,axis=1))
    ))

class ProbabilityCalibrator:
    def __init__(self,penalty=PENALTY): self.penalty=float(penalty)
    def fit(self,p,y,w=None):
        p=probs(p); y=np.asarray(y,int); self.k=p.shape[1]
        w=np.ones(len(y),float) if w is None else np.asarray(w,float)
        w=w/w.sum()
        if self.k==2:
            X=np.column_stack([np.log(p[:,1]),-np.log(p[:,0]),np.ones(len(p))])
            init=np.array([1.,1.,0.])
            def obj(th):
                z=X@th; delta=th-init
                return (
                    np.sum(w*(np.logaddexp(0,z)-y*z))+self.penalty*(delta@delta),
                    X.T@(w*(expit(z)-y))+2*self.penalty*delta
                )
            res=minimize(
                obj,init,jac=True,
                bounds=[(1e-8,None),(1e-8,None),(None,None)],
                method="L-BFGS-B"
            )
        else:
            X=np.log(p); T=np.eye(self.k)[y]
            init=np.r_[np.eye(self.k).ravel(),np.zeros(self.k)]
            def obj(th):
                W=th[:self.k**2].reshape(self.k,self.k)
                b=th[self.k**2:]
                z=X@W.T+b
                lp=z-logsumexp(z,axis=1,keepdims=True)
                e=w[:,None]*(np.exp(lp)-T)
                delta=th-init
                return (
                    -np.sum(w*lp[np.arange(len(y)),y])+self.penalty*(delta@delta),
                    np.r_[(e.T@X).ravel(),e.sum(0)]+2*self.penalty*delta
                )
            res=minimize(obj,init,jac=True,method="L-BFGS-B")
        if not res.success: raise RuntimeError(res.message)
        self.theta=res.x
        return self
    def predict_proba(self,p):
        p=probs(p)
        if self.k==2:
            X=np.column_stack([np.log(p[:,1]),-np.log(p[:,0]),np.ones(len(p))])
            q=expit(X@self.theta)
            return probs(np.c_[1-q,q])
        W=self.theta[:self.k**2].reshape(self.k,self.k)
        b=self.theta[self.k**2:]
        z=np.log(p)@W.T+b
        return probs(np.exp(z-logsumexp(z,axis=1,keepdims=True)))

class ResidualCalibrator:
    def __init__(self,k,ess,penalty=PENALTY):
        self.k=k; self.ess=max(float(ess),1.); self.penalty=float(penalty)
    def fit(self,p,y,w):
        p=probs(p); y=np.asarray(y,int)
        w=np.asarray(w,float); w=w/w.sum()
        X=np.log(p); T=np.eye(self.k)[y]
        init=np.r_[1.,np.zeros(self.k)]
        offpen=(self.k-1)/self.ess
        def obj(th):
            z=th[0]*X+th[1:]
            lp=z-logsumexp(z,axis=1,keepdims=True)
            e=w[:,None]*(np.exp(lp)-T)
            loss=(-np.sum(w*lp[np.arange(len(y)),y])
                  +self.penalty*(th[0]-1)**2
                  +offpen*np.dot(th[1:],th[1:]))
            grad=np.r_[
                np.sum(e*X)+2*self.penalty*(th[0]-1),
                e.sum(0)+2*offpen*th[1:]
            ]
            return loss,grad
        res=minimize(
            obj,init,jac=True,
            bounds=[(1e-8,None)]+[(None,None)]*self.k,
            method="L-BFGS-B"
        )
        if not res.success: raise RuntimeError(res.message)
        self.theta=res.x
        return self
    def predict_proba(self,p):
        z=self.theta[0]*np.log(probs(p))+self.theta[1:]
        return probs(np.exp(z-logsumexp(z,axis=1,keepdims=True)))

def multi_geometry(anchor,cands):
    widths=[]
    for q in cands.values():
        d=q-anchor
        center=np.sum(anchor*d,axis=1)
        z=d-center[:,None]
        widths.append(np.max(z,axis=1)-np.min(z,axis=1))
    return np.max(np.column_stack(widths),axis=1)

def choose_canaries(anchor,cands,m,rng):
    n=len(anchor)
    m=min(max(1,int(m)),max(1,n-1))
    width=multi_geometry(anchor,cands)
    H=min(8,m,n)
    chunks=[
        np.asarray(x,int)
        for x in np.array_split(np.argsort(width,kind="mergesort"),H)
        if len(x)
    ]
    masses=np.asarray([
        len(x)*max(float(np.max(width[x])),1e-12) for x in chunks
    ])
    caps=np.asarray([len(x) for x in chunks],int)
    counts=np.zeros(len(chunks),int)
    for _ in range(m):
        score=np.where(counts<caps,masses/(counts+1),-1)
        counts[int(np.argmax(score))]+=1
    strata=[]; chosen=[]
    for pop,nh in zip(chunks,counts):
        s=np.sort(rng.choice(pop,size=int(nh),replace=False))
        chosen+=s.tolist()
        strata.append({"population":pop,"sample":s})
    can=np.sort(np.asarray(chosen,int))
    mask=np.ones(n,bool); mask[can]=False
    return strata,can,np.flatnonzero(mask)

def estimate(anchor,q,y,strata,bulk):
    d=q-anchor
    center=np.sum(anchor*d,axis=1)
    z=d-center[:,None]
    M=max(1,len(bulk)); D=0.
    for st in strata:
        pop,samp=st["population"],st["sample"]
        if len(samp)==0: continue
        bulk_h=np.setdiff1d(pop,samp,assume_unique=True)
        D+=(len(bulk_h)/M)*float(np.mean(z[samp,y[samp]]))
    C=float(np.mean(np.sum(d[bulk]**2,axis=1))) if len(bulk) else 0.
    eta=float(np.clip(D/C,0,1)) if C>1e-18 else 0.
    gain=2*D*eta-C*eta*eta
    return D,C,eta,gain

def clip_H(anchor,q,bulk,can,eta,eps):
    if eta<=0: return 0.,0.
    def h(e):
        out=anchor.copy()
        out[bulk]=probs(anchor[bulk]+e*(q[bulk]-anchor[bulk]))
        out[can]=anchor[can]
        return prospective_H(anchor,out)
    h0=h(eta)
    if h0<=eps+1e-12: return eta,h0
    lo,hi=0.,eta
    for _ in range(70):
        mid=(lo+hi)/2
        if h(mid)<=eps: lo=mid
        else: hi=mid
    return lo,h(lo)

def download(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size>1_000_000:
        return
    last=None
    for url in URLS:
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req,timeout=600) as r,open(path,"wb") as f:
                shutil.copyfileobj(r,f)
            if path.stat().st_size>1_000_000: return
        except Exception as e:
            last=repr(e)
            if path.exists(): path.unlink()
    raise RuntimeError(last)

def load_airlines():
    # Prefer the already-downloaded canonical one-shot file.
    candidates=[
        PROJECT/"artifacts"/"dtrar_final_airlines_oneshot"/"raw"/"airlines.csv",
        PROJECT/"datasets"/"airlines"/"airlines.csv",
    ]
    p=next((x for x in candidates if x.exists()),candidates[0])
    download(p)

    df=pd.read_csv(p)
    df=df.loc[:,[c for c in df.columns if not str(c).lower().startswith("unnamed:")]]
    must(list(df.columns)==EXPECTED_COLUMNS,"Airlines header mismatch")
    must(len(df)==EXPECTED_ROWS,f"Airlines rows={len(df)}")
    must(not df.isna().any().any(),"Airlines contains missing values")

    n=len(df)
    source_n=int(math.floor(SOURCE_FRACTION*n))
    y=pd.to_numeric(df["Delay"],errors="raise").astype(int).to_numpy()
    labels=sorted(np.unique(y).tolist())
    must(labels==sorted(np.unique(y[:source_n]).tolist()),"source lacks class")

    X=np.empty((n,7),np.float32)
    for j,c in enumerate(EXPECTED_COLUMNS[:-1]):
        if c in CAT_COLS:
            levels=sorted(df[c].iloc[:source_n].astype(str).unique().tolist())
            mp={v:i for i,v in enumerate(levels)}
            X[:,j]=(df[c].astype(str).map(mp).fillna(-1)
                    .astype(np.float32).to_numpy())
        else:
            X[:,j]=pd.to_numeric(df[c],errors="raise").astype(np.float32).to_numpy()

    source=np.arange(source_n,dtype=int)
    target=np.arange(source_n,n,dtype=int)
    windows=[np.asarray(w,int) for w in np.array_split(target,TARGET_WINDOWS)]
    return p,X,y,source,target,windows,labels

def xgb_args(seed,k,rounds):
    return dict(
        n_estimators=int(rounds),
        max_depth=6,learning_rate=.05,
        subsample=.9,colsample_bytree=.9,min_child_weight=2,reg_lambda=1.,
        tree_method="hist",device="cuda",n_jobs=4,random_state=int(seed),
        objective="binary:logistic" if k==2 else "multi:softprob",
        eval_metric="logloss" if k==2 else "mlogloss",
    )

def choose_rounds(X,y,source,seed,k):
    cut=max(1,int(math.floor(.8*len(source))))
    tr=source[:cut]; va=source[cut:]
    imp=SimpleImputer(strategy="median").fit(X[tr])
    a=xgb_args(seed,k,900)
    m=XGBClassifier(**a,early_stopping_rounds=60)
    m.fit(
        imp.transform(X[tr]).astype(np.float32),y[tr],
        eval_set=[(imp.transform(X[va]).astype(np.float32),y[va])],
        verbose=False
    )
    return max(1,int(getattr(m,"best_iteration",899))+1)

def rolling_anchor(X,y,source,windows,seed,rounds,k,cache):
    cache.mkdir(parents=True,exist_ok=True)
    issued=[]; labels=[]; rowids=[]
    history=[]
    for t,ix in enumerate(windows):
        pf=cache/f"window_{t+1}.npz"
        if pf.exists():
            with np.load(pf) as z:
                p=np.asarray(z["p"],float)
        else:
            fit_idx=np.r_[source,np.concatenate(history) if history else np.array([],int)]
            imp=SimpleImputer(strategy="median").fit(X[fit_idx])
            m=XGBClassifier(**xgb_args(seed,k,rounds))
            m.fit(imp.transform(X[fit_idx]).astype(np.float32),y[fit_idx],verbose=False)
            p=np.asarray(
                m.predict_proba(imp.transform(X[ix]).astype(np.float32)),
                float
            )
            p=probs(p)
            np.savez_compressed(pf,p=p,row_id=ix)
        issued.append(p); labels.append(y[ix]); rowids.append(ix)
        history.append(ix.copy())
    return np.vstack(issued),np.concatenate(labels),np.concatenate(rowids)

def history_positions(window_positions,t):
    ids=[]; ws=[]
    for u in range(max(0,t-BUFFER),t):
        pos=np.asarray(window_positions[u],int)
        decay=DECAY**(t-1-u)
        ids.extend(pos.tolist())
        ws.extend([decay/max(1,len(pos))]*len(pos))
    return np.asarray(ids,int),np.asarray(ws,float)

def run_seed(X,y,source,target,windows,labels,seed,file_sha):
    k=len(labels)
    rounds=choose_rounds(X,y,source,seed,k)
    cache=OUT/"airlines_rolling_cache"/f"seed_{seed}"
    issued,yy,row_id=rolling_anchor(X,y,source,windows,seed,rounds,k,cache)
    must(np.array_equal(row_id,target),"row id mismatch")
    must(np.array_equal(yy,y[target]),"label mismatch")

    # Positions inside concatenated target probability matrix.
    pos={int(r):i for i,r in enumerate(row_id)}
    winpos=[
        np.asarray([pos[int(r)] for r in ix],int)
        for ix in windows
    ]

    pred=issued.copy()
    predS=issued.copy()
    traces=[]

    for t,posix in enumerate(winpos):
        anchor=issued[posix]
        local_y=yy[posix]
        hist,w=history_positions(winpos,t)

        if len(hist):
            full=ProbabilityCalibrator().fit(
                issued[hist],yy[hist],w
            ).predict_proba(anchor)
            tw=w/w.sum()
            ess=float(1.0/np.sum(tw**2))
            residual=ResidualCalibrator(
                anchor.shape[1],ess
            ).fit(issued[hist],yy[hist],w).predict_proba(anchor)
            cands={"fullcal":full,"residual":residual}
        else:
            cands={"fullcal":anchor.copy(),"residual":anchor.copy()}

        m=max(1,int(math.floor(CANARY_FRACTION*len(posix))))
        rng=np.random.default_rng(stable_seed("variel_highinfo",seed,t,file_sha))
        strata,can,bulk=choose_canaries(anchor,cands,m,rng)

        ests={}
        best="raw_retrain"; best_gain=0.; best_eta=0.
        for name,q in cands.items():
            D,C,eta,gain=estimate(anchor,q,local_y,strata,bulk)
            ests[name]=(D,C,eta,gain)
            if gain>best_gain:
                best=name; best_gain=gain; best_eta=eta

        out=anchor.copy()
        if best!="raw_retrain":
            out[bulk]=probs(
                anchor[bulk]+best_eta*(cands[best][bulk]-anchor[bulk])
            )
        out[can]=anchor[can]
        pred[posix]=out

        bestS="raw_retrain"; bestSgain=0.; bestSeta=0.
        for name,q in cands.items():
            D,C,eta,_=ests[name]
            etaS,_=clip_H(anchor,q,bulk,can,eta,EPS_SAFE)
            score=2*D*etaS-C*etaS*etaS
            if score>bestSgain:
                bestS=name; bestSgain=score; bestSeta=etaS
        outS=anchor.copy()
        if bestS!="raw_retrain":
            outS[bulk]=probs(
                anchor[bulk]+bestSeta*(cands[bestS][bulk]-anchor[bulk])
            )
        outS[can]=anchor[can]
        predS[posix]=outS
        H=prospective_H(anchor,outS)
        must(H<=EPS_SAFE+2e-9,f"H violation {H}")

        traces.append({
            "dataset":"airlines","seed":seed,"window":t+1,
            "canary_n":len(can),
            "utility_choice":best,"utility_eta":best_eta,
            "safety_choice":bestS,"safety_eta":bestSeta,
            "raw_retrain_brier":float(np.mean(brier_rows(anchor,local_y))),
            "utility_brier":float(np.mean(brier_rows(out,local_y))),
            "safety02_brier":float(np.mean(brier_rows(outS,local_y))),
            "safety_H":H,
        })

    raw=float(np.mean(brier_rows(issued,yy)))
    util=float(np.mean(brier_rows(pred,yy)))
    safe=float(np.mean(brier_rows(predS,yy)))
    return {
        "dataset":"airlines","seed":seed,
        "raw_retrain_brier":raw,
        "utility_brier":util,
        "safety02_brier":safe,
        "improvement_vs_retrain":raw-util,
        "beats_raw_retrain":bool(util<raw),
        "max_safety_H":float(max(x["safety_H"] for x in traces)),
        "rounds":int(rounds),
        "full_feedback_label_fraction":1.0,
        "highinfo_total_label_fraction":1.0,
        "current_canary_fraction_within_full_regime":CANARY_FRACTION,
    },pd.DataFrame(traces)

def main():
    print("VARIEL_HIGHINFO_DEVELOPMENT_BEGIN",flush=True)
    p,X,y,source,target,windows,labels=load_airlines()
    file_sha=hashlib.sha256(Path(p).read_bytes()).hexdigest()
    print(f"AIRLINES_DATASET_LOCK sha256={file_sha}",flush=True)

    rows=[]; traces=[]
    with threadpool_limits(limits=1):
        for seed in SEEDS:
            print(f"AIRLINES_HIGHINFO_BEGIN {seed+1}/10 seed={seed}",flush=True)
            s,tr=run_seed(X,y,source,target,windows,labels,seed,file_sha)
            rows.append(s); traces.append(tr)
            print(
                f"AIRLINES_HIGHINFO_DONE seed={seed} "
                f"raw={s['raw_retrain_brier']:.9f} "
                f"utility={s['utility_brier']:.9f} "
                f"delta={s['improvement_vs_retrain']:.9f} "
                f"H={s['max_safety_H']:.9f}",
                flush=True
            )

    air=pd.DataFrame(rows)
    air_trace=pd.concat(traces,ignore_index=True)
    air.to_csv(OUT/"AIRLINES_HIGHINFO_SEEDWISE.csv",index=False)
    air_trace.to_csv(OUT/"AIRLINES_HIGHINFO_WINDOW_TRACE.csv",index=False)

    frozen=pd.read_csv(HERE/"FROZEN_HIGHINFO_BEG_SEEDWISE.csv")
    dev=pd.concat([frozen,air[frozen.columns.intersection(air.columns)]],
                  ignore_index=True,sort=False)

    # Normalize to common required fields.
    required=["dataset","seed","raw_retrain_brier","utility_brier",
              "safety02_brier","improvement_vs_retrain",
              "beats_raw_retrain","max_safety_H"]
    dev=pd.concat([
        frozen[required],
        air[required]
    ],ignore_index=True)
    dev.to_csv(OUT/"VARIEL_HIGHINFO_FOURDEV_SEEDWISE.csv",index=False)

    sums=[]
    for ds,g in dev.groupby("dataset"):
        sums.append({
            "dataset":ds,
            "mean_raw_retrain_brier":float(g.raw_retrain_brier.mean()),
            "mean_highinfo_utility_brier":float(g.utility_brier.mean()),
            "mean_highinfo_safety_brier":float(g.safety02_brier.mean()),
            "mean_improvement_vs_raw_retrain":
                float(g.improvement_vs_retrain.mean()),
            "wins_vs_raw_retrain":
                int((g.improvement_vs_retrain>0).sum()),
            "beats_raw_full_retrain_mean":
                bool(g.utility_brier.mean()<g.raw_retrain_brier.mean()),
            "max_safety_H":float(g.max_safety_H.max()),
        })
    summary=pd.DataFrame(sums)
    summary.to_csv(OUT/"VARIEL_HIGHINFO_FOURDEV_SUMMARY.csv",index=False)

    assessment={
        "architecture":
            "VARIEL outer maintenance cabinet with HIGH-INFO FULL+REPAIR drawer",
        "development_datasets":
            ["bank","electricity","gas","airlines"],
        "external_datasets_touched":False,
        "frozen_sparse_variel_changed":False,
        "highinfo_action_total_label_fraction":1.0,
        "raw_full_retraining_total_label_fraction":1.0,
        "bank_electricity_gas_results_reused_from_frozen_preexisting_development_archive":True,
        "airlines_new_development_evaluation_only":True,
        "development_wins_vs_raw_full_retrain":
            int((dev.improvement_vs_retrain>0).sum()),
        "development_total_seed_decisions":int(len(dev)),
        "all_four_dataset_means_beat_raw_full_retrain":
            bool(summary.beats_raw_full_retrain_mean.all()),
        "all_40_seeds_beat_raw_full_retrain":
            bool((dev.improvement_vs_retrain>0).all()),
        "safety_H02_holds":
            bool((dev.max_safety_H<=EPS_SAFE+2e-9).all()),
        "raw_full_retraining_is_dominated_if_gate_passes":
            bool((dev.improvement_vs_retrain>0).all()),
        "next_step":
            "If all_40_seeds_beat_raw_full_retrain and safety_H02_holds are true, freeze the HIGH-INFO drawer and test it unchanged on Rialto, Spam, INSECTS, and BLE RSSI.",
    }
    (OUT/"VARIEL_HIGHINFO_DEVELOPMENT_ASSESSMENT.json").write_text(
        json.dumps(assessment,indent=2,sort_keys=True)
    )

    review=PROJECT/"VARIEL_HighInfoCabinet_Development_Review.zip"
    with zipfile.ZipFile(review,"w",zipfile.ZIP_DEFLATED) as z:
        for q in sorted(OUT.iterdir()):
            if q.is_file(): z.write(q,q.name)
        z.write(Path(__file__),Path(__file__).name)
        z.write(HERE/"FROZEN_HIGHINFO_BEG_SEEDWISE.csv",
                "FROZEN_HIGHINFO_BEG_SEEDWISE.csv")
        z.write(HERE/"FROZEN_HIGHINFO_BEG_SUMMARY.csv",
                "FROZEN_HIGHINFO_BEG_SUMMARY.csv")

    print("\n=== VARIEL HIGH-INFO FOUR-DEVELOPMENT ASSESSMENT ===")
    print(json.dumps(assessment,indent=2,sort_keys=True))
    print("\n=== FOUR-DEVELOPMENT SUMMARY ===")
    print(summary.to_string(index=False))
    print(f"\nVARIEL_HIGHINFO_DEVELOPMENT_COMPLETE review={review.name}",flush=True)

if __name__=="__main__":
    main()
