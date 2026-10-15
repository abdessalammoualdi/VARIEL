#!/usr/bin/env python3
"""
High-information retraining drawer experiment.

Goal
----
Test whether the historical rolling full-feedback XGBoost reference can be made
strictly stronger by adding a causal Brier-targeted probability-repair layer.

The frozen 5% sparse router is NOT modified.

For each current window t:
  anchor r_t = exact historical rolling full-feedback XGBoost probabilities.

Using only labels from PREVIOUS windows, construct two repair drawers:
  1) fullcal  : monotone-beta (binary) / regularized-Dirichlet (multiclass)
  2) residual : scalar log-temperature + class offsets, regularized to identity

About 1% of the current window is revealed early as geometry canaries. These
labels are a subset of the full-feedback labels that will be acquired anyway,
so total label usage remains 100% in this high-information regime.

The current canaries estimate Brier utility for each repair direction q-r.
For each drawer:
    eta = clip(Dhat/C,0,1)
    Ghat = 2 Dhat eta - C eta^2
Raw rolling retraining is the zero-action drawer. The best positive drawer is
applied to the non-canary bulk; canaries retain the already-issued retraining
probability.

We also evaluate an exact-H<=0.02 version relative to the rolling-retraining
anchor, and a retrospective oracle ceiling.

The exact rolling predictor is reproduced from the archived Focus Review
source_code/scripts/run_predictor_reference.py, not from a reconstructed model.
Per-seed Brier is checked against the authoritative recovery CSV.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, json, math, os, shutil, subprocess, sys, zipfile
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from threadpoolctl import threadpool_limits

import dtrar_final_core as fc
from dtra.data import load_dataset

PROJECT=Path(__file__).resolve().parent
OUT=PROJECT/"artifacts"/"highinfo_retrain_repair"
SOURCE=OUT/"focus_source"
ROLL=OUT/"rolling_outputs"

DATASETS=["bank","electricity","gas"]
SEEDS=list(range(10))
BUFFER=3
DECAY=0.9
PENALTY=0.001
CANARY_FRACTION=0.01
EPS_SAFE=0.02

FULL_FEEDBACK_REFERENCE={"bank":0.364310,"electricity":0.365481,"gas":0.585471}
FROZEN_SPARSE_ROUTER={"bank":0.347409,"electricity":0.355584,"gas":0.611952}

def must(c,m):
    if not c: raise RuntimeError(m)

def probs(p):
    p=np.asarray(p,float); p=np.clip(p,1e-12,np.inf); return p/p.sum(axis=1,keepdims=True)

def brier_rows(p,y):
    p=probs(p); y=np.asarray(y,int); return np.sum((p-np.eye(p.shape[1])[y])**2,axis=1)

def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str,parts)).encode()).digest()[:8],"little")%(2**32-1)

class ProbabilityCalibrator:
    def __init__(self,penalty=PENALTY): self.penalty=float(penalty)
    def fit(self,p,y,w=None):
        p=probs(p); y=np.asarray(y,int); self.k=p.shape[1]
        w=np.ones(len(y),float) if w is None else np.asarray(w,float); w=w/w.sum()
        if self.k==2:
            X=np.column_stack([np.log(p[:,1]),-np.log(p[:,0]),np.ones(len(p))]); init=np.array([1.,1.,0.])
            def obj(th):
                z=X@th; delta=th-init
                return np.sum(w*(np.logaddexp(0,z)-y*z))+self.penalty*(delta@delta), X.T@(w*(expit(z)-y))+2*self.penalty*delta
            res=minimize(obj,init,jac=True,bounds=[(1e-8,None),(1e-8,None),(None,None)],method="L-BFGS-B")
        else:
            X=np.log(p); T=np.eye(self.k)[y]; init=np.r_[np.eye(self.k).ravel(),np.zeros(self.k)]
            def obj(th):
                W=th[:self.k**2].reshape(self.k,self.k); b=th[self.k**2:]
                z=X@W.T+b; lp=z-logsumexp(z,axis=1,keepdims=True); e=w[:,None]*(np.exp(lp)-T); delta=th-init
                return -np.sum(w*lp[np.arange(len(y)),y])+self.penalty*(delta@delta), np.r_[(e.T@X).ravel(),e.sum(0)]+2*self.penalty*delta
            res=minimize(obj,init,jac=True,method="L-BFGS-B")
        if not res.success: raise RuntimeError(res.message)
        self.theta=res.x; return self
    def predict_proba(self,p):
        p=probs(p)
        if self.k==2:
            X=np.column_stack([np.log(p[:,1]),-np.log(p[:,0]),np.ones(len(p))]); q=expit(X@self.theta)
            return probs(np.c_[1-q,q])
        W=self.theta[:self.k**2].reshape(self.k,self.k); b=self.theta[self.k**2:]
        z=np.log(p)@W.T+b; return probs(np.exp(z-logsumexp(z,axis=1,keepdims=True)))

class ResidualCalibrator:
    def __init__(self,k,ess,penalty=PENALTY):
        self.k=k; self.ess=max(float(ess),1.); self.penalty=float(penalty)
    def fit(self,p,y,w):
        p=probs(p); y=np.asarray(y,int); w=np.asarray(w,float); w=w/w.sum()
        X=np.log(p); T=np.eye(self.k)[y]; init=np.r_[1.,np.zeros(self.k)]
        offpen=(self.k-1)/self.ess
        def obj(th):
            z=th[0]*X+th[1:]; lp=z-logsumexp(z,axis=1,keepdims=True); e=w[:,None]*(np.exp(lp)-T)
            loss=-np.sum(w*lp[np.arange(len(y)),y])+self.penalty*(th[0]-1)**2+offpen*np.dot(th[1:],th[1:])
            grad=np.r_[np.sum(e*X)+2*self.penalty*(th[0]-1), e.sum(0)+2*offpen*th[1:]]
            return loss,grad
        res=minimize(obj,init,jac=True,bounds=[(1e-8,None)]+[(None,None)]*self.k,method="L-BFGS-B")
        if not res.success: raise RuntimeError(res.message)
        self.theta=res.x; return self
    def predict_proba(self,p):
        z=self.theta[0]*np.log(probs(p))+self.theta[1:]
        return probs(np.exp(z-logsumexp(z,axis=1,keepdims=True)))

def find_focus_archive():
    cands=[]
    for name in ["DTRA_Focus_Review_03_of_03.zip","DTRA_Focus_Review_02_of_03.zip","DTRA_Focus_Review_01_of_03.zip"]:
        cands += list(PROJECT.rglob(name))
    cands=sorted(set(cands),key=lambda p:(0 if "focus_v05" in str(p) else 1,len(str(p))))
    for p in cands:
        try:
            with zipfile.ZipFile(p) as z:
                if "source_code/scripts/run_predictor_reference.py" in z.namelist():
                    return p
        except Exception:
            pass
    raise RuntimeError("Could not find a Focus Review ZIP containing source_code/scripts/run_predictor_reference.py")

def prepare_focus_source():
    marker=SOURCE/"scripts"/"run_predictor_reference.py"
    if not marker.exists():
        arc=find_focus_archive()
        SOURCE.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(arc) as z:
            for n in z.namelist():
                if not n.startswith("source_code/") or n.endswith("/"): continue
                rel=Path(n).relative_to("source_code")
                dest=SOURCE/rel; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(z.read(n))
        (OUT/"FOCUS_ARCHIVE_USED.txt").write_text(str(arc)+"\nsha256="+hashlib.sha256(arc.read_bytes()).hexdigest()+"\n")
    # data symlink
    (SOURCE/"data").mkdir(exist_ok=True)
    rawlink=SOURCE/"data"/"raw"
    if not rawlink.exists():
        os.symlink(PROJECT/"data"/"raw",rawlink,target_is_directory=True)
    # Reuse exact historical frozen source-model caches if present.
    (SOURCE/"artifacts").mkdir(exist_ok=True)
    model_link=SOURCE/"artifacts"/"models"
    historical=PROJECT/"focus_v05"/"artifacts"/"models"
    if historical.exists() and not model_link.exists():
        os.symlink(historical,model_link,target_is_directory=True)
    return marker

def expected_table():
    df=pd.read_csv(PROJECT/"AUTHORITATIVE_FULL_FEEDBACK_SEEDWISE.csv")
    return df[df.method.astype(str).eq("rolling_full_feedback_raw")].copy()

def locate_existing_output(ds,seed):
    likely=PROJECT/"focus_v05"/"artifacts"/"focus_after_primary"/"5c828cd0de3b"/"jobs"/f"{ds}_seed{seed}_rolling_predictor"
    if (likely/"issued_predictions.npz").exists() and (likely/"evaluation_labels.npz").exists():
        return likely
    return None

def ensure_rolling(ds,seed):
    own=ROLL/f"{ds}_seed{seed}"
    if (own/"issued_predictions.npz").exists() and (own/"evaluation_labels.npz").exists():
        return own
    ex=locate_existing_output(ds,seed)
    if ex is not None:
        return ex
    runner=prepare_focus_source()
    own.mkdir(parents=True,exist_ok=True)
    py=PROJECT/".venv"/"bin"/"python"
    if not py.exists(): py=Path(sys.executable)
    cmd=[str(py),"-u",str(runner),"--dataset",ds,"--seed",str(seed),"--output",str(own),"--threads","1","--delay","1"]
    print("FULL_FEEDBACK_REPLAY_BEGIN"," ".join(cmd),flush=True)
    subprocess.run(cmd,cwd=SOURCE,check=True)
    return own

def history_weights(window_positions,t):
    ids=[]; ws=[]
    for u in range(max(0,t-BUFFER),t):
        pos=np.asarray(window_positions[u],int)
        decay=DECAY**(t-1-u)
        ids.extend(pos.tolist()); ws.extend([decay/max(1,len(pos))]*len(pos))
    return np.asarray(ids,int),np.asarray(ws,float)

def multi_geometry(anchor,cands):
    widths=[]
    for q in cands.values():
        d=q-anchor; center=np.sum(anchor*d,axis=1); z=d-center[:,None]
        widths.append(np.max(z,axis=1)-np.min(z,axis=1))
    return np.max(np.column_stack(widths),axis=1) if widths else np.zeros(len(anchor))

def canaries(anchor,cands,m,rng):
    n=len(anchor); m=min(max(1,int(m)),max(1,n-1))
    w=multi_geometry(anchor,cands); H=min(8,m,n)
    chunks=[np.asarray(x,int) for x in np.array_split(np.argsort(w,kind="mergesort"),H) if len(x)]
    masses=np.array([len(x)*max(float(np.max(w[x])),1e-12) for x in chunks],float)
    caps=np.array([len(x) for x in chunks],int)
    # proportional integer allocation
    counts=np.zeros(len(chunks),int)
    for _ in range(m):
        score=np.where(counts<caps,masses/(counts+1),-1)
        counts[int(np.argmax(score))]+=1
    strata=[]; chosen=[]
    for h,(pop,nh) in enumerate(zip(chunks,counts)):
        s=np.sort(rng.choice(pop,size=int(nh),replace=False)); chosen+=s.tolist()
        strata.append({"population":pop,"sample":s})
    can=np.sort(np.asarray(chosen,int)); mask=np.ones(n,bool); mask[can]=False
    return strata,can,np.flatnonzero(mask)

def estimate(anchor,q,y,strata,bulk):
    d=q-anchor; center=np.sum(anchor*d,axis=1); z=d-center[:,None]
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
        out=anchor.copy(); out[bulk]=probs(anchor[bulk]+e*(q[bulk]-anchor[bulk])); out[can]=anchor[can]
        return float(fc.prospective_H(anchor,out-anchor))
    h0=h(eta)
    if h0<=eps+1e-12: return eta,h0
    lo,hi=0.,eta
    for _ in range(70):
        mid=(lo+hi)/2
        if h(mid)<=eps: lo=mid
        else: hi=mid
    return lo,h(lo)

def run_seed(ds,seed,expected):
    outdir=ensure_rolling(ds,seed)
    with np.load(outdir/"issued_predictions.npz") as z:
        row_id=np.asarray(z["row_id"],int); issued=probs(z["probabilities"])
    with np.load(outdir/"evaluation_labels.npz") as z:
        yy=np.asarray(z["y"],int)
    data=load_dataset(ds,PROJECT/"data"/"raw")
    must(np.array_equal(row_id,np.asarray(data.target,int)),"row_id mismatch")
    must(np.array_equal(yy,np.asarray(data.y[data.target],int)),"label mismatch")
    pos={int(r):i for i,r in enumerate(row_id)}
    winpos=[np.asarray([pos[int(r)] for r in ix],int) for ix in data.windows]

    raw_b=float(np.mean(brier_rows(issued,yy)))
    exp=float(expected.loc[(expected.dataset==ds)&(expected.seed==seed),"brier_sum_over_classes"].iloc[0])
    must(abs(raw_b-exp)<=2e-8,f"full-feedback replay mismatch {ds} seed={seed}: {raw_b} != {exp}")

    pred=issued.copy(); predS=issued.copy(); oracle=issued.copy(); traces=[]
    for t,posix in enumerate(winpos):
        anchor=issued[posix]; y=yy[posix]
        hist,w=history_weights(winpos,t)
        cands={}
        if len(hist):
            full=ProbabilityCalibrator().fit(issued[hist],yy[hist],w).predict_proba(anchor)
            tw=w/w.sum(); ess=float(1.0/np.sum(tw**2))
            residual=ResidualCalibrator(anchor.shape[1],ess).fit(issued[hist],yy[hist],w).predict_proba(anchor)
            cands={"fullcal":full,"residual":residual}
        else:
            cands={"fullcal":anchor.copy(),"residual":anchor.copy()}

        m=max(1,int(math.floor(CANARY_FRACTION*len(posix))))
        rng=np.random.default_rng(stable_seed("highinfo_canary",ds,seed,t))
        strata,can,bulk=canaries(anchor,cands,m,rng)

        best="raw_retrain"; best_score=0.; best_eta=0.
        ests={}
        for name,q in cands.items():
            D,C,eta,gain=estimate(anchor,q,y,strata,bulk)
            ests[name]=(D,C,eta,gain)
            if gain>best_score:
                best=name; best_score=gain; best_eta=eta

        out=anchor.copy()
        if best!="raw_retrain":
            q=cands[best]; out[bulk]=probs(anchor[bulk]+best_eta*(q[bulk]-anchor[bulk]))
        out[can]=anchor[can]; pred[posix]=out

        # Safety profile relative to the rolling-retraining anchor.
        bestS="raw_retrain"; bestSscore=0.; bestSeta=0.
        for name,q in cands.items():
            D,C,eta,gain=ests[name]
            etaS,_=clip_H(anchor,q,bulk,can,eta,EPS_SAFE)
            score=2*D*etaS-C*etaS*etaS
            if score>bestSscore:
                bestS=name; bestSscore=score; bestSeta=etaS
        outS=anchor.copy()
        if bestS!="raw_retrain":
            outS[bulk]=probs(anchor[bulk]+bestSeta*(cands[bestS][bulk]-anchor[bulk]))
        outS[can]=anchor[can]; predS[posix]=outS
        HS=float(fc.prospective_H(anchor,outS-anchor))
        must(HS<=EPS_SAFE+2e-9,"highinfo safety H violation")

        # retrospective ceiling
        bestOb=float(np.mean(brier_rows(anchor[bulk],y[bulk]))) if len(bulk) else float(np.mean(brier_rows(anchor,y)))
        outO=anchor.copy(); obest="raw_retrain"; oeta=0.
        if len(bulk):
            for name,q in cands.items():
                d=q-anchor; z=d[bulk,y[bulk]]-np.sum(anchor[bulk]*d[bulk],axis=1)
                D=float(np.mean(z)); C=float(np.mean(np.sum(d[bulk]**2,axis=1))); eta=float(np.clip(D/C,0,1)) if C>1e-18 else 0.
                cand=probs(anchor[bulk]+eta*d[bulk]); b=float(np.mean(brier_rows(cand,y[bulk])))
                if b<bestOb: bestOb=b; obest=name; oeta=eta; outO[bulk]=cand
        outO[can]=anchor[can]; oracle[posix]=outO

        tr={"dataset":ds,"seed":seed,"window":t+1,"canary_n":len(can),
            "chosen":best,"eta":best_eta,"gain_hat":best_score,
            "raw_retrain_brier":float(np.mean(brier_rows(anchor,y))),
            "utility_brier":float(np.mean(brier_rows(out,y))),
            "safety_brier":float(np.mean(brier_rows(outS,y))),
            "safety_H":HS,"oracle_brier":float(np.mean(brier_rows(outO,y))),
            "oracle_choice":obest,"oracle_eta":oeta}
        for name,(D,C,eta,gain) in ests.items():
            tr[f"{name}_Dhat"]=D; tr[f"{name}_eta"]=eta; tr[f"{name}_gain_hat"]=gain
        traces.append(tr)

    util=float(np.mean(brier_rows(pred,yy))); safe=float(np.mean(brier_rows(predS,yy))); ora=float(np.mean(brier_rows(oracle,yy)))
    return {"dataset":ds,"seed":seed,"raw_retrain_brier":raw_b,"utility_brier":util,"safety02_brier":safe,
            "oracle_brier":ora,"improvement_vs_retrain":raw_b-util,
            "beats_raw_retrain":bool(util<raw_b),"max_safety_H":max(x["safety_H"] for x in traces)}, pd.DataFrame(traces)

def main():
    OUT.mkdir(parents=True,exist_ok=True); ROLL.mkdir(parents=True,exist_ok=True)
    prepare_focus_source()
    expected=expected_table()
    rows=[]; traces=[]; total=len(DATASETS)*len(SEEDS); k=0
    with threadpool_limits(limits=1):
        for ds in DATASETS:
            for seed in SEEDS:
                k+=1
                s,tr=run_seed(ds,seed,expected); rows.append(s); traces.append(tr)
                print(f"HIGHINFO_JOB {k}/{total} dataset={ds} seed={seed} retrain={s['raw_retrain_brier']:.9f} "
                      f"utility={s['utility_brier']:.9f} safety={s['safety02_brier']:.9f} oracle={s['oracle_brier']:.9f}",flush=True)
    seedwise=pd.DataFrame(rows); trace=pd.concat(traces,ignore_index=True)
    sums=[]
    for ds,g in seedwise.groupby("dataset"):
        sums.append({"dataset":ds,"mean_raw_retrain_brier":float(g.raw_retrain_brier.mean()),
                     "mean_utility_brier":float(g.utility_brier.mean()),
                     "mean_safety02_brier":float(g.safety02_brier.mean()),
                     "mean_oracle_brier":float(g.oracle_brier.mean()),
                     "mean_improvement_vs_retrain":float(g.improvement_vs_retrain.mean()),
                     "wins_vs_retrain":int((g.improvement_vs_retrain>0).sum()),
                     "authoritative_retrain_reference":FULL_FEEDBACK_REFERENCE[ds],
                     "frozen_sparse_router_reference":FROZEN_SPARSE_ROUTER[ds],
                     "beats_full_feedback_mean":bool(g.utility_brier.mean()<FULL_FEEDBACK_REFERENCE[ds]),
                     "beats_frozen_sparse_mean":bool(g.utility_brier.mean()<FROZEN_SPARSE_ROUTER[ds]),
                     "max_safety_H":float(g.max_safety_H.max())})
    summary=pd.DataFrame(sums)
    seedwise.to_csv(OUT/"HIGHINFO_SEEDWISE.csv",index=False)
    trace.to_csv(OUT/"HIGHINFO_WINDOW_TRACE.csv",index=False)
    summary.to_csv(OUT/"HIGHINFO_SUMMARY.csv",index=False)
    assessment={
        "exact_retraining_reproduced":bool(np.all(np.abs(summary.mean_raw_retrain_brier-summary.authoritative_retrain_reference)<2e-6)),
        "gas_beats_full_feedback":bool(summary.loc[summary.dataset=="gas","beats_full_feedback_mean"].iloc[0]),
        "beats_full_feedback_all_three":bool(summary.beats_full_feedback_mean.all()),
        "safety_contract_holds":bool((summary.max_safety_H<=EPS_SAFE+2e-9).all()),
        "note":"High-information drawer is separate from the frozen 5% sparse router. Utility and H<=.02 modes are both reported."
    }
    (OUT/"HIGHINFO_ASSESSMENT.json").write_text(json.dumps(assessment,indent=2,sort_keys=True))
    review=PROJECT/"DTRAR_HighInfoRetrainRepair_Review.zip"
    with zipfile.ZipFile(review,"w",zipfile.ZIP_DEFLATED) as z:
        for p in OUT.iterdir():
            if p.is_file(): z.write(p,p.name)
        z.write(Path(__file__),Path(__file__).name)
    print("\n=== HIGH-INFORMATION SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== ASSESSMENT ===")
    print(json.dumps(assessment,indent=2,sort_keys=True))
    print(f"HIGHINFO_RETRAIN_REPAIR_COMPLETE review={review.name}",flush=True)

if __name__=="__main__":
    main()
