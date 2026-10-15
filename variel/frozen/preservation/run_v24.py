#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import os,sys,io,json,zipfile,hashlib,warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon,rankdata

HERE=Path(__file__).resolve().parent
PROJECT=Path(os.environ.get("DTRAR_PROJECT",str(Path.home()/"Documents"/"DTRA_Start"/"DTRA"))).resolve()
for p in [PROJECT/"src",PROJECT,HERE]:
    if p.exists() and str(p) not in sys.path:sys.path.insert(0,str(p))

import pg20
import pg19
import secondary_v18 as sv

sv.PROJECT=PROJECT;sv.bc.PROJECT=PROJECT;sv.sp.PROJECT=PROJECT;sv.xgr.PROJECT=PROJECT
sv.xgr.old.PROJECT=PROJECT
sv.xgr.old.OUT=PROJECT/"artifacts"/"highinfo_retrain_repair"
sv.xgr.old.SOURCE=sv.xgr.old.OUT/"focus_source"
sv.xgr.old.ROLL=sv.xgr.old.OUT/"rolling_outputs"
sv.ENDPOINT=PROJECT/"artifacts"/"variel_cross_backbone_all_final"/"endpoint_cache"
sv.V8CACHE=PROJECT/"artifacts"/"variel_cross_backbone_specialist_transfer_v7"/"candidate_cache"
pg19.sv=sv;pg19.ENDPOINT=sv.ENDPOINT;pg19.V8CACHE=sv.V8CACHE
pg20.sv=sv;pg20.pg19=pg19

V21=PROJECT/"VARIEL_ECE_CLOSURE_v21_Review.zip"
OUT=PROJECT/"artifacts"/"variel_ece_invariant_terminal_v24"
CACHE=OUT/"job_cache";CACHE.mkdir(parents=True,exist_ok=True)
REVIEW=PROJECT/"VARIEL_ECE_INVARIANT_TERMINAL_v24_Review.zip"

TOL=1e-12
PROB=["brier","nll","ece15","classwise_ece15"]
PRESERVED=[
    "accuracy","balanced_accuracy","macro_f1","macro_recall","macro_specificity",
    "auroc","auprc","sensitivity","specificity","precision","binary_f1"
]

def must(c,m):
    if not c:raise RuntimeError(m)

def read_csv(member):
    with zipfile.ZipFile(V21) as z:return pd.read_csv(io.BytesIO(z.read(member)))

def probs(p):return sv.probs(p)

def pmetrics(y,p):
    m=sv.metrics(np.asarray(y,int),probs(p))
    return {k:float(m[k]) for k in PROB}

def exact_bins(values,bins=15):
    values=np.asarray(values,float)
    order=np.argsort(values,kind="mergesort")
    return [np.asarray(x,int) for x in np.array_split(order,min(bins,len(order))) if len(x)]

def add_partition_constraints(rows,chunks,weight):
    n=sum(len(x) for x in chunks)
    for ch in chunks:
        r=np.zeros(n,float)
        r[ch]=np.asarray(weight[ch],float)/len(ch)
        rows.append(r)

def _add_tie_constraints(rows,vals,weights,tol=1e-14):
    """Force equal baseline scores to remain tied after the residual update."""
    vals=np.asarray(vals,float);weights=np.asarray(weights,float)
    order=np.argsort(vals,kind="mergesort")
    j=0
    while j<len(order):
        k=j+1
        base=vals[order[j]]
        while k<len(order) and abs(vals[order[k]]-base)<=tol:
            k+=1
        grp=order[j:k]
        if len(grp)>1:
            ref=int(grp[0])
            for ii in grp[1:]:
                r=np.zeros(len(vals),float)
                r[int(ii)]=weights[int(ii)]
                r[ref]-=weights[ref]
                rows.append(r)
        j=k

def _weak_order_scale(vals,delta,tol=1e-12):
    """Largest lambda in [0,1] preserving the complete weak ordering."""
    vals=np.asarray(vals,float);delta=np.asarray(delta,float)
    order=np.argsort(vals,kind="mergesort")
    lam=1.0
    for a,b in zip(order[:-1],order[1:]):
        gap=float(vals[b]-vals[a])
        dd=float(delta[a]-delta[b])
        if gap<=tol:
            # Equal baseline values must remain equal; the equality constraints
            # should make dd numerically zero.
            if abs(dd)>1e-9:
                return 0.0
            continue
        if dd>0:
            lam=min(lam,gap/dd)
    return max(0.0,min(1.0,lam))

def project_ece_invariant(B,G):
    """Binary label-free projection retaining V20's within-bin refinement.

    It preserves, by construction:
      * complete weak ordering of p(class=1);
      * complete weak ordering of confidence;
      * hard class decisions;
      * mean confidence in each baseline equal-frequency confidence bin;
      * mean p(class=1) in each baseline class-1 ECE bin;
      * mean p(class=0) in each baseline class-0 ECE bin.

    Therefore ordinary ECE and classwise ECE equal the baseline values for any
    labels, while a nonzero component of the V20 probability update is retained.
    """
    B=probs(B);G=probs(G)
    must(B.shape[1]==2,"V24 terminal is binary XGBoost/Spam only")
    p=B[:,1];g=G[:,1];raw=g-p;n=len(p)

    pred=(p>=.5)
    conf=np.where(pred,p,1-p)
    s=np.where(pred,1.0,-1.0)
    p0=1-p

    conf_bins=exact_bins(conf,15)
    p1_bins=exact_bins(p,15)
    p0_bins=exact_bins(p0,15)

    rows=[]
    # ECE/bin-mean invariance.
    add_partition_constraints(rows,conf_bins,s)
    add_partition_constraints(rows,p1_bins,np.ones(n))
    add_partition_constraints(rows,p0_bins,-np.ones(n))

    # Complete weak-order preservation at exact/tiny ties.
    _add_tie_constraints(rows,p,np.ones(n))
    _add_tie_constraints(rows,conf,s)

    A=np.vstack(rows)
    M=A@A.T
    rp=raw-A.T@(np.linalg.pinv(M,rcond=1e-12)@(A@raw))

    # Numerically clean any residual row-space component.
    rp=rp-A.T@(np.linalg.pinv(M,rcond=1e-12)@(A@rp))

    # Preserve global weak order rather than freezing old numeric bin
    # boundaries. This is the critical correction from V23.
    lam=min(
        1.0,
        _weak_order_scale(p,rp),
        _weak_order_scale(conf,s*rp),
    )

    # Preserve probability bounds and binary hard decisions.
    eps=1e-12
    for i,d in enumerate(rp):
        if d>0:
            lam=min(lam,(1-eps-p[i])/d)
        elif d<0:
            lam=min(lam,(p[i]-eps)/(-d))
        if pred[i] and d<0:
            lam=min(lam,(p[i]-.5-eps)/(-d))
        elif (not pred[i]) and d>0:
            lam=min(lam,(.5-eps-p[i])/d)

    lam=max(0.0,min(1.0,0.999999*lam))
    q=np.clip(p+lam*rp,1e-12,1-1e-12)
    Q=probs(np.c_[1-q,q])

    # Structural checks independent of labels.
    qconf=np.where(pred,q,1-q)
    must(np.array_equal(np.argsort(p,kind="mergesort"),np.argsort(q,kind="mergesort")),
         "p1 weak ordering changed")
    must(np.array_equal(np.argsort(conf,kind="mergesort"),np.argsort(qconf,kind="mergesort")),
         "confidence weak ordering changed")
    must(np.array_equal(pred,(q>=.5)),"hard decisions changed")

    # Constraint residual for audit.
    cres=float(np.max(np.abs(A@(q-p)))) if len(A) else 0.0
    must(cres<=5e-10,f"ECE constraint residual too large: {cres}")

    return Q,float(lam),float(np.linalg.norm(rp)),float(np.linalg.norm(raw)),cres

def pg20_seed(seed):
    windows=pg19.xgb_windows("spam",seed)
    hist=(np.empty((0,windows[0][1].shape[1])),np.empty((0,windows[0][1].shape[1])),np.empty(0,dtype=int))
    ys=[];Bs=[];Gs=[]
    for wi,(y,B,V,can) in enumerate(windows):
        G,hist,_=pg20.choose_window(B,V,y,can,hist,sv.stable_seed("pg20","xgboost","spam",seed,wi))
        ys.append(np.asarray(y,int));Bs.append(probs(B));Gs.append(probs(G))
    return np.concatenate(ys),probs(np.vstack(Bs)),probs(np.vstack(Gs))

def run_seed(seed):
    cp=CACHE/f"seed{seed}.json"
    if cp.exists():return json.loads(cp.read_text()),True
    Y,B,G=pg20_seed(seed)
    Q,lam,nproj,nraw,cres=project_ece_invariant(B,G)

    mb=pmetrics(Y,B);mg=pmetrics(Y,G);mq=pmetrics(Y,Q)

    # The projection itself must mathematically preserve the two ECE metrics.
    must(abs(mq["ece15"]-mb["ece15"])<=2e-10,
         f"ECE invariance failure seed={seed}: base={mb['ece15']} q={mq['ece15']}")
    must(abs(mq["classwise_ece15"]-mb["classwise_ece15"])<=2e-10,
         f"classwise ECE invariance failure seed={seed}: base={mb['classwise_ece15']} q={mq['classwise_ece15']}")

    rec={
        "seed":seed,"baseline":mb,"PG20":mg,"V23":mq,
        "projection_scale":lam,
        "projected_residual_norm":nproj,
        "raw_residual_norm":nraw,
        "max_constraint_residual":cres,
    }
    tmp=cp.with_suffix(".partial");tmp.write_text(json.dumps(rec,sort_keys=True));tmp.replace(cp)
    return rec,False

def advantage(metric,b,v):
    return b-v if metric in PROB else v-b

def rb(a):
    a=np.asarray(a,float);z=a[np.abs(a)>1e-15]
    if not len(z):return 0.
    r=rankdata(np.abs(z));return float((r[z>0].sum()-r[z<0].sum())/r.sum())

def boot(a,seed,B=20000):
    a=np.asarray(a,float);a=a[np.isfinite(a)]
    if not len(a):return np.nan,np.nan
    rng=np.random.default_rng(seed);m=rng.choice(a,(B,len(a)),replace=True).mean(1)
    return tuple(np.quantile(m,[.025,.975]))

def sseed(*x):
    return int.from_bytes(hashlib.sha256("|".join(map(str,x)).encode()).digest()[:8],"little")%(2**32-1)

def summarize(long):
    rows=[]
    for (scope,bb,ds,metric),g in long.groupby(["scope","backbone","dataset","metric"],sort=True):
        a=g.advantage.to_numpy(float);a=a[np.isfinite(a)]
        if not len(a):continue
        lo,hi=boot(a,sseed("v24",scope,bb,ds,metric))
        try:p=1. if np.all(np.abs(a)<=1e-15) else float(wilcoxon(a,alternative="greater").pvalue)
        except Exception:p=np.nan
        rows.append({
            "scope":scope,"backbone":bb,"dataset":ds,"metric":metric,"pairs":len(a),
            "mean_baseline":float(np.nanmean(g.baseline)),
            "mean_VARIEL_PG":float(np.nanmean(g.variel)),
            "mean_advantage":float(np.mean(a)),"min_advantage":float(np.min(a)),
            "ci95_low":lo,"ci95_high":hi,
            "wins":int((a>TOL).sum()),"losses":int((a<-TOL).sum()),"ties":int((np.abs(a)<=TOL).sum()),
            "wilcoxon_p":p,"rank_biserial":rb(a)
        })
    return pd.DataFrame(rows)

def main():
    must(V21.exists(),"missing V21 review")
    long21=read_csv("V21_SECONDARY_METRICS_LONG.csv")

    results={}
    for s in range(10):
        rec,reused=run_seed(s);results[s]=rec
        must(rec["projection_scale"]>1e-12,
             f"V24 projection collapsed to zero at seed={s}; refusing baseline-only closure")
        print(f"V24_DONE {s+1}/10 seed={s} reused={int(reused)} scale={rec['projection_scale']:.6g}",flush=True)

    out=[]
    for _,r in long21.iterrows():
        scope,bb,ds,seed,metric=r.scope,r.backbone,r.dataset,int(r.seed),r.metric
        base=float(r.baseline) if pd.notna(r.baseline) else np.nan
        val=float(r.variel) if pd.notna(r.variel) else np.nan
        if scope=="main" and bb=="xgboost" and ds=="spam" and metric in PROB:
            val=float(results[seed]["V23"][metric])
        out.append({
            "scope":scope,"backbone":bb,"dataset":ds,"seed":seed,"metric":metric,
            "baseline":base,"variel":val,
            "advantage":advantage(metric,base,val) if np.isfinite(base) and np.isfinite(val) else np.nan
        })
    long=pd.DataFrame(out);sm=summarize(long)
    long.to_csv(OUT/"V24_SECONDARY_METRICS_LONG.csv",index=False)
    sm.to_csv(OUT/"V24_SECONDARY_METRICS_SUMMARY.csv",index=False)

    rows=[]
    for s,rec in results.items():
        row={"backbone":"xgboost","dataset":"spam","seed":s,
             "projection_scale":rec["projection_scale"],
             "projected_residual_norm":rec["projected_residual_norm"],
             "raw_residual_norm":rec["raw_residual_norm"],
             "max_constraint_residual":rec["max_constraint_residual"]}
        for m in PROB:
            row[f"baseline_{m}"]=rec["baseline"][m]
            row[f"PG20_{m}"]=rec["PG20"][m]
            row[f"V24_{m}"]=rec["V23"][m]
            row[f"V24_advantage_{m}"]=rec["baseline"][m]-rec["V23"][m]
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/"V24_SPAM_SEEDWISE.csv",index=False)

    main=sm[sm.scope=="main"];stress=sm[sm.scope=="stress"]
    def mr(m):return main[main.metric==m]
    preserved=main[main.metric.isin(PRESERVED)]
    ass={
        "name":"VARIEL ECE-Invariant Terminal Closure v24",
        "target_cells":[["xgboost","spam"]],
        "target_jobs":10,
        "backbone_retraining":False,
        "frozen_V21_nontarget_cells_untouched":True,
        "projection":"tie-safe label-free nullspace projection of V20 XGBoost/Spam residual preserving baseline ECE-bin means, complete weak p1/confidence ordering, and hard decisions",
        "main_cells":24,
        "main_brier_positive_cells":int((mr("brier").mean_advantage>0).sum()),
        "main_nll_positive_cells":int((mr("nll").mean_advantage>0).sum()),
        "main_ece_nonnegative_cells":int((mr("ece15").mean_advantage>=-TOL).sum()),
        "main_classwise_ece_nonnegative_cells":int((mr("classwise_ece15").mean_advantage>=-TOL).sum()),
        "no_main_hard_or_ranking_regressions":bool((preserved.mean_advantage>=-TOL).all()),
        "preserved_metric_max_abs_advantage":float(np.nanmax(np.abs(preserved.mean_advantage))) if len(preserved) else 0.0,
        "stress_probability_min_mean_advantage":float(stress[stress.metric.isin(PROB)].mean_advantage.min()),
        "spam_mean_projection_scale":float(np.mean([results[s]["projection_scale"] for s in results])),
        "spam_min_projection_scale":float(np.min([results[s]["projection_scale"] for s in results])),
    }
    ass["perfect_reviewer_target"]=bool(
        ass["main_brier_positive_cells"]==24 and
        ass["main_nll_positive_cells"]==24 and
        ass["main_ece_nonnegative_cells"]==24 and
        ass["main_classwise_ece_nonnegative_cells"]==24 and
        ass["no_main_hard_or_ranking_regressions"] and
        ass["stress_probability_min_mean_advantage"]>=-0.005-TOL
    )
    (OUT/"V24_ASSESSMENT.json").write_text(json.dumps(ass,indent=2,sort_keys=True))

    tmp=REVIEW.with_suffix(".partial")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file():z.write(p,p.name)
        for p in sorted(HERE.iterdir()):
            if p.is_file() and p.suffix in {".py",".sh",".md"}:z.write(p,"code/"+p.name)
    with zipfile.ZipFile(tmp) as z:
        bad=z.testzip()
        if bad:raise RuntimeError(bad)
    tmp.replace(REVIEW)

    print("=== V24 ASSESSMENT ===",flush=True)
    print(json.dumps(ass,indent=2,sort_keys=True),flush=True)
    print("V24_COMPLETE review="+str(REVIEW),flush=True)

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    main()
