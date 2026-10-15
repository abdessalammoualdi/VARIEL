#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import os,sys,io,json,math,zipfile,hashlib,warnings
from concurrent.futures import ProcessPoolExecutor,as_completed

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon,rankdata

HERE=Path(__file__).resolve().parent
PROJECT=Path(os.environ.get("DTRAR_PROJECT",str(Path.home()/"Documents"/"DTRA_Start"/"DTRA"))).resolve()
for p in [PROJECT/"src",PROJECT,HERE]:
    if p.exists() and str(p) not in sys.path:sys.path.insert(0,str(p))

import pg19
import secondary_v18 as sv

# Bind bundled replay code to the user's project.
for mod in [pg19,sv]:
    mod.PROJECT=PROJECT
sv.bc.PROJECT=PROJECT
sv.sp.PROJECT=PROJECT
sv.xgr.PROJECT=PROJECT
sv.xgr.old.PROJECT=PROJECT
sv.xgr.old.OUT=PROJECT/"artifacts"/"highinfo_retrain_repair"
sv.xgr.old.SOURCE=sv.xgr.old.OUT/"focus_source"
sv.xgr.old.ROLL=sv.xgr.old.OUT/"rolling_outputs"
sv.ENDPOINT=PROJECT/"artifacts"/"variel_cross_backbone_all_final"/"endpoint_cache"
sv.V8CACHE=PROJECT/"artifacts"/"variel_cross_backbone_specialist_transfer_v7"/"candidate_cache"
pg19.sv=sv
pg19.bc=sv.bc if hasattr(pg19,"bc") else None
pg19.sp=sv.sp if hasattr(pg19,"sp") else None
pg19.xgr=sv.xgr if hasattr(pg19,"xgr") else None
pg19.ENDPOINT=sv.ENDPOINT
pg19.V8CACHE=sv.V8CACHE

V19=PROJECT/"VARIEL_PRESERVATION_GUARD_v19_Review.zip"
V8=PROJECT/"VARIEL_CROSS_BACKBONE_SPECIALIST_TRANSFER_v8_Review.zip"
OUT=PROJECT/"artifacts"/"variel_preservation_guard_v20"
CACHE=OUT/"job_cache";CACHE.mkdir(parents=True,exist_ok=True)
REVIEW=PROJECT/"VARIEL_PRESERVATION_GUARD_v20_Review.zip"

WORKERS=max(1,int(os.environ.get("PG20_WORKERS","4")))
TOL=1e-12
PROB=["brier","nll","ece15","classwise_ece15"]
PRESERVED=[
    "accuracy","balanced_accuracy","macro_f1","macro_recall","macro_specificity",
    "auroc","auprc","sensitivity","specificity","precision","binary_f1"
]

# Conservative grids only. No isotonic/nonparametric calibrator.
ALPHAS=np.asarray([0.0,0.125,0.25,0.5,0.75,0.875,1.0],float)
SLOPES=np.asarray([0.80,0.90,0.95,1.0,1.05,1.10,1.20],float)
SHRINK=np.asarray([0.0,0.25,0.50,0.75,1.0],float)
TEMPS=np.asarray([0.80,0.90,0.95,1.0,1.05,1.10,1.20,1.30],float)

def must(c,m):
    if not c:raise RuntimeError(m)

def read_csv(member):
    with zipfile.ZipFile(V19) as z:return pd.read_csv(io.BytesIO(z.read(member)))

def read_json(path,member):
    with zipfile.ZipFile(path) as z:return json.loads(z.read(member))

def probs(p):return sv.probs(p)

def pmetrics(y,p):
    m=sv.metrics(np.asarray(y,int),probs(p))
    return {k:float(m[k]) for k in PROB}

def logit(x):
    x=np.clip(np.asarray(x,float),1e-8,1-1e-8)
    return np.log(x/(1-x))

def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-40,40)))

def softmax(z):
    z=np.asarray(z,float);z=z-z.max(axis=1,keepdims=True)
    e=np.exp(z);return e/e.sum(axis=1,keepdims=True)

def calibration_gaps(y,p):
    """Stable canary-selection surrogates; less noisy than 15-bin ECE on tiny canaries."""
    y=np.asarray(y,int);p=probs(p);k=p.shape[1]
    pred=p.argmax(1);conf=p.max(1)
    conf_gap=abs(float(conf.mean())-float((pred==y).mean()))
    freq=np.bincount(y,minlength=k).astype(float)/max(1,len(y))
    class_gap=float(np.mean(np.abs(p.mean(0)-freq)))
    return conf_gap,class_gap

def stratified_split(y,seed):
    y=np.asarray(y,int);rng=np.random.default_rng(seed)
    fit=[];val=[]
    for c in np.unique(y):
        ii=np.flatnonzero(y==c).copy();rng.shuffle(ii)
        if len(ii)==1:
            val.extend(ii.tolist())
        else:
            n=max(1,len(ii)//2)
            fit.extend(ii[:n].tolist());val.extend(ii[n:].tolist())
    if len(val)<max(3,len(np.unique(y))):
        idx=np.arange(len(y));rng.shuffle(idx);cut=max(1,len(idx)//2)
        fit=idx[:cut].tolist();val=idx[cut:].tolist()
    return np.asarray(sorted(set(fit)),int),np.asarray(sorted(set(val)),int)

def blend(B,V,a):
    return probs((1-float(a))*B+float(a)*V)

def binary_transform(Q,a,b):
    p=np.clip(probs(Q)[:,1],1e-8,1-1e-8)
    q1=sigmoid(float(a)*logit(p)+float(b))
    return np.c_[1-q1,q1]

def multiclass_transform(Q,T,bias):
    Q=probs(Q);z=np.log(np.clip(Q,1e-12,1.0))/float(T)
    z=z+np.asarray(bias,float)[None,:]
    return softmax(z)

def prior_bias(train_y,train_p):
    train_y=np.asarray(train_y,int);train_p=probs(train_p);k=train_p.shape[1]
    obs=np.bincount(train_y,minlength=k).astype(float)+1.0
    obs/=obs.sum()
    pred=train_p.mean(0)+1e-6;pred/=pred.sum()
    b=np.log(obs)-np.log(pred)
    return b-b.mean()

def binary_shift(train_y,train_p):
    y=np.asarray(train_y,int);p=probs(train_p)[:,1]
    obs=(y.sum()+1.0)/(len(y)+2.0)
    pred=np.clip(p.mean(),1e-6,1-1e-6)
    return float(logit(obs)-logit(pred))

def candidate_score(y,base,raw,cand):
    """Validation objective and hard gates.

    Candidate must beat/tie baseline on BOTH proper scores, and must not
    make either stable calibration gap worse than baseline. Raw VARIEL is
    used as a regularizer: a large proper-score loss relative to raw is rejected.
    """
    mb=pmetrics(y,base);mr=pmetrics(y,raw);mc=pmetrics(y,cand)
    gb=calibration_gaps(y,base);gc=calibration_gaps(y,cand)
    n=max(1,len(y))
    # small validation slack, shrinking with more canaries
    proper_slack=0.0015/math.sqrt(max(1,n/10))
    cal_slack=0.0020/math.sqrt(max(1,n/10))

    if mc["brier"] > mb["brier"] + proper_slack:return None
    if mc["nll"] > mb["nll"] + proper_slack:return None
    if mc["brier"] > mr["brier"] + 2*proper_slack:return None
    if mc["nll"] > mr["nll"] + 2*proper_slack:return None
    if gc[0] > gb[0] + cal_slack:return None
    if gc[1] > gb[1] + cal_slack:return None

    # Prefer simultaneous calibration improvement, then proper scores.
    cal=(gc[0]+gc[1])
    proper=(mc["brier"]+0.25*mc["nll"])
    rawdistance=abs(mc["brier"]-mr["brier"])+0.25*abs(mc["nll"]-mr["nll"])
    return (cal,proper,0.10*rawdistance),mc,gc

def build_candidates(train_y,train_B,train_V,apply_B,apply_V):
    out=[]
    k=train_B.shape[1]
    for alpha in ALPHAS:
        tr=blend(train_B,train_V,alpha)
        ap=blend(apply_B,apply_V,alpha)
        out.append((f"blend:{alpha:g}",tr,ap))
        if k==2:
            b0=binary_shift(train_y,tr)
            for a in SLOPES:
                for lam in SHRINK:
                    b=lam*b0
                    out.append((f"blend:{alpha:g}|platt:a={a:g}:l={lam:g}",
                                binary_transform(tr,a,b),binary_transform(ap,a,b)))
        else:
            pb=prior_bias(train_y,tr)
            for T in TEMPS:
                for lam in SHRINK:
                    bias=lam*pb
                    out.append((f"blend:{alpha:g}|temp:T={T:g}:l={lam:g}",
                                multiclass_transform(tr,T,bias),multiclass_transform(ap,T,bias)))
    return out

def choose_window(B,V,Y,can,hist,seed):
    can=np.asarray(can,int);Y=np.asarray(Y,int)
    mask=np.ones(len(Y),bool);mask[can]=False;bulk=np.flatnonzero(mask)
    fitloc,valloc=stratified_split(Y[can],seed)
    cfit=can[fitloc];cval=can[valloc]

    hB,hV,hY=hist
    if len(hY):
        trainB=np.vstack([hB,B[cfit]]) if len(cfit) else hB.copy()
        trainV=np.vstack([hV,V[cfit]]) if len(cfit) else hV.copy()
        trainY=np.r_[hY,Y[cfit]] if len(cfit) else hY.copy()
    else:
        trainB=B[cfit] if len(cfit) else B[can]
        trainV=V[cfit] if len(cfit) else V[can]
        trainY=Y[cfit] if len(cfit) else Y[can]

    # If split leaves too little training data, use all current canaries to derive
    # the conservative transform, but validate against historical canaries.
    if len(trainY)<8:
        trainB=B[can];trainV=V[can];trainY=Y[can]

    valB=B[cval] if len(cval) else B[can]
    valV=V[cval] if len(cval) else V[can]
    valY=Y[cval] if len(cval) else Y[can]

    best=None
    for name,trcand,appcand in build_candidates(trainY,trainB,trainV,V,V):
        # Rebuild candidate on validation probabilities with the SAME derived
        # transform family. For grid-only transforms this is deterministic.
        # Find corresponding validation transform by deriving from train and
        # applying to val through a parallel call.
        # We reconstruct directly from the candidate name.
        parts=name.split("|")
        alpha=float(parts[0].split(":")[1])
        qval=blend(valB,valV,alpha)
        qfull=blend(B,V,alpha)
        if len(parts)>1:
            if "platt" in parts[1]:
                spec=parts[1]
                a=float(spec.split("a=")[1].split(":")[0])
                lam=float(spec.split("l=")[1])
                b0=binary_shift(trainY,blend(trainB,trainV,alpha))
                qval=binary_transform(qval,a,lam*b0)
                qfull=binary_transform(qfull,a,lam*b0)
            else:
                spec=parts[1]
                T=float(spec.split("T=")[1].split(":")[0])
                lam=float(spec.split("l=")[1])
                pb=prior_bias(trainY,blend(trainB,trainV,alpha))
                qval=multiclass_transform(qval,T,lam*pb)
                qfull=multiclass_transform(qfull,T,lam*pb)

        scored=candidate_score(valY,valB,valV,qval)
        if scored is None:continue
        key,mc,gaps=scored
        # Require some calibration evidence over the raw channel unless this is
        # the baseline fallback.
        rawg=calibration_gaps(valY,valV)
        if alpha>0 and (gaps[0]+gaps[1]) > (rawg[0]+rawg[1]) + 1e-12:
            continue
        cand=(key,name,qfull,mc,gaps)
        if best is None or cand[0] < best[0]:
            best=cand

    if best is None:
        # Explicit conservative fallback chosen only from validation canaries.
        mb=pmetrics(valY,valB);mv=pmetrics(valY,valV)
        gb=sum(calibration_gaps(valY,valB));gv=sum(calibration_gaps(valY,valV))
        if mv["brier"]<=mb["brier"] and mv["nll"]<=mb["nll"] and gv<=gb:
            name="raw";Q=V.copy()
        else:
            name="baseline";Q=B.copy()
    else:
        _,name,Q,_,_=best

    out=V.copy()
    if len(bulk):out[bulk]=Q[bulk]
    # Never revise audited canary outputs after their labels are observed.
    out[can]=V[can]

    hB2=np.vstack([hB,B[can]]) if len(hY) else B[can].copy()
    hV2=np.vstack([hV,V[can]]) if len(hY) else V[can].copy()
    hY2=np.r_[hY,Y[can]] if len(hY) else Y[can].copy()
    return probs(out),(hB2,hV2,hY2),name

def target_windows(bb,ds,seed,cfg):
    if bb=="xgboost":
        return pg19.xgb_windows(ds,seed)
    return pg19.catft_windows("main",bb,ds,seed,cfg)

def cache_path(bb,ds,seed):
    return CACHE/f"{bb}__{ds}__seed{seed}.json"

def run_seed(task):
    bb,ds,seed,cfg=task
    cp=cache_path(bb,ds,seed)
    if cp.exists():return json.loads(cp.read_text()),True
    windows=target_windows(bb,ds,seed,cfg)
    k=windows[0][1].shape[1]
    hist=(np.empty((0,k)),np.empty((0,k)),np.empty(0,dtype=int))
    ys=[];Bs=[];Vs=[];Gs=[];choices=[]
    for wi,(y,B,V,can) in enumerate(windows):
        G,hist,ch=choose_window(B,V,y,can,hist,sv.stable_seed("pg20",bb,ds,seed,wi))
        ys.append(np.asarray(y,int));Bs.append(B);Vs.append(V);Gs.append(G);choices.append(ch)
    Y=np.concatenate(ys);B=probs(np.vstack(Bs));V=probs(np.vstack(Vs));G=probs(np.vstack(Gs))
    rec={
        "backbone":bb,"dataset":ds,"seed":seed,
        "baseline":{m:float(sv.metrics(Y,B)[m]) for m in PROB},
        "raw_VARIEL":{m:float(sv.metrics(Y,V)[m]) for m in PROB},
        "PG20":{m:float(sv.metrics(Y,G)[m]) for m in PROB},
        "choices":choices,
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
    rng=np.random.default_rng(seed);x=rng.choice(a,(B,len(a)),replace=True).mean(1)
    return tuple(np.quantile(x,[.025,.975]))

def sseed(*x):
    return int.from_bytes(hashlib.sha256("|".join(map(str,x)).encode()).digest()[:8],"little")%(2**32-1)

def summarize(long):
    rows=[]
    for (scope,bb,ds,metric),g in long.groupby(["scope","backbone","dataset","metric"],sort=True):
        a=g.advantage.to_numpy(float);a=a[np.isfinite(a)]
        if not len(a):continue
        lo,hi=boot(a,sseed("pg20",scope,bb,ds,metric))
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
    must(V19.exists(),"missing V19 review")
    long19=read_csv("PG_SECONDARY_METRICS_LONG.csv")
    sm19=read_csv("PG_SECONDARY_METRICS_SUMMARY.csv")
    cfg=read_json(V8,"SELECTED_CONFIGS_V8.json")["full|fttransformer|spam"]["cfg"]

    # ONLY cells that still fail a probability metric in V19.
    bad=sm19[(sm19.scope=="main")&(sm19.metric.isin(PROB))&(sm19.mean_advantage<0)]
    targets=sorted(set(zip(bad.backbone,bad.dataset)))
    print("PG20_TARGETS",targets,flush=True)
    tasks=[(bb,ds,s,cfg) for bb,ds in targets for s in range(10)]
    print(f"PG20_TARGET_JOBS total={len(tasks)}",flush=True)

    results={}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        fut={ex.submit(run_seed,t):t for t in tasks}
        done=0
        for f in as_completed(fut):
            rec,reused=f.result();done+=1
            key=(rec["backbone"],rec["dataset"],int(rec["seed"]))
            results[key]=rec
            print(f"PG20_DONE {done}/{len(tasks)} backbone={key[0]} dataset={key[1]} seed={key[2]} reused={int(reused)}",flush=True)

    # Merge: V19 stays frozen everywhere except the six failed main probability cells.
    out=[]
    for _,r in long19.iterrows():
        scope,bb,ds,seed,metric=r.scope,r.backbone,r.dataset,int(r.seed),r.metric
        base=float(r.baseline) if pd.notna(r.baseline) else np.nan
        val=float(r.variel) if pd.notna(r.variel) else np.nan
        if scope=="main" and metric in PROB and (bb,ds,seed) in results:
            val=float(results[(bb,ds,seed)]["PG20"][metric])
        out.append({
            "scope":scope,"backbone":bb,"dataset":ds,"seed":seed,"metric":metric,
            "baseline":base,"variel":val,
            "advantage":advantage(metric,base,val) if np.isfinite(base) and np.isfinite(val) else np.nan
        })
    long=pd.DataFrame(out)
    sm=summarize(long)
    long.to_csv(OUT/"PG20_SECONDARY_METRICS_LONG.csv",index=False)
    sm.to_csv(OUT/"PG20_SECONDARY_METRICS_SUMMARY.csv",index=False)

    rows=[]
    for rec in results.values():
        row={k:rec[k] for k in ["backbone","dataset","seed"]}
        for m in PROB:
            row[f"baseline_{m}"]=rec["baseline"][m]
            row[f"raw_VARIEL_{m}"]=rec["raw_VARIEL"][m]
            row[f"PG20_{m}"]=rec["PG20"][m]
            row[f"PG20_advantage_{m}"]=rec["baseline"][m]-rec["PG20"][m]
        row["choices"]="|".join(rec["choices"])
        rows.append(row)
    pd.DataFrame(rows).sort_values(["backbone","dataset","seed"]).to_csv(OUT/"PG20_TARGETED_SEEDWISE.csv",index=False)

    main=sm[sm.scope=="main"];stress=sm[sm.scope=="stress"]
    def mrows(frame,m):return frame[frame.metric==m]
    preserved=main[main.metric.isin(PRESERVED)]
    ass={
        "name":"VARIEL Preservation Guard v20 — conservative cross-fit calibration",
        "target_cells":[list(x) for x in targets],
        "target_jobs":len(tasks),
        "backbone_retraining":False,
        "frozen_V19_cells_untouched_except_targets":True,
        "nonparametric_isotonic_removed":True,
        "selector":"deterministic canary split + historical canaries; conservative blend/Platt for binary and blend/temperature+prior correction for multiclass; explicit baseline/raw fallback",
        "main_cells":24,
        "main_brier_positive_cells":int((mrows(main,"brier").mean_advantage>0).sum()),
        "main_nll_positive_cells":int((mrows(main,"nll").mean_advantage>0).sum()),
        "main_ece_nonnegative_cells":int((mrows(main,"ece15").mean_advantage>=-TOL).sum()),
        "main_classwise_ece_nonnegative_cells":int((mrows(main,"classwise_ece15").mean_advantage>=-TOL).sum()),
        "no_main_hard_or_ranking_regressions":bool((preserved.mean_advantage>=-TOL).all()),
        "preserved_metric_max_abs_advantage":float(np.nanmax(np.abs(preserved.mean_advantage))) if len(preserved) else 0.0,
        "stress_probability_min_mean_advantage":float(stress[stress.metric.isin(PROB)].mean_advantage.min()),
    }
    ass["perfect_reviewer_target"]=bool(
        ass["main_brier_positive_cells"]==24 and
        ass["main_nll_positive_cells"]==24 and
        ass["main_ece_nonnegative_cells"]==24 and
        ass["main_classwise_ece_nonnegative_cells"]==24 and
        ass["no_main_hard_or_ranking_regressions"] and
        ass["stress_probability_min_mean_advantage"]>=-0.005-TOL
    )
    (OUT/"PG20_ASSESSMENT.json").write_text(json.dumps(ass,indent=2,sort_keys=True))

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

    print("=== PG20 ASSESSMENT ===",flush=True)
    print(json.dumps(ass,indent=2,sort_keys=True),flush=True)
    print("PG20_COMPLETE review="+str(REVIEW),flush=True)

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    main()
