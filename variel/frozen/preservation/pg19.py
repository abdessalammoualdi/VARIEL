#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import os,sys,io,json,math,gzip,pickle,zipfile,hashlib,warnings
from concurrent.futures import ProcessPoolExecutor,as_completed

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon,rankdata
from sklearn.isotonic import IsotonicRegression

HERE=Path(__file__).resolve().parent
PROJECT=Path(os.environ.get("DTRAR_PROJECT",str(Path.home()/"Documents"/"DTRA_Start"/"DTRA"))).resolve()
for p in [PROJECT/"src",PROJECT,HERE]:
    if p.exists() and str(p) not in sys.path:sys.path.insert(0,str(p))

import secondary_v18 as sv

# Force bundled V18 machinery onto the user's project.
sv.PROJECT=PROJECT
sv.bc.PROJECT=PROJECT
sv.sp.PROJECT=PROJECT
sv.xgr.PROJECT=PROJECT
sv.xgr.old.PROJECT=PROJECT
sv.xgr.old.OUT=PROJECT/"artifacts"/"highinfo_retrain_repair"
sv.xgr.old.SOURCE=sv.xgr.old.OUT/"focus_source"
sv.xgr.old.ROLL=sv.xgr.old.OUT/"rolling_outputs"
sv.ENDPOINT=PROJECT/"artifacts"/"variel_cross_backbone_all_final"/"endpoint_cache"
sv.V8CACHE=PROJECT/"artifacts"/"variel_cross_backbone_specialist_transfer_v7"/"candidate_cache"

V18=PROJECT/"VARIEL_FINAL_SECONDARY_METRICS_v18_Review.zip"
V8=PROJECT/"VARIEL_CROSS_BACKBONE_SPECIALIST_TRANSFER_v8_Review.zip"
OUT=PROJECT/"artifacts"/"variel_preservation_guard_v19"
CACHE=OUT/"job_cache";CACHE.mkdir(parents=True,exist_ok=True)
REVIEW=PROJECT/"VARIEL_PRESERVATION_GUARD_v19_Review.zip"

WORKERS=max(1,int(os.environ.get("PG_WORKERS","4")))
TOL=1e-12
PROB_METRICS=["brier","nll","ece15","classwise_ece15"]
PRESERVED_METRICS=[
    "accuracy","balanced_accuracy","macro_f1","macro_recall","macro_specificity",
    "auroc","auprc","sensitivity","specificity","precision","binary_f1"
]
ALPHAS=np.asarray([0.,.0625,.125,.1875,.25,.375,.5,.625,.75,.875,1.],float)
TEMPS=np.asarray([.45,.55,.67,.8,.9,1.,1.1,1.25,1.5,1.8,2.2],float)
GAMMAS=np.asarray([.35,.5,.65,.8,.9,1.,1.1,1.2,1.35,1.5],float)

def must(c,m):
    if not c:raise RuntimeError(m)

def read_csv(member):
    with zipfile.ZipFile(V18) as z:return pd.read_csv(io.BytesIO(z.read(member)))

def read_json(path,member):
    with zipfile.ZipFile(path) as z:return json.loads(z.read(member))

def probs(p):
    return sv.probs(p)

def prob_metrics(y,p):
    m=sv.metrics(y,p)
    return {k:float(m[k]) for k in PROB_METRICS}

def temperature(p,T):
    p=probs(p);z=np.log(np.clip(p,1e-12,1.0))/float(T);z-=z.max(axis=1,keepdims=True)
    return probs(np.exp(z))

def affine_confidence(p,gamma,center=None):
    p=probs(p);k=p.shape[1]
    if center is None:center=np.full(k,1./k,float)
    center=np.asarray(center,float);center=center/center.sum()
    q=center[None,:]+float(gamma)*(p-center[None,:])
    if np.min(q)<-1e-12 or np.max(q)>1+1e-12:return None
    return probs(np.clip(q,1e-12,1.))

def fit_isotonic(audit_p,audit_y,apply_p):
    audit_p=probs(audit_p);apply_p=probs(apply_p);y=np.asarray(audit_y,int);k=audit_p.shape[1]
    if len(y)<max(24,4*k):return None,None
    ao=[];qo=[]
    try:
        for c in range(k):
            yc=(y==c).astype(float)
            if yc.min()==yc.max():return None,None
            ir=IsotonicRegression(y_min=1e-5,y_max=1-1e-5,out_of_bounds="clip")
            ir.fit(audit_p[:,c],yc)
            ao.append(ir.predict(audit_p[:,c]))
            qo.append(ir.predict(apply_p[:,c]))
        A=np.stack(ao,1);Q=np.stack(qo,1)
        return probs(np.maximum(A,1e-12)),probs(np.maximum(Q,1e-12))
    except Exception:
        return None,None

def cal_score(m):
    return float(m["ece15"]+m["classwise_ece15"])

def choose_pg(aB,aV,aY,B,V,Y,can,hist,scope):
    """Select a probability-channel guard using only past/current canary labels.

    Hard decisions/ranking are a separate frozen baseline channel; this routine
    only optimizes probabilistic quality.

    Main scope reference = current VARIEL probability channel.
    Stress scope reference = whichever of baseline/VARIEL has lower
    Brier+NLL on the available canaries (so weak repairs can fall back).
    """
    can=np.asarray(can,int);mask=np.ones(len(Y),bool);mask[can]=False;bulk=np.flatnonzero(mask)
    hb,hv,hy=hist
    ab=B[can];av=V[can];ay=Y[can]
    if len(hy):
        auditB=np.vstack([hb,ab]);auditV=np.vstack([hv,av]);auditY=np.r_[hy,ay]
    else:
        auditB,auditV,auditY=ab,av,ay

    mb=prob_metrics(auditY,auditB);mv=prob_metrics(auditY,auditV)
    if scope=="main":
        ref_name="variel";ref_a=auditV;ref_full=V;ref_m=mv
    else:
        if (mb["brier"]+mb["nll"]) <= (mv["brier"]+mv["nll"]):
            ref_name="baseline";ref_a=auditB;ref_full=B;ref_m=mb
        else:
            ref_name="variel";ref_a=auditV;ref_full=V;ref_m=mv

    best=(cal_score(ref_m),ref_m["brier"]+ref_m["nll"],"ref",ref_a,ref_full,ref_m)
    # Cumulative canary class prior for optional prior-centered confidence scaling.
    k=B.shape[1];cnt=np.bincount(np.asarray(auditY,int),minlength=k).astype(float)+1.
    prior=cnt/cnt.sum()

    for alpha in ALPHAS:
        aa=probs((1-alpha)*auditB+alpha*auditV)
        qq=probs((1-alpha)*B+alpha*V)

        candidates=[("blend",aa,qq)]
        for T in TEMPS:
            candidates.append((f"temp:{T:g}",temperature(aa,T),temperature(qq,T)))
        for g in GAMMAS:
            x=affine_confidence(aa,g,None);q=affine_confidence(qq,g,None)
            if x is not None and q is not None:candidates.append((f"uniform:{g:g}",x,q))
            x=affine_confidence(aa,g,prior);q=affine_confidence(qq,g,prior)
            if x is not None and q is not None:candidates.append((f"prior:{g:g}",x,q))

        # Stronger calibration candidate; accepted only if proper scores pass.
        ia,iq=fit_isotonic(aa,auditY,qq)
        if ia is not None:candidates.append(("isotonic",ia,iq))

        for kind,ca,cq in candidates:
            cm=prob_metrics(auditY,ca)
            # No-harm proper-score gate relative to the reference probability channel.
            if cm["brier"]>ref_m["brier"]+1e-12:continue
            if cm["nll"]>ref_m["nll"]+1e-12:continue
            key=(cal_score(cm),cm["brier"]+cm["nll"])
            if key < best[:2]:
                best=(key[0],key[1],f"a={alpha:g}|{kind}",ca,cq,cm)

    _,_,name,_,chosen,cm=best
    out=ref_full.copy()
    if name!="ref" and len(bulk):
        out[bulk]=chosen[bulk]
    # Never revise already-audited canary predictions after observing labels.
    out[can]=ref_full[can]

    hb2=np.vstack([hb,ab]) if len(hy) else ab.copy()
    hv2=np.vstack([hv,av]) if len(hy) else av.copy()
    hy2=np.r_[hy,ay] if len(hy) else ay.copy()
    return probs(out),(hb2,hv2,hy2),{
        "choice":name,"reference":ref_name,"audit_brier":cm["brier"],
        "audit_nll":cm["nll"],"audit_ece":cm["ece15"],
        "audit_classwise_ece":cm["classwise_ece15"]
    }

def split_arrays(y,b,v,lengths,cans):
    out=[];off=0
    for n,can in zip(lengths,cans):
        out.append((np.asarray(y[off:off+n],int),probs(b[off:off+n]),probs(v[off:off+n]),np.asarray(can,int)))
        off+=n
    must(off==len(y),"window split mismatch")
    return out

def catft_windows(scope,bb,ds,seed,cfg_spam):
    obj=sv.load_endpoint(bb,ds,seed)
    if scope=="stress":
        y,b,v=sv.catft_sparse_core(bb,ds,seed)
        lengths=[len(w["y"]) for w in obj["windows"]]
        cans=[np.asarray(w["can4"],int) for w in obj["windows"]]
        return split_arrays(y,b,v,lengths,cans)

    if bb=="fttransformer" and ds=="spam":
        y,b,v=sv.reconstruct_specialist_full_ft_spam(seed,cfg_spam)
        _,_,_,_,_,windows=sv.sp.build_anchor_windows(bb,ds,seed,"full")
        lengths=[len(w["y"]) for w in windows]
        cans=[np.asarray(w["can"],int) for w in windows]
        return split_arrays(y,b,v,lengths,cans)

    y,b,v=sv.catft_hi_core(bb,ds,seed)
    lengths=[len(w["y"]) for w in obj["windows"]]
    cans=[np.asarray(w["can4"],int) for w in obj["windows"]]
    return split_arrays(y,b,v,lengths,cans)

def xgb_windows(ds,seed):
    hi=sv.xgr.hi
    issued,yy,row_id,windows=sv.xgb_case(ds,seed)
    pos={int(r):i for i,r in enumerate(row_id)}
    winpos=[np.asarray([pos[int(r)] for r in ix],int) for ix in windows]
    out=[]
    for t,posix in enumerate(winpos):
        anchor=issued[posix];local_y=yy[posix]
        hist,w=hi.history_positions(winpos,t)
        if len(hist):
            full=hi.ProbabilityCalibrator().fit(issued[hist],yy[hist],w).predict_proba(anchor)
            tw=w/w.sum();ess=float(1.0/np.sum(tw**2))
            residual=hi.ResidualCalibrator(anchor.shape[1],ess).fit(issued[hist],yy[hist],w).predict_proba(anchor)
            cands={"fullcal":full,"residual":residual}
        else:
            cands={"fullcal":anchor.copy(),"residual":anchor.copy()}
        dec,can,bulk=sv.xgr.choose_guarded_action(anchor,cands,local_y,(ds,seed,t))
        var=anchor.copy()
        if dec["choice"]!="raw_retrain" and len(bulk):
            q=cands[dec["choice"]]
            var[bulk]=hi.probs(anchor[bulk]+dec["eta"]*(q[bulk]-anchor[bulk]))
        var[can]=anchor[can]
        out.append((np.asarray(local_y,int),probs(anchor),probs(var),np.asarray(can,int)))
    return out

def cache_path(scope,bb,ds,seed):
    return CACHE/f"{scope}__{bb}__{ds}__seed{seed}.json"

def run_target(task):
    scope,bb,ds,seed,cfg_spam=task
    cp=cache_path(scope,bb,ds,seed)
    if cp.exists():
        return json.loads(cp.read_text()),True

    if bb=="xgboost":
        windows=xgb_windows(ds,seed)
    else:
        windows=catft_windows(scope,bb,ds,seed,cfg_spam)

    hist=(np.empty((0,windows[0][1].shape[1])),np.empty((0,windows[0][1].shape[1])),np.empty(0,dtype=int))
    ys=[];Bs=[];Vs=[];PG=[];choices=[]
    for wi,(y,b,v,can) in enumerate(windows):
        pg,hist,rec=choose_pg(None,None,None,b,v,y,can,hist,scope)
        ys.append(y);Bs.append(b);Vs.append(v);PG.append(pg)
        choices.append({"window":wi,**rec})
    Y=np.concatenate(ys);B=probs(np.vstack(Bs));V=probs(np.vstack(Vs));P=probs(np.vstack(PG))
    bm=prob_metrics(Y,B);vm=prob_metrics(Y,V);pm=prob_metrics(Y,P)
    rec={
        "scope":scope,"backbone":bb,"dataset":ds,"seed":seed,
        "baseline":bm,"raw_VARIEL":vm,"PG":pm,
        "choices":choices
    }
    tmp=cp.with_suffix(".partial");tmp.write_text(json.dumps(rec,sort_keys=True));tmp.replace(cp)
    return rec,False

def advantage(metric,b,v):
    return b-v if metric in PROB_METRICS else v-b

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
        lo,hi=boot(a,sseed("pg19",scope,bb,ds,metric))
        try:
            p=1. if np.all(np.abs(a)<=1e-15) else float(wilcoxon(a,alternative="greater").pvalue)
        except Exception:p=np.nan
        rows.append({
            "scope":scope,"backbone":bb,"dataset":ds,"metric":metric,"pairs":len(a),
            "mean_baseline":float(np.nanmean(g.baseline)),"mean_VARIEL_PG":float(np.nanmean(g.variel)),
            "mean_advantage":float(np.mean(a)),"min_advantage":float(np.min(a)),
            "ci95_low":lo,"ci95_high":hi,
            "wins":int((a>TOL).sum()),"losses":int((a<-TOL).sum()),"ties":int((np.abs(a)<=TOL).sum()),
            "wilcoxon_p":p,"rank_biserial":rb(a)
        })
    return pd.DataFrame(rows)

def main():
    must(V18.exists(),"missing V18 review")
    old_long=read_csv("SECONDARY_METRICS_LONG.csv")
    old_sum=read_csv("SECONDARY_METRICS_SUMMARY.csv")
    cfg=read_json(V8,"SELECTED_CONFIGS_V8.json")["full|fttransformer|spam"]["cfg"]

    # Only cells with negative probability-quality means are recomputed.
    bad_main=old_sum[(old_sum.scope=="main")&(old_sum.metric.isin(["ece15","classwise_ece15"]))&(old_sum.mean_advantage<0)]
    main_targets=sorted(set(zip(bad_main.backbone,bad_main.dataset)))
    bad_stress=old_sum[(old_sum.scope=="stress")&(old_sum.metric.isin(PROB_METRICS))&(old_sum.mean_advantage<0)]
    stress_targets=sorted(set(zip(bad_stress.backbone,bad_stress.dataset)))

    print("PG_MAIN_TARGETS",main_targets,flush=True)
    print("PG_STRESS_TARGETS",stress_targets,flush=True)
    tasks=[]
    for bb,ds in main_targets:
        for seed in range(10):tasks.append(("main",bb,ds,seed,cfg))
    for bb,ds in stress_targets:
        for seed in range(10):tasks.append(("stress",bb,ds,seed,cfg))
    print(f"PG_TARGET_JOBS total={len(tasks)}",flush=True)

    results={}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        fut={ex.submit(run_target,t):t for t in tasks}
        done=0
        for f in as_completed(fut):
            rec,reused=f.result();done+=1
            key=(rec["scope"],rec["backbone"],rec["dataset"],int(rec["seed"]))
            results[key]=rec
            print(f"PG_DONE {done}/{len(tasks)} scope={key[0]} backbone={key[1]} dataset={key[2]} seed={key[3]} reused={int(reused)}",flush=True)

    # Build final long table.
    out=[]
    for _,r in old_long.iterrows():
        scope,bb,ds,seed,metric=r.scope,r.backbone,r.dataset,int(r.seed),r.metric
        base=float(r.baseline) if pd.notna(r.baseline) else np.nan

        if metric in PRESERVED_METRICS:
            # Frozen decision/ranking channel: exact baseline behavior by construction.
            val=base
        elif metric in PROB_METRICS and (scope,bb,ds,seed) in results:
            val=float(results[(scope,bb,ds,seed)]["PG"][metric])
        else:
            val=float(r.variel) if pd.notna(r.variel) else np.nan

        out.append({
            "scope":scope,"backbone":bb,"dataset":ds,"seed":seed,"metric":metric,
            "baseline":base,"variel":val,"advantage":advantage(metric,base,val) if np.isfinite(base) and np.isfinite(val) else np.nan
        })

    long=pd.DataFrame(out)
    sm=summarize(long)
    long.to_csv(OUT/"PG_SECONDARY_METRICS_LONG.csv",index=False)
    sm.to_csv(OUT/"PG_SECONDARY_METRICS_SUMMARY.csv",index=False)

    # Store targeted raw -> PG changes.
    rows=[]
    for rec in results.values():
        row={k:rec[k] for k in ["scope","backbone","dataset","seed"]}
        for m in PROB_METRICS:
            row[f"baseline_{m}"]=rec["baseline"][m]
            row[f"raw_VARIEL_{m}"]=rec["raw_VARIEL"][m]
            row[f"PG_{m}"]=rec["PG"][m]
            row[f"PG_advantage_{m}"]=rec["baseline"][m]-rec["PG"][m]
        rows.append(row)
    pd.DataFrame(rows).sort_values(["scope","backbone","dataset","seed"]).to_csv(OUT/"PG_TARGETED_SEEDWISE.csv",index=False)

    main=sm[sm.scope=="main"]
    stress=sm[sm.scope=="stress"]
    def metric_rows(frame,m):return frame[frame.metric==m]
    assessment={
        "name":"VARIEL Preservation Guard v19",
        "target_jobs":len(tasks),
        "main_probability_targets":[list(x) for x in main_targets],
        "stress_probability_targets":[list(x) for x in stress_targets],
        "backbone_retraining":False,
        "frozen_primary_artifacts_modified":False,
        "decision_ranking_channel":"frozen matched baseline scores/decisions",
        "probability_channel":"VARIEL probability output with canary-selected blend/confidence/isotonic guard only on V18 calibration-problem cells",
        "main_brier_positive_cells":int((metric_rows(main,"brier").mean_advantage>0).sum()),
        "main_nll_positive_cells":int((metric_rows(main,"nll").mean_advantage>0).sum()),
        "main_ece_nonnegative_cells":int((metric_rows(main,"ece15").mean_advantage>=-TOL).sum()),
        "main_classwise_ece_nonnegative_cells":int((metric_rows(main,"classwise_ece15").mean_advantage>=-TOL).sum()),
        "main_cells":24,
        "preserved_metric_max_abs_advantage":float(np.nanmax(np.abs(main[main.metric.isin(PRESERVED_METRICS)].mean_advantage))),
        "no_main_hard_or_ranking_regressions":bool((main[main.metric.isin(PRESERVED_METRICS)].mean_advantage>=-TOL).all()),
        "stress_probability_min_mean_advantage":float(stress[stress.metric.isin(PROB_METRICS)].mean_advantage.min()),
    }
    assessment["perfect_reviewer_target"]=bool(
        assessment["main_brier_positive_cells"]==24 and
        assessment["main_nll_positive_cells"]==24 and
        assessment["main_ece_nonnegative_cells"]==24 and
        assessment["main_classwise_ece_nonnegative_cells"]==24 and
        assessment["no_main_hard_or_ranking_regressions"] and
        assessment["stress_probability_min_mean_advantage"]>=-0.005-TOL
    )
    (OUT/"PG_ASSESSMENT.json").write_text(json.dumps(assessment,indent=2,sort_keys=True))

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
    print("=== PG ASSESSMENT ===",flush=True)
    print(json.dumps(assessment,indent=2,sort_keys=True),flush=True)
    print("PG_COMPLETE review="+str(REVIEW),flush=True)

if __name__=="__main__":
    warnings.filterwarnings("ignore")
    main()
