#!/usr/bin/env python3
"""
ASPIRE-BOSA-NMCS ALL-SIX LABEL-EFFICIENCY SWEEP
================================================

Brier-Orthogonalized Sensitivity Audit with a Normal-Mixture Confidence Sequence.

WHY THIS EXISTS
---------------
The prior BSEI audit preserved perfect correctness (1080/1080), but still had
eight census cases. Its statistical interval used stagewise alpha spending:
late looks received extremely small alpha_t, making late-stage confidence
unnecessarily conservative.

This version removes stagewise alpha splitting entirely.

STATISTICAL CORE
----------------
After Brier midpoint orthogonalization and sensitivity-equalized importance
sampling,

    X_t = xi_{I_t} / (N q_{I_t})

are iid conditional on the finite labelled population and satisfy

    |X_t| <= c,
    c = sum_i h_i / N.

Hence the centered process is c-sub-Gaussian by Hoeffding's lemma.

We use the TWO-SIDED NORMAL-MIXTURE uniform boundary from Howard et al.:

    u(v) = sqrt((v + rho) *
                log((v + rho) / (alpha^2 * rho)))

with intrinsic time

    v_t = c^2 t.

Therefore, simultaneously for all t with probability >= 1-alpha,

    mean(X)_t +/- u(c^2 t)/t

covers the residual mean.

No alpha splitting across looks is required.

TUNING
------
The normal-mixture rho is chosen through the Howard et al. optimal-intrinsic-
time mapping

    rho = v_opt / (-W_{-1}(-alpha^2/e) - 1).

We parameterize v_opt label-free by a target draw horizon

    v_opt = c^2 * t_opt,
    t_opt = max(1, t_multiplier * N).

Candidate t_multipliers:
    0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1, 2, 5.

This is an ALL-SIX DEVELOPMENT SWEEP. Selection requires perfect empirical
correctness first, then minimizes mean UNIQUE label fraction.

PERFECT LAYERS LEFT UNCHANGED
-----------------------------
* UMAR kappa=.075, eta_cap=.70
* H <= .02
* SPARSE/FULL endpoints
* resource objective
* delta-regret tolerance=.001
* Brier midpoint orthogonalization
* q_i proportional to h_i
* zero-label dominance
* label caching
* exact finite-population envelope

NOVEL COMBINATION
-----------------
The normal-mixture CS itself is established prior art and must be cited.
The methodological object under test is its integration with:
  - Brier-specific label-free resource-gap orthogonalization,
  - minimax outcome-sensitivity importance allocation,
  - cached-label auditing,
  - exact finite-population envelope,
  - cost-aware FULL/SPARSE certification,
  - delta-regret stopping,
  - independent H<=epsilon action liability.
"""

from __future__ import annotations
from pathlib import Path
import os, sys, math, json, hashlib, zipfile
import numpy as np
import pandas as pd
from scipy.special import lambertw
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
PROJECT=Path(os.environ.get(
    "DTRAR_PROJECT",
    str(Path.home()/"Documents"/"DTRA_Start"/"DTRA")
)).resolve()
OUT=PROJECT/"artifacts"/"dtrar_aspire_bosa_nmcs_all_six"
OUT.mkdir(parents=True,exist_ok=True)

sys.path.insert(0,str(HERE))
sys.path.insert(0,str(PROJECT))

import run_bsei_all_six as bsei

# Reuse exact proven endpoint/data machinery.
DATASETS=bsei.DATASETS
SEEDS=bsei.SEEDS
LAMBDAS=bsei.LAMBDAS
BUDGET=bsei.BUDGET

ALPHA=.05
DELTA=.001

# This is a statistical-draw horizon, not label fraction. Repeats reuse labels.
MAX_DRAW_MULTIPLE=20.0

T_OPT_MULTIPLIERS=[
    .0025,.005,.01,.02,.05,.10,.25,.50,1.0,2.0,5.0
]

def must(c,m):
    if not c:
        raise RuntimeError(m)

def rho_from_vopt(v_opt,alpha):
    """
    Howard et al. normal-mixture optimal-time mapping:
      rho = v_opt / (-W_{-1}(-alpha^2/e) - 1)
    """
    a=float(alpha)
    v=max(float(v_opt),1e-30)
    w=lambertw(-(a*a)/math.e,k=-1)
    den=-float(np.real(w))-1.0
    must(np.isfinite(den) and den>0,"Invalid Lambert-W rho denominator")
    return v/den

def normal_mixture_radius(t,c,alpha,t_opt):
    """
    Two-sided normal-mixture time-uniform radius for a mean of variables whose
    centered process is c-sub-Gaussian.
    """
    tt=np.asarray(t,dtype=float)
    cc=float(c)
    if cc<=1e-30:
        return np.zeros_like(tt)

    v=(cc*cc)*tt
    v_opt=(cc*cc)*max(float(t_opt),1.0)
    rho=rho_from_vopt(v_opt,alpha)

    u=np.sqrt(
        np.maximum(
            0.0,
            (v+rho) *
            np.log((v+rho)/((alpha*alpha)*rho))
        )
    )
    return u/tt

def importance_path(ztrue,zclass,dataset,seed,profile):
    """
    Construct one deterministic (seeded) importance-sampling path and vectorize
    all statistical/exact-envelope quantities over every draw.
    """
    lo=np.min(zclass,axis=1)
    hi=np.max(zclass,axis=1)
    mid=.5*(lo+hi)
    half=.5*(hi-lo)

    N=len(ztrue)
    active=np.where(half>1e-15)[0]
    H=float(np.sum(half[active]))
    known_mid=float(np.mean(mid))
    zero_lo=float(np.mean(lo))
    zero_hi=float(np.mean(hi))

    if len(active)==0 or H<=1e-30:
        return {
            "N":N,"known_mid":known_mid,
            "zero_lo":zero_lo,"zero_hi":zero_hi,
            "c":0.0,"t":np.array([],int),
            "estimate":np.array([],float),
            "exact_lo":np.array([],float),
            "exact_hi":np.array([],float),
            "unique_count":np.array([],int),
            "draws":np.array([],int),
        }

    q=half[active]/H
    T=max(1,int(math.ceil(MAX_DRAW_MULTIPLE*N)))

    rng=np.random.default_rng(
        bsei.stable_seed("bosa_nmcs",dataset,seed,profile)
    )
    draw_pos=rng.choice(len(active),size=T,replace=True,p=q)
    draws=active[draw_pos]

    residual=ztrue[draws]-mid[draws]
    qdraw=q[draw_pos]
    x=residual/(N*qdraw)

    c=H/N
    must(
        float(np.max(np.abs(x)))<=c+1e-10,
        f"importance range violation {dataset}/{seed}/{profile}"
    )

    t=np.arange(1,T+1,dtype=int)
    estimate=known_mid+np.cumsum(x)/t

    # Exact finite-population envelope, updated only when a label is first bought.
    first=np.zeros(T,dtype=bool)
    seen=np.zeros(N,dtype=bool)
    unique_count=np.zeros(T,dtype=int)

    delta_lo=np.zeros(T,float)
    delta_hi=np.zeros(T,float)
    n_unique=0
    for j,idx in enumerate(draws):
        ii=int(idx)
        if not seen[ii]:
            seen[ii]=True
            first[j]=True
            n_unique+=1
            delta_lo[j]=(float(ztrue[ii])-float(lo[ii]))/N
            delta_hi[j]=(float(ztrue[ii])-float(hi[ii]))/N
        unique_count[j]=n_unique

    exact_lo=zero_lo+np.cumsum(delta_lo)
    exact_hi=zero_hi+np.cumsum(delta_hi)

    return {
        "N":N,"known_mid":known_mid,
        "zero_lo":zero_lo,"zero_hi":zero_hi,
        "c":c,"t":t,"estimate":estimate,
        "exact_lo":exact_lo,"exact_hi":exact_hi,
        "unique_count":unique_count,"draws":draws,
    }

def earliest_decisions(path,true_gain,t_opt_multiplier):
    """
    Return interval arrays for one normal-mixture tuning. A single confidence
    sequence is valid over every draw t; no stagewise alpha spending.
    """
    N=path["N"]

    # Zero-label envelope.
    zL=float(path["zero_lo"])
    zU=float(path["zero_hi"])

    if len(path["t"])==0:
        return {
            "zero":(zL,zU),
            "lower":np.array([],float),
            "upper":np.array([],float),
            "estimate":np.array([],float),
            "label_fraction":np.array([],float),
        }

    t=path["t"].astype(float)
    t_opt=max(1.0,float(t_opt_multiplier)*N)
    rad=normal_mixture_radius(
        t,path["c"],ALPHA,t_opt
    )

    stat_lo=path["estimate"]-rad
    stat_hi=path["estimate"]+rad

    lower=np.maximum(stat_lo,path["exact_lo"])
    upper=np.minimum(stat_hi,path["exact_hi"])

    # Numerical fallback: deterministic exact envelope is authoritative.
    bad=lower>upper+1e-12
    lower[bad]=path["exact_lo"][bad]
    upper[bad]=path["exact_hi"][bad]

    return {
        "zero":(zL,zU),
        "lower":lower,
        "upper":upper,
        "estimate":path["estimate"],
        "label_fraction":path["unique_count"]/N,
        "unique_count":path["unique_count"],
        "t":path["t"],
    }

def decide_from_arrays(arr,true_gain,lam):
    threshold=float(lam*(1.0-BUDGET))
    true_best="full" if true_gain>threshold else "sparse"

    # Zero-label exact/delta stop.
    L0,U0=arr["zero"]
    if L0>threshold:
        dep=("full","zero_exact_full",0.0,0.0,0,L0,U0)
        ex=("full","zero_exact_full",0.0,0.0,0,L0,U0)
        return ex,dep,true_best,threshold
    if U0<threshold:
        dep=("sparse","zero_exact_sparse",0.0,0.0,0,L0,U0)
        ex=("sparse","zero_exact_sparse",0.0,0.0,0,L0,U0)
        return ex,dep,true_best,threshold

    full_reg=max(0.0,threshold-L0)
    sparse_reg=max(0.0,U0-threshold)
    zero_dep=None
    if min(full_reg,sparse_reg)<=DELTA+1e-15:
        if full_reg<=sparse_reg:
            zero_dep=("full","zero_delta_full",full_reg,0.0,0,L0,U0)
        else:
            zero_dep=("sparse","zero_delta_sparse",sparse_reg,0.0,0,L0,U0)

    lower=arr["lower"];upper=arr["upper"]
    lf=arr["label_fraction"];uc=arr["unique_count"];tt=arr["t"]

    exact=None
    deploy=zero_dep

    # Find exact sign crossing.
    full_ix=np.flatnonzero(lower>threshold)
    sparse_ix=np.flatnonzero(upper<threshold)
    candidates=[]
    if len(full_ix):
        candidates.append((int(full_ix[0]),"full","exact_full"))
    if len(sparse_ix):
        candidates.append((int(sparse_ix[0]),"sparse","exact_sparse"))
    if candidates:
        j,act,mode=min(candidates,key=lambda x:x[0])
        exact=(
            act,mode,0.0,float(lf[j]),int(tt[j]),
            float(lower[j]),float(upper[j])
        )

    if deploy is None:
        reg_full=np.maximum(0.0,threshold-lower)
        reg_sparse=np.maximum(0.0,upper-threshold)
        minreg=np.minimum(reg_full,reg_sparse)
        dix=np.flatnonzero(
            (lower>threshold)
            |(upper<threshold)
            |(minreg<=DELTA+1e-15)
        )
        if len(dix):
            j=int(dix[0])
            if lower[j]>threshold:
                deploy=(
                    "full","exact_full",0.0,float(lf[j]),int(tt[j]),
                    float(lower[j]),float(upper[j])
                )
            elif upper[j]<threshold:
                deploy=(
                    "sparse","exact_sparse",0.0,float(lf[j]),int(tt[j]),
                    float(lower[j]),float(upper[j])
                )
            elif reg_full[j]<=reg_sparse[j]:
                deploy=(
                    "full","delta_full",float(reg_full[j]),
                    float(lf[j]),int(tt[j]),
                    float(lower[j]),float(upper[j])
                )
            else:
                deploy=(
                    "sparse","delta_sparse",float(reg_sparse[j]),
                    float(lf[j]),int(tt[j]),
                    float(lower[j]),float(upper[j])
                )

    # Exact census fallback preserves perfect endpoint identity.
    if exact is None:
        exact=(
            true_best,"census_exact",0.0,1.0,
            int(arr["t"][-1]) if len(arr["t"]) else 0,
            float(true_gain),float(true_gain)
        )

    if deploy is None:
        deploy=(
            true_best,"census_exact",0.0,1.0,
            int(arr["t"][-1]) if len(arr["t"]) else 0,
            float(true_gain),float(true_gain)
        )

    return exact,deploy,true_best,threshold

def append_row(dest,dataset,seed,profile,lam,dec,true_best,threshold,true_gain,mult):
    action,mode,reg,label_frac,draws,L,U=dec
    dest.append({
        "dataset":dataset,
        "seed":seed,
        "profile":profile,
        "lambda":float(lam),
        "t_opt_multiplier":float(mult),
        "action":action,
        "true_best":true_best,
        "empirical_best_or_tied":bool(action==true_best),
        "stop_mode":mode,
        "certified_worst_case_regret":float(reg),
        "label_fraction":float(label_frac),
        "draws":int(draws),
        "lower":float(L),
        "upper":float(U),
        "threshold":float(threshold),
        "true_gain":float(true_gain),
        "true_margin":float(true_gain-threshold),
    })

def main():
    # Build expensive model endpoints ONCE for all candidate CS tunings.
    cases=[]
    Hmax=[]

    with threadpool_limits(limits=1):
        job=0
        for dataset in DATASETS:
            for seed in SEEDS:
                job+=1
                data,target,util,safe,fullU,fullS,sig=bsei.load_case(
                    dataset,seed
                )

                if sig is not None and hasattr(sig,"columns") and "sparse_safety_H" in sig.columns:
                    Hmax.extend(sig.sparse_safety_H.astype(float).tolist())

                y=np.asarray(data.y[target],int)
                for profile,(ps,pf) in {
                    "utility":(util[target],fullU[target]),
                    "safety02":(safe[target],fullS[target]),
                }.items():
                    zclass=bsei.go.loss_difference_by_class(ps,pf)
                    ztrue=np.asarray([
                        zclass[i,int(y[i])]
                        for i in range(len(y))
                    ],float)
                    true_gain=float(np.mean(ztrue))
                    path=importance_path(
                        ztrue,zclass,dataset,seed,profile
                    )
                    cases.append({
                        "dataset":dataset,"seed":seed,"profile":profile,
                        "ztrue":ztrue,"zclass":zclass,
                        "true_gain":true_gain,"path":path,
                    })

                print(
                    f"BOSA_ENDPOINT_CACHE {job}/60 dataset={dataset} seed={seed}",
                    flush=True
                )

    sweep=[]
    saved={}

    for mult in T_OPT_MULTIPLIERS:
        dep_rows=[]
        ex_rows=[]

        for case in cases:
            arr=earliest_decisions(
                case["path"],case["true_gain"],mult
            )
            for lam in LAMBDAS:
                ex,dep,best,thr=decide_from_arrays(
                    arr,case["true_gain"],float(lam)
                )
                append_row(
                    ex_rows,case["dataset"],case["seed"],case["profile"],
                    lam,ex,best,thr,case["true_gain"],mult
                )
                append_row(
                    dep_rows,case["dataset"],case["seed"],case["profile"],
                    lam,dep,best,thr,case["true_gain"],mult
                )

        dep=pd.DataFrame(dep_rows)
        ex=pd.DataFrame(ex_rows)
        cells=dep.groupby(
            ["dataset","profile","lambda"],as_index=False
        ).agg(all_10of10=("empirical_best_or_tied","all"))

        rec={
            "t_opt_multiplier":float(mult),
            "deployment_correct":int(dep.empirical_best_or_tied.sum()),
            "deployment_total":int(len(dep)),
            "deployment_cells_10of10":int(cells.all_10of10.sum()),
            "deployment_cell_total":int(len(cells)),
            "deployment_mean_label_fraction":float(dep.label_fraction.mean()),
            "deployment_median_label_fraction":float(dep.label_fraction.median()),
            "deployment_max_label_fraction":float(dep.label_fraction.max()),
            "deployment_zero_labels":int((dep.label_fraction<=1e-15).sum()),
            "deployment_le_1pct":int((dep.label_fraction<=.01+1e-12).sum()),
            "deployment_le_5pct":int((dep.label_fraction<=.05+1e-12).sum()),
            "deployment_max_certified_regret":float(dep.certified_worst_case_regret.max()),
            "deployment_census_cases":int((dep.stop_mode=="census_exact").sum()),
            "exact_correct":int(ex.empirical_best_or_tied.sum()),
            "exact_total":int(len(ex)),
            "exact_mean_label_fraction":float(ex.label_fraction.mean()),
            "exact_max_label_fraction":float(ex.label_fraction.max()),
            "exact_census_cases":int((ex.stop_mode=="census_exact").sum()),
        }
        sweep.append(rec)
        saved[float(mult)]=(dep,ex)

        print(
            f"BOSA_VARIANT mult={mult:g} "
            f"correct={rec['deployment_correct']}/1080 "
            f"mean={rec['deployment_mean_label_fraction']:.6f} "
            f"max={rec['deployment_max_label_fraction']:.6f} "
            f"census={rec['deployment_census_cases']}",
            flush=True
        )

    sw=pd.DataFrame(sweep)
    valid=sw[
        (sw.deployment_correct==1080)
        &(sw.deployment_cells_10of10==108)
        &(sw.deployment_max_certified_regret<=DELTA+1e-12)
        &(sw.exact_correct==1080)
    ].copy()

    if len(valid):
        valid=valid.sort_values(
            [
                "deployment_mean_label_fraction",
                "deployment_max_label_fraction",
                "deployment_census_cases",
                "deployment_le_1pct",
                "deployment_le_5pct",
            ],
            ascending=[True,True,True,False,False]
        )
        best=valid.iloc[0]
        perfect_found=True
    else:
        best=sw.sort_values(
            [
                "deployment_correct",
                "deployment_cells_10of10",
                "deployment_mean_label_fraction",
            ],
            ascending=[False,False,True]
        ).iloc[0]
        perfect_found=False

    sw["selected"]=np.isclose(
        sw.t_opt_multiplier,float(best.t_opt_multiplier)
    )
    sw.to_csv(OUT/"BOSA_NMCS_TUNING_SWEEP.csv",index=False)

    dep,ex=saved[float(best.t_opt_multiplier)]
    dep.to_csv(OUT/"BOSA_SELECTED_DEPLOYMENT_SEEDWISE.csv",index=False)
    ex.to_csv(OUT/"BOSA_SELECTED_EXACT_SEEDWISE.csv",index=False)

    cells=dep.groupby(
        ["dataset","profile","lambda"],as_index=False
    ).agg(
        seeds_correct=("empirical_best_or_tied","sum"),
        all_10of10=("empirical_best_or_tied","all"),
        mean_label_fraction=("label_fraction","mean"),
        median_label_fraction=("label_fraction","median"),
        max_label_fraction=("label_fraction","max"),
        max_certified_regret=("certified_worst_case_regret","max"),
    )
    cells.to_csv(OUT/"BOSA_SELECTED_CELL_SUMMARY.csv",index=False)

    per=dep.groupby("dataset",as_index=False).agg(
        decisions_correct=("empirical_best_or_tied","sum"),
        decisions_total=("empirical_best_or_tied","size"),
        mean_label_fraction=("label_fraction","mean"),
        median_label_fraction=("label_fraction","median"),
        max_label_fraction=("label_fraction","max"),
        zero_labels=("label_fraction",lambda x:int((x<=1e-15).sum())),
        le_1pct=("label_fraction",lambda x:int((x<=.01+1e-12).sum())),
        le_5pct=("label_fraction",lambda x:int((x<=.05+1e-12).sum())),
        census_cases=("stop_mode",lambda x:int((x=="census_exact").sum())),
        max_certified_regret=("certified_worst_case_regret","max"),
    )
    cc=cells.groupby("dataset").agg(
        cells_10of10=("all_10of10","sum"),
        cell_total=("all_10of10","size"),
    ).reset_index()
    per=per.merge(cc,on="dataset",how="left")
    per.to_csv(OUT/"BOSA_SELECTED_PER_DATASET_SUMMARY.csv",index=False)

    assessment={
        "architecture":
            "ASPIRE-BOSA-NMCS: Brier-orthogonalized sensitivity auditing + cached importance labels + exact envelope + time-uniform normal-mixture CS",
        "development_on_all_six_known_domains":True,
        "datasets":DATASETS,
        "alpha_total":ALPHA,
        "alpha_spending_across_looks":False,
        "confidence_sequence":
            "two-sided normal-mixture uniform boundary (Howard et al.)",
        "delta_regret":DELTA,
        "sampling_rule":"q_i proportional to Brier half-width h_i",
        "zero_label_dominance":True,
        "cached_repeated_draws":True,
        "max_draw_multiple":MAX_DRAW_MULTIPLE,
        "candidate_t_opt_multipliers":T_OPT_MULTIPLIERS,
        "selected_t_opt_multiplier":float(best.t_opt_multiplier),
        "perfect_configuration_found":bool(perfect_found),
        "deployment_correct":int(best.deployment_correct),
        "deployment_total":int(best.deployment_total),
        "deployment_cells_10of10":int(best.deployment_cells_10of10),
        "deployment_cell_total":int(best.deployment_cell_total),
        "deployment_mean_label_fraction":float(best.deployment_mean_label_fraction),
        "deployment_median_label_fraction":float(best.deployment_median_label_fraction),
        "deployment_max_label_fraction":float(best.deployment_max_label_fraction),
        "deployment_zero_labels":int(best.deployment_zero_labels),
        "deployment_le_1pct":int(best.deployment_le_1pct),
        "deployment_le_5pct":int(best.deployment_le_5pct),
        "deployment_census_cases":int(best.deployment_census_cases),
        "deployment_max_certified_regret":float(best.deployment_max_certified_regret),
        "exact_correct":int(best.exact_correct),
        "exact_total":int(best.exact_total),
        "exact_mean_label_fraction":float(best.exact_mean_label_fraction),
        "exact_max_label_fraction":float(best.exact_max_label_fraction),
        "exact_census_cases":int(best.exact_census_cases),
        "max_sparse_H_if_available":float(max(Hmax)) if Hmax else None,
        "baseline_bsei_mean_label_fraction":0.06936960106496651,
        "baseline_bsei_max_label_fraction":1.0,
        "baseline_bsei_census_cases":8,
        "umar_changed":False,
        "H02_changed":False,
        "resource_endpoints_changed":False,
        "important_interpretation":
            "All six domains are development evidence for this efficiency refinement. If adopted, freeze the selected design and confirm on a fresh operational stream.",
    }

    (OUT/"BOSA_NMCS_FINAL_ASSESSMENT.json").write_text(
        json.dumps(assessment,indent=2,sort_keys=True)
    )

    review=PROJECT/"DTRAR_ASPIRE_BOSA_NMCS_AllSix_EfficiencySweep_Review.zip"
    with zipfile.ZipFile(review,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file():
                z.write(p,p.name)
        z.write(Path(__file__),Path(__file__).name)

    print("\n=== ASPIRE-BOSA-NMCS FINAL ASSESSMENT ===")
    print(json.dumps(assessment,indent=2,sort_keys=True))
    print("\n=== SELECTED PER-DATASET SUMMARY ===")
    print(per.to_string(index=False))
    print(f"\nBOSA_NMCS_COMPLETE review={review.name}")

if __name__=="__main__":
    main()
