#!/usr/bin/env python3
"""
ASPIRE-BURST-FACC
=================

Brier Universal Randomized Sequential Test + perfect FACC fallback.

This solver operates ONLY on the persistent Electricity/Gas endpoint-case
cache.  It performs no base-model training, no full-feedback reconstruction,
and no calibration reconstruction.

Mathematical reduction
----------------------
Sensitivity-equalized Brier importance sampling gives an iid residual draw

    X_t in [-c,+c].

Normalize:

    V_t = (X_t/c + 1)/2 in [0,1].

Binary classification:
    V_t is already exactly in {0,1}.

Multiclass classification:
    draw independent U_t ~ Uniform(0,1) and define

        B_t = 1{U_t <= V_t}.

Then, in BOTH cases,

    B_t ~ Bernoulli(p),  p = E[V_t],

and

    mean resource gap
      = known_mid + c(2p-1).

Thus both Electricity and multiclass Gas can use one exact Bernoulli
anytime-valid inference engine.

Bernoulli mixture confidence sequence
-------------------------------------
For a candidate p, the beta-binomial mixture likelihood ratio is

    M_t(p)
      = [ integral q^S (1-q)^F pi(dq) ]
        / [ p^S (1-p)^F ].

Under Bernoulli(p), M_t(p) is a nonnegative martingale.  By Ville,

    P_p( exists t: M_t(p) >= 1/alpha ) <= alpha.

Therefore

    C_t = { p : M_t(p) < 1/alpha }

is an anytime-valid confidence sequence.

The implementation uses a convex mixture of symmetric Beta priors.  Since the
denominator is common, inversion reduces to two monotone Bernoulli
log-likelihood root searches around the MLE.

No Bonferroni split over four separate resource tests is needed: one 95%
confidence sequence is transformed directly to the Brier resource-gap interval.

Deployment
----------
At each checkpoint transform [p_L,p_U] to [mu_L,mu_U], intersect with the
deterministic finite-population exact envelope, and apply the existing
delta=.001 resource-regret rule.

If BURST has not safely stopped by the candidate switch fraction, invoke the
known-perfect FACC controller on the SAME BOSA importance path.  Purchased
labels are unioned exactly, so fallback cost is not hidden.

Development scope
-----------------
Only Electricity/Gas are modified.  Bank/Rialto/Spam/Airlines retain the
already-perfect selected BETA-FACC results byte-for-byte in the final combined
assessment.

All current domains are development evidence.  Any selected final controller
must be frozen before a fresh external confirmation.
"""

from __future__ import annotations
from pathlib import Path
import os, sys, math, json, zipfile
import numpy as np
import pandas as pd
from scipy.special import betaln, logsumexp
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
PROJECT=Path(os.environ.get(
    "DTRAR_PROJECT",
    str(Path.home()/"Documents"/"DTRA_Start"/"DTRA")
)).resolve()
CACHE=PROJECT/"artifacts"/"dtrar_burst_endpoint_case_cache_v1"
OUT=PROJECT/"artifacts"/"dtrar_aspire_burst_facc_tail"
OUT.mkdir(parents=True,exist_ok=True)

sys.path.insert(0,str(HERE))
sys.path.insert(0,str(PROJECT))

import run_bosa_nmcs_all_six as bosa
import run_beta_facc_all_six as beta_base

DATASETS=["electricity","gas"]
SEEDS=list(range(10))
PROFILES=["utility","safety02"]
LAMBDAS=np.array([0.,.001,.0025,.005,.01,.025,.05,.10,.20],float)

ALPHA=.05
DELTA=.001
BUDGET=.05

# Development choices.  These affect only statistical efficiency.
PRIOR_SETS={
    "jeffreys":np.array([.5],float),
    "robust3":np.array([.5,1.,2.],float),
    "robust5":np.array([.25,.5,1.,2.,4.],float),
}
GAMMAS=[.50,.60,.70,.80,.90]
SWITCHES=[.02,.03,.05,.075,.10,.15,.20]

# Evaluate at a dense but finite set of UNIQUE-label fractions.
CHECKPOINTS=np.array([
    .0005,.001,.002,.003,.005,.0075,.01,.015,.02,.03,.04,.05,
    .075,.10,.15,.20
],float)

def must(c,m):
    if not c:
        raise RuntimeError(m)

def stable_seed(*parts):
    return bosa.bsei.stable_seed(*parts)

def load_cached_case(ds,seed,profile):
    p=CACHE/ds/f"seed_{seed}"/f"{profile}.npz"
    must(p.exists(),f"Missing BURST cache: {p}")
    with np.load(p,allow_pickle=False) as z:
        return {
            "dataset":ds,
            "seed":seed,
            "profile":profile,
            "ztrue":np.asarray(z["ztrue"],float),
            "zclass":np.asarray(z["zclass"],float),
            "true_gain":float(np.asarray(z["true_gain"],float).ravel()[0]),
            "n_classes":int(np.asarray(z["n_classes"],int).ravel()[0]),
            "N":int(np.asarray(z["N"],int).ravel()[0]),
            "max_sparse_H":float(np.asarray(z["max_sparse_H"],float).ravel()[0]),
        }

def prior_log_numerator(S,F,priors):
    """
    Convex equal-weight mixture of symmetric Beta(a,a) priors.
    Numerator is independent of candidate p.
    """
    S=np.asarray(S,float)
    F=np.asarray(F,float)
    vals=[]
    lw=-math.log(len(priors))
    for a in priors:
        vals.append(
            betaln(S+float(a),F+float(a))
            -betaln(float(a),float(a))
            +lw
        )
    return logsumexp(np.vstack(vals),axis=0)

def loglik(S,F,p):
    p=np.clip(np.asarray(p,float),1e-15,1-1e-15)
    return S*np.log(p)+F*np.log1p(-p)

def mixture_cs(S,F,alpha,priors):
    """
    Invert M_t(p)<1/alpha for arrays of Bernoulli sufficient statistics.
    Vectorized bisection around MLE.
    """
    S=np.asarray(S,float)
    F=np.asarray(F,float)
    t=S+F
    must(np.all(t>0),"mixture_cs t<=0")

    num=prior_log_numerator(S,F,priors)
    target=num-math.log(1.0/float(alpha))

    phat=S/t
    llhat=loglik(S,F,np.clip(phat,1e-15,1-1e-15))

    # Numerical safety: if mixture threshold is above max likelihood because of
    # extreme tiny t, return [0,1] rather than an invalid interval.
    feasible=target<=llhat+1e-12

    lower=np.zeros(len(t),float)
    upper=np.ones(len(t),float)

    # Lower root when S>0.
    mask=(S>0)&feasible
    if np.any(mask):
        lo=np.full(np.sum(mask),1e-15,float)
        hi=np.clip(phat[mask],1e-15,1-1e-15)
        Sm=S[mask];Fm=F[mask];Tm=target[mask]
        for _ in range(55):
            mid=.5*(lo+hi)
            val=Sm*np.log(mid)+Fm*np.log1p(-mid)
            # log-likelihood increases on [0, phat]
            left=val<Tm
            lo[left]=mid[left]
            hi[~left]=mid[~left]
        lower[mask]=.5*(lo+hi)

    # Upper root when F>0.
    mask=(F>0)&feasible
    if np.any(mask):
        lo=np.clip(phat[mask],1e-15,1-1e-15)
        hi=np.full(np.sum(mask),1-1e-15,float)
        Sm=S[mask];Fm=F[mask];Tm=target[mask]
        for _ in range(55):
            mid=.5*(lo+hi)
            val=Sm*np.log(mid)+Fm*np.log1p(-mid)
            # log-likelihood decreases on [phat,1]
            right=val<Tm
            hi[right]=mid[right]
            lo[~right]=mid[~right]
        upper[mask]=.5*(lo+hi)

    lower[~feasible]=0.
    upper[~feasible]=1.
    return lower,upper

def build_burst_path(case):
    """
    Reuse the exact BOSA/FACC sensitivity-equalized index path so a fallback can
    reuse purchased labels exactly.
    """
    path=bosa.importance_path(
        case["ztrue"],case["zclass"],
        case["dataset"],case["seed"],case["profile"]
    )
    N=case["N"]
    c=float(path["c"])

    if c<=1e-30 or len(path["draws"])==0:
        return {
            "path":path,
            "check_idx":np.array([],int),
            "S":np.array([],float),
            "F":np.array([],float),
            "label_fraction":np.array([],float),
            "exact_lo":np.array([],float),
            "exact_hi":np.array([],float),
            "known_mid":float(path["known_mid"]),
            "c":c,
        }

    # Reconstruct the exact importance residual from the deterministic
    # BOSA draw path.  bosa.importance_path stores draws / exact envelopes /
    # unique-label counts, but intentionally does not persist the raw x vector.
    #
    #   x_t = (ztrue[I_t] - mid[I_t]) / (N q[I_t])
    #   q_i = h_i / sum_j h_j
    #
    # This is algebraically identical to the x used internally by BOSA.
    lo=np.min(case["zclass"],axis=1)
    hi=np.max(case["zclass"],axis=1)
    mid=.5*(lo+hi)
    half=.5*(hi-lo)
    active=np.where(half>1e-15)[0]
    H=float(np.sum(half[active]))
    must(H>0.0, f"Invalid BURST sensitivity mass {case['dataset']}/{case['seed']}/{case['profile']}")

    draws=np.asarray(path["draws"],int)
    qdraw=half[draws]/H
    residual=np.asarray(case["ztrue"],float)[draws]-mid[draws]
    x=residual/(N*qdraw)

    # Verify reconstruction against the BOSA running estimate before use.
    reconstructed_est=float(path["known_mid"])+np.cumsum(x)/np.arange(1,len(x)+1,dtype=float)
    stored_est=np.asarray(path["estimate"],float)
    must(
        len(reconstructed_est)==len(stored_est)
        and np.allclose(reconstructed_est,stored_est,rtol=1e-11,atol=1e-12),
        f"BURST importance reconstruction mismatch {case['dataset']}/{case['seed']}/{case['profile']}"
    )

    v=(x/c + 1.0)/2.0
    v=np.clip(v,0.0,1.0)

    if case["n_classes"]==2:
        # The binary theorem gives exact 0/1 values up to floating error.
        err=np.minimum(np.abs(v),np.abs(v-1.0))
        must(float(np.max(err))<=1e-8,
             f"Binary BURST identity failed {case['dataset']}/{case['seed']}/{case['profile']}")
        bits=(v>=.5).astype(np.int64)
    else:
        # Exact randomized Bernoulliization for general bounded V in [0,1].
        rng=np.random.default_rng(
            stable_seed("burst_aux",case["dataset"],case["seed"],case["profile"])
        )
        u=rng.random(len(v))
        bits=(u<=v).astype(np.int64)

    Sfull=np.cumsum(bits)
    uc=np.asarray(path["unique_count"],int)
    N=int(path["N"])

    idx=[]
    for frac in CHECKPOINTS:
        target=max(1,int(math.ceil(float(frac)*N)))
        hits=np.flatnonzero(uc>=target)
        if not len(hits):
            continue
        # Last statistical draw before buying the NEXT new label, so evidence is
        # maximized at this purchased-label count.
        j=int(hits[0])
        count=int(uc[j])
        same=np.flatnonzero(uc==count)
        j=int(same[-1])
        idx.append(j)

    idx=np.unique(np.asarray(idx,int))
    S=Sfull[idx].astype(float)
    t=(idx+1).astype(float)
    F=t-S

    return {
        "path":path,
        "check_idx":idx,
        "S":S,
        "F":F,
        "label_fraction":uc[idx].astype(float)/N,
        "exact_lo":np.asarray(path["exact_lo"],float)[idx],
        "exact_hi":np.asarray(path["exact_hi"],float)[idx],
        "known_mid":float(path["known_mid"]),
        "c":c,
    }

def posterior_sign_conf(S,F,p0):
    # Jeffreys posterior is used ONLY to decide whether to trust a formally
    # delta-safe exact-side tie-break.  The regret guarantee comes from the CS.
    from scipy.stats import beta as beta_dist
    A=S+.5;B=F+.5
    if p0<=0:
        prob_full=np.ones(len(S),float)
    elif p0>=1:
        prob_full=np.zeros(len(S),float)
    else:
        prob_full=beta_dist.sf(p0,A,B)
    conf=np.maximum(prob_full,1.-prob_full)
    return prob_full,conf

def burst_decision(case,bp,lam,priors,gamma,switch):
    theta=float(lam*(1.-BUDGET))
    true_best="full" if case["true_gain"]>theta else "sparse"

    # Zero-label deterministic envelope.
    zlo=float(bp["path"]["zero_lo"])
    zhi=float(bp["path"]["zero_hi"])
    if zlo>theta:
        return dict(kind="burst",action="full",mode="zero_exact_full",
                    regret=0.,checkpoint=None,label_fraction=0.,
                    true_best=true_best,threshold=theta)
    if zhi<theta:
        return dict(kind="burst",action="sparse",mode="zero_exact_sparse",
                    regret=0.,checkpoint=None,label_fraction=0.,
                    true_best=true_best,threshold=theta)

    rF=max(0.,theta-zlo); rS=max(0.,zhi-theta)
    if min(rF,rS)<=DELTA+1e-15:
        act="full" if rF<=rS else "sparse"
        return dict(kind="burst",action=act,mode="zero_delta_"+act,
                    regret=min(rF,rS),checkpoint=None,label_fraction=0.,
                    true_best=true_best,threshold=theta)

    if not len(bp["check_idx"]):
        return dict(kind="fallback",action=None,mode="no_path",
                    regret=np.nan,checkpoint=None,label_fraction=0.,
                    true_best=true_best,threshold=theta)

    pL,pU=mixture_cs(bp["S"],bp["F"],ALPHA,priors)
    muL=bp["known_mid"]+bp["c"]*(2*pL-1)
    muU=bp["known_mid"]+bp["c"]*(2*pU-1)

    # Deterministic finite-population envelope can only tighten.
    L=np.maximum(muL,bp["exact_lo"])
    U=np.minimum(muU,bp["exact_hi"])
    bad=L>U+1e-12
    L[bad]=bp["exact_lo"][bad]
    U[bad]=bp["exact_hi"][bad]

    p0=.5*(1.+(theta-bp["known_mid"])/max(bp["c"],1e-30))
    pfull,conf=posterior_sign_conf(bp["S"],bp["F"],p0)

    for j in range(len(L)):
        f=float(bp["label_fraction"][j])

        if L[j]>theta:
            return dict(kind="burst",action="full",mode="burst_exact_full",
                        regret=0.,checkpoint=j,label_fraction=f,
                        true_best=true_best,threshold=theta,
                        lower=float(L[j]),upper=float(U[j]),
                        sign_conf=float(conf[j]))
        if U[j]<theta:
            return dict(kind="burst",action="sparse",mode="burst_exact_sparse",
                        regret=0.,checkpoint=j,label_fraction=f,
                        true_best=true_best,threshold=theta,
                        lower=float(L[j]),upper=float(U[j]),
                        sign_conf=float(conf[j]))

        regF=max(0.,theta-float(L[j]))
        regS=max(0.,float(U[j])-theta)
        rmin=min(regF,regS)

        if rmin<=DELTA+1e-15:
            if regF<regS:
                proposed="full"
            elif regS<regF:
                proposed="sparse"
            else:
                proposed="full" if pfull[j]>=.5 else "sparse"

            sideprob=float(pfull[j]) if proposed=="full" else float(1.-pfull[j])
            if sideprob>=float(gamma)-1e-15:
                return dict(kind="burst",action=proposed,
                            mode="burst_delta_"+proposed,
                            regret=rmin,checkpoint=j,label_fraction=f,
                            true_best=true_best,threshold=theta,
                            lower=float(L[j]),upper=float(U[j]),
                            sign_conf=float(conf[j]))
            return dict(kind="fallback",action=None,mode="fragile_delta",
                        regret=np.nan,checkpoint=j,label_fraction=f,
                        true_best=true_best,threshold=theta,
                        lower=float(L[j]),upper=float(U[j]),
                        sign_conf=float(conf[j]))

        if f>=float(switch)-1e-15:
            return dict(kind="fallback",action=None,mode="switch",
                        regret=np.nan,checkpoint=j,label_fraction=f,
                        true_best=true_best,threshold=theta,
                        lower=float(L[j]),upper=float(U[j]),
                        sign_conf=float(conf[j]))

    return dict(kind="fallback",action=None,mode="exhausted",
                regret=np.nan,checkpoint=len(L)-1,
                label_fraction=float(bp["label_fraction"][-1]),
                true_best=true_best,threshold=theta,
                lower=float(L[-1]),upper=float(U[-1]),
                sign_conf=float(conf[-1]))

def burst_label_set(bp,checkpoint):
    if checkpoint is None:
        return set()
    j=int(bp["check_idx"][int(checkpoint)])
    return set(map(int,bp["path"]["draws"][:j+1]))

def baseline_facc(case,lam):
    """
    Exact selected FACC fallback using the same cached statistical case.
    """
    ev=beta_base.facc.first_fast_event(
        case["arr"],case["true_gain"],float(lam),
        beta_base.FACC_KAPPA,beta_base.FACC_TAIL
    )
    fast_set=beta_base.facc.fast_label_set(case["path"],ev["draws"])

    if ev["kind"]=="fast":
        return {
            "action":ev["action"],
            "mode":"facc_"+ev["mode"],
            "regret":float(ev["regret"]),
            "label_set":fast_set,
            "true_best":ev["true_best"],
        }

    slow=beta_base.facc.go_deployment(
        case["ztrue"],case["zclass"],
        case["seed"],case["profile"],float(lam)
    )
    union=set(fast_set)
    union.update(slow["label_set"])
    return {
        "action":slow["action"],
        "mode":"facc_slow_"+slow["mode"],
        "regret":float(slow["regret"]),
        "label_set":union,
        "true_best":ev["true_best"],
    }

def prepare_case(raw):
    path=bosa.importance_path(
        raw["ztrue"],raw["zclass"],
        raw["dataset"],raw["seed"],raw["profile"]
    )
    arr=beta_base.facc.fast_arrays(path,raw["true_gain"])
    c=dict(raw)
    c["path"]=path
    c["arr"]=arr
    c["burst"]=build_burst_path(raw)
    return c

def append_tail_row(dest,case,lam,event,final_action,mode,regret,label_set,
                    route):
    dest.append({
        "dataset":case["dataset"],
        "seed":case["seed"],
        "profile":case["profile"],
        "n_classes":case["n_classes"],
        "lambda":float(lam),
        "action":final_action,
        "true_best":event["true_best"],
        "empirical_best_or_tied":bool(final_action==event["true_best"]),
        "route":route,
        "burst_mode":event["mode"],
        "final_mode":mode,
        "label_fraction":float(len(label_set)/case["N"]),
        "labels":int(len(label_set)),
        "certified_worst_case_regret":float(regret),
        "burst_label_fraction":float(event["label_fraction"]),
        "sign_conf":float(event.get("sign_conf",np.nan)),
    })

def main():
    manifest=CACHE/"MANIFEST.json"
    must(manifest.exists(),"BURST endpoint cache missing; run build_burst_tail_cache.py first")

    baseline=pd.read_csv(HERE/"BETA_FACC_SELECTED_SEEDWISE.csv")
    must(len(baseline)==1080,"Unexpected BETA-FACC baseline row count")
    must(bool(baseline.empirical_best_or_tied.all()),"Baseline is not perfect")

    raw_cases=[]
    for ds in DATASETS:
        for seed in SEEDS:
            for profile in PROFILES:
                raw_cases.append(load_cached_case(ds,seed,profile))

    # Cache-only statistical preparation.
    cases=[prepare_case(c) for c in raw_cases]

    sweep=[]
    saved={}

    for prior_name,priors in PRIOR_SETS.items():
        for gamma in GAMMAS:
            for switch in SWITCHES:
                tail_rows=[]
                burst_accepts=0
                fallbacks=0

                for case in cases:
                    for lam in LAMBDAS:
                        be=burst_decision(
                            case,case["burst"],float(lam),
                            priors,float(gamma),float(switch)
                        )
                        bset=burst_label_set(case["burst"],be["checkpoint"])

                        if be["kind"]=="burst":
                            burst_accepts+=1
                            append_tail_row(
                                tail_rows,case,lam,be,be["action"],be["mode"],
                                be["regret"],bset,"burst"
                            )
                        else:
                            fallbacks+=1
                            fd=baseline_facc(case,float(lam))
                            union=set(bset)
                            union.update(fd["label_set"])
                            append_tail_row(
                                tail_rows,case,lam,be,fd["action"],fd["mode"],
                                fd["regret"],union,"burst_to_facc"
                            )

                tail=pd.DataFrame(tail_rows)
                must(len(tail)==360,"Tail decision count must be 360")

                # Replace Electricity/Gas rows in the perfect six-domain baseline.
                keep=baseline[~baseline.dataset.isin(DATASETS)].copy()
                replacement=tail.rename(columns={
                    "route":"path",
                    "label_fraction":"label_fraction",
                })

                # Normalize only the columns needed for combined assessment.
                comb=pd.concat([
                    keep[["dataset","seed","profile","lambda","action","true_best",
                          "empirical_best_or_tied","label_fraction",
                          "certified_worst_case_regret"]],
                    replacement[["dataset","seed","profile","lambda","action","true_best",
                                 "empirical_best_or_tied","label_fraction",
                                 "certified_worst_case_regret"]],
                ],ignore_index=True)

                cells=comb.groupby(
                    ["dataset","profile","lambda"],as_index=False
                ).agg(all_10of10=("empirical_best_or_tied","all"))

                rec={
                    "prior":prior_name,
                    "gamma":float(gamma),
                    "switch":float(switch),
                    "correct":int(comb.empirical_best_or_tied.sum()),
                    "total":int(len(comb)),
                    "cells_10of10":int(cells.all_10of10.sum()),
                    "cell_total":int(len(cells)),
                    "mean_label_fraction":float(comb.label_fraction.mean()),
                    "median_label_fraction":float(comb.label_fraction.median()),
                    "max_label_fraction":float(comb.label_fraction.max()),
                    "zero_labels":int((comb.label_fraction<=1e-15).sum()),
                    "le_1pct":int((comb.label_fraction<=.01+1e-12).sum()),
                    "le_5pct":int((comb.label_fraction<=.05+1e-12).sum()),
                    "le_10pct":int((comb.label_fraction<=.10+1e-12).sum()),
                    "burst_accepts_tail":int(burst_accepts),
                    "fallbacks_tail":int(fallbacks),
                    "max_certified_regret":
                        float(comb.certified_worst_case_regret.max()),
                    "electricity_mean":
                        float(comb[comb.dataset=="electricity"].label_fraction.mean()),
                    "electricity_max":
                        float(comb[comb.dataset=="electricity"].label_fraction.max()),
                    "gas_mean":
                        float(comb[comb.dataset=="gas"].label_fraction.mean()),
                    "gas_max":
                        float(comb[comb.dataset=="gas"].label_fraction.max()),
                }
                sweep.append(rec)
                saved[(prior_name,float(gamma),float(switch))]=(tail,comb)

                pd.DataFrame(sweep).to_csv(
                    OUT/"BURST_SWEEP_INPROGRESS.csv",index=False
                )

                print(
                    f"BURST_VARIANT prior={prior_name} gamma={gamma:.2f} "
                    f"switch={switch:.3f} correct={rec['correct']}/1080 "
                    f"mean={rec['mean_label_fraction']:.6f} "
                    f"max={rec['max_label_fraction']:.6f} "
                    f"Emean={rec['electricity_mean']:.6f} "
                    f"Gmean={rec['gas_mean']:.6f}",
                    flush=True
                )

    sw=pd.DataFrame(sweep)
    valid=sw[
        (sw["correct"]==1080)
        &(sw["cells_10of10"]==108)
        &(sw["max_certified_regret"]<=DELTA+1e-12)
    ].copy()

    if len(valid):
        valid=valid.sort_values(
            ["mean_label_fraction","max_label_fraction",
             "electricity_mean","gas_mean","fallbacks_tail",
             "le_1pct","le_5pct"],
            ascending=[True,True,True,True,True,False,False]
        )
        best=valid.iloc[0]
        perfect=True
    else:
        best=sw.sort_values(
            ["correct","cells_10of10","mean_label_fraction"],
            ascending=[False,False,True]
        ).iloc[0]
        perfect=False

    key=(str(best["prior"]),float(best["gamma"]),float(best["switch"]))
    tail,comb=saved[key]

    sw["selected"]=(
        (sw["prior"]==str(best["prior"]))
        &np.isclose(sw["gamma"],float(best["gamma"]))
        &np.isclose(sw["switch"],float(best["switch"]))
    )
    sw.to_csv(OUT/"BURST_DEVELOPMENT_SWEEP.csv",index=False)
    tail.to_csv(OUT/"BURST_SELECTED_ELECTRICITY_GAS.csv",index=False)
    comb.to_csv(OUT/"BURST_SELECTED_ALL_SIX.csv",index=False)

    cells=comb.groupby(
        ["dataset","profile","lambda"],as_index=False
    ).agg(
        seeds_correct=("empirical_best_or_tied","sum"),
        all_10of10=("empirical_best_or_tied","all"),
        mean_label_fraction=("label_fraction","mean"),
        max_label_fraction=("label_fraction","max"),
        max_regret=("certified_worst_case_regret","max"),
    )
    cells.to_csv(OUT/"BURST_SELECTED_CELL_SUMMARY.csv",index=False)

    per=comb.groupby("dataset",as_index=False).agg(
        decisions_correct=("empirical_best_or_tied","sum"),
        decisions_total=("empirical_best_or_tied","size"),
        mean_label_fraction=("label_fraction","mean"),
        median_label_fraction=("label_fraction","median"),
        max_label_fraction=("label_fraction","max"),
        zero_labels=("label_fraction",lambda x:int((x<=1e-15).sum())),
        le_1pct=("label_fraction",lambda x:int((x<=.01+1e-12).sum())),
        le_5pct=("label_fraction",lambda x:int((x<=.05+1e-12).sum())),
        le_10pct=("label_fraction",lambda x:int((x<=.10+1e-12).sum())),
        max_regret=("certified_worst_case_regret","max"),
    )
    cc=cells.groupby("dataset").agg(
        cells_10of10=("all_10of10","sum"),
        cell_total=("all_10of10","size"),
    ).reset_index()
    per=per.merge(cc,on="dataset",how="left")
    per.to_csv(OUT/"BURST_SELECTED_PER_DATASET.csv",index=False)

    assessment={
        "architecture":
            "ASPIRE-BURST-FACC: universal Bernoulliized Brier resource CS + perfect FACC fallback",
        "development_scope":
            "Electricity/Gas tail solver; other four domains unchanged from perfect BETA-FACC",
        "bernoulliization":
            "V=(X/c+1)/2; binary V is already Bernoulli; multiclass uses B=1{U<=V}, giving Bernoulli(E[V]) exactly",
        "confidence_sequence":
            "single alpha=.05 beta-binomial mixture CS inverted directly; no four-test Bonferroni split",
        "persistent_case_cache":str(CACHE),
        "candidate_priors":list(PRIOR_SETS.keys()),
        "candidate_gammas":GAMMAS,
        "candidate_switches":SWITCHES,
        "selected_prior":str(best["prior"]),
        "selected_gamma":float(best["gamma"]),
        "selected_switch":float(best["switch"]),
        "perfect_configuration_found":bool(perfect),
        "deployment_correct":int(best["correct"]),
        "deployment_total":int(best["total"]),
        "deployment_cells_10of10":int(best["cells_10of10"]),
        "deployment_cell_total":int(best["cell_total"]),
        "deployment_mean_label_fraction":float(best["mean_label_fraction"]),
        "deployment_median_label_fraction":float(best["median_label_fraction"]),
        "deployment_max_label_fraction":float(best["max_label_fraction"]),
        "deployment_zero_labels":int(best["zero_labels"]),
        "deployment_le_1pct":int(best["le_1pct"]),
        "deployment_le_5pct":int(best["le_5pct"]),
        "deployment_le_10pct":int(best["le_10pct"]),
        "electricity_mean_label_fraction":float(best["electricity_mean"]),
        "electricity_max_label_fraction":float(best["electricity_max"]),
        "gas_mean_label_fraction":float(best["gas_mean"]),
        "gas_max_label_fraction":float(best["gas_max"]),
        "burst_accepts_electricity_gas":int(best["burst_accepts_tail"]),
        "fallbacks_electricity_gas":int(best["fallbacks_tail"]),
        "max_certified_worst_case_regret":float(best["max_certified_regret"]),
        "baseline_beta_facc_mean_label_fraction":0.04891277194046876,
        "baseline_beta_facc_max_label_fraction":0.9493694829760404,
        "baseline_electricity_mean":0.122571,
        "baseline_gas_mean":0.074465,
        "umar_changed":False,
        "H02_changed":False,
        "resource_endpoints_changed":False,
        "exact_layer_changed":False,
        "next_step":
            "If perfect and materially more efficient, freeze. Statistical sweeps can then be rerun cache-only without XGBoost.",
    }
    (OUT/"BURST_FINAL_ASSESSMENT.json").write_text(
        json.dumps(assessment,indent=2,sort_keys=True)
    )

    review=PROJECT/"DTRAR_ASPIRE_BURST_FACC_TailSolver_GPU_Review.zip"
    with zipfile.ZipFile(review,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file():
                z.write(p,p.name)
        z.write(Path(__file__),Path(__file__).name)
        z.write(CACHE/"MANIFEST.json","BURST_CACHE_MANIFEST.json")

    print("\n=== ASPIRE-BURST-FACC FINAL ASSESSMENT ===")
    print(json.dumps(assessment,indent=2,sort_keys=True))
    print("\n=== PER-DATASET ===")
    print(per.to_string(index=False))
    print(f"\nBURST_COMPLETE review={review.name}",flush=True)

if __name__=="__main__":
    main()
