#!/usr/bin/env python3
"""
PPARC-SHARP
===========

Prediction-Powered Active Resource Certification with
Studentized Heteroskedastic Adaptive Residual Precision.

This is a TERMINAL CACHE-ONLY repair for the 103 decisions left unresolved by
PPARC.  It does not touch the 977 already-certified decisions.

Why PPARC failed
----------------
PPARC's empirical Bernstein-Serfling interval is nonasymptotic, but the
worst-case residual-range term dominates the interval in this problem.
The selected unresolved intervals have mean width ~0.86, grossly larger than
the actual resource margins that need to be resolved.

SHARP keeps the same prediction-powered decomposition, but replaces only the
tail inference engine.

Stage 1 (already established)
-----------------------------
PPARC pilot fraction = .005
PPARC first-stage total labels = .05

If PPARC already certified the resource action, preserve it byte-for-byte.

Stage 2 (only the 103 unresolved decisions)
-------------------------------------------
1) Reconstruct the exact same disjoint pilot and first-stage residual audit.
2) Use first-stage residual sample variances to estimate which strata actually
   contain uncertainty.
3) Spend ONLY the remaining labels up to the 10% hard contract using
   residual-variance Neyman allocation:

       n_h approximately proportional to N_h * s_h.

   Existing first-stage labels are lower bounds and are reused.
4) Sample additional labels uniformly without replacement inside each stratum.
5) Estimate the finite-population residual correction with a stratified
   generalized-regression / prediction-powered estimator.
6) Estimate its design variance with the finite-population correction:

       Vhat = sum_h (N_h/N)^2 (1-n_h/N_h) s_h^2 / n_h.

7) Use Satterthwaite degrees of freedom:

       nu = Vhat^2 / sum_h v_h^2/(n_h-1).

8) Use ACTION-SPECIFIC one-sided 95% studentized bounds.
   No Bonferroni split is needed between FULL/SPARSE because a wrong
   operational action corresponds to rejecting the one true one-sided null.

Resource rules
--------------
theta=lambda*(1-.05), delta=.001.

EXACT FULL:
    lower_95 > theta

EXACT SPARSE:
    upper_95 < theta

DELTA-SPARSE:
    upper_95 <= theta+delta

DELTA-FULL:
    lower_95 >= theta-delta

When both actions are delta-valid, choose SPARSE.

This is a design-based/model-assisted finite-population inference stage.
It is asymptotic/studentized rather than a worst-case nonasymptotic
concentration bound.  The program reports all degrees of freedom and refuses
to certify if the design diagnostics are inadequate.

Hard contract
-------------
No unlimited fallback exists.

Required:
    operational_valid = 1080/1080
    operational_cells = 108/108
    unresolved = 0
    max true operational regret <= .001
    max labels <= .10

Runtime
-------
All endpoint probabilities/gaps are already cached.
Only small per-case pilot XGBoost regressors are fit on GPU and cached.
No base XGBoost, FULL branch, calibration, or conformal reconstruction occurs.
"""

from __future__ import annotations
from pathlib import Path
import os, sys, math, json, hashlib, zipfile
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from xgboost import XGBRegressor

HERE=Path(__file__).resolve().parent
PROJECT=Path(os.environ.get(
    "DTRAR_PROJECT",
    str(Path.home()/"Documents"/"DTRA_Start"/"DTRA")
)).resolve()

EG_CACHE=PROJECT/"artifacts"/"dtrar_burst_endpoint_case_cache_v1"
TAIL_CACHE=PROJECT/"artifacts"/"dtrar_terminal_tail_case_cache_v1"
PRED_CACHE=PROJECT/"artifacts"/"dtrar_pparc_sharp_predictor_cache_v1"
PRED_CACHE.mkdir(parents=True,exist_ok=True)
OUT=PROJECT/"artifacts"/"dtrar_pparc_sharp_terminal"
OUT.mkdir(parents=True,exist_ok=True)

BASELINE=pd.read_csv(HERE/"PPARC_SELECTED_ALL_SIX.csv")
UNRESOLVED0=pd.read_csv(HERE/"PPARC_UNRESOLVED_CASES.csv")

PILOT_FRAC=.005
STAGE1_TOTAL=.05
FINAL_CAP=.10
DELTA=.001
BUDGET=.05
ALPHA_ONE_SIDED=.05
N_STRATA=6

def must(c,m):
    if not c:
        raise RuntimeError(m)

def stable_seed(*parts):
    h=hashlib.sha256("|".join(map(str,parts)).encode()).hexdigest()
    return int(h[:8],16)

def case_path(ds,seed,profile):
    if ds in {"electricity","gas"}:
        return EG_CACHE/ds/f"seed_{seed}"/f"{profile}.npz"
    return TAIL_CACHE/ds/f"seed_{seed}"/f"{profile}.npz"

def load_case(ds,seed,profile):
    p=case_path(ds,seed,profile)
    must(p.exists(),f"Missing cached case: {p}")
    with np.load(p,allow_pickle=False) as z:
        return {
            "dataset":str(ds),
            "seed":int(seed),
            "profile":str(profile),
            "ztrue":np.asarray(z["ztrue"],float),
            "zclass":np.asarray(z["zclass"],float),
            "true_gain":float(np.asarray(z["true_gain"]).ravel()[0]),
            "n_classes":int(np.asarray(z["n_classes"]).ravel()[0]),
            "N":int(np.asarray(z["N"]).ravel()[0]),
        }

def row_features(zclass):
    z=np.asarray(zclass,float)
    mean=z.mean(axis=1)
    std=z.std(axis=1)
    lo=z.min(axis=1)
    hi=z.max(axis=1)
    med=np.median(z,axis=1)
    rng=hi-lo
    absmean=np.mean(np.abs(z),axis=1)
    return np.column_stack([z,mean,std,lo,hi,med,rng,absmean]).astype(np.float32)

def geometry_strata(zclass,n_strata=N_STRATA):
    z=np.asarray(zclass,float)
    lo=z.min(axis=1); hi=z.max(axis=1)
    half=.5*(hi-lo); mid=.5*(hi+lo)
    hrank=pd.Series(half).rank(method="first",pct=True).to_numpy()
    mrank=pd.Series(mid).rank(method="first",pct=True).to_numpy()
    score=.7*hrank+.3*mrank
    edges=np.quantile(score,np.linspace(0,1,n_strata+1))
    edges[0]-=1e-12; edges[-1]+=1e-12
    return np.digitize(score,edges[1:-1],right=False).astype(int)

def proportional_counts(total,sizes,weights=None,minimum=None):
    sizes=np.asarray(sizes,int)
    H=len(sizes)
    total=max(0,min(int(total),int(sizes.sum())))
    if minimum is None:
        lower=np.zeros(H,int)
    else:
        lower=np.minimum(np.asarray(minimum,int),sizes)
    must(lower.sum()<=total,"Lower-bound allocation exceeds budget")

    n=lower.copy()
    remaining=total-int(n.sum())
    if remaining<=0:
        return n

    capacity=sizes-n
    if weights is None:
        w=capacity.astype(float)
    else:
        w=np.asarray(weights,float)
        w=np.where(capacity>0,np.maximum(w,0.0),0.0)
        if w.sum()<=0:
            w=capacity.astype(float)

    if w.sum()<=0:
        return n

    raw=remaining*w/w.sum()
    add=np.floor(raw).astype(int)
    add=np.minimum(add,capacity)
    n+=add

    while n.sum()<total:
        cand=np.where(n<sizes)[0]
        if not len(cand): break
        deficit=raw[cand]-(n[cand]-lower[cand])
        h=cand[np.argmax(deficit)]
        n[h]+=1

    while n.sum()>total:
        cand=np.where(n>lower)[0]
        h=cand[np.argmax((n[cand]-lower[cand])-raw[cand])]
        n[h]-=1

    return n

def pilot_indices(case):
    N=case["N"]
    strata=geometry_strata(case["zclass"])
    H=int(strata.max())+1
    sizes=np.array([(strata==h).sum() for h in range(H)],int)
    total=max(H,int(math.ceil(PILOT_FRAC*N)))
    total=min(total,N)
    counts=proportional_counts(total,sizes)
    out=[]
    for h in range(H):
        ix=np.where(strata==h)[0]
        rng=np.random.default_rng(
            stable_seed("pparc_pilot",case["dataset"],case["seed"],case["profile"],h)
        )
        perm=rng.permutation(ix)
        out.extend(map(int,perm[:counts[h]]))
    return np.asarray(sorted(set(out)),int),strata

def predictor_cache_path(case):
    d=PRED_CACHE/case["dataset"]/f"seed_{case['seed']}"
    d.mkdir(parents=True,exist_ok=True)
    return d/f"{case['profile']}.npz"

def get_predictor(case,pilot):
    p=predictor_cache_path(case)
    if p.exists() and p.stat().st_size>0:
        try:
            with np.load(p,allow_pickle=False) as z:
                pred=np.asarray(z["pred"],float)
                cached_pilot=np.asarray(z["pilot"],int)
            if len(pred)==case["N"] and np.array_equal(cached_pilot,pilot):
                return pred
        except Exception:
            pass

    X=row_features(case["zclass"])
    y=case["ztrue"]
    lo=case["zclass"].min(axis=1)
    hi=case["zclass"].max(axis=1)

    model=XGBRegressor(
        objective="reg:squarederror",
        n_estimators=64,
        max_depth=3,
        learning_rate=.05,
        min_child_weight=2,
        subsample=.9,
        colsample_bytree=1.0,
        reg_lambda=10.0,
        reg_alpha=0.0,
        tree_method="hist",
        device="cuda",
        n_jobs=1,
        random_state=stable_seed(
            "pparc_xgb",case["dataset"],case["seed"],case["profile"],len(pilot)
        ),
    )
    model.fit(X[pilot],y[pilot],verbose=False)
    pred=np.asarray(model.predict(X),float)
    pred=np.minimum(np.maximum(pred,lo),hi)

    tmp=p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp,pred=pred,pilot=pilot)
    os.replace(tmp,p)
    return pred

def first_stage_and_extension(case):
    """
    Reconstruct selected PPARC first-stage labels, then extend unresolved
    cases to the 10% hard cap using residual-variance Neyman allocation.
    """
    N=case["N"]
    ztrue=case["ztrue"]; zclass=case["zclass"]
    lo=zclass.min(axis=1); hi=zclass.max(axis=1)

    pilot,strata=pilot_indices(case)
    pred=get_predictor(case,pilot)

    pmask=np.zeros(N,bool); pmask[pilot]=True
    remaining=np.where(~pmask)[0]
    H=int(strata.max())+1

    # Deterministic per-stratum permutations identical to PPARC.
    rem_by=[]; perms=[]; sizes=[]; ranges=[]
    for h in range(H):
        ix=remaining[strata[remaining]==h]
        rem_by.append(ix)
        sizes.append(len(ix))
        rng=np.random.default_rng(
            stable_seed("pparc_audit",case["dataset"],case["seed"],case["profile"],h)
        )
        perms.append(rng.permutation(ix))
        if len(ix):
            a=float(np.min(lo[ix]-pred[ix]))
            b=float(np.max(hi[ix]-pred[ix]))
        else:
            a=b=0.0
        ranges.append(max(0.0,b-a))

    sizes=np.asarray(sizes,int)

    # Stage-1 allocation: exactly the selected PPARC range allocation.
    total1=min(N,int(math.ceil(STAGE1_TOTAL*N)))
    audit1=max(0,total1-len(pilot))
    w1=sizes.astype(float)*np.asarray(ranges,float)
    n1=proportional_counts(audit1,sizes,w1)

    # Estimate residual SDs from the first-stage design.
    sd=np.zeros(H,float)
    pooled=[]
    for h in range(H):
        nh=int(n1[h])
        if nh>=2:
            samp=perms[h][:nh]
            r=ztrue[samp]-pred[samp]
            sd[h]=float(np.std(r,ddof=1))
            pooled.extend(map(float,r))
        elif nh==1:
            r=ztrue[perms[h][:1]]-pred[perms[h][:1]]
            pooled.extend(map(float,r))

    pooled_sd=float(np.std(pooled,ddof=1)) if len(pooled)>=2 else 0.0

    # Stabilize noisy stratum SDs toward pooled SD; zero-variance strata retain
    # a modest floor so no stratum can be starved by a lucky first-stage sample.
    shrink=.25
    floor=max(1e-12,.10*pooled_sd)
    sd_stable=np.maximum(
        floor,
        np.sqrt((1-shrink)*sd*sd + shrink*pooled_sd*pooled_sd)
    )

    # Final 10% audit allocation: Neyman n_h ∝ N_h*s_h, with stage-1
    # allocations as hard lower bounds.
    total2=min(N,int(math.ceil(FINAL_CAP*N)))
    audit2=max(0,total2-len(pilot))
    w2=sizes.astype(float)*sd_stable
    n2=proportional_counts(audit2,sizes,w2,minimum=n1)

    # Model-assisted finite-population point estimate.
    base=(float(ztrue[pilot].sum())+float(pred[remaining].sum()))/N

    correction=0.0
    V=0.0
    components=[]
    labels=set(map(int,pilot))

    for h in range(H):
        Nh=int(sizes[h])
        nh=int(n2[h])
        if Nh==0:
            continue
        must(nh>=2 or Nh<=1,
             f"Insufficient final stratum sample {case['dataset']}/{case['seed']}/{case['profile']}/h{h}: {nh}/{Nh}")

        samp=perms[h][:nh]
        labels.update(map(int,samp))
        r=ztrue[samp]-pred[samp]
        m=float(np.mean(r))
        s2=float(np.var(r,ddof=1)) if nh>=2 else 0.0

        weight=Nh/N
        correction += weight*m

        f=nh/Nh
        vh=(weight**2)*max(0.0,1.0-f)*s2/max(1,nh)
        V+=vh
        components.append((vh,nh))

    est=base+correction

    # Satterthwaite df for heteroskedastic stratified variance components.
    denom=0.0
    for vh,nh in components:
        if nh>1 and vh>0:
            denom += (vh*vh)/(nh-1)
    if V<=0:
        df=1e9
        se=0.0
    else:
        df=(V*V)/max(denom,1e-30)
        se=math.sqrt(V)

    # Require a healthy effective df before calling this tail certificate.
    diagnostic_ok=bool(df>=30 and len(labels)<=total2)

    crit=float(student_t.ppf(1.0-ALPHA_ONE_SIDED,df))
    lower=est-crit*se
    upper=est+crit*se

    return {
        "estimate":float(est),
        "se":float(se),
        "df":float(df),
        "lower":float(lower),
        "upper":float(upper),
        "labels":labels,
        "label_fraction":float(len(labels)/N),
        "pilot_labels":int(len(pilot)),
        "stage1_audit_labels":int(n1.sum()),
        "final_audit_labels":int(n2.sum()),
        "diagnostic_ok":diagnostic_ok,
        "min_final_stratum_n":int(min([n2[h] for h in range(H) if sizes[h]>0])),
        "max_final_stratum_n":int(max([n2[h] for h in range(H) if sizes[h]>0])),
    }

def decide(cert,theta):
    L=cert["lower"];U=cert["upper"]

    if not cert["diagnostic_ok"]:
        return None,"SHARP_DIAGNOSTIC_FAIL",None

    # Exact action-specific one-sided certificates.
    if L>theta:
        return "full","SHARP_EXACT_FULL",0.0
    if U<theta:
        return "sparse","SHARP_EXACT_SPARSE",0.0

    reg_sparse=max(0.0,U-theta)
    reg_full=max(0.0,theta-L)

    # Operational delta certificates. Prefer SPARSE if both are valid.
    if reg_sparse<=DELTA+1e-15:
        return "sparse","SHARP_DELTA_SPARSE",float(reg_sparse)
    if reg_full<=DELTA+1e-15:
        return "full","SHARP_DELTA_FULL",float(reg_full)

    return None,"SHARP_UNRESOLVED",None

def true_regret(action,true_gain,theta):
    if action is None:
        return np.nan
    margin=float(true_gain-theta)
    if action=="full":
        return max(0.0,-margin)
    return max(0.0,margin)

def main():
    # Cache-only preflight.
    for p in [
        EG_CACHE/"MANIFEST.json",
        TAIL_CACHE/"MANIFEST.json",
    ]:
        must(p.exists(),f"Missing prerequisite cache: {p}")
        m=json.loads(p.read_text())
        must(m.get("complete") is True,f"Incomplete cache: {m}")

    unresolved_keys=(
        UNRESOLVED0[["dataset","seed","profile"]]
        .drop_duplicates()
    )
    cases={}
    certs={}

    # Fit/cache predictor and compute ONE final 10% certificate per unresolved
    # seed/profile.  Resource lambda reuses the same certificate.
    for i,r in enumerate(unresolved_keys.itertuples(index=False),1):
        key=(str(r.dataset),int(r.seed),str(r.profile))
        case=load_case(*key)
        cases[key]=case
        certs[key]=first_stage_and_extension(case)
        print(
            f"SHARP_CASE {i}/{len(unresolved_keys)} "
            f"dataset={key[0]} seed={key[1]} profile={key[2]} "
            f"df={certs[key]['df']:.1f} se={certs[key]['se']:.6g}",
            flush=True
        )

    out=[]
    unresolved=0

    for _,br in BASELINE.iterrows():
        base=br.to_dict()

        # Preserve every PPARC-certified row exactly.
        if bool(br["operational_valid"]):
            out.append(base)
            continue

        key=(str(br["dataset"]),int(br["seed"]),str(br["profile"]))
        case=cases[key]
        cert=certs[key]
        theta=float(br["lambda"])*(1.-BUDGET)

        action,status,cert_reg=decide(cert,theta)
        if action is None:
            unresolved+=1
            # Keep explicit unresolved status; no hidden high-cost fallback.
            out.append({
                **base,
                "action":base["action"],
                "label_fraction":cert["label_fraction"],
                "labels":len(cert["labels"]),
                "certified_worst_case_regret":np.nan,
                "route":"pparc_sharp",
                "pparc_status":status,
                "sharp_estimate":cert["estimate"],
                "sharp_se":cert["se"],
                "sharp_df":cert["df"],
                "sharp_lower":cert["lower"],
                "sharp_upper":cert["upper"],
                "true_operational_regret":
                    true_regret(base["action"],case["true_gain"],theta),
                "operational_valid":False,
                "exact_winner_match":
                    bool(base["action"]==("full" if case["true_gain"]>theta else "sparse")),
            })
            continue

        true_best="full" if case["true_gain"]>theta else "sparse"
        treg=true_regret(action,case["true_gain"],theta)
        exact=bool(action==true_best)
        operational=bool(treg<=DELTA+1e-12)

        out.append({
            **base,
            "action":action,
            "true_best":true_best,
            "empirical_best_or_tied":exact,
            "label_fraction":cert["label_fraction"],
            "labels":len(cert["labels"]),
            "certified_worst_case_regret":float(cert_reg),
            "route":"pparc_sharp",
            "pparc_status":status,
            "sharp_estimate":cert["estimate"],
            "sharp_se":cert["se"],
            "sharp_df":cert["df"],
            "sharp_lower":cert["lower"],
            "sharp_upper":cert["upper"],
            "true_operational_regret":float(treg),
            "operational_valid":operational,
            "exact_winner_match":exact,
        })

    df=pd.DataFrame(out)
    must(len(df)==1080,"Row count mismatch")

    cells=df.groupby(
        ["dataset","profile","lambda"],as_index=False
    ).agg(
        operational_10of10=("operational_valid","all"),
        exact_10of10=("exact_winner_match","all"),
    )

    certvals=pd.to_numeric(df["certified_worst_case_regret"],errors="coerce").dropna()

    assessment={
        "architecture":
            "PPARC-SHARP: prediction-powered residual estimator + residual-variance Neyman extension + studentized finite-population GREG tail certification",
        "primary_bottleneck_fixed":
            "replace worst-case residual-range concentration with observed residual-variance design inference only for PPARC-unresolved cases",
        "stage1_preserved":
            "all 977 PPARC-certified decisions preserved byte-for-byte",
        "pilot_fraction":PILOT_FRAC,
        "stage1_total_fraction":STAGE1_TOTAL,
        "final_hard_cap":FINAL_CAP,
        "tail_interval":
            "stratified design-based finite-population variance + Satterthwaite t, action-specific one-sided 95% bounds",
        "operational_valid":int(df.operational_valid.sum()),
        "operational_total":int(len(df)),
        "operational_cells_10of10":int(cells.operational_10of10.sum()),
        "cell_total":int(len(cells)),
        "exact_winner_matches":int(df.exact_winner_match.sum()),
        "exact_cells_10of10":int(cells.exact_10of10.sum()),
        "unresolved":int(unresolved),
        "mean_label_fraction":float(df.label_fraction.mean()),
        "median_label_fraction":float(df.label_fraction.median()),
        "max_label_fraction":float(df.label_fraction.max()),
        "le_1pct":int((df.label_fraction<=.01+1e-12).sum()),
        "le_5pct":int((df.label_fraction<=.05+1e-12).sum()),
        "le_10pct":int((df.label_fraction<=.10+1e-12).sum()),
        "max_true_operational_regret":
            float(pd.to_numeric(df.true_operational_regret,errors="coerce").max()),
        "max_certified_regret":
            float(certvals.max()) if len(certvals) else None,
        "terminal_contract_satisfied":bool(
            df.operational_valid.all()
            and unresolved==0
            and cells.operational_10of10.all()
            and float(df.label_fraction.max())<=.10+1e-12
            and float(pd.to_numeric(df.true_operational_regret,errors="coerce").max())<=DELTA+1e-12
            and (len(certvals)==0 or float(certvals.max())<=DELTA+1e-12)
        ),
        "endpoint_models_retrained":False,
        "uses_existing_endpoint_caches":True,
        "umar_changed":False,
        "H02_changed":False,
        "canary_core_changed":False,
        "resource_endpoints_changed":False,
        "important_statistical_note":
            "SHARP tail certification is studentized design-based finite-population inference, not the worst-case nonasymptotic Bernstein-Serfling certificate used in PPARC stage 1.",
        "stop_rule":
            "If terminal_contract_satisfied=true, freeze. If false, inspect SHARP_UNRESOLVED_CASES.csv; do not add an unlimited fallback.",
    }

    df.to_csv(OUT/"SHARP_SELECTED_ALL_SIX.csv",index=False)
    df[~df.operational_valid].to_csv(
        OUT/"SHARP_UNRESOLVED_CASES.csv",index=False
    )

    per=df.groupby("dataset",as_index=False).agg(
        operational_valid=("operational_valid","sum"),
        decisions_total=("operational_valid","size"),
        exact_winner_matches=("exact_winner_match","sum"),
        mean_label_fraction=("label_fraction","mean"),
        median_label_fraction=("label_fraction","median"),
        max_label_fraction=("label_fraction","max"),
        le_1pct=("label_fraction",lambda x:int((x<=.01+1e-12).sum())),
        le_5pct=("label_fraction",lambda x:int((x<=.05+1e-12).sum())),
        le_10pct=("label_fraction",lambda x:int((x<=.10+1e-12).sum())),
        max_true_regret=("true_operational_regret","max"),
    )
    per.to_csv(OUT/"SHARP_PER_DATASET.csv",index=False)

    (OUT/"SHARP_FINAL_ASSESSMENT.json").write_text(
        json.dumps(assessment,indent=2,sort_keys=True)
    )

    review=PROJECT/"DTRAR_ASPIRE_PPARC_SHARP_TERMINAL_GPU_Review.zip"
    with zipfile.ZipFile(review,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file():
                z.write(p,p.name)
        z.write(Path(__file__),Path(__file__).name)

    print("\n=== PPARC-SHARP TERMINAL ASSESSMENT ===")
    print(json.dumps(assessment,indent=2,sort_keys=True))
    print("\n=== PER-DATASET ===")
    print(per.to_string(index=False))
    print(f"\nSHARP_COMPLETE review={review.name}",flush=True)

if __name__=="__main__":
    main()
