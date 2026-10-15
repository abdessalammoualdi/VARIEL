#!/usr/bin/env python3
"""
CPPI-GREG TERMINAL CLOSURE
==========================

Purpose
-------
Touch ONLY the 43 SHARP-unresolved decisions (42 unique dataset/seed/profile
cases). Preserve the 1037 already operationally certified decisions exactly.

Root cause found from SHARP
---------------------------
The remaining problem is BOTH:
  (a) interval width; and
  (b) point-estimation error.

Among the 43 SHARP-unresolved rows, the SHARP point estimate lies on the
correct side of the resource threshold only 29 times. Therefore merely
shrinking the same SHARP standard error cannot close the system.

Final idea
----------
Use the entire <=10% label contract for BOTH prediction and debiasing via
cross-fitting, rather than a .5% pilot predictor plus a largely separate audit.

This is a finite-population, stratified adaptation of cross-prediction-powered
mean inference.

For each unresolved seed/profile case:
  1. Select at most floor(.10*N) labels by a predeclared unlabeled Brier-
     geometry stratified design.
  2. Split those labels into K stratified folds.
  3. For each fold, train a predictor on the OTHER labeled folds.
  4. Predict:
       - every population row;
       - the held-out labeled fold.
  5. Form the cross-prediction estimate:

       mu_hat =
         mean_population( average_fold_prediction )
         + sum_h (N_h/N) * mean_{i in S_h}(y_i - pred_oof_i).

  6. Estimate finite-population residual variance stratum by stratum:

       V_res =
         sum_h (N_h/N)^2 (1-n_h/N_h) s_h^2 / n_h.

     A small cross-prediction model term Var(pred_bar)/N is included.

  7. Use one-sided Student-t bounds to certify exact or delta-optimal actions.

Predictor families
------------------
Global development sweep (same variant for every unresolved case):
  - RIDGE cross-prediction, alpha in {0.1, 1, 10}
  - XGB regression cross-prediction
  - XGB class-probability -> expected classwise Brier-gap cross-prediction
  - fixed 50/50 XGB-regression / XGB-class expectation blend

Two label-free stratification allocations:
  - proportional
  - geometry-range (N_h * R_h)

No per-case outcome-based router is used.  The selected global variant must
satisfy the full terminal contract on the six-domain DEVELOPMENT evidence.

Why the classifier proxy is principled
---------------------------------------
The realized paired Brier gap is z_i(Y_i).  If pi_i(k) approximates the
conditional class probability,

    E[z_i(Y_i)|x_i] = sum_k pi_i(k) z_i(k).

The classifier proxy therefore models the correct conditional expectation
rather than treating the realized Brier gap as an arbitrary continuous target.

Cross-fitting
-------------
Every labeled residual uses a prediction from a model that did not train on
that label. This removes the split-pilot waste while preventing direct
self-fitting leakage into the debiasing residual.

Hard contract
-------------
No unlimited fallback.

A selected variant must have:
    operational_valid = 1080 / 1080
    operational_cells_10of10 = 108 / 108
    unresolved = 0
    max true operational regret <= .001
    max certified regret <= .001
    max actual label fraction <= .10

All six current domains are DEVELOPMENT evidence for selecting this terminal
resource estimator. If the contract is achieved, freeze and confirm on a fresh
stream before making an unseen-domain claim.
"""

from __future__ import annotations
from pathlib import Path
import os, math, json, hashlib, zipfile
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor, XGBClassifier

HERE = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get(
    "DTRAR_PROJECT",
    str(Path.home()/"Documents"/"DTRA_Start"/"DTRA")
)).resolve()

EG_CACHE = PROJECT/"artifacts"/"dtrar_burst_endpoint_case_cache_v1"
TAIL_CACHE = PROJECT/"artifacts"/"dtrar_terminal_tail_case_cache_v1"
CASE_CACHE = PROJECT/"artifacts"/"dtrar_cppigreg_case_cache_v1"
CASE_CACHE.mkdir(parents=True,exist_ok=True)
OUT = PROJECT/"artifacts"/"dtrar_cppigreg_terminal"
OUT.mkdir(parents=True,exist_ok=True)

CASE_RESULT_CACHE = PROJECT/"artifacts"/"dtrar_cppigreg_case_results_v2"
CASE_RESULT_CACHE.mkdir(parents=True,exist_ok=True)
CASE_RESULT_CACHE_VERSION = "cppigreg_stratofix_v2"

def cert_cache_path(allocation,variant,key):
    ds,seed,profile=key
    d=CASE_RESULT_CACHE/allocation/variant/ds/f"seed_{seed}"
    d.mkdir(parents=True,exist_ok=True)
    return d/f"{profile}.json"

def load_cached_cert(allocation,variant,key):
    p=cert_cache_path(allocation,variant,key)
    if not p.exists() or p.stat().st_size<=0:
        return None
    try:
        obj=json.loads(p.read_text())
        if obj.get("cache_version")!=CASE_RESULT_CACHE_VERSION:
            return None
        cert=obj.get("cert")
        required={"estimate","se","df","lower","upper","labels",
                  "label_fraction","min_stratum_n"}
        if not isinstance(cert,dict) or not required.issubset(cert):
            return None
        return cert
    except Exception:
        return None

def save_cached_cert(allocation,variant,key,cert):
    p=cert_cache_path(allocation,variant,key)
    tmp=p.with_suffix(".tmp")
    payload={
        "cache_version":CASE_RESULT_CACHE_VERSION,
        "allocation":allocation,
        "variant":variant,
        "dataset":key[0],
        "seed":int(key[1]),
        "profile":key[2],
        "cert":cert,
    }
    tmp.write_text(json.dumps(payload,sort_keys=True))
    os.replace(tmp,p)

BASELINE = pd.read_csv(HERE/"SHARP_SELECTED_ALL_SIX.csv")
UNRESOLVED0 = pd.read_csv(HERE/"SHARP_UNRESOLVED_CASES.csv")

DELTA = .001
BUDGET = .05
MAX_LABEL_FRAC = .10
K_FOLDS = 5
N_STRATA = 6
ALPHA = .05

ALLOCATIONS = ["proportional","geometry"]
VARIANTS = [
    "ridge_a0.1",
    "ridge_a1",
    "ridge_a10",
    "xgb_reg",
    "xgb_cls",
    "xgb_blend",
]

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

def feature_matrix(zclass):
    z=np.asarray(zclass,float)
    lo=z.min(axis=1)
    hi=z.max(axis=1)
    mid=.5*(lo+hi)
    rng=hi-lo
    mean=z.mean(axis=1)
    std=z.std(axis=1)
    absmean=np.mean(np.abs(z),axis=1)
    q25=np.quantile(z,.25,axis=1)
    q75=np.quantile(z,.75,axis=1)
    return np.column_stack([
        z,mean,std,lo,hi,mid,rng,absmean,q25,q75
    ]).astype(np.float32)

def geometry_strata(zclass):
    z=np.asarray(zclass,float)
    lo=z.min(axis=1)
    hi=z.max(axis=1)
    half=.5*(hi-lo)
    mid=.5*(hi+lo)
    hr=pd.Series(half).rank(method="first",pct=True).to_numpy()
    mr=pd.Series(mid).rank(method="first",pct=True).to_numpy()
    score=.72*hr+.28*mr
    edges=np.quantile(score,np.linspace(0,1,N_STRATA+1))
    edges[0]-=1e-12
    edges[-1]+=1e-12
    return np.digitize(score,edges[1:-1],right=False).astype(int)

def alloc_counts(total,sizes,weights,minimum_per_nonempty=0):
    """
    Integer allocation with an explicit per-nonempty-stratum lower bound.

    For CPPI-GREG we require >=2 sampled labels per nonempty stratum so the
    design variance s_h^2 is estimable.  This lower bound is enforced BEFORE
    geometry/proportional mass is distributed and never exceeds the global
    label budget.
    """
    sizes=np.asarray(sizes,int)
    total=max(0,min(int(total),int(sizes.sum())))
    w=np.asarray(weights,float)
    w=np.where(sizes>0,np.maximum(w,0.),0.)

    nonempty=np.where(sizes>0)[0]
    lower=np.zeros_like(sizes)
    if minimum_per_nonempty>0:
        lower[nonempty]=np.minimum(
            int(minimum_per_nonempty),
            sizes[nonempty]
        )

    must(
        int(lower.sum())<=total,
        f"10% budget cannot support minimum stratum allocation: "
        f"need={int(lower.sum())}, budget={total}"
    )

    n=lower.copy()
    remaining=total-int(n.sum())
    if remaining<=0:
        return n

    capacity=sizes-n
    w=np.where(capacity>0,w,0.0)
    if w.sum()<=0:
        w=capacity.astype(float)

    raw=np.zeros_like(w,dtype=float)
    if w.sum()>0:
        raw=remaining*w/w.sum()

    add=np.minimum(np.floor(raw).astype(int),capacity)
    n+=add

    while n.sum()<total:
        c=np.where(n<sizes)[0]
        if not len(c):
            break
        deficit=raw[c]-(n[c]-lower[c])
        h=c[np.argmax(deficit)]
        n[h]+=1

    while n.sum()>total:
        c=np.where(n>lower)[0]
        if not len(c):
            break
        excess=(n[c]-lower[c])-raw[c]
        h=c[np.argmax(excess)]
        n[h]-=1

    must(int(n.sum())==total,
         f"Allocation mismatch: got={int(n.sum())}, target={total}")
    must(np.all(n[nonempty]>=np.minimum(
        int(minimum_per_nonempty),sizes[nonempty]
    )), "Minimum stratum allocation violated")
    return n

def select_sample(case,allocation):
    N=case["N"]
    z=case["zclass"]
    strata=geometry_strata(z)
    H=int(strata.max())+1
    sizes=np.array([(strata==h).sum() for h in range(H)],int)

    # FLOOR guarantees the realized label fraction never exceeds 10%.
    total=max(H,int(math.floor(MAX_LABEL_FRAC*N)))
    total=min(total,N)

    lo=z.min(axis=1)
    hi=z.max(axis=1)
    rr=hi-lo

    if allocation=="proportional":
        weights=sizes.astype(float)
    elif allocation=="geometry":
        weights=np.array([
            sizes[h]*(float(rr[strata==h].max()) if sizes[h] else 0.)
            for h in range(H)
        ],float)
    else:
        raise RuntimeError(allocation)

    counts=alloc_counts(total,sizes,weights,minimum_per_nonempty=2)

    selected=[]
    folds=np.full(N,-1,int)

    for h in range(H):
        ix=np.where(strata==h)[0]
        rng=np.random.default_rng(
            stable_seed("cppi_sample",case["dataset"],case["seed"],
                        case["profile"],allocation,h)
        )
        perm=rng.permutation(ix)
        take=perm[:counts[h]]
        selected.extend(map(int,take))

        # Stratified round-robin fold assignment with a rotating stratum
        # offset.  This prevents small strata from always populating only
        # folds 0/1 and makes empty global cross-fit folds much less likely.
        offset=h%K_FOLDS
        for j,ii in enumerate(take):
            folds[int(ii)]=(offset+j)%K_FOLDS

    sample=np.asarray(sorted(set(selected)),int)
    must(len(sample)<=math.floor(MAX_LABEL_FRAC*N),
         "10% label contract exceeded")
    must(np.all(folds[sample]>=0),"Unassigned fold")

    # Variance-estimation and cross-fitting integrity checks.
    for h in range(H):
        nh=int(np.sum(strata[sample]==h))
        Nh=int(sizes[h])
        if Nh>0:
            must(
                nh>=min(2,Nh),
                f"Stratum minimum failed h={h}: sampled={nh}, population={Nh}"
            )

    fold_counts=np.array([
        int(np.sum(folds[sample]==k))
        for k in range(K_FOLDS)
    ])
    must(
        np.all(fold_counts>0),
        f"Empty cross-fit fold after stratification: {fold_counts.tolist()}"
    )

    return sample,folds,strata

def recover_classes(ztrue,zclass):
    # ztrue is exactly one of the classwise possible paired Brier gaps.
    d=np.abs(zclass-ztrue[:,None])
    y=np.argmin(d,axis=1).astype(int)
    err=d[np.arange(len(y)),y]
    must(float(np.max(err))<=1e-8,
         f"Could not recover realized class; max error={float(np.max(err))}")
    return y

def standardized_features(X):
    mu=X.mean(axis=0)
    sd=X.std(axis=0)
    sd=np.where(sd>1e-8,sd,1.0)
    return ((X-mu)/sd).astype(np.float32)

def fit_predict_fold(case,X,yclass,train,variant):
    z=case["zclass"]
    y=case["ztrue"]

    if variant.startswith("ridge_a"):
        a=float(variant.split("a",1)[1])
        model=Ridge(alpha=a,fit_intercept=True)
        model.fit(X[train],y[train])
        return np.asarray(model.predict(X),float)

    if variant in {"xgb_reg","xgb_blend"}:
        reg=XGBRegressor(
            objective="reg:squarederror",
            n_estimators=96,
            max_depth=3,
            learning_rate=.05,
            min_child_weight=2,
            subsample=.9,
            colsample_bytree=1.0,
            reg_lambda=10.0,
            tree_method="hist",
            device="cuda",
            n_jobs=1,
            random_state=stable_seed(
                "cppi_reg",case["dataset"],case["seed"],case["profile"],
                len(train),int(train.sum()) if hasattr(train,"sum") else 0
            ),
        )
        reg.fit(X[train],y[train],verbose=False)
        preg=np.asarray(reg.predict(X),float)
        if variant=="xgb_reg":
            return preg

    if variant in {"xgb_cls","xgb_blend"}:
        classes=np.unique(yclass[train])
        if len(classes)<2:
            # Degenerate fold: class expectation under observed class only.
            pcls=z[:,int(classes[0])].astype(float)
        else:
            cls=XGBClassifier(
                objective="multi:softprob" if case["n_classes"]>2
                    else "binary:logistic",
                n_estimators=96,
                max_depth=3,
                learning_rate=.05,
                min_child_weight=2,
                subsample=.9,
                colsample_bytree=1.0,
                reg_lambda=10.0,
                tree_method="hist",
                device="cuda",
                n_jobs=1,
                eval_metric="mlogloss" if case["n_classes"]>2 else "logloss",
                random_state=stable_seed(
                    "cppi_cls",case["dataset"],case["seed"],case["profile"],
                    len(train),int(train.sum()) if hasattr(train,"sum") else 0
                ),
            )
            cls.fit(X[train],yclass[train],verbose=False)
            proba=np.asarray(cls.predict_proba(X),float)
            pcls=np.zeros(case["N"],float)
            for col,c in enumerate(cls.classes_):
                pcls += proba[:,col]*z[:,int(c)]
        if variant=="xgb_cls":
            return pcls

    if variant=="xgb_blend":
        return .5*preg+.5*pcls

    raise RuntimeError(variant)

def case_estimate(case,allocation,variant):
    N=case["N"]
    X=standardized_features(feature_matrix(case["zclass"]))
    yclass=recover_classes(case["ztrue"],case["zclass"])
    sample,folds,strata=select_sample(case,allocation)

    # Cross-fitted sample predictions and average full-population predictions.
    oof=np.full(N,np.nan,float)
    pop_preds=[]

    for k in range(K_FOLDS):
        held=sample[folds[sample]==k]
        train=sample[folds[sample]!=k]
        must(len(held)>0 and len(train)>0,
             f"Empty cross-fit fold {case['dataset']}/{case['seed']}/{case['profile']}/{k}")
        pred=fit_predict_fold(case,X,yclass,train,variant)
        pop_preds.append(pred)
        oof[held]=pred[held]

    must(np.isfinite(oof[sample]).all(),"Missing OOF predictions")
    pop_pred=np.mean(np.vstack(pop_preds),axis=0)

    # Cross-prediction point estimate.
    mu_pred=float(pop_pred.mean())
    residual=case["ztrue"]-oof

    H=int(strata.max())+1
    correction=0.0
    Vres=0.0
    denom_df=0.0
    min_n=10**9

    for h in range(H):
        pop_h=np.where(strata==h)[0]
        samp_h=sample[strata[sample]==h]
        Nh=len(pop_h); nh=len(samp_h)
        if Nh==0: continue
        must(nh>=2,
             f"Too few labels in stratum {h}: {nh}")
        min_n=min(min_n,nh)
        r=residual[samp_h]
        rbar=float(np.mean(r))
        s2=float(np.var(r,ddof=1))
        W=Nh/N
        correction += W*rbar
        vh=(W**2)*max(0.0,1.0-nh/Nh)*s2/nh
        Vres += vh
        if vh>0 and nh>1:
            denom_df += (vh*vh)/(nh-1)

    est=mu_pred+correction

    # Cross-prediction unlabeled/model term.  This is typically very small
    # because the whole finite target population is predicted.
    Vpred=float(np.var(pop_pred,ddof=1))/N if N>1 else 0.0
    V=Vres+Vpred

    if V<=0:
        se=0.0
        df=1e9
    else:
        se=math.sqrt(V)
        df=(Vres*Vres)/max(denom_df,1e-30) if Vres>0 else 1e9
        df=max(30.0,float(df))

    crit=float(student_t.ppf(1.0-ALPHA,df))
    lower=est-crit*se
    upper=est+crit*se

    return {
        "estimate":float(est),
        "se":float(se),
        "df":float(df),
        "lower":float(lower),
        "upper":float(upper),
        "labels":int(len(sample)),
        "label_fraction":float(len(sample)/N),
        "min_stratum_n":int(min_n),
    }

def decide(cert,theta):
    L=cert["lower"]; U=cert["upper"]

    if L>theta:
        return "full","CPPI_EXACT_FULL",0.0,True
    if U<theta:
        return "sparse","CPPI_EXACT_SPARSE",0.0,True

    rs=max(0.0,U-theta)
    rf=max(0.0,theta-L)

    # Operational preference: if SPARSE is certified within delta, use it.
    if rs<=DELTA+1e-15:
        return "sparse","CPPI_DELTA_SPARSE",float(rs),True
    if rf<=DELTA+1e-15:
        return "full","CPPI_DELTA_FULL",float(rf),True

    # Explicit unresolved. No hidden fallback.
    action="full" if cert["estimate"]>theta else "sparse"
    return action,"CPPI_UNRESOLVED",np.nan,False

def true_regret(action,true_gain,theta):
    m=float(true_gain-theta)
    if action=="full":
        return max(0.0,-m)
    return max(0.0,m)

def main():
    # Prerequisite cache checks.
    for p in [
        EG_CACHE/"MANIFEST.json",
        TAIL_CACHE/"MANIFEST.json",
    ]:
        must(p.exists(),f"Missing prerequisite cache manifest: {p}")
        m=json.loads(p.read_text())
        must(m.get("complete") is True,f"Incomplete cache: {m}")

    keys=(
        UNRESOLVED0[["dataset","seed","profile"]]
        .drop_duplicates()
        .sort_values(["dataset","seed","profile"])
    )
    cases={}
    for r in keys.itertuples(index=False):
        key=(str(r.dataset),int(r.seed),str(r.profile))
        cases[key]=load_case(*key)

    # Compute every case-level certificate ONCE per global variant.
    certs={}
    total_jobs=len(cases)*len(ALLOCATIONS)*len(VARIANTS)
    job=0
    for allocation in ALLOCATIONS:
        for variant in VARIANTS:
            for key,case in cases.items():
                job+=1
                cert=load_cached_cert(allocation,variant,key)
                cache_hit=cert is not None
                if cert is None:
                    cert=case_estimate(case,allocation,variant)
                    save_cached_cert(allocation,variant,key,cert)

                certs[(allocation,variant,key)]=cert
                print(
                    f"CPPI_CASE {job}/{total_jobs} "
                    f"allocation={allocation} variant={variant} "
                    f"dataset={key[0]} seed={key[1]} profile={key[2]} "
                    f"est={cert['estimate']:.6g} se={cert['se']:.6g} "
                    f"cache={'hit' if cache_hit else 'built'}",
                    flush=True
                )

    sweep=[]
    saved={}

    for allocation in ALLOCATIONS:
        for variant in VARIANTS:
            out=[]
            unresolved=0

            for _,br in BASELINE.iterrows():
                base=br.to_dict()

                # Preserve 1037 SHARP-certified decisions exactly.
                if bool(br["operational_valid"]):
                    out.append(base)
                    continue

                key=(str(br["dataset"]),int(br["seed"]),str(br["profile"]))
                case=cases[key]
                cert=certs[(allocation,variant,key)]
                theta=float(br["lambda"])*(1.-BUDGET)

                action,status,creg,ok=decide(cert,theta)
                treg=true_regret(action,case["true_gain"],theta)
                true_best="full" if case["true_gain"]>theta else "sparse"
                exact=bool(action==true_best)
                operational=bool(ok and treg<=DELTA+1e-12)

                if not ok:
                    unresolved+=1

                out.append({
                    **base,
                    "action":action,
                    "true_best":true_best,
                    "empirical_best_or_tied":exact,
                    "label_fraction":cert["label_fraction"],
                    "labels":cert["labels"],
                    "certified_worst_case_regret":
                        float(creg) if np.isfinite(creg) else np.nan,
                    "route":"cppi_greg",
                    "pparc_status":status,
                    "cppi_estimate":cert["estimate"],
                    "cppi_se":cert["se"],
                    "cppi_df":cert["df"],
                    "cppi_lower":cert["lower"],
                    "cppi_upper":cert["upper"],
                    "true_operational_regret":float(treg),
                    "operational_valid":operational,
                    "exact_winner_match":exact,
                })

            df=pd.DataFrame(out)
            must(len(df)==1080,"Row-count mismatch")
            cells=df.groupby(
                ["dataset","profile","lambda"],as_index=False
            ).agg(
                operational_10of10=("operational_valid","all"),
                exact_10of10=("exact_winner_match","all"),
            )

            certvals=pd.to_numeric(
                df["certified_worst_case_regret"],errors="coerce"
            ).dropna()

            rec={
                "allocation":allocation,
                "variant":variant,
                "operational_valid":int(df.operational_valid.sum()),
                "total":int(len(df)),
                "operational_cells_10of10":
                    int(cells.operational_10of10.sum()),
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
                    float(pd.to_numeric(
                        df.true_operational_regret,errors="coerce"
                    ).max()),
                "max_certified_regret":
                    float(certvals.max()) if len(certvals) else np.nan,
            }
            sweep.append(rec)
            saved[(allocation,variant)]=df

            print(
                f"CPPI_VARIANT allocation={allocation} variant={variant} "
                f"operational={rec['operational_valid']}/1080 "
                f"cells={rec['operational_cells_10of10']}/108 "
                f"unresolved={rec['unresolved']} "
                f"exact={rec['exact_winner_matches']}/1080 "
                f"mean={rec['mean_label_fraction']:.6f} "
                f"max={rec['max_label_fraction']:.6f}",
                flush=True
            )

    sw=pd.DataFrame(sweep)

    valid=sw[
        (sw.operational_valid==1080)
        &(sw.operational_cells_10of10==108)
        &(sw.unresolved==0)
        &(sw.max_true_operational_regret<=DELTA+1e-12)
        &(sw.max_certified_regret<=DELTA+1e-12)
        &(sw.max_label_fraction<=MAX_LABEL_FRAC+1e-12)
    ].copy()

    if len(valid):
        valid=valid.sort_values(
            ["mean_label_fraction","exact_winner_matches",
             "max_label_fraction"],
            ascending=[True,False,True]
        )
        best=valid.iloc[0]
        terminal=True
    else:
        best=sw.sort_values(
            ["operational_valid","unresolved",
             "operational_cells_10of10","exact_winner_matches"],
            ascending=[False,True,False,False]
        ).iloc[0]
        terminal=False

    key=(str(best["allocation"]),str(best["variant"]))
    final=saved[key].copy()

    sw["selected"]=(
        (sw.allocation==key[0])
        &(sw.variant==key[1])
    )
    sw.to_csv(OUT/"CPPI_GREG_SWEEP.csv",index=False)
    final.to_csv(OUT/"CPPI_GREG_SELECTED_ALL_SIX.csv",index=False)
    final[~final.operational_valid].to_csv(
        OUT/"CPPI_GREG_UNRESOLVED_CASES.csv",index=False
    )

    cells=final.groupby(
        ["dataset","profile","lambda"],as_index=False
    ).agg(
        operational_valid_seeds=("operational_valid","sum"),
        operational_10of10=("operational_valid","all"),
        exact_winner_seeds=("exact_winner_match","sum"),
        exact_10of10=("exact_winner_match","all"),
        mean_label_fraction=("label_fraction","mean"),
        max_label_fraction=("label_fraction","max"),
        max_true_regret=("true_operational_regret","max"),
    )
    cells.to_csv(OUT/"CPPI_GREG_CELL_SUMMARY.csv",index=False)

    per=final.groupby("dataset",as_index=False).agg(
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
    per.to_csv(OUT/"CPPI_GREG_PER_DATASET.csv",index=False)

    assessment={
        "architecture":
            "CPPI-GREG terminal closure: cross-prediction-powered finite-population resource-gap inference",
        "root_cause":
            "SHARP unresolved cases require both point-estimate correction and narrower uncertainty; SHARP point estimates were on the correct threshold side in only 29/43 failures",
        "labels":
            "at most floor(10%*N) per previously unresolved seed/profile; 1037 SHARP-certified decisions preserved",
        "cross_fitting_folds":K_FOLDS,
        "stratum_minimum_labels":2,
        "fold_assignment":"stratified round-robin with rotating stratum offsets",
        "case_result_cache_version":CASE_RESULT_CACHE_VERSION,
        "allocations_tested":ALLOCATIONS,
        "variants_tested":VARIANTS,
        "selected_allocation":key[0],
        "selected_variant":key[1],
        "terminal_contract_satisfied":bool(terminal),
        "operational_valid":int(best.operational_valid),
        "operational_total":int(best.total),
        "operational_cells_10of10":
            int(best.operational_cells_10of10),
        "cell_total":int(best.cell_total),
        "exact_winner_matches":int(best.exact_winner_matches),
        "exact_cells_10of10":int(best.exact_cells_10of10),
        "unresolved":int(best.unresolved),
        "mean_label_fraction":float(best.mean_label_fraction),
        "median_label_fraction":float(best.median_label_fraction),
        "max_label_fraction":float(best.max_label_fraction),
        "le_1pct":int(best.le_1pct),
        "le_5pct":int(best.le_5pct),
        "le_10pct":int(best.le_10pct),
        "max_true_operational_regret":
            float(best.max_true_operational_regret),
        "max_certified_regret":
            float(best.max_certified_regret)
            if np.isfinite(best.max_certified_regret) else None,
        "endpoint_models_retrained":False,
        "uses_existing_endpoint_caches":True,
        "umar_changed":False,
        "H02_changed":False,
        "canary_core_changed":False,
        "resource_endpoints_changed":False,
        "statistical_note":
            "Cross-prediction confidence intervals are studentized/asymptotic; all six current domains are development evidence for selecting the terminal variant.",
        "stop_rule":
            "If terminal_contract_satisfied=true, freeze this resource controller and validate the frozen rule on a fresh stream. If false, inspect the unresolved CSV; do not add an unlimited fallback.",
    }

    (OUT/"CPPI_GREG_FINAL_ASSESSMENT.json").write_text(
        json.dumps(assessment,indent=2,sort_keys=True)
    )

    review=PROJECT/"DTRAR_ASPIRE_CPPIGREG_TERMINAL_GPU_Review.zip"
    with zipfile.ZipFile(review,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file():
                z.write(p,p.name)
        z.write(Path(__file__),Path(__file__).name)

    print("\n=== CPPI-GREG TERMINAL ASSESSMENT ===")
    print(json.dumps(assessment,indent=2,sort_keys=True))
    print("\n=== PER-DATASET ===")
    print(per.to_string(index=False))
    print(f"\nCPPI_GREG_COMPLETE review={review.name}",flush=True)

if __name__=="__main__":
    main()
