#!/usr/bin/env python3
"""
PPARC: Prediction-Powered Active Resource Certification
=======================================================

Terminal development objective
------------------------------
The completed six-domain evidence shows that the remaining high-label tail is
NOT predominantly a near-tie problem:

    current >10%-label decisions: 106
    within |true margin| <= .001: 19
    outside .001: 87

Thus a pure equivalence controller cannot solve the tail.  The remaining
inefficiency is largely caused by conservative generic uncertainty bounds that
ignore predictable structure in the row-level Brier resource gaps.

PPARC changes only the resource-audit inference layer.

Prediction-powered decomposition
--------------------------------
For each target row i, the label-free classwise paired Brier resource gaps are

    z_i(k), k=1,...,K.

The realized gap is z_i(Y_i).

A small PILOT label sample is used to fit a predictor

    f(z_i(1),...,z_i(K), geometry_i) ~= z_i(Y_i).

For all non-pilot rows define clipped predictions

    zhat_i in [min_k z_i(k), max_k z_i(k)].

Then for the finite target population

    mu
      = [ sum_{i in P} z_i(Y_i)
          + sum_{i notin P} zhat_i
          + sum_{i notin P} (z_i(Y_i)-zhat_i) ] / N.

The first two terms are known after the pilot.
Only the residual population mean must be audited.

Validity does NOT require the predictor to be correct:
the disjoint audit sample estimates and corrects its residual bias.

Active residual audit
---------------------
Rows outside the pilot are divided into label-free Brier-sensitivity strata.
The audit budget is allocated in proportion to

    N_h * residual_range_h,

where residual_range_h is computable without labels because zhat_i is fixed
and z_i(Y_i) is bounded by the row's classwise Brier gap interval.

Within each stratum, audit labels are sampled uniformly WITHOUT replacement.

For each residual stratum, a two-sided empirical Bernstein-Serfling interval is
used.  This is variance-adaptive and exploits the finite population correction.

The per-stratum intervals are union-bounded into a 95% interval for mu.

Resource decision
-----------------
theta = lambda*(1-B), B=.05, delta=.001.

    L > theta              => EXACT FULL
    U < theta              => EXACT SPARSE
    U - theta <= delta     => SPARSE is certified delta-optimal
    theta - L <= delta     => FULL is certified delta-optimal
    otherwise              => unresolved under the label contract

There is NO hidden >10% fallback in the primary efficiency contract.

Development sweep
-----------------
All six known domains are development evidence.

Pilot fractions:
    .005, .01, .02

Total label caps:
    .05, .075, .10

The XGBoost pilot predictor is fit from cached arrays only and uses the selected
CUDA device.  Expensive endpoint models are NEVER rebuilt.

A configuration is terminally acceptable only if:
    operational_valid = 1080/1080
    operational_cells = 108/108
    max true operational regret <= .001
    max certified regret <= .001
    unresolved = 0
    max labels <= 10%

If no configuration satisfies this, the output identifies exactly which cases
remain information-limited under this contract.  We do not hide them behind
another unlimited fallback.
"""

from __future__ import annotations
from pathlib import Path
import os, sys, math, json, hashlib, zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

HERE=Path(__file__).resolve().parent
PROJECT=Path(os.environ.get(
    "DTRAR_PROJECT",
    str(Path.home()/"Documents"/"DTRA_Start"/"DTRA")
)).resolve()

EG_CACHE=PROJECT/"artifacts"/"dtrar_burst_endpoint_case_cache_v1"
TAIL_CACHE=PROJECT/"artifacts"/"dtrar_terminal_tail_case_cache_v1"
OUT=PROJECT/"artifacts"/"dtrar_pparc_terminal"
OUT.mkdir(parents=True,exist_ok=True)

BASELINE=pd.read_csv(HERE/"TERMINAL_EQUIV_SELECTED_ALL_SIX.csv")

TAIL_THRESHOLD=.10
DELTA=.001
BUDGET=.05
ALPHA=.05

PILOTS=[.005,.01,.02]
CAPS=[.05,.075,.10]
N_STRATA=6

KAPPA_EBS=7.0/3.0 + 3.0/math.sqrt(2.0)

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
            "max_sparse_H":float(np.asarray(z["max_sparse_H"]).ravel()[0]),
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
    # Direct classwise values are retained; model is fit per case so K is fixed.
    return np.column_stack([
        z,
        mean,std,lo,hi,med,rng,absmean,
    ]).astype(np.float32)

def geometry_strata(zclass,n_strata=N_STRATA):
    z=np.asarray(zclass,float)
    lo=z.min(axis=1)
    hi=z.max(axis=1)
    half=.5*(hi-lo)
    mid=.5*(hi+lo)

    # One-dimensional score that combines sensitivity and location.
    hrank=pd.Series(half).rank(method="first",pct=True).to_numpy()
    mrank=pd.Series(mid).rank(method="first",pct=True).to_numpy()
    score=.7*hrank+.3*mrank
    edges=np.quantile(score,np.linspace(0,1,n_strata+1))
    edges[0]-=1e-12
    edges[-1]+=1e-12
    strata=np.digitize(score,edges[1:-1],right=False)
    return strata.astype(int)

def proportional_counts(total,sizes,weights=None):
    sizes=np.asarray(sizes,int)
    H=len(sizes)
    total=max(0,min(int(total),int(sizes.sum())))
    if total==0:
        return np.zeros(H,int)

    if weights is None:
        w=sizes.astype(float)
    else:
        w=np.asarray(weights,float)
        w=np.where(sizes>0,np.maximum(w,0.0),0.0)
        if w.sum()<=0:
            w=sizes.astype(float)

    raw=total*w/w.sum()
    n=np.floor(raw).astype(int)
    n=np.minimum(n,sizes)

    # Ensure representation of nonempty strata when budget permits.
    nonempty=np.where(sizes>0)[0]
    if total>=len(nonempty):
        for h in nonempty:
            if n[h]==0:
                n[h]=1

    while n.sum()>total:
        candidates=np.where(n>0)[0]
        h=candidates[np.argmax(n[candidates]-raw[candidates])]
        n[h]-=1

    while n.sum()<total:
        candidates=np.where(n<sizes)[0]
        if not len(candidates):
            break
        deficit=raw[candidates]-n[candidates]
        h=candidates[np.argmax(deficit)]
        n[h]+=1

    return n

def pilot_indices(case,pilot_frac):
    N=case["N"]
    strata=geometry_strata(case["zclass"])
    H=int(strata.max())+1
    sizes=np.array([(strata==h).sum() for h in range(H)],int)
    total=max(H,int(math.ceil(float(pilot_frac)*N)))
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

def fit_predictor(case,pilot):
    X=row_features(case["zclass"])
    y=case["ztrue"]
    lo=case["zclass"].min(axis=1)
    hi=case["zclass"].max(axis=1)

    # Fixed small predictor; no per-case hyperparameter tuning.
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
    return pred

def ebs_radius(sample,Npop,a,b,alpha_two_sided):
    """
    Two-sided empirical Bernstein-Serfling radius.

    Bardenet-Maillard theorem is one-sided with success >=1-5*delta.
    For a two-sided stratum interval with total failure alpha_h:
        delta_internal = alpha_h / 10.
    """
    x=np.asarray(sample,float)
    n=len(x)
    if Npop<=0:
        return 0.0
    if n>=Npop:
        return 0.0
    if n<2:
        return float(b-a)

    shat=float(np.sqrt(np.mean((x-x.mean())**2)))
    if n<=Npop/2:
        rho=1.0-(n-1)/Npop
    else:
        rho=(1.0-n/Npop)*(1.0+1.0/n)
    rho=max(0.0,float(rho))

    delta_internal=max(float(alpha_two_sided)/10.0,1e-15)
    logterm=math.log(1.0/delta_internal)

    rad=(
        shat*math.sqrt(2.0*rho*logterm/n)
        +KAPPA_EBS*float(b-a)*logterm/n
    )
    return min(float(b-a),float(rad))

def evaluate_case(case,pilot_frac,total_cap):
    N=case["N"]
    ztrue=case["ztrue"]
    zclass=case["zclass"]
    lo=zclass.min(axis=1)
    hi=zclass.max(axis=1)

    pilot,strata=pilot_indices(case,pilot_frac)
    pred=fit_predictor(case,pilot)

    pilot_mask=np.zeros(N,bool)
    pilot_mask[pilot]=True
    remaining=np.where(~pilot_mask)[0]

    H=int(strata.max())+1
    total_labels=min(N,int(math.ceil(float(total_cap)*N)))
    audit_budget=max(0,total_labels-len(pilot))

    # Known prediction-powered base.
    base=(
        float(ztrue[pilot].sum())
        +float(pred[remaining].sum())
    )/N

    # Residual uncertainty ranges by stratum determine active allocation.
    rem_by=[]
    sizes=[]
    weights=[]
    residual_bounds=[]
    for h in range(H):
        ix=remaining[strata[remaining]==h]
        rem_by.append(ix)
        sizes.append(len(ix))
        if len(ix):
            a=float(np.min(lo[ix]-pred[ix]))
            b=float(np.max(hi[ix]-pred[ix]))
        else:
            a=b=0.0
        residual_bounds.append((a,b))
        weights.append(len(ix)*max(0.0,b-a))

    counts=proportional_counts(audit_budget,np.asarray(sizes,int),np.asarray(weights,float))

    lower=base
    upper=base
    audit_indices=[]

    nonempty=sum(1 for n in sizes if n>0)
    # union-bound two-sided coverage across residual strata
    alpha_h=ALPHA/max(1,nonempty)

    for h,ix in enumerate(rem_by):
        Nh=len(ix)
        if Nh==0:
            continue

        rng=np.random.default_rng(
            stable_seed("pparc_audit",case["dataset"],case["seed"],case["profile"],h)
        )
        perm=rng.permutation(ix)
        nh=int(counts[h])
        samp=perm[:nh]
        audit_indices.extend(map(int,samp))

        a,b=residual_bounds[h]
        if nh==0:
            rlo,rhi=a,b
        else:
            residual=ztrue[samp]-pred[samp]
            m=float(np.mean(residual))
            rad=ebs_radius(residual,Nh,a,b,alpha_h)
            rlo=max(a,m-rad)
            rhi=min(b,m+rad)

        weight=Nh/N
        lower += weight*rlo
        upper += weight*rhi

    labels=set(map(int,pilot))
    labels.update(audit_indices)

    must(
        len(labels)<=total_labels,
        f"Label contract overflow {case['dataset']}/{case['seed']}/{case['profile']}"
    )

    return {
        "L":float(lower),
        "U":float(upper),
        "labels":labels,
        "label_fraction":float(len(labels)/N),
        "pilot_labels":int(len(pilot)),
        "audit_labels":int(len(audit_indices)),
    }

def true_regret(action,true_gain,theta):
    margin=float(true_gain-theta)
    if action=="full":
        return max(0.0,-margin)
    return max(0.0,margin)

def resource_decision(L,U,theta):
    if L>theta:
        return "full","PPARC_EXACT_FULL",0.0,True
    if U<theta:
        return "sparse","PPARC_EXACT_SPARSE",0.0,True

    reg_sparse=max(0.0,U-theta)
    reg_full=max(0.0,theta-L)

    # Prefer SPARSE whenever it is certified delta-optimal.
    if reg_sparse<=DELTA+1e-15:
        return "sparse","PPARC_DELTA_SPARSE",reg_sparse,True
    if reg_full<=DELTA+1e-15:
        return "full","PPARC_DELTA_FULL",reg_full,True

    # No hidden fallback: this explicitly fails the efficiency contract.
    action="full" if (L+U)/2.0>theta else "sparse"
    return action,"PPARC_UNRESOLVED",np.nan,False

def build_cases():
    tail=BASELINE[BASELINE.label_fraction>TAIL_THRESHOLD+1e-12]
    targets=tail[["dataset","seed","profile"]].drop_duplicates()
    cases={}
    for r in targets.itertuples(index=False):
        key=(str(r.dataset),int(r.seed),str(r.profile))
        cases[key]=load_case(*key)
    return cases

def main():
    # Cache preflight: this run is endpoint-cache-only.
    for p in [
        EG_CACHE/"MANIFEST.json",
        TAIL_CACHE/"MANIFEST.json",
    ]:
        must(p.exists(),f"Missing prerequisite cache manifest: {p}")
        m=json.loads(p.read_text())
        must(m.get("complete") is True,f"Incomplete prerequisite cache: {m}")

    cases=build_cases()

    # Predictor outputs depend only on case + pilot fraction, not resource lambda
    # or total audit cap. Cache these to avoid repeated GPU fits.
    prepared={}
    for pilot_frac in PILOTS:
        for key,case in cases.items():
            pilot,strata=pilot_indices(case,pilot_frac)
            pred=fit_predictor(case,pilot)
            prepared[(pilot_frac,key)]=(pilot,strata,pred)
        print(f"PPARC_PREDICTORS_READY pilot={pilot_frac}",flush=True)

    def eval_prepared(case,pilot_frac,total_cap):
        N=case["N"]
        ztrue=case["ztrue"]
        zclass=case["zclass"]
        lo=zclass.min(axis=1)
        hi=zclass.max(axis=1)

        key=(case["dataset"],case["seed"],case["profile"])
        pilot,strata,pred=prepared[(pilot_frac,key)]

        pilot_mask=np.zeros(N,bool)
        pilot_mask[pilot]=True
        remaining=np.where(~pilot_mask)[0]

        H=int(strata.max())+1
        total_labels=min(N,int(math.ceil(float(total_cap)*N)))
        audit_budget=max(0,total_labels-len(pilot))

        base=(float(ztrue[pilot].sum())+float(pred[remaining].sum()))/N

        rem_by=[]; sizes=[]; weights=[]; rb=[]
        for h in range(H):
            ix=remaining[strata[remaining]==h]
            rem_by.append(ix); sizes.append(len(ix))
            if len(ix):
                a=float(np.min(lo[ix]-pred[ix]))
                b=float(np.max(hi[ix]-pred[ix]))
            else:
                a=b=0.0
            rb.append((a,b))
            weights.append(len(ix)*max(0.0,b-a))

        counts=proportional_counts(
            audit_budget,np.asarray(sizes,int),np.asarray(weights,float)
        )

        lower=base; upper=base
        labels=set(map(int,pilot))
        nonempty=sum(1 for n in sizes if n>0)
        alpha_h=ALPHA/max(1,nonempty)

        for h,ix in enumerate(rem_by):
            Nh=len(ix)
            if Nh==0: continue
            rng=np.random.default_rng(
                stable_seed("pparc_audit",case["dataset"],case["seed"],case["profile"],h)
            )
            perm=rng.permutation(ix)
            nh=int(counts[h]); samp=perm[:nh]
            labels.update(map(int,samp))
            a,b=rb[h]
            if nh==0:
                rlo,rhi=a,b
            else:
                residual=ztrue[samp]-pred[samp]
                m=float(residual.mean())
                rad=ebs_radius(residual,Nh,a,b,alpha_h)
                rlo=max(a,m-rad); rhi=min(b,m+rad)
            w=Nh/N
            lower+=w*rlo; upper+=w*rhi

        return float(lower),float(upper),labels

    sweep=[]
    saved={}

    for pilot_frac in PILOTS:
        for cap in CAPS:
            out=[]
            unresolved=0

            for _,br in BASELINE.iterrows():
                base=br.to_dict()

                # Preserve already-efficient baseline decisions exactly.
                if float(br["label_fraction"])<=TAIL_THRESHOLD+1e-12:
                    out.append({
                        **base,
                        "route":"preserved_baseline",
                        "pparc_status":"NOT_NEEDED",
                        "true_operational_regret":0.0,
                        "operational_valid":bool(br["operational_valid"]),
                        "exact_winner_match":bool(br["exact_winner_match"]),
                    })
                    continue

                key=(str(br["dataset"]),int(br["seed"]),str(br["profile"]))
                case=cases[key]
                L,U,labels=eval_prepared(case,pilot_frac,cap)
                theta=float(br["lambda"])*(1.-BUDGET)
                action,status,cert_reg,certified=resource_decision(L,U,theta)

                treg=true_regret(action,case["true_gain"],theta)
                exact=(action==("full" if case["true_gain"]>theta else "sparse"))
                operational=bool(certified and treg<=DELTA+1e-12)
                if not certified:
                    unresolved+=1

                out.append({
                    **base,
                    "action":action,
                    "true_best":"full" if case["true_gain"]>theta else "sparse",
                    "empirical_best_or_tied":exact,
                    "label_fraction":float(len(labels)/case["N"]),
                    "labels":int(len(labels)),
                    "certified_worst_case_regret":
                        float(cert_reg) if np.isfinite(cert_reg) else np.nan,
                    "route":"prediction_powered",
                    "pparc_status":status,
                    "pparc_lower":L,
                    "pparc_upper":U,
                    "true_operational_regret":float(treg),
                    "operational_valid":operational,
                    "exact_winner_match":exact,
                })

            df=pd.DataFrame(out)
            cells=df.groupby(
                ["dataset","profile","lambda"],as_index=False
            ).agg(
                operational_10of10=("operational_valid","all"),
                exact_10of10=("exact_winner_match","all"),
            )

            cert=df["certified_worst_case_regret"].dropna()
            rec={
                "pilot_fraction":float(pilot_frac),
                "total_cap":float(cap),
                "operational_valid":int(df.operational_valid.sum()),
                "total":int(len(df)),
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
                "max_true_operational_regret":float(df.true_operational_regret.max()),
                "max_certified_regret":
                    float(cert.max()) if len(cert) else np.nan,
            }
            sweep.append(rec)
            saved[(float(pilot_frac),float(cap))]=df

            print(
                f"PPARC_VARIANT pilot={pilot_frac:.3f} cap={cap:.3f} "
                f"operational={rec['operational_valid']}/1080 "
                f"unresolved={rec['unresolved']} "
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
        &(sw.max_label_fraction<=.10+1e-12)
    ].copy()

    if len(valid):
        valid=valid.sort_values(
            ["mean_label_fraction","max_label_fraction","exact_winner_matches",
             "le_5pct","le_1pct"],
            ascending=[True,True,False,False,False]
        )
        best=valid.iloc[0]
        perfect=True
    else:
        best=sw.sort_values(
            ["operational_valid","unresolved","max_label_fraction",
             "mean_label_fraction"],
            ascending=[False,True,True,True]
        ).iloc[0]
        perfect=False

    key=(float(best.pilot_fraction),float(best.total_cap))
    final=saved[key].copy()

    sw["selected"]=(
        np.isclose(sw.pilot_fraction,key[0])
        &np.isclose(sw.total_cap,key[1])
    )
    sw.to_csv(OUT/"PPARC_DEVELOPMENT_SWEEP.csv",index=False)
    final.to_csv(OUT/"PPARC_SELECTED_ALL_SIX.csv",index=False)

    unresolved_df=final[~final.operational_valid].copy()
    unresolved_df.to_csv(OUT/"PPARC_UNRESOLVED_CASES.csv",index=False)

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
    per.to_csv(OUT/"PPARC_PER_DATASET.csv",index=False)

    assessment={
        "architecture":
            "PPARC: Prediction-Powered Active Resource Certification",
        "root_cause":
            "87/106 >10%-label cases are outside delta=.001; the dominant remaining problem is conservative inference, not near-tie information limits",
        "prediction_powered_identity":
            "mu = known pilot contribution + predicted unlabeled contribution + audited residual correction",
        "predictor":
            "fixed small XGBoost regressor on label-free classwise Brier-gap geometry",
        "audit":
            "disjoint stratified sampling without replacement + empirical Bernstein-Serfling residual intervals",
        "candidate_pilot_fractions":PILOTS,
        "candidate_total_caps":CAPS,
        "selected_pilot_fraction":key[0],
        "selected_total_cap":key[1],
        "terminal_contract_satisfied":bool(perfect),
        "operational_valid":int(best.operational_valid),
        "operational_total":int(best.total),
        "operational_cells_10of10":int(best.operational_cells_10of10),
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
        "max_true_operational_regret":float(best.max_true_operational_regret),
        "max_certified_regret":float(best.max_certified_regret)
            if np.isfinite(best.max_certified_regret) else None,
        "baseline_terminal_mean_label_fraction":
            float(BASELINE.label_fraction.mean()),
        "baseline_terminal_max_label_fraction":
            float(BASELINE.label_fraction.max()),
        "endpoint_models_retrained":False,
        "uses_existing_endpoint_caches":True,
        "umar_changed":False,
        "H02_changed":False,
        "canary_core_changed":False,
        "resource_endpoints_changed":False,
        "stop_rule":
            "If terminal_contract_satisfied=true, freeze the six-domain controller. If false, inspect PPARC_UNRESOLVED_CASES.csv; do not introduce an unlimited fallback.",
    }
    (OUT/"PPARC_FINAL_ASSESSMENT.json").write_text(
        json.dumps(assessment,indent=2,sort_keys=True)
    )

    review=PROJECT/"DTRAR_ASPIRE_PPARC_TERMINAL_GPU_Review.zip"
    with zipfile.ZipFile(review,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file():
                z.write(p,p.name)
        z.write(Path(__file__),Path(__file__).name)

    print("\n=== PPARC TERMINAL ASSESSMENT ===")
    print(json.dumps(assessment,indent=2,sort_keys=True))
    print("\n=== PER-DATASET ===")
    print(per.to_string(index=False))
    print(f"\nPPARC_COMPLETE review={review.name}",flush=True)

if __name__=="__main__":
    main()
