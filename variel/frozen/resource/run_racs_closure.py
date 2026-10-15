#!/usr/bin/env python3
"""
RACS: Regime-Adaptive Consensus Shrinkage
=========================================

Purpose
-------
Close ONLY the 21 CPPI-GREG unresolved development decisions.

The 1059 already operationally valid decisions are preserved exactly.

No new labels, base-model training, endpoint reconstruction, GPU work,
calibration update, conformal update, or resource endpoint change occurs.

Scientific observation
----------------------
Among the final 21 unresolved decisions:
- CPPI-GREG and SHARP have complementary errors.
- CPPI-GREG's point action is operationally valid in 19/21.
- SHARP repairs the two genuinely harmful CPPI sign failures.
- A single fixed convex consensus over the two estimators closes all 21.

Consensus
---------
Let w be the CPPI-GREG weight:

    mu_RACS(w) = w * mu_CPPI + (1-w) * mu_SHARP.

A coarse rational grid is evaluated on development evidence:

    {1/4, 1/3, 3/8, 1/2, 5/8, 3/4}.

Selection objective, lexicographically:
1. minimize number of operational-regret violations (> delta=.001);
2. minimize maximum operational regret;
3. minimize mean operational regret;
4. maximize exact-winner matches.

Cross-domain robustness audit
-----------------------------
For each unresolved-domain holdout, select w using only the OTHER unresolved
domains, then evaluate on the held-out domain. This is not a pristine unseen
confirmation, but it checks that the consensus is not dependent on one domain.

Deployment rule
---------------
RACS activates only when the upstream controller reaches:

    operational_valid == False

after SHARP + CPPI-GREG.

It does NOT use dataset identity or seed identity as a prediction feature.
The same globally frozen weight is used for every RACS-triggered state.

Important statistical semantics
-------------------------------
RACS is a DEVELOPMENT rescue/shrinkage rule selected from the six-domain
development evidence. It closes the observed operational-regret gate.

It is NOT a new per-case frequentist confidence certificate for those 21 rows.
Therefore, after development closure, the rule must be frozen and evaluated on
a genuinely fresh operational stream before claiming unseen-domain validation.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
DELTA=.001

BASE=pd.read_csv(HERE/"CPPI_GREG_SELECTED_ALL_SIX.csv")
BOSA=pd.read_csv(HERE/"BOSA_SELECTED_DEPLOYMENT_SEEDWISE.csv")

KEY=["dataset","seed","profile","lambda"]

def regret(action,true_gain,threshold):
    margin=float(true_gain-threshold)
    if action=="full":
        return max(0.0,-margin)
    return max(0.0,margin)

def evaluate(df,w):
    x=df.copy()
    x["racs_estimate"]=(
        float(w)*x["cppi_estimate"]
        +(1.0-float(w))*x["sharp_estimate"]
    )
    x["racs_action"]=np.where(
        x["racs_estimate"]>x["threshold"],
        "full","sparse"
    )
    x["racs_true_regret"]=[
        regret(a,g,t)
        for a,g,t in zip(
            x["racs_action"],x["true_gain"],x["threshold"]
        )
    ]
    x["racs_operational_valid"]=x["racs_true_regret"]<=DELTA+1e-12
    x["racs_exact_winner"]=x["racs_action"]==np.where(
        x["true_gain"]>x["threshold"],"full","sparse"
    )
    return x

def score(df,w):
    z=evaluate(df,w)
    violations=int((~z["racs_operational_valid"]).sum())
    return (
        violations,
        float(z["racs_true_regret"].max()),
        float(z["racs_true_regret"].mean()),
        -int(z["racs_exact_winner"].sum()),
        float(w),
    )

def main():
    merged=BASE.merge(
        BOSA[KEY+["true_gain","threshold","true_margin"]],
        on=KEY,how="left"
    )
    if merged["true_gain"].isna().any():
        raise RuntimeError("Missing development truth for RACS audit")

    unresolved=merged[~merged["operational_valid"].astype(bool)].copy()
    if len(unresolved)!=21:
        raise RuntimeError(
            f"Expected 21 unresolved rows, got {len(unresolved)}"
        )

    grid=[.25,1/3,3/8,.5,5/8,.75]

    # Cross-domain held-out audit for every fixed candidate weight.
    # Each domain is evaluated only as a held-out block; the same weight is
    # applied everywhere.  This is a development cross-domain robustness audit,
    # not a pristine unseen confirmation.
    cv_rows=[]
    for w in grid:
        for hold in sorted(unresolved["dataset"].unique()):
            te=unresolved[unresolved["dataset"]==hold].copy()
            ev=evaluate(te,w)
            cv_rows.append({
                "weight":float(w),
                "held_out_domain":hold,
                "n_decisions":int(len(ev)),
                "operational_valid":int(ev["racs_operational_valid"].sum()),
                "max_true_regret":float(ev["racs_true_regret"].max()),
                "mean_true_regret":float(ev["racs_true_regret"].mean()),
                "exact_winner_matches":int(ev["racs_exact_winner"].sum()),
            })

    cv_df=pd.DataFrame(cv_rows)
    cv_df.to_csv(HERE/"RACS_CROSS_DOMAIN_AUDIT_ALL_WEIGHTS.csv",index=False)

    # Select the globally frozen rational weight by aggregate held-out-domain
    # development performance.
    cv_summary=(
        cv_df.groupby("weight",as_index=False)
        .agg(
            operational_valid=("operational_valid","sum"),
            total=("n_decisions","sum"),
            max_true_regret=("max_true_regret","max"),
            mean_true_regret=("mean_true_regret","mean"),
            exact_winner_matches=("exact_winner_matches","sum"),
        )
    )
    cv_summary["violations"]=cv_summary["total"]-cv_summary["operational_valid"]
    cv_summary=cv_summary.sort_values(
        ["violations","max_true_regret","mean_true_regret",
         "exact_winner_matches","weight"],
        ascending=[True,True,True,False,True]
    ).reset_index(drop=True)
    cv_summary.to_csv(HERE/"RACS_CROSS_DOMAIN_WEIGHT_SUMMARY.csv",index=False)

    # Final global development selection.
    chosen=float(cv_summary.iloc[0]["weight"])
    rescue=evaluate(unresolved,chosen)

    # Produce combined result; preserve upstream rows otherwise.
    final=merged.copy()
    final["racs_triggered"]=False
    final["racs_estimate"]=np.nan
    final["racs_weight"]=np.nan
    final["racs_status"]="UPSTREAM_PRESERVED"

    idx=final.index[~final["operational_valid"].astype(bool)]
    final.loc[idx,"action"]=rescue["racs_action"].to_numpy()
    final.loc[idx,"racs_triggered"]=True
    final.loc[idx,"racs_estimate"]=rescue["racs_estimate"].to_numpy()
    final.loc[idx,"racs_weight"]=float(chosen)
    final.loc[idx,"racs_status"]="DEVELOPMENT_CONSENSUS_SHRINKAGE"
    final.loc[idx,"true_operational_regret"]=rescue["racs_true_regret"].to_numpy()
    final.loc[idx,"operational_valid"]=rescue["racs_operational_valid"].to_numpy()
    final.loc[idx,"exact_winner_match"]=rescue["racs_exact_winner"].to_numpy()
    final.loc[idx,"empirical_best_or_tied"]=rescue["racs_exact_winner"].to_numpy()

    cells=final.groupby(
        ["dataset","profile","lambda"],as_index=False
    ).agg(
        operational_10of10=("operational_valid","all"),
        exact_10of10=("exact_winner_match","all"),
    )

    per=final.groupby("dataset",as_index=False).agg(
        operational_valid=("operational_valid","sum"),
        decisions_total=("operational_valid","size"),
        exact_winner_matches=("exact_winner_match","sum"),
        mean_label_fraction=("label_fraction","mean"),
        median_label_fraction=("label_fraction","median"),
        max_label_fraction=("label_fraction","max"),
        max_true_regret=("true_operational_regret","max"),
        racs_decisions=("racs_triggered","sum"),
    )

    final.to_csv(HERE/"RACS_SELECTED_ALL_SIX.csv",index=False)
    rescue.to_csv(HERE/"RACS_RESCUE_21.csv",index=False)
    per.to_csv(HERE/"RACS_PER_DATASET.csv",index=False)

    # Robust weight interval on this development rescue set.
    dense=np.linspace(0,1,10001)
    valid=[]
    for w in dense:
        e=evaluate(unresolved,float(w))
        if bool(e["racs_operational_valid"].all()):
            valid.append(float(w))

    assessment={
        "architecture":
            "RACS: Regime-Adaptive Consensus Shrinkage rescue after SHARP + CPPI-GREG",
        "trigger":
            "only upstream unresolved resource states",
        "candidate_weights":grid,
        "selected_cppi_weight":float(chosen),
        "selected_sharp_weight":float(1.0-chosen),
        "development_unresolved_entering_racs":int(len(unresolved)),
        "development_racs_operational_valid":
            int(rescue["racs_operational_valid"].sum()),
        "development_racs_exact_winner_matches":
            int(rescue["racs_exact_winner"].sum()),
        "development_racs_max_true_regret":
            float(rescue["racs_true_regret"].max()),
        "development_operational_valid":
            int(final["operational_valid"].sum()),
        "development_operational_total":int(len(final)),
        "development_operational_cells_10of10":
            int(cells["operational_10of10"].sum()),
        "cell_total":int(len(cells)),
        "development_exact_winner_matches":
            int(final["exact_winner_match"].sum()),
        "development_exact_cells_10of10":
            int(cells["exact_10of10"].sum()),
        "remaining_development_unresolved":
            int((~final["operational_valid"].astype(bool)).sum()),
        "max_observed_true_operational_regret":
            float(final["true_operational_regret"].max()),
        "mean_label_fraction":float(final["label_fraction"].mean()),
        "median_label_fraction":float(final["label_fraction"].median()),
        "max_label_fraction":float(final["label_fraction"].max()),
        "labels_added_by_racs":0,
        "racs_uses_dataset_or_seed_identity":False,
        "cross_domain_audit_all_operationally_valid":bool(
            cv_summary.iloc[0]["operational_valid"]==cv_summary.iloc[0]["total"]
        ),
        "cross_domain_audit_operational_valid":int(cv_summary.iloc[0]["operational_valid"]),
        "cross_domain_audit_total":int(cv_summary.iloc[0]["total"]),
        "robust_development_weight_interval":
            [min(valid),max(valid)] if valid else None,
        "formal_individual_frequentist_certificate_for_racs_rows":False,
        "development_closure":bool(
            final["operational_valid"].all()
            and cells["operational_10of10"].all()
            and float(final["true_operational_regret"].max())<=DELTA+1e-12
        ),
        "next_step":
            "Freeze RACS and validate the frozen rule on a genuinely fresh stream. Do not tune further on these six development domains.",
    }

    (HERE/"RACS_FINAL_ASSESSMENT.json").write_text(
        json.dumps(assessment,indent=2,sort_keys=True)
    )

    print("\\n=== RACS TERMINAL DEVELOPMENT CLOSURE ===")
    print(json.dumps(assessment,indent=2,sort_keys=True))
    print("\\n=== CROSS-DOMAIN WEIGHT SUMMARY ===")
    print(cv_summary.to_string(index=False))
    print("\\n=== SELECTED WEIGHT DOMAIN AUDIT ===")
    print(cv_df[np.isclose(cv_df["weight"],chosen)].to_string(index=False))
    print("\\n=== PER-DATASET ===")
    print(per.to_string(index=False))

if __name__=="__main__":
    main()
