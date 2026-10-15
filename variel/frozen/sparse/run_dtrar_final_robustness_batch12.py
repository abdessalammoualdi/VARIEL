#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import hashlib,json,zipfile
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import dtrar_final_core as fc
import run_dtrar_final_development_batch11 as b11

PROJECT=Path(__file__).resolve().parent
DEV=PROJECT/"artifacts"/"dtrar_final_development_batch11"
OUT=PROJECT/"artifacts"/"dtrar_final_robustness_batch12"
SEEDS=list(range(10))
XGB_CELLS=[
 ("xgboost","electricity"),("xgboost","bank"),("xgboost","gas"),("xgboost","insects"),
 ("xgboost_seen_unseen","occupancy_detection_v2"),("xgboost_seen_unseen","eeg_eye_state"),
 ("xgboost_consumed_holdout","pirvision_uci1101"),("xgboost_consumed_holdout","ai4i_uci601")]
NOISE_CELLS=XGB_CELLS[:6]


def must(c,m):
    if not c: raise RuntimeError(m)

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for x in iter(lambda:f.read(1024*1024),b""): h.update(x)
    return h.hexdigest()

def native(x):
    if isinstance(x,dict): return {str(k):native(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [native(v) for v in x]
    if isinstance(x,np.bool_): return bool(x)
    if isinstance(x,np.integer): return int(x)
    if isinstance(x,np.floating): return float(x)
    return x


def theorem_verify(reps=5000,seed=99173):
    rng=np.random.default_rng(seed); errs=[]; boundary=[]
    for _ in range(reps):
        k=int(rng.integers(2,8)); n=int(rng.integers(2,40))
        p=rng.dirichlet(np.ones(k),size=n); q=rng.dirichlet(np.ones(k),size=n)
        eta=float(rng.random()); d=eta*(q-p)
        H=fc.prospective_H(p,d)
        # Exact max over current labels decomposes rowwise.
        per=[]
        for c in range(k):
            e=np.zeros((n,k)); e[:,c]=1.0
            diff=np.sum((p+d-e)**2-(p-e)**2,axis=1)
            per.append(diff)
        brute=float(np.mean(np.max(np.column_stack(per),axis=1)))
        errs.append(abs(H-brute))
        emax,_=fc.exact_safe_eta(p,q,1.0,fc.EPS)
        h=fc.prospective_H(p,emax*(q-p))
        boundary.append(max(0.0,h-fc.EPS))
    return {"reps":reps,"max_identity_error":float(max(errs)),"max_boundary_violation":float(max(boundary))}


def cases():
    out=[]
    for rate in [0.01,0.02,0.04,0.08,0.10]:
        for cell in XGB_CELLS: out.append((f"budget_{rate:.2f}",cell,dict(budget_rate=rate,label_noise=0,label_dropout=0)))
    for noise in [0.05,0.10]:
        for cell in NOISE_CELLS: out.append((f"noise_{noise:.2f}",cell,dict(budget_rate=0.04,label_noise=noise,label_dropout=0)))
    for drop in [0.10,0.25]:
        for cell in NOISE_CELLS: out.append((f"dropout_{drop:.2f}",cell,dict(budget_rate=0.04,label_noise=0,label_dropout=drop)))
    return out


def summarize(df):
    rows=[]
    for (case,fam,ds),g in df.groupby(["case","family","dataset"]):
        x=g.brier_gain.to_numpy(float)
        rows.append({"case":case,"family":fam,"dataset":ds,"mean_gain":float(x.mean()),
                     "sd_gain":float(x.std(ddof=1)),"min_seed_gain":float(x.min()),
                     "positive_seeds":int((x>0).sum()),"seeds":len(x),
                     "max_exact_H":float(g.max_exact_H.max()),"all_budget_exact":bool(g.budget_exact.all()),
                     "mean_observed_labels":float(g.observed_canary_labels.mean()),"mean_requested_labels":float(g.requested_canary_labels.mean())})
    return pd.DataFrame(rows)


def assess(summary,seedwise,theory):
    exact=bool(seedwise.max_exact_H.max()<=fc.EPS+5e-9)
    budgets=bool(seedwise.budget_exact.all())
    nom=summary[summary.case.eq("budget_0.04")]
    primary=nom[nom.apply(lambda r:(r.family,r.dataset) in b11.PRIMARY[:4],axis=1)]
    nominal_primary_positive=bool(len(primary)==4 and (primary.mean_gain>0).all())
    def pick(case,ds):
        z=summary[(summary.case==case)&(summary.dataset==ds)]; return float(z.mean_gain.iloc[0]) if len(z) else float("nan")
    robust_bg=all(pick(c,ds)>=-0.001 for c in ["noise_0.10","dropout_0.25"] for ds in ["bank","gas"])
    low_budget_noncat=all(pick("budget_0.02",ds)>=-0.001 for ds in ["bank","gas","electricity","insects"])
    theory_ok=theory["max_identity_error"]<1e-12 and theory["max_boundary_violation"]<1e-10
    gates={"all_exact_contracts_hold":exact,"all_requested_budgets_exact":budgets,
           "nominal_xgb_primary_positive":nominal_primary_positive,
           "10pct_noise_and_25pct_dropout_bank_gas_no_material_harm":bool(robust_bg),
           "2pct_budget_xgb_primary_no_material_harm":bool(low_budget_noncat),
           "exact_H_identity_and_boundary_verified":bool(theory_ok)}
    return gates,bool(all(gates.values()))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    freeze=DEV/"FINAL_METHOD_FREEZE.json"; must(freeze.exists(),"Batch11 did not pass/freeze; refusing robustness")
    fr=json.loads(freeze.read_text()); must(fr["core_sha256"]==sha(PROJECT/"dtrar_final_core.py"),"final core changed after Batch11 freeze")
    print("FINAL_ROBUSTNESS_BATCH12_BEGIN frozen_core=true fresh_holdouts_opened=false",flush=True)
    rows=[]; total=len(cases())*10; n=0
    with threadpool_limits(limits=1):
        for case,(fam,ds),kw in cases():
            for seed in SEEDS:
                n+=1; print(f"ROBUST_JOB_BEGIN {n}/{total} case={case} family={fam} dataset={ds} seed={seed}",flush=True)
                data,raw=b11.load_job(fam,ds,seed)
                s,_=fc.run_stream(data,raw,seed,fam,ds,schedule_mode="balanced",sample_mode="geometry",action_mode="riskopt",exact_h=True,**kw)
                s["case"]=case; rows.append(s)
                print("ROBUST_JOB_RESULT "+json.dumps(native({"case":case,"family":fam,"dataset":ds,"seed":seed,"gain":s["brier_gain"],"max_H":s["max_exact_H"]}),sort_keys=True),flush=True)
    seedwise=pd.DataFrame(rows); summary=summarize(seedwise); theory=theorem_verify()
    seedwise.to_csv(OUT/"FINAL_ROBUSTNESS_SEEDWISE.csv",index=False); summary.to_csv(OUT/"FINAL_ROBUSTNESS_SUMMARY.csv",index=False)
    (OUT/"FINAL_THEOREM_VERIFICATION.json").write_text(json.dumps(theory,indent=2,sort_keys=True))
    gates,passed=assess(summary,seedwise,theory)
    assessj=native({"pass":passed,"gates":gates,"core_sha256":fr["core_sha256"],"fresh_holdouts_opened":False,
                    "budget_frontier":[0.01,0.02,0.04,0.08,0.10],"label_noise":[0.05,0.10],"label_dropout":[0.10,0.25]})
    (OUT/"FINAL_ROBUSTNESS_ASSESSMENT.json").write_text(json.dumps(assessj,indent=2,sort_keys=True))
    review=PROJECT/"DTRAR_FinalRobustness_Batch12_Review.zip"; tmp=review.with_suffix(".zip.partial")
    if tmp.exists(): tmp.unlink()
    members=[OUT/"FINAL_ROBUSTNESS_SEEDWISE.csv",OUT/"FINAL_ROBUSTNESS_SUMMARY.csv",OUT/"FINAL_THEOREM_VERIFICATION.json",OUT/"FINAL_ROBUSTNESS_ASSESSMENT.json",PROJECT/"dtrar_final_core.py",Path(__file__),freeze]
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:
        for p in members: z.write(p,p.name)
    with zipfile.ZipFile(tmp) as z: must(z.testzip() is None,"review CRC failure")
    tmp.replace(review)
    print("FINAL_ROBUSTNESS_BATCH12_ASSESSMENT "+json.dumps(gates,sort_keys=True),flush=True)
    print(f"FINAL_ROBUSTNESS_BATCH12_COMPLETE pass={str(passed).lower()} review={review.name}",flush=True)
    print(summary.to_string(index=False),flush=True)

if __name__=="__main__": main()
