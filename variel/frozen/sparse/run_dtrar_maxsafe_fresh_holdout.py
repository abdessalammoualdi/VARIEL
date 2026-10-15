#!/usr/bin/env python3
"""Frozen fresh-holdout confirmation of DTRA-RAPID MaxSafe.

Requires:
  1) structural holdout lock (no performance)
  2) MaxSafe development gate pass and source freeze

No MaxSafe rule may be changed here.
XGBoost is trained source-only with the same primary numerical protocol used in
the earlier unseen confirmation. Ten seeds are run on every structurally
selected fresh holdout.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib, importlib.util, json, math, os, platform, shutil, sys, zipfile
import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

PROJECT=Path(__file__).resolve().parent
BASE=PROJECT/"offset_check_v07"
SPEC=PROJECT/"MAXSAFE_SPEC.json"
LOCK=PROJECT/"artifacts"/"maxsafe_holdout_audit"/"MAXSAFE_HOLDOUT_LOCK.json"
FREEZE=PROJECT/"artifacts"/"dtrar_maxsafe_development"/"MAXSAFE_FREEZE.json"
GEN=PROJECT/"artifacts"/"dtrar_maxsafe_development"/"_generated_maxsafe_xgboost.py"
OUTROOT=PROJECT/"artifacts"/"maxsafe_fresh_holdout"
SEEDS=list(range(10))
MAXSAFE_METHOD="dtrar_maxsafe_0p005"

def must(c,m):
    if not c: raise SystemExit("ERROR: "+m)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

@dataclass
class Dataset:
    name:str
    X:pd.DataFrame
    y:np.ndarray
    classes:list
    categorical:list
    train:np.ndarray
    source:np.ndarray
    windows:list
    metadata:dict
    @property
    def target(self):
        return np.concatenate(self.windows)

MANIFEST=None
_CACHE={}

def load_data(name,*_args,**_kwargs):
    if name in _CACHE:
        return _CACHE[name]
    rec=MANIFEST["datasets"][name]
    p=PROJECT/rec["locked_csv"]
    must(p.exists(),f"missing locked holdout {p}")
    must(sha(p)==rec["locked_sha256"],f"{name}: holdout hash changed")
    df=pd.read_csv(p)
    yraw=df.pop("__target__")
    if "__sequence__" in df:
        df=df.drop(columns=["__sequence__"])
    must(not yraw.isna().any(),f"{name}: missing target")
    obj=df.select_dtypes(include=["object","string","category"]).columns.tolist()
    must(not obj,f"{name}: nonnumeric locked predictors {obj}")
    X=df.apply(pd.to_numeric,errors="raise")
    labels=sorted(pd.unique(yraw).tolist(),key=lambda x:str(x))
    mapping={v:i for i,v in enumerate(labels)}
    y=yraw.map(mapping).to_numpy(np.int64)
    n=len(y); a=int(0.50*n); b=int(0.65*n)
    data=Dataset(
        name=name,X=X,y=y,classes=[str(v) for v in labels],categorical=[],
        train=np.arange(a),source=np.arange(a,b),
        windows=[np.asarray(x,int) for x in np.array_split(np.arange(b,n),12)],
        metadata={"fresh_holdout":True,"locked_sha256":rec["locked_sha256"],
                  "ordering":rec["ordering"]}
    )
    tr=set(np.unique(y[data.train]).tolist())
    rr=set(np.unique(y[data.source]).tolist())
    tg=set(np.unique(y[data.target]).tolist())
    must(len(rr)>=2 and len(tg)>=2 and tg.issubset(tr),
         f"{name}: structural class-support lock no longer holds")
    _CACHE[name]=data
    return data

def source_hashes_holdout(_root,dataset):
    rec=MANIFEST["datasets"][dataset]
    return {"locked_sha256":rec["locked_sha256"]}

def primary_config():
    return json.loads((BASE/"configs"/"primary.json").read_text())

def train_or_reuse(data,seed):
    from xgboost import XGBClassifier
    rec=MANIFEST["datasets"][data.name]
    cfg=primary_config()
    ident={
        "dataset":data.name,"locked_sha256":rec["locked_sha256"],
        "seed":seed,"family":"xgboost",
        "tree_cfg":{k:cfg[k] for k in cfg if k.startswith("tree_")},
        "protocol":"maxsafe-fresh-holdout-v1",
    }
    sig=hashlib.sha256(json.dumps(ident,sort_keys=True).encode()).hexdigest()[:12]
    cache=OUTROOT/"models"/data.name/f"seed_{seed}_{sig}"
    cache.mkdir(parents=True,exist_ok=True)
    pf=cache/"frozen_probabilities.npz"
    cf=cache/"COMPLETE.json"
    if pf.exists() and cf.exists():
        saved=np.load(pf)
        raw=np.asarray(saved["probabilities"],float)
        must(np.array_equal(saved["row_id"],np.arange(len(data.y))),"cached row-id mismatch")
        must(raw.shape==(len(data.y),len(data.classes)),"cached probability shape mismatch")
        print(f"MAXSAFE_HOLDOUT_CACHE_REUSED dataset={data.name} seed={seed}",flush=True)
        return cache,raw

    boundary=int(math.floor(0.8*len(data.train)))
    inner,valid=data.train[:boundary],data.train[boundary:]
    must(len(np.unique(data.y[inner]))==len(data.classes),
         f"{data.name}: inner source training lacks a class")
    imp=SimpleImputer(strategy="median").fit(data.X.iloc[inner])
    Xin=imp.transform(data.X.iloc[inner]).astype(np.float32)
    Xva=imp.transform(data.X.iloc[valid]).astype(np.float32)
    k=len(data.classes)
    args=dict(
        n_estimators=cfg["tree_rounds"],max_depth=cfg["tree_depth"],
        learning_rate=cfg["tree_learning_rate"],subsample=0.9,colsample_bytree=0.9,
        min_child_weight=2,reg_lambda=1.0,tree_method="hist",device="cpu",n_jobs=1,
        random_state=seed,
        objective="binary:logistic" if k==2 else "multi:softprob",
        eval_metric="logloss" if k==2 else "mlogloss",
    )
    print(f"MAXSAFE_HOLDOUT_TRAIN_BEGIN dataset={data.name} seed={seed}",flush=True)
    sel=XGBClassifier(**args,early_stopping_rounds=cfg["tree_early_stopping"])
    sel.fit(Xin,data.y[inner],eval_set=[(Xva,data.y[valid])],verbose=False)
    rounds=max(1,int(sel.best_iteration)+1)

    fullimp=SimpleImputer(strategy="median").fit(data.X.iloc[data.train])
    Xall=fullimp.transform(data.X).astype(np.float32)
    args["n_estimators"]=rounds
    model=XGBClassifier(**args)
    model.fit(Xall[data.train],data.y[data.train],verbose=False)
    raw=np.asarray(model.predict_proba(Xall),float)
    must(raw.shape==(len(data.y),k),f"{data.name}: prediction shape {raw.shape}")
    tmp=cache/"frozen_probabilities.npz.tmp.npz"
    np.savez_compressed(tmp,probabilities=raw,row_id=np.arange(len(data.y)))
    tmp.replace(pf)
    joblib.dump(fullimp,cache/"imputer.joblib")
    (cache/"COMPLETE.json").write_text(json.dumps({**ident,"rounds":rounds},indent=2,sort_keys=True))
    print(f"MAXSAFE_HOLDOUT_TRAIN_COMPLETE dataset={data.name} seed={seed} rounds={rounds}",flush=True)
    return cache,raw

def frozen_cache_holdout(_root,data,family,seed,profile,hashes):
    must(family=="xgboost" and profile=="primary","unexpected base cache request")
    return train_or_reuse(data,seed)

def import_generated():
    spec=importlib.util.spec_from_file_location("dtrar_maxsafe_frozen",GEN)
    mod=importlib.util.module_from_spec(spec)
    sys.modules["dtrar_maxsafe_frozen"]=mod
    spec.loader.exec_module(mod)
    return mod

def main():
    global MANIFEST
    for p in [SPEC,LOCK,FREEZE,GEN]:
        must(p.exists(),f"missing required frozen input {p}")
    MANIFEST=json.loads(LOCK.read_text())
    freeze=json.loads(FREEZE.read_text())
    must(MANIFEST.get("performance_seen") is False,"holdout manifest says performance seen")
    must(MANIFEST.get("selection_locked_before_performance") is True,"holdout was not locked")
    must(sha(SPEC)==MANIFEST["maxsafe_spec_sha256"],"method spec differs from holdout lock")
    must(freeze.get("freeze_for_fresh_holdouts") is True,"MaxSafe development did not pass; refusing holdout")
    must(sha(SPEC)==freeze["spec_sha256"],"method spec differs from development freeze")
    must(sha(GEN)==freeze["generated_xgboost_sha256"],"generated MaxSafe source changed after freeze")
    datasets=list(MANIFEST["selected_fresh_holdouts"])
    must(datasets,"no selected fresh holdouts")

    ms=import_generated()
    ms.core.load_dataset=load_data
    ms.core.source_hashes=source_hashes_holdout
    ms.core.frozen_cache=frozen_cache_holdout

    sig=hashlib.sha256(json.dumps({
        "datasets":datasets,"seeds":SEEDS,"spec_sha256":sha(SPEC),
        "generated_sha256":sha(GEN)},sort_keys=True).encode()).hexdigest()[:12]
    out=OUTROOT/sig
    out.mkdir(parents=True,exist_ok=True)
    (OUTROOT/"LATEST.txt").write_text(str(out)+"\n")
    shutil.copy2(SPEC,out/"MAXSAFE_SPEC.json")
    shutil.copy2(LOCK,out/"MAXSAFE_HOLDOUT_LOCK.json")
    shutil.copy2(FREEZE,out/"MAXSAFE_FREEZE.json")

    print(f"MAXSAFE_FRESH_HOLDOUT_BEGIN datasets={len(datasets)} seeds=10 performance_was_unseen=true",flush=True)
    alls=[]; alld=[]
    with threadpool_limits(limits=1):
        for ds in datasets:
            for seed in SEEDS:
                s,d=ms.run_one(ds,seed,out)
                alls.extend(s); alld.extend(d)

    sdf=pd.DataFrame(alls)
    sdf.to_csv(out/"all_summaries.csv",index=False)
    trs=[]
    for p in sorted((out/"jobs").glob("*/transport_trace.json")):
        trs.extend(json.loads(p.read_text()))
    td=pd.DataFrame(trs); td.to_csv(out/"transport_trace.csv",index=False)

    piv=sdf.pivot_table(index=["dataset","seed"],columns="method",values="brier_sum_over_classes")
    rows=[]
    for ds in datasets:
        d=piv.loc[ds]; raw=d["raw"]
        for m in [MAXSAFE_METHOD,"dtrar_rapid","rapid_fixed_0p005","vamr","logistic"]:
            g=raw-d[m]
            try:
                wp=float(wilcoxon(g,alternative="greater").pvalue)
            except Exception:
                wp=float("nan")
            rows.append({
                "dataset":ds,"method":m,"mean_gain":float(g.mean()),
                "sd_gain":float(g.std(ddof=1)),"min_seed_gain":float(g.min()),
                "positive_seeds":int((g>0).sum()),"wilcoxon_p_one_sided":wp,
            })
    summary=pd.DataFrame(rows)
    summary.to_csv(out/"MAXSAFE_FRESH_HOLDOUT_SUMMARY.csv",index=False)

    mt=td[td.method.eq(MAXSAFE_METHOD)].copy()
    mt["observed_le_envelope"]=mt["observed_brier_excess_vs_raw"]<=mt["envelope_at_deploy_eta"]+1e-9
    mt["envelope_le_budget"]=mt["envelope_at_deploy_eta"]<=mt["harm_budget"]+1e-9
    safety=(mt.groupby("dataset",as_index=False)
            .agg(windows=("deployment_window","count"),
                 observed_le_envelope=("observed_le_envelope","all"),
                 envelope_le_budget=("envelope_le_budget","all"),
                 max_observed_excess=("observed_brier_excess_vs_raw","max"),
                 max_envelope=("envelope_at_deploy_eta","max"),
                 max_budget=("harm_budget","max")))
    safety.to_csv(out/"MAXSAFE_FRESH_HOLDOUT_SAFETY.csv",index=False)

    msum=summary[summary.method.eq(MAXSAFE_METHOD)]
    gates={
        "all_exact_contracts_hold":bool(safety.observed_le_envelope.all() and safety.envelope_le_budget.all()),
        "all_holdout_mean_nonharm":bool((msum.mean_gain>=-1e-10).all()),
        "at_least_one_holdout_positive":bool((msum.mean_gain>0).any()),
        "no_rule_change_after_lock":True,
    }
    assessment={
        "datasets":datasets,"seeds":SEEDS,"gates":gates,
        "pass":bool(all(gates.values())),
        "method_spec_sha256":sha(SPEC),"generated_source_sha256":sha(GEN),
    }
    (out/"MAXSAFE_FRESH_HOLDOUT_ASSESSMENT.json").write_text(json.dumps(assessment,indent=2,sort_keys=True))

    review=PROJECT/"DTRAR_MaxSafe_FreshHoldout_Review.zip"
    tmp=review.with_suffix(".zip.partial")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:
        for n in ["MAXSAFE_SPEC.json","MAXSAFE_HOLDOUT_LOCK.json","MAXSAFE_FREEZE.json",
                  "all_summaries.csv","transport_trace.csv","MAXSAFE_FRESH_HOLDOUT_SUMMARY.csv",
                  "MAXSAFE_FRESH_HOLDOUT_SAFETY.csv","MAXSAFE_FRESH_HOLDOUT_ASSESSMENT.json"]:
            z.write(out/n,n)
        z.write(Path(__file__),Path(__file__).name)
    with zipfile.ZipFile(tmp) as z:
        must(z.testzip() is None,"review CRC failure")
    tmp.replace(review)
    print("MAXSAFE_FRESH_HOLDOUT_ASSESSMENT "+json.dumps(gates,sort_keys=True),flush=True)
    print(f"MAXSAFE_FRESH_HOLDOUT_COMPLETE pass={str(assessment['pass']).lower()} review={review.name}",flush=True)
    print(msum.to_string(index=False),flush=True)

if __name__=="__main__":
    main()
