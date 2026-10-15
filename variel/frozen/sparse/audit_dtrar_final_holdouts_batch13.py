#!/usr/bin/env python3
"""Structural-only lock for FINAL untouched confirmation streams. No model is trained."""
from __future__ import annotations
from pathlib import Path
import hashlib,json,math,zipfile
import numpy as np
import pandas as pd

import dtrar_final_core as fc

PROJECT=Path(__file__).resolve().parent
DEV=PROJECT/"artifacts"/"dtrar_final_development_batch11"
ROB=PROJECT/"artifacts"/"dtrar_final_robustness_batch12"
OUT=PROJECT/"artifacts"/"dtrar_final_holdout_audit"
LOCKED=OUT/"locked"
MAX_ROWS=250000
PRIORITY=[
    {"id":755,"name":"uci755_accel_gyro_phone","target":"Activity"},
    {"id":780,"name":"uci780_har70plus","target":"label"},
    {"id":348,"name":"uci348_indoor_user_movement","target":None},
    {"id":196,"name":"uci196_localization_activity","target":"activity"},
    {"id":427,"name":"uci427_older_activity","target":"label"},
    {"id":760,"name":"uci760_multivariate_gait","target":"condition"},
]


def must(c,m):
    if not c: raise RuntimeError(m)

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def find_col(df,name):
    if name is None: return None
    m={str(c).strip().lower():c for c in df.columns}
    return m.get(str(name).strip().lower())

def fetch_record(spec):
    from ucimlrepo import fetch_ucirepo
    ds=fetch_ucirepo(id=spec["id"])
    original=getattr(ds.data,"original",None)
    X=getattr(ds.data,"features",None)
    Y=getattr(ds.data,"targets",None)
    if original is not None and isinstance(original,pd.DataFrame) and len(original):
        df=original.copy()
    elif X is not None:
        df=X.copy()
        if Y is not None and len(Y)==len(df):
            for c in Y.columns: df[c]=Y[c].to_numpy()
    else:
        raise RuntimeError("repository returned no tabular data")

    tc=find_col(df,spec.get("target"))
    if tc is None and Y is not None and isinstance(Y,pd.DataFrame) and Y.shape[1]>=1 and len(Y)==len(df):
        tc=Y.columns[0]
        if tc not in df: df[tc]=Y.iloc[:,0].to_numpy()
    if tc is None:
        raise RuntimeError(f"target column unresolved; columns={list(df.columns)[:30]}")
    y=df.pop(tc)
    return df,y,str(tc),getattr(ds,"metadata",{})

def numericize(X):
    out=pd.DataFrame(index=np.arange(len(X))); log=[]
    for c in X.columns:
        s=X[c]
        if pd.api.types.is_numeric_dtype(s):
            out[str(c)]=pd.to_numeric(s,errors="coerce").astype(float)
            log.append({"column":str(c),"transform":"numeric"})
            continue
        ss=s.astype("string")
        dt=pd.to_datetime(ss,errors="coerce",utc=True)
        if float(dt.notna().mean())>=0.90:
            vals=dt.astype("int64").astype(float)/1e9
            vals[dt.isna()]=np.nan
            out[str(c)]=vals
            log.append({"column":str(c),"transform":"datetime_unix_seconds"})
        else:
            # Stateless deterministic encoding: no future covariate fitting.
            def _h(v):
                if pd.isna(v): return np.nan
                b=hashlib.sha256(str(v).encode()).digest()[:8]
                return float(int.from_bytes(b,"little") % 1000000007)
            out[str(c)]=ss.map(_h).astype(float)
            log.append({"column":str(c),"transform":"stateless_sha256_category_code"})
    # Drop columns unusable everywhere. Remaining missing values are legal and are imputed source-only later.
    keep=[c for c in out.columns if out[c].notna().any() and out[c].nunique(dropna=True)>1]
    return out[keep],log

def structural_audit(spec):
    rec={"id":spec["id"],"name":spec["name"],"admissible":False,"performance_evaluated":False}
    try:
        X,y,target_col,meta=fetch_record(spec)
        rec["repository_rows"]=int(len(y)); rec["resolved_target_column"]=target_col
        must(len(X)==len(y) and len(y)>=5000,"too few or mismatched rows")
        must(not pd.isna(y).any(),"missing target")
        # Deterministic label-free temporal thinning for very large streams.
        if len(y)>MAX_ROWS:
            idx=np.unique(np.linspace(0,len(y)-1,MAX_ROWS,dtype=int))
            X=X.iloc[idx].reset_index(drop=True); y=y.iloc[idx].reset_index(drop=True)
            rec["thinning"]={"rule":"evenly_spaced_row_index","max_rows":MAX_ROWS,"original_n":rec["repository_rows"]}
        else:
            X=X.reset_index(drop=True); y=y.reset_index(drop=True); rec["thinning"]={"rule":"none"}
        X,trans=numericize(X); must(X.shape[1]>=2,"fewer than two usable predictors")
        labels=sorted([str(v) for v in pd.unique(y.astype("string"))]); must(2<=len(labels)<=20,f"class count {len(labels)} outside 2..20")
        ym={v:i for i,v in enumerate(labels)}; yy=y.astype("string").map(ym).to_numpy(int)
        n=len(yy); a=int(math.floor(.50*n)); b=int(math.floor(.65*n)); windows=[np.asarray(w,int) for w in np.array_split(np.arange(b,n),12)]
        tr=set(np.unique(yy[:a]).tolist()); src=set(np.unique(yy[a:b]).tolist()); tg=set(np.unique(yy[b:]).tolist())
        must(tg.issubset(tr),"target contains class absent from training")
        must(tg.issubset(src),"target contains class absent from source/calibration")
        must(len(tg)>=2,"target is single-class")
        sched,B=fc.horizon_schedule(windows,fc.BUDGET_RATE,"balanced")
        must(min(sched.values())>=fc.MIN_CANARY,"at least one target window cannot support minimum canary")
        lockdf=X.copy(); lockdf["__target__"]=yy; lockdf["__sequence__"]=np.arange(n,dtype=np.int64)
        LOCKED.mkdir(parents=True,exist_ok=True); p=LOCKED/f"{spec['name']}.csv"; lockdf.to_csv(p,index=False)
        rec.update({"admissible":True,"n":n,"features":int(X.shape[1]),"classes":labels,
                    "train_n":a,"source_n":b-a,"target_n":n-b,"target_windows":12,
                    "canary_budget_labels":B,"min_window_canary":int(min(sched.values())),
                    "locked_csv":str(p.relative_to(PROJECT)),"locked_sha256":sha(p),
                    "ordering":"repository row order; deterministic even-index thinning only if n>250000",
                    "feature_transforms":trans})
    except Exception as e:
        rec["rejection_reason"]=repr(e)
    return rec

def main():
    OUT.mkdir(parents=True,exist_ok=True); LOCKED.mkdir(parents=True,exist_ok=True)
    dev=DEV/"FINAL_DEVELOPMENT_ASSESSMENT.json"; rob=ROB/"FINAL_ROBUSTNESS_ASSESSMENT.json"; freeze=DEV/"FINAL_METHOD_FREEZE.json"
    for p in [dev,rob,freeze]: must(p.exists(),f"missing prerequisite {p}")
    dj=json.loads(dev.read_text()); rj=json.loads(rob.read_text()); fj=json.loads(freeze.read_text())
    must(dj.get("pass") is True and rj.get("pass") is True,"development/robustness did not pass; refusing fresh audit")
    core_hash=sha(PROJECT/"dtrar_final_core.py"); must(core_hash==fj["core_sha256"],"final core changed after freeze")
    print("FINAL_HOLDOUT_AUDIT_BEGIN performance_evaluation=false priority_locked=true",flush=True)
    records=[]
    for spec in PRIORITY:
        r=structural_audit(spec); records.append(r)
        print(f"FINAL_HOLDOUT_AUDIT dataset={r['name']} admissible={r['admissible']} reason={r.get('rejection_reason','OK')}",flush=True)
    selected=[r["name"] for r in records if r["admissible"]][:3]
    must(len(selected)==3,f"fewer than three structurally admissible holdouts: {selected}")
    gates={
        "all_exact_H_le_0p005":True,"all_requested_canary_budget_exact_4pct":True,
        "all_three_mean_brier_gain_nonnegative":True,
        "at_least_two_of_three_material_relative_brier_gain_ge_1pct":True,
        "at_least_two_of_three_positive_seeds_ge_8of10":True,
        "pooled_mean_brier_gain_positive":True,
        "no_source_change_after_lock":True,
    }
    manifest={"selection_locked_before_performance":True,"performance_seen":False,"priority_order":[x["name"] for x in PRIORITY],
              "selected_fresh_holdouts":selected,"datasets":{r["name"]:r for r in records},
              "final_core_sha256":core_hash,"final_method_freeze":fj,
              "predeclared_confirmation_gates":gates,"seeds":list(range(10)),"base_model_family":"xgboost"}
    pd.DataFrame([{k:v for k,v in r.items() if k not in {"feature_transforms","classes"}} for r in records]).to_csv(OUT/"FINAL_HOLDOUT_AUDIT.csv",index=False)
    (OUT/"FINAL_HOLDOUT_LOCK.json").write_text(json.dumps(manifest,indent=2,sort_keys=True))
    review=PROJECT/"DTRAR_FinalHoldoutAudit_Batch13_Review.zip"; tmp=review.with_suffix(".zip.partial")
    if tmp.exists(): tmp.unlink()
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:
        z.write(OUT/"FINAL_HOLDOUT_AUDIT.csv","FINAL_HOLDOUT_AUDIT.csv"); z.write(OUT/"FINAL_HOLDOUT_LOCK.json","FINAL_HOLDOUT_LOCK.json"); z.write(Path(__file__),Path(__file__).name)
        for name in selected:
            p=PROJECT/manifest["datasets"][name]["locked_csv"]; z.write(p,f"locked/{p.name}")
    with zipfile.ZipFile(tmp) as z: must(z.testzip() is None,"review CRC failure")
    tmp.replace(review)
    print("FINAL_HOLDOUT_SELECTION_LOCKED datasets="+",".join(selected),flush=True)
    print(f"FINAL_HOLDOUT_AUDIT_COMPLETE performance_evaluation=false review={review.name}",flush=True)

if __name__=="__main__": main()
