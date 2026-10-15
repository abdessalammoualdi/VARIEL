#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib,json,math,zipfile
import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

import dtrar_final_core as fc

PROJECT=Path(__file__).resolve().parent
AUD=PROJECT/"artifacts"/"dtrar_final_holdout_audit"
LOCK=AUD/"FINAL_HOLDOUT_LOCK.json"
DEV=PROJECT/"artifacts"/"dtrar_final_development_batch11"
ROB=PROJECT/"artifacts"/"dtrar_final_robustness_batch12"
OUT=PROJECT/"artifacts"/"dtrar_final_fresh_confirmation_batch14"
SEEDS=list(range(10))


def must(c,m):
    if not c: raise RuntimeError(m)

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def native(x):
    if isinstance(x,dict): return {str(k):native(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [native(v) for v in x]
    if isinstance(x,np.bool_): return bool(x)
    if isinstance(x,np.integer): return int(x)
    if isinstance(x,np.floating): return float(x)
    return x

@dataclass
class Dataset:
    name:str; X:pd.DataFrame; y:np.ndarray; classes:list; categorical:list
    train:np.ndarray; source:np.ndarray; windows:list; metadata:dict
    @property
    def target(self): return np.concatenate(self.windows)


def load_locked(name,manifest):
    rec=manifest["datasets"][name]; p=PROJECT/rec["locked_csv"]
    must(p.exists(),f"missing locked dataset {p}"); must(sha(p)==rec["locked_sha256"],f"hash changed {name}")
    df=pd.read_csv(p); must("__target__" in df,"missing target"); y=df.pop("__target__").to_numpy(int)
    if "__sequence__" in df: df=df.drop(columns=["__sequence__"])
    X=df.apply(pd.to_numeric,errors="coerce"); n=len(y); a=int(math.floor(.50*n)); b=int(math.floor(.65*n))
    data=Dataset(name=name,X=X,y=y,classes=[str(x) for x in rec["classes"]],categorical=[],
                 train=np.arange(a),source=np.arange(a,b),windows=[np.asarray(w,int) for w in np.array_split(np.arange(b,n),12)],
                 metadata={"fresh_final_confirmation":True,"locked_sha256":rec["locked_sha256"]})
    return data


def primary_config():
    p=PROJECT/"offset_check_v07"/"configs"/"primary.json"; must(p.exists(),f"missing {p}")
    return json.loads(p.read_text())


def train_xgb(data,seed,locked_sha):
    from xgboost import XGBClassifier
    cfg=primary_config(); ident={"dataset":data.name,"locked_sha256":locked_sha,"seed":seed,"family":"xgboost","protocol":"final-fresh-confirmation-v1","tree_cfg":{k:cfg[k] for k in cfg if k.startswith("tree_")}}
    sig=hashlib.sha256(json.dumps(ident,sort_keys=True).encode()).hexdigest()[:12]
    cache=OUT/"models"/data.name/f"seed_{seed}_{sig}"; cache.mkdir(parents=True,exist_ok=True)
    pf=cache/"frozen_probabilities.npz"; cf=cache/"COMPLETE.json"
    if pf.exists() and cf.exists():
        z=np.load(pf); raw=np.asarray(z["probabilities"],float); must(np.array_equal(z["row_id"],np.arange(len(data.y))),"cached row mismatch"); return raw
    boundary=int(math.floor(.8*len(data.train))); inner,valid=data.train[:boundary],data.train[boundary:]
    must(len(np.unique(data.y[inner]))==len(data.classes),f"{data.name}: training lacks class")
    imp=SimpleImputer(strategy="median").fit(data.X.iloc[inner]); Xin=imp.transform(data.X.iloc[inner]).astype(np.float32); Xva=imp.transform(data.X.iloc[valid]).astype(np.float32)
    k=len(data.classes); args=dict(n_estimators=cfg["tree_rounds"],max_depth=cfg["tree_depth"],learning_rate=cfg["tree_learning_rate"],subsample=.9,colsample_bytree=.9,min_child_weight=2,reg_lambda=1.,tree_method="hist",device="cpu",n_jobs=1,random_state=seed,objective="binary:logistic" if k==2 else "multi:softprob",eval_metric="logloss" if k==2 else "mlogloss")
    sel=XGBClassifier(**args,early_stopping_rounds=cfg["tree_early_stopping"]); sel.fit(Xin,data.y[inner],eval_set=[(Xva,data.y[valid])],verbose=False); rounds=max(1,int(sel.best_iteration)+1)
    fullimp=SimpleImputer(strategy="median").fit(data.X.iloc[data.train]); Xall=fullimp.transform(data.X).astype(np.float32); args["n_estimators"]=rounds
    model=XGBClassifier(**args); model.fit(Xall[data.train],data.y[data.train],verbose=False); raw=np.asarray(model.predict_proba(Xall),float)
    must(raw.shape==(len(data.y),k),"prediction shape mismatch"); np.savez_compressed(pf,probabilities=raw,row_id=np.arange(len(data.y))); joblib.dump(fullimp,cache/"preprocessor.joblib"); model.save_model(cache/"frozen.ubj"); cf.write_text(json.dumps({**ident,"rounds":rounds},indent=2,sort_keys=True)); return raw


def holm(p):
    p=np.asarray(p,float); o=np.argsort(p); out=np.empty_like(p); prev=0.; m=len(p)
    for r,i in enumerate(o):
        v=max(prev,min(1.,(m-r)*p[i])); out[i]=v; prev=v
    return out


def main():
    OUT.mkdir(parents=True,exist_ok=True); must(LOCK.exists(),"missing structural holdout lock")
    manifest=json.loads(LOCK.read_text()); must(manifest.get("performance_seen") is False,"lock says performance already seen"); must(manifest.get("selection_locked_before_performance") is True,"selection not locked")
    core_hash=sha(PROJECT/"dtrar_final_core.py"); must(core_hash==manifest["final_core_sha256"],"final core changed after lock")
    da=json.loads((DEV/"FINAL_DEVELOPMENT_ASSESSMENT.json").read_text()); ra=json.loads((ROB/"FINAL_ROBUSTNESS_ASSESSMENT.json").read_text()); must(da.get("pass") is True and ra.get("pass") is True,"prerequisites no longer pass")
    datasets=list(manifest["selected_fresh_holdouts"]); must(len(datasets)==3,"expected exactly three locked holdouts")
    print("FINAL_FRESH_CONFIRMATION_BATCH14_BEGIN datasets=3 seeds=10 source_frozen=true",flush=True)
    seedrows=[]; winframes=[]
    with threadpool_limits(limits=1):
        for ds in datasets:
            data=load_locked(ds,manifest)
            for seed in SEEDS:
                print(f"FINAL_FRESH_JOB_BEGIN dataset={ds} seed={seed}",flush=True)
                raw=train_xgb(data,seed,manifest["datasets"][ds]["locked_sha256"])
                s,w=fc.run_stream(data,raw,seed,"xgboost_fresh",ds,budget_rate=fc.BUDGET_RATE,schedule_mode="balanced",sample_mode="geometry",action_mode="riskopt",exact_h=True)
                seedrows.append(s); winframes.append(w)
                print("FINAL_FRESH_JOB_RESULT "+json.dumps(native({"dataset":ds,"seed":seed,"gain":s["brier_gain"],"raw_brier":s["raw_brier"],"max_H":s["max_exact_H"]}),sort_keys=True),flush=True)
    seedwise=pd.DataFrame(seedrows); windows=pd.concat(winframes,ignore_index=True); rows=[]
    for ds,g in seedwise.groupby("dataset"):
        x=g.brier_gain.to_numpy(float); rb=float(g.raw_brier.mean())
        try: wp=float(wilcoxon(x,alternative="greater").pvalue)
        except Exception: wp=float("nan")
        rows.append({"dataset":ds,"mean_gain":float(x.mean()),"sd_gain":float(x.std(ddof=1)),"min_seed_gain":float(x.min()),"positive_seeds":int((x>0).sum()),"seeds":len(x),"mean_raw_brier":rb,"relative_brier_reduction":float(x.mean()/rb) if rb>0 else np.nan,"wilcoxon_p_one_sided":wp,"max_exact_H":float(g.max_exact_H.max()),"all_budget_exact":bool(g.budget_exact.all()),"mean_nll_gain":float(g.nll_gain.mean()),"mean_accuracy_gain":float(g.accuracy_gain.mean()),"mean_macro_f1_gain":float(g.macro_f1_gain.mean()),"mean_ece_reduction":float(g.ece_reduction.mean())})
    summary=pd.DataFrame(rows); summary["holm_p_three"]=holm(summary.wilcoxon_p_one_sided.to_numpy(float))
    exact=bool(seedwise.max_exact_H.max()<=fc.EPS+5e-9); budget=bool(seedwise.budget_exact.all()); all_nonneg=bool((summary.mean_gain>=0).all()); material=int((summary.relative_brier_reduction>=0.01).sum())>=2; posseeds=int((summary.positive_seeds>=8).sum())>=2; pooled=float(seedwise.brier_gain.mean())>0
    gates={"all_exact_H_le_0p005":exact,"all_requested_canary_budget_exact_4pct":budget,"all_three_mean_brier_gain_nonnegative":all_nonneg,"at_least_two_of_three_material_relative_brier_gain_ge_1pct":bool(material),"at_least_two_of_three_positive_seeds_ge_8of10":bool(posseeds),"pooled_mean_brier_gain_positive":bool(pooled),"no_source_change_after_lock":True}
    pre=manifest["predeclared_confirmation_gates"]; must(set(pre)==set(gates),"gate names differ from lock")
    passed=bool(all(gates.values()))
    seedwise.to_csv(OUT/"FINAL_FRESH_SEEDWISE.csv",index=False); windows.to_csv(OUT/"FINAL_FRESH_WINDOW_TRACE.csv",index=False); summary.to_csv(OUT/"FINAL_FRESH_SUMMARY.csv",index=False)
    assess=native({"pass":passed,"gates":gates,"datasets":datasets,"seeds":SEEDS,"core_sha256":core_hash,"selection_locked_before_performance":True,"post_lock_method_change":False})
    (OUT/"FINAL_FRESH_ASSESSMENT.json").write_text(json.dumps(assess,indent=2,sort_keys=True))
    review=PROJECT/"DTRAR_FinalFreshConfirmation_Batch14_Review.zip"; tmp=review.with_suffix(".zip.partial")
    if tmp.exists(): tmp.unlink()
    members=[OUT/"FINAL_FRESH_SEEDWISE.csv",OUT/"FINAL_FRESH_WINDOW_TRACE.csv",OUT/"FINAL_FRESH_SUMMARY.csv",OUT/"FINAL_FRESH_ASSESSMENT.json",LOCK,PROJECT/"dtrar_final_core.py",Path(__file__)]
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:
        for p in members: z.write(p,p.name)
    with zipfile.ZipFile(tmp) as z: must(z.testzip() is None,"review CRC failure")
    tmp.replace(review)
    print("FINAL_FRESH_CONFIRMATION_ASSESSMENT "+json.dumps(gates,sort_keys=True),flush=True)
    print(f"FINAL_FRESH_CONFIRMATION_BATCH14_COMPLETE pass={str(passed).lower()} review={review.name}",flush=True)
    print(summary.to_string(index=False),flush=True)

if __name__=="__main__": main()
