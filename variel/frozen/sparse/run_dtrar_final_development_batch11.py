#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import hashlib, json, math, zipfile
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from threadpoolctl import threadpool_limits

import dtrar_final_core as fc

PROJECT = Path(__file__).resolve().parent
OUT = PROJECT / "artifacts" / "dtrar_final_development_batch11"
SEEDS = list(range(10))
PRIMARY = [
    ("xgboost","electricity"),("xgboost","bank"),("xgboost","gas"),("xgboost","insects"),
    ("catboost","electricity"),("catboost","bank"),("catboost","gas"),
    ("ft_transformer","electricity"),("ft_transformer","bank"),("ft_transformer","gas"),
]
EXTRAS = [
    ("xgboost_seen_unseen","occupancy_detection_v2"),
    ("xgboost_seen_unseen","eeg_eye_state"),
    ("xgboost_consumed_holdout","pirvision_uci1101"),
    ("xgboost_consumed_holdout","ai4i_uci601"),
]
VARIANTS = {
    "final": dict(schedule_mode="balanced", sample_mode="geometry", action_mode="riskopt", exact_h=True),
    "uniform_canary": dict(schedule_mode="balanced", sample_mode="uniform", action_mode="riskopt", exact_h=True),
    "terminal_blind": dict(schedule_mode="legacy_terminal_blind", sample_mode="geometry", action_mode="riskopt", exact_h=True),
    "maximal_safe": dict(schedule_mode="balanced", sample_mode="geometry", action_mode="maxsafe", exact_h=True),
    "no_exact_H": dict(schedule_mode="balanced", sample_mode="geometry", action_mode="riskopt", exact_h=False),
}


def must(c,m):
    if not c: raise RuntimeError(m)

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def native(x):
    if isinstance(x,dict): return {str(k):native(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [native(v) for v in x]
    if isinstance(x,np.bool_): return bool(x)
    if isinstance(x,np.integer): return int(x)
    if isinstance(x,np.floating): return float(x)
    return x

def patch_newest_cache(ro):
    core=ro.core
    original=core.frozen_cache
    g=original.__globals__
    ROOT=g.get("ROOT", getattr(core,"ROOT", None))
    file_hash=g.get("file_hash", getattr(core,"file_hash", None))
    probabilities=g.get("probabilities", getattr(core,"probabilities", lambda x:x))
    def resolver(root,data,family,seed,profile,hashes):
        base=Path(root)/"models"/data.name/family
        matches=[]
        for cfile in base.glob(f"seed_{seed}_{profile}_*/COMPLETE.json"):
            try: rec=json.loads(cfile.read_text())
            except Exception: continue
            ok=(rec.get("profile")==profile and rec.get("seed")==seed and rec.get("data_sha256")==hashes and not rec.get("sensitivity",False))
            if ROOT is not None and file_hash is not None:
                ok=ok and rec.get("models_code_sha256")==file_hash(ROOT/"dtra/models.py") and rec.get("data_code_sha256")==file_hash(ROOT/"dtra/data.py")
            if ok: matches.append(cfile.parent)
        if not matches:
            return original(root,data,family,seed,profile,hashes)
        matches.sort(key=lambda p:((p/"COMPLETE.json").stat().st_mtime_ns,p.name),reverse=True)
        folder=matches[0]
        with np.load(folder/"frozen_probabilities.npz") as saved:
            raw=probabilities(saved["probabilities"])
        must(raw.shape==(len(data.y),len(data.classes)),f"bad cache shape {data.name}/{family}/{seed}")
        return folder,raw
    core.frozen_cache=resolver


def load_job(family, ds, seed):
    import run_dtrar_risk_optimal_final as ro
    patch_newest_cache(ro)
    if family in {"xgboost","catboost","ft_transformer"}:
        if ds=="insects":
            import run_dtrar_stage2_primary_fixed as stage2
            data=stage2.load_dataset_ext(ds, ro.BASE/"data"/"raw")
            hashes=stage2.source_hashes_ext(ro.BASE/"data"/"raw", ds)
            _,raw=stage2.frozen_cache_ext(ro.BASE/"artifacts",data,family,seed,"primary",hashes)
        else:
            data=ro.core.load_dataset(ds, ro.BASE/"data"/"raw")
            hashes=ro.core.source_hashes(ro.BASE/"data"/"raw", ds)
            _,raw=ro.core.frozen_cache(ro.BASE/"artifacts",data,family,seed,"primary",hashes)
        return data,np.asarray(raw,float)

    if family=="xgboost_seen_unseen":
        import run_dtrar_unseen_confirmation_xgb as u
        if u.MANIFEST is None: u.MANIFEST=u.load_manifest()
        data=u.load_unseen(ds)
        _,raw=u.train_or_reuse_xgb(data,seed)
        return data,np.asarray(raw,float)

    if family=="xgboost_consumed_holdout":
        import run_dtrar_maxsafe_fresh_holdout as h
        if h.MANIFEST is None:
            must(h.LOCK.exists(),"missing consumed holdout lock")
            h.MANIFEST=json.loads(h.LOCK.read_text())
        data=h.load_data(ds)
        _,raw=h.train_or_reuse(data,seed)
        return data,np.asarray(raw,float)
    raise ValueError((family,ds))


def holm(p):
    p=np.asarray(p,float); order=np.argsort(p); out=np.empty_like(p); prev=0.; m=len(p)
    for r,i in enumerate(order):
        v=max(prev,min(1.,(m-r)*p[i])); out[i]=v; prev=v
    return out


def summarize(seedwise):
    rows=[]
    for (variant,fam,ds),g in seedwise.groupby(["variant","family","dataset"]):
        x=g.brier_gain.to_numpy(float)
        try: wp=float(wilcoxon(x,alternative="greater").pvalue)
        except Exception: wp=float("nan")
        rows.append({
            "variant":variant,"family":fam,"dataset":ds,"mean_gain":float(x.mean()),
            "sd_gain":float(x.std(ddof=1)),"min_seed_gain":float(x.min()),
            "positive_seeds":int((x>0).sum()),"nonnegative_seeds":int((x>=-1e-12).sum()),
            "seeds":len(x),"wilcoxon_p_one_sided":wp,
            "mean_nll_gain":float(g.nll_gain.mean()),"mean_accuracy_gain":float(g.accuracy_gain.mean()),
            "mean_macro_f1_gain":float(g.macro_f1_gain.mean()),"mean_ece_reduction":float(g.ece_reduction.mean()),
            "max_exact_H":float(g.max_exact_H.max()),"all_budget_exact":bool(g.budget_exact.all()),
            "mean_act_rate":float(g.mean_act_rate.mean()),"mean_eta_when_acted":float(g.mean_eta_when_acted.mean()),
        })
    s=pd.DataFrame(rows)
    mask=(s.variant=="final") & s.apply(lambda r:(r.family,r.dataset) in PRIMARY,axis=1)
    idx=s.index[mask].tolist()
    if idx:
        adj=holm(s.loc[idx,"wilcoxon_p_one_sided"].to_numpy(float))
        s.loc[idx,"holm_p_primary10"]=adj
    return s


def assess(summary, seedwise):
    f=summary[summary.variant.eq("final")].copy()
    def c(fam,ds):
        z=f[(f.family==fam)&(f.dataset==ds)]; must(len(z)==1,f"missing {fam}/{ds}"); return z.iloc[0]
    cells={k:c(*k) for k in PRIMARY+EXTRAS}
    primary_pos=all(float(cells[k].mean_gain)>0 for k in PRIMARY)
    primary_seed_strength=all(int(cells[k].positive_seeds)>=8 for k in PRIMARY)
    bg_material=all(float(cells[k].mean_gain)>=0.002 for k in [
        ("xgboost","bank"),("xgboost","gas"),("catboost","bank"),("catboost","gas"),
        ("ft_transformer","bank"),("ft_transformer","gas")])
    occ=float(cells[("xgboost_seen_unseen","occupancy_detection_v2")].mean_gain)>=0.001
    eeg=float(cells[("xgboost_seen_unseen","eeg_eye_state")].mean_gain)>=0.0
    consumed=all(float(cells[k].mean_gain)>=-1e-4 for k in [
        ("xgboost_consumed_holdout","pirvision_uci1101"),("xgboost_consumed_holdout","ai4i_uci601")])
    positives=sum(float(cells[k].mean_gain)>0 for k in PRIMARY+EXTRAS)
    finalrows=seedwise[seedwise.variant.eq("final")]
    exact=bool(finalrows.max_exact_H.max()<=fc.EPS+5e-9)
    budget=bool(finalrows.budget_exact.all())
    gates={
        "all_exact_contracts_hold":exact,
        "explicit_4pct_budget_exact":budget,
        "all_10_primary_cells_positive":bool(primary_pos),
        "all_10_primary_cells_at_least_8of10_positive_seeds":bool(primary_seed_strength),
        "bank_gas_material_all_three_backbones":bool(bg_material),
        "occupancy_material_and_eeg_nonharm":bool(occ and eeg),
        "consumed_pirvision_ai4i_nonharm":bool(consumed),
        "at_least_12_of_14_cells_positive":bool(positives>=12),
    }
    return gates,bool(all(gates.values()))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    print("FINAL_DEVELOPMENT_BATCH11_BEGIN method=HB-GO-Canary-RAPID fully_causal_candidate=true target_labels=current_canary_only budget_rate=0.04",flush=True)
    seed_rows=[]; win_frames=[]
    cells=PRIMARY+EXTRAS
    with threadpool_limits(limits=1):
        nall=len(cells)*len(SEEDS)*len(VARIANTS); n=0
        cache={}
        for fam,ds in cells:
            for seed in SEEDS:
                data,raw=load_job(fam,ds,seed)
                for variant,kw in VARIANTS.items():
                    n+=1
                    print(f"FINAL_DEV_JOB_BEGIN {n}/{nall} variant={variant} family={fam} dataset={ds} seed={seed}",flush=True)
                    s,w=fc.run_stream(data,raw,seed,fam,ds,budget_rate=fc.BUDGET_RATE,**kw)
                    s["variant"]=variant
                    w["variant"]=variant
                    seed_rows.append(s); win_frames.append(w)
                    print("FINAL_DEV_JOB_RESULT "+json.dumps(native({"variant":variant,"family":fam,"dataset":ds,"seed":seed,"gain":s["brier_gain"],"max_H":s["max_exact_H"],"budget_exact":s["budget_exact"]}),sort_keys=True),flush=True)
    seedwise=pd.DataFrame(seed_rows); windows=pd.concat(win_frames,ignore_index=True); summary=summarize(seedwise)
    seedwise.to_csv(OUT/"FINAL_DEVELOPMENT_SEEDWISE.csv",index=False)
    windows.to_csv(OUT/"FINAL_DEVELOPMENT_WINDOW_TRACE.csv",index=False)
    summary.to_csv(OUT/"FINAL_DEVELOPMENT_SUMMARY.csv",index=False)
    gates,passed=assess(summary,seedwise)
    core_hash=sha(PROJECT/"dtrar_final_core.py"); runner_hash=sha(Path(__file__))
    assessment=native({"method":"HB-GO-Canary-RAPID","pass":passed,"gates":gates,
        "nominal_budget_rate":fc.BUDGET_RATE,"epsilon":fc.EPS,"audit_alpha":fc.AUDIT_ALPHA,
        "candidate_information":"q*_t uses source + observed canary labels from windows < t only",
        "current_information":"window-t canary labels estimate only the frozen q*_t action on disjoint bulk",
        "core_sha256":core_hash,"runner_sha256":runner_hash,"fresh_holdouts_opened":False})
    (OUT/"FINAL_DEVELOPMENT_ASSESSMENT.json").write_text(json.dumps(assessment,indent=2,sort_keys=True))
    if passed:
        (OUT/"FINAL_METHOD_FREEZE.json").write_text(json.dumps(native({
            "frozen":True,"core_sha256":core_hash,"runner_sha256":runner_hash,
            "budget_rate":fc.BUDGET_RATE,"epsilon":fc.EPS,"audit_alpha":fc.AUDIT_ALPHA,
            "max_strata":fc.MAX_STRATA,"min_canary":fc.MIN_CANARY,
            "information_pattern":"horizon-balanced geometry-optimal current canary -> risk-optimal disjoint bulk -> exact H clipping; same canary labels train next candidate",
            "fresh_holdout_performance_seen":False}),indent=2,sort_keys=True))
    review=PROJECT/"DTRAR_FinalDevelopment_Batch11_Review.zip"; tmp=review.with_suffix(".zip.partial")
    if tmp.exists(): tmp.unlink()
    members=[OUT/"FINAL_DEVELOPMENT_SEEDWISE.csv",OUT/"FINAL_DEVELOPMENT_WINDOW_TRACE.csv",OUT/"FINAL_DEVELOPMENT_SUMMARY.csv",OUT/"FINAL_DEVELOPMENT_ASSESSMENT.json",PROJECT/"dtrar_final_core.py",Path(__file__)]
    if (OUT/"FINAL_METHOD_FREEZE.json").exists(): members.append(OUT/"FINAL_METHOD_FREEZE.json")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:
        for p in members: z.write(p,p.name)
    with zipfile.ZipFile(tmp) as z: must(z.testzip() is None,"review CRC failure")
    tmp.replace(review)
    print("FINAL_DEVELOPMENT_BATCH11_ASSESSMENT "+json.dumps(gates,sort_keys=True),flush=True)
    print(f"FINAL_DEVELOPMENT_BATCH11_COMPLETE pass={str(passed).lower()} review={review.name} core_sha256={core_hash}",flush=True)
    print(summary[summary.variant.eq("final")].to_string(index=False),flush=True)

if __name__=="__main__": main()
