#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import os,sys,io,json,math,time,zipfile,hashlib,gzip,pickle
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon,rankdata

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as cc

PROJECT=Path(os.environ.get("DTRAR_PROJECT",str(Path.home()/"Documents"/"DTRA_Start"/"DTRA"))).resolve()
V10=PROJECT/"VARIEL_FINAL_CODE_FREEZE_v10_Review.zip"
COMPASS=PROJECT/"VARIEL_COMPASS_SR_FINAL_Review.zip"
OUT=PROJECT/"artifacts"/"variel_xgboost_ltt_mc_final_v13"
V12_CACHE=PROJECT/"artifacts"/"variel_xgboost_ltt_mc_final_v12"/"xgb_static_cache"
RAW_CACHE=V12_CACHE if V12_CACHE.exists() else (OUT/"xgb_static_cache")
OUT.mkdir(parents=True,exist_ok=True); RAW_CACHE.mkdir(parents=True,exist_ok=True)
REVIEW=PROJECT/"VARIEL_XGBOOST_LTT_MC_FINAL_v13_Review.zip"

DATASETS=["gas","rialto","insects","ble_rssi"]
SEEDS=list(range(10))
TEMPS=[0.50,0.67,0.80,1.25,1.50,2.00]
ALPHA=.05; TOL=1e-12; AUDIT5=.05

def must(c,m):
    if not c: raise RuntimeError(m)

def read_zip_csv(path,name):
    with zipfile.ZipFile(path) as z:return pd.read_csv(io.BytesIO(z.read(name)))

def read_zip_json(path,name):
    with zipfile.ZipFile(path) as z:return json.loads(z.read(name))


def largest_counts(sizes,rate):
    sizes=np.asarray(sizes,int)
    B=int(math.floor(float(rate)*int(sizes.sum())))
    raw=B*sizes/max(1,int(sizes.sum()))
    a=np.floor(raw).astype(int)
    remain=B-int(a.sum())
    if remain>0:
        for j in np.argsort(-(raw-a),kind="mergesort")[:remain]:
            a[j]+=1
    return a,B

def geometry_canary_local(p,m,seed):
    p=cc.probs(p)
    n=len(p)
    if n<=1:
        return np.asarray([0],int)[:min(max(0,int(m)),n)]
    m=min(max(1,int(m)),n-1)
    ent=-np.sum(p*np.log(np.clip(p,1e-12,1.0)),axis=1)
    s=np.sort(p,axis=1)
    margin=s[:,-1]-s[:,-2] if p.shape[1]>1 else np.zeros(n)
    score=ent+1.0-margin
    order=np.argsort(score,kind="mergesort")
    H=min(8,m,n)
    chunks=[np.asarray(x,int) for x in np.array_split(order,H) if len(x)]
    masses=np.asarray([len(x)*max(float(np.ptp(score[x])),1e-6) for x in chunks],float)
    caps=np.asarray([len(x) for x in chunks],int)
    alloc=np.zeros(len(chunks),int)
    for _ in range(m):
        z=np.where(alloc<caps,masses/(alloc+1.0),-1.0)
        alloc[int(np.argmax(z))]+=1
    rng=np.random.default_rng(seed)
    out=[]
    for pop,a in zip(chunks,alloc):
        q=pop.copy()
        rng.shuffle(q)
        out.extend(q[:int(a)])
    return np.sort(np.asarray(out,int))

def brier_rows(p,y):
    p=cc.probs(p);y=np.asarray(y,int)
    z=np.zeros_like(p);z[np.arange(len(y)),y]=1.
    return np.sum((p-z)**2,axis=1)

def brier(p,y):
    return float(np.mean(brier_rows(p,y)))

def temp(p,T):
    p=cc.probs(p);z=np.log(np.clip(p,1e-12,1.))/float(T);z-=z.max(axis=1,keepdims=True)
    return cc.probs(np.exp(z))

def holm_reject(names,pvals,alpha=.05):
    order=np.argsort(pvals);rej=[]
    for rank,j in enumerate(order):
        if pvals[j] <= alpha/(len(pvals)-rank): rej.append(names[j])
        else: break
    return set(rej)

def ltt_window(raw,y,can):
    raw=cc.probs(raw); y=np.asarray(y,int); can=np.asarray(can,int); n=max(1,len(can))
    rawloss=brier_rows(raw[can],y[can]); names=[];pvals=[];cand={}
    for T in TEMPS:
        q=temp(raw,T);d=rawloss-brier_rows(q[can],y[can]);g=float(np.mean(d))
        pv=math.exp(-n*max(0.,g)**2/8.) if g>0 else 1.
        nm=f"T={T:g}";names.append(nm);pvals.append(min(1.,pv));cand[nm]=(q,g,float(np.mean(brier_rows(q[can],y[can]))))
    rejected=holm_reject(names,pvals,ALPHA)
    if not rejected:return raw,"STATIC",0,0.0
    chosen=min(rejected,key=lambda n:(cand[n][2],n))
    q=cand[chosen][0].copy(); q[can]=raw[can]
    return q,chosen,len(rejected),cand[chosen][1]

def raw_cache_path(ds,seed):
    return RAW_CACHE/f"{ds}_seed{seed}.pkl.gz"

def build_or_reuse_static_xgb(ds,seed):
    """Exact Compass XGBoost static/source model, checkpointed per seed."""
    cp=raw_cache_path(ds,seed)
    if cp.exists():
        with gzip.open(cp,"rb") as f:r=pickle.load(f)
        return r,True
    st=cc.load_stream(ds)
    rounds=cc.choose_rounds(st,seed)
    model,imp,trainsec=cc.fit_model(st,np.asarray(st.source,int),seed,rounds)
    preds=[];ys=[];windows=[]
    for ix0 in st.windows:
        ix=np.asarray(ix0,int)
        p=cc.pred(model,imp,st.X[ix])
        preds.append(p);ys.append(np.asarray(st.y[ix],int));windows.append(ix)
    r={
        "dataset":ds,"seed":seed,"rounds":int(rounds),"training_seconds":float(trainsec),
        "raw":np.vstack(preds),"y":np.concatenate(ys),
        "window_lengths":[len(w) for w in windows],
        "classes":list(st.classes)
    }
    tmp=cp.with_suffix(".partial")
    with gzip.open(tmp,"wb") as f:pickle.dump(r,f,pickle.HIGHEST_PROTOCOL)
    tmp.replace(cp)
    return r,False

def xgb_seed(ds,seed):
    r,cached=build_or_reuse_static_xgb(ds,seed)
    raw=cc.probs(r["raw"]); y=np.asarray(r["y"],int); lengths=r["window_lengths"]
    counts,B=largest_counts(lengths,AUDIT5)
    ps=[];ys=[];choices=[];sig=0;gains=[];offset=0
    t0=time.perf_counter()
    for t,n in enumerate(lengths):
        p=raw[offset:offset+n];yy=y[offset:offset+n]
        can=geometry_canary_local(p,counts[t],cc.stable_seed("xb5","xgboost",ds,seed,t))
        q,ch,k,g=ltt_window(p,yy,can)
        ps.append(q);ys.append(yy);choices.append(ch);sig+=k;gains.append(g);offset+=n
    sec=time.perf_counter()-t0
    return {
        "backbone":"xgboost","dataset":ds,"seed":seed,
        "ltt_mc_brier":brier(np.vstack(ps),np.concatenate(ys)),
        "ltt_selected_windows":int(sum(c!="STATIC" for c in choices)),
        "windows":len(lengths),
        "significant_hypotheses_total":sig,
        "mean_selected_audit_gain":float(np.mean(gains)),
        "selector_seconds":sec,
        "selector_seconds_per_window":sec/max(1,len(lengths)),
        "label_fraction":B/sum(lengths),
        "dominant_choice":pd.Series(choices).value_counts().index[0],
        "xgb_training_seconds":r["training_seconds"],
        "xgb_rounds":r["rounds"],
        "static_cache_reused":bool(cached)
    }

def rb(a):
    a=np.asarray(a,float);z=a[np.abs(a)>1e-15]
    if not len(z):return 0.
    r=rankdata(np.abs(z));return float((r[z>0].sum()-r[z<0].sum())/r.sum())

def boot(a,seed,B=20000):
    a=np.asarray(a,float);rng=np.random.default_rng(seed)
    m=rng.choice(a,(B,len(a)),replace=True).mean(1)
    return tuple(np.quantile(m,[.025,.975]))

def sseed(*x):
    h=hashlib.sha256("|".join(map(str,x)).encode()).digest()
    return int.from_bytes(h[:8],"little")%(2**32-1)

def summarize(p):
    rows=[]
    for (bb,ds),g in p.groupby(["backbone","dataset"],sort=True):
        a=g.variel_advantage.to_numpy(float);lo,hi=boot(a,sseed("v13",bb,ds))
        try:wp=float(wilcoxon(a,alternative="greater").pvalue)
        except Exception:wp=float("nan")
        rows.append({
            "backbone":bb,"dataset":ds,"pairs":len(g),
            "mean_LTT_MC_brier":float(g.ltt_mc_brier.mean()),
            "mean_VARIEL_brier":float(g.variel_brier.mean()),
            "mean_VARIEL_advantage":float(a.mean()),"min_VARIEL_advantage":float(a.min()),
            "ci95_low":lo,"ci95_high":hi,
            "wins":int((a>TOL).sum()),"losses":int((a<-TOL).sum()),"ties":int((np.abs(a)<=TOL).sum()),
            "wilcoxon_p":wp,"rank_biserial":rb(a),
            "mean_LTT_label_fraction":float(g.label_fraction.mean()),
            "mean_LTT_selector_seconds_per_window":float(g.selector_seconds_per_window.mean()),
            "selected_windows_total":int(g.ltt_selected_windows.sum())
        })
    return pd.DataFrame(rows)

def add_global_holm(sm):
    sm=sm.copy();pv=sm.wilcoxon_p.to_numpy(float);order=np.argsort(pv);adj=np.empty(len(pv));run=0.
    for rank,j in enumerate(order):
        val=min(1.,(len(pv)-rank)*pv[j]);run=max(run,val);adj[j]=run
    sm["holm_p"]=adj
    return sm

def main():
    a10=read_zip_json(V10,"LTT_MC_ASSESSMENT.json")
    must(bool(a10.get("perfect")) and int(a10.get("VARIEL_wins",-1))==80 and int(a10.get("VARIEL_losses",-1))==0,
         "V10 CatBoost/FT LTT-MC is not the expected 80/80 closure")
    old80=read_zip_csv(V10,"LTT_MC_CROSS_BACKBONE_SEEDWISE.csv")

    cs=read_zip_csv(COMPASS,"COMPASS_SR_VS_BASELINES_SEEDWISE.csv")
    xv=cs[(cs.baseline=="UPF") & cs.dataset.isin(DATASETS)][
        ["dataset","seed","compass_brier","compass_label_fraction","max_H","source"]
    ].rename(columns={"compass_brier":"variel_brier","compass_label_fraction":"variel_label_fraction"})
    must(len(xv)==40,f"Expected 40 frozen XGBoost VARIEL rows, got {len(xv)}")

    rows=[];done=0
    for ds in DATASETS:
        for s in SEEDS:
            rec=xgb_seed(ds,s);rows.append(rec);done+=1
            print(f"XGB_LTT_MC_DONE {done}/40 dataset={ds} seed={s} static_cached={int(rec['static_cache_reused'])}",flush=True)

    x=pd.DataFrame(rows).merge(xv,on=["dataset","seed"],validate="one_to_one")
    x["variel"]="VARIEL-Compass-SR-XGBoost"
    x["variel_advantage"]=x.ltt_mc_brier-x.variel_brier
    x.to_csv(OUT/"XGBOOST_LTT_MC_SEEDWISE.csv",index=False)
    xsm=add_global_holm(summarize(x));xsm.to_csv(OUT/"XGBOOST_LTT_MC_SUMMARY.csv",index=False)

    cols=old80.columns.tolist()
    for c in cols:
        if c not in x.columns:x[c]=np.nan
    all120=pd.concat([old80[cols],x[cols]],ignore_index=True)
    must(len(all120)==120,f"Expected 120 cells, got {len(all120)}")
    all120.to_csv(OUT/"LTT_MC_ALL3_BACKBONES_SEEDWISE.csv",index=False)
    allsm=add_global_holm(summarize(all120))
    allsm.to_csv(OUT/"LTT_MC_ALL3_BACKBONES_SUMMARY.csv",index=False)
    losses=all120[all120.variel_advantage<-TOL].copy()
    losses.to_csv(OUT/"LTT_MC_ALL3_BACKBONES_LOSSES.csv",index=False)

    ass={
        "name":"Final multiclass LTT-MC closure — all three backbones",
        "datasets":DATASETS,
        "backbones":["xgboost","catboost","fttransformer"],
        "paired_cells":int(len(all120)),
        "rows":int(len(allsm)),
        "VARIEL_wins":int((all120.variel_advantage>TOL).sum()),
        "VARIEL_losses":int((all120.variel_advantage<-TOL).sum()),
        "ties":int((np.abs(all120.variel_advantage)<=TOL).sum()),
        "perfect_10of10_rows":int(((allsm.wins==10)&(allsm.losses==0)).sum()),
        "perfect":bool((all120.variel_advantage>TOL).all()),
        "all_CI_lower_positive":bool((allsm.ci95_low>0).all()),
        "all_Holm_significant":bool((allsm.holm_p<.05).all()),
        "all_rank_biserial_one":bool(np.allclose(allsm.rank_biserial,1.,atol=1e-12)),
        "XGBoost_cells":40,
        "CatBoost_FT_cells_reused_from_V10":80,
        "XGBoost_static_source":"exact Compass common.py source-only XGBoost configuration; per-seed checkpointed static predictions",
        "XGBoost_VARIEL_source":"frozen VARIEL-Compass-SR review",
        "LTT_MC_budget":"5% geometry-stratified canaries",
        "status":"multiclass Brier LTT sensitivity extension; not original binary MCGrad benchmark",
        "future_label_leakage":False
    }
    (OUT/"LTT_MC_ALL3_BACKBONES_ASSESSMENT.json").write_text(json.dumps(ass,indent=2,sort_keys=True))

    rt=read_zip_json(V10,"RUNTIME_COMPUTE_ASSESSMENT.json")
    final={
        "coding_stop_ready":bool(ass["perfect"] and ass["all_CI_lower_positive"] and ass["all_Holm_significant"] and ass["all_rank_biserial_one"]),
        "multiclass_ltt_all3":ass,
        "runtime_compute_already_frozen_from_V10":rt,
        "what_remains_after_this":"No numerical experiments. Tables, figures, manuscript, supplement, repository packaging only."
    }
    (OUT/"FINAL_STOP_CODING_ASSESSMENT.json").write_text(json.dumps(final,indent=2,sort_keys=True))

    tmp=REVIEW.with_suffix(".partial")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p.parent==OUT:z.write(p,p.name)
        z.write(Path(__file__),Path(__file__).name)
        z.write(HERE/"common.py","common.py")
    with zipfile.ZipFile(tmp) as z:
        bad=z.testzip()
        if bad:raise RuntimeError(bad)
    tmp.replace(REVIEW)

    print("=== V13 ALL-3-BACKBONE LTT-MC ASSESSMENT ===",flush=True)
    print(json.dumps(ass,indent=2,sort_keys=True),flush=True)
    print("=== FINAL STOP-CODING ASSESSMENT ===",flush=True)
    print(json.dumps(final,indent=2,sort_keys=True),flush=True)
    print("V13_COMPLETE review="+str(REVIEW),flush=True)

if __name__=="__main__":
    main()
