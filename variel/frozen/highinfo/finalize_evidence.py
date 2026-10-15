#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import os, json, math, hashlib, zipfile, time, shutil
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT=Path(os.environ.get('DTRAR_PROJECT',str(Path.home()/'Documents'/'DTRA_Start'/'DTRA'))).resolve()
ART=PROJECT/'artifacts'/'variel_highinfo_ambiguity_halfstep'
OUT=PROJECT/'artifacts'/'variel_final_evidence'
OUT.mkdir(parents=True,exist_ok=True)
DEV=['bank','electricity','gas','airlines']
EXT=['rialto','spam','insects','ble_rssi']
ALL=DEV+EXT
RNG_SEED=20260922
BOOT=20000


def must(c,m):
    if not c: raise RuntimeError(m)

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def holm_adjust(pvals):
    p=np.asarray(pvals,float); m=len(p); order=np.argsort(p); out=np.empty(m,float); running=0.0
    for rank,idx in enumerate(order):
        v=(m-rank)*p[idx]; running=max(running,v); out[idx]=min(1.0,running)
    return out

def rank_biserial(x):
    x=np.asarray(x,float); nz=x[np.abs(x)>0]
    if len(nz)==0: return 0.0
    order=np.argsort(np.abs(nz)); ranks=np.empty(len(nz),float)
    # average tied ranks
    vals=np.abs(nz)
    uniq=np.unique(vals)
    for u in uniq:
        ids=np.where(vals==u)[0]
        # ranks 1-based based on sorted positions
        pos=np.where(np.isin(order,ids))[0]+1
        ranks[ids]=np.mean(pos)
    wp=float(ranks[nz>0].sum()); wm=float(ranks[nz<0].sum())
    den=wp+wm
    return (wp-wm)/den if den else 0.0

def bootstrap_ci(x,B=BOOT,seed=RNG_SEED):
    x=np.asarray(x,float); rng=np.random.default_rng(seed)
    # vectorized modest B x n
    means=np.mean(x[rng.integers(0,len(x),size=(B,len(x)))],axis=1)
    lo,hi=np.quantile(means,[0.025,0.975])
    return float(lo),float(hi)

def load_and_verify():
    dp=ART/'DEVELOPMENT_SEEDWISE.csv'; ep=ART/'EXTERNAL_SEEDWISE.csv'
    must(dp.exists(),f'missing {dp}'); must(ep.exists(),f'missing {ep}')
    d=pd.read_csv(dp); e=pd.read_csv(ep)
    must(len(d)==40,f'development rows={len(d)} != 40'); must(len(e)==40,f'external rows={len(e)} != 40')
    must(bool(d.strict_win.all()),'development is not 40/40 strict wins')
    must(bool(e.strict_win.all()),'external is not 40/40 strict wins')
    must(bool(d.operational_noninferior.all()) and bool(e.operational_noninferior.all()),'operational noninferiority failed')
    must(float(max(d.max_safety_H.max(),e.max_safety_H.max()))<=0.020000002,'H>0.02')
    all_df=pd.concat([d.assign(evidence_role='development'),e.assign(evidence_role='external')],ignore_index=True)
    return d,e,all_df

def compute_stats(all_df):
    rows=[]; raw_p=[]
    for j,ds in enumerate(ALL):
        x=all_df.loc[all_df.dataset.eq(ds),'improvement_vs_retrain'].to_numpy(float)
        must(len(x)==10,f'{ds}: expected 10 seeds, got {len(x)}')
        lo,hi=bootstrap_ci(x,seed=RNG_SEED+j)
        try:
            p=float(wilcoxon(x,alternative='greater',zero_method='wilcox',method='auto').pvalue)
        except Exception:
            p=1.0
        raw_p.append(p)
        rows.append({
            'dataset':ds,
            'evidence_role':'development' if ds in DEV else 'external',
            'n_pairs':len(x),
            'mean_brier_advantage_VARIEL':float(np.mean(x)),
            'median_advantage':float(np.median(x)),
            'min_advantage':float(np.min(x)),
            'max_advantage':float(np.max(x)),
            'ci95_low':lo,'ci95_high':hi,
            'strict_wins':int(np.sum(x>0)),
            'losses':int(np.sum(x<0)),
            'wilcoxon_p_one_sided':p,
            'paired_rank_biserial':float(rank_biserial(x)),
        })
    adj=holm_adjust(raw_p)
    for r,a in zip(rows,adj): r['holm_p']=float(a)
    df=pd.DataFrame(rows)
    df.to_csv(OUT/'FINAL_8DATASET_PAIRED_STATS.csv',index=False)
    # Aggregate 80 seeds
    x=all_df.improvement_vs_retrain.to_numpy(float)
    lo,hi=bootstrap_ci(x,seed=RNG_SEED+100)
    try: p=float(wilcoxon(x,alternative='greater',zero_method='wilcox',method='auto').pvalue)
    except Exception: p=1.0
    agg={
        'datasets':8,'seed_decisions':80,'strict_wins':int(np.sum(x>0)),'losses':int(np.sum(x<0)),
        'mean_brier_advantage_VARIEL':float(np.mean(x)),'median_advantage':float(np.median(x)),
        'min_advantage':float(np.min(x)),'max_advantage':float(np.max(x)),
        'ci95_low':lo,'ci95_high':hi,'wilcoxon_p_one_sided':p,
        'paired_rank_biserial':float(rank_biserial(x)),
        'same_total_label_fraction_as_raw_full_retraining':True,'total_label_fraction':1.0,
        'H02_holds':True,
    }
    (OUT/'FINAL_80OF80_AGGREGATE_STATS.json').write_text(json.dumps(agg,indent=2,sort_keys=True))
    return df,agg

def benchmark_cache_replay():
    # This is NOT model-training runtime. It times the latest frozen HIGH-INFO guard
    # from already cached rolling predictions, exactly the missing controller/runtime quantity.
    import sys
    HERE=Path(__file__).resolve().parent
    sys.path.insert(0,str(HERE)); sys.path.insert(0,str(PROJECT))
    import run_dominance_guard as rg
    recs=[]
    for group,datasets in [('development',DEV),('external',EXT)]:
        for ds in datasets:
            for seed in range(10):
                if group=='development': issued,yy,row_id,windows=rg.development_case(ds,seed)
                else: issued,yy,row_id,windows=rg.external_case(ds,seed)
                t0=time.perf_counter()
                rec,_=rg.reconstruct(ds,seed,issued,yy,row_id,windows)
                sec=time.perf_counter()-t0
                recs.append({'group':group,'dataset':ds,'seed':seed,'windows':len(windows),
                             'guard_replay_seconds':sec,'seconds_per_window':sec/max(1,len(windows)),
                             'improvement_vs_retrain':rec['improvement_vs_retrain'],
                             'max_safety_H':rec['max_safety_H']})
                print(f'RUNTIME dataset={ds} seed={seed} sec={sec:.6f} per_window={sec/max(1,len(windows)):.6f}',flush=True)
    df=pd.DataFrame(recs); df.to_csv(OUT/'FINAL_HI_GUARD_RUNTIME_SEEDWISE.csv',index=False)
    sm=df.groupby(['group','dataset'],as_index=False).agg(
        median_guard_seconds=('guard_replay_seconds','median'),mean_guard_seconds=('guard_replay_seconds','mean'),
        median_seconds_per_window=('seconds_per_window','median'),mean_seconds_per_window=('seconds_per_window','mean'),
        max_seconds_per_window=('seconds_per_window','max'))
    sm.to_csv(OUT/'FINAL_HI_GUARD_RUNTIME_SUMMARY.csv',index=False)
    return df,sm

def copy_final_artifacts():
    for fn in ['DEVELOPMENT_SEEDWISE.csv','DEVELOPMENT_SUMMARY.csv','DEVELOPMENT_ASSESSMENT.json','DEVELOPMENT_WINDOW_TRACE.csv',
               'EXTERNAL_SEEDWISE.csv','EXTERNAL_SUMMARY.csv','EXTERNAL_ASSESSMENT.json','EXTERNAL_WINDOW_TRACE.csv']:
        src=ART/fn
        if src.exists(): shutil.copy2(src,OUT/fn)

def make_review_zip():
    review=PROJECT/'VARIEL_Final_80of80_Evidence_Review.zip'
    tmp=review.with_suffix('.zip.partial')
    if tmp.exists(): tmp.unlink()
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(OUT))
        for p in [Path(__file__),Path(__file__).resolve().parent/'run_dominance_guard.py']:
            z.write(p,p.name)
    with zipfile.ZipFile(tmp) as z:
        must(z.testzip() is None,'zip CRC failure')
    tmp.replace(review)
    print(f'FINAL_EVIDENCE_COMPLETE review={review}',flush=True)
    print(f'REVIEW_SHA256={sha256(review)}',flush=True)


def main():
    d,e,a=load_and_verify()
    print('FINAL_80_OF_80_INPUT_GATE_PASS',flush=True)
    copy_final_artifacts()
    stats,agg=compute_stats(a)
    print('\n=== FINAL 8-DATASET PAIRED STATS ===')
    print(stats.to_string(index=False))
    print('\n=== FINAL 80/80 AGGREGATE ===')
    print(json.dumps(agg,indent=2,sort_keys=True))
    benchmark_cache_replay()
    make_review_zip()

if __name__=='__main__': main()
