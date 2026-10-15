#!/usr/bin/env python3
"""Batch 16: structural-only lock AFTER the liability policy is frozen.

Priority is fixed before performance: UCI 427, UCI 760, UCI 319 (MHEALTH).
UCI 427 and 760 were structurally audited in Batch 13 but were NOT among the
three datasets opened in Batch 14. Their locked files are reused if hashes match.
MHEALTH is newly acquired from the official UCI archive and audited structurally.
No predictive model or performance metric is computed here.
"""
from pathlib import Path
import hashlib,json,math,re,subprocess,urllib.request,zipfile,shutil
import numpy as np,pandas as pd
import dtrar_final_core as fc

PROJECT=Path(__file__).resolve().parent
OLD=PROJECT/'artifacts'/'dtrar_final_holdout_audit'/'FINAL_HOLDOUT_LOCK.json'
POLICY=PROJECT/'artifacts'/'dtrar_liability_frontier_batch15'/'LIABILITY_POLICY_FREEZE.json'
OUT=PROJECT/'artifacts'/'dtrar_liability_fresh_audit_batch16'; LOCKED=OUT/'locked'; CACHE=PROJECT/'data'/'liability_fresh_sources'
MAX_ROWS=250000
PRIORITY=['uci427_older_activity','uci760_multivariate_gait','uci319_mhealth']

def must(c,m):
    if not c: raise RuntimeError(m)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
      for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def structural_from_locked(name,old):
    rec=old['datasets'][name]
    must(rec.get('admissible') is True,f'{name} was not structurally admissible')
    must(rec.get('performance_evaluated') is False,f'{name} was already performance-evaluated')
    must(name not in old['selected_fresh_holdouts'],f'{name} was already opened in Batch14')
    src=PROJECT/rec['locked_csv']; must(src.exists(),f'missing prior locked CSV {src}'); must(sha(src)==rec['locked_sha256'],f'hash changed {name}')
    LOCKED.mkdir(parents=True,exist_ok=True); dst=LOCKED/f'{name}.csv'; shutil.copy2(src,dst); must(sha(dst)==rec['locked_sha256'],'copy hash mismatch')
    r=dict(rec); r['locked_csv']=str(dst.relative_to(PROJECT)); r['continued_from_batch13_unopened']=True; r['performance_evaluated']=False
    return r

def download_mhealth():
    CACHE.mkdir(parents=True,exist_ok=True); p=CACHE/'uci_319_mhealth.zip'
    urls=['https://archive.ics.uci.edu/static/public/319/mhealth%2Bdataset.zip','https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip']
    if not p.exists() or p.stat().st_size==0:
      ok=False; last=None
      for url in urls:
        tmp=p.with_suffix('.partial')
        if tmp.exists():tmp.unlink()
        try:
          req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 DTRAR structural audit'})
          with urllib.request.urlopen(req,timeout=300) as r,open(tmp,'wb') as f:
            while True:
              b=r.read(1024*1024)
              if not b:break
              f.write(b)
          with zipfile.ZipFile(tmp) as z: must(z.testzip() is None,'MHEALTH ZIP CRC failure')
          tmp.replace(p); ok=True; break
        except Exception as e:
          last=e
          if tmp.exists():tmp.unlink()
      must(ok,f'could not download official MHEALTH archive: {last!r}')
    return p

def audit_mhealth():
    rec={'id':319,'name':'uci319_mhealth','admissible':False,'performance_evaluated':False}
    try:
      p=download_mhealth(); zsha=sha(p)
      with zipfile.ZipFile(p) as z:
        names=[n for n in z.namelist() if re.search(r'mhealth_subject(\d+)\.log$',n,re.I)]
        def num(n):
          m=re.search(r'subject(\d+)',n,re.I); return int(m.group(1)) if m else 999
        names=sorted(names,key=num); must(len(names)>=10,f'expected 10 MHEALTH logs, found {len(names)}')
        frames=[]
        for n in names:
          with z.open(n) as f: d=pd.read_csv(f,sep=r'\s+',header=None,engine='python')
          must(d.shape[1]>=24,f'{n}: expected >=24 columns, got {d.shape[1]}')
          d=d.iloc[:,:24].copy(); d.columns=[f'f{i+1}' for i in range(23)]+['label']; frames.append(d)
      df=pd.concat(frames,ignore_index=True); y=pd.to_numeric(df.pop('label'),errors='coerce')
      must(y.notna().all(),'MHEALTH missing/non-numeric labels'); y=y.astype(int)
      rec['repository_rows']=int(len(y)); rec['source_transport']={'transport':'official_uci_static_zip','source_zip_sha256':zsha,'files':len(names)}
      if len(y)>MAX_ROWS:
        idx=np.unique(np.linspace(0,len(y)-1,MAX_ROWS,dtype=int)); df=df.iloc[idx].reset_index(drop=True); y=y.iloc[idx].reset_index(drop=True)
        rec['thinning']={'rule':'evenly_spaced_row_index','max_rows':MAX_ROWS,'original_n':rec['repository_rows']}
      else:
        df=df.reset_index(drop=True); y=y.reset_index(drop=True); rec['thinning']={'rule':'none'}
      X=df.apply(pd.to_numeric,errors='coerce'); keep=[c for c in X.columns if X[c].notna().any() and X[c].nunique(dropna=True)>1]; X=X[keep]; must(X.shape[1]>=2,'too few predictors')
      labels=sorted(np.unique(y).tolist()); must(2<=len(labels)<=20,f'class count {len(labels)} outside 2..20')
      mp={v:i for i,v in enumerate(labels)}; yy=y.map(mp).to_numpy(int); n=len(yy); a=int(math.floor(.50*n)); b=int(math.floor(.65*n))
      tr=set(np.unique(yy[:a])); sc=set(np.unique(yy[a:b])); tg=set(np.unique(yy[b:])); must(tg.issubset(tr),'target class absent from training'); must(tg.issubset(sc),'target class absent from source'); must(len(tg)>=2,'target single-class')
      windows=[np.asarray(w,int) for w in np.array_split(np.arange(b,n),12)]; sched,B=fc.horizon_schedule(windows,fc.BUDGET_RATE,'balanced'); must(min(sched.values())>=fc.MIN_CANARY,'window cannot support minimum canary')
      lockdf=X.copy(); lockdf['__target__']=yy; lockdf['__sequence__']=np.arange(n,dtype=np.int64); LOCKED.mkdir(parents=True,exist_ok=True); lp=LOCKED/'uci319_mhealth.csv'; lockdf.to_csv(lp,index=False)
      rec.update({'admissible':True,'n':n,'features':int(X.shape[1]),'classes':[str(v) for v in labels],'train_n':a,'source_n':b-a,'target_n':n-b,'target_windows':12,'canary_budget_labels':B,'min_window_canary':int(min(sched.values())),'locked_csv':str(lp.relative_to(PROJECT)),'locked_sha256':sha(lp),'ordering':'official subject1..subject10 file order; within-file acquisition order; deterministic even-row-index thinning only if n>250000'})
    except Exception as e:rec['rejection_reason']=repr(e)
    return rec

def main():
    OUT.mkdir(parents=True,exist_ok=True); LOCKED.mkdir(parents=True,exist_ok=True)
    must(POLICY.exists(),'Batch15 did not freeze a liability policy'); policy=json.loads(POLICY.read_text()); eps=float(policy['primary_epsilon']); core_hash=sha(PROJECT/'dtrar_final_core.py'); must(core_hash==policy['core_sha256'],'core changed after liability freeze')
    must(OLD.exists(),'missing Batch13 structural lock'); old=json.loads(OLD.read_text())
    print(f'LIABILITY_FRESH_AUDIT_BATCH16_BEGIN performance_evaluation=false primary_epsilon={eps}',flush=True)
    recs=[]
    for name in PRIORITY[:2]:
      r=structural_from_locked(name,old); recs.append(r); print(f'LIABILITY_FRESH_AUDIT dataset={name} admissible=true reused_unopened=true',flush=True)
    r=audit_mhealth(); recs.append(r); print(f"LIABILITY_FRESH_AUDIT dataset=uci319_mhealth admissible={str(r['admissible']).lower()} reason={r.get('rejection_reason','OK')}",flush=True)
    must(all(r['admissible'] for r in recs),'one of three predeclared fresh datasets failed structural audit; stop before performance')
    gates={'all_exact_H_le_primary_epsilon':True,'all_requested_canary_budget_exact_4pct':True,'all_three_mean_brier_gain_nonnegative':True,'at_least_two_of_three_relative_brier_gain_ge_1pct':True,'at_least_two_of_three_positive_seeds_ge_8of10':True,'pooled_mean_brier_gain_positive':True,'no_source_or_epsilon_change_after_lock':True}
    manifest={'selection_locked_before_performance':True,'performance_seen':False,'priority_order':PRIORITY,'selected_fresh_holdouts':PRIORITY,'datasets':{r['name']:r for r in recs},'final_core_sha256':core_hash,'liability_policy':policy,'predeclared_confirmation_gates':gates,'seeds':list(range(10)),'base_model_family':'xgboost'}
    (OUT/'LIABILITY_FRESH_LOCK.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)); pd.DataFrame([{k:v for k,v in r.items() if k not in {'classes','source_transport'}} for r in recs]).to_csv(OUT/'LIABILITY_FRESH_AUDIT.csv',index=False)
    review=PROJECT/'DTRAR_LiabilityFreshAudit_Batch16_Review.zip'; tmp=review.with_suffix('.zip.partial')
    if tmp.exists():tmp.unlink()
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
      z.write(OUT/'LIABILITY_FRESH_LOCK.json','LIABILITY_FRESH_LOCK.json'); z.write(OUT/'LIABILITY_FRESH_AUDIT.csv','LIABILITY_FRESH_AUDIT.csv'); z.write(Path(__file__),Path(__file__).name)
      for r in recs:
        p=PROJECT/r['locked_csv']; z.write(p,f'locked/{p.name}')
    with zipfile.ZipFile(tmp) as z:must(z.testzip() is None,'CRC failure')
    tmp.replace(review)
    print('LIABILITY_FRESH_SELECTION_LOCKED datasets='+','.join(PRIORITY),flush=True); print(f'LIABILITY_FRESH_AUDIT_BATCH16_COMPLETE performance_evaluation=false review={review.name}',flush=True)
if __name__=='__main__':main()
