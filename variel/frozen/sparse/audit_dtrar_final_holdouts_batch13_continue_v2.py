#!/usr/bin/env python3
"""Continue Batch 13 after ucimlrepo transport failures.

No performance metric is computed.  This script preserves the already frozen
priority order and structural admissibility rules.  It reuses the completed
structural result for UCI 755 (rejected) and the already-written locked CSV for
UCI 760, and audits only the unresolved transport-failed candidates 780/348/196/427
from the official UCI static archives.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, json, math, re, subprocess, urllib.request, zipfile
import numpy as np
import pandas as pd

import dtrar_final_core as fc

PROJECT=Path(__file__).resolve().parent
DEV=PROJECT/'artifacts'/'dtrar_final_development_batch11'
ROB=PROJECT/'artifacts'/'dtrar_final_robustness_batch12'
OUT=PROJECT/'artifacts'/'dtrar_final_holdout_audit'
LOCKED=OUT/'locked'
CACHE=PROJECT/'data'/'final_holdout_sources'
MAX_ROWS=250000

PRIORITY=[
    {'id':755,'name':'uci755_accel_gyro_phone','target':'Activity'},
    {'id':780,'name':'uci780_har70plus','target':'label'},
    {'id':348,'name':'uci348_indoor_user_movement','target':None},
    {'id':196,'name':'uci196_localization_activity','target':'activity'},
    {'id':427,'name':'uci427_older_activity','target':'label'},
    {'id':760,'name':'uci760_multivariate_gait','target':'condition'},
]

URLS={
  780:'https://archive.ics.uci.edu/static/public/780/har70.zip',
  348:'https://archive.ics.uci.edu/static/public/348/indoor%2Buser%2Bmovement%2Bprediction%2Bfrom%2Brss%2Bdata.zip',
  196:'https://archive.ics.uci.edu/static/public/196/localization%2Bdata%2Bfor%2Bperson%2Bactivity.zip',
  427:'https://archive.ics.uci.edu/static/public/427/activity%2Brecognition%2Bwith%2Bhealthy%2Bolder%2Bpeople%2Busing%2Ba%2Bbatteryless%2Bwearable%2Bsensor.zip',
}

def must(c,m):
    if not c: raise RuntimeError(m)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def download_official(dataset_id):
    CACHE.mkdir(parents=True,exist_ok=True)
    p=CACHE/f'uci_{dataset_id}.zip'
    url=URLS[dataset_id]
    if not p.exists() or p.stat().st_size==0:
        tmp=p.with_suffix('.zip.partial')
        if tmp.exists(): tmp.unlink()
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 DTRAR structural audit'})
            with urllib.request.urlopen(req,timeout=300) as r, open(tmp,'wb') as f:
                while True:
                    b=r.read(1024*1024)
                    if not b: break
                    f.write(b)
        except Exception:
            if tmp.exists(): tmp.unlink()
            subprocess.run(['curl','-L','--fail','--retry','3','--connect-timeout','30','-o',str(tmp),url],check=True)
        with zipfile.ZipFile(tmp) as z:
            must(z.testzip() is None,f'UCI {dataset_id} ZIP CRC failure')
        tmp.replace(p)
    with zipfile.ZipFile(p) as z:
        must(z.testzip() is None,f'UCI {dataset_id} cached ZIP CRC failure')
    return p,url,sha(p)

def _read_csv_member(z,name,**kw):
    with z.open(name) as f: return pd.read_csv(f,**kw)

def load_780(z):
    names=sorted([n for n in z.namelist() if re.search(r'har70plus/[^/]+\.csv$',n,re.I)],key=lambda s:[int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)',s)])
    must(len(names)>=10,f'HAR70+ CSV files not found; first members={z.namelist()[:20]}')
    frames=[]
    for n in names:
        d=_read_csv_member(z,n)
        cols={str(c).strip().lower():c for c in d.columns}
        must('label' in cols,f'HAR70+ label missing in {n}')
        d=d.rename(columns={cols['label']:'label'})
        d['__source_file__']=Path(n).name
        frames.append(d)
    df=pd.concat(frames,ignore_index=True)
    y=df.pop('label')
    return df,y,'label',{'transport':'official_uci_static_zip','files':len(names)}

def load_196(z):
    cand=[n for n in z.namelist() if Path(n).name.lower()=='conflongdemo_jsi.txt']
    must(len(cand)==1,f'Localization data file unresolved: {cand}')
    names=['sequence_name','tag_id','timestamp','date','x','y','z','activity']
    d=_read_csv_member(z,cand[0],header=None,names=names)
    # tolerate a stray malformed line/header if present, but never missing targets
    d=d[d['activity'].notna()].reset_index(drop=True)
    y=d.pop('activity').astype('string').str.strip()
    return d,y,'activity',{'transport':'official_uci_static_zip','file':cand[0]}

def load_427(z):
    members=[n for n in z.namelist() if ('s1_dataset/' in n.lower() or 's2_dataset/' in n.lower()) and not n.endswith('/')]
    members=sorted(members,key=lambda s:[int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)',s)])
    must(members,'Older-activity data files not found')
    frames=[]
    for n in members:
        try:
            d=_read_csv_member(z,n,header=None)
        except Exception:
            continue
        if d.shape[1]<9:
            try:
                with z.open(n) as f: d=pd.read_csv(f,header=None,sep=r'\s+',engine='python')
            except Exception:
                continue
        if d.shape[1]<9: continue
        d=d.iloc[:,:9].copy()
        d.columns=['time','acc_front','acc_vertical','acc_lateral','antenna_id','rssi','phase','frequency','label']
        lab=pd.to_numeric(d['label'],errors='coerce')
        d=d[lab.notna()].copy(); d['label']=lab[lab.notna()].astype(int).to_numpy()
        if len(d):
            d['__source_file__']=n
            frames.append(d)
    must(frames,'No parseable older-activity files')
    df=pd.concat(frames,ignore_index=True)
    y=df.pop('label')
    return df,y,'label',{'transport':'official_uci_static_zip','files':len(frames)}

def load_348(z):
    members=z.namelist()
    target=[n for n in members if Path(n).name.lower()=='movementaal_target.csv']
    must(len(target)==1,'MovementAAL_target.csv unresolved')
    td=_read_csv_member(z,target[0])
    td.columns=[str(c).strip().lstrip('#') for c in td.columns]
    cols={str(c).lower():c for c in td.columns}
    sidcol=cols.get('sequence_id') or td.columns[0]
    labcol=cols.get('class_label') or td.columns[-1]
    targets={int(r[sidcol]):r[labcol] for _,r in td.iterrows()}
    rss=[]
    for n in members:
        m=re.search(r'MovementAAL_RSS_(\d+)\.csv$',n,re.I)
        if not m: continue
        sid=int(m.group(1)); must(sid in targets,f'missing target for sequence {sid}')
        d=_read_csv_member(z,n)
        d.columns=[str(c).strip().lstrip('#') for c in d.columns]
        d['__sequence_id__']=sid
        d['__step__']=np.arange(len(d),dtype=np.int64)
        d['__target__']=targets[sid]
        rss.append((sid,d))
    rss.sort(key=lambda x:x[0]); must(len(rss)>=300,f'expected ~314 RSS sequences, found {len(rss)}')
    df=pd.concat([d for _,d in rss],ignore_index=True)
    y=df.pop('__target__')
    # sequence id is structural metadata, not a predictor
    df=df.drop(columns=['__sequence_id__','__step__'],errors='ignore')
    return df,y,'class_label',{'transport':'official_uci_static_zip','sequences':len(rss)}

LOADERS={780:load_780,348:load_348,196:load_196,427:load_427}

def direct_fetch_record(spec):
    p,url,zsha=download_official(spec['id'])
    with zipfile.ZipFile(p) as z:
        X,y,target_col,meta=LOADERS[spec['id']](z)
    meta={**meta,'official_url':url,'source_zip_sha256':zsha,'repository_rows':int(len(y))}
    return X,y,target_col,meta

def numericize(X):
    out=pd.DataFrame(index=np.arange(len(X))); log=[]
    for c in X.columns:
        s=X[c]
        if pd.api.types.is_numeric_dtype(s):
            out[str(c)]=pd.to_numeric(s,errors='coerce').astype(float); log.append({'column':str(c),'transform':'numeric'}); continue
        ss=s.astype('string')
        # deterministic date parse only for obvious date/time/timestamp columns; avoids noisy warnings on categorical strings
        cname=str(c).lower()
        if any(k in cname for k in ('date','time','timestamp')):
            dt=pd.to_datetime(ss,errors='coerce',utc=True)
        else:
            dt=pd.Series(pd.NaT,index=ss.index,dtype='datetime64[ns, UTC]')
        if float(dt.notna().mean())>=0.90:
            vals=dt.astype('int64').astype(float)/1e9; vals[dt.isna()]=np.nan
            out[str(c)]=vals; log.append({'column':str(c),'transform':'datetime_unix_seconds'})
        else:
            def _h(v):
                if pd.isna(v): return np.nan
                b=hashlib.sha256(str(v).encode()).digest()[:8]
                return float(int.from_bytes(b,'little') % 1000000007)
            out[str(c)]=ss.map(_h).astype(float); log.append({'column':str(c),'transform':'stateless_sha256_category_code'})
    keep=[c for c in out.columns if out[c].notna().any() and out[c].nunique(dropna=True)>1]
    return out[keep],log

def audit_direct(spec):
    rec={'id':spec['id'],'name':spec['name'],'admissible':False,'performance_evaluated':False}
    try:
        X,y,target_col,meta=direct_fetch_record(spec)
        rec['repository_rows']=int(len(y)); rec['resolved_target_column']=target_col; rec['source_transport']=meta
        must(len(X)==len(y) and len(y)>=5000,'too few or mismatched rows')
        must(not pd.isna(y).any(),'missing target')
        if len(y)>MAX_ROWS:
            idx=np.unique(np.linspace(0,len(y)-1,MAX_ROWS,dtype=int))
            X=X.iloc[idx].reset_index(drop=True); y=y.iloc[idx].reset_index(drop=True)
            rec['thinning']={'rule':'evenly_spaced_row_index','max_rows':MAX_ROWS,'original_n':rec['repository_rows']}
        else:
            X=X.reset_index(drop=True); y=y.reset_index(drop=True); rec['thinning']={'rule':'none'}
        X,trans=numericize(X); must(X.shape[1]>=2,'fewer than two usable predictors')
        labels=sorted([str(v) for v in pd.unique(y.astype('string'))]); must(2<=len(labels)<=20,f'class count {len(labels)} outside 2..20')
        ym={v:i for i,v in enumerate(labels)}; yy=y.astype('string').map(ym).to_numpy(int)
        n=len(yy); a=int(math.floor(.50*n)); b=int(math.floor(.65*n)); windows=[np.asarray(w,int) for w in np.array_split(np.arange(b,n),12)]
        tr=set(np.unique(yy[:a]).tolist()); src=set(np.unique(yy[a:b]).tolist()); tg=set(np.unique(yy[b:]).tolist())
        must(tg.issubset(tr),'target contains class absent from training')
        must(tg.issubset(src),'target contains class absent from source/calibration')
        must(len(tg)>=2,'target is single-class')
        sched,B=fc.horizon_schedule(windows,fc.BUDGET_RATE,'balanced')
        must(min(sched.values())>=fc.MIN_CANARY,'at least one target window cannot support minimum canary')
        lockdf=X.copy(); lockdf['__target__']=yy; lockdf['__sequence__']=np.arange(n,dtype=np.int64)
        LOCKED.mkdir(parents=True,exist_ok=True); lp=LOCKED/f"{spec['name']}.csv"; lockdf.to_csv(lp,index=False)
        rec.update({'admissible':True,'n':n,'features':int(X.shape[1]),'classes':labels,'train_n':a,'source_n':b-a,'target_n':n-b,'target_windows':12,'canary_budget_labels':B,'min_window_canary':int(min(sched.values())),'locked_csv':str(lp.relative_to(PROJECT)),'locked_sha256':sha(lp),'ordering':'official archive deterministic file order; within-file repository order; deterministic even-index thinning only if n>250000','feature_transforms':trans})
    except Exception as e:
        rec['rejection_reason']=repr(e)
    return rec

def reuse_760():
    lp=LOCKED/'uci760_multivariate_gait.csv'
    must(lp.exists(),'expected already-completed UCI760 locked CSV from first Batch13 attempt')
    d=pd.read_csv(lp); must('__target__' in d,'UCI760 locked CSV missing target')
    yy=d['__target__'].to_numpy(int); n=len(yy); a=int(math.floor(.50*n)); b=int(math.floor(.65*n)); windows=[np.asarray(w,int) for w in np.array_split(np.arange(b,n),12)]
    sched,B=fc.horizon_schedule(windows,fc.BUDGET_RATE,'balanced')
    labs=sorted(np.unique(yy).tolist())
    return {'id':760,'name':'uci760_multivariate_gait','admissible':True,'performance_evaluated':False,'n':n,'features':int(len([c for c in d.columns if c not in {'__target__','__sequence__'}])),'classes':[str(x) for x in labs],'train_n':a,'source_n':b-a,'target_n':n-b,'target_windows':12,'canary_budget_labels':B,'min_window_canary':int(min(sched.values())),'locked_csv':str(lp.relative_to(PROJECT)),'locked_sha256':sha(lp),'ordering':'reused from completed first structural audit; no performance opened','continued_from_first_batch13_attempt':True}

def main():
    OUT.mkdir(parents=True,exist_ok=True); LOCKED.mkdir(parents=True,exist_ok=True)
    dev=DEV/'FINAL_DEVELOPMENT_ASSESSMENT.json'; rob=ROB/'FINAL_ROBUSTNESS_ASSESSMENT.json'; freeze=DEV/'FINAL_METHOD_FREEZE.json'
    for p in [dev,rob,freeze]: must(p.exists(),f'missing prerequisite {p}')
    dj=json.loads(dev.read_text()); rj=json.loads(rob.read_text()); fj=json.loads(freeze.read_text())
    must(dj.get('pass') is True and rj.get('pass') is True,'development/robustness did not pass; refusing fresh audit')
    core_hash=sha(PROJECT/'dtrar_final_core.py'); must(core_hash==fj['core_sha256'],'final core changed after freeze')
    print('FINAL_HOLDOUT_AUDIT_CONTINUE_V2_BEGIN performance_evaluation=false prior_results_reused=uci755,uci760',flush=True)

    records=[]
    records.append({'id':755,'name':'uci755_accel_gyro_phone','admissible':False,'performance_evaluated':False,'rejection_reason':"RuntimeError('target contains class absent from training')",'continued_from_first_batch13_attempt':True})
    print("FINAL_HOLDOUT_AUDIT dataset=uci755_accel_gyro_phone admissible=False reason=RuntimeError('target contains class absent from training') reused=true",flush=True)

    for spec in PRIORITY[1:5]:
        r=audit_direct(spec); records.append(r)
        print(f"FINAL_HOLDOUT_AUDIT dataset={r['name']} admissible={r['admissible']} reason={r.get('rejection_reason','OK')} transport=official_static_zip",flush=True)

    r760=reuse_760(); records.append(r760)
    print('FINAL_HOLDOUT_AUDIT dataset=uci760_multivariate_gait admissible=True reason=OK reused=true',flush=True)

    # Restore exact frozen priority order before selection.
    by={r['name']:r for r in records}; records=[by[s['name']] for s in PRIORITY]
    selected=[r['name'] for r in records if r['admissible']][:3]
    must(len(selected)==3,f'fewer than three structurally admissible holdouts after transport repair: {selected}')
    gates={'all_exact_H_le_0p005':True,'all_requested_canary_budget_exact_4pct':True,'all_three_mean_brier_gain_nonnegative':True,'at_least_two_of_three_material_relative_brier_gain_ge_1pct':True,'at_least_two_of_three_positive_seeds_ge_8of10':True,'pooled_mean_brier_gain_positive':True,'no_source_change_after_lock':True}
    manifest={'selection_locked_before_performance':True,'performance_seen':False,'priority_order':[x['name'] for x in PRIORITY],'selected_fresh_holdouts':selected,'datasets':{r['name']:r for r in records},'final_core_sha256':core_hash,'final_method_freeze':fj,'predeclared_confirmation_gates':gates,'seeds':list(range(10)),'base_model_family':'xgboost','continuation_note':'Only UCI transport was repaired; no performance was evaluated before selection.'}
    slim=[]
    for r in records:
        slim.append({k:v for k,v in r.items() if k not in {'feature_transforms','classes','source_transport'}})
    pd.DataFrame(slim).to_csv(OUT/'FINAL_HOLDOUT_AUDIT.csv',index=False)
    (OUT/'FINAL_HOLDOUT_LOCK.json').write_text(json.dumps(manifest,indent=2,sort_keys=True))
    review=PROJECT/'DTRAR_FinalHoldoutAudit_Batch13_Review.zip'; tmp=review.with_suffix('.zip.partial')
    if tmp.exists(): tmp.unlink()
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(OUT/'FINAL_HOLDOUT_AUDIT.csv','FINAL_HOLDOUT_AUDIT.csv'); z.write(OUT/'FINAL_HOLDOUT_LOCK.json','FINAL_HOLDOUT_LOCK.json'); z.write(Path(__file__),Path(__file__).name)
        for name in selected:
            p=PROJECT/manifest['datasets'][name]['locked_csv']; z.write(p,f'locked/{p.name}')
    with zipfile.ZipFile(tmp) as z: must(z.testzip() is None,'review CRC failure')
    tmp.replace(review)
    print('FINAL_HOLDOUT_SELECTION_LOCKED datasets='+','.join(selected),flush=True)
    print(f'FINAL_HOLDOUT_AUDIT_COMPLETE performance_evaluation=false review={review.name}',flush=True)

if __name__=='__main__': main()
