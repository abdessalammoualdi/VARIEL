#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib,json,math,zipfile,joblib
import numpy as np,pandas as pd
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits
import dtrar_final_core as fc

PROJECT=Path(__file__).resolve().parent
AUD=PROJECT/'artifacts'/'dtrar_liability_fresh_audit_batch16'; LOCK=AUD/'LIABILITY_FRESH_LOCK.json'; OUT=PROJECT/'artifacts'/'dtrar_liability_fresh_confirmation_batch17'; SEEDS=list(range(10))

def must(c,m):
    if not c:raise RuntimeError(m)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
      for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def native(x):
    if isinstance(x,dict):return {str(k):native(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [native(v) for v in x]
    if isinstance(x,np.bool_):return bool(x)
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.floating):return float(x)
    return x
@dataclass
class Dataset:
    name:str;X:pd.DataFrame;y:np.ndarray;classes:list;categorical:list;train:np.ndarray;source:np.ndarray;windows:list;metadata:dict
    @property
    def target(self):return np.concatenate(self.windows)

def load_locked(name,m):
    r=m['datasets'][name];p=PROJECT/r['locked_csv'];must(p.exists(),f'missing {p}');must(sha(p)==r['locked_sha256'],f'hash changed {name}')
    df=pd.read_csv(p);y=df.pop('__target__').to_numpy(int);df=df.drop(columns=['__sequence__'],errors='ignore');X=df.apply(pd.to_numeric,errors='coerce');n=len(y);a=int(math.floor(.50*n));b=int(math.floor(.65*n))
    return Dataset(name,X,y,[str(x) for x in r['classes']],[],np.arange(a),np.arange(a,b),[np.asarray(w,int) for w in np.array_split(np.arange(b,n),12)],{'liability_fresh':True})
def cfg():
    p=PROJECT/'offset_check_v07'/'configs'/'primary.json';must(p.exists(),f'missing {p}');return json.loads(p.read_text())
def train_xgb(data,seed,locked_sha):
    from xgboost import XGBClassifier
    c=cfg();ident={'dataset':data.name,'locked_sha256':locked_sha,'seed':seed,'protocol':'liability-fresh-v1','family':'xgboost','tree_cfg':{k:c[k] for k in c if k.startswith('tree_')}};sig=hashlib.sha256(json.dumps(ident,sort_keys=True).encode()).hexdigest()[:12];cache=OUT/'models'/data.name/f'seed_{seed}_{sig}';cache.mkdir(parents=True,exist_ok=True);pf=cache/'frozen_probabilities.npz';cf=cache/'COMPLETE.json'
    if pf.exists() and cf.exists():
      z=np.load(pf);raw=np.asarray(z['probabilities'],float);must(np.array_equal(z['row_id'],np.arange(len(data.y))),'row mismatch');return raw
    boundary=int(math.floor(.8*len(data.train)));inner,valid=data.train[:boundary],data.train[boundary:];must(len(np.unique(data.y[inner]))==len(data.classes),f'{data.name}: training lacks class')
    imp=SimpleImputer(strategy='median').fit(data.X.iloc[inner]);Xin=imp.transform(data.X.iloc[inner]).astype(np.float32);Xva=imp.transform(data.X.iloc[valid]).astype(np.float32);k=len(data.classes);args=dict(n_estimators=c['tree_rounds'],max_depth=c['tree_depth'],learning_rate=c['tree_learning_rate'],subsample=.9,colsample_bytree=.9,min_child_weight=2,reg_lambda=1.,tree_method='hist',device='cpu',n_jobs=1,random_state=seed,objective='binary:logistic' if k==2 else 'multi:softprob',eval_metric='logloss' if k==2 else 'mlogloss')
    sel=XGBClassifier(**args,early_stopping_rounds=c['tree_early_stopping']);sel.fit(Xin,data.y[inner],eval_set=[(Xva,data.y[valid])],verbose=False);rounds=max(1,int(sel.best_iteration)+1);fullimp=SimpleImputer(strategy='median').fit(data.X.iloc[data.train]);Xall=fullimp.transform(data.X).astype(np.float32);args['n_estimators']=rounds;model=XGBClassifier(**args);model.fit(Xall[data.train],data.y[data.train],verbose=False);raw=np.asarray(model.predict_proba(Xall),float);must(raw.shape==(len(data.y),k),'prediction shape mismatch');np.savez_compressed(pf,probabilities=raw,row_id=np.arange(len(data.y)));joblib.dump(fullimp,cache/'preprocessor.joblib');model.save_model(cache/'frozen.ubj');cf.write_text(json.dumps({**ident,'rounds':rounds},indent=2,sort_keys=True));return raw

def holm(p):
    p=np.asarray(p,float);o=np.argsort(p);out=np.empty_like(p);prev=0.;m=len(p)
    for r,i in enumerate(o):
      v=max(prev,min(1.,(m-r)*p[i]));out[i]=v;prev=v
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True);must(LOCK.exists(),'missing Batch16 lock');m=json.loads(LOCK.read_text());must(m['performance_seen'] is False and m['selection_locked_before_performance'] is True,'invalid fresh lock');eps=float(m['liability_policy']['primary_epsilon']);core_hash=sha(PROJECT/'dtrar_final_core.py');must(core_hash==m['final_core_sha256'],'core changed after lock');dslist=m['selected_fresh_holdouts'];must(len(dslist)==3,'need 3 locked streams')
    print(f'LIABILITY_FRESH_CONFIRMATION_BATCH17_BEGIN epsilon={eps} datasets=3 seeds=10 source_frozen=true',flush=True)
    seeds=[];wins=[]
    with threadpool_limits(limits=1):
      for ds in dslist:
        data=load_locked(ds,m)
        for seed in SEEDS:
          print(f'LIABILITY_FRESH_JOB_BEGIN dataset={ds} seed={seed}',flush=True);raw=train_xgb(data,seed,m['datasets'][ds]['locked_sha256']);s,w=fc.run_stream(data,raw,seed,'xgboost_liability_fresh',ds,budget_rate=fc.BUDGET_RATE,schedule_mode='balanced',sample_mode='geometry',action_mode='riskopt',exact_h=True,eps=eps);seeds.append(s);wins.append(w);print('LIABILITY_FRESH_JOB_RESULT '+json.dumps(native({'dataset':ds,'seed':seed,'gain':s['brier_gain'],'raw_brier':s['raw_brier'],'max_H':s['max_exact_H']}),sort_keys=True),flush=True)
    sd=pd.DataFrame(seeds);wd=pd.concat(wins,ignore_index=True);rows=[]
    for ds,g in sd.groupby('dataset'):
      x=g.brier_gain.to_numpy(float);rb=float(g.raw_brier.mean())
      try:wp=float(wilcoxon(x,alternative='greater').pvalue)
      except Exception:wp=float('nan')
      rows.append({'dataset':ds,'epsilon':eps,'mean_gain':float(x.mean()),'sd_gain':float(x.std(ddof=1)),'min_seed_gain':float(x.min()),'positive_seeds':int((x>0).sum()),'seeds':len(x),'mean_raw_brier':rb,'relative_brier_reduction':float(x.mean()/rb) if rb>0 else np.nan,'wilcoxon_p_one_sided':wp,'max_exact_H':float(g.max_exact_H.max()),'all_budget_exact':bool(g.budget_exact.all()),'mean_nll_gain':float(g.nll_gain.mean()),'mean_accuracy_gain':float(g.accuracy_gain.mean()),'mean_macro_f1_gain':float(g.macro_f1_gain.mean()),'mean_ece_reduction':float(g.ece_reduction.mean())})
    sm=pd.DataFrame(rows);sm['holm_p_three']=holm(sm.wilcoxon_p_one_sided.to_numpy(float));exact=bool(sd.max_exact_H.max()<=eps+5e-9);budget=bool(sd.budget_exact.all());allnon=bool((sm.mean_gain>=0).all());material=int((sm.relative_brier_reduction>=0.01).sum())>=2;pos=int((sm.positive_seeds>=8).sum())>=2;pooled=float(sd.brier_gain.mean())>0
    gates={'all_exact_H_le_primary_epsilon':exact,'all_requested_canary_budget_exact_4pct':budget,'all_three_mean_brier_gain_nonnegative':allnon,'at_least_two_of_three_relative_brier_gain_ge_1pct':bool(material),'at_least_two_of_three_positive_seeds_ge_8of10':bool(pos),'pooled_mean_brier_gain_positive':bool(pooled),'no_source_or_epsilon_change_after_lock':True};must(set(gates)==set(m['predeclared_confirmation_gates']),'gate names changed');passed=bool(all(gates.values()))
    sd.to_csv(OUT/'LIABILITY_FRESH_SEEDWISE.csv',index=False);wd.to_csv(OUT/'LIABILITY_FRESH_WINDOW_TRACE.csv',index=False);sm.to_csv(OUT/'LIABILITY_FRESH_SUMMARY.csv',index=False);(OUT/'LIABILITY_FRESH_ASSESSMENT.json').write_text(json.dumps(native({'pass':passed,'gates':gates,'epsilon':eps,'datasets':dslist,'core_sha256':core_hash,'post_lock_change':False}),indent=2,sort_keys=True))
    review=PROJECT/'DTRAR_LiabilityFreshConfirmation_Batch17_Review.zip';tmp=review.with_suffix('.zip.partial');
    if tmp.exists():tmp.unlink()
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
      for p in [OUT/'LIABILITY_FRESH_SEEDWISE.csv',OUT/'LIABILITY_FRESH_WINDOW_TRACE.csv',OUT/'LIABILITY_FRESH_SUMMARY.csv',OUT/'LIABILITY_FRESH_ASSESSMENT.json',LOCK,Path(__file__)]:z.write(p,p.name)
    with zipfile.ZipFile(tmp) as z:must(z.testzip() is None,'CRC failure')
    tmp.replace(review);print('LIABILITY_FRESH_CONFIRMATION_ASSESSMENT '+json.dumps(gates,sort_keys=True),flush=True);print(f'LIABILITY_FRESH_CONFIRMATION_BATCH17_COMPLETE pass={str(passed).lower()} review={review.name}',flush=True);print(sm.to_string(index=False),flush=True)
if __name__=='__main__':main()
