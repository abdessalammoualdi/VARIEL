#!/usr/bin/env python3
"""Batch 15: consumed-data safety--utility frontier for frozen HB-GO-Canary-RAPID.

No algorithmic component is changed.  The exact prospective liability epsilon is
recognized as an explicit application contract.  We evaluate a predeclared grid
and choose the SMALLEST epsilon satisfying a materiality rule on consumed data.
The three Batch-14 streams are no longer called fresh after this analysis.
"""
from pathlib import Path
import hashlib,json,zipfile
import numpy as np, pandas as pd
from threadpoolctl import threadpool_limits

import dtrar_final_core as fc
import run_dtrar_final_development_batch11 as b11
import run_dtrar_final_fresh_confirmation_batch14 as b14

PROJECT=Path(__file__).resolve().parent
OUT=PROJECT/'artifacts'/'dtrar_liability_frontier_batch15'
DEV=PROJECT/'artifacts'/'dtrar_final_development_batch11'
ROB=PROJECT/'artifacts'/'dtrar_final_robustness_batch12'
LOCK=PROJECT/'artifacts'/'dtrar_final_holdout_audit'/'FINAL_HOLDOUT_LOCK.json'
EPS_GRID=[0.005,0.010,0.020,0.030,0.050]
SEEDS=list(range(10))
PRIMARY=b11.PRIMARY
DEV_EXTRAS=b11.EXTRAS

def must(c,m):
    if not c: raise RuntimeError(m)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''): h.update(x)
    return h.hexdigest()
def native(x):
    if isinstance(x,dict): return {str(k):native(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [native(v) for v in x]
    if isinstance(x,np.bool_): return bool(x)
    if isinstance(x,np.integer): return int(x)
    if isinstance(x,np.floating): return float(x)
    return x

def summarise(df):
    rows=[]
    for (eps,fam,ds,role),g in df.groupby(['epsilon','family','dataset','role']):
        x=g.brier_gain.to_numpy(float); rb=float(g.raw_brier.mean())
        rows.append({'epsilon':float(eps),'epsilon_fraction_of_brier_range_2':float(eps/2.0),
          'family':fam,'dataset':ds,'role':role,'mean_gain':float(x.mean()),'sd_gain':float(x.std(ddof=1)),
          'min_seed_gain':float(x.min()),'positive_seeds':int((x>0).sum()),'seeds':int(len(x)),
          'mean_raw_brier':rb,'relative_brier_reduction':float(x.mean()/rb) if rb>0 else np.nan,
          'max_exact_H':float(g.max_exact_H.max()),'all_budget_exact':bool(g.budget_exact.all()),
          'mean_nll_gain':float(g.nll_gain.mean()),'mean_ece_reduction':float(g.ece_reduction.mean()),
          'mean_act_rate':float(g.mean_act_rate.mean()),'mean_eta_when_acted':float(g.mean_eta_when_acted.mean())})
    return pd.DataFrame(rows)

def eligible(summary,eps,former_fresh):
    s=summary[np.isclose(summary.epsilon,eps)]
    prim=s[s.apply(lambda r:(r.family,r.dataset) in PRIMARY,axis=1)]
    ff=s[s.dataset.isin(former_fresh)]
    reasons={
      'primary_10_cells_present':len(prim)==10,
      'primary_all_mean_positive':len(prim)==10 and bool((prim.mean_gain>0).all()),
      'primary_all_at_least_8of10_positive':len(prim)==10 and bool((prim.positive_seeds>=8).all()),
      'former_fresh_3_present':len(ff)==3,
      'former_fresh_all_mean_nonnegative':len(ff)==3 and bool((ff.mean_gain>=0).all()),
      'former_fresh_two_of_three_relative_gain_ge_1pct':len(ff)==3 and int((ff.relative_brier_reduction>=0.01).sum())>=2,
      'former_fresh_two_of_three_positive_seeds_ge_8of10':len(ff)==3 and int((ff.positive_seeds>=8).sum())>=2,
      'exact_contract_for_epsilon':bool(len(s)>0 and (s.max_exact_H<=eps+5e-9).all()),
      'budget_exact':bool(len(s)>0 and s.all_budget_exact.all()),
    }
    return reasons,bool(all(reasons.values()))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    freeze=json.loads((DEV/'FINAL_METHOD_FREEZE.json').read_text())
    core_hash=sha(PROJECT/'dtrar_final_core.py'); must(core_hash==freeze['core_sha256'],'frozen core changed')
    must(json.loads((DEV/'FINAL_DEVELOPMENT_ASSESSMENT.json').read_text())['pass'] is True,'Batch11 did not pass')
    must(json.loads((ROB/'FINAL_ROBUSTNESS_ASSESSMENT.json').read_text())['pass'] is True,'Batch12 did not pass')
    manifest=json.loads(LOCK.read_text()); former_fresh=list(manifest['selected_fresh_holdouts']); must(len(former_fresh)==3,'expected three consumed Batch14 streams')
    print('LIABILITY_FRONTIER_BATCH15_BEGIN eps_grid=0.005,0.010,0.020,0.030,0.050 brier_range=2 consumed_data_only=true',flush=True)
    rows=[]
    with threadpool_limits(limits=1):
      # Manuscript development cells.
      for eps in EPS_GRID:
        for fam,ds in PRIMARY+DEV_EXTRAS:
          for seed in SEEDS:
            data,raw=b11.load_job(fam,ds,seed)
            s,_=fc.run_stream(data,raw,seed,fam,ds,budget_rate=fc.BUDGET_RATE,schedule_mode='balanced',sample_mode='geometry',action_mode='riskopt',exact_h=True,eps=eps)
            s.update({'epsilon':eps,'role':'development'})
            rows.append(s)
        # Former fresh Batch14 streams are explicitly consumed from now on.
        for ds in former_fresh:
          data=b14.load_locked(ds,manifest)
          for seed in SEEDS:
            raw=b14.train_xgb(data,seed,manifest['datasets'][ds]['locked_sha256'])
            s,_=fc.run_stream(data,raw,seed,'xgboost_consumed_batch14',ds,budget_rate=fc.BUDGET_RATE,schedule_mode='balanced',sample_mode='geometry',action_mode='riskopt',exact_h=True,eps=eps)
            s.update({'epsilon':eps,'role':'consumed_batch14'})
            rows.append(s)
        print(f'LIABILITY_FRONTIER_EPS_COMPLETE epsilon={eps:.3f}',flush=True)
    seedwise=pd.DataFrame(rows); summary=summarise(seedwise)
    selected=None; gate_table={}
    for eps in EPS_GRID:
      reasons,ok=eligible(summary,eps,former_fresh); gate_table[f'{eps:.3f}']=reasons
      if ok and selected is None: selected=eps
    passed=selected is not None
    seedwise.to_csv(OUT/'LIABILITY_FRONTIER_SEEDWISE.csv',index=False); summary.to_csv(OUT/'LIABILITY_FRONTIER_SUMMARY.csv',index=False)
    assessment=native({'pass':passed,'selected_epsilon':selected,'selection_rule':'smallest predeclared epsilon satisfying all materiality/safety gates on consumed data','epsilon_grid':EPS_GRID,'brier_score_range':[0,2],'selected_epsilon_fraction_of_full_brier_range':None if selected is None else selected/2.0,'former_fresh_are_now_consumed':former_fresh,'core_sha256':core_hash,'gates_by_epsilon':gate_table})
    (OUT/'LIABILITY_FRONTIER_ASSESSMENT.json').write_text(json.dumps(assessment,indent=2,sort_keys=True))
    if passed:
      (OUT/'LIABILITY_POLICY_FREEZE.json').write_text(json.dumps(native({'frozen':True,'core_sha256':core_hash,'primary_epsilon':selected,'strict_reference_epsilon':0.005,'budget_rate':fc.BUDGET_RATE,'audit_alpha':fc.AUDIT_ALPHA,'selection_rule':assessment['selection_rule'],'fresh_confirmation_after_this_freeze':False}),indent=2,sort_keys=True))
    review=PROJECT/'DTRAR_LiabilityFrontier_Batch15_Review.zip'; tmp=review.with_suffix('.zip.partial')
    if tmp.exists(): tmp.unlink()
    files=[OUT/'LIABILITY_FRONTIER_SEEDWISE.csv',OUT/'LIABILITY_FRONTIER_SUMMARY.csv',OUT/'LIABILITY_FRONTIER_ASSESSMENT.json',Path(__file__)]
    if (OUT/'LIABILITY_POLICY_FREEZE.json').exists(): files.append(OUT/'LIABILITY_POLICY_FREEZE.json')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
      for p in files:z.write(p,p.name)
    with zipfile.ZipFile(tmp) as z: must(z.testzip() is None,'CRC failure')
    tmp.replace(review)
    print('LIABILITY_FRONTIER_BATCH15_ASSESSMENT '+json.dumps(native({'pass':passed,'selected_epsilon':selected}),sort_keys=True),flush=True)
    print(f'LIABILITY_FRONTIER_BATCH15_COMPLETE pass={str(passed).lower()} review={review.name}',flush=True)
    # Print only the former-fresh materiality rows + selected primary cells.
    print(summary[summary.role.eq('consumed_batch14')].to_string(index=False),flush=True)

if __name__=='__main__':main()
