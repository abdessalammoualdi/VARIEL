#!/usr/bin/env python3
"""Controlled validation of the Temporal Actionability Gap.

The synthetic construction holds current safe repair opportunity approximately
constant while varying only the temporal persistence of the utility of two
opposing probability-repair directions.  It tests the theoretical claim that
stale labels are useful only when repair utility transports through time.

Outputs include exact source CSVs and vector PDF figures.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PERSISTENCE = [0.50, 0.60, 0.75, 0.90, 0.98]
STALENESS = [0, 1, 2, 3]
T = 80
N_AUDIT_POOL = 1000
N_DEPLOY = 1200
AUDIT_FRACTION = 0.05
SHIFT = 0.85
EPS = 0.02
DELTA = 0.05
DECAY = 0.9
BUFFER = 3


def sigmoid(x):
    x=np.clip(x,-30,30)
    return 1/(1+np.exp(-x))


def probs_from_logit(z):
    u=sigmoid(z)
    return np.column_stack([1-u,u])


def brier(p,y):
    return float(np.mean(np.sum((p-np.eye(p.shape[1])[y])**2,axis=1)))


def brier_rows(p,y):
    return np.sum((p-np.eye(p.shape[1])[y])**2,axis=1)


def theta_values(p,q,y):
    d=q-p
    return d[np.arange(len(y)),y]-np.sum(p*d,axis=1)


def C_value(p,q):
    d=q-p
    return float(np.mean(np.sum(d*d,axis=1)))


def serfling_lower(p,q,y,sample_ix,delta=DELTA):
    vals=theta_values(p,q,y)
    sample=np.asarray(sample_ix,int)
    dhat=float(np.mean(vals[sample]))
    d=q-p
    offset=np.sum(p*d,axis=1)
    potential=d-offset[:,None]
    lo=float(np.min(potential)); hi=float(np.max(potential)); span=max(1e-12,hi-lo)
    N=len(y); n=len(sample)
    if n>=N:
        rad=0.0
    else:
        fpc=1-(n-1)/N
        rad=span*math.sqrt(max(0.0,fpc)*math.log(2/delta)/(2*n))
    return dhat,dhat-rad,rad


def risk_opt_eta(C,Dhat,Dlo,eps=EPS):
    if C<=1e-15 or Dhat<=0:
        return 0.0
    plug=float(np.clip(Dhat/C,0,1))
    # audit upper excess C eta^2 - 2 Dlo eta <= eps
    cap=(Dlo+math.sqrt(max(0.0,Dlo*Dlo+C*eps)))/C
    return float(min(plug,np.clip(cap,0,1)))


def plhe_cap(p,q,eta,eps=EPS):
    if eta<=0: return 0.0,0.0
    d=q-p
    A=float(np.mean(np.sum(d*d,axis=1)))
    B=max(0.0,float(np.mean(np.sum(p*d,axis=1)-np.min(d,axis=1))))
    env=lambda e:A*e*e+2*B*e
    if env(eta)<=eps+1e-15:
        return eta,env(eta)
    if A<=1e-15:
        cap=eps/(2*B) if B>1e-15 else eta
    else:
        cap=(math.sqrt(max(0,B*B+A*eps))-B)/A
    e=float(min(eta,max(0,cap)))
    return e,env(e)


def interpolate(p,q,e):
    return p+e*(q-p)


def state_path(rho,T,rng):
    s=np.empty(T,dtype=int); s[0]=1 if rng.random()<0.5 else -1
    for t in range(1,T):
        s[t]=s[t-1] if rng.random()<rho else -s[t-1]
    return s


def simulate_one(rho,stale,rep,seed0=20260913):
    rng=np.random.default_rng(np.random.SeedSequence([seed0,rep,int(round(rho*1000)),stale]))
    state=state_path(rho,T,rng)
    audits=[]
    rows=[]
    for t in range(T):
        # Independent audit and deployment populations under the same current regime.
        za=rng.normal(0,1,N_AUDIT_POOL)
        zd=rng.normal(0,1,N_DEPLOY)
        pa=probs_from_logit(za); pd=probs_from_logit(zd)
        qa_plus=probs_from_logit(za+SHIFT); qa_minus=probs_from_logit(za-SHIFT)
        qd_plus=probs_from_logit(zd+SHIFT); qd_minus=probs_from_logit(zd-SHIFT)
        true_a=qa_plus[:,1] if state[t]>0 else qa_minus[:,1]
        true_d=qd_plus[:,1] if state[t]>0 else qd_minus[:,1]
        ya=(rng.random(N_AUDIT_POOL)<true_a).astype(int)
        yd=(rng.random(N_DEPLOY)<true_d).astype(int)
        n=max(2,int(round(AUDIT_FRACTION*N_AUDIT_POOL)))
        sample=np.sort(rng.choice(N_AUDIT_POOL,size=n,replace=False))
        dhp,dlp,rp=serfling_lower(pa,qa_plus,ya,sample,DELTA/2)
        dhm,dlm,rm=serfling_lower(pa,qa_minus,ya,sample,DELTA/2)
        audits.append({'plus':(dhp,dlp),'minus':(dhm,dlm),'state':state[t]})

        eligible=t-stale
        # Weighted average of up to BUFFER most recent available audit summaries.
        hist=[]
        if eligible>=0:
            for j in range(max(0,eligible-BUFFER+1),eligible+1):
                age=eligible-j
                hist.append((DECAY**age,audits[j]))
        if hist:
            sw=sum(w for w,_ in hist)
            est={}
            for key in ['plus','minus']:
                est[key]=(sum(w*a[key][0] for w,a in hist)/sw,
                          sum(w*a[key][1] for w,a in hist)/sw)
        else:
            est={'plus':(0.,-1.),'minus':(0.,-1.)}

        candidates={'plus':qd_plus,'minus':qd_minus}
        cand_stats=[]
        for key,q in candidates.items():
            C=C_value(pd,q); Dhat,Dlo=est[key]
            e0=risk_opt_eta(C,Dhat,Dlo,EPS)
            e,env=plhe_cap(pd,q,e0,EPS)
            predicted_gain=max(0.0,2*Dhat*e-C*e*e)
            cand_stats.append((predicted_gain,key,e,env))
        cand_stats.sort(reverse=True)
        _,chosen,e,env=cand_stats[0]
        if cand_stats[0][0]<=0:
            chosen='raw'; e=0.; env=0.; out=pd
        else:
            out=interpolate(pd,candidates[chosen],e)

        # Safe hindsight oracle: current labels choose plus/minus and eta, same PLHE budget.
        oracle=[]
        for key,q in candidates.items():
            C=C_value(pd,q); D=float(np.mean(theta_values(pd,q,yd)))
            eo=float(np.clip(D/C if C>1e-15 else 0,0,1)) if D>0 else 0.
            eo,envo=plhe_cap(pd,q,eo,EPS)
            po=interpolate(pd,q,eo)
            oracle.append((brier(po,yd),key,eo,envo,D))
        oracle.append((brier(pd,yd),'raw',0.,0.,0.))
        oracle.sort(key=lambda x:x[0])
        ob,okey,oe,oenv,oD=oracle[0]
        rb=brier(pd,yd); mb=brier(out,yd)
        og=rb-ob; mg=rb-mb
        curDplus=float(np.mean(theta_values(pd,qd_plus,yd)))
        curDminus=float(np.mean(theta_values(pd,qd_minus,yd)))
        if eligible>=0:
            ref=audits[eligible]
            util_var=max(abs(curDplus-ref['plus'][0]),abs(curDminus-ref['minus'][0]))
        else:
            util_var=np.nan
        rows.append({
            'rho':rho,'staleness':stale,'rep':rep,'window':t+1,'state':state[t],
            'raw_brier':rb,'method_brier':mb,'oracle_brier':ob,
            'method_gain':mg,'oracle_gain':og,
            'capture':mg/og if og>1e-12 else np.nan,
            'chosen':chosen,'oracle_chosen':okey,'eta':e,'oracle_eta':oe,
            'envelope':env,'oracle_envelope':oenv,'utility_variation':util_var,
            'audit_n':n,'epsilon':EPS,
        })
    return rows


def figures(summary,window,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    piv=summary.pivot(index='rho',columns='staleness',values='capture_mean').sort_index()
    fig,ax=plt.subplots(figsize=(7.0,4.8),layout='constrained')
    im=ax.imshow(piv.values,aspect='auto',origin='lower',vmin=0,vmax=1,cmap='viridis')
    ax.set_xticks(range(len(piv.columns)),[str(x) for x in piv.columns])
    ax.set_yticks(range(len(piv.index)),[f'{x:.2f}' for x in piv.index])
    ax.set_xlabel('Utility staleness (windows)'); ax.set_ylabel('State persistence $\\rho$')
    ax.set_title('Delayed repair captures less of the safe oracle when utility is nonpersistent')
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            v=piv.values[i,j]
            ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=9,
                    color='white' if v<0.45 else 'black')
    cb=fig.colorbar(im,ax=ax); cb.set_label('Mean fraction of safe-oracle Brier gain captured')
    fig.savefig(out/'temporal_actionability_heatmap.pdf',bbox_inches='tight')
    plt.close(fig)

    g=window.dropna(subset=['utility_variation']).copy()
    # Bin to keep the publication figure legible.
    g['var_bin']=pd.qcut(g['utility_variation'],q=20,duplicates='drop')
    b=g.groupby('var_bin',observed=True).agg(variation=('utility_variation','mean'),
                                            regret=('oracle_brier','mean'),method=('method_brier','mean')).reset_index()
    b['safe_oracle_regret']=b['method']-b['regret']
    fig,ax=plt.subplots(figsize=(6.6,4.5),layout='constrained')
    ax.plot(b['variation'],b['safe_oracle_regret'],marker='o',linewidth=1.8)
    ax.set_xlabel('Delayed-to-current repair-utility variation')
    ax.set_ylabel('Brier gap to current safe oracle')
    ax.set_title('Temporal utility variation explains the delayed-action gap')
    ax.grid(alpha=.2); ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'utility_variation_regret.pdf',bbox_inches='tight')
    plt.close(fig)
    b.to_csv(out/'utility_variation_regret_source.csv',index=False)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--reps',type=int,default=100); ap.add_argument('--output',type=Path,default=Path('artifacts/temporal_actionability_final'))
    a=ap.parse_args(); out=a.output; out.mkdir(parents=True,exist_ok=True)
    plan={'persistence':PERSISTENCE,'staleness':STALENESS,'reps':a.reps,'windows':T,'audit_fraction':AUDIT_FRACTION,
          'shift':SHIFT,'fixed_plhe_budget':EPS,'audit_delta':DELTA,'buffer':BUFFER,'decay':DECAY,
          'claim':'hold safe repair opportunity approximately constant while varying temporal persistence and feedback staleness'}
    (out/'LOCKED_PLAN.json').write_text(json.dumps(plan,indent=2,sort_keys=True))
    print(f'TEMPORAL_ACTIONABILITY_BEGIN conditions={len(PERSISTENCE)*len(STALENESS)} reps={a.reps}',flush=True)
    rows=[]
    for rho in PERSISTENCE:
        for stale in STALENESS:
            for rep in range(a.reps):
                rows.extend(simulate_one(rho,stale,rep))
            print(f'TEMPORAL_ACTIONABILITY_CONDITION_COMPLETE rho={rho:.2f} staleness={stale}',flush=True)
    w=pd.DataFrame(rows); w.to_csv(out/'WINDOW_RESULTS.csv',index=False)
    s=(w.groupby(['rho','staleness','rep'],as_index=False)
       .agg(raw_brier=('raw_brier','mean'),method_brier=('method_brier','mean'),oracle_brier=('oracle_brier','mean'),
            method_gain=('method_gain','mean'),oracle_gain=('oracle_gain','mean'),utility_variation=('utility_variation','mean')))
    s['capture']=s['method_gain']/s['oracle_gain'].where(s['oracle_gain'].abs()>1e-12,np.nan)
    s.to_csv(out/'REPLICATION_RESULTS.csv',index=False)
    summary=(s.groupby(['rho','staleness'],as_index=False)
             .agg(method_gain_mean=('method_gain','mean'),method_gain_sd=('method_gain','std'),
                  oracle_gain_mean=('oracle_gain','mean'),oracle_gain_sd=('oracle_gain','std'),
                  capture_mean=('capture','mean'),capture_sd=('capture','std'),
                  utility_variation_mean=('utility_variation','mean')))
    summary.to_csv(out/'SUMMARY.csv',index=False)

    # Predeclared qualitative gates for the theory: opportunity remains stable;
    # actionability rises strongly with persistence; staleness hurts when persistence is not perfect.
    oracle_by_rho=s.groupby('rho').oracle_gain.mean()
    oracle_cv=float(oracle_by_rho.std(ddof=1)/oracle_by_rho.mean())
    lag2=summary[summary.staleness==2].sort_values('rho')
    rho_corr=float(spearmanr(lag2.rho,lag2.capture_mean).statistic)
    high=summary[summary.rho==0.90].sort_values('staleness')
    stale_corr=float(spearmanr(high.staleness,high.capture_mean).statistic)
    assessment={
        'oracle_gain_cv_across_persistence':oracle_cv,
        'spearman_persistence_vs_capture_at_staleness2':rho_corr,
        'spearman_staleness_vs_capture_at_rho0p90':stale_corr,
        'gates':{
            'oracle_opportunity_stable_cv_le_0p10':bool(oracle_cv<=0.10),
            'capture_increases_with_persistence_rho_ge_0p80':bool(rho_corr>=0.80),
            'capture_decreases_with_staleness_rho_le_minus_0p80':bool(stale_corr<=-0.80),
            'all_plhe_envelopes_within_fixed_budget':bool((w.envelope<=EPS+1e-9).all()),
        }
    }
    assessment['pass']=bool(all(assessment['gates'].values()))
    (out/'ASSESSMENT.json').write_text(json.dumps(assessment,indent=2,sort_keys=True))
    figures(summary,w,out)
    # Review zip
    import zipfile
    review=Path('DTRAR_TemporalActionability_Final_Review.zip')
    with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as z:
        for fn in ['LOCKED_PLAN.json','SUMMARY.csv','REPLICATION_RESULTS.csv','ASSESSMENT.json','temporal_actionability_heatmap.pdf','utility_variation_regret.pdf','utility_variation_regret_source.csv']:
            z.write(out/fn,fn)
    print('TEMPORAL_ACTIONABILITY_ASSESSMENT '+json.dumps(assessment,sort_keys=True),flush=True)
    print(f'TEMPORAL_ACTIONABILITY_COMPLETE pass={str(assessment["pass"]).lower()} review={review.name}',flush=True)

if __name__=='__main__': main()
