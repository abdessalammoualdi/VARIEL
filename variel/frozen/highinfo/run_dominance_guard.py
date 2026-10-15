#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, math, os, sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
PROJECT=Path(os.environ.get('DTRAR_PROJECT',str(Path.home()/'Documents'/'DTRA_Start'/'DTRA'))).resolve()
sys.path.insert(0,str(PROJECT)); sys.path.insert(0,str(HERE))
import frozen_variel_highinfo_development as hi
import frozen_external_reference as ext
import frozen_highinfo_beg_reference as old
old.PROJECT=PROJECT
old.OUT=PROJECT/'artifacts'/'highinfo_retrain_repair'
old.SOURCE=old.OUT/'focus_source'
old.ROLL=old.OUT/'rolling_outputs'

SEEDS=list(range(10))
STAGES=(0.01,0.02,0.04)
ALPHA_TOTAL=0.05
EPS_SAFE=0.02
DELTA_OPERATIONAL=0.001
OUT=PROJECT/'artifacts'/'variel_highinfo_ambiguity_halfstep'
OUT.mkdir(parents=True,exist_ok=True)
LOCKED_HI_SHA='1b2fbbff9e47fb78174109ae48968fc7245e66ec6e226549afb9cd54e0708b62'
DEVELOPMENT=['bank','electricity','gas','airlines']
EXTERNAL=['rialto','spam','insects','ble_rssi']

def must(c,m):
    if not c: raise RuntimeError(m)

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def nested_design(anchor,cands,max_m,rng):
    n=len(anchor); max_m=min(max(1,int(max_m)),max(1,n-1))
    widths=hi.multi_geometry(anchor,cands)
    H=min(8,max_m,n)
    chunks=[np.asarray(x,int) for x in np.array_split(np.argsort(widths,kind='mergesort'),H) if len(x)]
    masses=np.asarray([len(x)*max(float(np.max(widths[x])),1e-12) for x in chunks],float)
    caps=np.asarray([len(x) for x in chunks],int)
    counts=np.zeros(len(chunks),int); count_path=[counts.copy()]
    for _ in range(max_m):
        score=np.where(counts<caps,masses/(counts+1),-1.)
        counts=counts.copy(); counts[int(np.argmax(score))]+=1
        count_path.append(counts.copy())
    perms=[]
    for pop in chunks:
        p=np.asarray(pop,int).copy(); rng.shuffle(p); perms.append(p)
    return chunks,perms,count_path

def strata_at(chunks,perms,count_path,m,n):
    m=min(max(1,int(m)),len(count_path)-1); counts=count_path[m]
    strata=[]; chosen=[]
    for pop,perm,nh in zip(chunks,perms,counts):
        s=np.sort(np.asarray(perm[:int(nh)],int)); chosen.extend(s.tolist())
        strata.append({'population':np.asarray(pop,int),'sample':s})
    can=np.sort(np.asarray(chosen,int)); mask=np.ones(n,bool); mask[can]=False
    return strata,can,np.flatnonzero(mask)

def estimate_with_radius(anchor,q,y,strata,bulk,alpha):
    d=q-anchor; center=np.sum(anchor*d,axis=1); z=d-center[:,None]
    M=max(1,len(bulk)); D=0.; terms=[]
    for st in strata:
        pop=np.asarray(st['population'],int); samp=np.asarray(st['sample'],int)
        if len(samp)==0: continue
        bulk_h=np.setdiff1d(pop,samp,assume_unique=True)
        if len(bulk_h)==0: continue
        D+=(len(bulk_h)/M)*float(np.mean(z[samp,y[samp]]))
        Rh=float(np.max(z[pop])-np.min(z[pop]))
        terms.append(((len(bulk_h)*Rh/M)**2)/len(samp))
    C=float(np.mean(np.sum(d[bulk]**2,axis=1))) if len(bulk) else 0.
    radius=math.sqrt(0.5*math.log(2.0/alpha)*sum(terms)) if terms else 0.
    return D,C,radius,D-radius,D+radius

def choose_guarded_action(anchor,cands,y,seed_tag):
    n=len(anchor); max_m=max(1,int(math.floor(max(STAGES)*n)))
    rng=np.random.default_rng(hi.stable_seed('variel_highinfo_dominance_guard',*seed_tag))
    chunks,perms,count_path=nested_design(anchor,cands,max_m,rng)
    alpha_each=ALPHA_TOTAL/(len(STAGES)*max(1,len(cands)))
    records=[]
    for frac in STAGES:
        m=max(1,int(math.floor(frac*n)))
        strata,can,bulk=strata_at(chunks,perms,count_path,m,n)
        choices=[]
        for name,q in cands.items():
            D,C,r,DL,DU=estimate_with_radius(anchor,q,y,strata,bulk,alpha_each)
            eta_plugin=float(np.clip(D/C,0,1)) if C>1e-18 else 0.
            eta_dom=float(min(eta_plugin,max(0.,2*DL/C))) if (C>1e-18 and DL>0) else 0.
            eta_h,H=hi.clip_H(anchor,q,bulk,can,eta_dom,EPS_SAFE)
            lower_gain=float(2*DL*eta_h-C*eta_h*eta_h) if eta_h>0 else 0.
            est_gain=float(2*D*eta_h-C*eta_h*eta_h) if eta_h>0 else 0.
            choices.append({'name':name,'Dhat':D,'C':C,'radius':r,'DL':DL,'DU':DU,'eta':eta_h,'H':H,'lower_gain':lower_gain,'estimated_gain':est_gain})
        viable=[x for x in choices if x['eta']>0 and x['lower_gain']>0]
        records.append({'fraction':frac,'canary_n':len(can),'choices':choices})
        if viable:
            best=max(viable,key=lambda x:(x['lower_gain'],x['estimated_gain']))
            return {'choice':best['name'],'eta':best['eta'],'H':best['H'],'fraction':frac,'canary_n':len(can),'DL':best['DL'],'Dhat':best['Dhat'],'radius':best['radius'],'lower_gain':best['lower_gain'],'stage_records':records},can,bulk
    # Final-stage ambiguity rule:
    # if the CI still straddles zero, do not throw away a positive point estimate.
    # For quadratic Brier gain G(eta)=2D*eta-C*eta^2, eta_plugin=D/C.
    # The half-step eta=0.5*eta_plugin retains 75% of the plug-in optimum
    # estimated gain while halving action magnitude. Exact H clipping remains.
    frac=STAGES[-1]; m=max(1,int(math.floor(frac*n)))
    strata,can,bulk=strata_at(chunks,perms,count_path,m,n)
    choices=[]
    for name,q in cands.items():
        D,C,r,DL,DU=estimate_with_radius(anchor,q,y,strata,bulk,ALPHA_TOTAL/(len(STAGES)*max(1,len(cands))))
        eta_plugin=float(np.clip(D/C,0,1)) if C>1e-18 else 0.
        eta_half=0.5*eta_plugin
        eta_h,H=hi.clip_H(anchor,q,bulk,can,eta_half,EPS_SAFE)
        est_gain=float(2*D*eta_h-C*eta_h*eta_h) if eta_h>0 else 0.
        choices.append({'name':name,'Dhat':D,'C':C,'radius':r,'DL':DL,'DU':DU,
                        'eta':eta_h,'H':H,'estimated_gain':est_gain})
    viable=[x for x in choices if x['Dhat']>0 and x['eta']>0 and x['estimated_gain']>0]
    if viable:
        best=max(viable,key=lambda x:x['estimated_gain'])
        return {'choice':best['name'],'eta':best['eta'],'H':best['H'],
                'fraction':frac,'canary_n':len(can),'DL':best['DL'],
                'Dhat':best['Dhat'],'radius':best['radius'],
                'lower_gain':0.0,'ambiguity_half_step':True,
                'estimated_gain':best['estimated_gain'],
                'stage_records':records},can,bulk
    return {'choice':'raw_retrain','eta':0.,'H':0.,'fraction':frac,
            'canary_n':len(can),'lower_gain':0.,
            'ambiguity_half_step':False,'stage_records':records},can,bulk

def reconstruct(ds,seed,issued,yy,row_id,windows):
    pos={int(r):i for i,r in enumerate(row_id)}
    winpos=[np.asarray([pos[int(r)] for r in ix],int) for ix in windows]
    pred=issued.copy(); traces=[]
    for t,posix in enumerate(winpos):
        anchor=issued[posix]; local_y=yy[posix]
        hist,w=hi.history_positions(winpos,t)
        if len(hist):
            full=hi.ProbabilityCalibrator().fit(issued[hist],yy[hist],w).predict_proba(anchor)
            tw=w/w.sum(); ess=float(1.0/np.sum(tw**2))
            residual=hi.ResidualCalibrator(anchor.shape[1],ess).fit(issued[hist],yy[hist],w).predict_proba(anchor)
            cands={'fullcal':full,'residual':residual}
        else:
            cands={'fullcal':anchor.copy(),'residual':anchor.copy()}
        dec,can,bulk=choose_guarded_action(anchor,cands,local_y,(ds,seed,t))
        out=anchor.copy()
        if dec['choice']!='raw_retrain' and len(bulk):
            q=cands[dec['choice']]
            out[bulk]=hi.probs(anchor[bulk]+dec['eta']*(q[bulk]-anchor[bulk]))
        out[can]=anchor[can]
        H=hi.prospective_H(anchor,out); must(H<=EPS_SAFE+2e-9,f'{ds}/{seed}/W{t+1}: H={H}')
        pred[posix]=out
        rb=float(np.mean(hi.brier_rows(anchor,local_y))); gb=float(np.mean(hi.brier_rows(out,local_y)))
        traces.append({'dataset':ds,'seed':seed,'window':t+1,'raw_retrain_brier':rb,'guarded_brier':gb,'improvement':rb-gb,'choice':dec['choice'],'eta':dec['eta'],'safety_H':H,'canary_fraction':dec['fraction'],'canary_n':dec['canary_n'],'Dhat':dec.get('Dhat',np.nan),'DL':dec.get('DL',np.nan),'radius':dec.get('radius',np.nan),'lower_gain':dec.get('lower_gain',0.),'ambiguity_half_step':bool(dec.get('ambiguity_half_step',False))})
    raw=float(np.mean(hi.brier_rows(issued,yy))); guarded=float(np.mean(hi.brier_rows(pred,yy)))
    return {'dataset':ds,'seed':seed,'raw_retrain_brier':raw,'guarded_brier':guarded,'improvement_vs_retrain':raw-guarded,'strict_win':bool(guarded<raw),'operational_noninferior':bool(guarded<=raw+DELTA_OPERATIONAL),'max_safety_H':float(max(x['safety_H'] for x in traces)),'mean_current_canary_fraction':float(np.mean([x['canary_fraction'] for x in traces])),'max_current_canary_fraction':float(max(x['canary_fraction'] for x in traces)),'total_label_fraction':1.0},pd.DataFrame(traces)

def development_case(ds,seed):
    if ds in ('bank','electricity','gas'):
        ex=old.locate_existing_output(ds,seed)
        if ex is None:
            own=old.ROLL/f'{ds}_seed{seed}'
            if (own/'issued_predictions.npz').exists() and (own/'evaluation_labels.npz').exists():
                ex=own
        if ex is None:
            print(f'ROLLING_CACHE_MISSING dataset={ds} seed={seed}; rebuilding exact historical rolling reference once',flush=True)
            ex=old.ensure_rolling(ds,seed)
            print(f'ROLLING_CACHE_READY dataset={ds} seed={seed} path={ex}',flush=True)
        with np.load(ex/'issued_predictions.npz') as z: row_id=np.asarray(z['row_id'],int); issued=hi.probs(z['probabilities'])
        with np.load(ex/'evaluation_labels.npz') as z: yy=np.asarray(z['y'],int)
        from dtra.data import load_dataset
        data=load_dataset(ds,PROJECT/'data'/'raw'); windows=[np.asarray(w,int) for w in data.windows]
        must(np.array_equal(row_id,np.asarray(data.target,int)),f'{ds}: row mismatch')
        return issued,yy,row_id,windows
    _,X,y,source,target,windows,labels=hi.load_airlines()
    cache=PROJECT/'artifacts'/'variel_highinfo_cabinet_development'/'airlines_rolling_cache'/f'seed_{seed}'
    must(all((cache/f'window_{i}.npz').exists() for i in range(1,15)),f'airlines/seed{seed}: rolling cache missing')
    issued,yy,row_id=hi.rolling_anchor(X,y,source,windows,seed,1,len(labels),cache)
    return issued,yy,row_id,windows

def external_case(ds,seed):
    _,X,y,source,target,windows,labels,_=ext.load_dataset(ds)
    cache=PROJECT/'artifacts'/'variel_highinfo_external_confirmation'/ds/f'seed_{seed}'/'rolling_anchor_cache'
    must(all((cache/f'window_{i}.npz').exists() for i in range(1,15)),f'{ds}/seed{seed}: external rolling cache missing')
    issued,yy,row_id=hi.rolling_anchor(X,y,source,windows,seed,1,len(labels),cache)
    return issued,yy,row_id,windows

def run_group(group,datasets):
    must(sha256(HERE/'frozen_variel_highinfo_development.py')==LOCKED_HI_SHA,'frozen HIGH-INFO source hash mismatch')
    rows=[]; traces=[]
    with threadpool_limits(limits=1):
        for ds in datasets:
            print(f'GUARD_DATASET_BEGIN group={group} dataset={ds}',flush=True)
            for seed in SEEDS:
                d=OUT/group/ds/f'seed_{seed}'; d.mkdir(parents=True,exist_ok=True); sp=d/'SUMMARY.json'; tp=d/'WINDOW_TRACE.csv'
                if sp.exists() and tp.exists():
                    rec=json.loads(sp.read_text()); tr=pd.read_csv(tp)
                    print(f'GUARD_REUSE dataset={ds} seed={seed} delta={rec["improvement_vs_retrain"]:.12f}',flush=True)
                else:
                    issued,yy,row_id,windows=(development_case(ds,seed) if group=='development' else external_case(ds,seed))
                    rec,tr=reconstruct(ds,seed,issued,yy,row_id,windows); sp.write_text(json.dumps(rec,indent=2,sort_keys=True)); tr.to_csv(tp,index=False)
                    print(f'GUARD_DONE dataset={ds} seed={seed} raw={rec["raw_retrain_brier"]:.9f} guarded={rec["guarded_brier"]:.9f} delta={rec["improvement_vs_retrain"]:.12f} win={rec["strict_win"]} H={rec["max_safety_H"]:.9f} canary_mean={rec["mean_current_canary_fraction"]:.4f}',flush=True)
                rows.append(rec); traces.append(tr)
            g=pd.DataFrame([r for r in rows if r['dataset']==ds])
            print('GUARD_DATASET_COMPLETE '+json.dumps({'dataset':ds,'strict_wins':int(g.strict_win.sum()),'operational_noninferior':int(g.operational_noninferior.sum()),'mean_improvement':float(g.improvement_vs_retrain.mean()),'min_improvement':float(g.improvement_vs_retrain.min()),'max_H':float(g.max_safety_H.max())},sort_keys=True),flush=True)
    df=pd.DataFrame(rows); tt=pd.concat(traces,ignore_index=True)
    df.to_csv(OUT/f'{group.upper()}_SEEDWISE.csv',index=False); tt.to_csv(OUT/f'{group.upper()}_WINDOW_TRACE.csv',index=False)
    sm=df.groupby('dataset',as_index=False).agg(strict_wins=('strict_win','sum'),operational_noninferior=('operational_noninferior','sum'),mean_improvement=('improvement_vs_retrain','mean'),min_improvement=('improvement_vs_retrain','min'),max_H=('max_safety_H','max'),mean_canary_fraction=('mean_current_canary_fraction','mean'),max_canary_fraction=('max_current_canary_fraction','max'))
    sm.to_csv(OUT/f'{group.upper()}_SUMMARY.csv',index=False)
    a={'architecture':'VARIEL HIGH-INFO Dominance Guard + Quadratic Ambiguity Half-Step','group':group,'datasets':datasets,'total_seeds':len(df),'strict_wins':int(df.strict_win.sum()),'all_strict_wins':bool(df.strict_win.all()),'operational_noninferior':int(df.operational_noninferior.sum()),'all_operationally_noninferior':bool(df.operational_noninferior.all()),'H02_holds':bool((df.max_safety_H<=EPS_SAFE+2e-9).all()),'total_label_fraction':1.0,'max_current_canary_fraction':float(df.max_current_canary_fraction.max()),'parameters':{'stages':list(STAGES),'alpha_total':ALPHA_TOTAL,'epsilon_H':EPS_SAFE,'operational_delta':DELTA_OPERATIONAL},'frozen_sparse_VARIEL_changed':False}
    (OUT/f'{group.upper()}_ASSESSMENT.json').write_text(json.dumps(a,indent=2,sort_keys=True))
    print(f'\n=== {group.upper()} DOMINANCE-GUARD ASSESSMENT ===\n'+json.dumps(a,indent=2,sort_keys=True),flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--group',choices=['development','external'],required=True); ap.add_argument('--datasets',nargs='*'); args=ap.parse_args()
    allowed=DEVELOPMENT if args.group=='development' else EXTERNAL; ds=args.datasets or allowed
    must(all(x in allowed for x in ds),'dataset outside group'); run_group(args.group,ds)
if __name__=='__main__': main()
