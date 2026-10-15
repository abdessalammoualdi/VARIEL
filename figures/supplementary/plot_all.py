#!/usr/bin/env python3
"""Eight source-grounded supplementary figures, Matplotlib only; no rendered-image reuse.
Run: python plot_all.py
"""
from pathlib import Path
import hashlib, json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parent; D=ROOT/'data'; OUT=ROOT/'figures'; PRE=ROOT/'previews'; AUD=ROOT/'audit'
for p in (OUT, PRE, AUD):p.mkdir(exist_ok=True)
MAN=json.loads((D/'SOURCE_MANIFEST.json').read_text())
for f,m in MAN.items():
 assert hashlib.sha256((D/f).read_bytes()).hexdigest()==m['sha256'], f'Changed evidence: {f}'
C={'blue':'#0048B8','green':'#00733F','orange':'#EA6500','red':'#B31226','cyan':'#0099BC','grey':'#8292A3','navy':'#153650','black':'#15191E','gold':'#A97700'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.titleweight':'bold','axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white','figure.facecolor':'white'})
def read(f):return pd.read_csv(D/f)
def name(x):
 return {'eeg_eye_state':'EEG','occupancy_detection_v2':'Occupancy','ble_rssi':'BLE RSSI','insects':'INSECTS','uci319_mhealth':'MHEALTH','uci427_older_activity':'Older Activity','uci760_multivariate_gait':'Multivariate Gait','fttransformer':'FT-Transformer','ft_transformer':'FT-Transformer','catboost':'CatBoost','xgboost':'XGBoost'}.get(str(x),str(x).replace('_',' ').title())
def clean(ax,grid='y'):
 ax.set_axisbelow(True)
 if grid:ax.grid(axis=grid,color='#DDE4EC',lw=.7)
 ax.tick_params(labelsize=9,color='#8090A0');ax.spines['left'].set_color('#B0BDC9');ax.spines['bottom'].set_color('#B0BDC9')
def figure():
 fig,aa=plt.subplots(2,2,figsize=(17.5,11.5),layout='constrained')
 fig.get_layout_engine().set(w_pad=.16,h_pad=.18,wspace=.17,hspace=.22)
 return fig,aa.ravel()
def title(ax,txt,letter):ax.set_title(f'({letter})  {txt}',loc='left',pad=16,color=C['navy'],fontsize=12.0,fontweight='bold')
def save(fig,key,desc,checks):
 pdf=OUT/(key+'.pdf');png=PRE/(key+'.png')
 fig.savefig(pdf,dpi=240,bbox_inches='tight',pad_inches=.19)
 fig.savefig(png,dpi=155,bbox_inches='tight',pad_inches=.19)
 plt.close(fig)
 import fitz
 doc=fitz.open(pdf)
 assert doc.page_count==1
 w,h=doc[0].rect.width,doc[0].rect.height
 assert w>300 and h>200
 (AUD/(key+'.json')).write_text(json.dumps({'description':desc,'data_sha256_checked':True,'pdf_page_count':1,'size_pt':[round(w,2),round(h,2)],'checks':checks},indent=2,default=lambda x:float(x) if isinstance(x,np.floating) else int(x) if isinstance(x,np.integer) else str(x)))
 print(key,desc,'PDF',round(pdf.stat().st_size/1024),'KiB')

def f1():
 dev=read('dev_windows.csv'); temp=read('temporal_summary.csv');fr=read('frontier_summary.csv')
 fig,axs=figure();a,b,c,d=axs
 # a: global horizon canary budget use final xgboost 3 datasets, seed0 cumulative actual allocation
 subset=dev[(dev.variant=='final')&(dev.family=='xgboost')&(dev.seed==0)&(dev.dataset.isin(['bank','electricity','gas']))]
 for ds,col,marker,style in [('electricity','green','s','-'),('gas','orange','o','-'),('bank','blue','D','--')]:
  g=subset[subset.dataset==ds].sort_values('window'); x=g.window.to_numpy(); y=g.canary_observed.cumsum()/g.canary_observed.sum()*100
  a.plot(x,y,color=C[col],marker=marker,lw=1.9,markersize=5,label=name(ds),linestyle=style,markerfacecolor='white' if ds=='bank' else C[col],zorder=5 if ds=='bank' else 3)
 title(a,'Horizon allocation of the 4% canary budget','a');a.set(xlabel='Maintenance window',ylabel='Cumulative share of allocated labels (%)',ylim=(0,104)); a.legend(frameon=False,loc='lower right',ncol=2);clean(a)
 # b: actual estimated gain geometry - freeze identity from one window, no claiming observed gains
 g=subset[(subset.dataset=='bank') & (subset.eta_deploy>0) & (subset.C_bulk>0)].head(1)
 if len(g)==0:g=subset[subset.C_bulk>0].head(1)
 r=g.iloc[0];etas=np.linspace(0,1,180);D=float(r.D_hat_bulk);Cb=float(r.C_bulk);G=2*D*etas-Cb*etas**2
 b.plot(etas,G,color=C['blue'],lw=2.6)
 b.axhline(0,color='#596A78',lw=.8);b.axvline(float(r.eta_deploy),ls='--',lw=1.6,color=C['orange'],label='Deployed action')
 b.scatter([float(r.eta_deploy)],[np.interp(float(r.eta_deploy),etas,G)],s=70,color=C['orange'],zorder=3)
 title(b,'Estimated quadratic gain on a frozen action path','b');b.set(xlabel='Action fraction η',ylabel='Estimated Brier gain  2D̂η − Cη²');b.legend(frameon=False,loc='best',fontsize=9);clean(b)
 b.text(.98,.06,f'{name(r.dataset)}, seed {int(r.seed)}, window {int(r.window)}',ha='right',transform=b.transAxes,fontsize=9,color=C['navy'])
 # c: exact liability boundary across cap grid (aggregate evidence)
 z=fr[(fr.family=='xgboost')&(fr.dataset.isin(['bank','electricity','gas']))].copy()
 for ds,col in zip(['bank','electricity','gas'],['blue','green','orange']):
  g=z[z.dataset==ds].sort_values('epsilon');c.plot(g.epsilon,g.max_exact_H,marker='o',lw=2,color=C[col],label=name(ds))
 mx=max(float(z.epsilon.max()),float(z.max_exact_H.max()))
 c.plot([0,mx],[0,mx],ls='--',lw=1.2,color=C['red'],label='H = cap')
 title(c,'Observed maximum liability against declared cap','c');c.set(xlabel='Prospective liability cap ε',ylabel='Maximum exact H across evaluated seeds');c.legend(frameon=False,ncol=2,fontsize=9);clean(c)
 # d temporal actionable capture heatmap
 pv=temp.pivot(index='rho',columns='staleness',values='capture_mean').sort_index(ascending=False)
 im=d.imshow(pv.to_numpy(),cmap='Blues',aspect='auto',vmin=min(0,float(np.nanmin(pv.to_numpy()))),vmax=1)
 for i in range(len(pv.index)):
  for j in range(len(pv.columns)):
   val=pv.iloc[i,j]; d.text(j,i,f'{val:.2f}',ha='center',va='center',fontsize=9,color='white' if val>.63 else C['black'])
 d.set_xticks(range(len(pv.columns)),[str(v) for v in pv.columns]);d.set_yticks(range(len(pv.index)),[f'{v:.2f}' for v in pv.index]);d.set(xlabel='Label staleness',ylabel='Persistence ρ')
 title(d,'Actionable capture across persistence and staleness','d');fig.colorbar(im,ax=d,shrink=.77,label='Capture fraction',pad=.02)
 save(fig,'S01_action_evidence_geometry','Horizon budget, estimated action geometry, exact liability, and actionable capture',{'trace_rows':len(subset),'temporal_grid':list(pv.shape),'frontier_rows':len(z)})

def f2():
 stages=read('racs_stages.csv');per=read('racs_dataset.csv');res=read('racs_rescue.csv');fig,ax=figure();a,b,c,d=ax
 s=stages[stages.stage!='Exact/equivalent fallback'].copy();ys=np.arange(len(s));sname=['ASPIRE-CB','PPARC','SHARP','CPPI-GREG','RACS']
 a.barh(ys,s.operational/s.total*100,color=[C['grey'],C['blue'],C['cyan'],C['orange'],C['green']],height=.6)
 for j,row in enumerate(s.itertuples()):a.text(min(99,float(row.operational)/float(row.total)*100+2),j,f'{int(row.operational)}/{int(row.total)}',va='center',fontweight='bold',fontsize=9)
 a.set(yticks=ys,yticklabels=sname,xlabel='Operationally valid decisions (%)',xlim=(0,113));a.invert_yaxis();title(a,'Resource resolution across certification stages','a');clean(a,'x')
 b.bar(ys,s.mean_label*100,color=[C['grey'],C['blue'],C['cyan'],C['orange'],C['green']],width=.62)
 b.set(xticks=ys,xticklabels=sname,ylabel='Mean acquired labels (%)');b.tick_params(axis='x',rotation=20)
 title(b,'Labels used by each resource design','b');clean(b)
 per=per.sort_values('mean_label_fraction');pos=np.arange(len(per));c.barh(pos,per.mean_label_fraction*100,color=C['blue'],height=.56)
 for j,row in enumerate(per.itertuples()):c.text(row.mean_label_fraction*100+.14,j,f'{int(row.exact_winner_matches)}/180 exact',va='center',fontsize=9,color=C['navy'])
 c.set(yticks=pos,yticklabels=[name(x) for x in per.dataset],xlabel='Mean label fraction (%)',xlim=(0, max(per.mean_label_fraction)*100+2.5));c.invert_yaxis()
 title(c,'Terminal Resource Cabinet by dataset','c');clean(c,'x')
 r=res.sort_values('true_margin');xx=np.arange(len(r));old=np.abs(r.true_operational_regret);new=np.abs(r.racs_true_regret)
 d.plot(xx,old,color=C['orange'],marker='o',lw=1.5,markersize=4,label='Prior unresolved route')
 d.plot(xx,new,color=C['green'],marker='D',lw=1.5,markersize=4,label='RACS final')
 d.axhline(0,color='#334452',lw=.6);d.set(xlabel='Previously unresolved case (sorted)',ylabel='True operational regret',xlim=(-.8,len(r)-.2))
 title(d,'Final residual consensus on 21 hard cases','d');d.legend(frameon=False,fontsize=9);clean(d)
 save(fig,'S02_resource_cabinet','Resource-stage operational validity, label expense and hard-case regret',{'RACS_operational':int(per.operational_valid.sum()),'RACS_exact':int(per.exact_winner_matches.sum()),'rescue_count':len(res),'racs_resolved':int(res.racs_operational_valid.sum())})

def f3():
 dev=read('dev_summary.csv');rob=read('robust_summary.csv');fresh=read('fresh_summary.csv');fig,ax=figure();a,b,c,d=ax
 z=dev[(dev.variant=='final')&(dev.family=='xgboost')&(dev.dataset.isin(['bank','gas','electricity']))].sort_values('mean_gain')
 x=np.arange(len(z));a.barh(x,z.mean_gain,color=[C['blue'],C['green'],C['orange']],height=.53);a.errorbar(z.mean_gain,x,xerr=z.sd_gain,fmt='none',ecolor=C['black'],capsize=4)
 a.set(yticks=x,yticklabels=[name(k) for k in z.dataset],xlabel='Mean Brier improvement ± seed SD');title(a,'Core sparse-maintenance outcomes','a');clean(a,'x')
 z=rob[(rob.family=='xgboost')&(rob.dataset.isin(['bank','electricity','gas']))]
 for ds,col in zip(['bank','electricity','gas'],['blue','green','orange']):
  g=z[(z.dataset==ds)&(z.case.str.startswith('budget_'))].copy();g['budget']=g.case.str.replace('budget_','',regex=False).astype(float)*100;g=g.sort_values('budget')
  b.plot(g.budget,g.mean_gain,marker='o',lw=2.2,color=C[col],label=name(ds))
 b.set(xlabel='Horizon label budget (%)',ylabel='Mean Brier improvement');title(b,'Feedback-budget frontier','b');b.legend(frameon=False);clean(b)
 styles={'noise_0.05':('red','5% label noise'),'noise_0.10':('orange','10% label noise'),'dropout_0.10':('cyan','10% dropout'),'dropout_0.25':('blue','25% dropout')}
 for key,(col,lab) in styles.items():
  g=z[z.case==key].sort_values('dataset'); c.scatter([name(v) for v in g.dataset],g.mean_gain,color=C[col],s=68,label=lab,zorder=2)
 c.set(ylabel='Mean Brier improvement');title(c,'Noise and canary-dropout stress','c');c.legend(frameon=False,ncol=2,fontsize=8.6,loc='upper center');clean(c)
 fres=fresh.copy();fres['pretty']=fres.dataset.map(name);ys=np.arange(len(fres));colors=[C['green'] if v>0 else C['grey'] for v in fres.mean_gain]
 d.barh(ys,fres.mean_gain,color=colors,height=.54)
 for i,rr in enumerate(fres.itertuples()):d.text(float(rr.mean_gain)+.0006,i,'HOLD (10/10)' if rr.mean_gain==0 else f'{int(rr.positive_seeds)}/{int(rr.seeds)} wins',va='center',fontsize=9,fontweight='bold')
 d.set(yticks=ys,yticklabels=fres.pretty,xlabel='Mean Brier improvement',xlim=(0,fres.mean_gain.max()*1.36));d.invert_yaxis();title(d,'Locked fresh confirmation and no-op control','d');clean(d,'x')
 save(fig,'S03_sparse_robustness_fresh','Sparse development, robustness and three locked fresh confirmations',{'primary_n':len(dev[(dev.variant=='final')&(dev.family=='xgboost')&(dev.dataset.isin(['bank','gas','electricity']))]),'fresh_datasets':len(fresh),'hold_controls':int((fresh.mean_gain==0).sum())})

def f4():
 sp=read('specialist.csv');mc=read('multiclass.csv');fig,ax=figure();a,b,c,d=ax
 datasets=sorted(sp.dataset.unique(),key=lambda x:name(x));comparators=sorted(sp.comparator.unique())
 for bb,axis,letter in [('catboost',a,'a'),('fttransformer',b,'b')]:
  g=sp[sp.backbone==bb]; mat=g.pivot_table(index='comparator',columns='dataset',values='mean_advantage',aggfunc='mean').reindex(index=comparators,columns=datasets)
  vals=mat.to_numpy();mask=np.ma.masked_invalid(vals)
  im=axis.imshow(mask,cmap='Blues',aspect='auto',vmin=0,vmax=max(.01,float(sp.mean_advantage.max())))
  axis.set_xticks(np.arange(len(datasets)),[name(x) for x in datasets],rotation=45,ha='right',fontsize=8)
  axis.set_yticks(range(len(comparators)),[x.replace('-XB','').replace('SeverityAware','Severity-Aware').replace('Periodic-5%-recalibration','Periodic 5%') for x in comparators],fontsize=8)
  for iy in range(len(comparators)):
   for ix in range(len(datasets)):
    if np.isfinite(vals[iy,ix]):axis.text(ix,iy,f'{vals[iy,ix]:.3f}',ha='center',va='center',fontsize=7.6,color='white' if vals[iy,ix]>.13 else C['navy'])
  title(axis,f'{name(bb)}: matched specialist comparisons',letter)
 fig.colorbar(im,ax=[a,b],shrink=.52,label='Mean Brier advantage',pad=.015)
 order=mc.sort_values(['dataset','backbone']); labels=[f'{name(r.dataset)} · {name(r.backbone)}' for r in order.itertuples()];yy=np.arange(len(order))
 c.errorbar(order.mean_VARIEL_advantage,yy,xerr=[order.mean_VARIEL_advantage-order.ci95_low,order.ci95_high-order.mean_VARIEL_advantage],fmt='o',color=C['blue'],ecolor=C['grey'],capsize=3,ms=4)
 c.axvline(0,color=C['red'],ls='--',lw=1.3)
 c.set(yticks=yy,yticklabels=labels,xlabel='VARIEL advantage over multiclass LTT (95% CI)');c.invert_yaxis();title(c,'All 12 three-backbone multiclass cells','c');clean(c,'x')
 cm=mc.groupby('backbone').agg(mean_adv=('mean_VARIEL_advantage','mean'),min_adv=('min_VARIEL_advantage','min'),fraction=('mean_LTT_label_fraction','mean'))
 xx=np.arange(len(cm));d.bar(xx-.18,cm.mean_adv,width=.36,color=C['green'],label='Mean Brier advantage');d.bar(xx+.18,cm.min_adv,width=.36,color=C['cyan'],label='Minimum seed-level advantage')
 d.set(xticks=xx,xticklabels=[name(s) for s in cm.index],ylabel='Brier advantage')
 d.legend(frameon=False,fontsize=8.3,loc='upper right',bbox_to_anchor=(0.98,1.20),borderaxespad=0.)
 title(d,'Multiclass consistency across backbones','d');clean(d)
 save(fig,'S04_crossbackbone_multiclass','Matched specialist advantages and 12 multiclass LTT cells',{'specialist_rows':len(sp),'specialist_negative_mean':int((sp.mean_advantage<0).sum()),'multiclass_rows':len(mc)})

def f5():
 hi=read('hi_paired.csv');dws=read('hi_dev_windows.csv');ews=read('hi_ext_windows.csv');both=pd.concat([dws.assign(role='Development'),ews.assign(role='External')],ignore_index=True)
 fig,ax=figure();a,b,c,d=ax
 z=hi.sort_values('mean_brier_advantage_VARIEL');yy=np.arange(len(z));colors=[C['orange'] if v=='development' else C['green'] for v in z.evidence_role]
 a.errorbar(z.mean_brier_advantage_VARIEL,yy,xerr=[z.mean_brier_advantage_VARIEL-z.ci95_low,z.ci95_high-z.mean_brier_advantage_VARIEL],fmt='none',ecolor=C['grey'],capsize=4,lw=1.8)
 a.scatter(z.mean_brier_advantage_VARIEL,yy,s=62,color=colors,zorder=3)
 a.axvline(0,lw=1,color=C['red'],ls='--');a.set(yticks=yy,yticklabels=[name(x) for x in z.dataset],xlabel='Brier advantage vs full retraining (95% CI)');a.invert_yaxis();title(a,'High-information: eight paired cells','a');clean(a,'x')
 ds=pd.concat([read('hi_dev_summary.csv').assign(role='Development'),read('hi_ext_summary.csv').assign(role='External')]);xs=ds.mean_canary_fraction*100;ys=ds.mean_improvement
 for role,col in [('Development','orange'),('External','green')]:
  g=ds[ds.role==role];b.scatter(g.mean_canary_fraction*100,g.mean_improvement,color=C[col],s=85,label=role,zorder=3)
 offsets={'ble_rssi':(-62,7),'rialto':(-44,8),'spam':(7,-15),'electricity':(-65,7),'gas':(5,8),'bank':(8,5),'insects':(7,5),'airlines':(5,5)}
 for r in ds.itertuples():
  offs=offsets.get(r.dataset,(5,5))
  b.annotate(name(r.dataset),(r.mean_canary_fraction*100,r.mean_improvement),xytext=offs,textcoords='offset points',fontsize=8,ha='right' if offs[0]<0 else 'left')
 b.set(xlabel='Mean nested canary fraction (%)',ylabel='Mean Brier improvement',xlim=(2.20,4.45));b.legend(frameon=False,loc='center right');title(b,'Evidence expense versus achieved repair','b');clean(b)
 for role,col in [('Development','orange'),('External','green')]:
  group=both[both.role==role]; c.hist(group.safety_H,bins=np.linspace(0,.0201,24),alpha=.6,color=C[col],label=role)
 c.axvline(.02,color=C['red'],ls='--',lw=1.5,label='Exact H cap = 0.02');c.set(xlabel='Exact prospective liability H',ylabel='Deployment windows');title(c,'High-information liability distribution','c');c.legend(frameon=False,fontsize=9);clean(c)
 states=both.copy();states['type']=np.where(states.eta.fillna(0)>0,'Guarded REPAIR','HOLD / raw FULL');pt=states.groupby(['role','type']).size().unstack(fill_value=0).reindex(['Development','External'])
 x=np.arange(len(pt));v=np.zeros(len(x));
 for nm,col in [('HOLD / raw FULL','grey'),('Guarded REPAIR','green')]:
  z0=pt.get(nm,pd.Series(np.zeros(len(pt)),index=pt.index)).to_numpy();d.bar(x,z0,bottom=v,color=C[col],width=.53,label=nm);v+=z0
 d.set(xticks=x,xticklabels=pt.index,ylabel='Evaluated deployment windows');d.legend(frameon=False);title(d,'Action types across information roles','d');clean(d)
 save(fig,'S05_high_information','High-information gains, canary use, exact H and action state frequencies',{'paired_cells':len(hi),'paired_wins':int(hi.strict_wins.sum()),'window_rows':len(both),'max_H':float(both.safety_H.max())})

def f6():
 sec=read('secondary.csv');main=sec[sec.scope=='main'];bs=['xgboost','catboost','fttransformer'];ds=sorted(main.dataset.unique(),key=name)
 fig,ax=figure();a,b,c,d=ax
 def heat(axis,metric,letter):
  g=main[main.metric==metric].pivot(index='dataset',columns='backbone',values='mean_advantage').reindex(index=ds,columns=bs)
  m=g.to_numpy(); maxv=max(abs(m.min()),abs(m.max()),.001)
  im=axis.imshow(m,cmap='RdBu',vmin=-maxv,vmax=maxv,aspect='auto')
  axis.set_xticks(range(3),[name(s) for s in bs],fontsize=9);axis.set_yticks(range(len(ds)),[name(s) for s in ds],fontsize=9)
  for i in range(len(ds)):
   for j in range(3):
    val=m[i,j];axis.text(j,i,f'{val:+.3f}',ha='center',va='center',fontsize=8,color='white' if abs(val)>.72*maxv else C['black'])
  title(axis,f'{metric.replace("ece15","ECE15").replace("brier","Brier").upper() if metric=="nll" else metric.replace("brier","Brier").replace("ece15","ECE15")}: mean advantage',letter)
  return m
 br=heat(a,'brier','a');nl=heat(b,'nll','b')
 # calibration ECE and classwise ECE -- paired points for each cell
 cal=main[main.metric.isin(['ece15','classwise_ece15'])].copy();idx=np.arange(len(ds)*3)
 for met,col,shift,lab in [('ece15','blue',-.14,'ECE15'),('classwise_ece15','orange',.14,'Classwise ECE15')]:
  vals=[float(cal[(cal.dataset==ds0)&(cal.backbone==bb)&(cal.metric==met)].mean_advantage.iloc[0]) for ds0 in ds for bb in bs]
  c.scatter(vals,idx+shift,s=25,color=C[col],label=lab,zorder=2)
 c.axvline(0,color=C['red'],lw=1.2,ls='--');c.set(yticks=idx,yticklabels=[f'{name(ds0)} · {name(bb)}' for ds0 in ds for bb in bs],xlabel='Calibration error advantage (lower is better)',ylim=(len(idx)-.3,-.9));c.tick_params(axis='y',labelsize=7.4)
 c.legend(frameon=False,ncol=2,fontsize=8.8,loc='lower right');title(c,'Calibration: all dataset × backbone cells','c');clean(c,'x')
 protected=['accuracy','balanced_accuracy','macro_f1','macro_recall','auroc','auprc','precision','macro_specificity'];mx=main[main.metric.isin(protected)].groupby('metric').mean_advantage.apply(lambda s:np.max(np.abs(s))).sort_values()
 d.barh(np.arange(len(mx)),mx.values,color=C['green'],height=.6)
 d.set(yticks=np.arange(len(mx)),yticklabels=[v.replace('_',' ').title() for v in mx.index],xlabel='Maximum absolute mean change');d.invert_yaxis();title(d,'Protected operating metrics','d');clean(d,'x')
 if mx.max()==0:d.set_xlim(0,.001)
 save(fig,'S06_probability_preservation','Main V24 Brier, NLL, calibration and protected outcomes',{'main_cells':len(main[main.metric=='brier']),'brier_positive_cells':int((br>0).sum()),'nll_positive_cells':int((nl>0).sum()),'max_protected_abs_delta':float(mx.max())})

def f7():
 run=read('compute.csv').sort_values('dataset');ds=[name(x) for x in run.dataset];yy=np.arange(len(run));fig,ax=figure();a,b,c,d=ax
 for lab,col,field,shift in [('VARIEL','blue','VARIEL_decision_seconds_per_window',-.24),('UPF','orange','UPF_decision_seconds_per_window',0),('Severity-Aware','red','Severity_decision_seconds_per_window',.24)]:
  a.barh(yy+shift,run[field],height=.23,color=C[col],label=lab)
 a.set(yticks=yy,yticklabels=ds,xlabel='Decision time (s / window; symlog)');a.set_xscale('symlog',linthresh=.2);a.invert_yaxis();a.legend(frameon=False,ncol=3,fontsize=8,loc='upper right',bbox_to_anchor=(0.99,1.18),borderaxespad=0.);title(a,'Controller-only decision overhead','a');clean(a,'x')
 b.plot(run.UPF_over_VARIEL_decision_ratio,yy,marker='o',color=C['orange'],label='UPF / VARIEL',lw=1.4)
 b.plot(run.Severity_over_VARIEL_decision_ratio,yy,marker='s',color=C['red'],label='Severity-Aware / VARIEL',lw=1.4)
 b.axvline(1,ls='--',color=C['navy'],lw=1.0);b.set(yticks=yy,yticklabels=ds,xlabel='Decision-time ratio (log; >1 favors VARIEL)');b.set_xscale('log');b.invert_yaxis();b.legend(frameon=False,fontsize=9);title(b,'Relative controller decision efficiency','b');clean(b,'x')
 c.barh(yy-.18,run.VARIEL_decision_seconds_per_window,color=C['blue'],label='Decision',height=.33)
 c.barh(yy+.18,run.VARIEL_training_seconds_per_window,color=C['cyan'],label='Training / endpoint generation',height=.33)
 c.set(yticks=yy,yticklabels=ds,xlabel='VARIEL matched time (s / window; symlog)');c.set_xscale('symlog',linthresh=.2);c.invert_yaxis();c.legend(frameon=False,fontsize=9);title(c,'Separate decision from training compute','c');clean(c,'x')
 for nm,col,field,shift in [('VARIEL','blue','VARIEL_total_seconds_per_window',-.23),('UPF','orange','UPF_total_seconds_per_window',0),('Severity-Aware','red','Severity_total_seconds_per_window',.23)]:
  d.barh(yy+shift,run[field],height=.22,color=C[col],label=nm)
 d.set(yticks=yy,yticklabels=ds,xlabel='End-to-end matched time (s / window; symlog)');d.set_xscale('symlog',linthresh=.2);d.invert_yaxis();d.legend(frameon=False,ncol=3,fontsize=8,loc='upper right',bbox_to_anchor=(0.99,1.18),borderaxespad=0.);title(d,'Total wall-clock: same measurement regime','d');clean(d,'x')
 save(fig,'S07_matched_compute','Controller, comparator ratios, training overhead and end-to-end timing',{'dataset_rows':len(run),'decision_faster_upf':int((run.UPF_decision_seconds_per_window>run.VARIEL_decision_seconds_per_window).sum()),'total_faster_upf':int((run.UPF_total_seconds_per_window>run.VARIEL_total_seconds_per_window).sum())})

def f8():
 ble=read('ble_seed0.csv').assign(domain='BLE RSSI');ins=read('insects_seed0.csv').assign(domain='INSECTS');df=pd.concat([ins,ble],ignore_index=True); fig,ax=figure();a,b,c,d=ax
 vmin=min(float(df.realized_gain.min()),-.001);vmax=max(float(df.realized_gain.max()),.001)
 # use fixed symmetric color range across both domains to be comparable
 norm=TwoSlopeNorm(vcenter=0,vmin=-max(abs(vmin),abs(vmax)),vmax=max(abs(vmin),abs(vmax)))
 for domain,axis,letter in [('INSECTS',a,'a'),('BLE RSSI',b,'b')]:
  g=df[df.domain==domain]
  im=axis.scatter(g.D_lower_bulk,g.whole_window_exact_H,s=45+500*g.eta_deploy,edgecolor='white',lw=.8,c=g.realized_gain,cmap='RdBu',norm=norm,zorder=3)
  axis.axvline(0,color=C['navy'],lw=1,ls=':');axis.axhline(.02,color=C['red'],lw=1,ls='--');axis.set(xlabel='Lower confidence bound Dᴸ',ylabel='Exact prospective liability H',ylim=(-.0004,.022))
  title(axis,f'{domain}: action, evidence and realized gain',letter);clean(axis)
 cb=fig.colorbar(im,ax=[a,b],shrink=.66,pad=.02);cb.set_label('Realized Brier gain (seed 0)')
 df['evidence']=np.where(df.D_lower_bulk>0,'Positive Dᴸ','Nonpositive Dᴸ');df['action']=np.where(df.eta_deploy>1e-12,'REPAIR','HOLD')
 gr=df.groupby(['domain','evidence']).action.apply(lambda z:(z=='REPAIR').mean()).unstack()
 gr=gr.reindex(index=['INSECTS','BLE RSSI'])
 for i,domain in enumerate(gr.index):
  for j,key in enumerate(['Nonpositive Dᴸ','Positive Dᴸ']):
   val=gr.get(key,pd.Series()).get(domain,np.nan)
   if np.isfinite(val):c.bar(i+j*.28-.14,val,width=.26,color=C['grey'] if j==0 else C['blue'],label=key if i==0 else '')
 c.set(xticks=[0,1],xticklabels=gr.index,ylabel='Fraction of windows repaired',ylim=(0,1.07));c.legend(frameon=False,fontsize=9);title(c,'Action frequency by evidence sign','c');clean(c)
 cc=df.groupby(['domain','action']).agg(windows=('window','size'),sum_positive=('realized_gain',lambda s:int((s>0).sum()))).reset_index()
 label_order=['INSECTS · HOLD','INSECTS · REPAIR','BLE RSSI · HOLD','BLE RSSI · REPAIR'];count=[];pos=[]
 for name0 in label_order:
  ds,action=name0.split(' · ');r=cc[(cc.domain==ds)&(cc.action==action)];count.append(int(r.windows.iloc[0]) if len(r) else 0);pos.append(int(r.sum_positive.iloc[0]) if len(r) else 0)
 xx=np.arange(4);d.bar(xx,count,color=C['grey'],width=.55,label='Zero/nonpositive gain');d.bar(xx,pos,color=C['green'],width=.55,label='Positive realized gain')
 # previous grey first gives total count + pos (wrong), replace via patch below
 d.cla();d.bar(xx,np.array(count)-np.array(pos),color='#DCE4EB',width=.56,label='Zero/nonpositive gain');d.bar(xx,pos,bottom=np.array(count)-np.array(pos),color=C['green'],width=.56,label='Positive realized gain')
 for i,(n,p) in enumerate(zip(count,pos)):d.text(i,n+.1,f'{n} windows',fontsize=9,ha='center')
 d.set(xticks=xx,xticklabels=['INSECTS\nHOLD','INSECTS\nREPAIR','BLE RSSI\nHOLD','BLE RSSI\nREPAIR'],ylabel='Deployment windows');d.legend(frameon=False,fontsize=8.5);title(d,'Action outcomes in the same frozen seed-0 traces','d');clean(d)
 save(fig,'S08_fresh_action_portraits','Real frozen seed-0 fresh-stream window traces; not a reconstruction from the uploaded PDF',{'total_windows':len(df),'INSECTS_windows':len(ins),'BLE_windows':len(ble),'max_exact_H':float(df.whole_window_exact_H.max()),'HOLD_windows':int((df.eta_deploy<=1e-12).sum())})

if __name__=='__main__':
 for f in [f1,f2,f3,f4,f5,f6,f7,f8]:f()
 print('DONE all eight source-checked figures')
