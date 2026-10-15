#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'sources'
PDF = ROOT / 'figures'
PNG = ROOT / 'png'
PDF.mkdir(exist_ok=True)
PNG.mkdir(exist_ok=True)

# ---------- Publication style ----------
NAVY = '#17365D'
BLUE = '#0B6DB7'
CYAN = '#17A9D2'
TEAL = '#0EA5A8'
GREEN = '#0AA678'
ORANGE = '#FF8A1F'
GOLD = '#F1C83D'
RED = '#EF4D4D'
GRAY = '#8C99A6'
DARK = '#111820'
GRID = '#D9E3EC'
LIGHT = '#EEF4F8'

mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12.0,
    'axes.titlesize': 17,
    'axes.labelsize': 14.5,
    'xtick.labelsize': 11.5,
    'ytick.labelsize': 11.5,
    'legend.fontsize': 11.0,
    'axes.edgecolor': DARK,
    'axes.linewidth': 1.2,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.facecolor': 'white',
    'figure.facecolor': 'white',
})

def stylize(ax, grid='y'):
    ax.spines[['top','right']].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

def ptitle(ax, letter, title):
    ax.set_title(f'({letter})  {title}', loc='left', color=NAVY, fontweight='bold', pad=13)

def save(fig, stem):
    fig.savefig(PDF / f'{stem}.pdf', bbox_inches='tight', pad_inches=0.07)
    fig.savefig(PNG / f'{stem}.png', dpi=180, bbox_inches='tight', pad_inches=0.07)
    plt.close(fig)

# ---------- Figure 2: Resource Cabinet ----------
def fig02():
    stages=pd.read_csv(SRC/'fig02_resource_stages.csv')
    dec=pd.read_csv(SRC/'fig02_racs_decisions.csv')
    per=pd.read_csv(SRC/'fig02_racs_per_dataset.csv')
    fig=plt.figure(figsize=(14.48,10.86))
    gs=fig.add_gridspec(2,2,left=.065,right=.97,bottom=.08,top=.965,hspace=.34,wspace=.23)

    ax=fig.add_subplot(gs[0,0]); ptitle(ax,'a','Operational resolution')
    x=np.arange(len(stages)); y=stages.resolved.to_numpy();
    ax.step(x,y,where='mid',color=BLUE,lw=3)
    ax.plot(x,y,'o',ms=10,mec=DARK,mew=1.0,mfc=CYAN,zorder=3)
    for xi,yi,u in zip(x,y,stages.unresolved):
        ax.text(xi, yi+45, f'{yi}/1080', ha='center', fontweight='bold', fontsize=13)
        ax.text(xi, 35, f'{int(u)} unresolved', ha='center', color=RED, fontsize=11)
    ax.set_xticks(x,stages.stage,rotation=22,ha='right')
    ax.set_ylabel('Operationally valid decisions')
    ax.set_ylim(0,1160); stylize(ax)

    ax=fig.add_subplot(gs[0,1]); ptitle(ax,'b','Where terminal rescue was needed')
    ds_order=['airlines','bank','electricity','gas','rialto','spam']
    # Reconstruct route stacks per dataset from all 180 decisions.
    labels={'preserved_baseline':'Resolved upstream','pparc_sharp':'SHARP','cppi_greg':'CPPI/RACS path','racs':'CPPI/RACS path'}
    stacks={'Resolved upstream':[],'SHARP':[],'CPPI/RACS path':[]}
    for ds in ds_order:
        g=dec[dec.dataset==ds]
        # upstream = not pparc_sharp/cppi_greg/racs
        up=(~g.route.isin(['pparc_sharp','cppi_greg','racs'])).sum()
        sh=(g.route=='pparc_sharp').sum()
        cp=len(g)-up-sh
        stacks['Resolved upstream'].append(up); stacks['SHARP'].append(sh); stacks['CPPI/RACS path'].append(cp)
    xx=np.arange(len(ds_order)); bottom=np.zeros(len(ds_order))
    for lab,col in [('Resolved upstream',BLUE),('SHARP',CYAN),('CPPI/RACS path',ORANGE)]:
        vals=np.asarray(stacks[lab]); ax.bar(xx,vals,bottom=bottom,width=.72,color=col,label=lab); bottom+=vals
    ax.set_xticks(xx,[d.title() for d in ds_order],rotation=28,ha='right')
    ax.set_ylabel('Decisions'); ax.set_ylim(0,195); stylize(ax)
    ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.53,1.01),fontsize=9.6,columnspacing=.9,handlelength=1.8)

    ax=fig.add_subplot(gs[1,0]); ptitle(ax,'c','Feedback cost across final decisions')
    groups=[100*dec.loc[dec.dataset==d,'label_fraction'].to_numpy() for d in ds_order]
    parts=ax.violinplot(groups,showmedians=True,showextrema=True,widths=.78)
    for body in parts['bodies']:
        body.set_facecolor('#6FD0CD'); body.set_edgecolor(NAVY); body.set_alpha(.88)
    for k in ['cmedians','cbars','cmins','cmaxes']:
        parts[k].set_color(NAVY); parts[k].set_linewidth(1.2)
    ax.axhline(10,color=RED,ls='--',lw=1.6)
    ax.text(6.22,10.18,'10% ceiling',color=RED,fontsize=10.5,ha='right')
    ax.set_xticks(np.arange(1,7),[d.title() for d in ds_order],rotation=28,ha='right')
    ax.set_ylabel('Target labels used (%)'); ax.set_ylim(-.25,10.6); stylize(ax)

    ax=fig.add_subplot(gs[1,1]); ptitle(ax,'d','RACS terminal rescue: exact vs equivalent')
    r=dec[dec.racs_triggered]
    exact=int(r.exact_winner_match.sum()); equiv=len(r)-exact
    ax.barh([0],[exact],height=.45,color=BLUE,label='Exact oracle action')
    ax.barh([0],[equiv],left=[exact],height=.45,color=GOLD,label='Operationally equivalent')
    ax.text(exact/2,0,str(exact),ha='center',va='center',color='white',fontweight='bold',fontsize=16)
    ax.text(exact+equiv/2,0,str(equiv),ha='center',va='center',color=DARK,fontweight='bold',fontsize=16)
    ax.set_xlim(0,21); ax.set_ylim(-.42,.50); ax.set_yticks([]); ax.set_xlabel('Previously unresolved decisions')
    ax.legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.54,1.03),fontsize=10.4)
    ax.text(0,-.32,'1080/1080 valid  |  0 unresolved  |  mean labels 2.35%  |  max regret 0.0009974',fontsize=10.2,color=NAVY)
    stylize(ax,grid=None)
    save(fig,'fig02_resource_evolution')

# ---------- Figure 3 ----------
def fig03():
    fr=pd.read_csv(SRC/'fig03_liability_frontier.csv')
    ab=pd.read_csv(SRC/'fig03_ablation_summary.csv')
    geo=pd.read_csv(SRC/'fig03_geometry_vs_uniform.csv').sort_values('difference',ascending=False)
    shortfam={'xgboost':'XGB','catboost':'CatBoost','ft_transformer':'FT'}
    shortds={'bank':'Bank','electricity':'Elec.','gas':'Gas','insects':'INSECTS'}
    labels=[f"{shortfam[r.family]} / {shortds[r.dataset]}" for r in geo.itertuples()]
    vals=geo.difference.to_numpy()
    fig=plt.figure(figsize=(14.48,10.86)); gs=fig.add_gridspec(2,2,left=.065,right=.97,bottom=.08,top=.965,hspace=.35,wspace=.27)
    ax=fig.add_subplot(gs[0,0]); ptitle(ax,'a','Utility across liability frontier')
    ax.fill_between(fr.epsilon,fr.lo,fr.hi,color=CYAN,alpha=.17,label='range across 10 primary cells')
    ax.plot(fr.epsilon,fr.mean_gain,'o-',color=BLUE,lw=3,ms=8)
    s=fr[np.isclose(fr.epsilon,.02)].iloc[0]
    ax.scatter([s.epsilon],[s.mean_gain],s=170,color=ORANGE,edgecolor=DARK,zorder=4)
    ax.text(.0215,s.mean_gain+.009,'selected ε = 0.02',color=ORANGE,fontsize=12)
    ax.set_xlabel('Prospective liability limit ε'); ax.set_ylabel('Mean Brier advantage'); ax.set_ylim(-.002,.102); stylize(ax)
    ax.legend(frameon=False,loc='lower right')

    ax=fig.add_subplot(gs[0,1]); ptitle(ax,'b','Exact H tracks safety level')
    ax.plot(fr.epsilon,fr.maxH,'o-',color=TEAL,lw=3,ms=8,label='observed max exact H')
    ax.plot(fr.epsilon,fr.epsilon,'--',color=RED,lw=1.8,label='H = ε boundary')
    for x,y in zip(fr.epsilon,fr.maxH): ax.text(x,y+.0011,f'{y:.3f}',ha='center',fontsize=9.8)
    ax.set_xlabel('ε'); ax.set_ylabel('Maximum exact H'); stylize(ax); ax.legend(frameon=False,loc='upper left')

    ax=fig.add_subplot(gs[1,0]); ptitle(ax,'c','Ablation phase space')
    cmap={'final':BLUE,'maximal_safe':GREEN,'no_exact_H':RED,'terminal_blind':GRAY,'uniform_canary':CYAN}
    lab={'final':'Final','maximal_safe':'Max-safe','no_exact_H':'No exact H','terminal_blind':'Terminal-blind','uniform_canary':'Uniform canary'}
    offsets={'final':(10,8),'maximal_safe':(10,20),'no_exact_H':(-12,-10),'terminal_blind':(10,-8),'uniform_canary':(10,-2)}
    for _,r in ab.iterrows():
        ax.scatter(r.maxH,r.mean_gain,s=135,color=cmap[r.variant],edgecolor=DARK,zorder=3)
        dx,dy=offsets[r.variant]
        ax.annotate(lab[r.variant],(r.maxH,r.mean_gain),xytext=(dx,dy),textcoords='offset points',fontsize=9.3,ha='right' if r.variant=='no_exact_H' else 'left')
    ax.axvline(.02,color=ORANGE,ls='--',lw=1.5)
    ax.set_xscale('log'); ax.set_xlabel('Maximum exact H (log scale)'); ax.set_ylabel('Mean Brier advantage'); ax.set_ylim(-.002,.074); stylize(ax)

    ax=fig.add_subplot(gs[1,1]); ptitle(ax,'d','Geometry canaries vs uniform')
    yy=np.arange(len(labels)); ax.barh(yy,vals,color=BLUE,height=.68)
    ax.set_yticks(yy,labels); ax.invert_yaxis(); ax.set_xlabel('Δ mean Brier advantage (geometry - uniform)'); stylize(ax,'x')
    save(fig,'fig03_sparse_geometry_liability')

# ---------- Figure 4 ----------
def fig04():
    df=pd.read_csv(SRC/'fig04_temporal_actionability.csv')
    # Slightly more horizontal room and more vertical separation so the wrapped
    # titles read cleanly without changing any data.
    fig=plt.figure(figsize=(14.9,10.86))
    gs=fig.add_gridspec(2,2,left=.072,right=.965,bottom=.08,top=.965,hspace=.41,wspace=.34)
    colors={.98:RED,.90:ORANGE,.75:BLUE,.60:CYAN,.50:GRAY}

    ax=fig.add_subplot(gs[0,0]); ptitle(ax,'a','Captured opportunity')
    piv=df.pivot(index='rho',columns='staleness',values='capture_mean').sort_index(ascending=False)
    cmap=LinearSegmentedColormap.from_list('cap',[RED,'#F8D8D8','white','#B7D9EA',BLUE])
    im=ax.imshow(piv.values,aspect='auto',cmap=cmap,vmin=-.05,vmax=1.0)
    ax.set_xticks(range(len(piv.columns)),[str(int(x)) for x in piv.columns])
    ax.set_yticks(range(len(piv.index)),[f'{x:.2f}' for x in piv.index])
    ax.set_xlabel('Label staleness'); ax.set_ylabel('Persistence ρ')
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j,i,f'{piv.values[i,j]:.2f}',ha='center',va='center',fontsize=10)
    cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.03); cb.set_label('Captured fraction')

    # Wrap the title intentionally so it stays inside its panel.
    ax=fig.add_subplot(gs[0,1]); ptitle(ax,'b','Capture falls as evidence\nbecomes stale')
    for rho in [.98,.90,.75,.60,.50]:
        g=df[np.isclose(df.rho,rho)].sort_values('staleness')
        ax.plot(g.staleness,g.capture_mean,'o-',lw=2.3,ms=8,color=colors[rho])
        y=float(g.capture_mean.iloc[-1]); off={.98:.012,.90:.010,.75:.012,.60:.020,.50:-.035}[rho]
        ax.text(3.12,y+off,rf'ρ={rho:.2f}',color=colors[rho],fontsize=10.4,va='center')
    ax.axhline(0,color=DARK,lw=.8)
    ax.set_xlim(0,3.85); ax.set_ylim(-.08,1.02)
    ax.set_xlabel('Label staleness'); ax.set_ylabel('Captured fraction'); stylize(ax)

    # The scientific variation is tiny (CV=0.018%). Keep the honest wide
    # contextual axis, but add a *small* upper-right zoom that magnifies only
    # the thin near-invariant band without disturbing the rest of the panel.
    ax=fig.add_subplot(gs[1,0]); ptitle(ax,'c','Oracle repair opportunity remains nearly fixed')
    series=[]
    for rho in [.98,.90,.75,.60,.50]:
        g=df[np.isclose(df.rho,rho)].sort_values('staleness')
        series.append((rho,g))
        ax.plot(g.staleness,g.oracle_gain_mean,'o-',lw=1.9,ms=6,color=colors[rho])
    ax.set_xlabel('Label staleness'); ax.set_ylabel('Oracle Brier opportunity')
    ax.set_ylim(.0068,.0078); stylize(ax)
    ax.text(.05,.00772,'CV across persistence ≈ 0.000183 (0.018%)',fontsize=10,color=NAVY,
            bbox=dict(boxstyle='round,pad=.28',fc='#F5F8FB',ec='#C9D7E4'))

    y_all=np.concatenate([g.oracle_gain_mean.to_numpy() for _,g in series])
    pad=max((y_all.max()-y_all.min())*0.18, 2e-7)
    # small upper-right zoom box in the previously empty space
    axin=ax.inset_axes([0.69,0.61,0.25,0.24])
    for rho,g in series:
        axin.plot(g.staleness,g.oracle_gain_mean,'o-',lw=1.55,ms=4.3,color=colors[rho])
    axin.set_xlim(-0.05,3.05)
    axin.set_ylim(y_all.min()-pad, y_all.max()+pad)
    axin.set_xticks([]); axin.set_yticks([])
    axin.set_facecolor('#FBFDFF')
    for side in ['top','right','left','bottom']:
        axin.spines[side].set_linewidth(1.0)
        axin.spines[side].set_color(NAVY)
    axin.set_title('zoom',fontsize=8.8,color=NAVY,pad=1.5)
    # Draw a small source window only (not a box over the whole band), then
    # connect that local window to the zoom inset manually.
    from matplotlib.patches import Rectangle, ConnectionPatch
    x0, x1 = 1.20, 1.66
    y0, y1 = 0.007205, 0.007355
    src = Rectangle((x0, y0), x1-x0, y1-y0, fill=False, ec=NAVY, lw=1.0)
    ax.add_patch(src)
    con1 = ConnectionPatch(xyA=(x1, y1), coordsA=ax.transData,
                           xyB=(0.0, 1.0), coordsB=axin.transAxes,
                           color=NAVY, lw=1.0)
    con2 = ConnectionPatch(xyA=(x1, y0), coordsA=ax.transData,
                           xyB=(0.0, 0.0), coordsB=axin.transAxes,
                           color=NAVY, lw=1.0)
    ax.add_artist(con1)
    ax.add_artist(con2)

    # Wrap the long title before it reaches the neighbouring panel.
    ax=fig.add_subplot(gs[1,1]); ptitle(ax,'d','Utility variation grows when old labels\nstop being actionable')
    for rho in [.98,.90,.75,.60,.50]:
        g=df[np.isclose(df.rho,rho)].sort_values('staleness')
        ax.plot(g.staleness,g.utility_variation_mean,'o-',lw=1.9,ms=6,color=colors[rho])
    ax.set_xlabel('Label staleness'); ax.set_ylabel('Utility variation'); stylize(ax)
    save(fig,'fig04_temporal_actionability')

def fig05():
    hi=pd.read_csv(SRC/'fig05_hi8.csv')
    st=pd.read_csv(SRC/'fig05_stress2.csv')
    fr=pd.read_csv(SRC/'fig05_fresh3.csv')
    fig=plt.figure(figsize=(14.48,10.86)); gs=fig.add_gridspec(2,2,left=.075,right=.965,bottom=.08,top=.965,hspace=.38,wspace=.35)
    def forest(ax, data, title, letter, color, xcol='mean_brier_advantage_VARIEL',lo='ci95_low',hi='ci95_high', labelcol='display', annotate='10/10', xlim=None, text_x=None):
        ptitle(ax,letter,title); data=data.copy().reset_index(drop=True); y=np.arange(len(data))[::-1]
        means=data[xcol].to_numpy(); lows=data[lo].to_numpy(); highs=data[hi].to_numpy()
        ax.errorbar(means,y,xerr=np.vstack([means-lows,highs-means]),fmt='o',color=color,mec=color,ms=9,elinewidth=2,capsize=4)
        ax.axvline(0,color=DARK,lw=.8); ax.set_yticks(y,data[labelcol]); ax.set_xlabel('Mean Brier advantage'); stylize(ax)
        if xlim: ax.set_xlim(*xlim)
        xr=ax.get_xlim(); xpos=text_x if text_x is not None else xr[1]-0.045*(xr[1]-xr[0])
        for yi in y: ax.text(xpos,yi,annotate,ha='right',va='center',color=NAVY,fontsize=10.8)

    # Give the annotations a clear right-side gutter.
    forest(fig.add_subplot(gs[0,0]),hi[hi.evidence_role=='development'],'HI development/challenge','a',BLUE,xlim=(-.0005,.0225),text_x=.0219)
    forest(fig.add_subplot(gs[0,1]),hi[hi.evidence_role=='external'],'Cross-domain external confirmation','b',TEAL,xlim=(-.0005,.0260),text_x=.0253)

    ax=fig.add_subplot(gs[1,0]); ptitle(ax,'c','Specialist/stress generalization')
    data=st.copy().sort_values('display'); y=np.arange(len(data))[::-1]
    means=data.mean_gain.to_numpy(); sd=data.sd_gain.to_numpy()
    ax.errorbar(means,y,xerr=sd,fmt='o',color=ORANGE,mec=ORANGE,ms=9,elinewidth=2,capsize=4)
    ax.axvline(0,color=DARK,lw=.8); ax.set_yticks(y,data.display); ax.set_xlabel('Mean Brier advantage')
    # Wider right margin: the 10/10 labels no longer touch the orange error bars.
    ax.set_xlim(-.00015,.00620); stylize(ax)
    for yi in y: ax.text(.00605,yi,'10/10',ha='right',va='center',color=NAVY,fontsize=10.8)

    ax=fig.add_subplot(gs[1,1]); ptitle(ax,'d','Locked fresh confirmation')
    y=np.arange(len(fr))[::-1]; means=fr.mean_gain.to_numpy(); sd=fr.sd_gain.to_numpy()
    ax.errorbar(means,y,xerr=sd,fmt='o',color=GREEN,mec=GREEN,ms=9,elinewidth=2,capsize=4)
    ax.axvline(0,color=DARK,lw=.8); ax.set_yticks(y,fr.display); ax.set_xlabel('Mean Brier advantage')
    # Same fix for Older Activity: put 10/10 in its own clear right-side gutter.
    ax.set_xlim(-.002,.061); stylize(ax)
    for yi,row in zip(y,fr.itertuples()):
        ax.text(.0600,yi,'HOLD' if row.mean_gain==0 else '10/10',ha='right',va='center',color=NAVY,fontsize=10.8)
    save(fig,'fig05_generalization_13datasets')

def fig06():
    v9=pd.read_csv(SRC/'fig06_v9_summary.csv'); v13=pd.read_csv(SRC/'fig06_v13_multiclass.csv')
    dorder=['bank','electricity','occupancy_detection_v2','eeg_eye_state','gas','airlines','rialto','spam','insects','ble_rssi']
    comp_order=['LTT-over-MCGrad-XB','MCGrad-0.1.5-XB','Periodic-5%-recalibration','Rolling-FULL','SeverityAware-XB-strong','UPF-XB']
    comp_label={'LTT-over-MCGrad-XB':'LTT','MCGrad-0.1.5-XB':'MCGrad','Periodic-5%-recalibration':'Periodic-5%','Rolling-FULL':'Rolling FULL','SeverityAware-XB-strong':'Severity-Aware','UPF-XB':'UPF'}
    dslabel={'bank':'Bank','electricity':'Electricity','occupancy_detection_v2':'Occupancy','eeg_eye_state':'EEG','gas':'Gas','airlines':'Airlines','rialto':'Rialto','spam':'Spam','insects':'INSECTS','ble_rssi':'BLE RSSI'}
    def matrix(backbone):
        mat=np.full((len(comp_order),len(dorder)),np.nan); win=np.full_like(mat,np.nan)
        g=v9[v9.backbone==backbone]
        for i,c in enumerate(comp_order):
            for j,d in enumerate(dorder):
                q=g[(g.comparator==c)&(g.dataset==d)]
                if len(q): mat[i,j]=q.iloc[0].mean_advantage; win[i,j]=q.iloc[0].wins
        return mat,win

    # Make this figure physically wider.  The bottom two panels now have their
    # own breathing room, so the panel-C colorbar and panel-D labels cannot collide.
    fig=plt.figure(figsize=(16.4,10.86))
    outer=fig.add_gridspec(3,1,left=.068,right=.965,bottom=.07,top=.97,height_ratios=[1.2,1.2,.78],hspace=.42)
    for ax,bb,letter,title in [(fig.add_subplot(outer[0]),'catboost','a','CatBoost: final specialist/system comparisons'),(fig.add_subplot(outer[1]),'fttransformer','b','FT-Transformer: final specialist/system comparisons')]:
        mat,win=matrix(bb); vmax=np.nanmax(mat); im=ax.imshow(mat,cmap='Blues',vmin=0,vmax=vmax,aspect='auto')
        ax.set_yticks(range(len(comp_order)),[comp_label[c] for c in comp_order]); ax.set_xticks(range(len(dorder)),[dslabel[d] for d in dorder],rotation=25,ha='right')
        ptitle(ax,letter,title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i,j]):
                    col='white' if mat[i,j]>.58*vmax else DARK
                    ax.text(j,i,f'{mat[i,j]:.3f}\n{int(win[i,j])}/10',ha='center',va='center',fontsize=8.8,color=col)
        cb=fig.colorbar(im,ax=ax,fraction=.014,pad=.012); cb.set_label('Mean Brier advantage')

    # Deliberately generous gap between panels C and D.
    bottom=outer[2].subgridspec(1,2,wspace=.62,width_ratios=[1.0,1.0])
    ax=fig.add_subplot(bottom[0,0]); ptitle(ax,'c','Multiclass LTT: all three backbones')
    bbs=['xgboost','catboost','fttransformer']; md=['gas','rialto','insects','ble_rssi']; ml={'xgboost':'XGBoost','catboost':'CatBoost','fttransformer':'FT-Transformer'}
    mat=np.zeros((3,4)); wins=np.zeros((3,4))
    for i,b in enumerate(bbs):
        for j,d in enumerate(md):
            q=v13[(v13.backbone==b)&(v13.dataset==d)].iloc[0]; mat[i,j]=q.mean_VARIEL_advantage; wins[i,j]=q.wins
    im=ax.imshow(mat,cmap='Blues',vmin=0,vmax=np.max(mat),aspect='auto')
    ax.set_yticks(range(3),[ml[x] for x in bbs]); ax.set_xticks(range(4),[dslabel[d] for d in md],rotation=18,ha='right')
    for i in range(3):
        for j in range(4):
            col='white' if mat[i,j]>.55*np.max(mat) else DARK
            ax.text(j,i,f'{mat[i,j]:.3f}\n10/10',ha='center',va='center',fontsize=8.8,color=col)
    cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.035); cb.set_label('Mean Brier advantage',labelpad=10)

    ax=fig.add_subplot(bottom[0,1]); ptitle(ax,'d','Consistency of the terminal closures')
    ax.barh([1,0],[100,100],height=.52,color=[BLUE,GREEN])
    ax.set_yticks([1,0],['V9 cross-backbone','V13 multiclass LTT'])
    ax.tick_params(axis='y',pad=7)
    ax.set_xlim(0,100); ax.set_ylim(-.65,1.45); ax.set_xlabel('Paired seed wins (%)')
    ax.text(50,1,'720/720',ha='center',va='center',color='white',fontweight='bold',fontsize=18)
    ax.text(50,0,'120/120',ha='center',va='center',color='white',fontweight='bold',fontsize=18)
    ax.text(2,-.42,'0 losses · 0 ties   |   all CI lower bounds > 0   |   all Holm-adjusted p < .05',fontsize=9.0,color=NAVY,va='center')
    stylize(ax,'x')
    save(fig,'fig06_crossbackbone_multiclass')

def fig07():
    rb=pd.read_csv(SRC/'fig07_robustness_summary.csv'); st=pd.read_csv(SRC/'fig08_v24_stress_summary.csv')
    names={'bank':'Bank','electricity':'Electricity','gas':'Gas','insects':'INSECTS','occupancy_detection_v2':'Occupancy','eeg_eye_state':'EEG'}
    cols={'bank':BLUE,'electricity':CYAN,'gas':ORANGE,'insects':GREEN,'occupancy_detection_v2':RED,'eeg_eye_state':GOLD}
    fig=plt.figure(figsize=(14.48,10.86)); gs=fig.add_gridspec(2,2,left=.07,right=.965,bottom=.08,top=.965,hspace=.35,wspace=.28)
    ax=fig.add_subplot(gs[0,0]); ptitle(ax,'a','Label-budget frontier')
    for ds in names:
        g=rb[rb.case.str.startswith('budget_') & (rb.dataset==ds)].copy();
        if g.empty: continue
        g['x']=g.case.str.replace('budget_','',regex=False).astype(float)*100; g=g.sort_values('x'); ax.plot(g.x,g.mean_gain,'o-',lw=2.3,ms=7,color=cols[ds],label=names[ds])
    ax.set_xlabel('Target-label budget (%)'); ax.set_ylabel('Mean Brier advantage'); stylize(ax); ax.legend(frameon=False,ncol=2,loc='center right')
    ax=fig.add_subplot(gs[0,1]); ptitle(ax,'b','Label-noise stress')
    for ds in names:
        g=rb[rb.case.str.startswith('noise_') & (rb.dataset==ds)].copy();
        if g.empty: continue
        g['x']=g.case.str.replace('noise_','',regex=False).astype(float)*100; g=g.sort_values('x'); ax.plot(g.x,g.mean_gain,'o-',lw=2.3,ms=7,color=cols[ds])
    ax.set_xlabel('Label noise (%)'); ax.set_ylabel('Mean Brier advantage'); stylize(ax)
    ax=fig.add_subplot(gs[1,0]); ptitle(ax,'c','Canary-dropout stress')
    for ds in names:
        g=rb[rb.case.str.startswith('dropout_') & (rb.dataset==ds)].copy();
        if g.empty: continue
        g['x']=g.case.str.replace('dropout_','',regex=False).astype(float)*100; g=g.sort_values('x'); ax.plot(g.x,g.mean_gain,'o-',lw=2.3,ms=7,color=cols[ds])
        # Mark the harsh endpoint if fewer than 10 positive seeds
        last=g.iloc[-1]
        if int(last.positive_seeds)<int(last.seeds): ax.text(last.x+.6,last.mean_gain,f'{int(last.positive_seeds)}/{int(last.seeds)}',color=RED,fontsize=11,va='center')
    ax.set_xlabel('Canary dropout (%)'); ax.set_ylabel('Mean Brier advantage'); stylize(ax)
    ax=fig.add_subplot(gs[1,1]); ptitle(ax,'d','Final stress tests after preservation')
    metrics=['brier','nll','ece15','classwise_ece15']; mlabel={'brier':'Brier','nll':'NLL','ece15':'ECE','classwise_ece15':'Class-ECE'}; mcol={'brier':BLUE,'nll':GREEN,'ece15':CYAN,'classwise_ece15':GOLD}
    groups=[('catboost','eeg_eye_state','EEG\nCatBoost'),('fttransformer','eeg_eye_state','EEG\nFT'),('catboost','occupancy_detection_v2','Occupancy\nCatBoost'),('fttransformer','occupancy_detection_v2','Occupancy\nFT')]
    xx=np.arange(len(groups)); width=.18
    for k,m in enumerate(metrics):
        vals=[]
        for b,d,_ in groups: vals.append(float(st[(st.backbone==b)&(st.dataset==d)&(st.metric==m)].mean_advantage.iloc[0]))
        ax.bar(xx+(k-1.5)*width,vals,width=width,color=mcol[m],label=mlabel[m])
    ax.axhline(0,color=DARK,lw=.8); ax.set_xticks(xx,[g[2] for g in groups]); ax.set_ylabel('Mean advantage'); stylize(ax); ax.legend(frameon=False,ncol=2,loc='upper right')
    save(fig,'fig07_robustness_stress')

# ---------- Figure 8 ----------
def fig08():
    df=pd.read_csv(SRC/'fig08_v24_main_summary.csv')
    bbs=['xgboost','catboost','fttransformer']; btitle={'xgboost':'XGBoost','catboost':'CatBoost','fttransformer':'FT-Transformer'}
    dorder=['airlines','bank','electricity','gas','rialto','spam','insects','ble_rssi']; dl={'airlines':'Airlines','bank':'Bank','electricity':'Electricity','gas':'Gas','rialto':'Rialto','spam':'Spam','insects':'INSECTS','ble_rssi':'BLE RSSI'}
    metrics=['brier','nll','ece15','classwise_ece15']; ml=['Brier','NLL','ECE','Class-ECE']
    fig=plt.figure(figsize=(14.48,10.86)); gs=fig.add_gridspec(2,2,left=.075,right=.965,bottom=.08,top=.965,hspace=.38,wspace=.42)
    axes=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,0])]
    for ax,b,letter in zip(axes,bbs,['a','b','c']):
        mat=np.zeros((4,8));
        for i,m in enumerate(metrics):
            for j,d in enumerate(dorder): mat[i,j]=float(df[(df.backbone==b)&(df.dataset==d)&(df.metric==m)].mean_advantage.iloc[0])
        vmax=np.max(mat); im=ax.imshow(mat,cmap='Blues',vmin=0,vmax=vmax,aspect='auto'); ptitle(ax,letter,btitle[b]); ax.set_yticks(range(4),ml); ax.set_xticks(range(8),[dl[d] for d in dorder],rotation=45,ha='right')
        for i in range(4):
            for j in range(8):
                val=mat[i,j]; col='white' if val>.58*vmax else DARK; ax.text(j,i, f'{val:.3g}',ha='center',va='center',fontsize=8.1,color=col)
        cb=fig.colorbar(im,ax=ax,fraction=.028,pad=.015)
        ax.tick_params(axis='y',labelsize=10.5)
    ax=fig.add_subplot(gs[1,1]); ptitle(ax,'d','Operational outputs are exactly preserved')
    ops=['accuracy','balanced_accuracy','macro_f1','macro_recall','macro_specificity','auroc','auprc','sensitivity','specificity','precision']; ol=['Accuracy','Balanced Accuracy','Macro F1','Macro Recall','Macro Specificity','AUROC','AUPRC','Sensitivity','Specificity','Precision']
    yy=np.arange(len(ops))[::-1]; ax.scatter(np.zeros(len(ops)),yy,s=95,color=BLUE)
    for yi in yy: ax.text(.035,yi,'0',va='center',fontsize=9.7)
    ax.axvline(0,color=DARK,lw=.9); ax.set_yticks(yy,ol); ax.set_xlim(-1,1); ax.set_xlabel('Maximum |mean advantage| across main cells'); stylize(ax)
    ax.text(.17,.45,'24/24 Brier positive\n24/24 NLL positive\n24/24 ECE nonnegative\n24/24 Class-ECE nonnegative',fontsize=9.4,color=NAVY,bbox=dict(boxstyle='round,pad=.35',fc='#F5F8FB',ec='#C9D7E4'))
    ax.tick_params(axis='y',labelsize=10.2)
    save(fig,'fig08_probability_preservation')

# ---------- Figure 9 ----------
def fig09():
    rt=pd.read_csv(SRC/'fig09_runtime_by_dataset.csv'); seed=pd.read_csv(SRC/'fig09_runtime_seedwise.csv')
    order=['airlines','bank','ble_rssi','electricity','gas','insects','rialto','spam']; labels=[x.replace('_',' ').title().replace('Ble Rssi','BLE RSSI') for x in order]
    rt=rt.set_index('dataset').loc[order].reset_index()
    fig=plt.figure(figsize=(14.48,10.86)); gs=fig.add_gridspec(2,2,left=.07,right=.965,bottom=.08,top=.955,hspace=.37,wspace=.30)

    ax=fig.add_subplot(gs[0,0]); ptitle(ax,'a','Controller decision overhead')
    xx=np.arange(len(rt)); w=.22
    ax.bar(xx-w,rt.VARIEL_decision_seconds_per_window,width=w,color=BLUE,label='VARIEL')
    ax.bar(xx,rt.UPF_decision_seconds_per_window,width=w,color=CYAN,label='UPF')
    ax.bar(xx+w,rt.Severity_decision_seconds_per_window,width=w,color=ORANGE,label='Severity-Aware')
    ax.set_yscale('log'); ax.set_xticks(xx,labels,rotation=35,ha='right'); ax.set_ylabel('Decision seconds / window (log)'); stylize(ax)
    # Keep the key below the title but above the bars, with explicit headroom so it
    # touches neither the title nor the orange Severity-Aware bar.
    ymin=min(rt[['VARIEL_decision_seconds_per_window','UPF_decision_seconds_per_window','Severity_decision_seconds_per_window']].min())
    ymax=max(rt[['VARIEL_decision_seconds_per_window','UPF_decision_seconds_per_window','Severity_decision_seconds_per_window']].max())
    ax.set_ylim(ymin/1.9, ymax*9.5)
    ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,.995),borderaxespad=0.0,columnspacing=1.6)

    ax=fig.add_subplot(gs[0,1]); ptitle(ax,'b','Per-dataset decision speedup')
    yy=np.arange(len(rt))[::-1]
    ax.barh(yy+.15,rt.UPF_over_VARIEL_decision_ratio,height=.28,color=CYAN,label='UPF / VARIEL')
    ax.barh(yy-.15,rt.Severity_over_VARIEL_decision_ratio,height=.28,color=ORANGE,label='Severity / VARIEL')
    ax.set_yticks(yy,labels); ax.set_xlabel('× slower than VARIEL controller'); stylize(ax,'x'); ax.legend(frameon=False,loc='upper right')

    ax=fig.add_subplot(gs[1,0]); ptitle(ax,'c','Information and safety footprint')
    ax.bar(xx,100*rt.VARIEL_label_fraction,color=BLUE,width=.65,label='VARIEL target labels (%)')
    ax.set_xticks(xx,labels,rotation=35,ha='right'); ax.set_ylabel('VARIEL target labels (%)'); ax.set_ylim(0,5.1); stylize(ax)
    ax2=ax.twinx(); ax2.plot(xx,rt.VARIEL_max_H,'o--',color=RED,lw=1.6,ms=8,label='Maximum H'); ax2.set_ylim(0,.024); ax2.set_ylabel('Maximum H',color=RED)
    handles=[Patch(color=BLUE,label='VARIEL target labels (%)'),Line2D([0],[0],color=RED,marker='o',ls='--',label='Maximum H')]
    ax.legend(handles=handles,frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.03))

    ax=fig.add_subplot(gs[1,1]); ptitle(ax,'d','Seed-level controller speed ratios')
    up=seed['UPF_decision_seconds_per_window']/seed['VARIEL_decision_seconds_per_window']; sev=seed['Severity_decision_seconds_per_window']/seed['VARIEL_decision_seconds_per_window']
    parts=ax.violinplot([up,sev],positions=[1,2],widths=.75,showmedians=True,showextrema=True)
    for body,col in zip(parts['bodies'],[CYAN,'#FFB264']): body.set_facecolor(col); body.set_edgecolor(NAVY); body.set_alpha(.85)
    for k in ['cmedians','cbars','cmins','cmaxes']: parts[k].set_color(NAVY); parts[k].set_linewidth(1.1)
    ax.set_xticks([1,2],['UPF / VARIEL','Severity / VARIEL']); ax.set_ylabel('Decision-time ratio'); stylize(ax)
    ax.text(.55,430,'80/80 faster vs UPF\n80/80 faster vs Severity-Aware',fontsize=11,color=NAVY,ha='left')
    save(fig,'fig09_efficiency')

def fig10():
    ins=pd.read_csv(SRC/'fig10_insects_seed0_trace.csv').sort_values('window'); ble=pd.read_csv(SRC/'fig10_ble_rssi_seed0_trace.csv').sort_values('window')
    fig=plt.figure(figsize=(14.48,10.86)); outer=fig.add_gridspec(1,2,left=.065,right=.97,bottom=.07,top=.965,wspace=.38)
    def col(sub,data,title,letter):
        gs=sub.subgridspec(4,1,height_ratios=[1.15,1.05,1.1,.52],hspace=.18)
        ax=fig.add_subplot(gs[0]); ptitle(ax,letter,title)
        ax.plot(data.window,data.raw_brier,color=GRAY,lw=2.5,label='baseline'); ax.plot(data.window,data.final_brier,color=BLUE,lw=2.5,label='VARIEL'); ax.set_ylabel('Brier'); ax.legend(frameon=False,ncol=2,loc='upper center'); stylize(ax); ax.tick_params(labelbottom=False)
        ax=fig.add_subplot(gs[1]); vals=np.maximum(data.realized_gain,0); ax.bar(data.window,vals,color=GREEN,width=.62); ax.set_ylabel('ΔBrier'); stylize(ax); ax.tick_params(labelbottom=False)
        for x,v in zip(data.window,vals):
            if v>0: ax.text(x,v+max(vals.max()*.035,1e-5),f'+{v:.3f}',ha='center',fontsize=8.2)
        ax=fig.add_subplot(gs[2]); ax.plot(data.window,data.eta_deploy,'o-',color=ORANGE,lw=2,label='η'); ax2=ax.twinx(); ax2.plot(data.window,data.whole_window_exact_H,'o-',color=RED,lw=1.6,label='H'); ax2.axhline(.02,color=RED,ls='--',lw=1); ax.set_ylabel('η',labelpad=-2); ax2.set_ylabel('H',color=RED,labelpad=-2); ax.tick_params(labelbottom=False); stylize(ax)
        ax=fig.add_subplot(gs[3]); y=.5; ax.set_xlim(data.window.min()-.5,data.window.max()+.5); ax.set_ylim(0,1); ax.set_yticks([]); ax.set_xticks(data.window); ax.set_xlabel('Deployment window')
        for _,r in data.iterrows():
            fc=BLUE if bool(r.act_bulk) else LIGHT; ax.scatter(r.window,y,s=110,facecolor=fc,edgecolor=NAVY,zorder=3); ax.text(r.window,.86,f'{100*r.canary_fraction:.1f}%',ha='center',fontsize=7.6,color=NAVY)
        ax.text(.0,-.56,'blue = REPAIR   light = HOLD',transform=ax.transAxes,fontsize=8.6,color=NAVY); ax.spines[['top','right','left']].set_visible(False)
    col(outer[0],ins,'INSECTS','a'); col(outer[1],ble,'BLE RSSI','b')
    save(fig,'fig10_deployment_trajectory')


# ---------- Figure 11: supplementary action phase portrait ----------
def fig11():
    aw=pd.read_csv(SRC/'fig11_action_phase_portrait.csv')
    fig=plt.figure(figsize=(14.48,10.86)); gs=fig.add_gridspec(2,2,left=.075,right=.965,bottom=.08,top=.965,hspace=.37,wspace=.43)
    for idx,name in enumerate(['INSECTS','BLE RSSI']):
        ax=fig.add_subplot(gs[0,idx]); ptitle(ax,['a','b'][idx],name)
        d=aw[aw.display_dataset==name].copy(); h=d.whole_window_exact_H.to_numpy(); u=d.D_lower_bulk.to_numpy(); g=d.realized_gain.to_numpy(); size=30+180*np.clip(d.eta_deploy.to_numpy(),0,1)
        vmax=max(abs(float(np.nanmin(g))),abs(float(np.nanmax(g))),0.002)
        sc=ax.scatter(u,h,s=size,c=g,cmap='RdBu',vmin=-vmax,vmax=vmax,alpha=.72,edgecolor=GRAY,linewidth=.5)
        ax.axvline(0,color=DARK,lw=.8); ax.axhline(.02,color=RED,ls='--',lw=1.2); ax.set_xlabel(r'Lower-bound repair utility $D_L$'); ax.set_ylabel('Exact prospective H'); stylize(ax)
        ax.text(.97,.95,'bubble size = deployed η',transform=ax.transAxes,ha='right',va='top',fontsize=10,color=NAVY)
        cb=fig.colorbar(sc,ax=ax,fraction=.030,pad=.015); cb.set_label('Realized Brier gain',labelpad=8)
    ax=fig.add_subplot(gs[1,0]); ptitle(ax,'c','Action rate by evidence sign')
    tmp=aw.assign(utility_positive=aw.D_lower_bulk>0).groupby(['display_dataset','utility_positive']).act_bulk.mean().unstack().reindex(['BLE RSSI','INSECTS'])
    x=np.arange(len(tmp)); ax.bar(x-.18,tmp.get(False,pd.Series(index=tmp.index,dtype=float)),width=.36,color=GRAY,label=r'$D_L\leq0$'); ax.bar(x+.18,tmp.get(True,pd.Series(index=tmp.index,dtype=float)),width=.36,color=BLUE,label=r'$D_L>0$')
    ax.set_xticks(x,tmp.index); ax.set_ylim(0,1.05); ax.set_ylabel('Fraction of windows repaired'); ax.legend(frameon=False); stylize(ax)
    ax=fig.add_subplot(gs[1,1]); ptitle(ax,'d','Safety occupancy and realized outcome')
    tab=pd.crosstab(np.where(aw.act_bulk,'REPAIR','HOLD'),np.where(aw.realized_gain>0,'positive gain','zero/negative')).reindex(index=['HOLD','REPAIR'],columns=['zero/negative','positive gain'],fill_value=0)
    bottom=np.zeros(2)
    for col,color in [('zero/negative',LIGHT),('positive gain',GREEN)]:
        vals=tab[col].to_numpy(); ax.bar(np.arange(2),vals,bottom=bottom,color=color,label=col)
        for i,v in enumerate(vals):
            if v: ax.text(i,bottom[i]+v/2,str(int(v)),ha='center',va='center',fontsize=13,color=NAVY)
        bottom+=vals
    ax.set_xticks([0,1],['HOLD','REPAIR']); ax.set_ylabel('Deployment windows'); ax.legend(frameon=False,loc='upper left'); stylize(ax)
    save(fig,'fig11_action_phase_portrait')

if __name__=='__main__':
    for fn in [fig02,fig03,fig04,fig05,fig06,fig07,fig08,fig09,fig10,fig11]: fn()
    print(f'Wrote PDFs to {PDF}')
