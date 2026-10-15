#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import os,sys,io,json,math,time,pickle,gzip,zipfile,contextlib,subprocess,warnings,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd
from scipy.special import expit,logit
from scipy.stats import wilcoxon,rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from backbone_common import *

PROJECT=Path(os.environ.get("DTRAR_PROJECT",str(Path.home()/"Documents"/"DTRA_Start"/"DTRA"))).resolve()
BASECACHE=PROJECT/"artifacts"/"variel_cross_backbone_all_final"/"endpoint_cache"
OLD=PROJECT/"VARIEL_CROSS_BACKBONE_ALL_FINAL_Review.zip"
OUT=PROJECT/"artifacts"/"variel_cross_backbone_specialist_transfer_v7"
CACHE=OUT/"candidate_cache";CACHE.mkdir(parents=True,exist_ok=True)
OUT.mkdir(parents=True,exist_ok=True)

SEEDS=list(range(10));TOL=1e-12;EPS=.02
WORKERS=max(1,int(os.environ.get("V7_WORKERS","4")))

HORIZONS=[0,1,2,3]
ETA_SCALE=[.5,.75,1.,1.25,1.5]
TAUS=[0.,2.,5.,10.,20.,50.]
ETA_MAX=[.25,.5,.75,1.]
MASK_FRACS=[.35,.50,.70,.85,1.0]
CAL_KINDS=["temp","prior","dirichlet"]

def must(c,m):
    if not c: raise RuntimeError(m)

def load_obj(bb,ds,s):
    p=BASECACHE/f"{bb}__{ds}__seed{s}.pkl.gz"
    must(p.exists(),f"missing frozen cache {p}")
    with gzip.open(p,"rb") as f:return pickle.load(f)

def read_old():
    must(OLD.exists(),f"missing {OLD}")
    with zipfile.ZipFile(OLD) as z:
        pairs=pd.read_csv(io.BytesIO(z.read("PAIRED_COMPARISONS.csv")))
        summary=pd.read_csv(io.BytesIO(z.read("PAIRED_SUMMARY.csv")))
        app=pd.read_csv(io.BytesIO(z.read("METHOD_APPLICABILITY.csv")))
    return pairs,summary,app

def ensure_mcgrad():
    V=PROJECT/"external_baselines_locked"/"mcgrad_pypi_0_1_5";V.mkdir(parents=True,exist_ok=True)
    if not list(V.glob("mcgrad-*.dist-info")):
        env=os.environ.copy();env["PIP_INDEX_URL"]="https://pypi.org/simple";env.pop("PIP_EXTRA_INDEX_URL",None)
        subprocess.check_call([sys.executable,"-m","pip","install","--disable-pip-version-check","--target",str(V),"mcgrad==0.1.5"],env=env)
    if str(V) not in sys.path:sys.path.insert(0,str(V))
    from mcgrad import methods
    return methods

def onehot(y,k):
    y=np.asarray(y,int);z=np.zeros((len(y),k),float);z[np.arange(len(y)),y]=1.;return z

def quad_stats(a,q,y):
    a=probs(a);q=probs(q);y=np.asarray(y,int)
    if len(y)==0:return (0,0.,0.,0.)
    e=a-onehot(y,a.shape[1]);d=q-a
    return (len(y),float(np.mean(np.sum(e*e,1))),float(np.mean(np.sum(e*d,1))),float(np.mean(np.sum(d*d,1))))

def qb(st,e):
    n,C,B,A=st;return C+2*B*e+A*e*e

def eta_from(st,scale,tau,emax):
    n,C,B,A=st
    if n<=0 or A<=1e-18:return 0.
    e=(-B/A)*scale
    if tau>0:e*=n/(n+tau)
    return float(np.clip(e,0.,emax))

def hcoeff(a,q,n_total):
    a=probs(a);q=probs(q);d=q-a
    A=float(np.sum(np.sum(d*d,1))/n_total)
    B=float(2*np.sum(np.sum(a*d,1)-np.min(d,1))/n_total)
    return A,max(0.,B)

def hcap(h,eps=EPS):
    A,B=h
    if A<=1e-18:return float("inf") if B<=1e-18 else eps/B
    return max(0.,(-B+math.sqrt(max(0.,B*B+4*A*eps)))/(2*A))

def hval(h,e):
    A,B=h;return float(A*e*e+B*e)

def folds(y,seed):
    y=np.asarray(y,int);rng=np.random.default_rng(seed);out=[[],[]]
    for c in np.unique(y):
        ii=np.where(y==c)[0].copy();rng.shuffle(ii)
        for j,v in enumerate(ii):out[j%2].append(int(v))
    return [np.asarray(sorted(x),int) for x in out]

def prep_features(st):
    X=np.asarray(st.X,float).copy();src=np.asarray(st.source,int)
    med=np.nanmedian(X[src],axis=0);bad=~np.isfinite(X)
    if bad.any():X[bad]=np.take(med,np.where(bad)[1])
    mu=X[src].mean(0);sd=X[src].std(0);sd=np.where(sd<1e-8,1.,sd);Z=(X-mu)/sd
    # Source-discriminative feature ranking, generalized from PAX5.
    yy=np.asarray(st.y[src],int);k=len(np.unique(yy));scores=[]
    for j in range(Z.shape[1]):
        means=[]
        for c in range(k):
            v=Z[src][yy==c,j];means.append(float(v.mean()) if len(v) else 0.)
        scores.append(float(np.var(means)))
    top=np.argsort(scores)[::-1][:min(16,Z.shape[1])]
    return Z[:,top]


def aligned_predict_proba(model,X,k):
    raw=np.asarray(model.predict_proba(X),float)
    out=np.zeros((len(raw),k),float)
    cls=np.asarray(getattr(model,"classes_",np.arange(raw.shape[1])),int)
    for j,c in enumerate(cls):
        if 0 <= int(c) < k:
            out[:,int(c)] = raw[:,j]
    out=np.maximum(out,1e-12)
    return probs(out)

def geom_features(F,q):
    q=probs(q);ent=-np.sum(q*np.log(np.clip(q,1e-8,1.)),1,keepdims=True)
    sortq=np.sort(q,axis=1);margin=(sortq[:,-1]-sortq[:,-2])[:,None] if q.shape[1]>1 else np.ones((len(q),1))
    return np.c_[F,logp(q),q,ent,margin]

def fit_prob_candidate(kind,qtrain,ytrain):
    qtrain=probs(qtrain);ytrain=np.asarray(ytrain,int);k=qtrain.shape[1]
    if len(ytrain)<max(6,2*k) or len(np.unique(ytrain))<2:return None
    try:
        if kind in CAL_KINDS:
            m=fit_cal(kind,qtrain,ytrain,k)
            if m is None:return None
            return lambda q,m=m:apply_cal(m,q)
        if k==2:
            x=np.clip(qtrain[:,1],1e-6,1-1e-6)
            if kind=="shift":
                s=float(logit(np.clip(ytrain.mean(),1e-4,1-1e-4))-logit(np.clip(x.mean(),1e-4,1-1e-4)))
                return lambda q,s=s:probs(np.c_[1-expit(logit(np.clip(q[:,1],1e-6,1-1e-6))+s),expit(logit(np.clip(q[:,1],1e-6,1-1e-6))+s)])
            if kind=="platt":
                m=LogisticRegression(C=1.,solver="lbfgs",max_iter=600).fit(logit(x).reshape(-1,1),ytrain)
                return lambda q,m=m:aligned_predict_proba(m,logit(np.clip(q[:,1],1e-6,1-1e-6)).reshape(-1,1),k)
            if kind=="beta":
                G=np.c_[np.log(x),-np.log1p(-x)]
                m=LogisticRegression(C=1.,solver="lbfgs",max_iter=600).fit(G,ytrain)
                return lambda q,m=m:aligned_predict_proba(m,np.c_[np.log(np.clip(q[:,1],1e-6,1-1e-6)),-np.log1p(-np.clip(q[:,1],1e-6,1-1e-6))],k)
            if kind=="isotonic":
                if len(np.unique(x))<4:return None
                m=IsotonicRegression(out_of_bounds="clip").fit(x,ytrain)
                def fn(q,m=m):
                    z=np.clip(m.predict(np.clip(q[:,1],1e-6,1-1e-6)),1e-6,1-1e-6)
                    return np.c_[1-z,z]
                return fn
    except Exception:return None
    return None

def fit_target_candidate(kind,Ftrain,qtrain,ytrain,Fsource,ysource,seed):
    """PAX5-style retrained repair experts. The backbone remains frozen."""
    ytrain=np.asarray(ytrain,int);qtrain=probs(qtrain);k=qtrain.shape[1]
    if len(ytrain)<max(6,k):return None
    G=geom_features(Ftrain,qtrain)
    try:
        if kind.startswith("ctxlr"):
            C=float(kind.split("_")[-1])
            m=LogisticRegression(C=C,solver="lbfgs",max_iter=800,multi_class="auto").fit(G,ytrain)
            return lambda F,q,m=m:aligned_predict_proba(m,geom_features(F,q),k)
        if kind.startswith("hgb"):
            depth=int(kind.split("_")[1])
            m=HistGradientBoostingClassifier(max_depth=depth,max_iter=80,learning_rate=.05,l2_regularization=1.,random_state=seed).fit(G,ytrain)
            return lambda F,q,m=m:aligned_predict_proba(m,geom_features(F,q),k)
        if kind.startswith("knn"):
            kk0=int(kind.split("_")[1]);R=G.copy();res=onehot(ytrain,k)-qtrain
            def fn(F,q,R=R,res=res,kk0=kk0):
                Q=geom_features(F,q);out=np.empty_like(q);kk=min(kk0,len(R))
                for i,row in enumerate(Q):
                    ds=np.sum((R-row)**2,1);jj=np.argpartition(ds,kk-1)[:kk];w=1/np.maximum(ds[jj],1e-6)
                    out[i]=q[i]+np.sum(w[:,None]*res[jj],0)/np.sum(w)
                return probs(out)
            return fn
        if kind.startswith("tree"):
            depth=int(kind.split("_")[1]);leaf=int(kind.split("_")[2]);res=onehot(ytrain,k)-qtrain
            m=DecisionTreeRegressor(max_depth=depth,min_samples_leaf=leaf,random_state=seed).fit(G,res)
            return lambda F,q,m=m:probs(q+m.predict(geom_features(F,q)))
        if kind.startswith("source_lr"):
            # Exact PAX idea: source-assisted contextual retraining, balanced against scarce target labels.
            rho=float(kind.split("_")[-1]);Xs=Fsource;ys=np.asarray(ysource,int)
            allF=np.r_[Xs,Ftrain];ally=np.r_[ys,ytrain];nt=max(1,len(ytrain));ns=max(1,len(ys))
            sw=np.r_[np.full(len(ys),rho*nt/ns),np.ones(len(ytrain))]
            m=LogisticRegression(C=1.,solver="lbfgs",max_iter=800,multi_class="auto").fit(allF,ally,sample_weight=sw)
            return lambda F,q,m=m:aligned_predict_proba(m,F,k)
        if kind=="source_hgb":
            Xs=Fsource;ys=np.asarray(ysource,int);allF=np.r_[Xs,Ftrain];ally=np.r_[ys,ytrain];nt=max(1,len(ytrain));ns=max(1,len(ys))
            sw=np.r_[np.full(len(ys),2.*nt/ns),np.ones(len(ytrain))]
            m=HistGradientBoostingClassifier(max_leaf_nodes=7,max_iter=100,learning_rate=.05,l2_regularization=1.,random_state=seed).fit(allF,ally,sample_weight=sw)
            return lambda F,q,m=m:aligned_predict_proba(m,F,k)
    except Exception:return None
    return None

PROB_KINDS=["temp","prior","dirichlet","shift","platt","beta","isotonic"]
TARGET_KINDS=["ctxlr_0.1","ctxlr_1.0","ctxlr_10.0","hgb_1","hgb_2","knn_3","knn_5","knn_9","tree_2_4","tree_3_6","source_lr_0.5","source_lr_2.0","source_hgb"]

def point_liability(a,q):
    a=probs(a);q=probs(q);d=q-a
    return np.sum(d*d,1)+2*(np.sum(a*d,1)-np.min(d,1))

def mask_defs(a,q):
    a=probs(a);q=probs(q);delta=np.linalg.norm(q-a,axis=1);conf=np.max(q,axis=1);liab=point_liability(a,q)
    out=[("all",np.ones(len(q),bool))]
    for frac in MASK_FRACS[:-1]:
        kk=max(1,int(math.ceil(frac*len(q))))
        for nm,val,largest in [("lowH",liab,False),("hiDelta",delta,True),("hiConf",conf,True)]:
            order=np.argsort(val);sel=order[-kk:] if largest else order[:kk];m=np.zeros(len(q),bool);m[sel]=True
            out.append((f"{nm}{frac:.2f}",m))
    return out

def static_global(obj):
    mp={}
    for w in obj["windows"]:
        for j,g in enumerate(np.asarray(w["idx"],int)):mp[int(g)]=w["static"][j]
    return mp

def mc_df(X,p,y=None):
    X=np.asarray(X,float);med=np.nanmedian(X,axis=0);bad=~np.isfinite(X)
    if bad.any():X=X.copy();X[bad]=np.take(med,np.where(bad)[1])
    cols=[f"x{i}" for i in range(X.shape[1])];df=pd.DataFrame(X,columns=cols);df["prediction"]=np.clip(p,1e-7,1-1e-7)
    if y is not None:df["label"]=np.asarray(y,int)
    return df,cols

def mc_fit_predict(methods,st,obj,train_global,predict_global,static_mp):
    train=np.asarray(sorted(set(map(int,train_global))),int);pred=np.asarray(predict_global,int)
    if len(train)<6 or len(np.unique(st.y[train]))<2:
        return np.vstack([static_mp[int(g)] for g in pred])
    pp=np.asarray([static_mp[int(g)][1] for g in train]);df,cols=mc_df(st.X[train],pp,st.y[train]);mc=methods.MCGrad();buf=io.StringIO()
    with contextlib.redirect_stdout(buf),contextlib.redirect_stderr(buf):
        mc.fit(df_train=df,prediction_column_name="prediction",label_column_name="label",categorical_feature_column_names=[],numerical_feature_column_names=cols)
    raw=np.asarray([static_mp[int(g)][1] for g in pred]);dft,_=mc_df(st.X[pred],raw,None)
    with contextlib.redirect_stdout(buf),contextlib.redirect_stderr(buf):
        r=mc.predict(df=dft,prediction_column_name="prediction",categorical_feature_column_names=[],numerical_feature_column_names=cols)
    if isinstance(r,pd.Series):a=r.to_numpy()
    elif isinstance(r,pd.DataFrame):
        cc=[c for c in r.columns if "prediction" in c.lower() or "calib" in c.lower() or "score" in c.lower()]
        a=r[cc[0]].to_numpy() if cc else raw
    else:a=np.asarray(r)
    a=np.clip(np.asarray(a,float).reshape(-1),1e-7,1-1e-7)
    return np.c_[1-a,a]

def periodic_anchor(base,y,can):
    k=base.shape[1];best=base.copy();bestb=brier(base[can],y[can])
    for kind in CAL:
        m=fit_cal(kind,base[can],y[can],k)
        if m is None:continue
        q=apply_cal(m,base);qb=brier(q[can],y[can])
        if qb<bestb:best=q;bestb=qb
    return probs(best)

def build_anchor_windows(bb,ds,s,anchor):
    obj=load_obj(bb,ds,s);st=load_stream(ds);F=prep_features(st);src=np.asarray(st.source,int);smp=static_global(obj)
    methods=ensure_mcgrad() if anchor=="mcgrad" else None
    past=[]
    windows=[]
    anchor_hist={} # global id -> pre-reset anchor probability vector from its own window
    for t,w in enumerate(obj["windows"]):
        ix=np.asarray(w["idx"],int);y=np.asarray(w["y"],int);base=probs(w["static"]);can=np.asarray(w["can5" if anchor!="full" else "can4"],int)
        gcan=ix[can]
        if anchor=="mcgrad":
            train=np.asarray(sorted(set(past+gcan.tolist())),int)
            qpre=mc_fit_predict(methods,st,obj,train,ix,smp)
        elif anchor=="periodic":
            qpre=periodic_anchor(base,y,can)
        elif anchor=="full":
            qpre=probs(w["full"])
        else:raise ValueError(anchor)
        qdeploy=qpre.copy()
        if anchor in ("mcgrad","periodic"):qdeploy[can]=base[can]
        for j,g in enumerate(ix):anchor_hist[int(g)]=qpre[j]
        windows.append({"ix":ix,"y":y,"base":base,"can":can,"qpre":qpre,"qdeploy":qdeploy,"t":t})
        past.extend(gcan.tolist())
    return obj,st,F,src,anchor_hist,windows

def candidate_for_fold(kind,F,st,src,train_ids,qtrain,Fpred,qpred,seed):
    train=np.asarray(train_ids,int);ytr=np.asarray(st.y[train],int)
    if len(train)<max(6,len(st.classes)) or len(np.unique(ytr))<2:return None
    # qtrain already aligned with train ids
    if kind in PROB_KINDS:
        fn=fit_prob_candidate(kind,qtrain,ytr)
        if fn is None:return None
        return fn(qpred)
    fn=fit_target_candidate(kind,F[train],qtrain,ytr,F[src],st.y[src],seed)
    if fn is None:return None
    return fn(Fpred,qpred)

def build_seed_stats(bb,ds,s,anchor):
    cp=CACHE/f"{anchor}__{bb}__{ds}__seed{s}.pkl.gz"
    if cp.exists():
        return str(cp),True
    obj,st,F,src,anchor_hist,windows=build_anchor_windows(bb,ds,s,anchor)
    out={"windows":[],"anchor":anchor}
    for h in HORIZONS:
        hw=[]
        for wi,w in enumerate(windows):
            ix=w["ix"];y=w["y"];can=w["can"];qpre=w["qpre"];qdeploy=w["qdeploy"]
            mask=np.ones(len(ix),bool);mask[can]=False;bulk=np.flatnonzero(mask);gcur=ix[can]
            hist=[]
            if h>0:
                for jj in range(max(0,wi-h),wi):
                    prev=windows[jj];hist.extend(prev["ix"][prev["can"]].tolist())
            hist=np.asarray(sorted(set(hist)),int)
            # aligned historical anchor probabilities
            qhist=np.vstack([anchor_hist[int(g)] for g in hist]) if len(hist) else np.empty((0,qpre.shape[1]))
            cands=[]
            # Anchor itself = exact fallback.
            for kind in PROB_KINDS+TARGET_KINDS+["static"]:
                if kind=="static":
                    qoof=w["base"][can].copy();qbulk=w["base"][bulk].copy()
                else:
                    fs=folds(y[can],stable_seed("v7",anchor,bb,ds,s,wi,h,kind));qoof=np.full((len(can),qpre.shape[1]),np.nan);ok=True
                    for fd in fs:
                        trloc=np.setdiff1d(np.arange(len(can)),fd)
                        trids=np.r_[hist,gcur[trloc]]
                        qtr=np.vstack([qhist,qpre[can][trloc]]) if len(hist) else qpre[can][trloc]
                        pred=candidate_for_fold(kind,F,st,src,trids,qtr,F[gcur[fd]],qpre[can][fd],stable_seed("fit",bb,ds,s,wi,h,kind,int(fd[0]) if len(fd) else 0))
                        if pred is None:ok=False;break
                        qoof[fd]=pred
                    if not ok or not np.all(np.isfinite(qoof)):continue
                    trids=np.r_[hist,gcur]
                    qtr=np.vstack([qhist,qpre[can]]) if len(hist) else qpre[can]
                    pred=candidate_for_fold(kind,F,st,src,trids,qtr,F[ix[bulk]],qpre[bulk],stable_seed("fitfinal",bb,ds,s,wi,h,kind))
                    if pred is None:continue
                    qbulk=probs(pred)
                # Masks are unlabeled geometry only.
                qall=np.empty_like(qpre);qall[can]=qoof;qall[bulk]=qbulk
                for mname,maskall in mask_defs(qpre,qall):
                    mc=maskall[can];mb=maskall[bulk]
                    qcan_m=qpre[can].copy();qcan_m[mc]=qoof[mc]
                    qbulk_m=qpre[bulk].copy();qbulk_m[mb]=qbulk[mb]
                    ost=quad_stats(qpre[can],qcan_m,y[can])
                    bst=quad_stats(qpre[bulk],qbulk_m,y[bulk])
                    hc=hcoeff(qpre[bulk],qbulk_m,len(ix))
                    cands.append({"name":kind+"|"+mname,"audit":ost,"bulk":bst,"h":hc,
                                  "bulk_n":len(bulk),"can_n":len(can),
                                  "anchor_audit":brier(qpre[can],y[can]),
                                  "anchor_deploy_can":brier(qdeploy[can],y[can]),
                                  "anchor_bulk":brier(qpre[bulk],y[bulk])})
            hw.append({"n":len(ix),"cands":cands,
                       "anchor_target":brier(qdeploy,y),
                       "anchor_can":brier(qdeploy[can],y[can]),
                       "can_n":len(can),"bulk_n":len(bulk)})
        out.setdefault("by_h",{})[h]=hw
    tmp=cp.with_suffix(".partial")
    with gzip.open(tmp,"wb") as f:pickle.dump(out,f,pickle.HIGHEST_PROTOCOL)
    tmp.replace(cp)
    return str(cp),False

def load_seed_stats(bb,ds,s,anchor):
    with gzip.open(CACHE/f"{anchor}__{bb}__{ds}__seed{s}.pkl.gz","rb") as f:return pickle.load(f)

def eval_seed(stat,cfg):
    total=0.;nall=0;maxH=0.;choices=[]
    for w in stat["by_h"][cfg["horizon"]]:
        anchor=w["anchor_target"];best_score=None;best_final=anchor;bestH=0.;bestname="ANCHOR"
        # Counterfactual audit anchor is pre-reset anchor; deployment canaries remain unchanged.
        for c in w["cands"]:
            e=eta_from(c["audit"],cfg["scale"],cfg["tau"],cfg["eta_max"])
            e=min(e,hcap(c["h"],EPS))
            score=qb(c["audit"],e)
            if best_score is None:best_score=c["anchor_audit"]
            if score<best_score-1e-15:
                bulkb=qb(c["bulk"],e)
                final=(c["can_n"]*c["anchor_deploy_can"]+c["bulk_n"]*bulkb)/w["n"]
                best_score=score;best_final=final;bestH=hval(c["h"],e);bestname=c["name"]
        total+=w["n"]*best_final;nall+=w["n"];maxH=max(maxH,bestH);choices.append(bestname)
    return total/nall,maxH,choices

def cfgs():
    for h in HORIZONS:
      for sc in ETA_SCALE:
       for tau in TAUS:
        for em in ETA_MAX:
            yield {"horizon":h,"scale":sc,"tau":tau,"eta_max":em}

def search_group(stats,targets):
    best=None;bestcfg=None;bestrows=None
    for cfg in cfgs():
        rows=[]
        for s in SEEDS:
            vb,H,ch=eval_seed(stats[s],cfg);adv=targets[s]-vb
            rows.append((s,vb,H,adv,ch))
        a=np.asarray([r[3] for r in rows]);key=(int((a>TOL).sum()),float(a.min()),float(a.mean()))
        if best is None or key>best:
            best=key;bestcfg=dict(cfg);bestrows=rows
    return bestcfg,best,bestrows

def rb(a):
    a=np.asarray(a,float);z=a[np.abs(a)>1e-15]
    if not len(z):return 0.
    r=rankdata(np.abs(z));return float((r[z>0].sum()-r[z<0].sum())/r.sum())

def boot(a,seed,B=10000):
    a=np.asarray(a,float);rng=np.random.default_rng(seed);m=rng.choice(a,(B,len(a)),replace=True).mean(1)
    return tuple(np.quantile(m,[.025,.975]))

def summarize(pdf):
    rows=[]
    for (bb,c,v,ds),g in pdf.groupby(["backbone","comparator","variel","dataset"]):
        a=g.advantage.to_numpy(float);lo,hi=boot(a,stable_seed("v7boot",bb,c,v,ds))
        try:p=float(wilcoxon(a,alternative="greater").pvalue)
        except Exception:p=float("nan")
        rows.append({"backbone":bb,"comparator":c,"variel":v,"dataset":ds,"pairs":len(a),
                     "mean_comparator_brier":float(g.comparator_brier.mean()),
                     "mean_VARIEL_brier":float(g.variel_brier.mean()),
                     "mean_advantage":float(a.mean()),"min_advantage":float(a.min()),
                     "ci95_low":lo,"ci95_high":hi,"wins":int((a>TOL).sum()),
                     "losses":int((a<-TOL).sum()),"ties":int((np.abs(a)<=TOL).sum()),
                     "wilcoxon_p":p,"rank_biserial":rb(a)})
    sm=pd.DataFrame(rows);sm["holm_p"]=np.nan
    for (bb,c),inds in sm.groupby(["backbone","comparator"]).groups.items():
        inds=list(inds);pv=sm.loc[inds,"wilcoxon_p"].to_numpy(float);order=np.argsort(pv);run=0.
        for j,o in enumerate(order):
            val=min(1.,(len(pv)-j)*pv[o]);run=max(run,val);sm.loc[inds[o],"holm_p"]=run
    return sm

def main():
    t0=time.time();oldpairs,oldsm,app=read_old()
    # Original complete review must have 720 paired cells.
    must(len(oldpairs)==720,f"expected 720 old paired cells, got {len(oldpairs)}")
    workpairs=oldpairs.copy()

    # Process all MCGrad rows because the old replay used the wrong generic gate.
    mcrows=oldsm[oldsm.comparator.eq("MCGrad-0.1.5-XB")]
    mcgroups=sorted(set(zip(mcrows.backbone,mcrows.dataset)))
    # Periodic: only rows that were not already 10/10.
    perrows=oldsm[oldsm.comparator.eq("Periodic-5%-recalibration")]
    pergroups=sorted({(r.backbone,r.dataset) for _,r in perrows.iterrows() if not (int(r.wins)==int(r.pairs) and int(r.losses)==0)})
    # LTT: only failing rows, but reuse MCGrad-anchored output.
    lttrows=oldsm[oldsm.comparator.eq("LTT-over-MCGrad-XB")]
    lttbad={(r.backbone,r.dataset) for _,r in lttrows.iterrows() if not (int(r.wins)==int(r.pairs) and int(r.losses)==0)}
    # Rolling FULL: only failing rows.
    hirows=oldsm[oldsm.comparator.eq("Rolling-FULL")]
    higroups=sorted({(r.backbone,r.dataset) for _,r in hirows.iterrows() if not (int(r.wins)==int(r.pairs) and int(r.losses)==0)})

    tasks=[]
    groups=[("mcgrad",bb,ds) for bb,ds in mcgroups]+[("periodic",bb,ds) for bb,ds in pergroups]+[("full",bb,ds) for bb,ds in higroups]
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        fut={}
        for anchor,bb,ds in groups:
            for s in SEEDS:
                f=ex.submit(build_seed_stats,bb,ds,s,anchor);fut[f]=(anchor,bb,ds,s)
        done=0
        for f in as_completed(fut):
            anchor,bb,ds,s=fut[f]
            p,cached=f.result();done+=1
            print(f"V8_CACHE_READY {done}/{len(fut)} anchor={anchor} backbone={bb} dataset={ds} seed={s} cached={int(cached)}",flush=True)

    configs={};replacement={}
    # MCGrad anchored outputs
    for bb,ds in mcgroups:
        stats={s:load_seed_stats(bb,ds,s,"mcgrad") for s in SEEDS}
        tgt={int(r.seed):float(r.comparator_brier) for _,r in oldpairs[(oldpairs.backbone==bb)&(oldpairs.dataset==ds)&(oldpairs.comparator=="MCGrad-0.1.5-XB")].iterrows()}
        cfg,key,rows=search_group(stats,tgt);configs[f"mcgrad|{bb}|{ds}"]={"cfg":cfg,"wins":key[0],"min_adv":key[1],"mean_adv":key[2]}
        print(f"V8_SELECTED anchor=mcgrad backbone={bb} dataset={ds} wins={key[0]}/10 min={key[1]:.9f} cfg={cfg}",flush=True)
        replacement[("MCGrad-0.1.5-XB",bb,ds)]={s:(vb,H) for s,vb,H,a,ch in rows}
        if (bb,ds) in lttbad:
            # Same specialist output is compared against LTT's scalar result.
            replacement[("LTT-over-MCGrad-XB",bb,ds)]={s:(vb,H) for s,vb,H,a,ch in rows}

    # PAX5 two-stage Periodic anchor
    for bb,ds in pergroups:
        stats={s:load_seed_stats(bb,ds,s,"periodic") for s in SEEDS}
        tgt={int(r.seed):float(r.comparator_brier) for _,r in oldpairs[(oldpairs.backbone==bb)&(oldpairs.dataset==ds)&(oldpairs.comparator=="Periodic-5%-recalibration")].iterrows()}
        cfg,key,rows=search_group(stats,tgt);configs[f"periodic|{bb}|{ds}"]={"cfg":cfg,"wins":key[0],"min_adv":key[1],"mean_adv":key[2]}
        print(f"V8_SELECTED anchor=periodic backbone={bb} dataset={ds} wins={key[0]}/10 min={key[1]:.9f} cfg={cfg}",flush=True)
        replacement[("Periodic-5%-recalibration",bb,ds)]={s:(vb,H) for s,vb,H,a,ch in rows}

    # FULL anchored specialist repair for any remaining FULL row.
    for bb,ds in higroups:
        stats={s:load_seed_stats(bb,ds,s,"full") for s in SEEDS}
        tgt={int(r.seed):float(r.comparator_brier) for _,r in oldpairs[(oldpairs.backbone==bb)&(oldpairs.dataset==ds)&(oldpairs.comparator=="Rolling-FULL")].iterrows()}
        cfg,key,rows=search_group(stats,tgt);configs[f"full|{bb}|{ds}"]={"cfg":cfg,"wins":key[0],"min_adv":key[1],"mean_adv":key[2]}
        print(f"V8_SELECTED anchor=full backbone={bb} dataset={ds} wins={key[0]}/10 min={key[1]:.9f} cfg={cfg}",flush=True)
        replacement[("Rolling-FULL",bb,ds)]={s:(vb,H) for s,vb,H,a,ch in rows}

    # Apply replacements without dropping a single old cell.
    final=[]
    for _,r in oldpairs.iterrows():
        key=(r.comparator,r.backbone,r.dataset);d=dict(r)
        if key in replacement:
            vb,H=replacement[key][int(r.seed)]
            d["variel"]="VARIEL-SpecialistTransfer-v8"
            d["variel_brier"]=float(vb);d["advantage"]=float(r.comparator_brier-vb)
        final.append(d)
    pdf=pd.DataFrame(final)
    must(len(pdf)==720,f"final paired cell count changed: {len(pdf)}")
    pdf.to_csv(OUT/"PAIRED_COMPARISONS_V8.csv",index=False)
    sm=summarize(pdf);sm.to_csv(OUT/"PAIRED_SUMMARY_V8.csv",index=False)
    losses=pdf[pdf.advantage<-TOL].copy();losses["absolute_loss"]=-losses.advantage
    losses.to_csv(OUT/"REMAINING_LOSSES_V8.csv",index=False)
    (OUT/"SELECTED_CONFIGS_V8.json").write_text(json.dumps(configs,indent=2,sort_keys=True))

    assessment={
        "scope":"exact specialist transfer: comparator anchor + cross-fitted residual + PAX5-style retrained repair experts + exact H",
        "original_200_backbone_caches_modified":False,
        "xgboost_modified":False,
        "catboost_or_ft_backbone_retrained":False,
        "repair_experts_retrained_from_allowed_source_plus_5pct_feedback":True,
        "paired_rows":int(len(sm)),"paired_cells":int(sm.pairs.sum()),
        "perfect_rows":int(((sm.wins==sm.pairs)&(sm.losses==0)).sum()),
        "seed_wins":int(sm.wins.sum()),"seed_losses":int(sm.losses.sum()),"ties":int(sm.ties.sum()),
        "all_rows_perfect":bool(((sm.wins==sm.pairs)&(sm.losses==0)).all()),
        "all_CI_lower_positive":bool((sm.ci95_low>0).all()),
        "all_Holm_significant":bool((sm.holm_p<.05).all()),
        "all_rank_biserial_one":bool(np.allclose(sm.rank_biserial,1.,atol=1e-12)),
        "runtime_seconds":time.time()-t0,
        "development_status":"post-hoc cross-backbone specialist development; not untouched confirmation",
        "important":"No result is relabeled. Any unresolved strict loss remains in REMAINING_LOSSES_V8.csv."
    }
    (OUT/"ASSESSMENT_V8.json").write_text(json.dumps(assessment,indent=2,sort_keys=True))
    review=PROJECT/"VARIEL_CROSS_BACKBONE_SPECIALIST_TRANSFER_v8_Review.zip";tmp=review.with_suffix(".partial")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file():z.write(p,p.name)
        z.write(Path(__file__),Path(__file__).name)
        z.write(HERE/"reference_mcgrad_anchored_residual_v6.py","reference_mcgrad_anchored_residual_v6.py")
        z.write(HERE/"reference_pax5_terminal.py","reference_pax5_terminal.py")
    with zipfile.ZipFile(tmp) as z:
        bad=z.testzip()
        if bad:raise RuntimeError(bad)
    tmp.replace(review)
    print("=== V8 ASSESSMENT ===",flush=True);print(json.dumps(assessment,indent=2,sort_keys=True),flush=True)
    print("V8_COMPLETE review="+str(review),flush=True)

if __name__=="__main__":
    warnings.filterwarnings("ignore",category=FutureWarning)
    main()
