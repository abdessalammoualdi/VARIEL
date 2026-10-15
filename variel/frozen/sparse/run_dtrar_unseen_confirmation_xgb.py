#!/usr/bin/env python3
"""Frozen RAPID risk-optimal confirmation on the two pre-performance locked unseen streams.

Runs ONLY the already-locked primary unseen datasets from
DTRAR_UnseenDataset_Audit_V2_Review:
  * occupancy_detection_v2
  * eeg_eye_state

No dataset-specific RAPID tuning is performed.  The base learner is XGBoost
with the same primary source-only training hyperparameters as the main project.
The already-validated run_dtrar_risk_optimal_final.py supplies the complete
RAPID mathematics, audit design, delayed-label protocol, PLHE, old-method
ablations, and fixed prospective safety frontier.

This script deliberately creates a compact review package: full per-job
probabilities remain in artifacts/ on the workstation; the upload contains the
summary/statistical/safety evidence plus provenance.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import fcntl
import hashlib
import io
import json
import math
import os
import platform
import shutil
import sys
import zipfile

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parent
BASE = PROJECT / "offset_check_v07"
AUDIT_JSON = PROJECT / "artifacts" / "unseen_dataset_audit_v2" / "UNSEEN_DATASET_AUDIT_V2.json"
RISKOPT = PROJECT / "run_dtrar_risk_optimal_final.py"
DATASETS = ["occupancy_detection_v2", "eeg_eye_state"]
SEEDS = list(range(10))
MODEL = "xgboost"


def must(cond, msg):
    if not cond:
        raise SystemExit("ERROR: " + msg)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    classes: list
    categorical: list
    train: np.ndarray
    source: np.ndarray
    windows: list
    metadata: dict

    @property
    def target(self):
        return np.concatenate(self.windows)


def load_manifest():
    must(AUDIT_JSON.exists(), f"missing locked audit manifest: {AUDIT_JSON}")
    manifest = json.loads(AUDIT_JSON.read_text())
    must(manifest.get("performance_seen") is False, "audit manifest says performance was already seen")
    must(manifest.get("selection_locked_before_performance") is True, "unseen selection was not locked before performance")
    must(manifest.get("primary_unseen_confirmation_datasets") == DATASETS,
         "primary unseen dataset lock differs from expected pair")
    return manifest


MANIFEST = None
_DATA_CACHE = {}


def dataset_record(name):
    return MANIFEST["datasets"][name]


def locked_path_and_hash(name):
    rec = dataset_record(name)
    if name == "occupancy_detection_v2":
        rel = rec["v2_locked_csv"]
        expected = rec["v2_locked_sha256"]
    elif name == "eeg_eye_state":
        rel = rec["locked_csv"]
        expected = rec["locked_sha256"]
    else:
        raise ValueError(name)
    p = PROJECT / rel
    must(p.exists(), f"locked CSV missing: {p}")
    actual = sha256(p)
    must(actual == expected, f"locked CSV checksum mismatch for {name}: {actual} != {expected}")
    return p, expected


def load_unseen(name, *_args, **_kwargs):
    if name in _DATA_CACHE:
        return _DATA_CACHE[name]
    p, expected = locked_path_and_hash(name)
    df = pd.read_csv(p)
    must("__target__" in df.columns, f"{name}: __target__ missing")
    yraw = df.pop("__target__")
    if "__sequence__" in df.columns:
        df = df.drop(columns=["__sequence__"])
    # Timestamp/order columns were already excluded by the blind audit lock.
    must(len(df) == dataset_record(name)["n"], f"{name}: row count changed after lock")
    must(not yraw.isna().any(), f"{name}: missing target after lock")
    labels = sorted(pd.unique(yraw).tolist(), key=lambda x: str(x))
    mapping = {v: i for i, v in enumerate(labels)}
    y = yraw.map(mapping).to_numpy(dtype=np.int64)
    # Primary locked streams are numeric; refusing silent object encoding keeps
    # the confirmation protocol identical and auditable.
    obj = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    must(not obj, f"{name}: unexpected categorical/object predictors after blind lock: {obj}")
    X = df.apply(pd.to_numeric, errors="raise")
    must(np.isfinite(X.to_numpy(float)).all(), f"{name}: nonfinite predictor values")
    n = len(y)
    a, b = math.floor(0.50 * n), math.floor(0.65 * n)
    data = Dataset(
        name=name,
        X=X,
        y=y,
        classes=[str(v) for v in labels],
        categorical=[],
        train=np.arange(a),
        source=np.arange(a, b),
        windows=[np.asarray(w, int) for w in np.array_split(np.arange(b, n), 12)],
        metadata={
            "unseen_confirmation": True,
            "locked_csv": str(p.relative_to(PROJECT)),
            "locked_sha256": expected,
            "ordering": dataset_record(name).get("ordering", dataset_record(name).get("correction", "locked chronological order")),
        },
    )
    # Recheck the locked support without changing it.
    src_classes = set(np.unique(y[np.concatenate([data.train, data.source])]).tolist())
    tar_classes = set(np.unique(y[data.target]).tolist())
    must(len(src_classes) >= 2 and tar_classes.issubset(src_classes) and len(tar_classes) >= 2,
         f"{name}: class-support lock no longer holds")
    _DATA_CACHE[name] = data
    return data


def primary_config():
    p = BASE / "configs" / "primary.json"
    must(p.exists(), f"missing primary config: {p}")
    return json.loads(p.read_text())


def train_or_reuse_xgb(data, seed):
    from xgboost import XGBClassifier
    cfg = primary_config()
    p, dhash = locked_path_and_hash(data.name)
    signature_obj = {
        "dataset": data.name,
        "locked_sha256": dhash,
        "seed": int(seed),
        "family": "xgboost",
        "config": {k: cfg[k] for k in cfg if k.startswith("tree_")},
        "protocol": "unseen-confirmation-v1",
    }
    sig = hashlib.sha256(json.dumps(signature_obj, sort_keys=True).encode()).hexdigest()[:12]
    cache = PROJECT / "artifacts" / "unseen_riskopt_xgb" / "models" / data.name / f"seed_{seed}_{sig}"
    cache.mkdir(parents=True, exist_ok=True)
    predfile = cache / "frozen_probabilities.npz"
    complete = cache / "COMPLETE.json"
    if predfile.exists() and complete.exists():
        saved = np.load(predfile)
        raw = saved["probabilities"]
        must(raw.shape == (len(data.y), len(data.classes)), f"{data.name}/seed{seed}: cached shape mismatch")
        must(np.array_equal(saved["row_id"], np.arange(len(data.y))), f"{data.name}/seed{seed}: cached row IDs mismatch")
        print(f"UNSEEN_CACHE_REUSED dataset={data.name} seed={seed} cache={cache.name}", flush=True)
        return cache, raw

    # Exact numeric branch of the project's primary XGBoost training protocol.
    boundary = math.floor(0.8 * len(data.train))
    inner, valid = data.train[:boundary], data.train[boundary:]
    must(len(np.unique(data.y[inner])) == len(data.classes), f"{data.name}: development training lacks a class")
    imputer = SimpleImputer(strategy="median").fit(data.X.iloc[inner])
    Xin = imputer.transform(data.X.iloc[inner]).astype(np.float32)
    Xva = imputer.transform(data.X.iloc[valid]).astype(np.float32)
    k = len(data.classes)
    args = dict(
        n_estimators=cfg["tree_rounds"], max_depth=cfg["tree_depth"],
        learning_rate=cfg["tree_learning_rate"], subsample=0.9,
        colsample_bytree=0.9, min_child_weight=2, reg_lambda=1.0,
        tree_method="hist", device="cpu", n_jobs=1, random_state=seed,
        objective="binary:logistic" if k == 2 else "multi:softprob",
        eval_metric="logloss" if k == 2 else "mlogloss",
    )
    print(f"UNSEEN_TRAIN_BEGIN dataset={data.name} model=xgboost seed={seed}", flush=True)
    selection = XGBClassifier(**args, early_stopping_rounds=cfg["tree_early_stopping"])
    selection.fit(Xin, data.y[inner], eval_set=[(Xva, data.y[valid])], verbose=False)
    rounds = max(1, int(selection.best_iteration) + 1)

    full_imputer = SimpleImputer(strategy="median").fit(data.X.iloc[data.train])
    Xall = full_imputer.transform(data.X).astype(np.float32)
    args["n_estimators"] = rounds
    model = XGBClassifier(**args)
    model.fit(Xall[data.train], data.y[data.train], verbose=False)
    raw = np.asarray(model.predict_proba(Xall), float)
    must(raw.shape == (len(data.y), k), f"{data.name}: prediction shape mismatch")
    model.save_model(cache / "frozen.ubj")
    joblib.dump(full_imputer, cache / "preprocessor.joblib")
    tmp = cache / "frozen_probabilities.tmp.npz"
    np.savez_compressed(tmp, probabilities=raw, row_id=np.arange(len(data.y)))
    tmp.replace(predfile)
    complete.write_text(json.dumps(signature_obj, indent=2, sort_keys=True))
    print(f"UNSEEN_TRAIN_COMPLETE dataset={data.name} model=xgboost seed={seed} rounds={rounds}", flush=True)
    return cache, raw


def source_hashes_unseen(_root, name):
    p, h = locked_path_and_hash(name)
    return {p.name: h}


def frozen_cache_unseen(_artifact_root, data, family, seed, profile, hashes):
    must(family == "xgboost", f"unseen confirmation expected xgboost, got {family}")
    must(profile == "primary", f"unseen confirmation expected primary profile, got {profile}")
    return train_or_reuse_xgb(data, int(seed))


def holm_adjust(pvals):
    pvals = np.asarray(pvals, float)
    order = np.argsort(pvals)
    out = np.empty_like(pvals)
    prev = 0.0
    m = len(pvals)
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        val = max(prev, val)
        out[idx] = val
        prev = val
    return out


def summarize(ro, out, all_summaries, all_decisions):
    df = pd.DataFrame(all_summaries)
    dd = pd.DataFrame(all_decisions)
    df.to_csv(out / "all_summaries.csv", index=False)
    dd.to_csv(out / "decision_trace.csv", index=False)

    transports = []
    for jp in sorted((out / "jobs").glob("*/transport_trace.json")):
        transports.extend(json.loads(jp.read_text()))
    td = pd.DataFrame(transports)
    td.to_csv(out / "transport_trace.csv", index=False)

    piv = df.pivot_table(index=["dataset", "seed"], columns="method", values="brier_sum_over_classes")
    assessment = {}
    paired = []
    p_primary = []
    p_primary_ix = []
    for ds in DATASETS:
        d = piv.loc[ds]
        raw = d["raw"]
        rap = d["dtrar_rapid"]
        old = d["rapidt_old"]
        strict5 = d["rapid_fixed_0p005"]
        g = raw - rap
        gf = raw - strict5
        assessment[ds] = {
            "raw_mean": float(raw.mean()),
            "risk_optimal_mean": float(rap.mean()),
            "risk_optimal_gain_mean": float(g.mean()),
            "risk_optimal_gain_min_seed": float(g.min()),
            "risk_optimal_gain_max_seed": float(g.max()),
            "risk_optimal_positive_seed_count": int(np.sum(g > 0)),
            "old_rapid_gain_mean": float((raw-old).mean()),
            "fixed_0p005_gain_mean": float(gf.mean()),
            "fixed_0p005_gain_min_seed": float(gf.min()),
            "vamr_mean": float(d["vamr"].mean()),
            "logistic_mean": float(d["logistic"].mean()),
        }
        for m in ["rapidt_old", "dtrar_rapid", "rapid_fixed_0p005"]:
            gain = raw - d[m]
            try:
                pv = float(wilcoxon(gain, alternative="greater", zero_method="wilcox").pvalue)
            except Exception:
                pv = float("nan")
            row = {
                "dataset": ds, "method": m, "mean_gain": float(gain.mean()),
                "median_gain": float(gain.median()), "min_gain": float(gain.min()),
                "positive_seeds": int(np.sum(gain > 0)), "wilcoxon_p_one_sided": pv,
            }
            paired.append(row)
            if m == "dtrar_rapid":
                p_primary.append(pv); p_primary_ix.append(len(paired)-1)
    adj = holm_adjust([1.0 if not np.isfinite(x) else x for x in p_primary])
    for idx, val in zip(p_primary_ix, adj):
        paired[idx]["holm_p_primary"] = float(val)
    pd.DataFrame(paired).to_csv(out / "PAIRED_STATS.csv", index=False)

    checked = td[td["method"].isin(["dtrar_rapid", "rapid_fixed_0p005"])] .copy()
    checked["observed_le_envelope"] = checked["observed_brier_excess_vs_raw"] <= checked["envelope_at_deploy_eta"] + 1e-9
    checked["envelope_le_budget"] = checked["envelope_at_deploy_eta"] <= checked["harm_budget"] + 1e-9
    safety = checked.groupby(["dataset", "method"], as_index=False).agg(
        windows=("deployment_window", "count"),
        observed_le_envelope=("observed_le_envelope", "all"),
        envelope_le_budget=("envelope_le_budget", "all"),
        max_observed_excess=("observed_brier_excess_vs_raw", "max"),
        max_envelope=("envelope_at_deploy_eta", "max"),
        max_budget=("harm_budget", "max"),
    )
    safety.to_csv(out / "SAFETY_CONTRACT_CHECK.csv", index=False)

    frontier = []
    for ds in DATASETS:
        d = piv.loc[ds]
        for eps, m in zip(ro.FRONTIER_BUDGETS, ro.FRONTIER_METHODS):
            gain = d["raw"] - d[m]
            frontier.append({
                "dataset": ds, "prospective_harm_budget": float(eps),
                "brier_gain_mean": float(gain.mean()), "brier_gain_sd": float(gain.std(ddof=1)),
                "brier_gain_min_seed": float(gain.min()), "brier_gain_max_seed": float(gain.max()),
            })
    pd.DataFrame(frontier).to_csv(out / "SAFETY_PERFORMANCE_FRONTIER.csv", index=False)

    gates = {
        "both_unseen_mean_nonworse": bool(all(assessment[d]["risk_optimal_gain_mean"] >= -1e-10 for d in DATASETS)),
        "both_fixed_0p005_mean_nonworse": bool(all(assessment[d]["fixed_0p005_gain_mean"] >= -1e-10 for d in DATASETS)),
        "all_prospective_contracts_hold": bool(checked["observed_le_envelope"].all() and checked["envelope_le_budget"].all()),
    }
    assessment["gates"] = gates
    assessment["unseen_confirmation_pass"] = bool(all(gates.values()))
    (out / "UNSEEN_CONFIRMATION_ASSESSMENT.json").write_text(json.dumps(assessment, indent=2, sort_keys=True))
    return assessment


def main():
    global MANIFEST
    MANIFEST = load_manifest()
    must(RISKOPT.exists(), "run_dtrar_risk_optimal_final.py missing")
    # Import only after locked-manifest validation.
    sys.path.insert(0, str(PROJECT))
    import run_dtrar_risk_optimal_final as ro
    ro.self_test()
    ro.DATASETS = DATASETS
    ro.SEEDS = SEEDS
    # Exact frozen method, only the data/cache hooks change.
    ro.core.load_dataset = load_unseen
    ro.core.source_hashes = source_hashes_unseen
    ro.core.frozen_cache = frozen_cache_unseen

    identity = {
        "version": "RAPID-risk-optimal-unseen-XGB-v1",
        "datasets": DATASETS, "seeds": SEEDS, "model": MODEL,
        "method_source_sha256": sha256(RISKOPT),
        "unseen_audit_sha256": sha256(AUDIT_JSON),
        "beta": ro.BETA, "delay": ro.DELAY, "harm_cap": ro.HARM_CAP,
        "frontier_budgets": ro.FRONTIER_BUDGETS,
        "rule": "frozen risk-optimal RAPID unchanged; no unseen-specific tuning",
    }
    sig = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    root = PROJECT / "artifacts" / "unseen_riskopt_xgb"
    out = root / sig
    out.mkdir(parents=True, exist_ok=True)
    lockf = (root / "RUNNER.lock").open("w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("ERROR: unseen confirmation already running")
    (root / "LATEST.txt").write_text(str(out) + "\n")
    (out / "LOCKED_PLAN.json").write_text(json.dumps(identity, indent=2, sort_keys=True))
    (out / "ENVIRONMENT.json").write_text(json.dumps({"python": sys.version, "platform": platform.platform()}, indent=2))
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    shutil.copy2(AUDIT_JSON, out / "UNSEEN_DATASET_AUDIT_V2.json")

    print(f"UNSEEN_CONFIRMATION_BEGIN datasets={len(DATASETS)} seeds={len(SEEDS)} jobs={len(DATASETS)*len(SEEDS)} performance_was_unseen=true", flush=True)
    all_summaries, all_decisions = [], []
    with threadpool_limits(limits=1):
        for ds in DATASETS:
            for seed in SEEDS:
                s, d = ro.run_one(ds, seed, out)
                all_summaries.extend(s); all_decisions.extend(d)
    assessment = summarize(ro, out, all_summaries, all_decisions)

    review = PROJECT / "DTRAR_UnseenConfirmation_XGB_Review.zip"
    tmp = review.with_suffix(".zip.partial")
    keep = [
        "LOCKED_PLAN.json", "ENVIRONMENT.json", "UNSEEN_DATASET_AUDIT_V2.json",
        "all_summaries.csv", "decision_trace.csv", "transport_trace.csv",
        "PAIRED_STATS.csv", "SAFETY_CONTRACT_CHECK.csv", "SAFETY_PERFORMANCE_FRONTIER.csv",
        "UNSEEN_CONFIRMATION_ASSESSMENT.json", Path(__file__).name,
    ]
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name in keep:
            p = out / name
            must(p.exists(), f"review input missing: {p}")
            z.write(p, name)
    with zipfile.ZipFile(tmp) as z:
        must(z.testzip() is None, "review ZIP CRC failure")
    tmp.replace(review)
    print("UNSEEN_CONFIRMATION_ASSESSMENT " + json.dumps(assessment["gates"], sort_keys=True), flush=True)
    print(f"UNSEEN_CONFIRMATION_COMPLETE jobs={len(DATASETS)*len(SEEDS)} pass={str(assessment['unseen_confirmation_pass']).lower()} review={review.name}", flush=True)


if __name__ == "__main__":
    main()
