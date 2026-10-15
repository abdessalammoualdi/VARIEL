#!/usr/bin/env python3
"""DTRA-R + RAPID falsification pilot.

Purpose
-------
Test the new repairability mechanism before any full journal-scale matrix.
The pilot reuses frozen XGBoost caches and the three existing real streams.
No classifier retraining is started.

Locked pilot design
-------------------
- datasets: Electricity, Bank, Gas
- model: XGBoost
- seeds: 0,1,2
- target-label ceiling: 5%
- feedback delay: one deployment window
- label roles, fixed before labels are revealed: 40% repair-fit, 40% repair-audit,
  20% conformal-score
- repair portfolio: raw anchor, VAMR, feature-logistic repair
- RAPID certificate: simultaneous finite-population Serfling bounds over the
  last three audit-eligible completed windows
- minimum worthwhile Brier gain: 0.005
- methods evaluated under one shared label schedule: raw, VAMR, logistic,
  hard audit selector, DTRA-R+RAPID

The finite-population certificate is explicitly for the completed audit-eligible
populations used by RAPID. Future-window performance remains empirical.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import zipfile
import zlib

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parent
BASE = PROJECT / "offset_check_v07"
if not (BASE / "dtra" / "__init__.py").exists():
    raise SystemExit("ERROR: offset_check_v07 is missing. Run this script from the main DTRA folder beside README.md.")
sys.path.insert(0, str(BASE))

from dtra.acquisition import allocate, DelayedOracle
from dtra.adapt import frozen_cache
from dtra.cli import source_hashes
from dtra.conformal import candidate_scores, split_quantile, empirical_weighted_quantile
from dtra.data import load_dataset
from dtra.io import file_hash
from dtra.metrics import evaluate
from dtra.monitoring import monitoring_roles
from vamr.core import moments, solve_repair, simplex
from vamr.features import representation
from vamr.logit_control import fit_control

DATASETS = ["electricity", "bank", "gas"]
SEEDS = [0, 1, 2]
METHODS = ["raw", "vamr", "logistic", "hard_selector", "dtrar_rapid"]
BETA = 0.05
DELAY = 1
ALPHA = 0.05
ROLE_SHARE = {"fit": 0.40, "audit": 0.40, "cp": 0.20}
CERT_ALPHA = 0.05
MIN_GAIN = 0.005
BUFFER_WINDOWS = 3
SOURCE_MASS = 0.1
DECAY = 0.9
FAILURE_MARGIN = 0.02


def jdump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_default))


def json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x).__name__)


def brier_rows(p, y):
    p = np.asarray(p, float)
    y = np.asarray(y, int)
    return np.sum((p - np.eye(p.shape[1])[y]) ** 2, axis=1)


def serfling_radius(N: int, n: int, delta: float, span: float = 2.0) -> float:
    """Two-sided Serfling/Hoeffding-style finite-population radius.

    For a bounded variable in an interval of width ``span`` sampled uniformly
    without replacement from a finite population of size N. A union bound is
    applied by the caller across the finite set of planned certificate looks.
    """
    if not (1 <= n <= N):
        raise ValueError((N, n))
    if n == N:
        return 0.0
    if not (0 < delta < 1):
        raise ValueError(delta)
    fpc = 1.0 - (n - 1.0) / N
    return float(span * math.sqrt(fpc * math.log(2.0 / delta) / (2.0 * n)))


def project_simplex(v):
    v = np.asarray(v, float)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, len(v) + 1)
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.maximum(v - theta, 0)


def solve_portfolio(G, theta_bound):
    """Minimize the certified excess Brier-risk quadratic on the simplex."""
    G = np.asarray(G, float)
    theta_bound = np.asarray(theta_bound, float)
    J = G.shape[0]
    assert G.shape == (J, J) and theta_bound.shape == (J - 1,)
    c = np.r_[0.0, theta_bound]

    def fun(w):
        return float(w @ G @ w - G[0, 0] - 2.0 * c @ w)

    def jac(w):
        return 2.0 * G @ w - 2.0 * c

    starts = [np.eye(J)[j] for j in range(J)] + [np.ones(J) / J]
    best = None
    for x0 in starts:
        res = minimize(fun, x0, jac=jac, method="SLSQP",
                       bounds=[(0.0, 1.0)] * J,
                       constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0,
                                    "jac": lambda w: np.ones_like(w)},
                       options={"ftol": 1e-12, "maxiter": 500})
        if res.success:
            w = project_simplex(res.x)
            value = fun(w)
            if best is None or value < best[0]:
                best = (value, w)
    if best is None:
        raise RuntimeError("RAPID simplex optimization failed")
    return best[1], float(best[0])


def source_thresholds(p, y, score_ids, uniforms, alpha):
    scores = candidate_scores(p[score_ids], uniforms[score_ids])
    sy = y[score_ids]
    true = scores[np.arange(len(score_ids)), sy]
    return np.asarray([split_quantile(true[sy == k], alpha) for k in range(p.shape[1])]), true, sy


def role_quotas(windows, ceiling):
    """Allocate the global 5% budget only to origins that can affect a future window."""
    actionable = max(0, len(windows) - (DELAY + 1))
    caps = np.asarray([len(windows[t]) for t in range(actionable)], int)
    fit_total = int(math.floor(ROLE_SHARE["fit"] * ceiling))
    audit_total = int(math.floor(ROLE_SHARE["audit"] * ceiling))
    cp_total = ceiling - fit_total - audit_total
    if fit_total + audit_total + cp_total != ceiling:
        raise AssertionError
    # Since the overall ceiling is only 5%, independently allocated role quotas
    # cannot exhaust a window. We still verify the exact constraint below.
    fq = allocate(fit_total, caps) if fit_total else np.zeros(actionable, int)
    aq = allocate(audit_total, caps) if audit_total else np.zeros(actionable, int)
    cq = allocate(cp_total, caps) if cp_total else np.zeros(actionable, int)
    if np.any(fq + aq + cq > caps):
        raise RuntimeError("Role allocation overfilled a window")
    return actionable, fq, aq, cq, {"fit": fit_total, "audit": audit_total, "cp": cp_total}


def sample_roles(ix, nfit, naudit, ncp, seed, t):
    """Disjoint role assignment before labels are revealed.

    Fit is drawn first. Conditional on the realized fit sample, audit is a SRS
    from the full post-fit population, which is the finite population certified
    by RAPID. Conformal-score rows are drawn only after the audit role is fixed.
    """
    ix = np.asarray(ix, int)
    rng = np.random.default_rng(np.random.SeedSequence([seed, t, 55109]))
    fit = rng.choice(ix, nfit, replace=False) if nfit else np.empty(0, int)
    rem = ix[~np.isin(ix, fit)]
    audit = rng.choice(rem, naudit, replace=False) if naudit else np.empty(0, int)
    rem2 = rem[~np.isin(rem, audit)]
    cp = rng.choice(rem2, ncp, replace=False) if ncp else np.empty(0, int)
    return fit, audit, cp, rem


def fit_repairs(raw, z, basis, source_fit, source_y, fit_cohorts, known_fit, clock):
    if not fit_cohorts:
        return raw.copy(), raw.copy(), {"target_effective_n": 0.0, "probability_floor": 0.0}
    h, g, v = moments(basis, raw, fit_cohorts, known_fit, source_fit, source_y, clock,
                      decay=DECAY, source_mass=SOURCE_MASS)
    coef, info = solve_repair(h, g, v, raw.shape[1], shrink=True, feature=True)
    qv = simplex(raw + basis @ coef)
    ql = fit_control(raw, z, fit_cohorts, known_fit, source_fit, clock)
    weight_sum = 0.0
    weight_sq = 0.0
    for origin in sorted(fit_cohorts)[-BUFFER_WINDOWS:]:
        cohort = fit_cohorts[origin]
        for st in cohort["strata"]:
            wi = DECAY ** (clock - origin) / (len(cohort["indices"]) * st["pi"])
            nh = len(st["sample_ids"])
            weight_sum += nh * wi
            weight_sq += nh * wi * wi
    neff = weight_sum * weight_sum / weight_sq if weight_sq > 0 else 0.0
    floor = 1.0 / (neff + raw.shape[1]) if neff > 0 else 0.0
    if neff > 0:
        qv = (neff * qv + 1.0) / (neff + raw.shape[1])
        ql = (neff * ql + 1.0) / (neff + raw.shape[1])
    info = {**info, "target_effective_n": neff, "probability_floor": floor}
    return qv, ql, info


def rapid_certificate(raw, qv, ql, audit_cohorts, known_audit, update_number, max_updates):
    """Return a simultaneous completed-population repairability certificate."""
    origins = sorted(audit_cohorts)[-BUFFER_WINDOWS:]
    if not origins:
        return None
    J = 3
    # Union bound across every candidate contrast, every buffered cohort, and
    # every possible update in this locked pilot. This is deliberately
    # conservative; it avoids pretending a fixed-time interval is anytime-valid.
    delta_cell = CERT_ALPHA / max(1, (J - 1) * BUFFER_WINDOWS * max_updates)
    populations = []
    totalN = 0
    theta_hat_num = np.zeros(J - 1)
    L_num = np.zeros(J - 1)
    U_num = np.zeros(J - 1)
    F_all = []
    for origin in origins:
        c = audit_cohorts[origin]
        pop = np.asarray(c["population_ids"], int)
        sample = np.asarray(c["audit_ids"], int)
        if len(sample) == 0:
            continue
        N, n = len(pop), len(sample)
        Fpop = np.stack([raw[pop], qv[pop], ql[pop]], axis=1)  # N x J x K
        F_all.append(Fpop)
        totalN += N
        ys = np.asarray([known_audit[int(i)] for i in sample], int)
        for j, q in enumerate([qv, ql]):
            x = q[sample, ys] - raw[sample, ys]
            mean = float(np.mean(x))
            rad = serfling_radius(N, n, delta_cell, span=2.0)
            theta_hat_num[j] += N * mean
            L_num[j] += N * max(-1.0, mean - rad)
            U_num[j] += N * min(1.0, mean + rad)
        populations.append({"origin_window": origin + 1, "population_n": N,
                            "audit_n": n, "delta_cell": delta_cell})
    if totalN == 0:
        return None
    F = np.concatenate(F_all, axis=0)
    G = np.einsum("njk,nlk->jl", F, F) / len(F)
    theta_hat = theta_hat_num / totalN
    L = L_num / totalN
    U = U_num / totalN
    w_robust, upper_excess = solve_portfolio(G, L)
    w_optimistic, lower_excess = solve_portfolio(G, U)
    gain_lower = max(0.0, -upper_excess)
    gain_upper = max(0.0, -lower_excess)
    # Hard-selector comparator: choose the vertex with smallest pessimistic excess.
    vertex = []
    for j in range(J):
        w = np.eye(J)[j]
        cvec = np.r_[0.0, L]
        val = float(w @ G @ w - G[0, 0] - 2.0 * cvec @ w)
        vertex.append(val)
    hard_index = int(np.argmin(vertex))
    # Oracle is diagnostic only; full labels are added by the evaluator later.
    return {
        "G": G, "theta_hat": theta_hat, "theta_lower": L, "theta_upper": U,
        "weights": w_robust, "optimistic_weights": w_optimistic,
        "gain_lower": gain_lower, "gain_upper": gain_upper,
        "hard_index": hard_index, "populations": populations,
        "update_number": update_number,
    }


def oracle_portfolio(raw, qv, ql, populations, y):
    if not populations:
        return {"oracle_weights": np.array([1.0, 0.0, 0.0]), "oracle_gain": 0.0}
    pop = np.unique(np.concatenate([np.asarray(p, int) for p in populations]))
    F = np.stack([raw[pop], qv[pop], ql[pop]], axis=1)
    G = np.einsum("njk,nlk->jl", F, F) / len(F)
    theta = np.asarray([
        np.mean(qv[pop, y[pop]] - raw[pop, y[pop]]),
        np.mean(ql[pop, y[pop]] - raw[pop, y[pop]])
    ])
    w, excess = solve_portfolio(G, theta)
    return {"oracle_weights": w, "oracle_gain": max(0.0, -excess)}


def update_cp_thresholds(method, source_true, source_y, source_cutoff, cp_cohorts, known_cp, clock, k):
    active = sorted(cp_cohorts)[-BUFFER_WINDOWS:]
    if not active:
        return source_cutoff.copy()
    scores = source_true.tolist()
    labels = source_y.tolist()
    weights = [SOURCE_MASS / len(source_true)] * len(source_true)
    target_scores = []
    target_weights = []
    target_labels = []
    for origin in active:
        c = cp_cohorts[origin]
        ids = np.asarray(c["cp_ids"], int)
        if len(ids) == 0:
            continue
        ys = np.asarray([known_cp[int(i)] for i in ids], int)
        ts = np.asarray(c["issued_true_scores"][method], float)
        wi = DECAY ** (clock - origin) / len(ids)
        scores.extend(ts.tolist()); labels.extend(ys.tolist()); weights.extend([wi] * len(ids))
        target_scores.extend(ts.tolist()); target_labels.extend(ys.tolist()); target_weights.extend([wi] * len(ids))
    if not target_scores:
        return source_cutoff.copy()
    s = np.asarray(scores); lab = np.asarray(labels); w = np.asarray(weights)
    ts = np.asarray(target_scores); tl = np.asarray(target_labels); tw = np.asarray(target_weights)
    pooled = empirical_weighted_quantile(ts, tw, 1 - ALPHA)
    cutoff = source_cutoff.copy()
    for cls in range(k):
        take_t = tl == cls
        neff = (tw[take_t].sum() ** 2 / np.sum(tw[take_t] ** 2)) if np.any(take_t) else 0.0
        take = lab == cls
        if neff < math.ceil(1 / ALPHA):
            cutoff[cls] = max(source_cutoff[cls], pooled)
        else:
            cutoff[cls] = empirical_weighted_quantile(s[take], w[take], 1 - ALPHA)
    return cutoff


def run_one(dataset, seed, out):
    data = load_dataset(dataset, BASE / "data" / "raw")
    hashes = source_hashes(BASE / "data" / "raw", dataset)
    cache, raw = frozen_cache(BASE / "artifacts", data, "xgboost", seed, "primary", hashes)
    z, _ = representation(data)
    roles = monitoring_roles(data.source, data.y)
    source_fit, source_score, source_monitor = roles
    source_y = data.y[source_fit]
    uniforms = np.random.default_rng(np.random.SeedSequence([seed, 19073])).random(len(raw))
    source_cutoff, source_true, source_score_y = source_thresholds(raw, data.y, source_score, uniforms, ALPHA)
    source_ref_brier = float(np.mean(brier_rows(raw[source_monitor], data.y[source_monitor])))
    basis = np.column_stack([np.ones(len(z)), raw[:, :-1], z])
    target = np.concatenate(data.windows)
    ceiling = math.floor(BETA * len(target))
    actionable, fitq, auditq, cpq, role_totals = role_quotas(data.windows, ceiling)
    max_updates = max(1, actionable)

    oracle = DelayedOracle(data.y, raw.shape[1], seed=np.random.SeedSequence([seed, 90731]))
    known_fit = {int(i): int(data.y[i]) for i in data.source}
    known_audit = {}
    known_cp = {}
    pending = {}
    fit_cohorts = {}
    audit_cohorts = {}
    cp_cohorts = {}

    qv = raw.copy(); ql = raw.copy()
    rapid_w = np.array([1.0, 0.0, 0.0])
    hard_idx = 0
    decision = "HOLD"
    cert = None
    thresholds = {m: source_cutoff.copy() for m in METHODS}

    issued_p = {m: np.zeros((len(target), raw.shape[1])) for m in METHODS}
    issued_s = {m: np.zeros((len(target), raw.shape[1]), bool) for m in METHODS}
    pos = 0
    decisions = []
    query_records = []
    update_number = 0

    for t, ix in enumerate(data.windows):
        nwin = len(ix)
        rapid_prob_all = rapid_w[0] * raw + rapid_w[1] * qv + rapid_w[2] * ql
        hard_prob_all = [raw, qv, ql][hard_idx]
        prob_now = {
            "raw": raw[ix], "vamr": qv[ix], "logistic": ql[ix],
            "hard_selector": hard_prob_all[ix], "dtrar_rapid": rapid_prob_all[ix],
        }
        for m in METHODS:
            issued_p[m][pos:pos+nwin] = prob_now[m]
            scores = candidate_scores(prob_now[m], uniforms[ix])
            issued_s[m][pos:pos+nwin] = scores <= thresholds[m][None, :]

        if t < actionable:
            fit_ids, audit_ids, cp_ids, audit_population = sample_roles(
                ix, int(fitq[t]), int(auditq[t]), int(cpq[t]), seed, t)
            rec = {"fit_ids": fit_ids, "audit_ids": audit_ids, "cp_ids": cp_ids,
                   "audit_population": audit_population,
                   "issued_cp_scores": {}}
            for m in METHODS:
                if len(cp_ids):
                    loc = {int(r): j for j, r in enumerate(ix)}
                    cp_pos = np.asarray([loc[int(r)] for r in cp_ids], int)
                    sc = candidate_scores(prob_now[m], uniforms[ix])
                    rec["issued_cp_scores"][m] = sc[cp_pos]
                else:
                    rec["issued_cp_scores"][m] = np.empty((0, raw.shape[1]))
            pending[t] = rec
            for kind, ids, denom in [
                ("fit", fit_ids, len(ix)),
                ("audit", audit_ids, len(audit_population)),
                ("cp", cp_ids, len(ix) - len(fit_ids) - len(audit_ids)),
            ]:
                if len(ids):
                    pi = np.full(len(ids), len(ids) / denom)
                    oracle.request(ids, kind, t, t, DELAY, pi)
                    query_records.extend({"row_id": int(i), "kind": kind,
                                          "origin_window": t+1, "request_after_window": t+1,
                                          "arrival_after_window": t+DELAY+1,
                                          "conditional_inclusion_probability": float(pii)}
                                         for i, pii in zip(ids, pi))

        due = oracle.release(t)
        if due:
            origins_due = sorted(set(int(job["origin"]) for job in due))
            for origin in origins_due:
                rec = pending[origin]
                # Prepare containers; labels are filled from role-specific jobs below.
                fit_ids = rec["fit_ids"]
                if len(fit_ids):
                    fit_cohorts[origin] = {"indices": data.windows[origin], "strata": [{
                        "stratum": 0, "population_n": len(data.windows[origin]),
                        "sample_ids": fit_ids, "pi": len(fit_ids) / len(data.windows[origin])}]}
                if len(rec["audit_ids"]):
                    audit_cohorts[origin] = {"population_ids": rec["audit_population"],
                                             "audit_ids": rec["audit_ids"]}
                if len(rec["cp_ids"]):
                    cp_cohorts[origin] = {"cp_ids": rec["cp_ids"], "issued_true_scores": {}}
            for job in due:
                origin = int(job["origin"])
                if job["kind"] == "fit":
                    for i, yy in zip(job["indices"], job["labels"]):
                        known_fit[int(i)] = int(yy)
                elif job["kind"] == "audit":
                    for i, yy in zip(job["indices"], job["labels"]):
                        known_audit[int(i)] = int(yy)
                elif job["kind"] == "cp":
                    ids = np.asarray(job["indices"], int)
                    ys = np.asarray(job["labels"], int)
                    for i, yy in zip(ids, ys):
                        known_cp[int(i)] = int(yy)
                    rec = pending[origin]
                    for m in METHODS:
                        matrix = np.asarray(rec["issued_cp_scores"][m])
                        cp_cohorts[origin]["issued_true_scores"][m] = matrix[np.arange(len(ids)), ys]
                else:
                    raise RuntimeError(job["kind"])

            qv, ql, fit_info = fit_repairs(raw, z, basis, source_fit, source_y,
                                            fit_cohorts, known_fit, t)
            update_number += 1
            cert = rapid_certificate(raw, qv, ql, audit_cohorts, known_audit,
                                     update_number, max_updates)
            if cert is None:
                rapid_w = np.array([1.0, 0.0, 0.0]); hard_idx = 0; decision = "HOLD"
            else:
                rapid_w = np.asarray(cert["weights"])
                hard_idx = int(cert["hard_index"])
                # Estimate whether raw itself is clearly degraded on the active audit sample.
                active = sorted(audit_cohorts)[-BUFFER_WINDOWS:]
                aud = np.concatenate([audit_cohorts[o]["audit_ids"] for o in active])
                ay = np.asarray([known_audit[int(i)] for i in aud], int)
                raw_audit = float(np.mean(brier_rows(raw[aud], ay))) if len(aud) else source_ref_brier
                if cert["gain_lower"] > MIN_GAIN:
                    decision = "REPAIR"
                elif cert["gain_upper"] <= MIN_GAIN and raw_audit > source_ref_brier + FAILURE_MARGIN:
                    decision = "ESCALATE"
                    rapid_w = np.array([1.0, 0.0, 0.0])
                else:
                    decision = "HOLD"
                    rapid_w = np.array([1.0, 0.0, 0.0])
                pops = [audit_cohorts[o]["population_ids"] for o in active]
                oracle_diag = oracle_portfolio(raw, qv, ql, pops, data.y)
                cert.update(oracle_diag)
                cert["repair_regret_bound_proxy"] = max(0.0, cert["oracle_gain"] - cert["gain_lower"])

            for m in METHODS:
                thresholds[m] = update_cp_thresholds(m, source_true, source_score_y,
                                                     source_cutoff, cp_cohorts, known_cp,
                                                     t, raw.shape[1])
            decisions.append({
                "after_window": t+1, "effective_from_window": t+2,
                "decision": decision, "weights_raw": float(rapid_w[0]),
                "weights_vamr": float(rapid_w[1]), "weights_logistic": float(rapid_w[2]),
                "hard_choice": ["raw", "vamr", "logistic"][hard_idx],
                "gain_lower": None if cert is None else float(cert["gain_lower"]),
                "gain_upper": None if cert is None else float(cert["gain_upper"]),
                "oracle_gain_completed_population": None if cert is None else float(cert["oracle_gain"]),
                "oracle_weights": None if cert is None else np.asarray(cert["oracle_weights"]).tolist(),
                "theta_hat": None if cert is None else np.asarray(cert["theta_hat"]).tolist(),
                "theta_lower": None if cert is None else np.asarray(cert["theta_lower"]).tolist(),
                "theta_upper": None if cert is None else np.asarray(cert["theta_upper"]).tolist(),
                "fit_info": fit_info,
            })
        pos += nwin

    if pos != len(target):
        raise AssertionError
    # Every requested observation has exactly one role.
    qids = [r["row_id"] for r in query_records]
    if len(qids) != len(set(qids)) or len(qids) > ceiling:
        raise AssertionError("Budget or role-disjointness failure")

    job = out / "jobs" / f"{dataset}_seed{seed}"
    job.mkdir(parents=True, exist_ok=True)
    summaries = []
    for m in METHODS:
        metrics = evaluate(issued_p[m], data.y[target], issued_s[m])
        row = {"dataset": dataset, "seed": seed, "method": m, "model": "xgboost",
               "budget_fraction": BETA, "delay": DELAY, "alpha": ALPHA,
               "labels_requested": len(qids), "budget_ceiling": ceiling,
               "fit_labels_requested": sum(r["kind"] == "fit" for r in query_records),
               "audit_labels_requested": sum(r["kind"] == "audit" for r in query_records),
               "cp_labels_requested": sum(r["kind"] == "cp" for r in query_records),
               "base_model_retrained": False, **metrics}
        summaries.append(row)
        np.savez_compressed(job / f"issued_{m}.npz", row_id=target,
                            probabilities=issued_p[m], prediction_set=issued_s[m])
    np.savez_compressed(job / "evaluation_labels.npz", row_id=target, y=data.y[target])
    jdump(job / "summaries.json", summaries)
    jdump(job / "queries.json", query_records)
    jdump(job / "decision_trace.json", decisions)
    jdump(job / "job_record.json", {
        "dataset": dataset, "seed": seed, "frozen_cache": str(cache),
        "role_totals": role_totals, "actionable_origin_windows": actionable,
        "certificate_scope": "Completed post-fit audit-eligible populations; future deployment empirical.",
        "certificate_alpha": CERT_ALPHA, "minimum_worthwhile_brier_gain": MIN_GAIN,
        "source_reference_brier": source_ref_brier,
    })
    print("RAPID_JOB_RESULT " + json.dumps({"dataset": dataset, "seed": seed,
          "raw_brier": summaries[0]["brier_sum_over_classes"],
          "rapid_brier": summaries[-1]["brier_sum_over_classes"],
          "labels": len(qids), "decisions": [d["decision"] for d in decisions]}), flush=True)
    return summaries, decisions


def publication_diagnostics(out, all_summaries, all_decisions):
    df = pd.DataFrame(all_summaries)
    df.to_csv(out / "all_summaries.csv", index=False)
    jdump(out / "all_summaries.json", all_summaries)
    dd = pd.DataFrame(all_decisions)
    dd.to_csv(out / "decision_trace.csv", index=False)
    jdump(out / "decision_trace.json", all_decisions)
    grouped = df.groupby(["dataset", "method"])[["brier_sum_over_classes", "coverage",
                  "worst_present_class_coverage", "set_size", "aurc_discrete"]].agg(["mean", "std"])
    grouped.to_csv(out / "mean_sd.csv")

    # Gate is intentionally predeclared: the pilot passes only if RAPID protects
    # Electricity while recovering useful Bank/Gas improvement relative to raw.
    pivot = df.pivot_table(index=["dataset", "seed"], columns="method",
                           values="brier_sum_over_classes")
    checks = {}
    for ds in DATASETS:
        d = pivot.loc[ds]
        raw = d["raw"].mean(); rapid = d["dtrar_rapid"].mean(); logit = d["logistic"].mean()
        checks[ds] = {"raw_mean": float(raw), "rapid_mean": float(rapid),
                      "logistic_mean": float(logit),
                      "rapid_minus_raw": float(rapid - raw)}
    checks["gates"] = {
        "electricity_protection": checks["electricity"]["rapid_minus_raw"] <= 0.01,
        "bank_improvement": checks["bank"]["rapid_minus_raw"] < -0.01,
        "gas_improvement": checks["gas"]["rapid_minus_raw"] < -0.03,
    }
    checks["pilot_pass"] = bool(all(checks["gates"].values()))
    jdump(out / "PILOT_ASSESSMENT.json", checks)

    # Vector PDFs only, plus CSV source data.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figdir = out / "figures"; figdir.mkdir(exist_ok=True)
        methods = METHODS
        for ds in DATASETS:
            sub = df[df.dataset == ds]
            vals = [sub[sub.method == m].brier_sum_over_classes.mean() for m in methods]
            errs = [sub[sub.method == m].brier_sum_over_classes.std(ddof=1) for m in methods]
            fig, ax = plt.subplots(figsize=(8.4, 4.8), layout="constrained")
            x = np.arange(len(methods))
            ax.errorbar(x, vals, yerr=errs, marker="o", capsize=4, linewidth=1.6)
            ax.set_xticks(x, ["Raw", "VAMR", "Logistic", "Hard selector", "DTRA-R+RAPID"], rotation=15)
            ax.set_ylabel("Brier loss (lower is better)")
            ax.set_title(f"RAPID falsification pilot — {ds}")
            ax.grid(alpha=0.2)
            fig.savefig(figdir / f"{ds}_brier_pilot.pdf", bbox_inches="tight")
            plt.close(fig)
        # Decision/repairability timeline for every dataset and seed.
        ddf = pd.DataFrame(all_decisions)
        for (ds, seed), sub in ddf.groupby(["dataset", "seed"]):
            sub = sub.sort_values("effective_from_window")
            if sub["gain_lower"].notna().sum() == 0:
                continue
            fig, ax = plt.subplots(figsize=(8.4, 4.8), layout="constrained")
            x = sub["effective_from_window"].to_numpy()
            lo = sub["gain_lower"].fillna(0).to_numpy(float)
            hi = sub["gain_upper"].fillna(0).to_numpy(float)
            ax.fill_between(x, lo, hi, alpha=0.2)
            ax.plot(x, lo, marker="o", label="Certified gain lower bound")
            ax.plot(x, hi, marker="s", label="Possible gain upper bound")
            ax.axhline(MIN_GAIN, linestyle="--", linewidth=1, label="Worthwhile-gain threshold")
            ax.set(xlabel="Effective deployment window", ylabel="Brier improvement",
                   title=f"RAPID repairability envelope — {ds}, seed {seed}")
            ax.legend(frameon=False)
            ax.grid(alpha=0.2)
            fig.savefig(figdir / f"{ds}_seed{seed}_repairability_timeline.pdf", bbox_inches="tight")
            plt.close(fig)
    except Exception as exc:
        (out / "FIGURE_WARNING.txt").write_text(str(exc))
    return checks


def export_review(out, project_root):
    paths = [p for p in out.rglob("*") if p.is_file() and "review" not in p.relative_to(out).parts]
    groups, part, size = [], [], 120_000
    for p in sorted(paths):
        cost = len(zlib.compress(p.read_bytes(), 6)) + 2000
        if cost > 7 * 2**20:
            raise RuntimeError("Unexpected oversized review input: " + str(p))
        if part and size + cost > 7 * 2**20:
            groups.append(part); part = []; size = 120_000
        part.append(p); size += cost
    if part: groups.append(part)
    review = out / "review"; review.mkdir(exist_ok=True)
    for i, group in enumerate(groups, 1):
        dest = review / f"DTRAR_RAPID_Pilot_Review_{i:02d}_of_{len(groups):02d}.zip"
        tmp = dest.with_suffix(".zip.partial")
        manifest = []
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in group:
                b = p.read_bytes(); rel = str(p.relative_to(out))
                zf.writestr(rel, b)
                manifest.append({"path": rel, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()})
            zf.writestr("PART_MANIFEST.json", json.dumps({"part": i, "parts": len(groups), "files": manifest}, indent=2))
        with zipfile.ZipFile(tmp) as zf:
            if zf.testzip() is not None: raise RuntimeError("Review ZIP CRC failure")
        tmp.replace(dest)
        shutil.copy2(dest, project_root / dest.name)
        print("RAPID_REVIEW_PART_READY " + str(project_root / dest.name), flush=True)
    return len(groups)


def self_test():
    # Portfolio optimizer returns the raw vertex when every repair is inferior.
    G = np.array([[0.4, 0.4, 0.4], [0.4, 0.7, 0.5], [0.4, 0.5, 0.8]])
    w, val = solve_portfolio(G, np.array([0.0, 0.0]))
    assert np.argmax(w) == 0 and val <= 1e-9
    # A strong true-class advantage should move weight off raw.
    w2, val2 = solve_portfolio(G, np.array([0.25, 0.25]))
    assert val2 < 0 and (w2[1] + w2[2]) > 0.1
    # Serfling radius is finite and vanishes when the population is exhausted.
    assert serfling_radius(100, 100, 1e-3) == 0
    assert 0 < serfling_radius(100, 20, 1e-3) < 2
    # Role design is disjoint.
    ix = np.arange(100)
    f, a, c, rem = sample_roles(ix, 20, 20, 10, 1, 0)
    assert not set(f) & set(a) and not set(f) & set(c) and not set(a) & set(c)
    assert set(a).issubset(set(rem))
    print("RAPID_SELF_TEST_COMPLETE checks=8", flush=True)


def main():
    self_test()
    # Validate prerequisites without modifying old result folders.
    for required in [BASE / "configs" / "raw_manifest.json", BASE / "artifacts" / "models"]:
        if not required.exists():
            raise SystemExit("ERROR: required project input missing: " + str(required))

    identity = {
        "version": "DTRA-RAPID-falsification-v0.8",
        "datasets": DATASETS, "seeds": SEEDS, "methods": METHODS,
        "beta": BETA, "delay": DELAY, "alpha": ALPHA,
        "role_share": ROLE_SHARE, "certificate_alpha": CERT_ALPHA,
        "minimum_worthwhile_brier_gain": MIN_GAIN,
        "buffer_windows": BUFFER_WINDOWS,
        "base_code": {str(p.relative_to(BASE)): file_hash(p) for folder in ["dtra", "vamr"]
                      for p in (BASE / folder).rglob("*.py")},
        "runner_sha256": file_hash(Path(__file__)),
    }
    sig = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    root = PROJECT / "artifacts" / "dtrar_rapid_pilot"
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "RUNNER.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("DTRA-R RAPID pilot is already running. Read the existing log.")
    out = root / sig; out.mkdir(exist_ok=True)
    (root / "LATEST.txt").write_text(str(out) + "\n")
    jdump(out / "LOCKED_PLAN.json", identity)
    jdump(out / "ENVIRONMENT.json", {"python": sys.version, "platform": platform.platform()})
    shutil.copy2(Path(__file__), out / "run_dtrar_rapid_pilot.py")

    print("DTRAR_RAPID_PILOT_BEGIN jobs=9 base_model_retrained=false", flush=True)
    all_summaries, all_decisions = [], []
    with threadpool_limits(limits=1):
        for ds in DATASETS:
            for seed in SEEDS:
                s, d = run_one(ds, seed, out)
                all_summaries.extend(s)
                all_decisions.extend({"dataset": ds, "seed": seed, **row} for row in d)
    assessment = publication_diagnostics(out, all_summaries, all_decisions)
    parts = export_review(out, PROJECT)
    print("RAPID_PILOT_ASSESSMENT " + json.dumps(assessment["gates"]), flush=True)
    print(f"DTRAR_RAPID_PILOT_COMPLETE jobs=9 method_rows={len(all_summaries)} review_parts={parts} pass={str(assessment['pilot_pass']).lower()}", flush=True)


if __name__ == "__main__":
    main()
