#!/usr/bin/env python3
"""Core implementation for the final HB-GO-Canary-RAPID controller.

Information pattern
-------------------
For target window t:
  1. Freeze q*_t using source labels and canary labels from windows < t only.
  2. Allocate a predeclared current-window canary budget using current unlabeled
     repair geometry. Canary predictions remain RAW permanently.
  3. Observe only those canary labels, estimate the Brier linear term for the
     disjoint non-canary bulk, solve the current-window risk-optimal scalar
     action, and clip it by the exact prospective worst-current-label Brier
     liability H_t <= epsilon.
  4. Recycle the same observed canary labels into the repair fit for q*_{t+1}.

No second set of delayed repair labels exists. The same label budget performs
both current utility estimation and future candidate learning.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

EPS = 0.005
AUDIT_ALPHA = 0.05
MAX_STRATA = 8
MIN_CANARY = 8
BUDGET_RATE = 0.04
BUFFER_WINDOWS = 3
DECAY = 0.9
NUMERIC_TOL = 1e-12


def must(c, m):
    if not c:
        raise RuntimeError(m)


def stable_seed(tag, family, dataset, seed, window, extra=0):
    s = f"{tag}|{family}|{dataset}|{seed}|{window}|{extra}".encode()
    return int.from_bytes(hashlib.sha256(s).digest()[:8], "little") % (2**32)


def simplex(p):
    p = np.asarray(p, float)
    p = np.clip(p, 0.0, 1.0)
    s = p.sum(axis=1, keepdims=True)
    s[s <= 0] = 1.0
    return p / s


def brier_rows(p, y):
    p = np.asarray(p, float)
    y = np.asarray(y, int)
    e = np.zeros_like(p)
    e[np.arange(len(y)), y] = 1.0
    return np.sum((p - e) ** 2, axis=1)


def prospective_H(raw, delta):
    raw = np.asarray(raw, float)
    delta = np.asarray(delta, float)
    if len(raw) == 0:
        return 0.0
    return float(np.mean(
        np.sum(delta * delta, axis=1)
        + 2.0 * np.sum(raw * delta, axis=1)
        - 2.0 * np.min(delta, axis=1)
    ))


def repair_geometry(p, q):
    d = q - p
    offset = np.sum(p * d, axis=1)
    potential = d - offset[:, None]
    lo = potential.min(axis=1)
    hi = potential.max(axis=1)
    center = 0.5 * (lo + hi)
    halfwidth = 0.5 * (hi - lo)
    return potential, center, halfwidth


def largest_remainder(total, caps, mass, min_each=True):
    caps = np.asarray(caps, int)
    mass = np.asarray(mass, float)
    H = len(caps)
    if total <= 0 or H == 0:
        return np.zeros(H, int)
    total = int(min(total, int(caps.sum())))
    base = np.zeros(H, int)
    if min_each and total >= 2 * H:
        base[:] = 2
    elif min_each and total >= H:
        base[:] = 1
    base = np.minimum(base, caps)
    remain = int(total - base.sum())
    if remain <= 0:
        return base
    avail = caps - base
    m = np.where(avail > 0, np.maximum(mass, 0.0), 0.0)
    if m.sum() <= 0:
        m = np.where(avail > 0, caps.astype(float), 0.0)
    ideal = remain * m / m.sum()
    add = np.minimum(np.floor(ideal).astype(int), avail)
    base += add
    remain = int(total - base.sum())
    frac = ideal - np.floor(ideal)
    while remain > 0:
        cand = np.where(base < caps)[0]
        must(len(cand) > 0, "unable to complete allocation")
        j = max(cand.tolist(), key=lambda z: (frac[z], m[z], -z))
        base[j] += 1
        frac[j] = -1.0
        remain -= 1
    must(int(base.sum()) == total, "allocation count mismatch")
    return base


def horizon_schedule(windows, budget_rate=BUDGET_RATE, mode="balanced"):
    lens = np.asarray([len(w) for w in windows], int)
    N = int(lens.sum())
    B = int(math.floor(float(budget_rate) * N))
    must(B >= 0 and B < N, "invalid total canary budget")
    if B == 0:
        return {i + 1: 0 for i in range(len(windows))}, 0

    if mode == "balanced":
        caps = np.maximum(lens - 1, 0)
        mass = lens.astype(float)
        alloc = largest_remainder(B, caps, mass, min_each=False)
    elif mode == "legacy_terminal_blind":
        # Reproduce the information blackout of a one-window-delay protocol:
        # the final DELAY+1 windows cannot contribute labels before horizon end.
        actionable = max(1, len(windows) - 2)
        caps = np.zeros_like(lens)
        caps[:actionable] = np.maximum(lens[:actionable] - 1, 0)
        must(B <= int(caps.sum()), "legacy schedule cannot hold requested budget")
        alloc = largest_remainder(B, caps, lens.astype(float), min_each=False)
    else:
        raise ValueError(f"unknown schedule mode {mode}")

    must(int(alloc.sum()) == B, "schedule changed total budget")
    return {i + 1: int(alloc[i]) for i in range(len(windows))}, B


def canary_design(p, q, m, rng, mode="geometry"):
    n = len(p)
    m = int(min(max(m, 0), max(0, n - 1)))
    if m <= 0:
        return [], np.empty(0, int), np.arange(n, dtype=int)

    potential, center, halfwidth = repair_geometry(p, q)
    if mode == "uniform":
        pop = np.arange(n, dtype=int)
        samp = np.sort(rng.choice(pop, size=m, replace=False))
        chosen = np.zeros(n, bool)
        chosen[samp] = True
        bulk = np.flatnonzero(~chosen)
        return [{
            "stratum": 0, "population": pop, "sample": samp, "bulk": bulk,
            "Nh": n, "nh": m, "Mh": len(bulk),
            "hmax": float(np.max(halfwidth)) if n else 0.0,
            "design_pi": float(m / n),
        }], samp, bulk

    H = int(max(1, min(MAX_STRATA, m, n)))
    order = np.argsort(halfwidth, kind="mergesort")
    blocks = [np.asarray(x, int) for x in np.array_split(order, H) if len(x)]
    caps = np.asarray([len(x) for x in blocks], int)
    hmax = np.asarray([float(np.max(halfwidth[x])) for x in blocks])
    mass = caps.astype(float) * np.maximum(2.0 * hmax, 1e-15)
    nh = largest_remainder(m, caps, mass, min_each=True)

    strata, canary_parts, bulk_parts = [], [], []
    for h, (pop, take) in enumerate(zip(blocks, nh)):
        take = int(take)
        samp = np.sort(rng.choice(pop, size=take, replace=False)) if take else np.empty(0, int)
        chosen = np.isin(pop, samp)
        bulk = pop[~chosen]
        canary_parts.append(samp)
        bulk_parts.append(bulk)
        strata.append({
            "stratum": int(h), "population": pop, "sample": samp, "bulk": bulk,
            "Nh": int(len(pop)), "nh": int(len(samp)), "Mh": int(len(bulk)),
            "hmax": float(hmax[h]),
            "design_pi": float(len(samp) / len(pop)) if len(pop) else 0.0,
        })
    canary = np.sort(np.concatenate(canary_parts)) if canary_parts else np.empty(0, int)
    bulk = np.sort(np.concatenate(bulk_parts)) if bulk_parts else np.empty(0, int)
    must(len(canary) == m, "canary size mismatch")
    must(len(canary) + len(bulk) == n, "canary/bulk partition mismatch")
    return strata, canary, bulk


def perturb_labels(true_y, k, rng, noise=0.0):
    out = np.asarray(true_y, int).copy()
    if noise <= 0 or len(out) == 0:
        return out
    mask = rng.random(len(out)) < float(noise)
    for j in np.flatnonzero(mask):
        r = int(rng.integers(0, k - 1))
        if r >= out[j]:
            r += 1
        out[j] = r
    return out


def observed_strata(strata, dropout, rng):
    """Apply MCAR label dropout after canary selection; canary rows stay reserved/raw."""
    out = []
    for st in strata:
        samp = np.asarray(st["sample"], int)
        if len(samp) and dropout > 0:
            keep = rng.random(len(samp)) >= float(dropout)
            obs = samp[keep]
        else:
            obs = samp.copy()
        out.append({**st, "observed": np.asarray(obs, int), "nobs": int(len(obs))})
    return out


def estimate_bulk_D(p, q, y_observed_by_local, strata_obs, bulk_local):
    if len(bulk_local) == 0:
        return {"D_hat": 0.0, "D_lower": 0.0, "D_upper": 0.0,
                "radius": 0.0, "C_bulk": 0.0, "n_observed": 0, "estimable": False}
    potential, center, halfwidth = repair_geometry(p, q)
    d = q - p
    M = int(len(bulk_local))
    total_hat = 0.0
    variance_proxy = 0.0
    nobs = 0

    for st in strata_obs:
        bulk = np.asarray(st["bulk"], int)
        Mh = int(len(bulk))
        if Mh <= 0:
            continue
        obs = np.asarray(st["observed"], int)
        if len(obs) == 0:
            return {"D_hat": 0.0, "D_lower": 0.0, "D_upper": 0.0,
                    "radius": float("inf"), "C_bulk": 0.0,
                    "n_observed": nobs, "estimable": False}
        ys = np.asarray([y_observed_by_local[int(i)] for i in obs], int)
        z_s = potential[obs, ys]
        r_s = z_s - center[obs]
        rbar = float(np.mean(r_s))
        total_hat += float(np.sum(center[bulk])) + Mh * rbar
        nobs += len(obs)
        Nh = int(st["Nh"])
        rng_h = 2.0 * float(st["hmax"])
        variance_proxy += ((Nh / M) * rng_h) ** 2 / len(obs)

    D_hat = float(total_hat / M)
    radius = float(math.sqrt(max(0.0, 0.5 * math.log(2.0 / AUDIT_ALPHA) * variance_proxy)))
    global_lower = float(np.min(potential[bulk_local]))
    global_upper = float(np.max(potential[bulk_local]))
    D_lower = float(max(global_lower, D_hat - radius))
    D_upper = float(min(global_upper, D_hat + radius))
    C_bulk = float(np.mean(np.sum(d[bulk_local] ** 2, axis=1)))
    return {"D_hat": D_hat, "D_lower": D_lower, "D_upper": D_upper,
            "radius": radius, "C_bulk": C_bulk, "n_observed": int(nobs), "estimable": True}


def current_risk_optimal_eta(D_hat, D_lower, C, eps=EPS):
    if C <= 1e-18 or D_hat <= NUMERIC_TOL:
        return 0.0, 0.0, 0.0
    eta_plugin = float(np.clip(D_hat / C, 0.0, 1.0))
    disc = D_lower * D_lower + C * eps
    eta_audit_max = float(np.clip((D_lower + math.sqrt(max(0.0, disc))) / C, 0.0, 1.0))
    eta = float(min(eta_plugin, eta_audit_max))
    est_gain = float(2.0 * eta * D_hat - eta * eta * C)
    audit_upper_excess = float(eta * eta * C - 2.0 * eta * D_lower)
    return eta, est_gain, audit_upper_excess


def exact_safe_eta(p, q, eta_requested, eps=EPS):
    eta_requested = float(np.clip(eta_requested, 0.0, 1.0))
    if eta_requested <= 0 or len(p) == 0:
        return 0.0, 0.0
    d = q - p
    def hh(e):
        return prospective_H(p, e * d)
    hreq = hh(eta_requested)
    if hreq <= eps + 1e-12:
        return eta_requested, hreq
    lo, hi = 0.0, eta_requested
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if hh(mid) <= eps:
            lo = mid
        else:
            hi = mid
    return float(lo), hh(lo)


def choose_inner_mix(qv, ql, fit_cohorts, known_fit, clock):
    active = sorted(fit_cohorts)[-BUFFER_WINDOWS:]
    ids, ys, ws = [], [], []
    for origin in active:
        cohort = fit_cohorts[origin]
        for st in cohort["strata"]:
            sample = np.asarray(st["sample_ids"], int)
            if len(sample) == 0:
                continue
            decay = DECAY ** max(0, clock - origin)
            pi = float(st["pi"])
            if pi <= 0:
                continue
            wi = decay / (len(cohort["indices"]) * pi)
            ids.extend(sample.tolist())
            ys.extend([known_fit[int(i)] for i in sample])
            ws.extend([wi] * len(sample))
    if not ids:
        return 0.5, {"past_n": 0, "quadratic_A": 0.0, "linear_B": 0.0}
    ids = np.asarray(ids, int)
    ys = np.asarray(ys, int)
    w = np.asarray(ws, float)
    w /= w.sum()
    a = qv[ids] - ql[ids]
    e = np.eye(qv.shape[1])[ys]
    A = float(np.sum(w * np.sum(a * a, axis=1)))
    B = float(np.sum(w * np.sum(a * (ql[ids] - e), axis=1)))
    lam = float(np.clip(-B / A, 0.0, 1.0)) if A > 1e-18 else 0.5
    return lam, {"past_n": int(len(ids)), "quadratic_A": A, "linear_B": B}


def secondary_metrics(p, y):
    from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
    p = np.asarray(p, float)
    y = np.asarray(y, int)
    pred = np.argmax(p, axis=1)
    out = {
        "brier": float(np.mean(brier_rows(p, y))),
        "nll": float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)))),
        "accuracy": float(np.mean(pred == y)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }
    conf = np.max(p, axis=1)
    corr = (pred == y).astype(float)
    ece = 0.0
    bins = np.linspace(0, 1, 16)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if np.any(m):
            ece += float(np.mean(m) * abs(np.mean(corr[m]) - np.mean(conf[m])))
    out["ece15"] = float(ece)
    k = p.shape[1]
    try:
        if k == 2:
            out["auroc"] = float(roc_auc_score(y, p[:, 1]))
            out["auprc"] = float(average_precision_score(y, p[:, 1]))
        else:
            Y = np.eye(k)[y]
            out["auroc"] = float(roc_auc_score(Y, p, average="macro", multi_class="ovr"))
            out["auprc"] = float(average_precision_score(Y, p, average="macro"))
    except Exception:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def run_stream(data, raw, seed, family, dataset,
               budget_rate=BUDGET_RATE,
               schedule_mode="balanced",
               sample_mode="geometry",
               label_noise=0.0,
               label_dropout=0.0,
               action_mode="riskopt",
               exact_h=True,
               eps=EPS,
               return_predictions=False):
    """Run one fully causal stream. Current canary labels are the only target labels used."""
    import run_dtrar_risk_optimal_final as ro
    core = ro.core

    raw = simplex(raw)
    y = np.asarray(data.y, int)
    must(raw.shape == (len(y), len(data.classes)), f"raw shape mismatch {raw.shape}")
    z, _ = core.representation(data)
    source_fit, _, _ = core.monitoring_roles(data.source, data.y)
    source_y = data.y[source_fit]
    basis = np.column_stack([np.ones(len(z)), raw[:, :-1], z])

    schedule, B = horizon_schedule(data.windows, budget_rate, schedule_mode)
    target = np.concatenate(data.windows)
    pred = raw.copy()
    known_fit = {int(i): int(y[i]) for i in data.source}
    fit_cohorts = {}
    qv = raw.copy()
    ql = raw.copy()
    inner_lambda = 0.5
    rows = []
    requested_total = observed_total = 0

    for t0, ix0 in enumerate(data.windows):
        t = t0 + 1
        ix = np.asarray(ix0, int)
        p = raw[ix]
        yy = y[ix]
        qstar_all = core.simplex(inner_lambda * qv + (1.0 - inner_lambda) * ql)
        q = qstar_all[ix]
        m = int(min(schedule.get(t, 0), max(0, len(ix) - 1)))
        requested_total += m
        rng = np.random.default_rng(stable_seed("final_canary", family, dataset, seed, t))
        strata, can_local, bulk_local = canary_design(p, q, m, rng, sample_mode)
        strata_obs = observed_strata(
            strata, label_dropout,
            np.random.default_rng(stable_seed("final_dropout", family, dataset, seed, t))
        )
        true_map = {int(i): int(yy[i]) for i in can_local}
        obs_local = np.concatenate([st["observed"] for st in strata_obs]) if strata_obs else np.empty(0, int)
        obs_true = np.asarray([true_map[int(i)] for i in obs_local], int) if len(obs_local) else np.empty(0, int)
        obs_y = perturb_labels(
            obs_true, raw.shape[1],
            np.random.default_rng(stable_seed("final_noise", family, dataset, seed, t)),
            label_noise,
        )
        obs_map = {int(i): int(v) for i, v in zip(obs_local, obs_y)}
        observed_total += len(obs_local)

        est = estimate_bulk_D(p, q, obs_map, strata_obs, bulk_local)
        if action_mode == "riskopt" and est["estimable"] and len(obs_local) >= MIN_CANARY:
            eta_pre, est_gain, audit_upper = current_risk_optimal_eta(
                est["D_hat"], est["D_lower"], est["C_bulk"], eps
            )
        elif action_mode == "maxsafe" and len(bulk_local):
            eta_pre, est_gain, audit_upper = 1.0, float("nan"), float("nan")
        else:
            eta_pre = est_gain = audit_upper = 0.0

        if exact_h:
            eta, h_bulk = exact_safe_eta(p[bulk_local], q[bulk_local], eta_pre, eps)
        else:
            eta = float(np.clip(eta_pre, 0.0, 1.0))
            h_bulk = prospective_H(p[bulk_local], eta * (q[bulk_local] - p[bulk_local])) if len(bulk_local) else 0.0

        if eta > 0 and len(bulk_local):
            global_bulk = ix[bulk_local]
            pred[global_bulk] = simplex(raw[global_bulk] + eta * (qstar_all[global_bulk] - raw[global_bulk]))
        global_canary = ix[can_local]
        if len(global_canary):
            must(np.max(np.abs(pred[global_canary] - raw[global_canary])) <= 1e-12,
                 "retroactive canary repair detected")

        h_window = prospective_H(raw[ix], pred[ix] - raw[ix])
        if exact_h:
            must(h_window <= eps + 5e-9, f"exact contract failure {dataset}/seed{seed}/w{t}: {h_window}")

        raw_b = float(np.mean(brier_rows(raw[ix], yy)))
        out_b = float(np.mean(brier_rows(pred[ix], yy)))

        rows.append({
            "family": family, "dataset": dataset, "seed": int(seed), "window": int(t),
            "window_n": int(len(ix)), "canary_requested": int(len(can_local)),
            "canary_observed": int(len(obs_local)), "canary_fraction": float(len(can_local)/len(ix)),
            "schedule_mode": schedule_mode, "sample_mode": sample_mode,
            "label_noise": float(label_noise), "label_dropout": float(label_dropout),
            "candidate_inner_vamr_weight": float(inner_lambda),
            "candidate_inner_logistic_weight": float(1.0-inner_lambda),
            "D_hat_bulk": float(est["D_hat"]), "D_lower_bulk": float(est["D_lower"]),
            "D_radius_bulk": float(est["radius"]), "C_bulk": float(est["C_bulk"]),
            "eta_before_exact_H": float(eta_pre), "eta_deploy": float(eta),
            "act_bulk": bool(eta > 0), "bulk_exact_H": float(h_bulk),
            "whole_window_exact_H": float(h_window), "raw_brier": raw_b,
            "final_brier": out_b, "realized_gain": raw_b-out_b,
            "estimated_gain_at_eta": float(est_gain), "audit_upper_excess": float(audit_upper),
        })

        # Recycle only OBSERVED current canary labels into the NEXT repair candidate.
        if len(obs_local):
            design_fit = []
            for st in strata_obs:
                obs = np.asarray(st["observed"], int)
                if len(obs) == 0:
                    continue
                gobs = ix[obs]
                # Probability of being selected AND observed under MCAR dropout.
                pi = float(st["design_pi"] * (1.0 - float(label_dropout)))
                if label_dropout <= 0:
                    pi = float(st["design_pi"])
                design_fit.append({
                    "stratum": int(st["stratum"]), "population_n": int(st["Nh"]),
                    "sample_ids": np.asarray(gobs, int), "pi": max(pi, 1e-12),
                })
                for li, gi in zip(obs, gobs):
                    known_fit[int(gi)] = int(obs_map[int(li)])
            if design_fit:
                fit_cohorts[t0] = {"indices": np.asarray(ix, int), "strata": design_fit}
                qv, ql, _ = core.fit_repairs(
                    raw, z, basis, source_fit, source_y, fit_cohorts, known_fit, t0
                )
                inner_lambda, _ = choose_inner_mix(qv, ql, fit_cohorts, known_fit, t0)

    final_metrics = secondary_metrics(pred[target], y[target])
    raw_metrics = secondary_metrics(raw[target], y[target])
    summary = {
        "family": family, "dataset": dataset, "seed": int(seed),
        "budget_rate": float(budget_rate), "schedule_mode": schedule_mode,
        "sample_mode": sample_mode, "action_mode": action_mode, "exact_h": bool(exact_h),
        "label_noise": float(label_noise), "label_dropout": float(label_dropout),
        "target_n": int(len(target)), "requested_canary_labels": int(requested_total),
        "predeclared_budget_labels": int(B), "observed_canary_labels": int(observed_total),
        "budget_exact": bool(requested_total == B),
        "mean_act_rate": float(np.mean([r["act_bulk"] for r in rows])),
        "mean_eta_when_acted": float(np.mean([r["eta_deploy"] for r in rows if r["act_bulk"]])) if any(r["act_bulk"] for r in rows) else 0.0,
        "max_exact_H": float(max(r["whole_window_exact_H"] for r in rows)),
    }
    for k, v in raw_metrics.items(): summary[f"raw_{k}"] = v
    for k, v in final_metrics.items(): summary[f"final_{k}"] = v
    summary["brier_gain"] = float(raw_metrics["brier"] - final_metrics["brier"])
    summary["nll_gain"] = float(raw_metrics["nll"] - final_metrics["nll"])
    summary["accuracy_gain"] = float(final_metrics["accuracy"] - raw_metrics["accuracy"])
    summary["macro_f1_gain"] = float(final_metrics["macro_f1"] - raw_metrics["macro_f1"])
    summary["ece_reduction"] = float(raw_metrics["ece15"] - final_metrics["ece15"])
    if return_predictions:
        return summary, pd.DataFrame(rows), pred
    return summary, pd.DataFrame(rows)
