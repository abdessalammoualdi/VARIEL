#!/usr/bin/env python3
"""
DTRA-R + RAPID geometry-aware repairability falsification pilot (frozen v2.2).

This launcher reuses the existing frozen XGBoost caches. It does not retrain
base classifiers and does not repeat the earlier 78/36 comparison batches.

Locked scientific design
------------------------
* Datasets: Electricity, Bank, Gas.
* Seeds: 0, 1, 2.
* Total target-label ceiling: 5%.
* Feedback delay: one window.
* Label roles: 80% repair stream, 20% disjoint conformal-score stream.
* Repair labels are causal-recycled: a label may audit a candidate that was
  frozen before its reveal; only afterward may it help fit a later candidate.
* Inner repair direction: past-label convex combination of VAMR and the
  feature-conditioned logistic repair head, anchored at the raw probabilities.
* RAPID audit: eight deterministic strata formed from label-free repair
  geometry; quotas favor rows that can most shrink repair-certification
  uncertainty and use exact within-stratum inclusion probabilities.
* Exact Brier line geometry: Delta R(eta)=eta^2 C-2 eta D. C is completely
  label-free on the completed audit population; only scalar D requires labels.
* Sequential multiplicity: a 5% family-wise error budget is divided across all
  actionable repair decisions in each dataset/seed run.
* Full repair requires a minimum certified Brier gain of 0.005. A cautious
  partial repair is permitted only when the rigorous upper excess-risk bound
  stays below the predeclared Brier harm cap of 0.005 and estimated gain is at
  least 0.001. Raw probabilities remain the safety anchor.
* Guarantees refer to each completed audited finite population. Performance on
  future deployment windows is evaluated empirically.

Methods exported
----------------
raw; VAMR; feature-logistic repair; DTRA-R strict certificate; DTRA-R + RAPID.

Pilot gates
-----------
Electricity: RAPID minus raw Brier <= +0.01.
Bank:        RAPID improves raw Brier by > 0.01.
Gas:         RAPID improves raw Brier by > 0.03.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import zipfile
import zlib

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parent

# Reuse the already-tested project loader/repair/conformal machinery from the
# previous pilot.  The user ran that pilot immediately before this one, so the
# file should be beside README.md.  Nothing in its main() executes on import.
try:
    import run_dtrar_rapid_pilot as core
except Exception as exc:
    raise SystemExit(
        "ERROR: run_dtrar_rapid_pilot.py must remain in the main DTRA folder "
        "beside this launcher. Import failed: " + repr(exc)
    )

BASE = core.BASE
DATASETS = ["electricity", "bank", "gas"]
SEEDS = [0, 1, 2]
METHODS = ["raw", "vamr", "logistic", "dtrar_strict", "dtrar_rapid"]

BETA = 0.05
DELAY = 1
ALPHA = 0.05
REPAIR_SHARE = 0.80
CP_SHARE = 0.20
MIN_GAIN = 0.005
FAMILY_ALPHA = 0.05
HARM_CAP = 0.005
MIN_CAUTION_GAIN = 0.001
AUDIT_STRATA = 8
SOURCE_MASS = 0.1
DECAY = 0.9
BUFFER_WINDOWS = 3
FAILURE_MARGIN = 0.02


def jdump(path: Path, obj):
    core.jdump(path, obj)


def brier_rows(p, y):
    return core.brier_rows(p, y)


def allocate_roles(windows, ceiling):
    """Global 5% ceiling; only windows whose labels can affect deployment."""
    actionable = max(0, len(windows) - (DELAY + 1))
    caps = np.asarray([len(windows[t]) for t in range(actionable)], int)
    repair_total = int(math.floor(REPAIR_SHARE * ceiling))
    cp_total = ceiling - repair_total
    rq = core.allocate(repair_total, caps) if repair_total else np.zeros(actionable, int)
    cq = core.allocate(cp_total, caps) if cp_total else np.zeros(actionable, int)
    if np.any(rq + cq > caps):
        raise RuntimeError("Role allocation overfilled a window")
    if int(rq.sum() + cq.sum()) != ceiling:
        raise AssertionError("Budget allocation mismatch")
    return actionable, rq, cq, {
        "repair": int(repair_total),
        "cp": int(cp_total),
        "total": int(ceiling),
    }


def largest_remainder_alloc(total, caps, mass, min_each=True):
    caps = np.asarray(caps, int)
    mass = np.asarray(mass, float)
    H = len(caps)
    if total <= 0 or H == 0:
        return np.zeros(H, int)
    base = np.zeros(H, int)
    if min_each and total >= 2 * H:
        base[:] = 2
    elif min_each and total >= H:
        base[:] = 1
    base = np.minimum(base, caps)
    remain = int(total - base.sum())
    avail = caps - base
    if remain <= 0:
        return base
    m = np.where(avail > 0, np.maximum(mass, 0.0), 0.0)
    if m.sum() <= 0:
        m = np.where(avail > 0, caps.astype(float), 0.0)
    ideal = remain * m / m.sum()
    add = np.minimum(np.floor(ideal).astype(int), avail)
    base += add
    remain = int(total - base.sum())
    frac = ideal - np.floor(ideal)
    while remain > 0:
        candidates = np.where(base < caps)[0]
        if len(candidates) == 0:
            raise RuntimeError("Unable to allocate audit quota")
        # largest fractional remainder; break ties by larger mass then index
        j = max(candidates.tolist(), key=lambda q: (frac[q], m[q], -q))
        base[j] += 1
        frac[j] = -1.0
        remain -= 1
    return base


def repair_geometry(raw, qstar, ix):
    """Label-free Brier alignment geometry for a frozen repair direction."""
    ix = np.asarray(ix, int)
    d = qstar[ix] - raw[ix]
    offset = np.sum(raw[ix] * d, axis=1)
    potential = d - offset[:, None]
    lo = potential.min(axis=1)
    hi = potential.max(axis=1)
    center = 0.5 * (lo + hi)
    halfwidth = 0.5 * (hi - lo)
    return potential, center, halfwidth


def geometry_strata(ix, raw, qstar, H=AUDIT_STRATA):
    """Deterministic strata based only on pre-label repair ambiguity."""
    ix = np.asarray(ix, int)
    _, center, halfwidth = repair_geometry(raw, qstar, ix)
    H = int(max(1, min(H, len(ix))))
    order = np.argsort(halfwidth, kind="mergesort")
    blocks = [b for b in np.array_split(order, H) if len(b)]
    out = []
    for h, loc in enumerate(blocks):
        ids = ix[loc]
        hw = halfwidth[loc]
        out.append({
            "stratum": h,
            "population_ids": ids,
            "population_n": int(len(ids)),
            "halfwidth_max": float(hw.max()) if len(hw) else 0.0,
            "halfwidth_mean": float(hw.mean()) if len(hw) else 0.0,
            "center_mean": float(center[loc].mean()) if len(loc) else 0.0,
        })
    return out


def sample_roles(ix, nrepair, ncp, seed, t, raw, qstar):
    """Geometry-aware audit + disjoint CP draw, all fixed before labels."""
    ix = np.asarray(ix, int)
    rng = np.random.default_rng(np.random.SeedSequence([seed, t, 771091]))
    strata = geometry_strata(ix, raw, qstar, AUDIT_STRATA)
    caps = np.asarray([st["population_n"] for st in strata], int)
    # Worst-case Hoeffding contribution is minimized by n_h proportional to
    # W_h * range_h = N_h * (2 h_h); factor 2 is common and omitted.
    mass = np.asarray([
        st["population_n"] * max(st["halfwidth_max"], 1e-12)
        for st in strata
    ], float)
    nh = largest_remainder_alloc(int(nrepair), caps, mass, min_each=True)
    repair_parts = []
    design = []
    for st, take in zip(strata, nh):
        pop = np.asarray(st["population_ids"], int)
        samp = rng.choice(pop, int(take), replace=False) if take else np.empty(0, int)
        repair_parts.append(samp)
        design.append({
            **{k:v for k,v in st.items() if k != "population_ids"},
            "sample_ids": samp,
            "sample_n": int(len(samp)),
            "pi": float(len(samp) / len(pop)) if len(pop) else 0.0,
        })
    repair = np.concatenate(repair_parts) if repair_parts else np.empty(0, int)
    rem = ix[~np.isin(ix, repair)]
    cp = rng.choice(rem, int(ncp), replace=False) if ncp else np.empty(0, int)
    return np.asarray(repair, int), np.asarray(cp, int), design

def choose_inner_mix(qv, ql, fit_cohorts, known_fit, clock):
    """Choose VAMR/logistic mixture from PAST labels only."""
    active = sorted(fit_cohorts)[-BUFFER_WINDOWS:]
    ids, ys, ws = [], [], []
    for origin in active:
        cohort = fit_cohorts[origin]
        for st in cohort["strata"]:
            sample = np.asarray(st["sample_ids"], int)
            if len(sample) == 0:
                continue
            decay = DECAY ** max(0, clock - origin)
            # IPW estimate of a window mean under stratified SRS.
            pi = float(st["pi"])
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
    lam = 0.5 if A <= 1e-14 else float(np.clip(-B / A, 0.0, 1.0))
    return lam, {"past_n": int(len(ids)), "quadratic_A": A, "linear_B": B}

def max_gain_from_D(D, C):
    if C <= 1e-14 or D <= 0:
        return 0.0, 0.0
    eta = float(np.clip(D / C, 0.0, 1.0))
    gain = float(max(0.0, 2.0 * eta * D - eta * eta * C))
    return eta, gain


def risk_capped_eta(D_hat, D_lower, C, harm_cap=HARM_CAP):
    if C <= 1e-14 or D_hat <= 0:
        return 0.0, 0.0, 0.0
    eta_hat = float(np.clip(D_hat / C, 0.0, 1.0))
    disc = D_lower * D_lower + C * max(0.0, harm_cap)
    eta_max = float(np.clip((D_lower + math.sqrt(max(0.0, disc))) / C, 0.0, 1.0))
    eta = min(eta_hat, eta_max)
    est_gain = float(max(0.0, 2.0 * eta * D_hat - eta * eta * C))
    upper_excess = float(eta * eta * C - 2.0 * eta * D_lower)
    return eta, est_gain, upper_excess


def alignment_audit(raw, qstar, pop, sample, known, source_ref_brier, audit_design, audit_alpha):
    """Geometry-centered, stratified finite-population audit.

    For z_i = d_{i,y_i} - p_i^T d_i, all possible z_{i,k} are known before
    labels.  We subtract each row's label-free interval midpoint a_i, so only
    the residual r_i=z_i-a_i must be estimated.  Repair labels are drawn by
    SRS within deterministic strata formed from the row half-width h_i.

    The combined Hoeffding bound is design-valid for independent stratified
    SRS; sampling without replacement is no less concentrated than the
    corresponding with-replacement sample.  No post-label stratum formation or
    allocation occurs.
    """
    pop = np.asarray(pop, int)
    sample = np.asarray(sample, int)
    N, n = len(pop), len(sample)
    d = qstar - raw
    C = float(np.mean(np.sum(d[pop] ** 2, axis=1)))
    if n == 0 or C <= 1e-14:
        return {
            "N": N, "n": n, "C": C, "D_hat": 0.0,
            "strict_lower": 0.0, "strict_upper": 0.0,
            "strict_eta": 0.0, "strict_gain": 0.0,
            "possible_eta_strict": 0.0, "possible_gain_strict": 0.0,
            "riskcap_eta": 0.0, "riskcap_est_gain": 0.0,
            "riskcap_upper_excess": 0.0,
            "raw_audit_brier": source_ref_brier,
            "geometry_radius": 0.0, "geometry_center_mean": 0.0,
        }

    # Global lookup for label-free geometry.
    potential, center, halfwidth = repair_geometry(raw, qstar, pop)
    loc = {int(i): j for j, i in enumerate(pop)}
    center_bar = float(center.mean())

    residual_est = 0.0
    variance_proxy = 0.0
    audit_rows = []
    for st in audit_design:
        ids = np.asarray(st["sample_ids"], int)
        Nh = int(st["population_n"])
        nh = len(ids)
        if Nh <= 0:
            continue
        Wh = Nh / N
        # Because strata are deterministic blocks of pop ids, recover all rows
        # belonging to this stratum by h-order using stored membership bounds is
        # not necessary for estimation: the exact hmax was frozen in design.
        hmax = float(st["halfwidth_max"])
        if nh <= 0:
            raise RuntimeError("Every geometry stratum must receive >=1 audit label")
        ys = np.asarray([known[int(i)] for i in ids], int)
        jj = np.asarray([loc[int(i)] for i in ids], int)
        z = potential[jj, ys]
        r = z - center[jj]
        residual_est += Wh * float(np.mean(r))
        # Hoeffding proxy for W_h * mean(r_h), r_h in [-hmax,hmax].
        range_h = 2.0 * hmax
        variance_proxy += (Wh * Wh) * (range_h * range_h) / nh
        audit_rows.append({
            "stratum": int(st["stratum"]), "population_n": Nh,
            "sample_n": nh, "pi": float(st["pi"]),
            "halfwidth_max": hmax, "residual_mean": float(np.mean(r)),
        })

    D_hat = center_bar + residual_est
    radius = math.sqrt(max(0.0, 0.5 * math.log(2.0 / audit_alpha) * variance_proxy))
    # Logical support from the complete label-free possible-value matrix.
    global_lower = float(potential.min())
    global_upper = float(potential.max())
    strict_lower = max(global_lower, D_hat - radius)
    strict_upper = min(global_upper, D_hat + radius)

    seta, sgain = max_gain_from_D(strict_lower, C)
    speta, spgain = max_gain_from_D(strict_upper, C)
    rc_eta, rc_est_gain, rc_upper_excess = risk_capped_eta(
        D_hat, strict_lower, C, HARM_CAP
    )
    ys_all = np.asarray([known[int(i)] for i in sample], int)
    raw_audit = float(np.mean(brier_rows(raw[sample], ys_all)))
    return {
        "N": N, "n": n, "C": C, "D_hat": float(D_hat),
        "strict_lower": float(strict_lower), "strict_upper": float(strict_upper),
        "strict_eta": seta, "strict_gain": sgain,
        "possible_eta_strict": speta, "possible_gain_strict": spgain,
        "riskcap_eta": rc_eta, "riskcap_est_gain": rc_est_gain,
        "riskcap_upper_excess": rc_upper_excess,
        "raw_audit_brier": raw_audit,
        "geometry_radius": float(radius),
        "geometry_center_mean": center_bar,
        "audit_strata": audit_rows,
    }

def oracle_alignment(raw, qstar, pop, y):
    pop = np.asarray(pop, int)
    d = qstar - raw
    C = float(np.mean(np.sum(d[pop] ** 2, axis=1)))
    z = (
        d[pop, y[pop]]
        - np.sum(raw[pop] * d[pop], axis=1)
    )
    D = float(np.mean(z))
    eta, gain = max_gain_from_D(D, C)
    return {"oracle_D": D, "oracle_eta": eta, "oracle_gain": gain}


def source_thresholds(*args, **kwargs):
    return core.source_thresholds(*args, **kwargs)


def update_cp_thresholds(*args, **kwargs):
    return core.update_cp_thresholds(*args, **kwargs)


def run_one(dataset, seed, out):
    data = core.load_dataset(dataset, BASE / "data" / "raw")
    hashes = core.source_hashes(BASE / "data" / "raw", dataset)
    cache, raw = core.frozen_cache(
        BASE / "artifacts", data, "xgboost", seed, "primary", hashes
    )
    z, _ = core.representation(data)
    source_fit, source_score, source_monitor = core.monitoring_roles(data.source, data.y)
    source_y = data.y[source_fit]
    uniforms = np.random.default_rng(
        np.random.SeedSequence([seed, 29073])
    ).random(len(raw))
    source_cutoff, source_true, source_score_y = source_thresholds(
        raw, data.y, source_score, uniforms, ALPHA
    )
    source_ref_brier = float(
        np.mean(brier_rows(raw[source_monitor], data.y[source_monitor]))
    )
    basis = np.column_stack([np.ones(len(z)), raw[:, :-1], z])

    target = np.concatenate(data.windows)
    ceiling = math.floor(BETA * len(target))
    actionable, repairq, cpq, role_totals = allocate_roles(data.windows, ceiling)

    oracle = core.DelayedOracle(
        data.y, raw.shape[1], seed=np.random.SeedSequence([seed, 990731])
    )
    known_fit = {int(i): int(data.y[i]) for i in data.source}
    known_repair = {}
    known_cp = {}
    fit_cohorts = {}
    cp_cohorts = {}
    pending = {}

    # Shadow repair model: built from labels that have already arrived.
    qv = raw.copy()
    ql = raw.copy()
    inner_lambda = 0.5

    # Deployed versions are exact snapshots of candidates that were audited
    # after being frozen.  They are not silently replaced by a newly refit head.
    deploy_rapid_q = raw.copy()
    deploy_rapid_eta = 0.0
    deploy_strict_q = raw.copy()
    deploy_strict_eta = 0.0

    thresholds = {m: source_cutoff.copy() for m in METHODS}
    issued_p = {m: np.zeros((len(target), raw.shape[1])) for m in METHODS}
    issued_s = {m: np.zeros((len(target), raw.shape[1]), bool) for m in METHODS}
    query_records = []
    decisions = []
    pos = 0

    for t, ix in enumerate(data.windows):
        nwin = len(ix)

        # 1) Issue current predictions BEFORE labels due after this window arrive.
        rapid_all = core.simplex(
            raw + deploy_rapid_eta * (deploy_rapid_q - raw)
        )
        strict_all = core.simplex(
            raw + deploy_strict_eta * (deploy_strict_q - raw)
        )
        prob_now = {
            "raw": raw[ix],
            "vamr": qv[ix],
            "logistic": ql[ix],
            "dtrar_strict": strict_all[ix],
            "dtrar_rapid": rapid_all[ix],
        }
        for m in METHODS:
            issued_p[m][pos:pos+nwin] = prob_now[m]
            scores = core.candidate_scores(prob_now[m], uniforms[ix])
            issued_s[m][pos:pos+nwin] = scores <= thresholds[m][None, :]

        # 2) Labels requested in earlier windows now arrive.  First audit the
        # candidate snapshot that PRE-DATED those labels.  Only then recycle the
        # same repair labels into training for the next candidate.
        due = oracle.release(t)
        if due:
            origins_due = sorted(set(int(job["origin"]) for job in due))

            # Load labels into role-specific dictionaries.
            for job in due:
                origin = int(job["origin"])
                if job["kind"] == "repair":
                    for i, yy in zip(job["indices"], job["labels"]):
                        known_repair[int(i)] = int(yy)
                elif job["kind"] == "cp":
                    for i, yy in zip(job["indices"], job["labels"]):
                        known_cp[int(i)] = int(yy)
                else:
                    raise RuntimeError("Unexpected role " + str(job["kind"]))

            # Audit each due frozen candidate and update deployment state.
            for origin in origins_due:
                rec = pending[origin]
                rids = np.asarray(rec["repair_ids"], int)

                if len(rids):
                    audit = alignment_audit(
                        raw, rec["candidate_qstar"], rec["population_ids"],
                        rids, known_repair, source_ref_brier, rec["audit_design"], rec["audit_alpha"]
                    )
                    oracle_diag = oracle_alignment(
                        raw, rec["candidate_qstar"], rec["population_ids"], data.y
                    )
                    audit.update(oracle_diag)

                    if audit["strict_gain"] > MIN_GAIN:
                        rapid_decision = "CERTIFIED_REPAIR"
                        deploy_rapid_q = rec["candidate_qstar"].copy()
                        deploy_rapid_eta = float(audit["strict_eta"])
                    elif audit["riskcap_est_gain"] >= MIN_CAUTION_GAIN and audit["riskcap_eta"] > 0:
                        rapid_decision = "CAUTIOUS_REPAIR"
                        deploy_rapid_q = rec["candidate_qstar"].copy()
                        deploy_rapid_eta = float(audit["riskcap_eta"])
                    elif (
                        audit["possible_gain_strict"] <= MIN_GAIN
                        and audit["raw_audit_brier"] > source_ref_brier + FAILURE_MARGIN
                    ):
                        rapid_decision = "ESCALATE"
                        deploy_rapid_q = raw.copy()
                        deploy_rapid_eta = 0.0
                    else:
                        rapid_decision = "HOLD"
                        deploy_rapid_q = raw.copy()
                        deploy_rapid_eta = 0.0

                    if audit["strict_gain"] > MIN_GAIN:
                        strict_decision = "REPAIR"
                        deploy_strict_q = rec["candidate_qstar"].copy()
                        deploy_strict_eta = float(audit["strict_eta"])
                    else:
                        strict_decision = "HOLD"
                        deploy_strict_q = raw.copy()
                        deploy_strict_eta = 0.0

                    # AFTER auditing, recycle these random labels into the next
                    # candidate's repair fit.  Same-version fit/cert reuse never
                    # occurs.
                    fit_cohorts[origin] = {
                        "indices": np.asarray(rec["population_ids"], int),
                        "strata": [
                            {
                                "stratum": int(st["stratum"]),
                                "population_n": int(st["population_n"]),
                                "sample_ids": np.asarray(st["sample_ids"], int),
                                "pi": float(st["pi"]),
                            }
                            for st in rec["audit_design"]
                        ],
                    }
                    for i in rids:
                        known_fit[int(i)] = known_repair[int(i)]

                    decisions.append({
                        "dataset": dataset,
                        "seed": seed,
                        "audited_origin_window": origin + 1,
                        "labels_arrived_after_window": t + 1,
                        "rapid_effective_from_window": t + 2,
                        "rapid_decision": rapid_decision,
                        "strict_decision": strict_decision,
                        "candidate_inner_vamr_weight": float(
                            rec["candidate_inner_lambda"]
                        ),
                        "candidate_inner_logistic_weight": float(
                            1.0 - rec["candidate_inner_lambda"]
                        ),
                        **audit,
                    })

                # CP labels remain disjoint and update only the conformal score layer.
                cpids = np.asarray(rec["cp_ids"], int)
                if len(cpids):
                    ys = np.asarray([known_cp[int(i)] for i in cpids], int)
                    cp_cohorts[origin] = {
                        "cp_ids": cpids,
                        "issued_true_scores": {},
                    }
                    for m in METHODS:
                        matrix = np.asarray(rec["issued_cp_scores"][m])
                        cp_cohorts[origin]["issued_true_scores"][m] = (
                            matrix[np.arange(len(cpids)), ys]
                        )

            # Fit NEXT shadow heads only after all due candidates were audited.
            qv, ql, fit_info = core.fit_repairs(
                raw, z, basis, source_fit, source_y,
                fit_cohorts, known_fit, t
            )
            inner_lambda, mix_info = choose_inner_mix(
                qv, ql, fit_cohorts, known_fit, t
            )

            for m in METHODS:
                thresholds[m] = update_cp_thresholds(
                    m, source_true, source_score_y, source_cutoff,
                    cp_cohorts, known_cp, t, raw.shape[1]
                )
        else:
            fit_info = {"target_effective_n": 0.0}
            mix_info = {"past_n": 0}

        # 3) At the end of the completed current window, propose a NEW shadow
        # candidate using only labels already available from the past.  Freeze
        # that exact candidate BEFORE requesting current-window repair labels.
        if t < actionable:
            candidate_qstar = core.simplex(
                inner_lambda * qv + (1.0 - inner_lambda) * ql
            )
            repair_ids, cp_ids, audit_design = sample_roles(
                ix, int(repairq[t]), int(cpq[t]), seed, t, raw, candidate_qstar
            )

            rec = {
                "population_ids": np.asarray(ix, int),
                "repair_ids": repair_ids,
                "cp_ids": cp_ids,
                "candidate_qstar": candidate_qstar.astype(np.float32),
                "candidate_inner_lambda": float(inner_lambda),
                "audit_design": audit_design,
                "audit_alpha": float(FAMILY_ALPHA / max(1, actionable)),
                "issued_cp_scores": {},
            }

            loc = {int(r): j for j, r in enumerate(ix)}
            cp_pos = np.asarray([loc[int(r)] for r in cp_ids], int) if len(cp_ids) else np.empty(0, int)
            for m in METHODS:
                if len(cp_ids):
                    sc = core.candidate_scores(prob_now[m], uniforms[ix])
                    rec["issued_cp_scores"][m] = sc[cp_pos]
                else:
                    rec["issued_cp_scores"][m] = np.empty(
                        (0, raw.shape[1]), float
                    )
            pending[t] = rec

            if len(repair_ids):
                pi_map = {}
                for st in audit_design:
                    for i in st["sample_ids"]:
                        pi_map[int(i)] = float(st["pi"])
                pis = np.asarray([pi_map[int(i)] for i in repair_ids], float)
                oracle.request(repair_ids, "repair", t, t, DELAY, pis)
                query_records.extend({
                    "row_id": int(i),
                    "kind": "repair",
                    "origin_window": t + 1,
                    "request_after_window": t + 1,
                    "arrival_after_window": t + DELAY + 1,
                    "inclusion_probability": float(pi_map[int(i)]),
                } for i in repair_ids)

            if len(cp_ids):
                denom = len(ix) - len(repair_ids)
                pi = float(len(cp_ids) / denom)
                oracle.request(
                    cp_ids, "cp", t, t, DELAY,
                    np.full(len(cp_ids), pi)
                )
                query_records.extend({
                    "row_id": int(i),
                    "kind": "cp",
                    "origin_window": t + 1,
                    "request_after_window": t + 1,
                    "arrival_after_window": t + DELAY + 1,
                    "conditional_inclusion_probability": pi,
                } for i in cp_ids)

        pos += nwin

    if pos != len(target):
        raise AssertionError("Target traversal mismatch")

    qids = [r["row_id"] for r in query_records]
    if len(qids) != len(set(qids)):
        raise AssertionError("Role overlap detected")
    if len(qids) > ceiling:
        raise AssertionError("Global label budget exceeded")

    job = out / "jobs" / f"{dataset}_seed{seed}"
    job.mkdir(parents=True, exist_ok=True)
    summaries = []
    for m in METHODS:
        metrics = core.evaluate(
            issued_p[m], data.y[target], issued_s[m]
        )
        row = {
            "dataset": dataset,
            "seed": seed,
            "method": m,
            "model": "xgboost",
            "budget_fraction": BETA,
            "delay": DELAY,
            "alpha": ALPHA,
            "labels_requested": int(len(qids)),
            "budget_ceiling": int(ceiling),
            "repair_labels_requested": int(
                sum(r["kind"] == "repair" for r in query_records)
            ),
            "cp_labels_requested": int(
                sum(r["kind"] == "cp" for r in query_records)
            ),
            "base_model_retrained": False,
            **metrics,
        }
        summaries.append(row)
        np.savez_compressed(
            job / f"issued_{m}.npz",
            row_id=target,
            probabilities=issued_p[m],
            prediction_set=issued_s[m],
        )

    np.savez_compressed(
        job / "evaluation_labels.npz", row_id=target, y=data.y[target]
    )
    jdump(job / "summaries.json", summaries)
    jdump(job / "queries.json", query_records)
    jdump(job / "decision_trace.json", decisions)
    jdump(job / "job_record.json", {
        "dataset": dataset,
        "seed": seed,
        "frozen_cache": str(cache),
        "role_totals": role_totals,
        "actionable_origin_windows": int(actionable),
        "strict_interval": (
            "geometry-centered stratified Hoeffding finite-population audit; "
            "strata and allocation fixed from pre-label repair geometry"
        ),
        "risk_cap": HARM_CAP,
        "guarantee_scope": (
            "Strict interval concerns only each completed audited finite "
            "population. Future-window performance is empirical."
        ),
        "minimum_worthwhile_brier_gain": MIN_GAIN,
        "source_reference_brier": source_ref_brier,
    })

    smap = {r["method"]: r for r in summaries}
    print("RAPID_GEOMETRY_JOB_RESULT " + json.dumps({
        "dataset": dataset,
        "seed": seed,
        "raw_brier": smap["raw"]["brier_sum_over_classes"],
        "rapid_brier": smap["dtrar_rapid"]["brier_sum_over_classes"],
        "strict_brier": smap["dtrar_strict"]["brier_sum_over_classes"],
        "labels": len(qids),
        "rapid_decisions": [d["rapid_decision"] for d in decisions],
        "strict_repairs": int(sum(d["strict_decision"] == "REPAIR" for d in decisions)),
    }), flush=True)
    return summaries, decisions


def diagnostics(out, all_summaries, all_decisions):
    df = pd.DataFrame(all_summaries)
    df.to_csv(out / "all_summaries.csv", index=False)
    jdump(out / "all_summaries.json", all_summaries)
    dd = pd.DataFrame(all_decisions)
    dd.to_csv(out / "decision_trace.csv", index=False)
    jdump(out / "decision_trace.json", all_decisions)

    metrics = [
        "brier_sum_over_classes", "coverage",
        "worst_present_class_coverage", "set_size", "aurc_discrete",
    ]
    grouped = df.groupby(["dataset", "method"])[metrics].agg(["mean", "std"])
    grouped.to_csv(out / "mean_sd.csv")

    assessment = {}
    piv = df.pivot_table(
        index=["dataset", "seed"], columns="method",
        values="brier_sum_over_classes"
    )
    for ds in DATASETS:
        d = piv.loc[ds]
        assessment[ds] = {
            "raw_mean": float(d["raw"].mean()),
            "rapid_mean": float(d["dtrar_rapid"].mean()),
            "strict_mean": float(d["dtrar_strict"].mean()),
            "vamr_mean": float(d["vamr"].mean()),
            "logistic_mean": float(d["logistic"].mean()),
            "rapid_minus_raw": float(
                d["dtrar_rapid"].mean() - d["raw"].mean()
            ),
        }

    assessment["gates"] = {
        "electricity_protection": (
            assessment["electricity"]["rapid_minus_raw"] <= 0.01
        ),
        "bank_improvement": (
            assessment["bank"]["rapid_minus_raw"] < -0.01
        ),
        "gas_improvement": (
            assessment["gas"]["rapid_minus_raw"] < -0.03
        ),
    }
    assessment["pilot_pass"] = bool(all(assessment["gates"].values()))
    jdump(out / "PILOT_ASSESSMENT.json", assessment)

    # Vector PDF evidence + exact source CSVs.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figdir = out / "figures"
        figdir.mkdir(exist_ok=True)
        labels = {
            "raw": "Raw",
            "vamr": "VAMR",
            "logistic": "Feature repair",
            "dtrar_strict": "DTRA-R strict",
            "dtrar_rapid": "DTRA-R + RAPID",
        }
        for ds in DATASETS:
            sub = df[df.dataset == ds]
            vals = [
                sub[sub.method == m].brier_sum_over_classes.mean()
                for m in METHODS
            ]
            errs = [
                sub[sub.method == m].brier_sum_over_classes.std(ddof=1)
                for m in METHODS
            ]
            fig, ax = plt.subplots(figsize=(8.4, 4.8), layout="constrained")
            x = np.arange(len(METHODS))
            ax.errorbar(
                x, vals, yerr=errs, marker="o", capsize=4, linewidth=1.5
            )
            ax.set_xticks(
                x, [labels[m] for m in METHODS], rotation=15
            )
            ax.set_ylabel("Brier loss (lower is better)")
            ax.set_title(f"Repair-alignment falsification pilot — {ds}")
            ax.grid(alpha=0.2)
            fig.savefig(
                figdir / f"{ds}_alignment_brier.pdf",
                bbox_inches="tight"
            )
            plt.close(fig)

        if not dd.empty:
            for (ds, seed), sub in dd.groupby(["dataset", "seed"]):
                sub = sub.sort_values("rapid_effective_from_window")
                fig, ax = plt.subplots(
                    figsize=(8.4, 4.8), layout="constrained"
                )
                x = sub["rapid_effective_from_window"].to_numpy()
                ax.plot(
                    x, sub["riskcap_est_gain"].to_numpy(),
                    marker="o", label="RAPID estimated gain"
                )
                ax.plot(
                    x, sub["strict_gain"].to_numpy(),
                    marker="s", label="Strict certified gain"
                )
                ax.plot(
                    x, sub["oracle_gain"].to_numpy(),
                    marker="^", label="Oracle completed-population gain"
                )
                ax.axhline(
                    MIN_GAIN, linestyle="--", linewidth=1,
                    label="Worthwhile-gain threshold"
                )
                ax.set_xlabel("Effective deployment window")
                ax.set_ylabel("Brier improvement")
                ax.set_title(
                    f"RAPID scalar repairability — {ds}, seed {seed}"
                )
                ax.legend(frameon=False)
                ax.grid(alpha=0.2)
                fig.savefig(
                    figdir / f"{ds}_seed{seed}_alignment_timeline.pdf",
                    bbox_inches="tight"
                )
                plt.close(fig)
    except Exception as exc:
        (out / "FIGURE_WARNING.txt").write_text(repr(exc))

    return assessment


def export_review(out, project_root):
    paths = [
        p for p in out.rglob("*")
        if p.is_file() and "review" not in p.relative_to(out).parts
    ]
    groups, part, size = [], [], 120_000
    for p in sorted(paths):
        cost = len(zlib.compress(p.read_bytes(), 6)) + 2000
        if cost > 7 * 2**20:
            raise RuntimeError("Unexpected oversized review input: " + str(p))
        if part and size + cost > 7 * 2**20:
            groups.append(part)
            part, size = [], 120_000
        part.append(p)
        size += cost
    if part:
        groups.append(part)

    review = out / "review"
    review.mkdir(exist_ok=True)
    total = len(groups)
    for i, group in enumerate(groups, 1):
        dest = review / (
            f"DTRAR_RAPID_Final_Review_{i:02d}_of_{total:02d}.zip"
        )
        tmp = dest.with_suffix(".zip.partial")
        manifest = []
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in group:
                b = p.read_bytes()
                rel = str(p.relative_to(out))
                zf.writestr(rel, b)
                manifest.append({
                    "path": rel,
                    "bytes": len(b),
                    "sha256": hashlib.sha256(b).hexdigest(),
                })
            zf.writestr(
                "PART_MANIFEST.json",
                json.dumps(
                    {"part": i, "parts": total, "files": manifest},
                    indent=2
                )
            )
        with zipfile.ZipFile(tmp) as zf:
            if zf.testzip() is not None:
                raise RuntimeError("Review ZIP CRC failure")
        tmp.replace(dest)
        shutil.copy2(dest, project_root / dest.name)
        print(
            "RAPID_GEOMETRY_REVIEW_PART_READY " + str(project_root / dest.name),
            flush=True
        )
    return total


def self_test():
    # Exact Brier line geometry on a tiny binary population.
    p = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    q = np.array([[0.7, 0.3], [0.2, 0.8], [0.4, 0.6]])
    y = np.array([0, 1, 1])
    d = q - p
    C = float(np.mean(np.sum(d * d, axis=1)))
    D = float(np.mean(
        d[np.arange(len(y)), y] - np.sum(p * d, axis=1)
    ))
    eta = 0.37
    lhs = float(np.mean(brier_rows(p + eta * d, y) - brier_rows(p, y)))
    rhs = eta * eta * C - 2.0 * eta * D
    assert abs(lhs - rhs) < 1e-12

    # Label recycling is version-safe only when the candidate predates labels;
    # this is enforced structurally by pending candidate snapshots.
    assert abs(REPAIR_SHARE + CP_SHARE - 1.0) < 1e-12

    # Gain maximizer agrees with analytic line solution.
    e, g = max_gain_from_D(0.1, 0.2)
    assert abs(e - 0.5) < 1e-12 and abs(g - 0.05) < 1e-12
    print("RAPID_GEOMETRY_SELF_TEST_COMPLETE checks=3", flush=True)


def main():
    self_test()

    for required in [
        BASE / "configs" / "raw_manifest.json",
        BASE / "artifacts" / "models",
    ]:
        if not required.exists():
            raise SystemExit(
                "ERROR: required project input missing: " + str(required)
            )

    identity = {
        "version": "DTRA-RAPID-geometry-frozen-v2.2",
        "datasets": DATASETS,
        "seeds": SEEDS,
        "methods": METHODS,
        "beta": BETA,
        "delay": DELAY,
        "alpha": ALPHA,
        "repair_share": REPAIR_SHARE,
        "cp_share": CP_SHARE,
        "minimum_worthwhile_brier_gain": MIN_GAIN,
        "familywise_alpha": FAMILY_ALPHA,
        "harm_cap": HARM_CAP,
        "candidate_rule": (
            "past-label VAMR/logistic convex repair direction; current "
            "geometry-stratified repair labels audit the frozen direction "
            "before they may train a later version"
        ),
        "base_code": {
            str(p.relative_to(BASE)): core.file_hash(p)
            for folder in ["dtra", "vamr"]
            for p in (BASE / folder).rglob("*.py")
        },
        "runner_sha256": core.file_hash(Path(__file__)),
    }
    sig = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()[:12]

    root = PROJECT / "artifacts" / "dtrar_rapid_geometry_final"
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "RUNNER.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(
            "DTRA-R RAPID final pilot is already running."
        )

    out = root / sig
    out.mkdir(exist_ok=True)
    (root / "LATEST.txt").write_text(str(out) + "\n")
    jdump(out / "LOCKED_PLAN.json", identity)
    jdump(out / "ENVIRONMENT.json", {
        "python": sys.version,
        "platform": platform.platform(),
    })
    shutil.copy2(
        Path(__file__), out / "run_dtrar_rapid_geometry_final.py"
    )

    print(
        "DTRAR_RAPID_FINAL_BEGIN jobs=9 "
        "base_model_retrained=false old_pilot_repeated=false",
        flush=True
    )

    all_summaries, all_decisions = [], []
    with threadpool_limits(limits=1):
        for ds in DATASETS:
            for seed in SEEDS:
                s, d = run_one(ds, seed, out)
                all_summaries.extend(s)
                all_decisions.extend(d)

    assessment = diagnostics(out, all_summaries, all_decisions)
    parts = export_review(out, PROJECT)
    print(
        "RAPID_GEOMETRY_ASSESSMENT "
        + json.dumps(assessment["gates"]),
        flush=True
    )
    print(
        f"DTRAR_RAPID_FINAL_COMPLETE jobs=9 "
        f"method_rows={len(all_summaries)} review_parts={parts} "
        f"pass={str(assessment['pilot_pass']).lower()}",
        flush=True
    )


if __name__ == "__main__":
    main()
