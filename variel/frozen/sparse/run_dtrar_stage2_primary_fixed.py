#!/usr/bin/env python3
"""
DTRA-R + RAPID Stage 2: frozen primary confirmation + untouched fourth stream.

What this launcher does
-----------------------
1) Verifies that the exact DTRA-R + RAPID v2.2 runner used in the passing
   falsification pilot is still present and unchanged.
2) Keeps all DTRA-R/RAPID scientific parameters frozen.
3) Adds one untouched real stream: INSECTS abrupt-balanced (52,848 rows,
   33 features, 6 classes), preserving published stream order.
4) Runs the frozen method with XGBoost for seeds 0..9 on:
      Electricity, Bank, Gas, INSECTS.
5) Reuses existing completed XGBoost caches for the first three datasets.
   Missing INSECTS XGBoost caches are trained from source data only.
6) Produces paired statistics, external-confirmation diagnostics, vector-PDF
   figures and a compact review package.

This stage DOES NOT implement or tune MEBTR. MEBTR is reserved for the
contemporary-comparator stage after this untouched confirmation run.

Scientific lock
---------------
DTRA-R/RAPID version: DTRA-RAPID-geometry-frozen-v2.2
Expected runner SHA256:
46d8b77626030995ef84da03dc78a13665cb50db7de9579b2e52af4e59e97a1f

No parameter may be changed in response to INSECTS results.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import urllib.request
import zipfile
import zlib

import numpy as np
import pandas as pd
from scipy.io import arff
from scipy.stats import wilcoxon
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parent
FINAL_RUNNER = PROJECT / "run_dtrar_rapid_geometry_final.py"
EXPECTED_FINAL_SHA = "46d8b77626030995ef84da03dc78a13665cb50db7de9579b2e52af4e59e97a1f"

if not FINAL_RUNNER.exists():
    raise SystemExit(
        "ERROR: run_dtrar_rapid_geometry_final.py is missing. "
        "Keep it beside README.md and this Stage-2 launcher."
    )

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

actual_final_sha = sha256(FINAL_RUNNER)
if actual_final_sha != EXPECTED_FINAL_SHA:
    raise SystemExit(
        "ERROR: the frozen DTRA-R/RAPID runner has changed.\n"
        f"expected={EXPECTED_FINAL_SHA}\nactual={actual_final_sha}\n"
        "Do not continue until the frozen runner is restored."
    )

import run_dtrar_rapid_geometry_final as final
from dtra.calibration import probabilities as normalize_probabilities

BASE = final.BASE
DATASETS = ["electricity", "bank", "gas", "insects"]
SEEDS = list(range(10))
METHODS = final.METHODS
INSECTS_FILENAME = "INSECTS-abrupt_balanced_norm.arff"
INSECTS_CSV = "INSECTS-abrupt_balanced_norm.csv"

# Primary original source, plus an independently hosted mirror fallback.
INSECTS_URLS = [
    (
        "arff",
        "https://sites.labic.icmc.usp.br/vsouza/repository/creme/"
        "INSECTS-abrupt_balanced_norm.arff"
    ),
    (
        "csv",
        "https://raw.githubusercontent.com/durga256/OnlineLearning_ML/"
        "main/INSECTS-abrupt_balanced_norm.csv"
    ),
]

EXPECTED_INSECTS_ROWS = 52848
EXPECTED_INSECTS_FEATURES = 33
EXPECTED_INSECTS_CLASSES = 6

RAW_DIR = BASE / "data" / "raw"
STAGE_ROOT = PROJECT / "artifacts" / "dtrar_stage2_primary"
MODEL_ROOT = PROJECT / "artifacts" / "dtrar_stage2_models"

_orig_load_dataset = final.core.load_dataset
_orig_source_hashes = final.core.source_hashes
_orig_frozen_cache = final.core.frozen_cache


def jdump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=final.core.json_default))


def download_insects():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    arff_path = RAW_DIR / INSECTS_FILENAME
    csv_path = RAW_DIR / INSECTS_CSV
    if arff_path.exists() and arff_path.stat().st_size > 1_000_000:
        return arff_path, INSECTS_URLS[0][1]
    if csv_path.exists() and csv_path.stat().st_size > 1_000_000:
        return csv_path, INSECTS_URLS[1][1]

    errors = []
    for kind, url in INSECTS_URLS:
        dest = arff_path if kind == "arff" else csv_path
        tmp = dest.with_suffix(dest.suffix + ".partial")
        try:
            print(f"INSECTS_DOWNLOAD_BEGIN source={kind} url={url}", flush=True)
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 DTRA-R-research/1.0"},
            )
            with urllib.request.urlopen(req, timeout=90) as r, tmp.open("wb") as f:
                shutil.copyfileobj(r, f)
            if tmp.stat().st_size < 1_000_000:
                raise RuntimeError(f"download unexpectedly small: {tmp.stat().st_size} bytes")
            tmp.replace(dest)
            print(
                f"INSECTS_DOWNLOAD_COMPLETE source={kind} bytes={dest.stat().st_size}",
                flush=True,
            )
            return dest, url
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            errors.append(f"{url}: {type(exc).__name__}: {exc}")

    raise SystemExit(
        "ERROR: automatic INSECTS download failed.\n"
        + "\n".join(errors)
        + "\nDownload INSECTS-abrupt_balanced_norm.arff manually into "
          "offset_check_v07/data/raw/ and rerun this same command."
    )


def _decode_series(s):
    if getattr(s.dtype, "kind", "") in {"S", "O"}:
        return s.map(lambda v: v.decode("utf8") if isinstance(v, (bytes, bytearray)) else str(v))
    return s.astype(str)


def insects_frame(path: Path):
    if path.suffix.lower() == ".arff":
        records, _ = arff.loadarff(path)
        frame = pd.DataFrame(records)
    else:
        frame = pd.read_csv(path)
        # Some mirrors may write an unnamed pandas index. Never silently keep it.
        unnamed = [c for c in frame.columns if str(c).lower().startswith("unnamed")]
        if unnamed and frame.shape[1] - len(unnamed) == EXPECTED_INSECTS_FEATURES + 1:
            frame = frame.drop(columns=unnamed)

    if len(frame) != EXPECTED_INSECTS_ROWS:
        raise ValueError(
            f"INSECTS row-count mismatch: got {len(frame)}, "
            f"expected {EXPECTED_INSECTS_ROWS}"
        )
    if frame.shape[1] != EXPECTED_INSECTS_FEATURES + 1:
        raise ValueError(
            f"INSECTS column-count mismatch: got {frame.shape[1]}, "
            f"expected {EXPECTED_INSECTS_FEATURES + 1}"
        )

    # Published files place the class in the final column; ARFF commonly names it class.
    target_col = "class" if "class" in frame.columns else frame.columns[-1]
    labels = _decode_series(frame[target_col])
    X = frame.drop(columns=[target_col]).copy()

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="raise")
    if not np.isfinite(X.to_numpy(float)).all():
        raise ValueError("INSECTS contains non-finite predictor values")

    class_values = sorted(labels.unique().tolist())
    if len(class_values) != EXPECTED_INSECTS_CLASSES:
        raise ValueError(
            f"INSECTS class-count mismatch: got {len(class_values)}, "
            f"expected {EXPECTED_INSECTS_CLASSES}"
        )
    mapping = {v: i for i, v in enumerate(class_values)}
    y = labels.map(mapping).to_numpy(dtype=np.int64)
    return X, y, class_values


def load_dataset_ext(name, root, sensitivity=False):
    if name != "insects":
        return _orig_load_dataset(name, root, sensitivity=sensitivity)

    path = RAW_DIR / INSECTS_FILENAME
    if not path.exists():
        path = RAW_DIR / INSECTS_CSV
    X, y, class_values = insects_frame(path)
    n = len(y)
    train_end = math.floor(0.50 * n)
    source_end = math.floor(0.65 * n)
    train = np.arange(train_end)
    source = np.arange(train_end, source_end)
    windows = list(np.array_split(np.arange(source_end, n), 12))

    Dataset = type(_orig_load_dataset("electricity", RAW_DIR))
    return Dataset(
        "insects",
        X,
        y,
        class_values,
        [],
        train,
        source,
        windows,
        {
            "ordering": (
                "published INSECTS abrupt-balanced stream order; "
                "temperature-driven drift order is preserved"
            ),
            "variant": "abrupt_balanced",
            "external_confirmation": True,
            "method_tuning_permitted": False,
        },
    )


def source_hashes_ext(data_dir, name):
    if name != "insects":
        return _orig_source_hashes(data_dir, name)
    p = RAW_DIR / INSECTS_FILENAME
    if not p.exists():
        p = RAW_DIR / INSECTS_CSV
    return {p.name: sha256(p)}


def _train_or_load_insects(data, family, seed, profile, hashes):
    if family != "xgboost":
        raise ValueError("Stage 2 trains only XGBoost for the untouched INSECTS confirmation")
    folder = MODEL_ROOT / "insects" / family / f"seed_{seed}_{profile}"
    complete = folder / "COMPLETE.json"
    probs_path = folder / "frozen_probabilities.npz"

    if complete.exists() and probs_path.exists():
        rec = json.loads(complete.read_text())
        if rec.get("data_sha256") == hashes and rec.get("frozen_runner_sha256") == EXPECTED_FINAL_SHA:
            with np.load(probs_path) as z:
                if np.array_equal(z["row_id"], np.arange(len(data.y))):
                    p = normalize_probabilities(z["probabilities"])
                    print(
                        f"INSECTS_FROZEN_CACHE_REUSED seed={seed} cache={folder.name}",
                        flush=True,
                    )
                    return folder, p

    from dtra.models import fit_tree

    cfg = json.loads((BASE / "configs" / "primary.json").read_text())
    folder.mkdir(parents=True, exist_ok=True)

    # The original Stage-2 attempt could crash immediately after fit_tree()
    # returned, leaving a valid frozen.ubj + preprocessor.joblib but no NPZ.
    # Reconstruct predictions from that completed model instead of retraining.
    frozen_model = folder / "frozen.ubj"
    preproc_path = folder / "preprocessor.joblib"
    training_path = folder / "training.json"
    if frozen_model.exists() and preproc_path.exists() and training_path.exists():
        try:
            import joblib
            from xgboost import XGBClassifier
            model = XGBClassifier()
            model.load_model(str(frozen_model))
            encoding = joblib.load(preproc_path)
            p = model.predict_proba(encoding.transform(data.X))
            p = normalize_probabilities(p)
            info = json.loads(training_path.read_text())
            print(
                f"INSECTS_PARTIAL_CACHE_RECOVERED seed={seed} "
                f"cache={folder.name}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"INSECTS_PARTIAL_CACHE_RECOVERY_FAILED seed={seed} "
                f"reason={type(exc).__name__}:{exc}",
                flush=True,
            )
            print(f"INSECTS_TRAIN_BEGIN model=xgboost seed={seed}", flush=True)
            p, info = fit_tree(data, "xgboost", seed, cfg, folder, threads=8)
            p = normalize_probabilities(p)
    else:
        print(f"INSECTS_TRAIN_BEGIN model=xgboost seed={seed}", flush=True)
        p, info = fit_tree(
            data, "xgboost", seed, cfg, folder, threads=8
        )
        p = normalize_probabilities(p)
    np.savez_compressed(
        probs_path, row_id=np.arange(len(data.y)), probabilities=p
    )
    jdump(
        complete,
        {
            "dataset": "insects",
            "model": "xgboost",
            "seed": seed,
            "profile": profile,
            "data_sha256": hashes,
            "frozen_runner_sha256": EXPECTED_FINAL_SHA,
            "method_parameters_frozen_before_external_dataset": True,
            "training": info,
        },
    )
    print(f"INSECTS_TRAIN_COMPLETE model=xgboost seed={seed}", flush=True)
    return folder, p


def frozen_cache_ext(root, data, family, seed, profile, hashes):
    if data.name != "insects":
        return _orig_frozen_cache(root, data, family, seed, profile, hashes)
    return _train_or_load_insects(data, family, seed, profile, hashes)


def paired_rank_biserial(diff):
    """Paired rank-biserial correlation; positive means first method is better."""
    d = np.asarray(diff, float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0:
        return 0.0
    order = np.argsort(np.abs(d))
    ranks = np.empty(len(d), float)
    # average tied absolute ranks
    vals = np.abs(d)
    sorted_vals = vals[order]
    i = 0
    while i < len(d):
        j = i + 1
        while j < len(d) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg
        i = j
    pos = ranks[d > 0].sum()
    neg = ranks[d < 0].sum()
    den = pos + neg
    return float((pos - neg) / den) if den else 0.0


def bootstrap_ci(values, seed=20260912, B=20000):
    x = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = np.empty(B, float)
    for b in range(B):
        means[b] = np.mean(rng.choice(x, size=len(x), replace=True))
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def compact_review(out: Path):
    include = [
        "LOCKED_STAGE2_PLAN.json",
        "ENVIRONMENT.json",
        "DATASET_AUDIT.json",
        "all_summaries.csv",
        "all_summaries.json",
        "decision_trace.csv",
        "PAIRWISE_STATS.csv",
        "STAGE2_ASSESSMENT.json",
    ]
    files = [out / n for n in include if (out / n).exists()]
    files += sorted((out / "figures").glob("*.pdf"))
    for p in sorted((out / "jobs").glob("*")):
        for n in ["summaries.json", "decision_trace.json", "queries.json", "job_record.json"]:
            q = p / n
            if q.exists():
                files.append(q)

    max_part = int(7.0 * 1024 * 1024)
    groups, current, approx = [], [], 100_000
    for p in files:
        cost = len(zlib.compress(p.read_bytes(), 6)) + 1500
        if current and approx + cost > max_part:
            groups.append(current)
            current, approx = [], 100_000
        current.append(p)
        approx += cost
    if current:
        groups.append(current)

    outputs = []
    for i, group in enumerate(groups, 1):
        dest = PROJECT / f"DTRAR_Stage2_Primary_Review_{i:02d}_of_{len(groups):02d}.zip"
        tmp = dest.with_suffix(".zip.partial")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            manifest = []
            for p in group:
                rel = str(p.relative_to(out))
                b = p.read_bytes()
                z.writestr(rel, b)
                manifest.append({"path": rel, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()})
            z.writestr(
                "PART_MANIFEST.json",
                json.dumps({"part": i, "parts": len(groups), "files": manifest}, indent=2),
            )
        with zipfile.ZipFile(tmp) as z:
            if z.testzip() is not None:
                raise RuntimeError("review ZIP CRC failure")
        tmp.replace(dest)
        outputs.append(dest)
        print(f"STAGE2_REVIEW_PART_READY {dest}", flush=True)
    return outputs


def main():
    insects_path, insects_source = download_insects()
    X, y, classes = insects_frame(insects_path)
    insects_hash = sha256(insects_path)
    print(
        f"INSECTS_AUDIT_OK n={len(y)} features={X.shape[1]} "
        f"classes={len(classes)} sha256={insects_hash}",
        flush=True,
    )

    # Monkeypatch only the dataset/cache hooks. DTRA-R/RAPID code and parameters
    # remain the exact frozen passing implementation.
    final.core.load_dataset = load_dataset_ext
    final.core.source_hashes = source_hashes_ext
    final.core.frozen_cache = frozen_cache_ext

    stage_plan = {
        "stage": "Stage 2 primary confirmation",
        "frozen_method": "DTRA-RAPID-geometry-frozen-v2.2",
        "frozen_runner_sha256": EXPECTED_FINAL_SHA,
        "datasets": DATASETS,
        "seeds": SEEDS,
        "model": "xgboost",
        "method_parameters_changed": False,
        "external_dataset": "INSECTS abrupt_balanced",
        "external_dataset_rows": EXPECTED_INSECTS_ROWS,
        "external_dataset_features": EXPECTED_INSECTS_FEATURES,
        "external_dataset_classes": EXPECTED_INSECTS_CLASSES,
        "external_dataset_tuning_permitted": False,
        "budget_fraction": final.BETA,
        "delay": final.DELAY,
        "alpha": final.ALPHA,
        "familywise_alpha": final.FAMILY_ALPHA,
        "harm_cap": final.HARM_CAP,
        "minimum_worthwhile_brier_gain": final.MIN_GAIN,
        "insects_source_used": insects_source,
        "insects_sha256": insects_hash,
        "note": (
            "INSECTS is an untouched confirmation stream. Results must not be "
            "used to modify DTRA-R/RAPID parameters."
        ),
    }
    sig = hashlib.sha256(json.dumps(stage_plan, sort_keys=True).encode()).hexdigest()[:12]
    out = STAGE_ROOT / sig
    out.mkdir(parents=True, exist_ok=True)
    jdump(out / "LOCKED_STAGE2_PLAN.json", stage_plan)
    jdump(
        out / "ENVIRONMENT.json",
        {"python": sys.version, "platform": platform.platform()},
    )
    shutil.copy2(Path(__file__), out / Path(__file__).name)

    # Dataset audit recorded before any INSECTS method result is inspected.
    audits = {}
    for ds in DATASETS:
        d = load_dataset_ext(ds, RAW_DIR)
        audits[ds] = {
            "n": int(len(d.y)),
            "features": int(d.X.shape[1]),
            "classes": int(len(d.classes)),
            "train_n": int(len(d.train)),
            "source_n": int(len(d.source)),
            "target_n": int(len(np.concatenate(d.windows))),
            "windows": int(len(d.windows)),
            "metadata": d.metadata,
        }
    jdump(out / "DATASET_AUDIT.json", audits)

    print(
        "DTRAR_STAGE2_BEGIN datasets=4 seeds=10 jobs=40 "
        "method_frozen=true insects_tuning=false",
        flush=True,
    )

    summaries, decisions = [], []
    reused_jobs = 0
    executed_jobs = 0
    with threadpool_limits(limits=1):
        for ds in DATASETS:
            for seed in SEEDS:
                job = out / "jobs" / f"{ds}_seed{seed}"
                summary_path = job / "summaries.json"
                decisions_path = job / "decision_trace.json"
                record_path = job / "job_record.json"

                reusable = False
                if summary_path.exists() and decisions_path.exists() and record_path.exists():
                    try:
                        s = json.loads(summary_path.read_text())
                        d = json.loads(decisions_path.read_text())
                        methods_seen = {r.get("method") for r in s}
                        reusable = (
                            len(s) == len(METHODS)
                            and methods_seen == set(METHODS)
                            and all(r.get("dataset") == ds for r in s)
                            and all(int(r.get("seed")) == seed for r in s)
                        )
                    except Exception:
                        reusable = False

                if reusable:
                    reused_jobs += 1
                    print(
                        f"STAGE2_JOB_REUSED dataset={ds} seed={seed} "
                        f"rows={len(s)}",
                        flush=True,
                    )
                else:
                    s, d = final.run_one(ds, seed, out)
                    executed_jobs += 1

                summaries.extend(s)
                decisions.extend(d)

    print(
        f"STAGE2_RESUME_SUMMARY reused_jobs={reused_jobs} "
        f"executed_jobs={executed_jobs}",
        flush=True,
    )

    df = pd.DataFrame(summaries)
    dd = pd.DataFrame(decisions)
    df.to_csv(out / "all_summaries.csv", index=False)
    jdump(out / "all_summaries.json", summaries)
    dd.to_csv(out / "decision_trace.csv", index=False)

    # Pairwise 10-seed statistics for the primary Brier endpoint.
    stats = []
    for ds in DATASETS:
        sub = df[df.dataset == ds].pivot(
            index="seed", columns="method", values="brier_sum_over_classes"
        )
        raw = sub["raw"].to_numpy()
        for comp in ["dtrar_rapid", "dtrar_strict", "vamr", "logistic"]:
            other = sub[comp].to_numpy()
            # Positive improvement = raw Brier - comparator Brier.
            improvement = raw - other
            try:
                w = wilcoxon(improvement, alternative="two-sided", zero_method="wilcox")
                pval = float(w.pvalue)
            except ValueError:
                pval = 1.0
            stats.append({
                "dataset": ds,
                "comparison": f"{comp}_vs_raw",
                "mean_brier_improvement": float(np.mean(improvement)),
                "ci95_low": bootstrap_ci(improvement, 20260912 + len(stats))[0],
                "ci95_high": bootstrap_ci(improvement, 20260912 + len(stats))[1],
                "wilcoxon_p": pval,
                "rank_biserial_positive_is_better": paired_rank_biserial(improvement),
            })
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(out / "PAIRWISE_STATS.csv", index=False)

    assessment = {}
    for ds in DATASETS:
        sub = df[df.dataset == ds]
        means = sub.groupby("method").mean(numeric_only=True)
        assessment[ds] = {
            "raw_brier": float(means.loc["raw", "brier_sum_over_classes"]),
            "rapid_brier": float(means.loc["dtrar_rapid", "brier_sum_over_classes"]),
            "rapid_minus_raw": float(
                means.loc["dtrar_rapid", "brier_sum_over_classes"]
                - means.loc["raw", "brier_sum_over_classes"]
            ),
            "rapid_coverage": float(means.loc["dtrar_rapid", "coverage"]),
            "rapid_worst_class_coverage": float(
                means.loc["dtrar_rapid", "worst_present_class_coverage"]
            ),
            "rapid_set_size": float(means.loc["dtrar_rapid", "set_size"]),
            "rapid_singleton_rate": float(means.loc["dtrar_rapid", "singleton_rate"]),
            "rapid_agreement_singleton_accuracy": float(
                means.loc["dtrar_rapid", "agreement_singleton_accuracy"]
            ) if np.isfinite(means.loc["dtrar_rapid", "agreement_singleton_accuracy"]) else None,
        }

    # These gates were defined before inspecting INSECTS:
    # - no dataset may suffer >0.01 mean Brier harm vs raw;
    # - at least 3/4 datasets should improve mean Brier vs raw.
    protected = {
        ds: assessment[ds]["rapid_minus_raw"] <= 0.01 for ds in DATASETS
    }
    improved = {
        ds: assessment[ds]["rapid_minus_raw"] < 0.0 for ds in DATASETS
    }
    assessment["gates"] = {
        "all_four_protected": bool(all(protected.values())),
        "at_least_three_improved": int(sum(improved.values())) >= 3,
        "protected_by_dataset": protected,
        "improved_by_dataset": improved,
    }
    assessment["primary_confirmation_pass"] = bool(
        assessment["gates"]["all_four_protected"]
        and assessment["gates"]["at_least_three_improved"]
    )
    assessment["conformal_note"] = (
        "No conformal guarantee is inferred from this gate. Marginal/worst-class "
        "coverage and singleton diagnostics are exported separately for the "
        "dedicated uncertainty-comparison stage."
    )
    jdump(out / "STAGE2_ASSESSMENT.json", assessment)

    # Vector-PDF figures only.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figdir = out / "figures"
        figdir.mkdir(exist_ok=True)
        label_map = {
            "raw": "Raw",
            "vamr": "VAMR",
            "logistic": "Feature repair",
            "dtrar_strict": "DTRA-R strict",
            "dtrar_rapid": "DTRA-R + RAPID",
        }
        for ds in DATASETS:
            sub = df[df.dataset == ds]
            fig, ax = plt.subplots(figsize=(8.4, 4.8), layout="constrained")
            x = np.arange(len(METHODS))
            vals = [
                sub[sub.method == m].brier_sum_over_classes.mean()
                for m in METHODS
            ]
            errs = [
                sub[sub.method == m].brier_sum_over_classes.std(ddof=1)
                for m in METHODS
            ]
            ax.errorbar(x, vals, yerr=errs, marker="o", capsize=4)
            ax.set_xticks(x, [label_map[m] for m in METHODS], rotation=15)
            ax.set_ylabel("Brier loss (lower is better)")
            ax.set_title(f"Frozen DTRA-R + RAPID — {ds}, 10 seeds")
            ax.grid(alpha=0.2)
            fig.savefig(figdir / f"{ds}_brier_10seed.pdf", bbox_inches="tight")
            plt.close(fig)

        # Safety/improvement overview.
        overview = pd.DataFrame([
            {
                "dataset": ds,
                "raw": assessment[ds]["raw_brier"],
                "rapid": assessment[ds]["rapid_brier"],
                "improvement": -assessment[ds]["rapid_minus_raw"],
            }
            for ds in DATASETS
        ])
        overview.to_csv(out / "figures" / "brier_improvement_source.csv", index=False)
        fig, ax = plt.subplots(figsize=(8.4, 4.8), layout="constrained")
        xx = np.arange(len(DATASETS))
        ax.axhline(0.0, linewidth=1)
        ax.plot(xx, overview["improvement"], marker="o")
        ax.set_xticks(xx, DATASETS)
        ax.set_ylabel("Raw − DTRA-R/RAPID Brier")
        ax.set_title("Repair benefit under frozen method (positive is better)")
        ax.grid(alpha=0.2)
        fig.savefig(figdir / "four_dataset_repair_benefit.pdf", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        (out / "FIGURE_WARNING.txt").write_text(repr(exc))

    parts = compact_review(out)
    print(
        "DTRAR_STAGE2_ASSESSMENT " + json.dumps(assessment["gates"]),
        flush=True,
    )
    print(
        f"DTRAR_STAGE2_COMPLETE jobs=40 method_rows={len(df)} "
        f"review_parts={len(parts)} "
        f"pass={str(assessment['primary_confirmation_pass']).lower()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
