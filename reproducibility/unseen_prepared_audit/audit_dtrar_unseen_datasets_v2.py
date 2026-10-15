#!/usr/bin/env python3
"""Corrected blind structural audit for unseen confirmation streams.

Purpose
-------
Fixes a *data-ingestion/audit bug* in the v1 unseen audit without changing any
scientific gate or looking at RAPID performance.

What v1 exposed
---------------
UCI Occupancy Detection officially contains 20,560 observations and no missing
values.  The locally locked v1 file contains 20,562 rows, exactly two of which
have a missing target.  Because v1 converted np.unique(...) to strings, NaN was
mistaken for an additional target class and the class-support gate fired.

V2 keeps the original 50/15/35 split and the original source/target class-
support requirement.  It applies the generic data-quality rule that rows with
missing/invalid targets are not admissible supervised observations, documents
those removals, and re-runs the structural audit before any model performance.

RT-IoT is *not rescued*: its already-locked repository-order binary stream is
re-audited under the same class-support rule and remains excluded if either
source or deployment lacks a class.

To restore domain diversity before any RAPID performance is seen, v2 adds a new
pre-performance candidate fixed now: UCI EEG Eye State (id=264), whose official
metadata states that all 14,980 observations are in chronological order from one
continuous EEG measurement.  No performance is evaluated here.

Primary unseen confirmation streams if structurally admissible:
  1. Occupancy Detection (binary environmental sensing)
  2. EEG Eye State (binary physiological sensing)
Secondary unseen confirmation stream:
  - Room Occupancy Estimation (4-class environmental sensing), if admissible.

The original v1 files/logs are left untouched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent
LOCKED = PROJECT / "data" / "unseen_locked"
OUT = PROJECT / "artifacts" / "unseen_dataset_audit_v2"
EEG_ID = 264


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ensure_ucimlrepo() -> None:
    try:
        import ucimlrepo  # noqa: F401
        return
    except Exception:
        pass
    print("INSTALL dependency=ucimlrepo", flush=True)
    # Avoid inheriting a broken vendor-only index as the sole source.
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--index-url", "https://pypi.org/simple", "ucimlrepo"]
    subprocess.run(cmd, check=True)


def split_indices(n: int):
    a = int(np.floor(0.50 * n))
    b = int(np.floor(0.65 * n))
    return np.arange(0, a), np.arange(a, b), np.arange(b, n)


def clean_label_series(y: pd.Series) -> pd.Series:
    # Preserve categorical labels but normalize missing/blank targets.
    out = y.copy()
    if out.dtype == object:
        s = out.astype("string").str.strip()
        s = s.mask(s.isin(["", "nan", "NaN", "None", "<NA>"]))
        out = s
    return out


def counts(y: pd.Series, idx) -> dict:
    s = y.iloc[np.asarray(idx, int)]
    vc = s.value_counts(dropna=False)
    return {str(k): int(v) for k, v in vc.items()}


def support_audit(y: pd.Series) -> dict:
    tr, cal, tar = split_indices(len(y))
    source = pd.concat([y.iloc[tr], y.iloc[cal]], ignore_index=True)
    src_classes = set(map(str, pd.unique(source.dropna())))
    tar_classes = set(map(str, pd.unique(y.iloc[tar].dropna())))
    pass_support = len(src_classes) >= 2 and tar_classes.issubset(src_classes) and len(tar_classes) >= 2
    wins = np.array_split(tar, 12)
    return {
        "n": int(len(y)),
        "train_n": int(len(tr)),
        "cal_n": int(len(cal)),
        "target_n": int(len(tar)),
        "train_counts": counts(y, tr),
        "cal_counts": counts(y, cal),
        "target_counts": counts(y, tar),
        "source_classes": sorted(src_classes),
        "target_classes": sorted(tar_classes),
        "target_window_counts": [counts(y, w) for w in wins],
        "pass_source_target_class_support": bool(pass_support),
    }


def load_locked(name: str) -> tuple[Path, pd.DataFrame]:
    path = LOCKED / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required v1 locked file missing: {path}")
    return path, pd.read_csv(path)


def audit_occupancy() -> tuple[dict, Path]:
    src, df = load_locked("occupancy_detection")
    if "__target__" not in df:
        raise RuntimeError("occupancy_detection: __target__ missing from v1 lock")
    raw_rows = len(df)
    raw_sha = sha256(src)
    y0 = pd.to_numeric(df["__target__"], errors="coerce")
    invalid = y0.isna()
    removed = int(invalid.sum())
    # This is a loader/data-quality correction, not a class-support relaxation.
    clean = df.loc[~invalid].copy().reset_index(drop=True)
    clean["__sequence__"] = np.arange(len(clean), dtype=int)
    clean["__target__"] = pd.to_numeric(clean["__target__"], errors="raise").astype(int)
    y = clean["__target__"]
    audit = support_audit(y)
    audit.update({
        "dataset": "occupancy_detection",
        "uci_id": 357,
        "v1_locked_rows": int(raw_rows),
        "v1_locked_sha256": raw_sha,
        "invalid_target_rows_removed": removed,
        "v2_rows": int(len(clean)),
        "expected_official_rows": 20560,
        "row_count_matches_official": bool(len(clean) == 20560),
        "correction": "drop rows with missing/invalid target before chronological split; scientific split/gate unchanged",
        "performance_seen": False,
    })
    if removed != 2 or len(clean) != 20560:
        raise SystemExit(
            f"ERROR: occupancy ingestion correction did not match the documented v1 anomaly: removed={removed} rows={len(clean)}"
        )
    if not audit["pass_source_target_class_support"]:
        raise SystemExit("ERROR: occupancy still fails the unchanged class-support gate after ingestion correction")
    out = LOCKED / "occupancy_detection_v2.csv"
    clean.to_csv(out, index=False)
    audit["v2_locked_csv"] = str(out.relative_to(PROJECT))
    audit["v2_locked_sha256"] = sha256(out)
    return audit, out


def audit_rt() -> dict:
    src, df = load_locked("rt_iot2022_binary")
    if "__target__" not in df:
        raise RuntimeError("rt_iot2022_binary: __target__ missing")
    y = pd.to_numeric(df["__target__"], errors="coerce")
    missing = int(y.isna().sum())
    if missing:
        raise SystemExit(f"ERROR: RT-IoT locked binary target unexpectedly has missing values: {missing}")
    y = y.astype(int)
    audit = support_audit(y)
    audit.update({
        "dataset": "rt_iot2022_binary",
        "uci_id": 942,
        "locked_rows": int(len(df)),
        "locked_sha256": sha256(src),
        "ordering": "repository sequence order; no wall-clock chronology claimed",
        "performance_seen": False,
        "selection_status": "exclude if unchanged source/target class-support gate fails",
    })
    return audit


def audit_room() -> dict:
    src, df = load_locked("room_occupancy")
    if "__target__" not in df:
        raise RuntimeError("room_occupancy: __target__ missing")
    y = clean_label_series(df["__target__"])
    missing = int(y.isna().sum())
    if missing:
        raise SystemExit(f"ERROR: Room Occupancy locked target has missing values: {missing}")
    audit = support_audit(y)
    audit.update({
        "dataset": "room_occupancy",
        "uci_id": 864,
        "locked_rows": int(len(df)),
        "locked_sha256": sha256(src),
        "ordering": "explicit Date+Time chronology from v1 lock",
        "performance_seen": False,
    })
    return audit


def audit_eeg() -> tuple[dict, Path]:
    ensure_ucimlrepo()
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=EEG_ID)
    X = ds.data.features.copy().reset_index(drop=True)
    yd = ds.data.targets.copy().reset_index(drop=True)
    if yd.shape[1] != 1:
        raise RuntimeError(f"eeg_eye_state: expected one target column, got {list(yd.columns)}")
    y = clean_label_series(yd.iloc[:, 0])
    if y.isna().any():
        raise RuntimeError(f"eeg_eye_state: missing targets={int(y.isna().sum())}")
    if len(X) != 14980:
        raise RuntimeError(f"eeg_eye_state: expected official 14980 rows, found {len(X)}")
    if X.isna().any().any():
        raise RuntimeError(f"eeg_eye_state: missing feature cells={int(X.isna().sum().sum())}")

    # Official metadata states repository rows are chronological from one continuous EEG measurement.
    outdf = X.copy()
    outdf.insert(0, "__sequence__", np.arange(len(outdf), dtype=int))
    outdf["__target__"] = y.to_numpy()
    out = LOCKED / "eeg_eye_state.csv"
    outdf.to_csv(out, index=False)

    audit = support_audit(pd.Series(y).reset_index(drop=True))
    audit.update({
        "dataset": "eeg_eye_state",
        "uci_id": EEG_ID,
        "ordering": "repository order; official UCI metadata states all observations are chronological from one continuous EEG measurement",
        "locked_rows": int(len(outdf)),
        "n_features": int(X.shape[1]),
        "missing_feature_cells": int(X.isna().sum().sum()),
        "locked_csv": str(out.relative_to(PROJECT)),
        "locked_sha256": sha256(out),
        "performance_seen": False,
    })
    return audit, out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOCKED.mkdir(parents=True, exist_ok=True)
    print("UNSEEN_AUDIT_V2_BEGIN performance_evaluation=false scientific_gates_changed=false", flush=True)

    occ, _ = audit_occupancy()
    print(
        "UNSEEN_AUDIT_V2 dataset=occupancy_detection "
        f"pass={occ['pass_source_target_class_support']} rows={occ['n']} "
        f"removed_invalid_targets={occ['invalid_target_rows_removed']}", flush=True
    )

    rt = audit_rt()
    print(
        "UNSEEN_AUDIT_V2 dataset=rt_iot2022_binary "
        f"pass={rt['pass_source_target_class_support']} rows={rt['n']} "
        f"source={rt['source_classes']} target={rt['target_classes']}", flush=True
    )

    room = audit_room()
    print(
        "UNSEEN_AUDIT_V2 dataset=room_occupancy "
        f"pass={room['pass_source_target_class_support']} rows={room['n']}", flush=True
    )

    eeg, _ = audit_eeg()
    print(
        "UNSEEN_AUDIT_V2 dataset=eeg_eye_state "
        f"pass={eeg['pass_source_target_class_support']} rows={eeg['n']} "
        f"source={eeg['source_classes']} target={eeg['target_classes']}", flush=True
    )

    if not occ["pass_source_target_class_support"]:
        raise SystemExit("ERROR: corrected Occupancy stream is not structurally admissible")
    if not eeg["pass_source_target_class_support"]:
        raise SystemExit("ERROR: EEG Eye State failed the pre-performance class-support audit")

    primary = ["occupancy_detection_v2", "eeg_eye_state"]
    secondary = ["room_occupancy"] if room["pass_source_target_class_support"] else []
    excluded = []
    if not rt["pass_source_target_class_support"]:
        excluded.append({
            "dataset": "rt_iot2022_binary",
            "reason": "unchanged source/target class-support gate failed under repository sequence order",
        })

    manifest = {
        "performance_seen": False,
        "selection_locked_before_performance": True,
        "scientific_gates_changed_from_v1": False,
        "v1_audit_preserved": True,
        "v2_reason": "correct v1 Occupancy ingestion artifact in which two invalid-target rows were mistaken for a new class; add a domain-diverse unseen candidate before any RAPID performance",
        "split_protocol": "first 50% source training, next 15% source calibration/audit reference, final 35% deployment split into 12 consecutive windows",
        "primary_unseen_confirmation_datasets": primary,
        "secondary_unseen_confirmation_datasets": secondary,
        "excluded_candidates": excluded,
        "later_method_rule": "apply the already-frozen RAPID risk-optimal mathematics unchanged; no dataset-specific method tuning",
        "datasets": {
            "occupancy_detection_v2": occ,
            "rt_iot2022_binary": rt,
            "room_occupancy": room,
            "eeg_eye_state": eeg,
        },
    }

    jpath = OUT / "UNSEEN_DATASET_AUDIT_V2.json"
    jpath.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    rows = []
    for name, a in manifest["datasets"].items():
        rows.append({
            "dataset": name,
            "uci_id": a.get("uci_id"),
            "primary_selected": name in primary,
            "secondary_selected": name in secondary,
            "pass_support": a["pass_source_target_class_support"],
            "n": a["n"],
            "train_n": a["train_n"],
            "cal_n": a["cal_n"],
            "target_n": a["target_n"],
            "source_classes": "|".join(a["source_classes"]),
            "target_classes": "|".join(a["target_classes"]),
            "train_counts": json.dumps(a["train_counts"], sort_keys=True),
            "cal_counts": json.dumps(a["cal_counts"], sort_keys=True),
            "target_counts": json.dumps(a["target_counts"], sort_keys=True),
        })
    cpath = OUT / "UNSEEN_DATASET_AUDIT_V2.csv"
    pd.DataFrame(rows).to_csv(cpath, index=False)

    review = PROJECT / "DTRAR_UnseenDataset_Audit_V2_Review.zip"
    with zipfile.ZipFile(review, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jpath, jpath.name)
        z.write(cpath, cpath.name)
        z.write(Path(__file__), Path(__file__).name)
    print("UNSEEN_DATASET_SELECTION_LOCKED_V2 primary=" + ",".join(primary) + " secondary=" + ",".join(secondary), flush=True)
    print(f"UNSEEN_AUDIT_V2_COMPLETE review={review.name}", flush=True)


if __name__ == "__main__":
    main()
