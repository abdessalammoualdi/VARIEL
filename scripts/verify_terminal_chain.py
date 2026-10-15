#!/usr/bin/env python3
"""Verify frozen SHARP, CPPI-GREG, and RACS handoffs without fitting models."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "reproducibility/resource_cabinet/terminal_upstream"
RACS = ROOT / "reproducibility/resource_cabinet/racs"

def read_csv(path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))

def seq(rows):
    return [(x["dataset"],x["seed"],x["profile"],x["lambda"]) for x in rows]

def valid(row):
    return row["operational_valid"].strip().lower() == "true"

sharp=read_csv(HERE/"SHARP_SELECTED_ALL_SIX.csv")
sharp_unres=read_csv(HERE/"SHARP_UNRESOLVED_CASES.csv")
cppi=read_csv(HERE/"CPPI_GREG_SELECTED_ALL_SIX.csv")
cppi_unres=read_csv(HERE/"CPPI_GREG_UNRESOLVED_CASES.csv")
racs=read_csv(RACS/"RACS_SELECTED_ALL_SIX.csv")
bosa=read_csv(HERE/"BOSA_SELECTED_DEPLOYMENT_SEEDWISE.csv")

assert len(sharp)==len(cppi)==len(racs)==len(bosa)==1080
assert len(set(seq(sharp))) == 1080
assert seq(sharp)==seq(cppi)==seq(racs)
assert set(seq(sharp))==set(seq(bosa))  # BOSA rows use a different dataset ordering
assert len(sharp_unres)==43 and len(cppi_unres)==21
assert {k for k in seq(sharp_unres)}=={k for k,x in zip(seq(sharp),sharp) if not valid(x)}
assert {k for k in seq(cppi_unres)}=={k for k,x in zip(seq(cppi),cppi) if not valid(x)}
assert sum(map(valid,sharp))==1037
assert sum(map(valid,cppi))==1059
assert sum(map(valid,racs))==1080
assert {k for k,x in zip(seq(sharp),sharp) if valid(x)}.issubset({k for k,x in zip(seq(cppi),cppi) if valid(x)})
# CPPI-GREG does not revise decisions certified by SHARP.
for s,c in zip(sharp,cppi):
    if valid(s):
        assert s["action"]==c["action"]
        assert abs(float(s["label_fraction"])-float(c["label_fraction"])) < 1e-12
# RACS does not revise decisions already operationally valid under CPPI-GREG.
for c,r in zip(cppi,racs):
    if valid(c):
        assert c["action"]==r["action"]
        assert abs(float(c["label_fraction"])-float(r["label_fraction"])) < 1e-12
assert sum(r["racs_triggered"].lower()=="true" for r in racs)==21
assert sum(r["exact_winner_match"].lower()=="true" for r in racs)==1075
# Match original archived source runners and terminal input bytes.
paths={
    "SHARP_SELECTED_ALL_SIX.csv":"DTRAR_ASPIRE_PPARC_SHARP_TERMINAL_GPU_Review.zip",
    "SHARP_UNRESOLVED_CASES.csv":"DTRAR_ASPIRE_PPARC_SHARP_TERMINAL_GPU_Review.zip",
    "CPPI_GREG_SELECTED_ALL_SIX.csv":"DTRAR_ASPIRE_CPPIGREG_TERMINAL_GPU_Review.zip",
    "CPPI_GREG_UNRESOLVED_CASES.csv":"DTRAR_ASPIRE_CPPIGREG_TERMINAL_GPU_Review.zip",
    "BOSA_SELECTED_DEPLOYMENT_SEEDWISE.csv":"DTRAR_ASPIRE_BOSA_NMCS_AllSix_EfficiencySweep_Review.zip",
}
from zipfile import ZipFile
for filename,archive in paths.items():
    with ZipFile(HERE/archive) as z:
        assert hashlib.sha256(z.read(filename)).digest()==hashlib.sha256((HERE/filename).read_bytes()).digest(),filename
for filename,archive in [
    ("run_pparc_sharp.py","DTRAR_ASPIRE_PPARC_SHARP_TERMINAL_GPU_Review.zip"),
    ("run_cppigreg_terminal.py","DTRAR_ASPIRE_CPPIGREG_TERMINAL_GPU_Review.zip"),
    ("run_racs_closure.py","DTRAR_ASPIRE_RACS_TERMINAL_CLOSURE.zip"),
    ("run_bosa_nmcs_all_six.py","DTRAR_ASPIRE_BOSA_NMCS_AllSix_EfficiencySweep_Review.zip"),
    ("run_burst_facc_tail.py","DTRAR_ASPIRE_BURST_FACC_TailSolver_GPU_Review.zip"),
]:
    with ZipFile(HERE/archive) as z:
        source=(ROOT/"variel/frozen/resource"/filename).read_bytes()
        assert hashlib.sha256(z.read(filename)).digest()==hashlib.sha256(source).digest(),filename

assess=json.loads((RACS/"RACS_FINAL_ASSESSMENT.json").read_text())
assert assess["development_operational_valid"]==1080
assert assess["development_exact_winner_matches"]==1075
print("PASS: SHARP 1037/1080 -> CPPI-GREG 1059/1080 -> RACS 1080/1080 operationally valid")
print("PASS: 43 -> 21 -> 0 unresolved; RACS 21 rescue decisions, 1075/1080 exact oracle actions")
print("PASS: source and input bytes match the original terminal review archives")
