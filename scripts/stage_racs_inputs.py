#!/usr/bin/env python3
"""Stage the frozen RACS script and exact input CSVs; never execute model training."""
from pathlib import Path
import shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'reproducibility/resource_cabinet/racs'
DST=ROOT/'runs/racs_frozen_inputs'
DST.mkdir(parents=True,exist_ok=True)
for name in ['BOSA_SELECTED_DEPLOYMENT_SEEDWISE.csv','CPPI_GREG_SELECTED_ALL_SIX.csv']:
    shutil.copy2(SRC/name,DST/name)
shutil.copy2(ROOT/'variel/frozen/resource/run_racs_closure.py',DST/'run_racs_closure.py')
print('Staged exact frozen RACS inputs and runner:',DST)
print('Use python scripts/verify_racs_linkage.py for a training-free verification against the archived terminal outputs.')
