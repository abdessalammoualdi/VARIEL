#!/usr/bin/env python3
"""Validate locked sources and frozen plotting inputs without running experiments."""
import csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
errors=[];count=0
with (ROOT/'reproducibility/SOURCE_LOCK.csv').open(newline='') as f:
 for r in csv.DictReader(f):
  p=ROOT/'variel'/r['path']
  if not p.is_file():errors.append('Missing '+str(p))
  elif hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:errors.append('Checksum mismatch '+str(p))
  count+=1
base=ROOT/'figures/supplementary/data'
manifest=json.loads((base/'SOURCE_MANIFEST.json').read_text())
for name,record in manifest.items():
 p=base/name
 if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']:
  errors.append('Supplementary figure input mismatch: '+name)
print(f'Checked {count} frozen source identities and {len(manifest)} supplementary plot inputs')
for e in errors:print('FAILED',e)
if errors:raise SystemExit(1)
print('PASS — source and figure-input checksums; no experiment executed')
