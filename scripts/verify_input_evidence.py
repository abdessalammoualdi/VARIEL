#!/usr/bin/env python3
"""Validate the exact archived source-to-input evidence manifest."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'reproducibility/UPSTREAM_SHA256.json').read_text())
for r in manifest['files']:
    p=ROOT/r['path']
    assert p.is_file() and p.stat().st_size==r['bytes'],r['path']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],r['path']
print('PASS: %d frozen source-to-input archives, tables, scripts and assessment hashes'%len(manifest['files']))
