#!/usr/bin/env python3
"""Check terminal RACS sources, inputs, decisions and original output without retraining."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'reproducibility/resource_cabinet/racs'
meta = json.loads((HERE / 'PROVENANCE.json').read_text())
for record in meta['files']:
    b = (HERE / record['filename']).read_bytes()
    assert hashlib.sha256(b).hexdigest() == record['sha256'], record['filename']
source = ROOT/'variel/frozen/resource/run_racs_closure.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == meta['terminal_script_sha256']

def rows(name):
    with (HERE/name).open(newline='') as f:
        return list(csv.DictReader(f))

base = rows('CPPI_GREG_SELECTED_ALL_SIX.csv')
truth = rows('BOSA_SELECTED_DEPLOYMENT_SEEDWISE.csv')
final = rows('RACS_SELECTED_ALL_SIX.csv')
rescue = rows('RACS_RESCUE_21.csv')
assert len(base) == len(truth) == len(final) == 1080
assert len(rescue) == 21
key = ['dataset', 'seed', 'profile', 'lambda']
def key_of(r):return tuple(r[n] for n in key)
indexed = {key_of(r): r for r in truth}
assert len(indexed)==1080
assert [key_of(x) for x in base] == [key_of(x) for x in final]
violations = 0
exact = 0
racs = 0
weight = 0.375
for b,f in zip(base,final):
    t=indexed[key_of(b)]
    valid = str(b['operational_valid']).lower() == 'true'
    flagged = str(f['racs_triggered']).lower() == 'true'
    assert flagged == (not valid), key_of(b)
    g=float(t['true_gain']);threshold=float(t['threshold'])
    if flagged:
        racs += 1
        mu = weight*float(b['cppi_estimate'])+(1-weight)*float(b['sharp_estimate'])
        action='full' if mu>threshold else 'sparse'
        assert f['action']==action, key_of(b)
        assert abs(float(f['racs_estimate'])-mu)<1e-10
        assert abs(float(f['racs_weight'])-weight)<1e-12
    else:
        assert f['action']==b['action']
        assert f['label_fraction']==b['label_fraction']
    regret = max(0., g-threshold) if f['action']=='sparse' else max(0., threshold-g)
    if regret>0.001+1e-12:violations+=1
    exact += int(f['action']==('full' if g>threshold else 'sparse'))
assert racs == 21
assert violations == 0
assert exact == 1075
assessment=json.loads((HERE/'RACS_FINAL_ASSESSMENT.json').read_text())
assert assessment['development_operational_valid']==1080
assert assessment['development_exact_winner_matches']==1075
assert abs(assessment['selected_cppi_weight']-weight)<1e-12
print('PASS: all RACS input/output hashes, 1,080 aligned decisions, 21 consensus cases, 1,075 exact actions.')
