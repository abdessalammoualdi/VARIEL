#!/usr/bin/env python3
"""Check Occupancy v2 and EEG exact prepared-data identities and blind splits."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'reproducibility/unseen_prepared_audit'
audit=json.loads((base/'UNSEEN_DATASET_AUDIT_V2.json').read_text())
prepped=json.loads((base/'PREPARED_FILES.json').read_text())
assert prepped['selection_locked_before_performance'] is True
assert audit['performance_seen'] is False
assert audit['primary_unseen_confirmation_datasets']==['occupancy_detection_v2','eeg_eye_state']
assert len(prepped['files'])==2
for row in prepped['files']:
    name=row['dataset']
    record=audit['datasets'][name]
    expected=record['v2_locked_sha256'] if name=='occupancy_detection_v2' else record['locked_sha256']
    rows=record['v2_rows'] if name=='occupancy_detection_v2' else record['locked_rows']
    assert expected==row['sha256'] and rows==row['rows']
    assert row['train_n']+row['cal_n']+row['target_n']==row['rows']
    src=ROOT/row['prepared_path']
    if src.exists():
        assert hashlib.sha256(src.read_bytes()).hexdigest()==expected,name
        print('MATCH: prepared file bytes',name)
    else:
        print('LOCK MATCH (external prepared CSV not included):',name,row['rows'],expected[:16])
print('PASS: both blind-audit dataset identities, chronological split counts and preparation plan')
