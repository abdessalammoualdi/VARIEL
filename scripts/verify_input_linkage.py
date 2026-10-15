#!/usr/bin/env python3
"""Verify included frozen resource, fresh-data and archive identity records."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
record=json.loads((ROOT/'reproducibility/INPUT_LINKAGE.json').read_text())
for filename,expected in record['terminal_archives'].items():
    p=ROOT/'reproducibility/terminal_archives'/filename
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected['sha256'], filename
    with ZipFile(p) as z:assert z.testzip() is None, filename
fresh=ROOT/'reproducibility/fresh_confirmation'
lockbytes=(fresh/'LIABILITY_FRESH_LOCK.json').read_bytes()
lock=json.loads(lockbytes)
audit=json.loads((fresh/'SOURCE_TO_PREPARED_DATA.json').read_text())
assert hashlib.sha256(lockbytes).hexdigest()==audit['fresh_lock_sha256']
source=ROOT/'variel/frozen/sparse/audit_dtrar_liability_fresh_batch16.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()==audit['audit_source_sha256']
for row in audit['prepared_files']:
    locked=lock['datasets'][row['dataset']]
    assert locked['locked_sha256']==row['sha256']
    assert locked['n']==row['rows']
    for k in ('train_n','source_n','target_n','target_windows'):
        assert locked[k]==row[k]
    assert locked['train_n']+locked['source_n']+locked['target_n']==locked['n']
assert len(audit['prepared_files'])==3
print('PASS: 3 exact fresh prepared-data locks, original structural source,',len(record['terminal_archives']),'terminal ZIP hashes and CRCs.')
print('Note: prepared CSV bytes were independently verified in the archived Batch16 evidence; they are not distributed here.')
