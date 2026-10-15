#!/usr/bin/env python3
"""Run source/data/terminal integrity checks; does not train any model."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
checks=[
 'verify.py', 'verify_input_linkage.py', 'verify_racs_linkage.py',
 'verify_terminal_chain.py', 'verify_unseen_data.py', 'verify_input_evidence.py'
]
for name in checks:
    code=subprocess.call([sys.executable,str(ROOT/'scripts'/name)],cwd=str(ROOT))
    if code:raise SystemExit('%s failed (exit %d)'%(name,code))
print('PASS: frozen code and source-to-input integrity checks')
print('No original model training, fitting or result reruns were requested by this command.')
