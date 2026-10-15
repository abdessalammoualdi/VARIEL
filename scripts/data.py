#!/usr/bin/env python3
"""Inspect exact dataset source records; fetch only checksum-pinned external originals."""
import argparse,hashlib,json,subprocess,sys,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REG=json.loads((ROOT/'data/sources.json').read_text())
parser=argparse.ArgumentParser()
parser.add_argument('command',choices=['list','fetch','fetch-core'])
parser.add_argument('dataset',nargs='?',choices=sorted(REG))
a=parser.parse_args()
if a.command=='list':
 for k,v in REG.items():print(f'{k:23} {v["group"]:18} {v["url"]}')
 sys.exit(0)
if a.command=='fetch-core':
 script=ROOT/'variel/infrastructure/scripts/download_data.py'
 raise SystemExit(subprocess.call([sys.executable,str(script),'--data-dir',str(ROOT/'data/raw')]))
if not a.dataset:parser.error('fetch requires dataset')
x=REG[a.dataset]
if not x['hash'] or x['file'].find('...')>=0:
 sys.exit('No single-source SHA-locked direct file for this dataset. Use the listed official source and original loader.')
if x['url'].find('raw.githubusercontent.com')<0:
 sys.exit('The record is a dataset landing page or an archive requiring extraction. Use original acquisition script or source page.')
out=ROOT/'data/raw'/x['file'];out.parent.mkdir(parents=True,exist_ok=True)
if out.exists() and hashlib.sha256(out.read_bytes()).hexdigest()==x['hash']:
 print('VERIFIED',out)
 sys.exit(0)
req=urllib.request.Request(x['url'],headers={'User-Agent':'VARIEL-source-verification/1.0'})
with urllib.request.urlopen(req,timeout=90) as r:raw=r.read(120_000_001)
if len(raw)>120_000_000:sys.exit('Unexpected download size')
sha=hashlib.sha256(raw).hexdigest()
if sha!=x['hash']:sys.exit(f'Checksum mismatch for {a.dataset}: {sha}')
out.write_bytes(raw)
print('VERIFIED',out)
