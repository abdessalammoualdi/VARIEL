#!/usr/bin/env python3
"""Dispatch the unmodified original experiment runners."""
import argparse,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN={
 'sparse': 'sparse/run_dtrar_final_development_batch11.py',
 'robustness':'sparse/run_dtrar_final_robustness_batch12.py',
 'liability':'sparse/run_dtrar_liability_frontier_batch15.py',
 'fresh_sparse':'sparse/run_dtrar_liability_fresh_confirmation_batch17.py',
 'resource_pparc':'resource/run_pparc_terminal.py',
 'resource_sharp':'resource/run_pparc_sharp.py',
 'resource_cppi':'resource/run_cppigreg_terminal.py',
 'resource_racs':'resource/run_racs_closure.py',
 'resource_bosa':'resource/run_bosa_nmcs_all_six.py',
 'resource_burst':'resource/run_burst_facc_tail.py',
 'high_information':'highinfo/run_dominance_guard.py',
 'preservation':'preservation/run_v24.py',
 'multiclass':'multiclass/run_xgboost_ltt_mc_v13.py',
 'temporal':'temporal/run_dtrar_temporal_actionability.py',
}
parser=argparse.ArgumentParser()
parser.add_argument('--list',action='store_true')
parser.add_argument('--component',choices=sorted(RUN))
parser.add_argument('--execute',action='store_true',help='Allow original runner to execute; may train models and need substantial input data.')
a=parser.parse_args()
if a.list or not a.component:
 for n in sorted(RUN):print(n)
 sys.exit(0)
script=ROOT/'variel/frozen'/RUN[a.component]
if not script.is_file():sys.exit(f'Missing original runner: {script}')
if not a.execute:
 print('Original runner:',script.relative_to(ROOT))
 print('Execution requires --execute. Consult reproducibility/REPRODUCTION_STATUS.md first.')
 sys.exit(0)
if a.component=='resource_racs':
 # Original RACS runner loads two frozen CSVs from its own directory.
 from shutil import copy2
 prepared=ROOT/'runs/racs_frozen_inputs';prepared.mkdir(parents=True,exist_ok=True)
 frozen=ROOT/'reproducibility/resource_cabinet/racs'
 for filename in ['BOSA_SELECTED_DEPLOYMENT_SEEDWISE.csv','CPPI_GREG_SELECTED_ALL_SIX.csv']:
  copy2(frozen/filename,prepared/filename)
 copy2(script,prepared/script.name)
 script=prepared/script.name
 print('Using frozen RACS source and original input CSVs:',prepared)
else:
 print('Run requires original dataset, endpoint/cache and configuration preparation. See reproducibility/INPUT_LINKAGE.json')
env=os.environ.copy()
env.setdefault('DTRAR_PROJECT',str(ROOT))
env['PYTHONPATH']=str(ROOT/'variel/infrastructure')+os.pathsep+str(script.parent)+os.pathsep+str(ROOT/'variel/frozen/sparse')+os.pathsep+env.get('PYTHONPATH','')
raise SystemExit(subprocess.call([sys.executable,str(script)],cwd=ROOT,env=env))
