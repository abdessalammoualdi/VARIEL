#!/usr/bin/env python3
"""Run the original source-training pipeline without changing its source."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'variel' / 'infrastructure'

def main():
    parser = argparse.ArgumentParser(description='Source-only training and static evaluation: Bank, Electricity, Gas')
    parser.add_argument('--datasets', default='bank', help='bank,electricity,gas or all')
    parser.add_argument('--models', default='xgboost', help='xgboost,catboost,ft_transformer or all')
    parser.add_argument('--seeds', default='0')
    parser.add_argument('--profile', choices=['smoke','primary'], default='smoke')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--score', choices=['aps_randomized','lac','aps_deterministic'], default='aps_randomized')
    parser.add_argument('--sensitivity', action='store_true')
    parser.add_argument('--execute', action='store_true', help='Actually train the selected models')
    args = parser.parse_args()
    selected_datasets = ['electricity','bank','gas'] if args.datasets=='all' else args.datasets.split(',')
    selected_models = ['xgboost','catboost','ft_transformer'] if args.models=='all' else args.models.split(',')
    if not set(selected_datasets) <= {'bank','electricity','gas'}:
        parser.error('Original training pipeline supports bank,electricity,gas; other datasets use separate original runners')
    if not set(selected_models) <= {'xgboost','catboost','ft_transformer'}:
        parser.error('Models: xgboost, catboost, ft_transformer or all')
    cmd=[sys.executable,'-m','dtra','static','--datasets',args.datasets,'--models',args.models,
         '--seeds',args.seeds,'--profile',args.profile,'--threads',str(args.threads),
         '--device',args.device,'--alpha',str(args.alpha),'--score',args.score,'--no-plots','--data-dir',str(ROOT/'data/raw'),
         '--output',str(ROOT/'artifacts')]
    if args.sensitivity:
        cmd.append('--sensitivity')
    if not args.execute:
        print('Preview: the original training and static-evaluation pipeline')
        print(' '.join(cmd))
        print('Use --execute after dataset acquisition and environment setup.')
        return 0
    env=os.environ.copy()
    env['PYTHONPATH']=str(ENGINE)+os.pathsep+env.get('PYTHONPATH','')
    return subprocess.call(cmd,cwd=ROOT,env=env)

if __name__=='__main__':
    raise SystemExit(main())
