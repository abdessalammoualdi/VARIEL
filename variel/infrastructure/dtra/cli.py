import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .calibration import ProbabilityCalibrator
from .conformal import SourceMondrian
from .data import load_dataset, source_roles, audit_dataset
from .io import write_json, file_hash
from .metrics import evaluate

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['electricity', 'bank', 'gas']
MODELS = ['xgboost', 'catboost', 'ft_transformer']


def names(value, available):
    result = available if value == 'all' else value.split(',')
    if not result or any(v not in available for v in result):
        raise ValueError(f'Choose all or comma-separated values from {available}')
    return result


def load_config(profile):
    config = json.loads((ROOT / 'configs/primary.json').read_text())
    if profile == 'smoke':
        config.update(tree_rounds=40, tree_early_stopping=10, ft_max_epochs=2,
                      ft_patience=2, ft_blocks=1, ft_width=64, ft_heads=4)
    return config


def source_hashes(data_dir, name):
    files = {'electricity': ['electricity-normalized.arff'], 'bank': ['bank-full.csv'],
             'gas': [f'batch{i}.dat' for i in range(1, 11)]}[name]
    expected = json.loads((ROOT / 'configs/raw_manifest.json').read_text())
    actual = {f: file_hash(Path(data_dir) / f) for f in files}
    for f in files:
        if actual[f] != expected[f]['sha256']:
            raise ValueError(f'Raw-data checksum mismatch: {f}')
    return actual


def audit(args):
    results, rows = [], []
    for name in names(args.datasets, DATASETS):
        hashes = source_hashes(args.data_dir, name)
        data = load_dataset(name, args.data_dir)
        result = audit_dataset(data)
        result['sha256'] = hashes
        results.append(result)
        rows.append({'dataset': name, 'n': len(data.y), 'features': data.X.shape[1], 'classes': len(data.classes),
                     'train': len(data.train), 'source_calibration': len(data.source), 'target': len(data.target),
                     'windows': len(data.windows), 'source_probability_fit': result['probability_fit']['n'],
                     'source_conformal_fit': result['conformal_fit']['n']})
        print(f'AUDIT_OK dataset={name} n={len(data.y)} features={data.X.shape[1]} train={len(data.train)} source={len(data.source)} target={len(data.target)}', flush=True)
    output = Path(args.output) / 'audit'
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'dataset_audit.json', results)
    pd.DataFrame(rows).to_csv(output / 'dataset_audit.csv', index=False)
    print(f'AUDIT_SAVED {output}', flush=True)


def static_run(args):
    from .models import fit_tree, fit_ft
    from .plots import figures
    config = load_config(args.profile)
    config['alpha'] = args.alpha
    config['score'] = args.score
    package_versions = {}
    for package in ['numpy', 'scipy', 'pandas', 'scikit-learn', 'xgboost', 'xgboost-cpu', 'catboost', 'torch', 'rtdl-revisiting-models']:
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    seed_list = [int(v) for v in args.seeds.split(',')]
    model_names = names(args.models, MODELS)
    write_json(Path(args.output) / 'last_requested_run.json', {**vars(args), 'config': config})
    all_summaries = []
    for dataset_name in names(args.datasets, DATASETS):
        data = load_dataset(dataset_name, args.data_dir, args.sensitivity)
        audit_dataset(data)
        hashes = source_hashes(args.data_dir, dataset_name)
        probability_ix, conformal_ix = source_roles(data)
        for family in model_names:
            for seed in seed_list:
                model_signature = {'dataset': dataset_name, 'model': family, 'seed': seed, 'profile': args.profile,
                                   'sensitivity': args.sensitivity, 'data_sha256': hashes,
                                   'package_versions': package_versions, 'requested_ft_device': args.device,
                                   'threads': args.threads,
                                   'model_config': {k: v for k, v in config.items() if k.startswith(('tree_', 'ft_'))},
                                   'models_code_sha256': file_hash(ROOT / 'dtra/models.py'),
                                   'data_code_sha256': file_hash(ROOT / 'dtra/data.py')}
                signature = hashlib.sha256(json.dumps(model_signature, sort_keys=True).encode()).hexdigest()[:12]
                cache = Path(args.output) / 'models' / dataset_name / family / f'seed_{seed}_{args.profile}_{signature}'
                cache.mkdir(parents=True, exist_ok=True)
                prediction_file = cache / 'frozen_probabilities.npz'
                if prediction_file.exists() and (cache / 'COMPLETE.json').exists():
                    print(f'REUSE_FROZEN_MODEL dataset={dataset_name} model={family} seed={seed}', flush=True)
                    with np.load(prediction_file) as saved:
                        p = saved['probabilities']
                        if not np.array_equal(saved['row_id'], np.arange(len(data.y))):
                            raise ValueError('Cached prediction row IDs differ')
                    train_info = json.loads((cache / 'training.json').read_text())
                else:
                    print(f'TRAIN_BEGIN dataset={dataset_name} model={family} seed={seed} profile={args.profile}', flush=True)
                    if family == 'ft_transformer':
                        p, train_info = fit_ft(data, seed, config, cache, args.device, args.threads)
                    else:
                        p, train_info = fit_tree(data, family, seed, config, cache, args.threads)
                    if p.shape != (len(data.y), len(data.classes)):
                        raise ValueError('Model probabilities have incorrect dimensions')
                    temporary = cache / 'frozen_probabilities.tmp.npz'
                    np.savez_compressed(temporary, probabilities=p, row_id=np.arange(len(data.y)))
                    temporary.replace(prediction_file)
                    write_json(cache / 'COMPLETE.json', model_signature)
                print(f'TRAIN_COMPLETE dataset={dataset_name} model={family} seed={seed} seconds={train_info["runtime_seconds"]:.2f}', flush=True)
                alpha_label = str(args.alpha).replace('.', 'p')
                destination = Path(args.output) / 'static' / dataset_name / family / f'seed_{seed}_{args.profile}_{signature}' / f'alpha_{alpha_label}_{args.score}'
                destination.mkdir(parents=True, exist_ok=True)
                calibrator = ProbabilityCalibrator(config['calibrator_penalty']).fit(p[probability_ix], data.y[probability_ix])
                calibrated = calibrator.predict_proba(p)
                # Common, label-independent randomization across model families for paired comparisons.
                uniforms = np.random.default_rng(np.random.SeedSequence([seed, 19073])).random(len(data.y))
                conformal = SourceMondrian(args.alpha, args.score).fit(calibrated[conformal_ix], data.y[conformal_ix], uniforms[conformal_ix])
                windows, sets = [], []
                for t, indices in enumerate(data.windows):
                    issued_sets = conformal.predict(calibrated[indices], uniforms[indices])
                    sets.append(issued_sets)
                    metrics = evaluate(calibrated[indices], data.y[indices], issued_sets, args.alpha)
                    windows.append({'window': t + 1, **metrics})
                target = data.target
                target_sets = np.concatenate(sets)
                summary = {'dataset': dataset_name, 'model': family, 'seed': seed, 'profile': args.profile,
                           'sensitivity': args.sensitivity, 'method': 'B0_static_source', 'alpha': args.alpha,
                           'score': args.score, 'labels_acquired': 0,
                           'training_seconds': train_info['runtime_seconds'],
                           **evaluate(calibrated[target], data.y[target], target_sets, args.alpha)}
                write_json(destination / 'summary.json', summary)
                write_json(destination / 'window_metrics.json', windows)
                write_json(destination / 'raw_base_metrics.json', evaluate(p[target], data.y[target]))
                write_json(destination / 'conformal.json', {'class_names': data.classes, 'counts': conformal.counts,
                                                          'thresholds': [str(q) if np.isinf(q) else float(q) for q in conformal.thresholds],
                                                          'infinite_cutoff_classes': np.flatnonzero(np.isinf(conformal.thresholds)),
                                                          'coverage_statement': 'source split-conformal construction; target drift coverage is empirical'})
                write_json(destination / 'configuration.json', {'config': config, 'model_signature': model_signature,
                                                              'runtime_python': sys.version, 'runtime_platform': platform.platform()})
                write_json(destination / 'calibration_fit.json', calibrator.fit_info)
                joblib.dump(calibrator, destination / 'probability_calibrator.joblib')
                np.savez_compressed(destination / 'source_roles.npz', train=data.train, probability_fit=probability_ix,
                                    conformal_fit=conformal_ix, target=target)
                np.savez_compressed(destination / 'target_predictions.npz', row_id=target,
                                    window=np.concatenate([np.full(len(ix), t + 1) for t, ix in enumerate(data.windows)]),
                                    raw_probability=p[target], calibrated_probability=calibrated[target],
                                    prediction_set=target_sets, aps_uniform=uniforms[target])
                # Full labels are kept in a separate evaluator file, not in the prediction cache.
                np.savez_compressed(destination / 'evaluation_labels.npz', row_id=target, y=data.y[target])
                flat = [{k: v for k, v in record.items() if not isinstance(v, dict)} for record in windows]
                pd.DataFrame(flat).to_csv(destination / 'window_metrics.csv', index=False)
                if not getattr(args,'no_plots',False):
                    figures(calibrated[target], data.y[target], windows, destination / 'figures',
                            f'{dataset_name.title()} | {family} | Static source | {args.profile}', args.alpha)
                all_summaries.append(summary)
                print('STATIC_RESULT ' + json.dumps({k: summary[k] for k in ['dataset', 'model', 'seed', 'coverage', 'set_size', 'brier_sum_over_classes', 'auroc']}), flush=True)
    write_json(Path(args.output) / 'last_static_summaries.json', all_summaries)
    print('STATIC_RUN_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description='DTRA source-model and static reliability experiments')
    sub = parser.add_subparsers(dest='command', required=True)
    for action in ['audit', 'static']:
        child = sub.add_parser(action)
        child.add_argument('--data-dir', type=Path, default=ROOT / 'data/raw')
        child.add_argument('--output', type=Path, default=ROOT / 'artifacts')
        child.add_argument('--datasets', default='all' if action == 'audit' else 'electricity')
        if action == 'static':
            child.add_argument('--models', default='xgboost')
            child.add_argument('--seeds', default='0')
            child.add_argument('--profile', choices=['smoke', 'primary'], default='primary')
            child.add_argument('--threads', type=int, default=12)
            child.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto', help='FT-Transformer device; tree models use CPU')
            child.add_argument('--alpha', type=float, default=0.05)
            child.add_argument('--score', choices=['aps_randomized', 'lac', 'aps_deterministic'], default='aps_randomized')
            child.add_argument('--sensitivity', action='store_true', help='Remove day/period for Electricity; day/month for Bank')
            child.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if args.command == 'audit':
        audit(args)
    else:
        if args.threads < 1 or not 0 < args.alpha < 1:
            parser.error('Use positive threads and alpha in (0,1)')
        static_run(args)


if __name__ == '__main__':
    main()
