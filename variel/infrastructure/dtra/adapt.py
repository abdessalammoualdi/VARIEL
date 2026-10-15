"""Run the first causal DTRA comparison using existing frozen-model caches."""
import argparse
import fcntl
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .acquisition import DelayedOracle
from .calibration import probabilities
from .cli import ROOT, DATASETS, MODELS, names, source_hashes
from .data import load_dataset, audit_dataset
from .io import file_hash, write_json
from .metrics import evaluate
from .monitoring import monitoring_roles, FeatureMonitor, ReliabilityMonitor
from .reliability import ReliabilityLayer
from .sequential import simulate, METHODS


def frozen_cache(root, data, family, seed, profile, hashes):
    matches = []
    for path in (Path(root) / 'models' / data.name / family).glob(f'seed_{seed}_{profile}_*/COMPLETE.json'):
        record = json.loads(path.read_text())
        if (record['profile'] == profile and record['seed'] == seed and record['data_sha256'] == hashes
            and not record['sensitivity'] and record['models_code_sha256'] == file_hash(ROOT / 'dtra/models.py')
            and record['data_code_sha256'] == file_hash(ROOT / 'dtra/data.py')):
            matches.append(path.parent)
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one completed {profile} cache for {data.name}/{family}/seed={seed}; found {len(matches)}. '
                         'No training will be started automatically. Check --cache-root or finish the static run.')
    folder = matches[0]
    with np.load(folder / 'frozen_probabilities.npz') as saved:
        if not np.array_equal(saved['row_id'], np.arange(len(data.y))):
            raise ValueError('Frozen prediction row identifiers do not match the dataset')
        p = probabilities(saved['probabilities'])
    if p.shape != (len(data.y), len(data.classes)):
        raise ValueError('Frozen probabilities have incorrect dimensions')
    return folder, p


def comparison_figure(records, destination, title, alpha):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'b0': '#626262', 'periodic': '#D97919', 'c2st_fixed': '#8060A4',
              'uncertainty': '#B64848', 'dtra': '#007E87'}
    labels = {'b0': 'Static', 'periodic': 'Periodic random', 'c2st_fixed': 'Feature trigger / fixed quota',
              'uncertainty': 'Uncertainty acquisition', 'dtra': 'DTRA'}
    with plt.rc_context({'font.family': 'DejaVu Serif', 'font.size': 10, 'text.color': 'black',
                         'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black'}):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
        panels = [('coverage', 'Empirical coverage'), ('worst_present_class_coverage', 'Worst present-class coverage'),
                  ('set_size', 'Mean prediction-set size'), ('cumulative_label_requests', 'Cumulative label requests')]
        for ax, (key, label) in zip(axes.flat, panels):
            for method, rows in records.items():
                ax.plot([r['window'] for r in rows], [r[key] for r in rows], label=labels[method],
                        color=colors[method], linewidth=2.2 if method == 'dtra' else 1.4, marker='o', markersize=3)
            if key in {'coverage', 'worst_present_class_coverage'}:
                ax.axhline(1 - alpha, color='black', linestyle='--', linewidth=1)
                ax.set_ylim(0, 1.03)
            ax.set(xlabel='Deployment window', ylabel=label)
            ax.grid(alpha=0.15)
            ax.spines[['top', 'right']].set_visible(False)
        handles, legend = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legend, loc='outside lower center', ncol=3, frameon=False)
        fig.suptitle(title)
        destination.mkdir(parents=True, exist_ok=True)
        for ext in ['pdf', 'svg', 'png']:
            fig.savefig(destination / f'adaptation_comparison.{ext}', dpi=300, bbox_inches='tight')
        plt.close(fig)


def run(args):
    config = json.loads((ROOT / 'configs/adaptation.json').read_text())
    for key in ['budget_fraction', 'delay', 'label_noise', 'alpha']:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.score:
        config['score'] = args.score
    config['calibration_guard'] = args.calibration_guard
    config['guard_delta'] = 0.05
    config['sparse_class_rule'] = args.sparse_class_rule
    config['target_calibrator'] = getattr(args, 'target_calibrator', 'legacy')
    config['fixed_score_map'] = getattr(args, 'fixed_score_map', False)
    config['budget_pacing'] = getattr(args, 'budget_pacing', False)
    config['target_supported_quantile'] = getattr(args, 'target_supported_quantile', False)
    config['uniform_mixture'] = getattr(args, 'uniform_mixture', False)
    if config['target_calibrator'] == 'prequential_mixture': config['protocol_version']='DTRA-development-v0.4-audited-AdaHedge'
    if config['target_calibrator'] == 'prequential_mixture' and (not config['fixed_score_map'] or config['calibration_guard']):
        raise ValueError('The prequential mixture requires fixed scores and does not use the old hard gate')
    if args.calibration_guard:
        config['protocol_version'] = 'DTRA-development-v0.3-guard'
    if args.calibration_guard and (args.no_calibration_update or args.no_conformal_update):
        raise ValueError('The guarded-joint arm requires calibration and score updates')
    for argument, setting in [('no_calibration_update', 'update_calibration'), ('no_conformal_update', 'update_conformal'),
                              ('no_ipw', 'use_ipw'), ('marginal', 'class_conditional')]:
        if getattr(args, argument):
            config[setting] = False
    if args.no_decay:
        config['decay'] = 1.0
    if args.no_source_anchor:
        config['source_mass'] = 0.0
    if not (0 < config['budget_fraction'] <= 0.10 and 0 < config['alpha'] < 1 and config['delay'] >= 0
            and 0 <= config['label_noise'] < 1 and args.threads > 0):
        raise ValueError('Invalid budget, significance, delay, noise or thread count')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    modules = ['adapt.py', 'acquisition.py', 'monitoring.py', 'reliability.py', 'sequential.py', 'calibration.py', 'calibration_guard.py', 'conformal.py', 'metrics.py', 'residual_calibration.py', 'prequential_mixture.py']
    code_hashes = {name: file_hash(ROOT / 'dtra' / name) for name in modules}
    config_id = hashlib.sha256(json.dumps({'config': config, 'code': code_hashes}, sort_keys=True).encode()).hexdigest()[:12]
    all_summaries = []
    for name in names(args.datasets, DATASETS):
        data = load_dataset(name, args.data_dir)
        config['deployment_windows'] = len(data.windows)
        config['inverse_fit_fraction_bound'] = max(len(ix)/max(1,min(len(ix)-1,int(len(ix)*config['target_probability_role_fraction']))) for ix in data.windows)
        audit_dataset(data)
        hashes = source_hashes(args.data_dir, name)
        if args.export_replay:
            for filename in hashes:
                dest = args.export_replay / 'data/raw' / filename
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(args.data_dir) / filename, dest)
        roles = monitoring_roles(data.source, data.y)
        known_source = {int(i): int(data.y[i]) for i in data.source}
        features = FeatureMonitor(data, roles[2], config, getattr(args,'detector_cache_root',None) or output / 'detector_cache')
        for family in names(args.models, MODELS):
            for seed in [int(s) for s in args.seeds.split(',')]:
                cache, p = frozen_cache(args.cache_root, data, family, seed, args.profile, hashes)
                print(f'FROZEN_CACHE_REUSED dataset={name} model={family} seed={seed} cache={cache.name}', flush=True)
                if args.export_replay:
                    replay = args.export_replay / 'artifacts/models' / name / family / cache.name
                    replay.mkdir(parents=True, exist_ok=True)
                    for filename in ['COMPLETE.json', 'training.json', 'frozen_probabilities.npz']:
                        shutil.copy2(cache / filename, replay / filename)
                uniforms = np.random.default_rng(np.random.SeedSequence([seed, 19073])).random(len(data.y))
                initial = ReliabilityLayer(p, known_source, roles[0], roles[1], uniforms, np.full(len(p), -1), config)
                monitor_p, monitor_sets = initial.predict(roles[2])
                monitor = ReliabilityMonitor(features, monitor_p, monitor_sets, data.y[roles[2]], config)
                comparison = {}
                parent = output / name / family / f'seed_{seed}_{args.profile}_{cache.name.split("_")[-1]}_{config_id}'
                write_json(parent / 'configuration.json', {'config': config, 'code_sha256': code_hashes,
                           'raw_sha256': hashes, 'frozen_model_cache': str(cache), 'seed': seed,
                           'scope': 'empirical prospective adaptation; primary external literature baselines are not included yet'})
                np.savez_compressed(parent / 'source_roles.npz', probability_fit=roles[0], score_fit=roles[1], monitor=roles[2])
                np.savez_compressed(parent / 'evaluation_labels.npz', row_id=data.target, y=data.y[data.target],
                                    window=np.concatenate([np.full(len(ix), t + 1) for t, ix in enumerate(data.windows)]))
                write_json(parent / 'class_names.json', dict(enumerate(data.classes)))
                baseline_raw = evaluate(p[data.target], data.y[data.target])
                write_json(parent / 'raw_base_metrics.json', baseline_raw)
                for method in names(args.methods, METHODS):
                    print(f'ADAPT_BEGIN dataset={name} model={family} seed={seed} method={method}', flush=True)
                    oracle = DelayedOracle(data.y, len(data.classes), np.random.SeedSequence([seed, 8101]), config['label_noise'])
                    result = simulate(p, known_source, roles, data.windows, uniforms, monitor, oracle, config, seed, method)
                    # Evaluation starts only after the policy has finished issuing all predictions.
                    metrics = evaluate(result['probabilities'], data.y[data.target], result['sets'], config['alpha'])
                    summary = {'dataset': name, 'model': family, 'seed': seed, 'method': method, 'profile': args.profile,
                               'budget_fraction': config['budget_fraction'], 'delay': config['delay'], 'label_noise': config['label_noise'],
                               'config_id': config_id, 'base_auroc': baseline_raw['auroc'],
                               **{k: result[k] for k in ['budget_ceiling', 'labels_acquired', 'sentinel_labels', 'adaptive_labels', 'pending_labels', 'updates', 'runtime_seconds']},
                               'labels_requested': result['labels_acquired'],
                               'labels_revealed': result['labels_acquired'] - result['pending_labels'],
                               'actual_label_fraction': result['labels_acquired'] / len(data.target), **metrics}
                    rows, offset = [], 0
                    for t, ix in enumerate(data.windows):
                        sl = slice(offset, offset + len(ix))
                        window_metrics = evaluate(result['probabilities'][sl], data.y[ix], result['sets'][sl], config['alpha'])
                        rows.append({**result['window_logs'][t], **window_metrics})
                        offset += len(ix)
                    destination = parent / method
                    write_json(destination / 'summary.json', summary)
                    write_json(destination / 'window_metrics.json', rows)
                    write_json(destination / 'updates.json', result['update_logs'])
                    write_json(destination / 'calibration_guard.json', result['calibration_guard_logs'])
                    write_json(destination / 'mixture_audits.json', result['mixture_audit_logs'])
                    write_json(destination / 'sampling_designs.json', result['acquisition_designs'])
                    pd.DataFrame(result['request_records']).to_csv(destination / 'label_requests.csv', index=False)
                    pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, dict)} for row in rows]).to_csv(destination / 'window_metrics.csv', index=False)
                    np.savez_compressed(destination / 'issued_predictions.npz', row_id=result['row_id'], probabilities=result['probabilities'],
                                        prediction_set=result['sets'], target_role=result['roles'][data.target], aps_uniform=uniforms[data.target],
                                        raw_probability=p[data.target])
                    all_summaries.append(summary)
                    comparison[method] = rows
                    print('ADAPT_RESULT ' + json.dumps({k: summary[k] for k in ['dataset', 'model', 'seed', 'method', 'coverage', 'worst_present_class_coverage', 'set_size', 'labels_acquired', 'auroc']}), flush=True)
                    write_json(output / 'latest_summaries.json', all_summaries)
                    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in all_summaries]).to_csv(output / 'latest_summary.csv', index=False)
                write_json(parent / 'monitor_thresholds.json', {str(n): q for n, q in monitor.threshold_cache.items()})
                if not args.no_plots:
                    comparison_figure(comparison, parent / 'figures', f'{name.title()} | {family} | seed {seed} | {args.profile}', config['alpha'])
                features.save()
    print('ADAPTATION_RUN_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/raw')
    parser.add_argument('--cache-root', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--detector-cache-root', type=Path, help='Reuse deterministic unlabeled C2ST calculations across jobs')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/adaptation')
    parser.add_argument('--datasets', default='all')
    parser.add_argument('--models', default='all')
    parser.add_argument('--methods', default='all')
    parser.add_argument('--seeds', default='0')
    parser.add_argument('--profile', choices=['smoke', 'primary'], default='primary')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--budget-fraction', type=float)
    parser.add_argument('--delay', type=int)
    parser.add_argument('--label-noise', type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--score', choices=['aps_randomized', 'aps_deterministic', 'lac'])
    parser.add_argument('--calibration-guard', action='store_true', help='Test candidates only on later, fit-role sentinel labels before deployment')
    parser.add_argument('--target-calibrator', choices=['legacy', 'temperature', 'support_residual', 'prevalence', 'prequential_mixture'], default='legacy')
    parser.add_argument('--fixed-score-map', action='store_true', help='Keep the source score map fixed while adapting probabilities and quantiles')
    parser.add_argument('--uniform-mixture', action='store_true', help='Ablation: keep equal expert weights')
    parser.add_argument('--budget-pacing', action='store_true', help='On an alarm, pace remaining labels over remaining useful request rounds')
    parser.add_argument('--target-supported-quantile', action='store_true', help='Use the empirical class quantile once target score labels of that class exist')
    parser.add_argument('--sparse-class-rule', choices=['legacy', 'class_aware'], default='legacy')
    parser.add_argument('--no-plots', action='store_true', help='Keep numerical evidence without rendering development figures')
    parser.add_argument('--export-replay', type=Path, help='Export public raw data and frozen probabilities for independent replay without training')
    for flag in ['no-calibration-update', 'no-conformal-update', 'no-ipw', 'no-decay', 'no-source-anchor', 'marginal']:
        parser.add_argument('--' + flag, action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'runner.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('An adaptation job already uses this output directory. Let it finish.')
        with threadpool_limits(limits=args.threads):
            run(args)


if __name__ == '__main__':
    main()
