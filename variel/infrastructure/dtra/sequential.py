"""Prospective window simulation. The policy never receives unrevealed target labels."""
import math
import time
import numpy as np
from .prequential_mixture import PrequentialMixture
from .acquisition import Budget, allocate, sample_remainder
from .reliability import ReliabilityLayer


METHODS = ['b0', 'periodic', 'c2st_fixed', 'uncertainty', 'dtra']


def simulate(p, source_labels, source_roles, windows, uniforms, monitor, oracle, config, seed, method):
    if method not in METHODS:
        raise ValueError(method)
    fit_source, score_source, monitor_source = source_roles
    target = np.concatenate(windows)
    config = dict(config)
    if method == 'b0': config['target_calibrator'] = 'legacy'
    config['deployment_windows'] = len(windows)
    config['inverse_fit_fraction_bound'] = max(len(ix)/max(1,min(len(ix)-1,math.floor(len(ix)*config['target_probability_role_fraction']))) for ix in windows)
    role_rng = np.random.default_rng(np.random.SeedSequence([seed, 7601]))
    roles = np.full(len(p), -1, int)
    for ix in windows:
        shuffled = role_rng.permutation(ix)
        boundary = max(1, min(len(ix) - 1, math.floor(len(ix) * config['target_probability_role_fraction'])))
        roles[shuffled[:boundary]], roles[shuffled[boundary:]] = 0, 1
    layer = ReliabilityLayer(p, source_labels, fit_source, score_source, uniforms, roles, config)
    budget = Budget.create(len(target), config['budget_fraction'], config['sentinel_share'])
    if budget.sentinel_reserve < len(windows) and method != 'b0':
        raise ValueError('Budget too small for a sentinel observation in every window')
    sentinel_counts = allocate(budget.sentinel_reserve, list(map(len, windows)))
    eligible = max(0, len(windows) - 1 - config['delay'])
    capacities = [len(windows[t]) - sentinel_counts[t] if t < eligible else 0 for t in range(len(windows))]
    fixed_counts = allocate(min(budget.adaptive_remaining, sum(capacities)), capacities)
    prediction, sets = np.zeros_like(p), np.zeros_like(p, dtype=bool)
    reference = monitor_source
    cohorts, logs, update_logs, designs, audit_history, guard_logs = {}, [], [], [], [], []
    mixture_logs = []
    start = time.perf_counter()

    def acquire(origin, clock, severity):
        c = cohorts[origin]
        if c['adaptive_requested'] or clock + config['delay'] >= len(windows) - 1:
            return 0
        remaining = np.setdiff1d(c['indices'], c['sentinel_ids'], assume_unique=True)
        if method in {'periodic', 'c2st_fixed'}:
            wanted = int(fixed_counts[origin])
        else:
            rate = config['rate_min'] + (config['rate_max'] - config['rate_min']) * severity
            wanted = math.ceil(rate * len(c['indices']))
            if config.get('budget_pacing', False):
                useful_rounds = len(windows) - 1 - config['delay'] - clock
                wanted = max(wanted, math.ceil(budget.adaptive_remaining / useful_rounds))
        count = min(wanted, budget.adaptive_remaining, len(remaining))
        if count <= 0:
            return 0
        design = 'hybrid' if method == 'dtra' else ('uncertainty' if method == 'uncertainty' else 'random')
        rng = np.random.default_rng(np.random.SeedSequence([seed, origin, 9107]))
        ix, pi, details = sample_remainder(remaining, prediction[remaining], count, rng, design, config['max_strata'])
        budget.debit('adaptive', count)
        oracle.request(ix, 'adaptive', origin, clock, config['delay'], pi)
        c['adaptive_requested'], c['probability_design'] = True, details['probability_design']
        designs.append({'origin_window': origin + 1, 'request_after_window': clock + 1, **details})
        return count

    for t, ix in enumerate(windows):
        if isinstance(layer.calibrator, PrequentialMixture):
            layer.calibrator.active_lane = t % layer.calibrator.lanes
        # Labels with arrival_after_window=t are still unavailable for these predictions.
        prediction[ix], sets[ix] = layer.predict(ix)
        guard_probe = layer.guard_probe(ix)
        mixture_probe = layer.mixture_probe(ix,t)
        issued_version = layer.version
        d = 0.0
        feature_alarm, severity = False, 0.0
        if method not in {'b0', 'periodic'}:
            d = monitor.features.score(reference, ix)
            feature_alarm, severity, _ = monitor.decision(d, 0, sentinel_counts[t])
        cohorts[t] = {'indices': ix, 'observed': [], 'sentinel_count': int(sentinel_counts[t]),
                      'sentinel_ids': np.array([], int), 'adaptive_requested': False, 'adaptive_arrived': False,
                      'probability_design': True, 'armed': feature_alarm, 'feature_score': d}
        cohorts[t]['guard_probe'] = guard_probe
        cohorts[t]['mixture_probe'] = mixture_probe
        before_adaptive = budget.adaptive_used
        if method != 'b0':
            rng = np.random.default_rng(np.random.SeedSequence([seed, t, 5701]))
            sentinel = rng.choice(ix, int(sentinel_counts[t]), replace=False)
            cohorts[t]['sentinel_ids'] = sentinel
            budget.debit('sentinel', len(sentinel))
            oracle.request(sentinel, 'sentinel', t, t, config['delay'], np.full(len(sentinel), len(sentinel) / len(ix)))
            if method == 'periodic' or feature_alarm:
                acquire(t, t, severity)
        need_update, max_harm, harm_alarm, dual_arrival_alarm = False, 0.0, False, False
        arrivals = 0
        # Re-enter for zero-delay second-stage requests; never revise issued predictions.
        while True:
            due = oracle.release(t)
            if not due:
                break
            for job in due:
                origin, rows, y = job['origin'], job['indices'], job['labels']
                c = cohorts[origin]
                arrivals += len(rows)
                for row, label, pi in zip(rows, y, job['pi']):
                    c['observed'].append({'row_id': int(row), 'label': int(label), 'pi': float(pi), 'kind': job['kind']})
                if job['kind'] == 'adaptive':
                    c['adaptive_arrived'] = True
                    need_update = True
                else:
                    if c['mixture_probe'] is not None:
                        item = layer.calibrator.audit(c['mixture_probe'], rows, y, c['sentinel_count'])
                        item.update({'audit_after_window':t+1,'next_same_lane_window':origin+config['delay']+2})
                        mixture_logs.append(item)
                        c['mixture_probe'] = None
                        layer.version += 1
                        layer.calibration_version += 1
                    if config.get('calibration_guard', False):
                        gate = layer.audit_challenger(c['guard_probe'], rows, y, len(windows))
                        guard_logs.append({'origin_window': origin + 1, 'audit_after_window': t + 1,
                                           'effective_from_window': t + 2, **gate})
                        need_update = need_update or gate['accepted']
                    # Do not retain population-sized probe arrays after their
                    # single audit, or allow the same sentinel set to be reused.
                    c['guard_probe'] = None
                    missed = (~sets[rows, y]).astype(float)
                    loss = np.sum((prediction[rows] - np.eye(p.shape[1])[y]) ** 2, axis=1)
                    audit_history.append((origin, missed, loss))
                    audit_history = audit_history[-config['buffer_windows']:]
                    all_m = np.concatenate([a[1] for a in audit_history])
                    all_l = np.concatenate([a[2] for a in audit_history])
                    h = monitor.harm(all_m, all_l)
                    max_harm = max(max_harm, h)
                    alarm, severity, _ = monitor.decision(c['feature_score'], h, len(all_m))
                    if method in {'dtra', 'uncertainty'} and alarm:
                        h_only, _, _ = monitor.decision(0, h, len(all_m))
                        harm_alarm = harm_alarm or h_only
                        dual_arrival_alarm, c['armed'], need_update = True, True, True
                        acquire(origin, t, severity)
                    if c['armed'] or method == 'periodic':
                        need_update = True
        updated = False
        if need_update:
            updated = layer.update(cohorts, t)
            if updated:
                latest = max(u for u, c in cohorts.items() if c['observed'])
                reference = windows[latest]
                audit_history = []
                update_logs.append({'after_window': t + 1, 'effective_from_window': t + 2,
                                    **layer.last_diagnostics})
        if budget.sentinel_used + budget.adaptive_used > budget.ceiling:
            raise AssertionError('Global label ceiling violated')
        pending = sum(len(job['indices']) for job in oracle.jobs)
        logs.append({'window': t + 1, 'issued_layer_version': issued_version, 'feature_score': d,
                     'feature_alarm': feature_alarm, 'reliability_alarm_on_arrival': harm_alarm,
                     'dual_alarm_on_audit_arrival': dual_arrival_alarm,
                     'max_arrived_audit_harm': max_harm, 'new_adaptive_requests': budget.adaptive_used - before_adaptive,
                     'labels_arrived': arrivals, 'pending_labels': pending,
                     'cumulative_sentinel_requests': budget.sentinel_used, 'cumulative_adaptive_requests': budget.adaptive_used,
                     'cumulative_label_requests': budget.sentinel_used + budget.adaptive_used,
                     'updated_after_window': updated})
    return {'probabilities': prediction[target], 'sets': sets[target], 'row_id': target, 'roles': roles,
            'window_logs': logs, 'update_logs': update_logs, 'acquisition_designs': designs,
            'calibration_guard_logs': guard_logs,
            'mixture_audit_logs': mixture_logs,
            'request_records': oracle.records, 'budget_ceiling': budget.ceiling,
            'labels_acquired': budget.sentinel_used + budget.adaptive_used,
            'sentinel_labels': budget.sentinel_used, 'adaptive_labels': budget.adaptive_used,
            'pending_labels': sum(len(job['indices']) for job in oracle.jobs),
            'updates': len(update_logs), 'runtime_seconds': time.perf_counter() - start}
