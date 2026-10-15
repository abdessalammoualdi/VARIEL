"""Disjoint probability/score roles and explicitly empirical target quantiles."""
import math
import numpy as np
from .calibration import ProbabilityCalibrator
from .conformal import candidate_scores, split_quantile, empirical_weighted_quantile
from .calibration_guard import ProbabilityMixture, brier_gate
from .residual_calibration import ResidualCalibration
from .prequential_mixture import PrequentialMixture


class ReliabilityLayer:
    def __init__(self, p, source_labels, probability_indices, score_indices, uniforms, roles, config):
        self.p, self.labels = p, dict(source_labels)
        self.fit_source, self.score_source = probability_indices, score_indices
        self.uniforms, self.roles, self.config = uniforms, roles, config
        self.k, self.version = p.shape[1], 0
        self.calibrator = ProbabilityCalibrator(config['calibrator_penalty']).fit(
            p[probability_indices], [self.labels[int(i)] for i in probability_indices])
        self.source_calibrator = self.calibrator
        if config.get('target_calibrator') == 'prequential_mixture':
            self.calibrator = PrequentialMixture(self.source_calibrator, config['delay']+1,
                                config['deployment_windows'], config['inverse_fit_fraction_bound'], config['calibrator_penalty'], config.get('uniform_mixture',False))
        self.calibration_version = 0
        self.challenger = None
        self.challenger_clock = None
        self.source_thresholds = self.source_cutoffs()
        self.thresholds = self.source_thresholds.copy()
        self.last_diagnostics = {'target_effective_size': [0.] * self.k, 'fallback_classes': list(range(self.k))}

    def source_cutoffs(self):
        ix = self.score_source
        y = np.array([self.labels[int(i)] for i in ix])
        scores = candidate_scores(self.score_probabilities(ix), self.uniforms[ix], self.config['score'])
        s = scores[np.arange(len(ix)), y]
        if self.config['class_conditional']:
            return np.array([split_quantile(s[y == k], self.config['alpha']) for k in range(self.k)])
        return np.full(self.k, split_quantile(s, self.config['alpha']))

    def predict(self, indices):
        p = self.calibrator.predict_proba(self.p[indices])
        scores = candidate_scores(self.score_probabilities(indices), self.uniforms[indices], self.config['score'])
        return p, scores <= self.thresholds[None, :]

    def score_probabilities(self, indices):
        model = self.source_calibrator if self.config.get('fixed_score_map', False) else self.calibrator
        return model.predict_proba(self.p[indices])

    def mixture_probe(self, indices, clock):
        if not isinstance(self.calibrator, PrequentialMixture):
            return None
        fit = indices[self.roles[indices] == 0]
        return {'fit_ids': fit, 'expert_probabilities': self.calibrator.component_probabilities(self.p[fit]),
                'fit_fraction': len(fit)/len(indices), 'lane': clock % self.calibrator.lanes,
                'origin_window': clock+1}

    def guard_probe(self, indices):
        """Snapshot both maps before this window's sentinel sample is drawn."""
        if not self.config.get('calibration_guard', False) or self.challenger is None:
            return None
        ids = np.asarray(indices)[self.roles[indices] == 0]
        return {'ids': ids, 'incumbent_version': self.calibration_version,
                'challenger_fit_after_window': self.challenger_clock,
                'challenger_map': self.challenger,
                'incumbent_p': self.calibrator.predict_proba(self.p[ids]),
                'challenger_p': self.challenger.predict_proba(self.p[ids])}

    def audit_challenger(self, probe, rows, labels, max_looks):
        if probe is None:
            return {'accepted': False, 'reason': 'no_challenger_at_prediction_time'}
        if probe['incumbent_version'] != self.calibration_version:
            return {'accepted': False, 'reason': 'stale_incumbent'}
        if self.config.get('label_noise', 0.) > 0:
            return {'accepted': False, 'reason': 'clean_label_bound_disabled_for_noisy_audit'}
        positions = {int(row): j for j, row in enumerate(probe['ids'])}
        selected = [(positions[int(row)], int(y)) for row, y in zip(rows, labels) if int(row) in positions]
        ix = np.array([x[0] for x in selected], int)
        y = np.array([x[1] for x in selected], int)
        result = brier_gate(probe['incumbent_p'], probe['challenger_p'], ix, y,
                            self.config.get('guard_delta', .05) / max_looks)
        result.update({'incumbent_version': probe['incumbent_version'],
                       'challenger_fit_after_window': probe['challenger_fit_after_window'],
                       'audit_row_ids': [int(probe['ids'][j]) for j in ix]})
        if result['accepted']:
            self.calibrator = ProbabilityMixture(self.calibrator, probe['challenger_map'], result['mixture_fraction'])
            self.calibration_version += 1
        return result

    def window_weights(self, cohort, role):
        """Conditional design weights; NOT an unconditional triggered-risk claim."""
        items = [item for item in cohort['observed'] if self.roles[item['row_id']] == role]
        if not items:
            return [], []
        total, population = len(cohort['indices']), int(np.sum(self.roles[cohort['indices']] == role))
        if cohort['adaptive_arrived']:
            if cohort['probability_design'] and self.config['use_ipw']:
                weights = [1 / population if x['kind'] == 'sentinel' else 1 / (population * x['pi']) for x in items]
            else:
                weights = [1 / len(items)] * len(items)
        else:
            weights = [total / (cohort['sentinel_count'] * population)] * len(items)
        return [x['row_id'] for x in items], weights

    def update(self, cohorts, clock):
        recent = sorted([u for u, c in cohorts.items() if c['observed']])[-self.config['buffer_windows']:]
        if not recent:
            return False
        fit_ids, fit_weights, score_ids, score_weights, target_flags = [], [], [], [], []
        for source, ids, weights in [(self.fit_source, fit_ids, fit_weights), (self.score_source, score_ids, score_weights)]:
            ids.extend(source.tolist())
            weights.extend([self.config['source_mass'] / len(source)] * len(source))
        target_flags.extend([False] * len(score_ids))
        for u in recent:
            cohort = cohorts[u]
            for item in cohort['observed']:
                self.labels[item['row_id']] = item['label']
            decay = self.config['decay'] ** (clock - u)
            for role, ids, weights in [(0, fit_ids, fit_weights), (1, score_ids, score_weights)]:
                new_ids, new_weights = self.window_weights(cohort, role)
                ids.extend(new_ids)
                weights.extend(np.asarray(new_weights) * decay)
                if role == 1:
                    target_flags.extend([True] * len(new_ids))
        fit_ids, score_ids = np.asarray(fit_ids, int), np.asarray(score_ids, int)
        fit_weights, score_weights = np.asarray(fit_weights), np.asarray(score_weights)
        target_flags = np.asarray(target_flags)
        if self.config['update_calibration'] and fit_weights.sum() > 0:
            family = self.config.get('target_calibrator', 'legacy')
            tw = fit_weights[len(self.fit_source):]
            effective = float(tw.sum()**2 / (tw @ tw)) if len(tw) and tw @ tw > 0 else 0.
            if family == 'prequential_mixture':
                calibrator = self.calibrator.fit(self.p[fit_ids], [self.labels[int(i)] for i in fit_ids], fit_weights, effective)
            else:
                calibrator = (ProbabilityCalibrator(self.config['calibrator_penalty']) if family == 'legacy' else
                              ResidualCalibration(self.source_calibrator, family, effective, self.config['calibrator_penalty']))
                calibrator.fit(self.p[fit_ids], [self.labels[int(i)] for i in fit_ids], fit_weights)
            if self.config.get('calibration_guard', False):
                self.challenger, self.challenger_clock = calibrator, clock + 1
            else:
                self.calibrator = calibrator
                self.calibration_version += 1
        if self.config['update_conformal'] and score_weights.sum() > 0:
            y = np.array([self.labels[int(i)] for i in score_ids])
            values = candidate_scores(self.score_probabilities(score_ids), self.uniforms[score_ids], self.config['score'])
            s = values[np.arange(len(y)), y]
            source_cutoffs = self.source_cutoffs()
            pooled = empirical_weighted_quantile(s[target_flags], score_weights[target_flags], 1 - self.config['alpha'])
            ess, fallback, thresholds = [], [], []
            for k in range(self.k):
                in_class = y == k if self.config['class_conditional'] else np.ones(len(y), bool)
                tw = score_weights[target_flags & in_class]
                n_eff = float(tw.sum() ** 2 / (tw @ tw)) if len(tw) and (tw @ tw) > 0 else 0.
                ess.append(n_eff)
                supported = self.config.get('target_supported_quantile', False) and np.any(target_flags & in_class)
                if n_eff < math.ceil(1 / self.config['alpha']) and not supported:
                    # A transparent fallback, not a class-coverage guarantee.
                    cutoff = max(source_cutoffs[k], pooled) if np.any(target_flags) else source_cutoffs[k]
                    if self.config.get('sparse_class_rule', 'legacy') == 'class_aware':
                        own_class = target_flags & in_class
                        if np.any(own_class):
                            own_cutoff = empirical_weighted_quantile(s[own_class], score_weights[own_class], 1 - self.config['alpha'])
                            cutoff = max(cutoff, own_cutoff)
                    fallback.append(k)
                else:
                    cutoff = empirical_weighted_quantile(s[in_class], score_weights[in_class], 1 - self.config['alpha'])
                thresholds.append(cutoff)
            self.thresholds = np.asarray(thresholds)
            self.last_diagnostics = {'target_effective_size': ess, 'fallback_classes': fallback}
        self.version += 1
        self.last_diagnostics.update({'version': self.version, 'retained_origin_windows': [u + 1 for u in recent],
                                    'calibration_version': self.calibration_version,
                                    'candidate_fitted': self.challenger is not None,
                                    'calibrator_fit': getattr(self.calibrator, 'fit_info', {}),
                                    'fixed_score_map': self.config.get('fixed_score_map', False),
                                    'calibration_guard': self.config.get('calibration_guard', False),
                                    'sparse_class_rule': self.config.get('sparse_class_rule', 'legacy'),
                                    'probability_observations': len(fit_ids), 'score_observations': len(score_ids),
                                    'thresholds': [str(v) if np.isinf(v) else float(v) for v in self.thresholds]})
        return True
