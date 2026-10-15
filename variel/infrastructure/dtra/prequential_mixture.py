"""Delay-lane AdaHedge using pre-draw fit-role sentinel snapshots.

AdaHedge: de Rooij et al., JMLR 15 (2014), equations 8--9 / Figure 1.
The expert algorithm is established; the sampling/delay integration is DTRA."""
import math
import numpy as np
from .calibration import probabilities, ProbabilityCalibrator
from .residual_calibration import ResidualCalibration


class SourceComposition:
    def __init__(self, source, residual):
        self.source, self.residual = source, residual
    def fit(self, p, y, w):
        self.residual.fit(self.source.predict_proba(p),y,w)
        return self
    def predict_proba(self,p):
        return self.residual.predict_proba(self.source.predict_proba(p))


class RawProbabilities:
    def predict_proba(self, p): return probabilities(p)


class PrequentialMixture:
    names = ('raw', 'source', 'temperature', 'support_residual', 'full_residual')
    def __init__(self, source, lanes, horizon, inverse_fit_fraction, penalty, uniform=False):
        self.source, self.penalty = source, penalty
        self.experts = [RawProbabilities()] + [source] * (len(self.names)-1)
        self.lanes, self.horizon = lanes, horizon
        self.cumulative_loss = np.zeros((lanes, len(self.names)))
        self.gaps = np.zeros(lanes)
        self.active_lane = 0
        self.uniform = uniform
        self.fit_info = {'kind': 'prequential_mixture', 'experts': list(self.names), 'aggregation': 'AdaHedge', 'lanes': lanes}

    def eta_for(self, lane):
        return math.inf if self.gaps[lane] == 0 else math.log(len(self.names))/self.gaps[lane]

    @staticmethod
    def mix(eta, cumulative):
        minimum = float(cumulative.min())
        values = ((cumulative == minimum).astype(float) if math.isinf(eta)
                  else np.exp(-eta * (cumulative-minimum)))
        potential = minimum if math.isinf(eta) else minimum-math.log(values.mean())/eta
        return values/values.sum(), potential

    def weights(self, lane=None):
        lane = self.active_lane if lane is None else lane
        if self.uniform: return np.full(len(self.names), 1/len(self.names))
        return self.mix(self.eta_for(lane), self.cumulative_loss[lane])[0]

    def observe_losses(self, lane, loss):
        eta = self.eta_for(lane)
        before = self.weights(lane)
        _, old = self.mix(eta, self.cumulative_loss[lane])
        self.cumulative_loss[lane] += loss
        _, new = self.mix(eta, self.cumulative_loss[lane])
        gap = max(0., float(before@loss)-(new-old))
        if not self.uniform: self.gaps[lane] += gap
        return {'eta': None if math.isinf(eta) else eta, 'eta_is_infinite': math.isinf(eta),
                'mixability_gap': gap, 'cumulative_gap': float(self.gaps[lane]),
                'estimated_hedge_loss': float(before@loss),
                'estimated_mix_loss': float(new-old),
                'weights_before': before.tolist(), 'weights_after': self.weights(lane).tolist()}

    def component_probabilities(self, p):
        return np.stack([m.predict_proba(p) for m in self.experts])

    def predict_proba(self, p):
        return probabilities(np.einsum('e,enk->nk', self.weights(), self.component_probabilities(p)))

    def fit(self, p, y, sample_weight, effective_size):
        self.experts = [RawProbabilities(), self.source,
            ResidualCalibration(self.source, 'temperature', effective_size, self.penalty).fit(p,y,sample_weight),
            ResidualCalibration(self.source, 'support_residual', effective_size, self.penalty).fit(p,y,sample_weight),
            SourceComposition(self.source, ProbabilityCalibrator(self.penalty)).fit(p,y,sample_weight)]
        self.fit_info.update({'n': len(y), 'target_effective_size': effective_size})
        return self

    def audit(self, snapshot, rows, labels, sentinel_count):
        positions = {int(row): j for j,row in enumerate(snapshot['fit_ids'])}
        selected = [(positions[int(row)], int(y)) for row,y in zip(rows,labels) if int(row) in positions]
        ix = np.asarray([a[0] for a in selected],int)
        y = np.asarray([a[1] for a in selected],int)
        p = snapshot['expert_probabilities'][:, ix, :]
        loss = np.sum((p - np.eye(p.shape[-1])[y][None, :, :])**2, axis=2) / 2
        estimate = loss.sum(axis=1) / (sentinel_count * snapshot['fit_fraction'])
        lane = snapshot['lane']
        info = self.observe_losses(lane, estimate)
        return {'origin_window': snapshot['origin_window'], 'lane': lane,
                'sentinel_count': int(sentinel_count), 'fit_sentinel_count': len(y),
                'audit_row_ids': [int(snapshot['fit_ids'][j]) for j in ix],
                'fit_fraction': snapshot['fit_fraction'], 'estimated_half_brier': estimate.tolist(),
                **info, 'expert_names': list(self.names)}
