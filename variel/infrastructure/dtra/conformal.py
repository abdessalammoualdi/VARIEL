import math
import numpy as np
from .calibration import probabilities


def candidate_scores(p, uniform=None, score='aps_randomized'):
    p = probabilities(p)
    if score == 'lac':
        return 1 - p
    if score not in {'aps_randomized', 'aps_deterministic'}:
        raise ValueError('Unknown conformal score')
    order = np.argsort(-p, axis=1, kind='stable')
    cumulative = np.cumsum(np.take_along_axis(p, order, axis=1), axis=1)
    inverse = np.argsort(order, axis=1)
    result = np.take_along_axis(cumulative, inverse, axis=1)
    if score == 'aps_randomized':
        uniform = np.asarray(uniform, dtype=float)
        if uniform.shape != (len(p),) or not np.all((uniform >= 0) & (uniform <= 1)):
            raise ValueError('Provide one independent Uniform[0,1] variate per row')
        result = result - (1 - uniform[:, None]) * p
    return np.clip(result, 0, 1)


def split_quantile(scores, alpha):
    """Unweighted source split-conformal rank, including the infinity atom."""
    if not 0 < alpha < 1:
        raise ValueError('alpha must be in (0,1)')
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    rank = math.ceil((n + 1) * (1 - alpha))
    return float('inf') if rank > n else float(np.partition(scores, rank - 1)[rank - 1])


def empirical_weighted_quantile(scores, weights, level):
    """Empirical quantile only. No finite-sample coverage claim for these weights."""
    s, w = np.asarray(scores, float), np.asarray(weights, float)
    if s.shape != w.shape or not 0 < level <= 1 or np.any(w < 0) or not np.isfinite(w).all() or not np.isfinite(s).all():
        raise ValueError('Invalid weighted quantile input')
    positive = w > 0
    s, w = s[positive], w[positive]
    if not len(w):
        return float('inf')
    order = np.argsort(s, kind='stable')
    index = np.searchsorted(np.cumsum(w[order]) / w.sum(), level, side='left')
    return float(s[order[min(index, len(order) - 1)]])


class SourceMondrian:
    def __init__(self, alpha=0.05, score='aps_randomized'):
        self.alpha, self.score = alpha, score

    def fit(self, p, y, uniform):
        scores = candidate_scores(p, uniform, self.score)
        y = np.asarray(y, dtype=int)
        self.counts = np.bincount(y, minlength=p.shape[1])
        true_scores = scores[np.arange(len(y)), y]
        self.thresholds = np.asarray([split_quantile(true_scores[y == k], self.alpha) for k in range(p.shape[1])])
        return self

    def predict(self, p, uniform):
        return candidate_scores(p, uniform, self.score) <= self.thresholds[None, :]
