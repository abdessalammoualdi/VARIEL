"""Budget accounting and explicit finite-population sampling designs."""
from dataclasses import dataclass
import math
import numpy as np


def allocate(total, capacities, masses=None, minimum=0):
    """Deterministic capped allocation, with largest-remainder tie breaking."""
    caps = np.asarray(capacities, dtype=int)
    if total < 0 or np.any(caps < 0) or total > caps.sum():
        raise ValueError('Infeasible allocation')
    mass = np.ones(len(caps)) if masses is None else np.asarray(masses, float)
    if mass.shape != caps.shape or np.any(mass < 0) or not np.isfinite(mass).all():
        raise ValueError('Invalid allocation masses')
    result = np.minimum(caps, minimum)
    if result.sum() > total:
        raise ValueError('Budget cannot satisfy positive stratum quotas')
    while result.sum() < total:
        room = caps - result
        active = room > 0
        weights = mass * active
        if weights.sum() == 0:
            weights = active.astype(float)
        desired = (total - result.sum()) * weights / weights.sum()
        addition = np.minimum(room, np.floor(desired).astype(int))
        if addition.sum() == 0:
            # A final fractional allocation. Stable ties use the stratum index.
            order = np.argsort(-np.where(active, desired, -1), kind='stable')
            order = order[:min(int(total - result.sum()), int(active.sum()))]
            addition[order] = 1
        result += addition
    return result


@dataclass
class Budget:
    ceiling: int
    sentinel_reserve: int
    sentinel_used: int = 0
    adaptive_used: int = 0

    @classmethod
    def create(cls, n, fraction, share):
        ceiling = math.floor(n * fraction)
        return cls(ceiling, math.floor(ceiling * share))

    @property
    def adaptive_remaining(self):
        return self.ceiling - self.sentinel_reserve - self.adaptive_used

    def debit(self, kind, count):
        if count < 0:
            raise ValueError('Negative label request')
        if kind == 'sentinel':
            if self.sentinel_used + count > self.sentinel_reserve:
                raise ValueError('Sentinel reserve exceeded')
            self.sentinel_used += count
        elif kind == 'adaptive':
            if count > self.adaptive_remaining:
                raise ValueError('Adaptive budget exceeded')
            self.adaptive_used += count
        else:
            raise ValueError(kind)


def sample_remainder(indices, probabilities, count, rng, design='hybrid', max_strata=8):
    """pi is conditional on the realized sentinel set, trigger and predictions.

    Returns selected global row IDs, their pi, and population stratum records.
    Deterministic uncertainty acquisition is explicitly NOT an IPW design.
    """
    indices = np.asarray(indices, dtype=int)
    p = np.asarray(probabilities, float)
    if p.shape[0] != len(indices) or not 0 < count <= len(indices):
        raise ValueError('Invalid acquisition request')
    confidence, prediction = p.max(1), p.argmax(1)
    if design == 'uncertainty':
        pos = np.argsort(confidence, kind='stable')[:count]
        return indices[pos], np.ones(count), {'design': design, 'probability_design': False,
                'population_ids': indices, 'inclusion_probabilities': np.isin(np.arange(len(indices)), pos).astype(float)}
    if design == 'random':
        groups = np.zeros(len(indices), dtype=int)
        quota = np.array([count])
        sizes = np.array([len(indices)])
    elif design == 'hybrid':
        frequencies = np.bincount(prediction, minlength=p.shape[1]) / len(prediction)
        priority = 0.5 * (1 - confidence) + 0.5 * (1 - frequencies[prediction])
        edges = np.unique(np.quantile(priority, np.linspace(0, 1, min(max_strata, count) + 1)[1:-1]))
        _, groups = np.unique(np.searchsorted(edges, priority, side='right'), return_inverse=True)
        sizes = np.bincount(groups)
        priority_mass = np.bincount(groups, weights=priority)
        preferential = priority_mass / priority_mass.sum() if priority_mass.sum() > 0 else sizes / sizes.sum()
        masses = 0.5 * sizes / sizes.sum() + 0.5 * preferential
        quota = allocate(count, sizes, masses, minimum=1)
    else:
        raise ValueError(design)
    population_pi = quota[groups] / sizes[groups]
    selected = np.concatenate([rng.choice(np.flatnonzero(groups == h), int(n), replace=False)
                               for h, n in enumerate(quota)])
    return indices[selected], population_pi[selected], {
        'design': design, 'probability_design': True, 'population_ids': indices,
        'strata': groups, 'stratum_sizes': sizes, 'stratum_quotas': quota,
        'inclusion_probabilities': population_pi}


class DelayedOracle:
    """Simulator-owned truth: policies receive only due, previously requested labels."""
    def __init__(self, truth, classes, seed=0, noise=0.0):
        self.__truth = np.asarray(truth, int).copy()
        self.classes, self.noise = classes, noise
        self.rng = np.random.default_rng(seed)
        self.jobs, self.requested, self.records = [], set(), []

    def request(self, indices, kind, origin, clock, delay, probabilities):
        indices = np.asarray(indices, int)
        if len(np.unique(indices)) != len(indices) or self.requested.intersection(indices.tolist()):
            raise ValueError('An observation was requested more than once')
        self.requested.update(indices.tolist())
        job = {'indices': indices, 'kind': kind, 'origin': origin, 'requested_after_window': clock,
               'arrival_after_window': clock + delay, 'pi': np.asarray(probabilities, float)}
        self.jobs.append(job)
        for i, pi in zip(indices, probabilities):
            self.records.append({'row_id': int(i), 'kind': kind, 'origin_window': origin + 1,
                                 'request_after_window': clock + 1, 'arrival_after_window': clock + delay + 1,
                                 'conditional_inclusion_probability': float(pi)})

    def release(self, clock):
        due = [job for job in self.jobs if job['arrival_after_window'] <= clock]
        self.jobs = [job for job in self.jobs if job['arrival_after_window'] > clock]
        for job in due:
            labels = self.__truth[job['indices']].copy()
            corrupt = self.rng.random(len(labels)) < self.noise
            offsets = self.rng.integers(1, self.classes, size=len(labels))
            labels[corrupt] = (labels[corrupt] + offsets[corrupt]) % self.classes
            job['labels'] = labels
        return due
