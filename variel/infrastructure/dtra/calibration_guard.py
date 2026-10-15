"""Prospective Brier safeguard. Its bound concerns the completed audit population.

The incumbent and challenger must be fixed before an SRS sentinel draw. Only
probability-fit-role sentinels enter the gate; conformal-role labels never do.
Nothing in this module certifies risk or coverage in an unseen future window.
"""
import numpy as np
from .calibration import probabilities


class ProbabilityMixture:
    """A fixed convex combination of two previously fitted probability maps."""
    def __init__(self, incumbent, challenger, fraction):
        if not 0 <= fraction <= 1:
            raise ValueError('Mixture fraction must be in [0,1]')
        self.incumbent, self.challenger, self.fraction = incumbent, challenger, float(fraction)
        self.k = incumbent.k

    def predict_proba(self, p):
        return probabilities((1 - self.fraction) * self.incumbent.predict_proba(p)
                             + self.fraction * self.challenger.predict_proba(p))


def brier_gate(incumbent, challenger, sampled_positions, labels, delta):
    """Hoeffding upper bound for A in R(p+lambda*d)-R(p)=lambda*A+lambda²*C.

    p,q contain ALL rows in the fixed audit population, not just queried rows.
    sampled_positions must be an SRS without replacement within that population.
    C is computed exactly from unlabeled predictions. The range of A_i across
    every possible label and every population row is also known without labels.
    The one-sided SRS bound holds simultaneously for all lambda in [0,1].
    """
    p, q = probabilities(incumbent), probabilities(challenger)
    ix, y = np.asarray(sampled_positions, int), np.asarray(labels, int)
    if (p.shape != q.shape or not len(p) or ix.shape != y.shape or ix.ndim != 1
            or len(np.unique(ix)) != len(ix) or np.any(ix < 0) or np.any(ix >= len(p))
            or np.any(y < 0) or np.any(y >= p.shape[1]) or not 0 < delta < 1):
        raise ValueError('Invalid Brier gate input')
    d = q - p
    known = 2 * np.sum(p * d, axis=1)
    lower = float(np.min(known - 2 * d.max(axis=1)))
    upper = float(np.max(known - 2 * d.min(axis=1)))
    curvature = float(np.mean(np.sum(d * d, axis=1)))
    result = {'audit_population_n': len(p), 'audit_n': len(ix), 'delta': float(delta),
              'coefficient_range': [lower, upper], 'curvature': curvature,
              'mixture_fraction': 0., 'accepted': False,
              'bound_scope': 'completed probability-fit-role population; no future guarantee'}
    if not len(ix):
        return {**result, 'reason': 'no_probability_role_sentinels'}
    estimate = float(np.mean(known[ix] - 2 * d[ix, y]))
    # Sampling without replacement satisfies this Hoeffding bound. A census
    # reveals A exactly. No unsupported ESS substitution or normal approximation.
    radius = 0. if len(ix) == len(p) else (upper - lower) * np.sqrt(np.log(1 / delta) / (2 * len(ix)))
    bound = min(upper, estimate + radius)
    fraction = float(np.clip(-bound / (2 * curvature), 0, 1)) if curvature > 1e-20 else 0.
    result.update({'coefficient_estimate': estimate, 'coefficient_upper': float(bound),
                   'radius': float(radius), 'mixture_fraction': fraction,
                   'risk_change_upper': float(fraction * bound + fraction**2 * curvature),
                   'accepted': fraction > 1e-12,
                   'reason': 'audit_supported_blend' if fraction > 1e-12 else 'insufficient_evidence'})
    return result
