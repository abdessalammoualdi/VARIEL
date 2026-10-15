"""Source-referenced low-dimensional target updates; no coverage theorem."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from .calibration import probabilities


class ResidualCalibration:
    def __init__(self, source_calibrator, family, target_effective_size, penalty=.001):
        self.source_calibrator = source_calibrator
        self.family = family
        self.target_effective_size = float(target_effective_size)
        self.penalty = penalty

    def fit(self, p, y, sample_weight):
        p = self.source_calibrator.predict_proba(p)
        y = np.asarray(y, int)
        w = np.asarray(sample_weight, float)
        if len(y) != len(p) or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError('Invalid residual calibration data')
        w = w / w.sum()
        self.k = p.shape[1]
        if self.family == 'prevalence':
            self.prior = probabilities(np.bincount(y, weights=w, minlength=self.k)[None, :])[0]
            self.fit_info = {'kind': self.family, 'n': len(y), 'target_effective_size': self.target_effective_size}
            return self
        z = np.log(probabilities(p))
        use_offsets = self.family == 'support_residual'
        if not use_offsets and self.family != 'temperature':
            raise ValueError(self.family)
        offset_penalty = (self.k - 1) / max(self.target_effective_size, 1.)
        initial = np.r_[1., np.zeros(self.k)] if use_offsets else np.ones(1)
        truth = np.eye(self.k)[y]

        def objective(theta):
            scores = theta[0] * z
            if use_offsets:
                scores = scores + theta[1:]
            lp = scores - logsumexp(scores, axis=1, keepdims=True)
            error = w[:, None] * (np.exp(lp) - truth)
            loss = -np.sum(w * lp[np.arange(len(y)), y]) + self.penalty * (theta[0] - 1)**2
            grad = [np.sum(error * z) + 2 * self.penalty * (theta[0] - 1)]
            if use_offsets:
                loss += offset_penalty * np.dot(theta[1:], theta[1:])
                grad.extend(error.sum(axis=0) + 2 * offset_penalty * theta[1:])
            return loss, np.asarray(grad)

        bounds = [(1e-8, None)] + ([(None, None)] * self.k if use_offsets else [])
        result = minimize(objective, initial, jac=True, bounds=bounds, method='L-BFGS-B',
                          options={'maxiter': 1500, 'ftol': 1e-12})
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f'Residual calibration failed: {result.message}')
        self.theta = result.x
        self.fit_info = {'kind': self.family, 'n': len(y), 'target_effective_size': self.target_effective_size,
                         'inverse_temperature': float(self.theta[0]), 'degrees_of_freedom': self.k if use_offsets else 1,
                         'offsets': self.theta[1:].tolist(), 'offset_penalty': offset_penalty if use_offsets else None,
                         'objective': float(result.fun), 'converged': True}
        return self

    def predict_proba(self, p):
        if self.family == 'prevalence':
            return np.broadcast_to(self.prior, (len(p), self.k)).copy()
        scores = self.theta[0] * np.log(probabilities(self.source_calibrator.predict_proba(p)))
        if self.family == 'support_residual':
            scores += self.theta[1:]
        return probabilities(np.exp(scores - logsumexp(scores, axis=1, keepdims=True)))
