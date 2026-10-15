import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp


def probabilities(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[1] < 2 or not np.isfinite(p).all():
        raise ValueError('Expected finite N by K probabilities')
    p = np.clip(p, 1e-12, 1)
    return p / p.sum(axis=1, keepdims=True)


class ProbabilityCalibrator:
    """Monotone beta (K=2) or regularized log-linear Dirichlet (K>2)."""
    def __init__(self, penalty=0.001):
        self.penalty = penalty

    def fit(self, p, y, sample_weight=None):
        p = probabilities(p)
        y = np.asarray(y, dtype=int)
        self.k = p.shape[1]
        if len(y) != len(p) or np.any(y < 0) or np.any(y >= self.k):
            raise ValueError('Invalid calibration labels')
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError('Invalid calibration weights')
        w = w / w.sum()
        if self.k == 2:
            self.kind = 'monotone_beta'
            design = np.column_stack([np.log(p[:, 1]), -np.log(p[:, 0]), np.ones(len(p))])
            initial = np.array([1., 1., 0.])
            def objective(theta):
                z = design @ theta
                loss = np.sum(w * (np.logaddexp(0., z) - y * z))
                gradient = design.T @ (w * (expit(z) - y))
                delta = theta - initial
                return loss + self.penalty * np.dot(delta, delta), gradient + 2 * self.penalty * delta
            bounds = [(1e-8, None), (1e-8, None), (None, None)]
        else:
            self.kind = 'regularized_dirichlet'
            design = np.log(p)
            target = np.eye(self.k)[y]
            initial = np.r_[np.eye(self.k).ravel(), np.zeros(self.k)]
            def objective(theta):
                matrix, offset = theta[:self.k**2].reshape(self.k, self.k), theta[self.k**2:]
                logits = design @ matrix.T + offset
                log_p = logits - logsumexp(logits, axis=1, keepdims=True)
                error = w[:, None] * (np.exp(log_p) - target)
                gradient = np.r_[(error.T @ design).ravel(), error.sum(axis=0)]
                delta = theta - initial
                return -np.sum(w * log_p[np.arange(len(y)), y]) + self.penalty * np.dot(delta, delta), gradient + 2 * self.penalty * delta
            bounds = None
        result = minimize(objective, initial, jac=True, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 1500, 'ftol': 1e-12})
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f'Calibration failed to converge: {result.message}')
        self.theta = result.x
        self.fit_info = {'kind': self.kind, 'n': len(y), 'objective': float(result.fun),
                         'iterations': int(result.nit), 'penalty': self.penalty, 'converged': bool(result.success)}
        return self

    def predict_proba(self, p):
        p = probabilities(p)
        if self.k == 2:
            design = np.column_stack([np.log(p[:, 1]), -np.log(p[:, 0]), np.ones(len(p))])
            positive = expit(design @ self.theta)
            return probabilities(np.column_stack([1 - positive, positive]))
        matrix, offset = self.theta[:self.k**2].reshape(self.k, self.k), self.theta[self.k**2:]
        z = np.log(p) @ matrix.T + offset
        return probabilities(np.exp(z - logsumexp(z, axis=1, keepdims=True)))
