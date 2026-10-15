import math
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from .calibration import probabilities


def ece_equal_frequency(confidence, outcomes, bins=15):
    order = np.argsort(confidence, kind='stable')
    return float(sum(len(ix) / len(order) * abs(np.mean(confidence[ix]) - np.mean(outcomes[ix]))
                     for ix in np.array_split(order, min(bins, len(order))) if len(ix)))


def reliability_bins(p, y, bins=15):
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    order = np.argsort(confidence, kind='stable')
    return [{'n': len(ix), 'confidence': float(confidence[ix].mean()), 'accuracy': float(correct[ix].mean())}
            for ix in np.array_split(order, min(bins, len(y))) if len(ix)]


def risk_curve(p, y):
    # Ties retain original observation order; no ordering by outcomes.
    order = np.argsort(-p.max(axis=1), kind='stable')
    errors = (p.argmax(axis=1) != y)[order]
    risk = np.cumsum(errors) / np.arange(1, len(y) + 1)
    coverage = np.arange(1, len(y) + 1) / len(y)
    return coverage, risk


def expected_tie_risk_curve(p, y):
    """Expected risk if equal-confidence observations are randomly ordered.

    Retain the legacy curve separately; this diagnostic avoids rewarding the
    arbitrary chronological order within an otherwise indistinguishable tie.
    """
    confidence = p.max(axis=1)
    order = np.argsort(-confidence, kind='stable')
    values = confidence[order]
    errors = (p.argmax(axis=1) != y)[order].astype(float)
    boundaries = np.r_[0, np.flatnonzero(np.diff(values) != 0) + 1, len(y)]
    expected = np.empty(len(y))
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        expected[start:end] = errors[start:end].mean()
    return np.arange(1, len(y) + 1) / len(y), np.cumsum(expected) / np.arange(1, len(y) + 1)


def evaluate(p, y, sets=None, alpha=0.05):
    p = probabilities(p)
    y = np.asarray(y, int)
    n, k = p.shape
    predicted = p.argmax(axis=1)
    truth = np.eye(k)[y]
    present = np.unique(y)
    result = {'n': n, 'accuracy': float(np.mean(predicted == y)),
              'balanced_accuracy': float(np.mean([np.mean(predicted[y == j] == j) for j in present])),
              'macro_f1_present_classes': float(f1_score(y, predicted, labels=present, average='macro', zero_division=0)),
              'brier_sum_over_classes': float(np.mean(np.sum((p - truth)**2, axis=1))),
              'nll': float(-np.log(p[np.arange(n), y]).mean()),
              'ece_15_equal_frequency': ece_equal_frequency(p.max(axis=1), predicted == y),
              'classwise_ece_15': float(np.mean([ece_equal_frequency(p[:, j], y == j) for j in range(k)]))}
    auc, ap = [], []
    for j in ([1] if k == 2 else range(k)):
        binary = y == j
        if binary.any() and not binary.all():
            auc.append(roc_auc_score(binary, p[:, j]))
            ap.append(average_precision_score(binary, p[:, j]))
    result['auroc'] = float(np.mean(auc)) if auc else None
    result['auprc'] = float(np.mean(ap)) if ap else None
    curve_coverage, curve_risk = risk_curve(p, y)
    result['aurc_discrete'] = float(curve_risk.mean())
    result['risk_at_80_percent_coverage'] = float(curve_risk[math.ceil(0.8 * n) - 1])
    acceptable = curve_coverage[curve_risk <= 0.05]
    result['max_coverage_at_risk_005'] = float(acceptable.max()) if len(acceptable) else 0.
    _, expected_risk = expected_tie_risk_curve(p, y)
    result['aurc_expected_random_ties'] = float(expected_risk.mean())
    if sets is not None:
        sets = np.asarray(sets, bool)
        if sets.shape != p.shape:
            raise ValueError('Set shape mismatch')
        included = sets[np.arange(n), y]
        size = sets.sum(axis=1)
        single = size == 1
        accepted_predictions = sets.argmax(axis=1)
        result.update({'coverage': float(included.mean()),
                       'absolute_coverage_error': float(abs(included.mean() - (1 - alpha))),
                       'undercoverage': float(max(0., 1 - alpha - included.mean())),
                       'set_size': float(size.mean()), 'singleton_rate': float(single.mean()),
                       'empty_rate': float((size == 0).mean()), 'abstention_rate': float((~single).mean()),
                       'selective_accuracy': float((accepted_predictions[single] == y[single]).mean()) if single.any() else None})
        class_coverage = {str(j): float(included[y == j].mean()) if np.any(y == j) else None for j in range(k)}
        result['class_coverage'] = class_coverage
        result['worst_present_class_coverage'] = min(v for v in class_coverage.values() if v is not None)
        agreement = single & (accepted_predictions == predicted)
        result['singleton_count'] = int(single.sum())
        result['correct_singleton_count'] = int(np.sum(included & single))
        result['singleton_argmax_disagreement_count'] = int(np.sum(single & ~agreement))
        result['agreement_singleton_rate'] = float(agreement.mean())
        result['agreement_singleton_accuracy'] = float(included[agreement].mean()) if agreement.any() else None
    return result
