from dataclasses import dataclass
from pathlib import Path
import io
import math
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

PARTITION_SEED = 20260910


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    classes: list
    categorical: list
    train: np.ndarray
    source: np.ndarray
    windows: list
    metadata: dict

    @property
    def target(self):
        return np.concatenate(self.windows)


def load_dataset(name, root, sensitivity=False):
    root = Path(root)
    if name == 'electricity':
        records, _ = arff.loadarff(root / 'electricity-normalized.arff')
        frame = pd.DataFrame(records)
        y = (frame['class'].str.decode('utf8') == 'UP').to_numpy(dtype=np.int64)
        date = frame['date'].to_numpy(float)
        X = frame.drop(columns=['date', 'class']).copy()
        X['day'] = X['day'].str.decode('utf8').astype(float)
        if sensitivity:
            X = X.drop(columns=['day', 'period'])
        classes, categorical = ['DOWN', 'UP'], []
        metadata = {'ordering': 'canonical benchmark row order; normalized date is not a sorting key',
                    'date_decrease_rows_1_based': (np.flatnonzero(np.diff(date) < 0) + 2).tolist(),
                    'preprocessed_source': 'published normalization and missing-value replacement'}
    elif name == 'bank':
        frame = pd.read_csv(root / 'bank-full.csv', sep=';')
        y = (frame['y'] == 'yes').to_numpy(dtype=np.int64)
        X = frame.drop(columns=['y', 'duration']).copy()
        if sensitivity:
            X = X.drop(columns=['day', 'month'])
        classes = ['no', 'yes']
        categorical = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
        metadata = {'ordering': 'published date-ordered row sequence; no sorting by month/day',
                    'unknown_categories': {c: int((X[c] == 'unknown').sum()) for c in categorical},
                    'excluded_predictor': 'duration'}
    elif name == 'gas':
        features, outcomes, batches = [], [], []
        for batch in range(1, 11):
            x, y = load_svmlight_file(str(root / f'batch{batch}.dat'), n_features=128, zero_based=False)
            features.append(x.toarray())
            outcomes.append(y.astype(np.int64) - 1)
            batches.extend([batch] * len(y))
        X = pd.DataFrame(np.concatenate(features), columns=[f'sensor_{i:03d}' for i in range(1, 129)])
        y = np.concatenate(outcomes)
        batches = np.asarray(batches)
        indices = np.flatnonzero(batches <= 3)
        train, source = train_test_split(indices, test_size=0.3, stratify=y[indices], random_state=PARTITION_SEED)
        return Dataset(name, X, y, [str(i) for i in range(1, 7)], [], np.sort(train), np.sort(source),
                       [np.flatnonzero(batches == i) for i in range(4, 11)],
                       {'ordering': 'batch chronology; no within-batch timestamp claim',
                        'source_batches': [1, 2, 3], 'target_batches': list(range(4, 11))})
    else:
        raise ValueError('Unknown dataset: ' + name)
    n = len(y)
    train_end, source_end = math.floor(0.5 * n), math.floor(0.65 * n)
    return Dataset(name, X, y, classes, categorical, np.arange(train_end),
                   np.arange(train_end, source_end), list(np.array_split(np.arange(source_end, n), 12)), metadata)


def source_roles(data):
    fit, conformal = train_test_split(data.source, test_size=0.5, stratify=data.y[data.source], random_state=PARTITION_SEED + 1)
    return np.sort(fit), np.sort(conformal)


def development_roles(data):
    if data.name == 'gas':
        fit, valid = train_test_split(data.train, test_size=0.2, stratify=data.y[data.train], random_state=PARTITION_SEED + 2)
    else:
        boundary = math.floor(0.8 * len(data.train))
        fit, valid = data.train[:boundary], data.train[boundary:]
    return np.sort(fit), np.sort(valid)


def audit_dataset(data):
    fit, conformal = source_roles(data)
    target = data.target
    parts = [data.train, data.source, target]
    joined = np.concatenate(parts)
    if len(np.unique(joined)) != len(data.y) or len(joined) != len(data.y):
        raise AssertionError('Partitions overlap or lose rows')
    if np.intersect1d(fit, conformal).size:
        raise AssertionError('Probability and conformal calibration roles overlap')
    numeric = data.X.drop(columns=data.categorical).to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError('Unexpected nonfinite numeric features')
    def part(indices):
        return {'n': len(indices), 'class_counts': {c: int(np.sum(data.y[indices] == k)) for k, c in enumerate(data.classes)}}
    return {'dataset': data.name, 'n': len(data.y), 'n_features': data.X.shape[1],
            'categorical_features': data.categorical, 'metadata': data.metadata,
            'train': part(data.train), 'source': part(data.source), 'probability_fit': part(fit),
            'conformal_fit': part(conformal), 'target': part(target),
            'windows': [{'window': t + 1, **part(ix)} for t, ix in enumerate(data.windows)]}
