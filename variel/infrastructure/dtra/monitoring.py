"""Source-only block replay and a dual empirical alarm; no sequential FPR theorem."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
from .data import PARTITION_SEED
from .models import TabularEncoding
from .io import file_hash, write_json


def monitoring_roles(source, labels):
    fit, rest = train_test_split(source, test_size=0.6, stratify=labels[source], random_state=PARTITION_SEED + 21)
    conformal, monitor = train_test_split(rest, test_size=1 / 3, stratify=labels[rest], random_state=PARTITION_SEED + 22)
    return np.sort(fit), np.sort(conformal), np.sort(monitor)


class FeatureMonitor:
    def __init__(self, data, monitor_indices, config, cache_dir):
        encoding = TabularEncoding(data.categorical).fit(data.X.iloc[data.train])
        values = encoding.transform(data.X)
        self.x = np.asarray(values.toarray() if hasattr(values, 'toarray') else values, np.float32)
        self.monitor_indices = np.asarray(monitor_indices, int)
        self.config = config
        self.n = min(config['domain_cap'], len(monitor_indices) // 2, min(map(len, data.windows)))
        if self.n < 10:
            raise ValueError('Too few held-out source observations for the C2ST')
        key = hashlib.sha256(self.x.tobytes() + self.monitor_indices.tobytes() +
                             json.dumps({k: config[k] for k in config if k.startswith(('domain_', 'null_'))}, sort_keys=True).encode() +
                             file_hash(Path(__file__)).encode()).hexdigest()[:16]
        self.path = Path(cache_dir) / f'{data.name}_{key}.json'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.saved = json.loads(self.path.read_text()) if self.path.exists() else {'scores': {}}
        rng = np.random.default_rng(PARTITION_SEED + 51)
        blocks = [self.monitor_indices[i:i + config['null_block_size']]
                  for i in range(0, len(self.monitor_indices), config['null_block_size'])]
        self.null_targets, null_d = [], []
        for r in range(config['null_replays']):
            order = rng.permutation(len(blocks))
            shuffled = np.concatenate([blocks[j] for j in order])[:2 * self.n]
            reference, target = shuffled[:self.n], shuffled[self.n:]
            self.null_targets.append(target)
            null_d.append(self.score(reference, target, seed=PARTITION_SEED + 100 + r))
            if (r + 1) % 16 == 0:
                print(f'SOURCE_REPLAY dataset={data.name} completed={r + 1}/{config["null_replays"]}', flush=True)
        self.null_d = np.asarray(null_d)
        self.save()

    def save(self):
        write_json(self.path, self.saved)

    def score(self, reference, target, seed=PARTITION_SEED + 31):
        reference, target = np.asarray(reference, int), np.asarray(target, int)
        key = hashlib.sha256(reference.tobytes() + target.tobytes() + str(seed).encode()).hexdigest()
        if key in self.saved['scores']:
            return self.saved['scores'][key]
        rng = np.random.default_rng(seed)
        n = min(self.n, len(reference), len(target))
        a = rng.choice(reference, n, replace=False)
        b = rng.choice(target, n, replace=False)
        values = self.x[np.r_[a, b]]
        domain = np.r_[np.zeros(n, int), np.ones(n, int)]
        cv = StratifiedKFold(n_splits=self.config['domain_folds'], shuffle=True, random_state=seed)
        prediction = np.empty(len(domain))
        with threadpool_limits(limits=1):
            for train, valid in cv.split(values, domain):
                model = HistGradientBoostingClassifier(max_iter=self.config['domain_iterations'], max_leaf_nodes=15,
                         min_samples_leaf=10, l2_regularization=1.0, early_stopping=False, random_state=seed)
                model.fit(values[train], domain[train])
                prediction[valid] = model.predict_proba(values[valid])[:, 1]
        value = max(0.0, 2 * (float(roc_auc_score(domain, prediction)) - 0.5))
        self.saved['scores'][key] = value
        return value


class ReliabilityMonitor:
    def __init__(self, feature_monitor, source_p, source_sets, source_y, config):
        self.features, self.config = feature_monitor, config
        self.losses = np.sum((source_p - np.eye(source_p.shape[1])[source_y]) ** 2, axis=1)
        self.misses = (~source_sets[np.arange(len(source_y)), source_y]).astype(float)
        self.mean = float(self.losses.mean())
        self.sd = max(float(self.losses.std(ddof=1)), 1e-8)
        self.local = {int(row): i for i, row in enumerate(feature_monitor.monitor_indices)}
        self.threshold_cache = {}

    def harm(self, misses, losses):
        n = len(misses)
        if not n:
            return 0.0
        alpha = self.config['alpha']
        cov = (np.mean(misses) - alpha) / np.sqrt(alpha * (1 - alpha) / n)
        brier = (np.mean(losses) - self.mean) / (self.sd / np.sqrt(n))
        return float(max(0, cov, brier))

    def thresholds(self, n):
        n = max(1, int(n))
        if n not in self.threshold_cache:
            h = []
            for r, targets in enumerate(self.features.null_targets):
                local = np.array([self.local[int(i)] for i in targets])
                rng = np.random.default_rng(np.random.SeedSequence([PARTITION_SEED, n, r, 81]))
                draw = rng.choice(local, n, replace=n > len(local))
                h.append(self.harm(self.misses[draw], self.losses[draw]))
            h = np.asarray(h)
            cut = len(h) // 2
            q = self.config['null_quantile']
            dx = max(float(np.quantile(self.features.null_d[:cut], q, method='higher')), 1e-8)
            hy = max(float(np.quantile(h[:cut], q, method='higher')), 1e-8)
            joint = np.maximum(self.features.null_d[cut:] / dx, h[cut:] / hy)
            gamma = max(float(np.quantile(joint, q, method='higher')), 1e-8)
            self.threshold_cache[n] = {'feature_scale': dx, 'harm_scale': hy, 'joint_cutoff': gamma}
        return self.threshold_cache[n]

    def decision(self, d, h, n):
        q = self.thresholds(n)
        statistic = max(d / q['feature_scale'], h / q['harm_scale'])
        alarm = statistic > q['joint_cutoff']
        severity = max(0.0, (statistic - q['joint_cutoff']) / (statistic + q['joint_cutoff']))
        return bool(alarm), float(severity), {'joint_statistic': statistic, **q}
