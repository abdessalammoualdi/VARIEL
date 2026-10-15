"""Three fixed model families; source-only model selection and full-source refit."""
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .calibration import probabilities
from .data import development_roles
from .io import write_json


class TabularEncoding:
    def __init__(self, categorical, neural=False):
        self.categorical = categorical
        self.neural = neural

    def fit(self, X):
        self.numeric = [c for c in X.columns if c not in self.categorical]
        if self.neural:
            self.imputer = SimpleImputer(strategy='median')
            self.scaler = StandardScaler()
            self.scaler.fit(self.imputer.fit_transform(X[self.numeric]))
            self.category_maps = {c: {value: j + 1 for j, value in enumerate(sorted(X[c].fillna('__missing__').astype(str).unique()))} for c in self.categorical}
            self.cardinalities = [len(self.category_maps[c]) + 1 for c in self.categorical]
        else:
            transformers = [('numeric', SimpleImputer(strategy='median'), self.numeric)]
            if self.categorical:
                transformers.append(('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float32), self.categorical))
            self.transformer = ColumnTransformer(transformers, sparse_threshold=1.0)
            self.transformer.fit(X)
        return self

    def transform(self, X):
        if not self.neural:
            return self.transformer.transform(X).astype(np.float32)
        numerical = self.scaler.transform(self.imputer.transform(X[self.numeric])).astype(np.float32)
        categorical = np.column_stack([X[c].fillna('__missing__').astype(str).map(self.category_maps[c]).fillna(0).to_numpy(dtype=np.int64) for c in self.categorical]) if self.categorical else None
        return numerical, categorical


def cat_frame(X, categorical):
    X = X.copy()
    for c in categorical:
        X[c] = X[c].fillna('__missing__').astype(str)
    return X


def fit_tree(data, family, seed, config, out, threads=12):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    inner, valid = development_roles(data)
    if len(np.unique(data.y[inner])) != len(data.classes):
        raise ValueError('The development training partition lacks a class')
    k = len(data.classes)
    start = time.perf_counter()
    if family == 'xgboost':
        from xgboost import XGBClassifier
        args = dict(n_estimators=config['tree_rounds'], max_depth=config['tree_depth'],
                    learning_rate=config['tree_learning_rate'], subsample=0.9, colsample_bytree=0.9,
                    min_child_weight=2, reg_lambda=1., tree_method='hist', device='cpu',
                    n_jobs=threads, random_state=seed,
                    objective='binary:logistic' if k == 2 else 'multi:softprob',
                    eval_metric='logloss' if k == 2 else 'mlogloss')
        encoding = TabularEncoding(data.categorical).fit(data.X.iloc[inner])
        selection = XGBClassifier(**args, early_stopping_rounds=config['tree_early_stopping'])
        selection.fit(encoding.transform(data.X.iloc[inner]), data.y[inner],
                      eval_set=[(encoding.transform(data.X.iloc[valid]), data.y[valid])], verbose=False)
        rounds = int(selection.best_iteration) + 1
        history = selection.evals_result()
        selection.save_model(out / 'selection_best.ubj')
        args['n_estimators'] = rounds
        encoding = TabularEncoding(data.categorical).fit(data.X.iloc[data.train])
        model = XGBClassifier(**args)
        model.fit(encoding.transform(data.X.iloc[data.train]), data.y[data.train], verbose=False)
        model.save_model(out / 'frozen.ubj')
        joblib.dump(encoding, out / 'preprocessor.joblib')
        predictions = model.predict_proba(encoding.transform(data.X))
    elif family == 'catboost':
        from catboost import CatBoostClassifier
        args = dict(iterations=config['tree_rounds'], depth=config['tree_depth'],
                    learning_rate=config['tree_learning_rate'], l2_leaf_reg=3.,
                    loss_function='Logloss' if k == 2 else 'MultiClass', random_seed=seed,
                    thread_count=threads, task_type='CPU', verbose=False,
                    allow_writing_files=False, cat_features=data.categorical)
        X = cat_frame(data.X, data.categorical)
        selection = CatBoostClassifier(**args)
        selection.fit(X.iloc[inner], data.y[inner], eval_set=(X.iloc[valid], data.y[valid]),
                      early_stopping_rounds=config['tree_early_stopping'], use_best_model=True)
        rounds = max(1, int(selection.get_best_iteration()) + 1)
        history = selection.get_evals_result()
        selection.save_model(str(out / 'selection_best.cbm'))
        args['iterations'] = rounds
        model = CatBoostClassifier(**args)
        model.fit(X.iloc[data.train], data.y[data.train])
        model.save_model(str(out / 'frozen.cbm'))
        write_json(out / 'categorical_columns.json', data.categorical)
        predictions = model.predict_proba(X)
    else:
        raise ValueError(family)
    elapsed = time.perf_counter() - start
    info = {'family': family, 'seed': seed, 'selected_rounds': rounds, 'runtime_seconds': elapsed,
            'selection_train_n': len(inner), 'selection_valid_n': len(valid), 'final_train_n': len(data.train),
            'device': 'cpu', 'history': history, 'model_parameters': args}
    write_json(out / 'training.json', info)
    return probabilities(predictions), info


def fit_ft(data, seed, config, out, device='auto', threads=12):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from rtdl_revisiting_models import FTTransformer
    torch.set_num_threads(threads)
    resolved_device = 'cuda' if device == 'auto' and torch.cuda.is_available() else ('cpu' if device == 'auto' else device)
    if resolved_device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Run scripts/preflight.py --gpu-test.')
    if resolved_device == 'cuda':
        probe = torch.randn(32, 32, device='cuda')
        _ = probe @ probe
        torch.cuda.synchronize()
        torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    inner, valid = development_roles(data)
    k = len(data.classes)
    start = time.perf_counter()

    def stage(train_ix, valid_ix, epochs, name):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        encoding = TabularEncoding(data.categorical, neural=True).fit(data.X.iloc[train_ix])
        continuous, categorical = encoding.transform(data.X)
        xc = torch.from_numpy(continuous)
        xcat = torch.from_numpy(categorical) if categorical is not None else torch.empty((len(data.y), 0), dtype=torch.long)
        labels = torch.tensor(data.y, dtype=torch.long)
        kwargs = dict(n_cont_features=continuous.shape[1], cat_cardinalities=encoding.cardinalities,
                      d_out=k, n_blocks=config['ft_blocks'], d_block=config['ft_width'],
                      attention_n_heads=config['ft_heads'], attention_dropout=0.2,
                      ffn_d_hidden=None, ffn_d_hidden_multiplier=4 / 3, ffn_dropout=0.1, residual_dropout=0.0)
        model = FTTransformer(**kwargs).to(resolved_device)
        optimizer = torch.optim.AdamW(model.make_parameter_groups(), lr=config['ft_learning_rate'], weight_decay=config['ft_weight_decay'])
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(TensorDataset(xc[train_ix], xcat[train_ix], labels[train_ix]),
                            batch_size=config['ft_batch_size'], shuffle=True, num_workers=0,
                            generator=generator, pin_memory=resolved_device == 'cuda')
        history, best, best_epoch, stale = [], float('inf'), 0, 0
        def predict(indices):
            model.eval()
            outputs = []
            with torch.inference_mode():
                for chunk in np.array_split(indices, max(1, int(np.ceil(len(indices) / 512)))):
                    if not len(chunk):
                        continue
                    a = xc[chunk].to(resolved_device)
                    b = xcat[chunk].to(resolved_device) if data.categorical else None
                    outputs.append(model(a, b).softmax(-1).cpu().numpy())
            return np.concatenate(outputs)
        for epoch in range(epochs):
            model.train()
            total_loss, count = 0., 0
            for a, b, y in loader:
                a, y = a.to(resolved_device), y.to(resolved_device)
                b = b.to(resolved_device) if data.categorical else None
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(a, b), y)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite FT-Transformer training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(y)
                count += len(y)
            record = {'epoch': epoch + 1, 'train_nll': total_loss / count}
            if valid_ix is not None:
                pv = probabilities(predict(valid_ix))
                value = float(-np.log(pv[np.arange(len(valid_ix)), data.y[valid_ix]]).mean())
                record['validation_nll'] = value
                improved = value < best - 1e-6
                if improved:
                    best, best_epoch, stale = value, epoch + 1, 0
                else:
                    stale += 1
            else:
                improved, best_epoch = True, epoch + 1
            state = {'epoch': epoch + 1, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                     'model_kwargs': kwargs, 'seed': seed, 'phase': name,
                     'torch_rng': torch.get_rng_state(), 'shuffle_rng': generator.get_state(),
                     'best_epoch': best_epoch, 'best_validation_nll': best, 'device': resolved_device}
            torch.save(state, out / f'{name}_last.pt')
            if improved:
                torch.save(state, out / f'{name}_best.pt')
            history.append(record)
            write_json(out / f'{name}_history.json', history)
            print(f'FT_EPOCH dataset={data.name} seed={seed} phase={name} epoch={epoch + 1} train_nll={record["train_nll"]:.5f}' +
                  (f' validation_nll={record["validation_nll"]:.5f}' if valid_ix is not None else ''), flush=True)
            if valid_ix is not None and stale >= config['ft_patience']:
                break
        if valid_ix is None:
            torch.save({'model_state': model.state_dict(), 'model_kwargs': kwargs, 'seed': seed,
                        'selected_epochs': epochs, 'device': resolved_device}, out / 'frozen.pt')
            joblib.dump(encoding, out / 'preprocessor.joblib')
            return predict(np.arange(len(data.y))), history, epochs
        return None, history, best_epoch

    _, selection_history, best_epoch = stage(inner, valid, config['ft_max_epochs'], 'selection')
    prediction, refit_history, _ = stage(data.train, None, best_epoch, 'refit')
    info = {'family': 'ft_transformer', 'implementation': 'rtdl-revisiting-models==0.0.2',
            'seed': seed, 'selected_epochs': best_epoch, 'runtime_seconds': time.perf_counter() - start,
            'selection_train_n': len(inner), 'selection_valid_n': len(valid), 'final_train_n': len(data.train),
            'device': resolved_device, 'selection_history': selection_history, 'refit_history': refit_history,
            'torch_version': torch.__version__, 'cuda_version': torch.version.cuda}
    write_json(out / 'training.json', info)
    return probabilities(prediction), info
