"""
General-purpose utility functions: reproducibility, data prep, training
helpers, and introspection. Pure NumPy, no external dependencies.
"""
from .backend import np
from .text_utils import one_hot_encode as one_hot


def set_seed(seed):
    """Seed the active backend's global RNG for reproducible runs.

    Reproducibility is only guaranteed within one backend: NumPy's legacy
    Mersenne Twister and CuPy's cuRAND-backed generator do not produce
    identical sequences for the same seed, so don't expect CPU and GPU
    runs with the same seed to match bit-for-bit.
    """
    np.random.seed(seed)


def train_test_split(X, Y, test_size=0.2, shuffle=True, seed=None):
    """Split X, Y into train/test sets.

    test_size: fraction (0 < test_size < 1) or an absolute number of samples.
    Returns (X_train, X_test, Y_train, Y_test).
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    n = X.shape[0]
    if isinstance(test_size, float):
        n_test = int(round(n * test_size))
    else:
        n_test = int(test_size)

    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    indices = rng.permutation(n) if shuffle else np.arange(n)
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    return X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]


def k_fold_split(X, Y, k=5, shuffle=True, seed=None):
    """Yield (X_train, X_val, Y_train, Y_val) for each of k folds."""
    X = np.asarray(X)
    Y = np.asarray(Y)
    n = X.shape[0]
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    indices = rng.permutation(n) if shuffle else np.arange(n)

    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1
    current = 0
    for fold_size in fold_sizes:
        val_idx = indices[current:current + fold_size]
        train_idx = np.concatenate([indices[:current], indices[current + fold_size:]])
        yield X[train_idx], X[val_idx], Y[train_idx], Y[val_idx]
        current += fold_size


def iterate_minibatches(X, Y, batch_size, shuffle=True):
    """Yield (X_batch, Y_batch) pairs covering the full dataset once."""
    n = X.shape[0]
    indices = np.random.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield X[batch_idx], Y[batch_idx]


def count_parameters(model):
    """Return (total_params, per_layer_dict) for a NeuralNet -- the
    programmatic counterpart to NeuralNet.summary(), which only prints."""
    per_layer = {}
    total = 0
    for i, layer in enumerate(model.layers):
        n = 0
        for key in ("weights", "bias", "gamma", "beta", "Wq", "bq", "Wk", "bk",
                    "Wv", "bv", "Wo", "bo", "Wx", "Wh", "b", "bx", "bh"):
            if key in layer:
                n += layer[key].size
        per_layer[i] = {"type": layer["type"], "params": n}
        total += n
    return total, per_layer


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    mode: "min" (e.g. loss) or "max" (e.g. accuracy).
    Usage: es = EarlyStopping(patience=5); ... ; if es.step(val_loss): break
    """
    def __init__(self, patience=5, min_delta=0.0, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def _is_improvement(self, metric):
        if self.best is None:
            return True
        if self.mode == "min":
            return metric < self.best - self.min_delta
        return metric > self.best + self.min_delta

    def step(self, metric):
        """Call once per epoch with the latest metric value. Returns True if
        training should stop."""
        if self._is_improvement(metric):
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        self.should_stop = self.num_bad_epochs >= self.patience
        return self.should_stop


class ModelCheckpoint:
    """Train(..., callbacks=[...]) callback: snapshots the model's weights
    (via get_weights(), not a full Save()) whenever the monitored metric
    improves, mirroring EarlyStopping's own improvement logic. Call
    .restore(model) after training to load the best snapshot back.

    monitor: a key expected in the epoch's `logs` dict -- "loss"/
        "accuracy"/"lr" always present, "val_loss"/"val_accuracy" only if
        X_val/Y_val were given to Train(). Silently does nothing on an
        epoch where `monitor` isn't present in `logs` (e.g. "val_loss"
        requested but no validation data was given).
    mode: "min" (e.g. loss) or "max" (e.g. accuracy).
    """
    def __init__(self, monitor="val_loss", mode="min", min_delta=0.0):
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.best_weights = None
        self.best_epoch = None

    def _is_improvement(self, metric):
        if self.best is None:
            return True
        if self.mode == "min":
            return metric < self.best - self.min_delta
        return metric > self.best + self.min_delta

    def on_epoch_end(self, epoch, logs, model=None):
        if self.monitor not in logs:
            return
        metric = logs[self.monitor]
        if self._is_improvement(metric):
            self.best = metric
            self.best_weights = model.get_weights()
            self.best_epoch = epoch

    def restore(self, model):
        """Load the best snapshot's weights back into `model` (no-op if
        no epoch ever improved, e.g. training never called on_epoch_end)."""
        if self.best_weights is not None:
            model.set_weights(self.best_weights)


class CSVLogger:
    """Train(..., callbacks=[...]) callback: appends one row per epoch to
    a CSV file (NOT a single growing in-memory table written once at the
    end), so an interrupted run leaves a valid, parseable partial log.
    Column header is written on the first epoch based on that epoch's
    `logs` keys."""
    def __init__(self, path):
        self.path = path
        self._fieldnames = None

    def on_epoch_end(self, epoch, logs, model=None):
        import csv
        row = {"epoch": epoch}
        row.update({k: float(v) for k, v in logs.items()})
        write_header = self._fieldnames is None
        if write_header:
            self._fieldnames = list(row.keys())
        with open(self.path, "w" if write_header else "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


class JSONLogger:
    """Train(..., callbacks=[...]) callback: appends one JSON object per
    line (JSON-lines format -- NOT a single growing JSON array), so an
    interrupted run leaves a valid, parseable partial log (each completed
    line parses independently with json.loads)."""
    def __init__(self, path):
        self.path = path
        self._opened = False

    def on_epoch_end(self, epoch, logs, model=None):
        import json
        row = {"epoch": epoch}
        row.update({k: float(v) for k, v in logs.items()})
        mode = "w" if not self._opened else "a"
        with open(self.path, mode) as f:
            f.write(json.dumps(row) + "\n")
        self._opened = True
