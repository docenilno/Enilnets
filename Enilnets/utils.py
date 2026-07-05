"""
General-purpose utility functions: reproducibility, data prep, training
helpers, and introspection. Pure NumPy, no external dependencies.
"""
import numpy as np
from .text_utils import one_hot_encode as one_hot


def set_seed(seed):
    """Seed NumPy's global RNG for reproducible runs."""
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
