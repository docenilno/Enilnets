import json
import pickle
import os
import numpy as np

# Every layer dict key whose value is a weight/statistic array that must
# round-trip as a numpy array (JSON has no array type, so these come back as
# plain Python lists after json.load and need explicit restoration).
_LAYER_ARRAY_KEYS = [
    "weights", "bias", "mask", "gamma", "beta", "running_mean", "running_var",
    "Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo", "pe",
    "Wx", "Wh", "b", "bx", "bh",
]

def _numpy_encoder(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def _restore_grad_accum(raw_accum):
    if not raw_accum:
        return []
    restored = []
    for entry in raw_accum:
        if entry is None:
            restored.append(None)
        else:
            restored.append({k: np.array(v, dtype=np.float64) for k, v in entry.items()})
    return restored

def Save(self, file, save_opt_state=True, extra_state=None):
    """
    Save the full trainable state of the model: layer weights, every
    optimizer/training hyperparameter that can be overridden per-model
    (learning rate, l2, momentum, grad clipping, mixed precision, per-
    optimizer betas/epsilons), optimizer state (momentum/velocity buffers),
    in-progress gradient accumulation buffers, the auto-shape-inference and
    residual-connection bookkeeping needed to keep building the model after
    loading, and the training/eval mode flag.

    Parameters
    ----------
    file : str
        Path to save file (.pkl or .json)
    save_opt_state : bool
        Whether to save optimizer state (momentum buffers, etc.)
    extra_state : dict or None
        Additional state to save (e.g., EMA weights, persistent CD buffer)
    """
    payload = {
        "version": 5,
        "layers": self.layers,
        "optimizer": self.optimizer_type,
        "learning_rate": self.learning_rate,
        "l2_lambda": self.l2_lambda,
        "momentum": self.momentum,
        "grad_clip_norm": self.grad_clip_norm,
        "use_mixed_precision": self.use_mixed_precision,
        "adam_beta1": self.adam_beta1,
        "adam_beta2": self.adam_beta2,
        "adam_epsilon": self.adam_epsilon,
        "rmsprop_decay": self.rmsprop_decay,
        "rmsprop_epsilon": self.rmsprop_epsilon,
        "adagrad_epsilon": self.adagrad_epsilon,
        "t": self.t,
        "training": self.training,
        "accum_steps": self._accum_steps,
        "grad_accum": self._grad_accum,
        "last_width": self._last_width,
        "last_spatial": self._last_spatial,
        "residual_stack": self._residual_stack,
    }
    if save_opt_state:
        payload["opt_state"] = self.opt_state
    if extra_state is not None:
        payload["extra_state"] = extra_state

    ext = os.path.splitext(file)[1].lower()
    if ext == ".pkl":
        with open(file, "wb") as f:
            pickle.dump(payload, f)
    else:
        with open(file, "w") as f:
            json.dump(payload, f, default=_numpy_encoder)

def Load(self, file, load_opt_state=True):
    """
    Restore everything Save() wrote: layer weights, optimizer/training
    hyperparameters, optimizer state, gradient accumulation buffers, and
    auto-shape-inference/residual bookkeeping.

    Parameters
    ----------
    file : str
        Path to save file (.pkl or .json)
    load_opt_state : bool
        Whether to restore optimizer state

    Returns
    -------
    extra_state : dict or None
        Any extra state that was saved (e.g., EMA weights)
    """
    ext = os.path.splitext(file)[1].lower()
    if ext == ".pkl":
        with open(file, "rb") as f:
            raw = pickle.load(f)
    else:
        with open(file, "r") as f:
            raw = json.load(f)

    self.layers = []
    for l in raw.get("layers", []):
        for k in _LAYER_ARRAY_KEYS:
            if k in l:
                l[k] = np.array(l[k], dtype=np.float64)
        self.layers.append(l)

    self.opt_state = []
    if load_opt_state and "opt_state" in raw:
        for state in raw["opt_state"]:
            if state is None:
                self.opt_state.append(None)
            else:
                restored = {}
                for k, v in state.items():
                    if isinstance(v, list):
                        restored[k] = np.array(v, dtype=np.float64)
                    else:
                        restored[k] = v
                self.opt_state.append(restored)

    self.t = raw.get("t", 0)
    self.learning_rate = raw.get("learning_rate", self.learning_rate)
    self.optimizer_type = raw.get("optimizer", self.optimizer_type)
    self.l2_lambda = raw.get("l2_lambda", self.l2_lambda)
    self.momentum = raw.get("momentum", self.momentum)
    self.grad_clip_norm = raw.get("grad_clip_norm", self.grad_clip_norm)
    self.use_mixed_precision = raw.get("use_mixed_precision", self.use_mixed_precision)
    self.adam_beta1 = raw.get("adam_beta1", self.adam_beta1)
    self.adam_beta2 = raw.get("adam_beta2", self.adam_beta2)
    self.adam_epsilon = raw.get("adam_epsilon", self.adam_epsilon)
    self.rmsprop_decay = raw.get("rmsprop_decay", self.rmsprop_decay)
    self.rmsprop_epsilon = raw.get("rmsprop_epsilon", self.rmsprop_epsilon)
    self.adagrad_epsilon = raw.get("adagrad_epsilon", self.adagrad_epsilon)
    self.training = raw.get("training", self.training)
    self._accum_steps = raw.get("accum_steps", 0)
    self._grad_accum = _restore_grad_accum(raw.get("grad_accum"))
    self._last_width = raw.get("last_width", self._last_width)
    last_spatial = raw.get("last_spatial")
    if last_spatial is not None:
        self._last_spatial = tuple(last_spatial)
    residual_stack = raw.get("residual_stack")
    if residual_stack is not None:
        self._residual_stack = list(residual_stack)

    return raw.get("extra_state", None)
