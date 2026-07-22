"""NeuralNet.ComputeLoss: a library of loss functions bound as a method
(see base.py)."""
import math
from typing import Any

from ..core.backend import np
from ..core import backend
from ..core import constants

def ComputeLoss(self: Any, output: Any, target: Any, function: str = "mse",
                 reduction: str = "mean", **kwargs: Any) -> Any:
    """Compute a loss between `output` and `target`.

    function: "mse" | "mae" | "huber" (kwarg: delta=1.0) | "smooth_l1"
        (kwarg: beta=1.0) | "binary_cross_entropy" | "cross_entropy" /
        "categorical_cross_entropy" (one-hot target) |
        "sparse_cross_entropy" (integer-class-index target) | "focal"
        (kwargs: alpha=0.25, gamma=2.0) | "hinge" | "kl_divergence"
        (kwargs: mu, logvar -- operates on VAE mu/logvar tensors, not
        output/target) | "bce_logits" | "wasserstein" |
        "cosine_similarity" | "triplet" (kwargs: margin=1.0, negative) |
        "ntxent" (kwarg: temperature=0.5).
    reduction: "mean" | "sum" | anything else returns the unreduced
        per-element/per-row loss array.
    """
    o = np.asarray(output, dtype=backend.default_dtype())
    t = np.asarray(target, dtype=backend.default_dtype())
    eps = kwargs.get("eps", constants.EPS_LOG)
    eps_div = kwargs.get("eps_div", constants.EPS_DIV)
    if function == "mse":
        loss = (o - t) ** 2
    elif function == "mae":
        loss = np.abs(o - t)
    elif function == "huber":
        delta = kwargs.get("delta", 1.0)
        diff = np.abs(o - t)
        loss = np.where(diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))
    elif function == "smooth_l1":
        beta = kwargs.get("beta", 1.0)
        diff = np.abs(o - t)
        loss = np.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    elif function == "binary_cross_entropy":
        o = np.clip(o, eps, 1 - eps)
        loss = -(t * np.log(o) + (1 - t) * np.log(1 - o))
    elif function in ("cross_entropy", "categorical_cross_entropy"):
        o = np.clip(o, eps, 1.0)
        loss = -t * np.log(o)
        # Divide by every leading (non-class) dim -- batch_size*seq_len for
        # (B,S,V) sequence output, not just batch_size (o.shape[0]), else the
        # reported loss/gradient scale is off by a factor of seq_len.
        n = math.prod(o.shape[:-1]) if o.ndim > 1 else o.shape[0]
        if reduction == "mean":
            return float(np.sum(loss) / n)
        if reduction == "sum":
            return float(np.sum(loss))
        return loss
    elif function == "sparse_cross_entropy":
        # Like cross_entropy but `target` is integer class indices (shape
        # matching output.shape[:-1]) instead of a one-hot array -- avoids
        # manually one-hotting large-vocab targets (e.g. language modeling).
        o = np.clip(o, eps, 1.0)
        idx = np.asarray(target).astype(np.int64)
        if idx.shape != o.shape[:-1]:
            raise ValueError(
                f"sparse_cross_entropy target shape {idx.shape} must match "
                f"output.shape[:-1] {o.shape[:-1]}"
            )
        picked = np.take_along_axis(o, idx[..., None], axis=-1)[..., 0]
        loss = -np.log(picked)
        n = math.prod(o.shape[:-1]) if o.ndim > 1 else o.shape[0]
        if reduction == "mean":
            return float(np.sum(loss) / n)
        if reduction == "sum":
            return float(np.sum(loss))
        return loss
    elif function == "focal":
        alpha = kwargs.get("alpha", 0.25)
        gamma = kwargs.get("gamma", 2.0)
        o = np.clip(o, eps, 1.0 - eps)
        pt = o * t + (1 - o) * (1 - t)
        loss = - (alpha * t * (1 - pt) ** gamma * np.log(o) + (1 - alpha) * (1 - t) * (1 - pt) ** gamma * np.log(1 - o))
    elif function == "hinge":
        loss = np.maximum(0, 1 - t * o)
    elif function == "kl_divergence":
        if "mu" not in kwargs or "logvar" not in kwargs:
            raise ValueError(
                "kl_divergence requires explicit 'mu' and 'logvar' kwargs "
                "(it operates on VAE mu/logvar tensors, not output/target)."
            )
        mu = kwargs["mu"]
        logvar = kwargs["logvar"]
        loss = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar), axis=-1)
    elif function == "bce_logits":
        loss = np.maximum(o, 0) - o * t + np.log(1 + np.exp(-np.abs(o)))
    elif function == "wasserstein":
        loss = -o * t
    elif function == "cosine_similarity":
        # Cosine embedding loss: 1 - cos(x, y)
        o_norm = o / (np.linalg.norm(o, axis=-1, keepdims=True) + eps_div)
        t_norm = t / (np.linalg.norm(t, axis=-1, keepdims=True) + eps_div)
        loss = 1 - np.sum(o_norm * t_norm, axis=-1)
    elif function == "triplet":
        margin = kwargs.get("margin", 1.0)
        # o: anchor, t: positive, kwargs['negative']: negative
        neg = kwargs.get("negative", None)
        if neg is None:
            raise ValueError("triplet loss requires 'negative' kwarg")
        d_pos = np.sum((o - t) ** 2, axis=-1)
        d_neg = np.sum((o - neg) ** 2, axis=-1)
        loss = np.maximum(0, d_pos - d_neg + margin)
    elif function == "ntxent":
        # Normalized Temperature-scaled Cross Entropy (SimCLR)
        temperature = kwargs.get("temperature", 0.5)
        o_norm = o / (np.linalg.norm(o, axis=-1, keepdims=True) + eps_div)
        t_norm = t / (np.linalg.norm(t, axis=-1, keepdims=True) + eps_div)
        sim = np.dot(o_norm, t_norm.T) / temperature
        sim_max = np.max(sim, axis=-1, keepdims=True)
        exp_sim = np.exp(sim - sim_max)
        pos_mask = np.eye(o.shape[0], dtype=backend.default_dtype())
        pos = np.sum(exp_sim * pos_mask, axis=-1)
        neg = np.sum(exp_sim, axis=-1)
        loss = -np.log(pos / (neg + eps_div) + eps_div)
    else:
        raise ValueError(f"Unknown loss function: {function}")

    if reduction == "mean":
        return float(np.mean(loss))
    if reduction == "sum":
        return float(np.sum(loss))
    return loss
