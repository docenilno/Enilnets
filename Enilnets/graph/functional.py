"""Stateless functional API over the graph ops (roadmap item 27) --
``Enilnets.functional.relu(x)``-style calls with no layer objects and no
stored state. Also importable as ``Enilnets.functional``.

Everything here is a thin composition of the differentiable ops in
``ops.py``, so gradients come for free and every function works on
Tensors or raw arrays alike.
"""

from typing import Any, Optional

from ..core.backend import np
from .tensor import Tensor, as_tensor
from . import ops

# Direct re-exports: already stateless functions.
relu = ops.relu
sigmoid = ops.sigmoid
tanh = ops.tanh
softmax = ops.softmax
log_softmax = ops.log_softmax
matmul = ops.matmul
pad = ops.pad
exp = ops.exp
log = ops.log
sqrt = ops.sqrt
conj = ops.conj
abs = ops.absolute  # noqa: A001 -- deliberate, mirrors torch.functional style


def linear(x: Any, weight: Any, bias: Optional[Any] = None) -> Tensor:
    """``y = x @ weight.T (+ bias)`` with the library-wide ``(n_out, n_in)``
    weight convention (same as ``add_dense`` / ``graph.Linear``)."""
    out = ops.matmul(as_tensor(x), ops.transpose(as_tensor(weight)))
    if bias is not None:
        out = ops.add(out, bias)
    return out


def dropout(x: Any, rate: float = 0.5, training: bool = True) -> Tensor:
    """Inverted dropout as a pure function (mask drawn per call). Identity
    when ``training=False`` or ``rate == 0``."""
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"dropout rate must be in [0, 1); got {rate}")
    x = as_tensor(x)
    if not training or rate == 0.0:
        return x
    keep = 1.0 - rate
    mask = (np.random.rand(*x.shape) < keep).astype(x.dtype) / keep
    return ops.mul(x, Tensor(mask))


def pixel_shuffle(x: Any, upscale_factor: int) -> Tensor:
    """Rearrange ``(B, C*r^2, H, W)`` into ``(B, C, H*r, W*r)`` (sub-pixel
    convolution upsampling, roadmap item 32). A pure reshape/transpose
    composite -- gradients come from the existing ops."""
    x = as_tensor(x)
    B, Cr2, H, W = x.shape
    r = int(upscale_factor)
    if Cr2 % (r * r) != 0:
        raise ValueError(
            f"pixel_shuffle needs channels divisible by upscale_factor^2 "
            f"({r * r}); got {Cr2}")
    C = Cr2 // (r * r)
    out = ops.reshape(x, shape=(B, C, r, r, H, W))
    out = ops.transpose(out, axes=(0, 1, 4, 2, 5, 3))   # (B, C, H, r, W, r)
    return ops.reshape(out, shape=(B, C, H * r, W * r))


def pixel_unshuffle(x: Any, downscale_factor: int) -> Tensor:
    """Inverse of :func:`pixel_shuffle`: ``(B, C, H*r, W*r)`` ->
    ``(B, C*r^2, H, W)``."""
    x = as_tensor(x)
    B, C, Hr, Wr = x.shape
    r = int(downscale_factor)
    if Hr % r != 0 or Wr % r != 0:
        raise ValueError(
            f"pixel_unshuffle needs spatial dims divisible by "
            f"downscale_factor ({r}); got {(Hr, Wr)}")
    H, W = Hr // r, Wr // r
    out = ops.reshape(x, shape=(B, C, H, r, W, r))
    out = ops.transpose(out, axes=(0, 1, 3, 5, 2, 4))   # (B, C, r, r, H, W)
    return ops.reshape(out, shape=(B, C * r * r, H, W))


def mse_loss(prediction: Any, target: Any) -> Tensor:
    """Mean squared error over every element (matches ``ComputeLoss('mse',
    reduction='mean')``)."""
    return ((as_tensor(prediction) - as_tensor(target)) ** 2).mean()


def cross_entropy(logits: Any, target_indices: Any) -> Tensor:
    """Cross-entropy from raw (unnormalized) logits and integer class
    indices: ``-mean(log_softmax(logits)[i, target_i])``. Numerically
    stable (log-sum-exp inside log_softmax); the gather uses a
    unique-row index, so its gradient scatter is exact."""
    logits = as_tensor(logits)
    idx = np.asarray(target_indices).astype(np.int64)
    if idx.shape != logits.shape[:-1]:
        raise ValueError(
            f"cross_entropy target shape {idx.shape} must match "
            f"logits.shape[:-1] {tuple(logits.shape[:-1])}")
    ls = ops.log_softmax(logits, axis=-1)
    if ls.ndim == 2:
        picked = ls[np.arange(ls.shape[0]), idx]
    else:
        flat = ls.reshape(-1, ls.shape[-1])
        picked = flat[np.arange(flat.shape[0]), idx.reshape(-1)]
    return -picked.mean()
