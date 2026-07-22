"""Pooling variants for the graph API (roadmap item 33): adaptive
average/max pooling, fractional max pooling, and MaxUnpool.

Everything here is a composite of existing differentiable ops (slices,
reductions, gathers), so no new gradient rules exist to get wrong -- the
per-output-bin Python loops run over the (small) output size, not the
input, which is the readable trade this library prefers.
"""

from typing import Any, List, Optional, Tuple

from ..core.backend import np
from .tensor import Tensor, as_tensor
from . import ops


def _adaptive_bounds(in_size: int, out_size: int) -> List[Tuple[int, int]]:
    """PyTorch-style adaptive bin edges: bin i covers
    [floor(i*In/Out), ceil((i+1)*In/Out)) -- bins tile the input exactly
    and adjacent bins may overlap by at most one element."""
    return [(int(np.floor(i * in_size / out_size)),
             int(np.ceil((i + 1) * in_size / out_size)))
            for i in range(out_size)]


def _pool_bins(x: Tensor, h_bounds, w_bounds, reduce: str) -> Tensor:
    """Shared bin-reduction: slice each (rows, cols) bin, reduce it to
    (B, C, 1, 1), and stitch the grid back with concatenate."""
    rows = []
    for hs, he in h_bounds:
        cells = []
        for ws, we in w_bounds:
            window = x[:, :, hs:he, ws:we]
            if reduce == "avg":
                cells.append(window.mean(axis=(2, 3), keepdims=True))
            else:
                cells.append(window.max(axis=(2, 3), keepdims=True))
        rows.append(ops.concatenate(*cells, axis=3))
    return ops.concatenate(*rows, axis=2)


def adaptive_avg_pool2d(x: Any, output_size: Any) -> Tensor:
    """Average-pool ``(B, C, H, W)`` to an arbitrary fixed ``(oh, ow)``
    output regardless of input size (bin edges spread as evenly as
    possible). ``output_size``: int or (oh, ow)."""
    x = as_tensor(x)
    oh, ow = (output_size, output_size) if isinstance(output_size, int) else output_size
    return _pool_bins(x, _adaptive_bounds(x.shape[2], oh),
                      _adaptive_bounds(x.shape[3], ow), "avg")


def adaptive_max_pool2d(x: Any, output_size: Any) -> Tensor:
    """Max-pool ``(B, C, H, W)`` to a fixed ``(oh, ow)`` output (see
    :func:`adaptive_avg_pool2d` for the binning rule)."""
    x = as_tensor(x)
    oh, ow = (output_size, output_size) if isinstance(output_size, int) else output_size
    return _pool_bins(x, _adaptive_bounds(x.shape[2], oh),
                      _adaptive_bounds(x.shape[3], ow), "max")


def _fractional_bounds(in_size: int, out_size: int, u: float) -> List[Tuple[int, int]]:
    """Graham-style pseudo-random bin edges: interior boundaries at
    floor(alpha * (i + u)) for one uniform draw u, giving window sizes that
    vary between floor(alpha) and ceil(alpha)."""
    alpha = in_size / out_size
    edges = [0] + [min(in_size - 1, int(np.floor(alpha * (i + u))))
                   for i in range(1, out_size)] + [in_size]
    return list(zip(edges[:-1], edges[1:]))


def fractional_max_pool2d(x: Any, output_size: Any,
                          random_u: Optional[Tuple[float, float]] = None) -> Tensor:
    """Fractional max pooling (Graham 2014): max-pool to ``(oh, ow)`` over
    pseudo-randomly placed bins, so repeated applications don't always cut
    the image on the same grid. ``random_u``: optionally fix the two
    uniform draws (row, col) in [0, 1) for reproducibility; fresh draws
    per call otherwise. Requires ``H > oh`` and ``W > ow``."""
    x = as_tensor(x)
    oh, ow = (output_size, output_size) if isinstance(output_size, int) else output_size
    H, W = int(x.shape[2]), int(x.shape[3])
    if H <= oh or W <= ow:
        raise ValueError(
            f"fractional_max_pool2d needs input spatial dims strictly larger "
            f"than the output; got {(H, W)} -> {(oh, ow)}")
    if random_u is None:
        uh, uw = float(np.random.rand()), float(np.random.rand())
    else:
        uh, uw = random_u
    return _pool_bins(x, _fractional_bounds(H, oh, uh),
                      _fractional_bounds(W, ow, uw), "max")


def max_pool2d_with_indices(x: Any, kernel_size: int,
                            stride: Optional[int] = None) -> Tuple[Tensor, Any]:
    """Max pooling that also returns the flat spatial argmax indices
    ``(B, C, OH, OW)`` -- the piece :func:`max_unpool2d` needs to invert
    the pooling. ``stride`` defaults to ``kernel_size`` (non-overlapping,
    matching ``nn/``'s ``add_maxpool2d``)."""
    x = as_tensor(x)
    k = int(kernel_size)
    stride = k if stride is None else int(stride)
    B, C, H, W = (int(s) for s in x.shape)
    OH = (H - k) // stride + 1
    OW = (W - k) // stride + 1

    # Window membership as flat spatial indices, (OH*OW, k*k), host-built.
    hs = (np.arange(OH) * stride)[:, None, None, None]
    ws = (np.arange(OW) * stride)[None, :, None, None]
    dh = np.arange(k)[None, None, :, None]
    dw = np.arange(k)[None, None, None, :]
    flat = ((hs + dh) * W + (ws + dw)).reshape(OH * OW, k * k)

    # Argmax per window from the raw values (index selection isn't
    # differentiable; the gather below is).
    windows = x.data.reshape(B, C, H * W)[:, :, flat]        # (B, C, L, k*k)
    argmax = windows.argmax(axis=-1)                          # (B, C, L)
    indices = np.take_along_axis(
        np.broadcast_to(flat, (B, C) + flat.shape), argmax[..., None], axis=-1
    )[..., 0]                                                 # (B, C, L) flat spatial

    b_idx = np.arange(B)[:, None, None]
    c_idx = np.arange(C)[None, :, None]
    pooled = ops.reshape(x, shape=(B, C, H * W))[b_idx, c_idx, indices]
    return (ops.reshape(pooled, shape=(B, C, OH, OW)),
            indices.reshape(B, C, OH, OW))


def max_pool2d(x: Any, kernel_size: int, stride: Optional[int] = None) -> Tensor:
    """Plain max pooling (indices discarded); see
    :func:`max_pool2d_with_indices`."""
    return max_pool2d_with_indices(x, kernel_size, stride)[0]


def max_unpool2d(x: Any, indices: Any, output_size: Tuple[int, int]) -> Tensor:
    """Invert a max pooling: place each pooled value back at its argmax
    position (from :func:`max_pool2d_with_indices`), zeros elsewhere.
    ``output_size``: the pre-pooling spatial ``(H, W)``.

    Implemented as a gather from ``concat(values, zero_cell)`` -- every
    non-argmax output position indexes the shared zero cell -- so the
    whole inverse stays inside existing differentiable ops."""
    x = as_tensor(x)
    B, C, OH, OW = (int(s) for s in x.shape)
    H, W = output_size
    L = OH * OW

    values = ops.reshape(x, shape=(B * C * L,))
    pool = ops.concatenate(values, Tensor(np.zeros(1, dtype=x.dtype)), axis=0)
    zero_slot = B * C * L

    # Gather map (built on the active backend throughout -- mixing host and
    # device index arrays is a CuPy crash): output cell -> its value's slot,
    # or the shared zero slot. If overlapping windows ever produce duplicate
    # indices, the last write wins (documented; matches scatter semantics).
    idx = np.asarray(indices).reshape(B, C, L).astype(np.int64)
    gather = np.full(B * C * H * W, zero_slot, dtype=np.int64)
    flat_targets = ((np.arange(B)[:, None, None] * C +
                     np.arange(C)[None, :, None]) * (H * W) + idx).reshape(-1)
    gather[flat_targets] = np.arange(B * C * L)
    return ops.reshape(pool[gather], shape=(B, C, H, W))
