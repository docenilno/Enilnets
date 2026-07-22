"""Tiled ("Flash") attention (roadmap item 43).

The fused-kernel half of FlashAttention is not expressible in NumPy/CuPy;
the algorithm half is, and it is where the memory win comes from. This
streams over key blocks carrying an ONLINE SOFTMAX (running max,
normalizer and output), so the full ``(B, H, S, S)`` score matrix never
exists: O(S * block_size) instead of O(S^2), with results identical to
the plain path up to float rounding. Backward recomputes each score block
from Q/K/V using ``D = rowsum(dO * O)``, so it is O(S * block_size) too.

A strictly optional backend behind the ordinary implementation:
``tiled_block_size=None`` keeps the original code path untouched."""

from typing import Any, Optional, Tuple

from ..core.backend import np
from ..core import backend

#: Additive score for masked positions. Finite, so a fully-masked block
#: underflows to zero weight instead of producing inf - inf = NaN.
MASK_VALUE = -1e9


def _block_bias(q_pos: Any, k_pos: Any, causal: bool, window_size: Optional[int],
                slopes: Optional[Any]) -> Optional[Any]:
    """Mask + ALiBi bias for one (query block, key block) tile, or None."""
    bias = None
    if causal or window_size is not None:
        diff = q_pos[:, None] - k_pos[None, :]
        blocked = diff < 0 if causal else np.zeros(diff.shape, dtype=bool)
        if window_size is not None:
            blocked = blocked | (np.abs(diff) > window_size)
        bias = np.where(blocked, MASK_VALUE, 0.0)[None, None, :, :]
    if slopes is not None:
        diff = q_pos[:, None] - k_pos[None, :]
        alibi = -slopes[:, None, None] * (diff if causal else np.abs(diff))
        alibi = alibi[None, :, :, :]
        bias = alibi if bias is None else bias + alibi
    return bias


def flash_attention_forward(Qh: Any, Kh: Any, Vh: Any, block_size: int,
                            causal: bool = False,
                            window_size: Optional[int] = None,
                            slopes: Optional[Any] = None) -> Tuple[Any, Tuple]:
    """Streaming attention over ``(B, H, S, Dh)`` Q/K/V. Returns
    ``(context, cache)``; `cache` holds only the O(S) softmax statistics."""
    B, H, S, Dh = Qh.shape
    scale = 1.0 / np.sqrt(Dh)
    dtype = backend.default_dtype()

    out = np.zeros((B, H, S, Dh), dtype=dtype)
    row_max = np.full((B, H, S, 1), -np.inf, dtype=dtype)
    row_sum = np.zeros((B, H, S, 1), dtype=dtype)
    q_pos = np.arange(S)

    for start in range(0, S, block_size):
        stop = min(start + block_size, S)
        if causal and start > S - 1:
            break
        k_pos = np.arange(start, stop)
        scores = np.matmul(Qh, Kh[:, :, start:stop].transpose(0, 1, 3, 2)) * scale
        bias = _block_bias(q_pos, k_pos, causal, window_size, slopes)
        if bias is not None:
            scores = scores + bias

        # Online softmax: rescale everything accumulated so far to the new
        # running maximum, then fold this block in. exp(-inf - finite) is 0,
        # so the very first block initializes cleanly without a special case.
        block_max = np.max(scores, axis=-1, keepdims=True)
        new_max = np.maximum(row_max, block_max)
        rescale = np.exp(row_max - new_max)
        p = np.exp(scores - new_max)
        row_sum = row_sum * rescale + np.sum(p, axis=-1, keepdims=True)
        out = out * rescale + np.matmul(p, Vh[:, :, start:stop])
        row_max = new_max

    out = out / row_sum
    return out, (Qh, Kh, Vh, out, row_max, row_sum, block_size, causal,
                 window_size, slopes)


def flash_attention_backward(dout: Any, cache: Tuple) -> Tuple[Any, Any, Any]:
    """Gradients (dQh, dKh, dVh), recomputing score blocks rather than
    keeping the attention matrix."""
    (Qh, Kh, Vh, out, row_max, row_sum, block_size, causal,
     window_size, slopes) = cache
    B, H, S, Dh = Qh.shape
    scale = 1.0 / np.sqrt(Dh)

    dQ = np.zeros_like(Qh)
    dK = np.zeros_like(Kh)
    dV = np.zeros_like(Vh)
    # The one extra statistic FlashAttention's backward needs; it is exactly
    # the sum(dattn * attn) term of the softmax Jacobian, precomputed once.
    D = np.sum(dout * out, axis=-1, keepdims=True)
    q_pos = np.arange(S)

    for start in range(0, S, block_size):
        stop = min(start + block_size, S)
        k_pos = np.arange(start, stop)
        scores = np.matmul(Qh, Kh[:, :, start:stop].transpose(0, 1, 3, 2)) * scale
        bias = _block_bias(q_pos, k_pos, causal, window_size, slopes)
        if bias is not None:
            scores = scores + bias
        p = np.exp(scores - row_max) / row_sum          # the real attention weights

        dV[:, :, start:stop] += np.matmul(p.transpose(0, 1, 3, 2), dout)
        dp = np.matmul(dout, Vh[:, :, start:stop].transpose(0, 1, 3, 2))
        ds = p * (dp - D) * scale
        dQ += np.matmul(ds, Kh[:, :, start:stop])
        dK[:, :, start:stop] += np.matmul(ds.transpose(0, 1, 3, 2), Qh)

    return dQ, dK, dV
