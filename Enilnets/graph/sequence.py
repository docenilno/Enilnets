"""Variable-length sequence batching for the graph API (roadmap item 28):
padding masks, packed sequences, and a graph-native multi-head attention
layer that honors them.

A **padding mask** is a boolean ``(batch, seq_len)`` array, True = real
token, built by :func:`lengths_to_mask`. A **PackedSequence** stores only
the real tokens, concatenated example-major into
``(total_tokens, features)``, plus the lengths; packing and unpacking are
differentiable gathers, so gradients flow through both directions."""

import math
from typing import Any, Optional

from ..core.backend import np
from ..core import backend
from ..nn.weight_init import init_weights
from ..nn.layers import _validate_num_kv_heads
from .tensor import Tensor, as_tensor
from .layers import Layer, Parameter
from . import ops


def lengths_to_mask(lengths: Any, max_len: Optional[int] = None) -> Any:
    """Per-example lengths -> boolean padding mask ``(batch, max_len)``,
    True where a token is real. Plain array (masks aren't differentiable)."""
    lengths = np.asarray(lengths).astype(np.int64)
    if max_len is None:
        max_len = int(lengths.max())
    return np.arange(max_len)[None, :] < lengths[:, None]


class PackedSequence:
    """Real tokens only, concatenated example-major: ``data`` is a Tensor of
    shape ``(sum(lengths), features)``; ``lengths`` records each example's
    token count. Produced by :func:`pack_padded`; restored by
    :func:`pad_packed`. Gradients flow through both."""

    def __init__(self, data: Tensor, lengths: Any) -> None:
        self.data = data
        self.lengths = np.asarray(lengths).astype(np.int64)

    @property
    def batch_size(self) -> int:
        return len(self.lengths)

    def __repr__(self) -> str:
        return (f"PackedSequence(total_tokens={self.data.shape[0]}, "
                f"batch_size={self.batch_size})")


def pack_padded(padded: Any, lengths: Any) -> PackedSequence:
    """``(batch, seq, features)`` + lengths -> :class:`PackedSequence`,
    dropping every padding position. Differentiable (a gather)."""
    padded = as_tensor(padded)
    lengths = np.asarray(lengths).astype(np.int64)
    B, S = padded.shape[0], padded.shape[1]
    mask = lengths_to_mask(lengths, S)                  # (B, S) bool
    flat = ops.reshape(padded, shape=(B * S,) + tuple(padded.shape[2:]))
    token_index = np.arange(B * S)[mask.reshape(-1)]    # positions of real tokens
    return PackedSequence(flat[token_index], lengths)


def pad_packed(packed: PackedSequence, max_len: Optional[int] = None,
               pad_value: float = 0.0) -> Tensor:
    """Inverse of :func:`pack_padded`: scatter tokens back into a padded
    ``(batch, max_len, features)`` Tensor.

    Implemented as a gather from ``concat(tokens, pad_row)`` -- every
    padding slot indexes the shared pad row -- so the whole round trip
    stays inside existing differentiable ops (the pad row's gradient is
    simply discarded with it)."""
    lengths = packed.lengths
    B = packed.batch_size
    S = int(lengths.max()) if max_len is None else max_len
    feat_shape = tuple(packed.data.shape[1:])

    pad_row = Tensor(np.full((1,) + feat_shape, pad_value,
                             dtype=packed.data.dtype))
    pool = ops.concatenate(packed.data, pad_row, axis=0)   # (total+1, F)
    pad_slot = int(packed.data.shape[0])

    # Build the (B*S,) gather map on the host: real slots point at their
    # token's row in the pool, padding slots at the shared pad row.
    mask = backend.to_numpy(lengths_to_mask(lengths, S)).reshape(-1)
    index = np.full(B * S, pad_slot, dtype=np.int64)
    index[np.asarray(mask)] = np.arange(pad_slot)
    return ops.reshape(pool[index], shape=(B, S) + feat_shape)


class MultiHeadAttention(Layer):
    """Graph-native multi-head self-attention with padding-mask support.

    Weight conventions match ``add_multihead_attention`` (``Wq``/``Wo``
    ``(embed_dim, embed_dim)``, ``Wk``/``Wv`` ``(num_kv_heads*head_dim,
    embed_dim)``, projections ``x @ W.T + b``), so weights are shareable
    with an ``nn/`` layer dict by reference.

    forward(x, key_padding_mask=None): ``x`` is ``(batch, seq, embed_dim)``;
    ``key_padding_mask`` is ``(batch, seq)`` boolean, True = real token.
    ``causal=True`` additionally applies the autoregressive mask.
    ``num_kv_heads``: None = num_heads (MHA), 1 = MQA, any divisor = GQA.
    ``window_size``: sliding window -- attend only where ``|i - j| <= w``.
    """

    #: Additive score for masked positions. Finite (not -inf) so exp() is a
    #: clean zero-underflow instead of inf-minus-inf NaN bait in backward.
    _MASK_VALUE = -1e9

    def __init__(self, embed_dim: int, num_heads: int = 4, causal: bool = False,
                 init_method: str = "xavier_uniform",
                 num_kv_heads: Optional[int] = None,
                 window_size: Optional[int] = None) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = _validate_num_kv_heads(num_heads, num_kv_heads,
                                                   "MultiHeadAttention")
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        if window_size is not None and window_size < 0:
            raise ValueError(f"window_size must be >= 0 or None, got {window_size}")
        self.window_size = window_size
        kv_dim = self.num_kv_heads * self.head_dim
        for name, out_dim in (("Wq", embed_dim), ("Wk", kv_dim),
                              ("Wv", kv_dim), ("Wo", embed_dim)):
            w, b = init_weights(embed_dim, out_dim, method=init_method)
            setattr(self, name, Parameter(w, name=name))
            setattr(self, "b" + name[1:].lower(), Parameter(b, name="b" + name[1:].lower()))

    def _repeat_kv(self, t: Tensor, repeats: int, B: int, S: int) -> Tensor:
        """Expand num_kv_heads K/V heads to num_heads query heads (MQA/GQA).
        Built from concatenate so the group-sum in backward comes for free:
        every copy is the same tensor, so its gradient accumulates."""
        if repeats == 1:
            return t
        Hkv, Dh = self.num_kv_heads, self.head_dim
        grouped = ops.reshape(t, shape=(B, Hkv, 1, S, Dh))
        expanded = ops.concatenate(*([grouped] * repeats), axis=2)
        return ops.reshape(expanded, shape=(B, Hkv * repeats, S, Dh))

    def forward(self, x: Tensor, key_padding_mask: Optional[Any] = None) -> Tensor:
        B, S, E = x.shape
        H, Hkv, Dh = self.num_heads, self.num_kv_heads, self.head_dim

        def project(w: Parameter, b: Parameter) -> Tensor:
            return ops.add(ops.matmul(x, ops.transpose(w)), b)

        def split_heads(t: Tensor, heads: int) -> Tensor:
            return ops.transpose(ops.reshape(t, shape=(B, S, heads, Dh)),
                                 axes=(0, 2, 1, 3))

        Qh = split_heads(project(self.Wq, self.bq), H)
        Kh = self._repeat_kv(split_heads(project(self.Wk, self.bk), Hkv), H // Hkv, B, S)
        Vh = self._repeat_kv(split_heads(project(self.Wv, self.bv), Hkv), H // Hkv, B, S)

        scores = ops.matmul(Qh, ops.transpose(Kh, axes=(0, 1, 3, 2)))
        scores = ops.mul(scores, Tensor(np.asarray(1.0 / math.sqrt(Dh),
                                                   dtype=x.dtype)))
        bias = np.zeros((B, 1, S, S), dtype=x.dtype)
        if key_padding_mask is not None:
            mask = np.asarray(key_padding_mask)
            if mask.shape != (B, S):
                raise ValueError(
                    f"key_padding_mask shape {mask.shape} must be (batch, seq) = {(B, S)}")
            bias = bias + np.where(mask, 0.0, self._MASK_VALUE)[:, None, None, :]
        if self.causal or self.window_size is not None:
            # Same rule as nn/'s _window_mask, but with a finite mask value
            # (see _MASK_VALUE) instead of -inf.
            diff = np.arange(S)[:, None] - np.arange(S)[None, :]
            blocked = diff < 0 if self.causal else np.zeros(diff.shape, dtype=bool)
            if self.window_size is not None:
                blocked = blocked | (np.abs(diff) > self.window_size)
            bias = bias + np.where(blocked, self._MASK_VALUE, 0.0)[None, None, :, :]
        scores = ops.add(scores, Tensor(bias.astype(x.dtype)))

        attn = ops.softmax(scores, axis=-1)
        context = ops.matmul(attn, Vh)
        context = ops.reshape(ops.transpose(context, axes=(0, 2, 1, 3)),
                              shape=(B, S, E))
        return ops.add(ops.matmul(context, ops.transpose(self.Wo)), self.bo)


def masked_mean(x: Any, mask: Any) -> Tensor:
    """Mean over the sequence axis counting only real tokens: ``x`` is
    ``(batch, seq, features)``, ``mask`` ``(batch, seq)`` boolean. The
    standard padding-aware pooling before a classification head."""
    x = as_tensor(x)
    m = np.asarray(mask).astype(x.dtype)
    weighted = ops.mul(x, Tensor(m[:, :, None]))            # zero the padding
    counts = Tensor(np.maximum(m.sum(axis=1), 1.0)[:, None]) # (B, 1) real-token counts
    return ops.div(weighted.sum(axis=1), counts)             # (B, F)
