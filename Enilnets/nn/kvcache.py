"""KV-cache incremental decoding for ``nn/`` causal attention stacks
(roadmap item 37).

Supported layer types: ``embedding``, ``positional_encoding``,
``layernorm``, ``dense``, ``dropout`` (inference no-op), ``residual_save``/
``residual_add``, ``moe``, and CAUSAL ``multihead_attention`` (absolute / RoPE /
ALiBi, MHA / MQA / GQA, with or without a sliding window). Anything else
raises.
"""

from typing import Any, Dict

from ..core.backend import np
from ..core import constants
from .activations import activate
from .forward import (_rope_cos_sin, _rope_rotate, _alibi_slopes, _repeat_kv,
                      _window_mask)
from .attention_kernels import feature_map, performer_stab
from .sparse_attention import block_selection, MASK_VALUE
from .moe import moe_forward


class KVCache:
    """Per-stream decoding state: cached K/V per attention layer, cached
    residual-save activations, and the absolute position of the next token.

    ``kv[layer_index]`` is ``(K, V, start)`` -- `start` is the absolute
    position of the cache's column 0, which is nonzero once a sliding
    window has evicted the oldest entries. Linearized-kernel layers instead
    keep a fixed-size running state in ``linear[layer_index]``."""

    def __init__(self) -> None:
        self.kv: Dict[int, Any] = {}
        self.linear: Dict[int, Any] = {}
        self.residual: Dict[int, Any] = {}
        self.position: int = 0

    def __repr__(self) -> str:
        return (f"KVCache(position={self.position}, "
                f"attention_layers={sorted(self.kv)})")

    def reorder(self, index: Any) -> None:
        """Gather every cached tensor along the BATCH axis by `index`.

        Beam search needs this: after pruning, each surviving beam descends
        from some parent beam, so its cached keys and values are the
        parent's. Without the reorder every beam would keep reading the
        history of whatever row it happens to occupy."""
        idx = np.asarray(index)
        for store, layout in ((self.kv, "kv"), (self.linear, "linear"),
                              (self.residual, "plain")):
            for key, value in list(store.items()):
                if layout == "plain":
                    store[key] = value[idx]
                elif layout == "kv":
                    k, v, start = value
                    store[key] = (k[idx], v[idx], start)
                else:
                    kv, ksum, stab = value
                    store[key] = (kv[idx], ksum[idx],
                                  None if stab is None else stab[idx])

    def expand(self, size: int) -> None:
        """Replicate a batch-1 cache to `size` rows -- how a beam search
        starts, from one primed prompt."""
        self.reorder([0] * int(size))


def cached_forward_step(model: Any, token_ids: Any, cache: KVCache,
                        advance_position: bool = True) -> Any:
    """Step `token_ids` -- an int array ``(batch, n_new)``, or ``(n_new,)``
    treated as batch 1 -- through `model`'s layers, reading and updating
    `cache`. Returns ``(batch, n_new, out_dim)`` for the new positions only.

    Multi-token steps get the causal mask among themselves, so this is
    exactly equivalent to a full ``Forward()`` over the same tokens.
    ``advance_position=False`` leaves ``cache.position`` to the caller."""
    x = np.asarray(token_ids)
    if x.ndim == 1:
        x = x[None, :]
    B, S_new = int(x.shape[0]), int(x.shape[1])
    pos = cache.position

    for idx, layer in enumerate(model.layers):
        t = layer["type"]
        if t == "embedding":
            x = layer["weights"][x]                               # (B, S, E)
        elif t == "positional_encoding":
            table = layer["weights"] if layer.get("_pos_type") == "learnable" \
                else layer["pe"]
            x = x + table[pos:pos + S_new][None, :, :]
        elif t == "layernorm":
            eps = layer.get("epsilon", 1e-5)
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.var(x, axis=-1, keepdims=True)
            x = layer["gamma"].reshape(1, 1, -1) * ((x - mean) / np.sqrt(var + eps)) \
                + layer["beta"].reshape(1, 1, -1)
        elif t == "dense":
            z = np.dot(x, layer["weights"].T) + layer["bias"]
            x = activate(layer["activation"], z, **layer.get("activation_params", {}))
        elif t == "dropout":
            pass                                                  # inference no-op
        elif t == "residual_save":
            cache.residual[idx] = x
        elif t == "residual_add":
            x = x + cache.residual[layer["save_index"]]
        elif t == "moe":
            # Position-wise, like dense: each token routes independently, so
            # a step needs no history at all.
            x, _ = moe_forward(x, layer, training=False)
        elif t == "multihead_attention":
            x = _attention_step(layer, x, idx, cache, B, S_new, pos)
        else:
            raise ValueError(
                f"KV-cache stepping doesn't support layer type '{t}' "
                f"(layer {idx}); use a full Forward() for this architecture."
            )
    if advance_position:
        cache.position += S_new
    return x


def _attention_step(layer: Dict[str, Any], x: Any, idx: int, cache: KVCache,
                    B: int, S_new: int, pos: int) -> Any:
    """One attention layer's step: project the new tokens, append their K/V
    to the cache, attend over [cached + new]. Raises if non-causal."""
    if not layer.get("causal", False):
        raise ValueError(
            f"KV-cache stepping requires causal attention; layer {idx} is "
            "non-causal (its full output at earlier positions would change "
            "when later tokens arrive, which a cache cannot represent)."
        )
    H, Dh, E = layer["num_heads"], layer["head_dim"], layer["embed_dim"]
    Hkv = layer.get("num_kv_heads", H)
    scheme = layer.get("positional_scheme", "absolute")

    Q = np.dot(x, layer["Wq"].T) + layer["bq"]
    K_new = np.dot(x, layer["Wk"].T) + layer["bk"]
    V_new = np.dot(x, layer["Wv"].T) + layer["bv"]
    Qh = Q.reshape(B, S_new, H, Dh).transpose(0, 2, 1, 3)
    Kh_new = K_new.reshape(B, S_new, Hkv, Dh).transpose(0, 2, 1, 3)
    Vh_new = V_new.reshape(B, S_new, Hkv, Dh).transpose(0, 2, 1, 3)

    if scheme == "rope":
        # Rotate Q/K at their ABSOLUTE positions (pos .. pos+S_new-1); the
        # cache stores already-rotated K, the standard trick.
        cos, sin = _rope_cos_sin(pos + S_new, Dh, constants.SINUSOIDAL_BASE)
        Qh = _rope_rotate(Qh, cos[pos:], sin[pos:])
        Kh_new = _rope_rotate(Kh_new, cos[pos:], sin[pos:])

    kernel = layer.get("attention_kernel", "softmax")
    if kernel != "softmax":
        return _linear_attention_step(layer, kernel, Qh, Kh_new, Vh_new,
                                      idx, cache, B, S_new, H, Hkv, E)

    prev = cache.kv.get(idx)
    if prev is None:
        Kh_all, Vh_all, start = Kh_new, Vh_new, pos
    else:
        Kh_all = np.concatenate([prev[0], Kh_new], axis=2)
        Vh_all = np.concatenate([prev[1], Vh_new], axis=2)
        start = prev[2]

    # Sliding-window attention: positions older than the newest query's
    # window can never be attended to again, so DROP them rather than mask
    # them. That is what makes SWA decode in bounded memory -- the whole
    # point of the variant. `start` tracks the absolute position of the
    # cache's column 0, which the ALiBi bias and the mask below need.
    window_size = layer.get("window_size")
    if window_size is not None:
        # Keep what the OLDEST query in this step needs (position `pos`), not
        # the newest: in a multi-token step the earlier queries still see
        # further back, and evicting on the newest query's window would leave
        # their rows fully masked -- a NaN softmax. Steady state after a
        # 1-token step is still exactly window_size + 1 entries.
        oldest_needed = pos - window_size
        drop = max(0, oldest_needed - start)
        if drop:
            Kh_all, Vh_all = Kh_all[:, :, drop:], Vh_all[:, :, drop:]
            start += drop

    # The cache holds the UNexpanded Hkv heads -- shrinking it by
    # num_heads/num_kv_heads is the entire point of MQA/GQA. The expansion
    # to H query heads happens per step, on the read side only.
    cache.kv[idx] = (Kh_all, Vh_all, start)
    S_tot = int(Kh_all.shape[2])
    repeats = H // Hkv
    Kh_all, Vh_all = _repeat_kv(Kh_all, repeats), _repeat_kv(Vh_all, repeats)

    q_pos = pos + np.arange(S_new)
    k_pos = start + np.arange(S_tot)

    sparse_spec = layer.get("sparse_pattern")
    if sparse_spec is not None:
        return _sparse_attention_step(layer, sparse_spec, Qh, Kh_all, Vh_all,
                                      q_pos, k_pos, B, S_new, H, Dh, E)

    scores = np.matmul(Qh, Kh_all.transpose(0, 1, 3, 2)) / np.sqrt(Dh)
    if scheme == "alibi":
        # Distance bias at absolute positions: query i vs key j.
        slopes = _alibi_slopes(H)
        distance = q_pos[:, None] - k_pos[None, :]
        scores = scores + (-slopes[:, None, None] * distance[None, :, :])[None]
    # Causality among the new tokens themselves (a cached key is always at a
    # position < pos, hence always visible), plus the window's lower edge.
    mask = _window_mask(q_pos, k_pos, causal=True, window_size=window_size)
    if mask is not None:
        scores = scores + mask[None, None, :, :]

    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    context = np.matmul(attn, Vh_all).transpose(0, 2, 1, 3).reshape(B, S_new, E)
    return np.dot(context, layer["Wo"].T) + layer["bo"]


def _linear_attention_step(layer, kernel, Qh, Kh_new, Vh_new, idx, cache,
                           B, S_new, H, Hkv, E):
    """Incremental step for a linearized-kernel layer. Causal linear
    attention is a linear RNN: its whole history compresses into a running
    ``sum(phi(k) (x) v)`` and ``sum(phi(k))``, so decoding needs a
    FIXED-size state -- no growing K/V cache at all, at any length."""
    omega = layer.get("omega")
    repeats = H // Hkv
    Kh_full = _repeat_kv(Kh_new, repeats)
    Vh = _repeat_kv(Vh_new, repeats)
    prev = cache.linear.get(idx)

    # Every key in the running sum must share ONE stabilizer scale, or the
    # sum mixes incompatible units. Merge the carried scale with this step's
    # the logsumexp way: take the larger, rescale the smaller side down.
    # (The final ratio is scale-invariant, so any consistent scale is exact.)
    stab = None
    rescale = None
    if kernel == "performer":
        stab_new = performer_stab(Kh_full, omega, "global")
        if prev is None:
            stab = stab_new
        else:
            stab = np.maximum(prev[2], stab_new)
            rescale = np.exp(prev[2] - stab)[..., 0]       # (B, H, 1)
    Qp = feature_map(Qh, kernel, omega, "row")
    Kp = feature_map(Kh_full, kernel, omega, "global", stab=stab)

    # Prefix sums WITHIN this step, offset by the state carried in.
    kv_step = np.cumsum(Kp[..., :, None] * Vh[..., None, :], axis=2)
    ksum_step = np.cumsum(Kp, axis=2)
    if prev is not None:
        kv_prev, ksum_prev = prev[0], prev[1]
        if rescale is not None:
            kv_prev = kv_prev * rescale[..., None]
            ksum_prev = ksum_prev * rescale
        kv_step = kv_step + kv_prev[:, :, None, :, :]
        ksum_step = ksum_step + ksum_prev[:, :, None, :]
    cache.linear[idx] = (kv_step[:, :, -1], ksum_step[:, :, -1], stab)

    num = np.einsum("bhsf,bhsfd->bhsd", Qp, kv_step)
    den = np.sum(Qp * ksum_step, axis=-1, keepdims=True) + 1e-20
    context = (num / den).transpose(0, 2, 1, 3).reshape(B, S_new, E)
    return np.dot(context, layer["Wo"].T) + layer["bo"]


def _sparse_attention_step(layer, spec, Qh, Kh_all, Vh_all, q_pos, k_pos,
                           B, S_new, H, Dh, E):
    """Incremental step for a block-sparse layer.

    The cache keeps the full K/V history -- a sparse pattern may select any
    earlier block, so nothing can be evicted -- but each query only ever
    scores the keys inside its own selected blocks, so the per-step compute
    stays bounded rather than growing with the context."""
    bs = spec["block_size"]
    n_blocks = int(k_pos[-1]) // bs + 1
    index, valid = block_selection(spec, n_blocks, causal=True)

    # Union of key blocks any of this step's queries needs, so the gather is
    # one contiguous slice-free take rather than a per-query loop.
    q_blocks = sorted({int(p) // bs for p in q_pos})
    needed = sorted({int(index[i, w]) for i in q_blocks
                     for w in range(index.shape[1]) if valid[i, w]})
    wanted = np.zeros(int(k_pos[-1]) + 1, dtype=bool)
    for b in needed:
        wanted[b * bs:(b + 1) * bs] = True
    take = np.asarray([j for j, p in enumerate(k_pos) if wanted[int(p)]])
    Kg, Vg, kg_pos = Kh_all[:, :, take], Vh_all[:, :, take], k_pos[take]

    scores = np.matmul(Qh, Kg.transpose(0, 1, 3, 2)) / np.sqrt(Dh)
    if layer.get("positional_scheme") == "alibi":
        slopes = _alibi_slopes(H)
        distance = q_pos[:, None] - kg_pos[None, :]
        scores = scores + (-slopes[:, None, None] * distance[None, :, :])[None]

    # A key is visible iff it is causally allowed AND its block is in this
    # particular query's selection (not merely in the union gathered above).
    allowed = np.zeros((len(q_pos), len(kg_pos)), dtype=bool)
    for qi, p in enumerate(q_pos):
        sel = {int(index[int(p) // bs, w]) for w in range(index.shape[1])
               if valid[int(p) // bs, w]}
        for ki, kp in enumerate(kg_pos):
            allowed[qi, ki] = int(kp) <= int(p) and int(kp) // bs in sel
    scores = scores + np.where(allowed, 0.0, MASK_VALUE)[None, None, :, :]

    scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    context = np.matmul(attn, Vg).transpose(0, 2, 1, 3).reshape(B, S_new, E)
    return np.dot(context, layer["Wo"].T) + layer["bo"]
