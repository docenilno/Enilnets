"""NeuralNet.Forward and the per-layer-type forward-pass math it dispatches
to. Populates self.outputs/self.pre_activations/self.*_cache as it goes,
which backward.py's Backward() then consumes."""
from typing import Any, Dict, Optional, Tuple

from ..core.backend import np
from ..core import backend
from .activations import activate
from . import attention_kernels
from . import sparse_attention
from . import moe
from . import flash_attention
from ..core import constants

def im2col(input_data: Any, filter_h: int, filter_w: int, stride: int = 1, pad: int = 0) -> Any:
    """Unroll every filter_h x filter_w patch of a (N,C,H,W) image batch
    into a row, via a zero-copy strided view -- the standard trick for
    turning a convolution into one big matmul against the flattened
    kernel. Returns (N*out_h*out_w, C*filter_h*filter_w)."""
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant')

    N_stride, C_stride, H_stride, W_stride = img.strides
    shape = (N, C, filter_h, filter_w, out_h, out_w)
    strides = (N_stride, C_stride, H_stride, W_stride, H_stride * stride, W_stride * stride)

    col = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    return col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

def im2col1d(input_data: Any, filter_k: int, stride: int = 1, pad: int = 0) -> Any:
    """1D analogue of im2col: unrolls every filter_k-wide patch of a
    (N,C,L) sequence batch into a row. Returns (N*out_l, C*filter_k)."""
    N, C, L = input_data.shape
    out_l = (L + 2 * pad - filter_k) // stride + 1
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad)], mode='constant')

    N_stride, C_stride, L_stride = img.strides
    shape = (N, C, filter_k, out_l)
    strides = (N_stride, C_stride, L_stride, L_stride * stride)

    col = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    return col.transpose(0, 3, 1, 2).reshape(N * out_l, -1)

def _rope_cos_sin(S: int, head_dim: int, base: float) -> Tuple[Any, Any]:
    """Standard RoPE angles: theta_i = base**(-2i/head_dim) for i in
    0..head_dim/2-1, applied at integer positions 0..S-1. Returns
    (cos, sin), each (S, head_dim/2) -- broadcasts directly against a
    (..., S, head_dim/2) array (numpy aligns trailing dims)."""
    half = head_dim // 2
    theta = base ** (-2.0 * np.arange(half) / head_dim)
    pos = np.arange(S, dtype=backend.default_dtype())
    angles = pos[:, None] * theta[None, :]
    return np.cos(angles), np.sin(angles)

def _attn_dropout_forward(attn: Any, layer: Dict[str, Any], training: bool) -> Tuple[Any, Optional[Any]]:
    """Apply attention-weight dropout (post-softmax, pre-context-matmul --
    the standard place for it in a Transformer). Returns (attn_used, mask):
    attn_used is what should actually be matmul'd against V; mask is cached
    on `layer["_attn_dropout_mask"]` (None if inactive) for
    _attn_dropout_backward to replay the identical mask/scale. `attn`
    itself (the real softmax probabilities) is left untouched and still
    what's cached in attention_cache, so existing consumers (e.g. the
    softmax-sums-to-1 sanity check) are unaffected."""
    rate = layer.get("dropout", 0.0)
    if not (training and rate > 0):
        layer["_attn_dropout_mask"] = None
        return attn, None
    if rate >= 1.0:
        mask = np.zeros_like(attn, dtype=backend.default_dtype())
        attn_used = np.zeros_like(attn)
    else:
        mask = (np.random.rand(*attn.shape) > rate).astype(backend.default_dtype())
        attn_used = attn * mask / (1.0 - rate)
    layer["_attn_dropout_mask"] = mask
    return attn_used, mask


def _rope_rotate(x: Any, cos: Any, sin: Any) -> Any:
    """Rotate the last axis of `x` (..., head_dim) by per-position-pair
    angles (the "rotate_half" formulation used by most modern RoPE
    implementations -- mathematically equivalent to the interleaved-pairs
    formulation up to a fixed permutation of head_dim, which cancels out
    since it's applied identically to Q and K before their dot product)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

def _alibi_slopes(num_heads: int) -> Any:
    """Standard ALiBi geometric-sequence per-head slopes."""
    def slopes_pow2(n):
        start = 2.0 ** (-8.0 / n)
        return [start ** (i + 1) for i in range(n)]
    if (num_heads & (num_heads - 1)) == 0:  # power of 2
        return np.array(slopes_pow2(num_heads))
    closest_pow2 = 2 ** int(np.floor(np.log2(num_heads)))
    slopes = slopes_pow2(closest_pow2)
    extra = slopes_pow2(2 * closest_pow2)[0::2][: num_heads - closest_pow2]
    return np.array(slopes + extra)

def _window_mask(q_pos: Any, k_pos: Any, causal: bool,
                 window_size: Optional[int]) -> Optional[Any]:
    """Additive attention mask for absolute query/key positions: 0 where a
    query may attend, -inf where it may not. Returns None when nothing is
    masked. `causal` forbids j > i; `window_size` (None = unbounded) forbids
    |i - j| > window_size, i.e. sliding-window attention.

    Note the two compose orthogonally, and j == i is always allowed, so no
    row is ever fully masked (which would make the softmax NaN)."""
    if not causal and window_size is None:
        return None
    diff = q_pos[:, None] - k_pos[None, :]              # i - j
    blocked = diff < 0 if causal else np.zeros(diff.shape, dtype=bool)
    if window_size is not None:
        blocked = blocked | (np.abs(diff) > window_size)
    return np.where(blocked, -np.inf, 0.0)


def spp_bounds(size: int, out: int) -> Any:
    """Adaptive pooling bin edges: bin i covers
    [floor(i*size/out), ceil((i+1)*size/out)). Bins tile the input exactly
    and adjacent bins may overlap by one, matching the usual adaptive-pool
    convention."""
    return [(int(np.floor(i * size / out)), int(np.ceil((i + 1) * size / out)))
            for i in range(out)]


def _argmax_over(x: Any, axes: Any) -> Any:
    """Flat argmax of `x` over `axes`, as an index into those axes raveled.
    Used so the backward pass can route a max's gradient to exactly the
    element that produced it."""
    moved = np.moveaxis(x, axes, range(-len(axes), 0))
    flat = moved.reshape(moved.shape[:-len(axes)] + (-1,))
    return flat.argmax(axis=-1)


def gate_broadcast(gate: Any, target: Any) -> Any:
    """Shape `gate` so it broadcasts against `target` by appending trailing
    singleton axes until the ranks match.

    A channel gate is (B, C) against a (B, C, H, W) feature map, and a
    spatial gate is (B, 1, H, W) which already aligns -- appending on the
    RIGHT is what makes both work, since NumPy aligns trailing axes and
    would otherwise read (B, C) as gating the last two dims."""
    extra = target.ndim - gate.ndim
    if extra < 0:
        raise ValueError(
            f"gate has more axes than what it gates: {gate.shape} vs {target.shape}")
    return gate.reshape(gate.shape + (1,) * extra) if extra else gate


def gate_reduce(grad: Any, gate_shape: Any) -> Any:
    """Adjoint of :func:`gate_broadcast`: sum `grad` back down to the gate's
    own shape, over both the appended axes and any axis the gate had as a
    singleton (a spatial gate's channel axis, say)."""
    extra = grad.ndim - len(gate_shape)
    out = grad.sum(axis=tuple(range(len(gate_shape), grad.ndim))) if extra else grad
    for axis, size in enumerate(gate_shape):
        if size == 1 and out.shape[axis] != 1:
            out = out.sum(axis=axis, keepdims=True)
    return out.reshape(gate_shape)


def _repeat_kv(t: Any, repeats: int) -> Any:
    """MQA/GQA: expand K/V heads to the query heads, (B, Hkv, S, Dh) ->
    (B, Hkv*repeats, S, Dh). Group-major: query head h reads K/V group
    h // repeats (backward.py's ``_sum_kv_groups`` assumes this)."""
    if repeats == 1:
        return t
    return np.repeat(t, repeats, axis=1)


def batchnorm_forward(x: Any, layer: Dict[str, Any], training: bool) -> Tuple[Any, Optional[Tuple]]:
    """BatchNorm forward for a (N,C) or (N,C,H,W) input. In training mode,
    normalizes over the batch (and spatial, for 4D) axes and updates the
    layer's running mean/var; in eval mode, normalizes using the stored
    running statistics instead. Returns (output, cache) -- cache is None
    in eval mode (Backward() never needs it there)."""
    epsilon = layer.get("epsilon", 1e-5)
    momentum = layer.get("momentum", 0.1)
    ndim = x.ndim

    if ndim == 2:
        axes = 0
        shape = (1, -1)
    elif ndim == 4:
        axes = (0, 2, 3)
        shape = (1, -1, 1, 1)
    else:
        raise ValueError(f"BatchNorm only supports 2D or 4D inputs, got {ndim}D")

    if training:
        mean = np.mean(x, axis=axes)
        variance = np.var(x, axis=axes)
        x_norm = (x - mean.reshape(shape)) / np.sqrt(variance.reshape(shape) + epsilon)
        out = layer["gamma"].reshape(shape) * x_norm + layer["beta"].reshape(shape)
        layer["running_mean"] = (1 - momentum) * layer["running_mean"] + momentum * mean
        layer["running_var"] = (1 - momentum) * layer["running_var"] + momentum * variance
        cache = (x, x_norm, mean, variance, layer["gamma"], epsilon, axes)
    else:
        x_norm = (x - layer["running_mean"].reshape(shape)) / np.sqrt(layer["running_var"].reshape(shape) + epsilon)
        out = layer["gamma"].reshape(shape) * x_norm + layer["beta"].reshape(shape)
        cache = None
    return out, cache

def layernorm_forward(x: Any, layer: Dict[str, Any], training: bool) -> Tuple[Any, Tuple]:
    """LayerNorm forward for a (N,C) 2D, (N,S,E) 3D (transformer-style,
    normalized per-token over the embedding axis), or (N,C,H,W) 4D input.
    Always normalizes using the CURRENT input's own statistics (unlike
    batchnorm, there's no running-average state, so `training` doesn't
    change the computation -- only whether it matters is which axes are
    normalized over)."""
    epsilon = layer.get("epsilon", 1e-5)
    ndim = x.ndim
    if ndim == 2:
        axes = 1
        # gamma/beta are 1D arrays of length num_features
        # For 2D input (batch, features), broadcast as (1, features)
        gamma = layer["gamma"].reshape(1, -1)
        beta = layer["beta"].reshape(1, -1)
    elif ndim == 4:
        axes = (1, 2, 3)
        # For 4D input (batch, C, H, W): normalized_shape as a bare int or
        # 1-tuple means per-channel affine (gamma/beta sized C, broadcast
        # (1,C,1,1)); a 3-tuple (C,H,W) means a full elementwise affine
        # over every normalized position (gamma/beta sized C*H*W,
        # broadcast (1,C,H,W)) -- add_layernorm sizes gamma/beta as
        # math.prod(normalized_shape) either way, so this must match
        # whichever shape the caller actually asked for instead of always
        # assuming size C (which crashed whenever normalized_shape was a
        # multi-element tuple, since gamma's real size was C*H*W).
        shape = layer["normalized_shape"]
        if isinstance(shape, int) or len(shape) == 1:
            gamma_bcast_shape = (1, x.shape[1], 1, 1)
        else:
            gamma_bcast_shape = (1,) + tuple(shape)
        gamma = layer["gamma"].reshape(gamma_bcast_shape)
        beta = layer["beta"].reshape(gamma_bcast_shape)
    elif ndim == 3:
        # For 3D sequence input (batch, seq_len, embed_dim), normalize per-token
        # over the embedding axis only, as in standard Transformer LayerNorm.
        axes = 2
        gamma = layer["gamma"].reshape(1, 1, -1)
        beta = layer["beta"].reshape(1, 1, -1)
    else:
        raise ValueError(f"LayerNorm only supports 2D, 3D or 4D inputs, got {ndim}D")

    mean = np.mean(x, axis=axes, keepdims=True)
    variance = np.var(x, axis=axes, keepdims=True)
    x_norm = (x - mean) / np.sqrt(variance + epsilon)
    out = gamma * x_norm + beta
    cache = (x, x_norm, mean, variance, layer["gamma"], epsilon, axes)
    return out, cache

def _initial_rnn_state(layer, B, H, key="_state_h"):
    """Initial hidden/cell state for an RNN-family layer: zeros normally;
    the carried state from the previous Forward call when the layer was
    built with stateful=True (roadmap item 29). Carried state is treated
    as a constant for BPTT (standard truncated-BPTT semantics)."""
    if layer.get("stateful") and layer.get(key) is not None:
        state = layer[key]
        if state.shape[0] != B:
            raise ValueError(
                f"stateful {layer['type']} carried a state for batch size "
                f"{state.shape[0]} but this batch has {B} examples -- call "
                "reset_rnn_state() when the stream/batch layout changes."
            )
        return state
    return np.zeros((B, H), dtype=backend.default_dtype())

def Forward(self: Any, inputs: Any, training: bool = False, dropout_rate: float = 0.0) -> Any:
    """Run `inputs` through every layer in self.layers in order, populating
    self.outputs (each layer's output, outputs[0] is the input itself),
    self.pre_activations, and self.{batchnorm,layernorm,attention,conv,
    rnn}_cache for Backward() to consume afterward. Returns the final
    layer's output.

    training: enables dropout/batchnorm's training-mode behavior (batch
        statistics instead of running averages, actual masking instead of
        a no-op).
    dropout_rate: fallback rate for any "dropout" layer that didn't get an
        explicit per-layer rate at add_dropout() time.
    """
    x = np.asarray(inputs, dtype=backend.default_dtype())
    first_type = self.layers[0]["type"] if self.layers else None
    if x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim == 3 and first_type == "conv2d":
        # (C, H, W) single-sample conv convenience path. Any other layer type
        # taking 3D input (attention, positional encoding, layernorm, dense)
        # uses (batch, seq_len, embed_dim) directly, already batched.
        x = x.reshape(1, *x.shape)

    self.outputs = [x]
    self.pre_activations = [None]
    self.batchnorm_cache = []
    self.layernorm_cache = []
    self.attention_cache = []
    self.moe_cache = []
    self.conv_cache = []
    self.rnn_cache = []

    for layer in self.layers:
        x = self.outputs[-1]
        attn_cache_entry = None
        conv_cache_entry = None
        rnn_cache_entry = None
        moe_cache_entry = None
        # Master weights/activations stay at the model's working dtype
        # (backend.default_dtype() -- float32 by default, float64 if
        # use_float64(True) was called); when use_mixed_precision is on,
        # only the matmul itself is forced down to float32 for a real
        # BLAS speedup on the hot path -- a no-op if the working dtype is
        # already float32.
        compute_dtype = np.float32 if getattr(self, "use_mixed_precision", False) else backend.default_dtype()
        if layer["type"] in ("dense", "sparse"):
            z = np.dot(x.astype(compute_dtype), layer["weights"].astype(compute_dtype).T) + \
                layer["bias"].astype(compute_dtype)
            z = z.astype(backend.default_dtype())
            x = activate(layer["activation"], z, **layer.get("activation_params", {}))
            self.pre_activations.append(z)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "conv2d":
            B, C, H, W = x.shape
            F, _, K, _ = layer["weights"].shape
            stride = layer.get("stride", 1)
            pad = layer.get("pad", 0)
            out_h, out_w = (H + 2 * pad - K) // stride + 1, (W + 2 * pad - K) // stride + 1
            col = im2col(x, K, K, stride=stride, pad=pad)
            conv_cache_entry = col
            weights_flat = layer["weights"].reshape(F, -1)
            out = np.dot(col.astype(compute_dtype), weights_flat.astype(compute_dtype).T) \
                .astype(backend.default_dtype()).reshape(B, out_h, out_w, F).transpose(0, 3, 1, 2)
            z = out + layer["bias"][None, :, None, None]
            x = activate(layer["activation"], z, **layer.get("activation_params", {}))
            self.pre_activations.append(z)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "conv1d":
            B, C, L = x.shape
            F, _, K = layer["weights"].shape
            stride = layer.get("stride", 1)
            pad = layer.get("pad", 0)
            out_l = (L + 2 * pad - K) // stride + 1
            col = im2col1d(x, K, stride=stride, pad=pad)
            conv_cache_entry = col
            weights_flat = layer["weights"].reshape(F, -1)
            out = np.dot(col.astype(compute_dtype), weights_flat.astype(compute_dtype).T) \
                .astype(backend.default_dtype()).reshape(B, out_l, F).transpose(0, 2, 1)
            z = out + layer["bias"][None, :, None]
            x = activate(layer["activation"], z, **layer.get("activation_params", {}))
            self.pre_activations.append(z)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "flatten":
            x = x.reshape(x.shape[0], -1)
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "maxpool2d":
            B, C, H, W, p = *x.shape, layer["p"]
            x = x[:, :, : H // p * p, : W // p * p].reshape(B, C, H // p, p, W // p, p).max(axis=(3, 5))
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "avgpool2d":
            B, C, H, W, p = *x.shape, layer["p"]
            x = x[:, :, : H // p * p, : W // p * p].reshape(B, C, H // p, p, W // p, p).mean(axis=(3, 5))
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "globalavgpool2d":
            x = np.mean(x, axis=(2, 3), keepdims=True)
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "upsample2d":
            scale = layer.get("scale_factor", 2)
            B, C, H, W = x.shape
            x = x.repeat(scale, axis=2).repeat(scale, axis=3)
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "batchnorm":
            normalized, cache = batchnorm_forward(x, layer, training)
            x = normalized
            self.pre_activations.append(None)
            self.batchnorm_cache.append(cache)
            self.layernorm_cache.append(None)
        elif layer["type"] == "layernorm":
            normalized, cache = layernorm_forward(x, layer, training)
            x = normalized
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(cache)
        elif layer["type"] == "dropout":
            rate = layer.get("rate", dropout_rate)
            if training and rate > 0:
                if rate >= 1.0:
                    mask = np.zeros_like(x, dtype=backend.default_dtype())
                    x = np.zeros_like(x)
                else:
                    mask = (np.random.rand(*x.shape) > rate).astype(backend.default_dtype())
                    x = x * mask / (1.0 - rate)
                layer["mask"] = mask
            else:
                layer["mask"] = None
                x = x
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "embedding":
            # x should be integer indices (batch, seq_len) or (batch,)
            x_int = np.asarray(x, dtype=np.int32)
            if x_int.ndim == 1:
                x_int = x_int.reshape(-1, 1)
            x = layer["weights"][x_int]
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "multihead_attention":
            x_in = x
            B, S, E = x_in.shape
            H = layer["num_heads"]
            Dh = layer["head_dim"]
            Q = np.dot(x_in, layer["Wq"].T) + layer["bq"]
            K = np.dot(x_in, layer["Wk"].T) + layer["bk"]
            V = np.dot(x_in, layer["Wv"].T) + layer["bv"]

            Hkv = layer.get("num_kv_heads", H)
            Qh = Q.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
            Kh = K.reshape(B, S, Hkv, Dh).transpose(0, 2, 1, 3)
            Vh = V.reshape(B, S, Hkv, Dh).transpose(0, 2, 1, 3)

            positional_scheme = layer.get("positional_scheme", "absolute")
            if positional_scheme == "rope":
                # Rotate Q/K (not V) per head before the score dot product.
                # Computed lazily from runtime S, like the causal mask below.
                # Rotation is per-position, identical for every head, so
                # doing it before the MQA/GQA expansion below is equivalent
                # to doing it after -- and cheaper.
                rope_cos, rope_sin = _rope_cos_sin(S, Dh, constants.SINUSOIDAL_BASE)
                Qh_scored = _rope_rotate(Qh, rope_cos, rope_sin)
                Kh_scored = _rope_rotate(Kh, rope_cos, rope_sin)
            else:
                Qh_scored, Kh_scored = Qh, Kh

            # MQA/GQA: expand the Hkv K/V heads to all H query heads. Plain
            # MHA (Hkv == H) is a no-op here and stays byte-identical.
            repeats = H // Hkv
            Kh_scored = _repeat_kv(Kh_scored, repeats)
            Vh = _repeat_kv(Vh, repeats)

            kernel = layer.get("attention_kernel", "softmax")
            if kernel != "softmax":
                # Linearized attention: no S x S score matrix ever exists,
                # so the mask/bias machinery below cannot apply (the layer
                # builder rejects the combinations that would need it).
                # `attn_state` takes attn's cache slot; backward.py picks the
                # matching branch off layer["attention_kernel"].
                omega = layer.get("omega")
                Qp = attention_kernels.feature_map(Qh_scored, kernel, omega, "row")
                Kp = attention_kernels.feature_map(Kh_scored, kernel, omega, "global")
                context_h, attn_state = attention_kernels.linear_attention_forward(
                    Qp, Kp, Vh, layer.get("causal", False))
                context = context_h.transpose(0, 2, 1, 3).reshape(B, S, E)
                x = np.dot(context, layer["Wo"].T) + layer["bo"]
                attn_cache_entry = (Q, K, V, Qh_scored, Kh_scored, Vh,
                                    attn_state, context, x_in)
                self.pre_activations.append(None)
                self.batchnorm_cache.append(None)
                self.layernorm_cache.append(None)
                self.attention_cache.append(attn_cache_entry)
                self.conv_cache.append(conv_cache_entry)
                self.rnn_cache.append(rnn_cache_entry)
                self.moe_cache.append(moe_cache_entry)
                self.outputs.append(x)
                continue

            sparse_spec = layer.get("sparse_pattern")
            if sparse_spec is not None:
                # Block-sparse: only the selected key blocks are gathered, so
                # the S x S score matrix is never built.
                slopes = _alibi_slopes(H) if positional_scheme == "alibi" else None
                context_h, attn_state = sparse_attention.sparse_attention_forward(
                    Qh_scored, Kh_scored, Vh, sparse_spec,
                    layer.get("causal", False), slopes)
                context = context_h.transpose(0, 2, 1, 3).reshape(B, S, E)
                x = np.dot(context, layer["Wo"].T) + layer["bo"]
                attn_cache_entry = (Q, K, V, Qh_scored, Kh_scored, Vh,
                                    attn_state, context, x_in)
                self.pre_activations.append(None)
                self.batchnorm_cache.append(None)
                self.layernorm_cache.append(None)
                self.attention_cache.append(attn_cache_entry)
                self.conv_cache.append(conv_cache_entry)
                self.rnn_cache.append(rnn_cache_entry)
                self.moe_cache.append(moe_cache_entry)
                self.outputs.append(x)
                continue

            tiled_block = layer.get("tiled_block_size")
            if tiled_block is not None:
                # Streaming softmax: same numbers, no S x S matrix.
                slopes = _alibi_slopes(H) if positional_scheme == "alibi" else None
                context_h, attn_state = flash_attention.flash_attention_forward(
                    Qh_scored, Kh_scored, Vh, tiled_block,
                    layer.get("causal", False), layer.get("window_size"), slopes)
                context = context_h.transpose(0, 2, 1, 3).reshape(B, S, E)
                x = np.dot(context, layer["Wo"].T) + layer["bo"]
                attn_cache_entry = (Q, K, V, Qh_scored, Kh_scored, Vh,
                                    attn_state, context, x_in)
                self.pre_activations.append(None)
                self.batchnorm_cache.append(None)
                self.layernorm_cache.append(None)
                self.attention_cache.append(attn_cache_entry)
                self.conv_cache.append(conv_cache_entry)
                self.rnn_cache.append(rnn_cache_entry)
                self.moe_cache.append(moe_cache_entry)
                self.outputs.append(x)
                continue

            scores = np.matmul(Qh_scored, Kh_scored.transpose(0, 1, 3, 2)) / np.sqrt(Dh)
            if positional_scheme == "alibi":
                # Static per-head linear-distance bias -- no extra params,
                # no backward-pass change needed (constant w.r.t. Q/K/V,
                # exactly like the causal mask below).
                slopes = _alibi_slopes(H)
                positions = np.arange(S)
                diff = positions[:, None] - positions[None, :]  # (S, S), i - j
                distance = diff if layer.get("causal", False) else np.abs(diff)
                alibi_bias = -slopes[:, None, None] * distance[None, :, :]
                scores = scores + alibi_bias[None, :, :, :]
            # Causal and/or sliding-window masking. No backward-pass change
            # is needed for either: masked entries get attn=0, so the
            # softmax-Jacobian gradient vanishes there too.
            positions = np.arange(S)
            mask = _window_mask(positions, positions, layer.get("causal", False),
                                layer.get("window_size"))
            if mask is not None:
                scores = scores + mask[None, None, :, :]
            scores_max = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
            attn_used, _ = _attn_dropout_forward(attn, layer, training)

            context = np.matmul(attn_used, Vh)
            context = context.transpose(0, 2, 1, 3).reshape(B, S, E)
            x = np.dot(context, layer["Wo"].T) + layer["bo"]

            attn_cache_entry = (Q, K, V, Qh_scored, Kh_scored, Vh, attn, context, x_in)
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "cross_attention":
            # Q from the normal sequential x; K/V from an earlier layer's
            # output (kv_source_index, direct convention like "goto"/
            # "concat_at", NOT residual's -1-adjusted save_index).
            x_in = x
            kv_source = self.outputs[layer["kv_source_index"] + 1]
            B, Sq, E = x_in.shape
            Skv = kv_source.shape[1]
            H = layer["num_heads"]
            Dh = layer["head_dim"]
            Q = np.dot(x_in, layer["Wq"].T) + layer["bq"]
            K = np.dot(kv_source, layer["Wk"].T) + layer["bk"]
            V = np.dot(kv_source, layer["Wv"].T) + layer["bv"]

            Hkv = layer.get("num_kv_heads", H)
            Qh = Q.reshape(B, Sq, H, Dh).transpose(0, 2, 1, 3)
            Kh = K.reshape(B, Skv, Hkv, Dh).transpose(0, 2, 1, 3)
            Vh = V.reshape(B, Skv, Hkv, Dh).transpose(0, 2, 1, 3)
            Kh = _repeat_kv(Kh, H // Hkv)          # MQA/GQA; no-op when Hkv == H
            Vh = _repeat_kv(Vh, H // Hkv)

            scores = np.matmul(Qh, Kh.transpose(0, 1, 3, 2)) / np.sqrt(Dh)
            scores_max = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
            attn_used, _ = _attn_dropout_forward(attn, layer, training)

            context = np.matmul(attn_used, Vh)
            context = context.transpose(0, 2, 1, 3).reshape(B, Sq, E)
            x = np.dot(context, layer["Wo"].T) + layer["bo"]

            attn_cache_entry = (Q, K, V, Qh, Kh, Vh, attn, context, x_in, kv_source)
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "moe":
            x, moe_cache_entry = moe.moe_forward(x, layer, training)
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "positional_encoding":
            S = x.shape[1]
            table = layer["weights"] if layer.get("_pos_type") == "learnable" else layer["pe"]
            x = x + table[:S][None, :, :]
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "rnn":
            B, S, _ = x.shape
            H = layer["hidden_dim"]
            h = _initial_rnn_state(layer, B, H)
            h_all = np.zeros((B, S + 1, H), dtype=backend.default_dtype())
            h_all[:, 0, :] = h  # so BPTT's t=0 step sees the true initial state
            for t in range(S):
                z = np.dot(x[:, t, :], layer["Wx"].T) + np.dot(h, layer["Wh"].T) + layer["b"]
                h = np.tanh(z)
                h_all[:, t + 1, :] = h
            if layer.get("stateful"):
                layer["_state_h"] = h
            rnn_cache_entry = {"x": x, "h_all": h_all}
            x = h_all[:, 1:, :] if layer["return_sequences"] else h_all[:, -1, :]
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "lstm":
            B, S, _ = x.shape
            H = layer["hidden_dim"]
            h = _initial_rnn_state(layer, B, H)
            c = _initial_rnn_state(layer, B, H, key="_state_c")
            h_all = np.zeros((B, S + 1, H), dtype=backend.default_dtype())
            c_all = np.zeros((B, S + 1, H), dtype=backend.default_dtype())
            h_all[:, 0, :] = h
            c_all[:, 0, :] = c
            i_all = np.zeros((B, S, H), dtype=backend.default_dtype())
            f_all = np.zeros((B, S, H), dtype=backend.default_dtype())
            g_all = np.zeros((B, S, H), dtype=backend.default_dtype())
            o_all = np.zeros((B, S, H), dtype=backend.default_dtype())
            tanh_c_all = np.zeros((B, S, H), dtype=backend.default_dtype())
            for t in range(S):
                gates = np.dot(x[:, t, :], layer["Wx"].T) + np.dot(h, layer["Wh"].T) + layer["b"]
                i_t = 1.0 / (1.0 + np.exp(-gates[:, 0 * H:1 * H]))
                f_t = 1.0 / (1.0 + np.exp(-gates[:, 1 * H:2 * H]))
                g_t = np.tanh(gates[:, 2 * H:3 * H])
                o_t = 1.0 / (1.0 + np.exp(-gates[:, 3 * H:4 * H]))
                c = f_t * c + i_t * g_t
                tanh_c = np.tanh(c)
                h = o_t * tanh_c
                i_all[:, t, :], f_all[:, t, :], g_all[:, t, :], o_all[:, t, :] = i_t, f_t, g_t, o_t
                tanh_c_all[:, t, :] = tanh_c
                h_all[:, t + 1, :] = h
                c_all[:, t + 1, :] = c
            if layer.get("stateful"):
                layer["_state_h"], layer["_state_c"] = h, c
            rnn_cache_entry = {"x": x, "h_all": h_all, "c_all": c_all, "i": i_all, "f": f_all,
                                "g": g_all, "o": o_all, "tanh_c": tanh_c_all}
            x = h_all[:, 1:, :] if layer["return_sequences"] else h_all[:, -1, :]
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "gru":
            B, S, _ = x.shape
            H = layer["hidden_dim"]
            h = _initial_rnn_state(layer, B, H)
            h_all = np.zeros((B, S + 1, H), dtype=backend.default_dtype())
            h_all[:, 0, :] = h
            r_all = np.zeros((B, S, H), dtype=backend.default_dtype())
            z_all = np.zeros((B, S, H), dtype=backend.default_dtype())
            n_all = np.zeros((B, S, H), dtype=backend.default_dtype())
            gh_n_all = np.zeros((B, S, H), dtype=backend.default_dtype())
            for t in range(S):
                gi = np.dot(x[:, t, :], layer["Wx"].T) + layer["bx"]
                gh = np.dot(h, layer["Wh"].T) + layer["bh"]
                r_t = 1.0 / (1.0 + np.exp(-(gi[:, 0 * H:1 * H] + gh[:, 0 * H:1 * H])))
                z_t = 1.0 / (1.0 + np.exp(-(gi[:, 1 * H:2 * H] + gh[:, 1 * H:2 * H])))
                gh_n = gh[:, 2 * H:3 * H]
                n_t = np.tanh(gi[:, 2 * H:3 * H] + r_t * gh_n)
                h = (1 - z_t) * n_t + z_t * h
                r_all[:, t, :], z_all[:, t, :], n_all[:, t, :] = r_t, z_t, n_t
                gh_n_all[:, t, :] = gh_n
                h_all[:, t + 1, :] = h
            if layer.get("stateful"):
                layer["_state_h"] = h
            rnn_cache_entry = {"x": x, "h_all": h_all, "r": r_all, "z": z_all, "n": n_all, "gh_n": gh_n_all}
            x = h_all[:, 1:, :] if layer["return_sequences"] else h_all[:, -1, :]
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "residual_save":
            # Identity passthrough; the saved value is simply self.outputs at
            # this layer's input index, referenced later by residual_add.
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "globalmaxpool2d":
            B, C, H, W = x.shape
            flat = x.reshape(B, C, H * W)
            idx = flat.argmax(axis=-1)
            conv_cache_entry = (x.shape, idx)
            x = np.take_along_axis(flat, idx[..., None], axis=-1).reshape(B, C, 1, 1)
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "channel_pool":
            # CBAM's spatial-attention input: (B,C,H,W) -> (B,2,H,W), the
            # per-position mean and max ACROSS channels.
            B, C, H, W = x.shape
            idx = x.argmax(axis=1)                       # (B, H, W)
            conv_cache_entry = (x.shape, idx)
            mean_c = np.mean(x, axis=1, keepdims=True)
            max_c = np.max(x, axis=1, keepdims=True)
            x = np.concatenate([mean_c, max_c], axis=1)
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "spp":
            # Spatial Pyramid Pooling: max-pool to each grid size in turn and
            # concatenate the flattened results, so any input resolution
            # yields one fixed-length vector.
            B, C, H, W = x.shape
            pieces, index_map = [], []
            for n in layer["pool_sizes"]:
                for hs, he in spp_bounds(H, n):
                    for ws, we in spp_bounds(W, n):
                        window = x[:, :, hs:he, ws:we]
                        wh, ww = he - hs, we - ws
                        flat = window.reshape(B, C, wh * ww)
                        pos = flat.argmax(axis=-1)
                        pieces.append(np.take_along_axis(flat, pos[..., None],
                                                         axis=-1)[..., 0])
                        index_map.append((hs, ws, wh, ww, pos))
            conv_cache_entry = (x.shape, index_map)
            x = np.stack(pieces, axis=-1).reshape(B, -1)
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "cbam_channel":
            # CBAM channel attention as ONE layer, because the paper's MLP is
            # SHARED between the average- and max-pooled paths and a layer
            # list has no way to tie two dense layers' weights together.
            B, C, H, W = x.shape
            flat = x.reshape(B, C, H * W)
            argmax = flat.argmax(axis=-1)
            avg = np.mean(flat, axis=-1)
            mx = np.take_along_axis(flat, argmax[..., None], axis=-1)[..., 0]
            pre_a = np.dot(avg, layer["W1"].T) + layer["b1"]
            pre_m = np.dot(mx, layer["W1"].T) + layer["b1"]
            act_name = layer["activation"]
            h_a, h_m = activate(act_name, pre_a), activate(act_name, pre_m)
            z = (np.dot(h_a, layer["W2"].T) + np.dot(h_m, layer["W2"].T)
                 + 2.0 * layer["b2"])
            gate = 1.0 / (1.0 + np.exp(-z))
            conv_cache_entry = (x, avg, mx, argmax, pre_a, pre_m, h_a, h_m, gate)
            x = x * gate[:, :, None, None]
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "residual_mul":
            # Multiplicative counterpart of residual_add: gate the saved
            # activations by whatever the branch since then produced.
            # Squeeze-and-Excitation, CBAM and EfficientNet are all this.
            saved = self.outputs[layer["save_index"]]
            x = saved * gate_broadcast(x, saved)
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "residual_add":
            x = x + self.outputs[layer["save_index"]]
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "goto":
            # Jumps to an earlier layer's output, REPLACING x entirely
            # (unlike residual_add, which adds). Indexing convention:
            # stored_index names "the layer whose output is referenced"
            # directly -- self.outputs[stored_index + 1] is the target, no
            # -1 adjustment (unlike residual's save_index, which points AT
            # the residual_save marker layer itself).
            x = self.outputs[layer["stored_index"] + 1]
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "concat_at":
            # Concatenates two earlier layers' outputs along the feature
            # axis, REPLACING x (same non-additive, index-direct convention
            # as "goto"). Used to join bidirectional RNN directions.
            x = np.concatenate(
                [self.outputs[layer["idx_a"] + 1], self.outputs[layer["idx_b"] + 1]], axis=-1
            )
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "reverse_sequence":
            # Reverses the sequence (time) axis -- a plain sequential
            # passthrough layer (not multi-source), self-inverse in
            # backward too.
            x = x[:, ::-1, :]
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        else:
            raise ValueError(f"Unknown layer type: {layer['type']}")
        act_quant = layer.get("act_quant")
        if act_quant is not None:
            # Post-training activation quantization (roadmap item 63): round
            # this layer's output onto the calibrated integer grid, which is
            # exactly what an integer kernel would carry forward.
            from ..compression.quantization import fake_quantize
            x = fake_quantize(x, act_quant["scale"], act_quant["zero_point"],
                              act_quant["bits"], act_quant["scheme"])
        self.attention_cache.append(attn_cache_entry)
        self.conv_cache.append(conv_cache_entry)
        self.rnn_cache.append(rnn_cache_entry)
        self.moe_cache.append(moe_cache_entry)
        self.outputs.append(x)
    return self.outputs[-1]
