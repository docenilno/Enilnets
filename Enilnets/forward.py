import numpy as np
from .activations import activate
from . import constants

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant')

    N_stride, C_stride, H_stride, W_stride = img.strides
    shape = (N, C, filter_h, filter_w, out_h, out_w)
    strides = (N_stride, C_stride, H_stride, W_stride, H_stride * stride, W_stride * stride)

    col = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    return col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

def im2col1d(input_data, filter_k, stride=1, pad=0):
    N, C, L = input_data.shape
    out_l = (L + 2 * pad - filter_k) // stride + 1
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad)], mode='constant')

    N_stride, C_stride, L_stride = img.strides
    shape = (N, C, filter_k, out_l)
    strides = (N_stride, C_stride, L_stride, L_stride * stride)

    col = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    return col.transpose(0, 3, 1, 2).reshape(N * out_l, -1)

def _rope_cos_sin(S, head_dim, base):
    """Standard RoPE angles: theta_i = base**(-2i/head_dim) for i in
    0..head_dim/2-1, applied at integer positions 0..S-1. Returns
    (cos, sin), each (S, head_dim/2) -- broadcasts directly against a
    (..., S, head_dim/2) array (numpy aligns trailing dims)."""
    half = head_dim // 2
    theta = base ** (-2.0 * np.arange(half) / head_dim)
    pos = np.arange(S, dtype=np.float64)
    angles = pos[:, None] * theta[None, :]
    return np.cos(angles), np.sin(angles)

def _rope_rotate(x, cos, sin):
    """Rotate the last axis of `x` (..., head_dim) by per-position-pair
    angles (the "rotate_half" formulation used by most modern RoPE
    implementations -- mathematically equivalent to the interleaved-pairs
    formulation up to a fixed permutation of head_dim, which cancels out
    since it's applied identically to Q and K before their dot product)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

def _alibi_slopes(num_heads):
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

def batchnorm_forward(x, layer, training):
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

def layernorm_forward(x, layer, training):
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
        # For 4D input (batch, C, H, W), broadcast as (1, C, 1, 1)
        gamma = layer["gamma"].reshape(1, -1, 1, 1)
        beta = layer["beta"].reshape(1, -1, 1, 1)
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

def Forward(self, inputs, training=False, dropout_rate=0.0):
    x = np.asarray(inputs, dtype=np.float64)
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
    self.conv_cache = []
    self.rnn_cache = []

    for layer in self.layers:
        x = self.outputs[-1]
        attn_cache_entry = None
        conv_cache_entry = None
        rnn_cache_entry = None
        compute_dtype = np.float32 if getattr(self, "use_mixed_precision", False) else np.float64
        if layer["type"] in ("dense", "sparse"):
            # Master weights stay float64 (updated by the optimizer at full
            # precision); only the matmul itself runs in float32 when mixed
            # precision is on, for a real BLAS speedup on the hot path.
            z = np.dot(x.astype(compute_dtype), layer["weights"].astype(compute_dtype).T) + \
                layer["bias"].astype(compute_dtype)
            z = z.astype(np.float64)
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
                .astype(np.float64).reshape(B, out_h, out_w, F).transpose(0, 3, 1, 2)
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
                .astype(np.float64).reshape(B, out_l, F).transpose(0, 2, 1)
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
                    mask = np.zeros_like(x, dtype=np.float64)
                    x = np.zeros_like(x)
                else:
                    mask = (np.random.rand(*x.shape) > rate).astype(np.float64)
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

            Qh = Q.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
            Kh = K.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)
            Vh = V.reshape(B, S, H, Dh).transpose(0, 2, 1, 3)

            positional_scheme = layer.get("positional_scheme", "absolute")
            if positional_scheme == "rope":
                # Rotate Q/K (not V) per head before the score dot product.
                # Computed lazily from runtime S, like the causal mask below.
                rope_cos, rope_sin = _rope_cos_sin(S, Dh, constants.SINUSOIDAL_BASE)
                Qh_scored = _rope_rotate(Qh, rope_cos, rope_sin)
                Kh_scored = _rope_rotate(Kh, rope_cos, rope_sin)
            else:
                Qh_scored, Kh_scored = Qh, Kh

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
            if layer.get("causal", False):
                # Position i may only attend to positions <= i (autoregressive
                # language modeling). No backward-pass changes are needed for
                # this: masked entries get attn=0, so the softmax-Jacobian
                # gradient naturally vanishes there too.
                causal_mask = np.where(np.arange(S)[None, :] > np.arange(S)[:, None], -np.inf, 0.0)
                scores = scores + causal_mask[None, None, :, :]
            scores_max = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

            context = np.matmul(attn, Vh)
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

            Qh = Q.reshape(B, Sq, H, Dh).transpose(0, 2, 1, 3)
            Kh = K.reshape(B, Skv, H, Dh).transpose(0, 2, 1, 3)
            Vh = V.reshape(B, Skv, H, Dh).transpose(0, 2, 1, 3)

            scores = np.matmul(Qh, Kh.transpose(0, 1, 3, 2)) / np.sqrt(Dh)
            scores_max = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

            context = np.matmul(attn, Vh)
            context = context.transpose(0, 2, 1, 3).reshape(B, Sq, E)
            x = np.dot(context, layer["Wo"].T) + layer["bo"]

            attn_cache_entry = (Q, K, V, Qh, Kh, Vh, attn, context, x_in, kv_source)
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "positional_encoding":
            S = x.shape[1]
            x = x + layer["pe"][:S][None, :, :]
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "rnn":
            B, S, _ = x.shape
            H = layer["hidden_dim"]
            h = np.zeros((B, H), dtype=np.float64)
            h_all = np.zeros((B, S + 1, H), dtype=np.float64)
            for t in range(S):
                z = np.dot(x[:, t, :], layer["Wx"].T) + np.dot(h, layer["Wh"].T) + layer["b"]
                h = np.tanh(z)
                h_all[:, t + 1, :] = h
            rnn_cache_entry = {"x": x, "h_all": h_all}
            x = h_all[:, 1:, :] if layer["return_sequences"] else h_all[:, -1, :]
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "lstm":
            B, S, _ = x.shape
            H = layer["hidden_dim"]
            h = np.zeros((B, H), dtype=np.float64)
            c = np.zeros((B, H), dtype=np.float64)
            h_all = np.zeros((B, S + 1, H), dtype=np.float64)
            c_all = np.zeros((B, S + 1, H), dtype=np.float64)
            i_all = np.zeros((B, S, H), dtype=np.float64)
            f_all = np.zeros((B, S, H), dtype=np.float64)
            g_all = np.zeros((B, S, H), dtype=np.float64)
            o_all = np.zeros((B, S, H), dtype=np.float64)
            tanh_c_all = np.zeros((B, S, H), dtype=np.float64)
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
            rnn_cache_entry = {"x": x, "h_all": h_all, "c_all": c_all, "i": i_all, "f": f_all,
                                "g": g_all, "o": o_all, "tanh_c": tanh_c_all}
            x = h_all[:, 1:, :] if layer["return_sequences"] else h_all[:, -1, :]
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
            self.layernorm_cache.append(None)
        elif layer["type"] == "gru":
            B, S, _ = x.shape
            H = layer["hidden_dim"]
            h = np.zeros((B, H), dtype=np.float64)
            h_all = np.zeros((B, S + 1, H), dtype=np.float64)
            r_all = np.zeros((B, S, H), dtype=np.float64)
            z_all = np.zeros((B, S, H), dtype=np.float64)
            n_all = np.zeros((B, S, H), dtype=np.float64)
            gh_n_all = np.zeros((B, S, H), dtype=np.float64)
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
        self.attention_cache.append(attn_cache_entry)
        self.conv_cache.append(conv_cache_entry)
        self.rnn_cache.append(rnn_cache_entry)
        self.outputs.append(x)
    return self.outputs[-1]
