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
            out_h, out_w = (H - K) // stride + 1, (W - K) // stride + 1
            col = im2col(x, K, K, stride=stride)
            conv_cache_entry = col
            weights_flat = layer["weights"].reshape(F, -1)
            out = np.dot(col.astype(compute_dtype), weights_flat.astype(compute_dtype).T) \
                .astype(np.float64).reshape(B, out_h, out_w, F).transpose(0, 3, 1, 2)
            z = out + layer["bias"][None, :, None, None]
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

            scores = np.matmul(Qh, Kh.transpose(0, 1, 3, 2)) / np.sqrt(Dh)
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

            attn_cache_entry = (Q, K, V, Qh, Kh, Vh, attn, context, x_in)
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
        else:
            raise ValueError(f"Unknown layer type: {layer['type']}")
        self.attention_cache.append(attn_cache_entry)
        self.conv_cache.append(conv_cache_entry)
        self.rnn_cache.append(rnn_cache_entry)
        self.outputs.append(x)
    return self.outputs[-1]
