import numpy as np
from .weight_init import init_weights, init_conv_weights, init_embedding_weights

def _require_width(self, what):
    if self._last_width is None:
        raise ValueError(
            f"Cannot auto-infer input size for {what}: no previous layer to infer it from. "
            f"Pass the size explicitly for the first layer in the model."
        )
    return self._last_width

def add_dense(self, n_in=None, n_out=128, activation="relu", init_method="xavier_uniform",
              use_bias=True, activation_params=None):
    if n_in is None:
        n_in = _require_width(self, "add_dense")
    w, b = init_weights(n_in, n_out, method=init_method)
    if not use_bias:
        b = np.zeros(n_out, dtype=np.float64)
    self.layers.append({"type": "dense", "weights": w, "bias": b, "activation": activation,
                        "use_bias": use_bias, "activation_params": activation_params or {}})
    self._last_width = n_out
    self._last_spatial = None

def add_sparse(self, n_in=None, n_out=128, connectivity=0.5, activation="relu",
               init_method="xavier_uniform", activation_params=None):
    if n_in is None:
        n_in = _require_width(self, "add_sparse")
    w, b = init_weights(n_in, n_out, method=init_method)
    mask = (np.random.rand(n_out, n_in) < connectivity).astype(np.float64)
    self.layers.append({"type": "sparse", "weights": w * mask, "bias": b, "mask": mask,
                        "activation": activation, "activation_params": activation_params or {}})
    self._last_width = n_out
    self._last_spatial = None

def add_conv2d(self, in_ch=None, out_ch=32, k=3, activation="relu", init_method="he_normal",
               stride=1, activation_params=None, input_size=None):
    """input_size: optional (H, W) -- only needed on the very first conv2d call
    if you want add_flatten() to later auto-infer its output width. Later
    conv2d/pool/upsample calls propagate spatial size automatically."""
    if in_ch is None:
        if self._last_spatial is not None:
            in_ch = self._last_spatial[0]
        else:
            raise ValueError(
                "Cannot auto-infer in_ch for add_conv2d: no previous conv/pool layer. "
                "Pass in_ch explicitly for the first conv2d layer in the model."
            )
    w, b = init_conv_weights(in_ch, out_ch, k, method=init_method)
    self.layers.append({"type": "conv2d", "weights": w, "bias": b, "in_ch": in_ch, "out_ch": out_ch,
                        "k": k, "activation": activation, "stride": stride,
                        "activation_params": activation_params or {}})
    self._last_width = out_ch
    if self._last_spatial is not None:
        H, W = self._last_spatial[1], self._last_spatial[2]
        self._last_spatial = (out_ch, (H - k) // stride + 1, (W - k) // stride + 1)
    elif input_size is not None:
        H, W = input_size
        self._last_spatial = (out_ch, (H - k) // stride + 1, (W - k) // stride + 1)
    else:
        self._last_spatial = None

def add_flatten(self):
    if self._last_spatial is not None:
        C, H, W = self._last_spatial
        self._last_width = C * H * W
    self._last_spatial = None
    self.layers.append({"type": "flatten"})

def add_maxpool2d(self, pool_size=2):
    self.layers.append({"type": "maxpool2d", "p": pool_size})
    if self._last_spatial is not None:
        C, H, W = self._last_spatial
        self._last_spatial = (C, H // pool_size, W // pool_size)

def add_avgpool2d(self, pool_size=2):
    self.layers.append({"type": "avgpool2d", "p": pool_size})
    if self._last_spatial is not None:
        C, H, W = self._last_spatial
        self._last_spatial = (C, H // pool_size, W // pool_size)

def add_global_avgpool2d(self):
    self.layers.append({"type": "globalavgpool2d"})
    if self._last_spatial is not None:
        C = self._last_spatial[0]
        self._last_spatial = (C, 1, 1)

def add_upsample2d(self, scale_factor=2):
    self.layers.append({"type": "upsample2d", "scale_factor": scale_factor})
    if self._last_spatial is not None:
        C, H, W = self._last_spatial
        self._last_spatial = (C, H * scale_factor, W * scale_factor)

def add_batchnorm(self, num_features=None, epsilon=1e-5, momentum=0.1):
    if num_features is None:
        num_features = _require_width(self, "add_batchnorm")
    self.layers.append({"type": "batchnorm", "num_features": num_features, "epsilon": epsilon, "momentum": momentum,
                        "running_mean": np.zeros(num_features, dtype=np.float64),
                        "running_var": np.ones(num_features, dtype=np.float64),
                        "gamma": np.ones(num_features, dtype=np.float64),
                        "beta": np.zeros(num_features, dtype=np.float64)})

def add_layernorm(self, normalized_shape=None, epsilon=1e-5):
    """Add Layer Normalization layer.
    normalized_shape: int or tuple (e.g., 256 or (C,H,W))
    """
    if normalized_shape is None:
        normalized_shape = _require_width(self, "add_layernorm")
    if isinstance(normalized_shape, int):
        num_features = normalized_shape
    else:
        num_features = int(np.prod(normalized_shape))
    self.layers.append({"type": "layernorm", "normalized_shape": normalized_shape,
                        "epsilon": epsilon,
                        "gamma": np.ones(num_features, dtype=np.float64),
                        "beta": np.zeros(num_features, dtype=np.float64)})

def add_dropout(self, rate=0.5):
    self.layers.append({"type": "dropout", "rate": rate})

def add_embedding(self, vocab_size, embed_dim, init_method="normal"):
    """Add embedding layer for sequence/token data.
    vocab_size: number of unique tokens
    embed_dim: dimension of embedding vectors
    """
    w = init_embedding_weights(vocab_size, embed_dim, method=init_method)
    self.layers.append({"type": "embedding", "weights": w, "vocab_size": vocab_size, "embed_dim": embed_dim})
    self._last_width = embed_dim
    self._last_spatial = None

def add_multihead_attention(self, embed_dim=None, num_heads=4, dropout=0.0, init_method="xavier_uniform",
                             causal=False):
    """Add a multi-head self-attention layer.

    Input/output shape: (batch, seq_len, embed_dim).
    Stored as a single layer holding its own Q/K/V/output projection weights
    (Wq/bq, Wk/bk, Wv/bv, Wo/bo), so the shared input is used for all three
    projections rather than chaining separate dense layers.

    causal: if True, position i can only attend to positions <= i (for
        autoregressive / language-model style decoders).
    """
    if embed_dim is None:
        embed_dim = _require_width(self, "add_multihead_attention")
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
    head_dim = embed_dim // num_heads
    Wq, bq = init_weights(embed_dim, embed_dim, method=init_method)
    Wk, bk = init_weights(embed_dim, embed_dim, method=init_method)
    Wv, bv = init_weights(embed_dim, embed_dim, method=init_method)
    Wo, bo = init_weights(embed_dim, embed_dim, method=init_method)
    self.layers.append({
        "type": "multihead_attention",
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "dropout": dropout,
        "causal": causal,
        "Wq": Wq, "bq": bq,
        "Wk": Wk, "bk": bk,
        "Wv": Wv, "bv": bv,
        "Wo": Wo, "bo": bo,
    })
    self._last_width = embed_dim
    self._last_spatial = None

def add_mlp_block(self, hidden_dims, in_dim=None, out_dim=None, activation="relu",
                   out_activation="linear", init_method="xavier_uniform"):
    """Convenience: add a stack of dense layers in one call.

    add_mlp_block([64, 32], out_dim=10) is equivalent to:
        add_dense(in_dim, 64, activation)
        add_dense(64, 32, activation)
        add_dense(32, 10, out_activation)
    `in_dim` is only needed if this is the very first layer in the model.
    """
    n_in = in_dim
    for h in hidden_dims:
        self.add_dense(n_in, h, activation=activation, init_method=init_method)
        n_in = None  # subsequent layers auto-infer from the one just added
    if out_dim is not None:
        self.add_dense(n_in, out_dim, activation=out_activation, init_method=init_method)

def add_conv_block(self, out_ch, k=3, activation="relu", init_method="he_normal",
                    in_ch=None, stride=1, batchnorm=False, pool=None, input_size=None):
    """Convenience: one conv2d (with its activation), optionally followed by
    batchnorm and/or pooling: conv2d(activation) -> [batchnorm] -> [pool].

    pool: None, "max", or "avg" (pool_size=2).
    """
    self.add_conv2d(in_ch, out_ch, k, activation=activation,
                    init_method=init_method, stride=stride, input_size=input_size)
    if batchnorm:
        self.add_batchnorm(out_ch)
    if pool == "max":
        self.add_maxpool2d(2)
    elif pool == "avg":
        self.add_avgpool2d(2)

def add_rnn(self, n_in=None, hidden_dim=128, return_sequences=True, init_method="xavier_uniform"):
    """Add a vanilla (tanh) recurrent layer. Input/output: (batch, seq_len,
    features); if return_sequences=False, output is just the last timestep's
    hidden state, (batch, hidden_dim)."""
    if n_in is None:
        n_in = _require_width(self, "add_rnn")
    Wx, b = init_weights(n_in, hidden_dim, method=init_method)
    Wh, _ = init_weights(hidden_dim, hidden_dim, method=init_method)
    self.layers.append({"type": "rnn", "Wx": Wx, "Wh": Wh, "b": b,
                        "n_in": n_in, "hidden_dim": hidden_dim,
                        "return_sequences": return_sequences})
    self._last_width = hidden_dim
    self._last_spatial = None

def add_lstm(self, n_in=None, hidden_dim=128, return_sequences=True, init_method="xavier_uniform"):
    """Add an LSTM layer. Gate weights are stacked as a single (4*hidden, ·)
    matrix in [i, f, g, o] order. Input/output: (batch, seq_len, features);
    if return_sequences=False, output is just the last hidden state."""
    if n_in is None:
        n_in = _require_width(self, "add_lstm")
    Wx, b = init_weights(n_in, 4 * hidden_dim, method=init_method)
    Wh, _ = init_weights(hidden_dim, 4 * hidden_dim, method=init_method)
    # Forget-gate bias initialized to 1 (standard trick: start with the cell
    # remembering by default, avoids vanishing gradients early in training).
    b[hidden_dim:2 * hidden_dim] = 1.0
    self.layers.append({"type": "lstm", "Wx": Wx, "Wh": Wh, "b": b,
                        "n_in": n_in, "hidden_dim": hidden_dim,
                        "return_sequences": return_sequences})
    self._last_width = hidden_dim
    self._last_spatial = None

def add_gru(self, n_in=None, hidden_dim=128, return_sequences=True, init_method="xavier_uniform"):
    """Add a GRU layer. Gate weights are stacked as a single (3*hidden, ·)
    matrix in [r, z, n] order, with separate input/hidden biases (bx/bh) since
    the reset gate only multiplies the hidden contribution to the candidate
    gate, not the input contribution. Input/output: (batch, seq_len,
    features); if return_sequences=False, output is just the last hidden
    state."""
    if n_in is None:
        n_in = _require_width(self, "add_gru")
    Wx, bx = init_weights(n_in, 3 * hidden_dim, method=init_method)
    Wh, bh = init_weights(hidden_dim, 3 * hidden_dim, method=init_method)
    self.layers.append({"type": "gru", "Wx": Wx, "Wh": Wh, "bx": bx, "bh": bh,
                        "n_in": n_in, "hidden_dim": hidden_dim,
                        "return_sequences": return_sequences})
    self._last_width = hidden_dim
    self._last_spatial = None

def add_residual_start(self):
    """Mark the start of a residual/skip-connection block. Pair with a later
    add_residual_end() to compute `x = x + saved_x` (e.g. around attention or
    an MLP sub-block, as in a standard Transformer block). Nestable."""
    self._residual_stack.append(len(self.layers))
    self.layers.append({"type": "residual_save"})

def add_residual_end(self):
    """Add the input saved at the matching add_residual_start() back to the
    current activations: x = x + saved_x."""
    if not self._residual_stack:
        raise ValueError("add_residual_end() called without a matching add_residual_start()")
    save_index = self._residual_stack.pop()
    self.layers.append({"type": "residual_add", "save_index": save_index})
