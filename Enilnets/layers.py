import numpy as np
from .weight_init import init_weights, init_conv_weights, init_conv1d_weights, init_embedding_weights

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
    self._last_spatial_1d = None

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
    self._last_spatial_1d = None

def add_conv2d(self, in_ch=None, out_ch=32, k=3, activation="relu", init_method="he_normal",
               stride=1, activation_params=None, input_size=None, padding="valid"):
    """input_size: optional (H, W) -- only needed on the very first conv2d call
    if you want add_flatten() to later auto-infer its output width. Later
    conv2d/pool/upsample calls propagate spatial size automatically.

    padding: "valid" (default, no padding -- output shrinks with kernel
        size, exact pre-existing behavior) or "same" (output spatial size
        equals input spatial size). "same" only supports stride=1 and an
        odd kernel size k (raises ValueError otherwise -- avoids asymmetric
        padding edge cases)."""
    if in_ch is None:
        if self._last_spatial is not None:
            in_ch = self._last_spatial[0]
        else:
            raise ValueError(
                "Cannot auto-infer in_ch for add_conv2d: no previous conv/pool layer. "
                "Pass in_ch explicitly for the first conv2d layer in the model."
            )
    if padding == "valid":
        pad = 0
    elif padding == "same":
        if stride != 1 or k % 2 == 0:
            raise ValueError(
                f'padding="same" only supports stride=1 and an odd kernel size '
                f"(got stride={stride}, k={k})."
            )
        pad = (k - 1) // 2
    else:
        raise ValueError(f"Unknown padding: {padding!r}. Expected 'valid' or 'same'.")
    w, b = init_conv_weights(in_ch, out_ch, k, method=init_method)
    self.layers.append({"type": "conv2d", "weights": w, "bias": b, "in_ch": in_ch, "out_ch": out_ch,
                        "k": k, "activation": activation, "stride": stride, "padding": padding, "pad": pad,
                        "activation_params": activation_params or {}})
    self._last_width = out_ch
    self._last_spatial_1d = None
    if self._last_spatial is not None:
        H, W = self._last_spatial[1], self._last_spatial[2]
        self._last_spatial = (out_ch, (H + 2 * pad - k) // stride + 1, (W + 2 * pad - k) // stride + 1)
    elif input_size is not None:
        H, W = input_size
        self._last_spatial = (out_ch, (H + 2 * pad - k) // stride + 1, (W + 2 * pad - k) // stride + 1)
    else:
        self._last_spatial = None

def add_conv1d(self, in_ch=None, out_ch=32, k=3, activation="relu", init_method="he_normal",
               stride=1, activation_params=None, input_size=None, padding="valid"):
    """1D convolution for `(batch, channels, length)` data (audio, raw
    sequences, time series), mirroring `add_conv2d` throughout.

    input_size: optional int length L -- only needed on the very first
        conv1d call, to let a later add_flatten() auto-infer its output
        width. Later conv1d calls propagate length automatically via a
        separate `_last_spatial_1d` tracker (kept distinct from
        `add_conv2d`'s `_last_spatial` so a 1D `(C, L)` tuple is never
        misread as a 2D `(C, H, W)` one, or vice versa).

    padding: "valid" (default) or "same" (stride=1 + odd kernel size k
        only, same restriction and rationale as add_conv2d)."""
    if in_ch is None:
        if self._last_spatial_1d is not None:
            in_ch = self._last_spatial_1d[0]
        else:
            raise ValueError(
                "Cannot auto-infer in_ch for add_conv1d: no previous conv1d layer. "
                "Pass in_ch explicitly for the first conv1d layer in the model."
            )
    if padding == "valid":
        pad = 0
    elif padding == "same":
        if stride != 1 or k % 2 == 0:
            raise ValueError(
                f'padding="same" only supports stride=1 and an odd kernel size '
                f"(got stride={stride}, k={k})."
            )
        pad = (k - 1) // 2
    else:
        raise ValueError(f"Unknown padding: {padding!r}. Expected 'valid' or 'same'.")
    w, b = init_conv1d_weights(in_ch, out_ch, k, method=init_method)
    self.layers.append({"type": "conv1d", "weights": w, "bias": b, "in_ch": in_ch, "out_ch": out_ch,
                        "k": k, "activation": activation, "stride": stride, "padding": padding, "pad": pad,
                        "activation_params": activation_params or {}})
    self._last_width = out_ch
    self._last_spatial = None
    if self._last_spatial_1d is not None:
        L = self._last_spatial_1d[1]
        self._last_spatial_1d = (out_ch, (L + 2 * pad - k) // stride + 1)
    elif input_size is not None:
        self._last_spatial_1d = (out_ch, (input_size + 2 * pad - k) // stride + 1)
    else:
        self._last_spatial_1d = None

def add_flatten(self):
    if self._last_spatial_1d is not None:
        C, L = self._last_spatial_1d
        self._last_width = C * L
    elif self._last_spatial is not None:
        C, H, W = self._last_spatial
        self._last_width = C * H * W
    self._last_spatial = None
    self._last_spatial_1d = None
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
    self._last_spatial_1d = None

def add_multihead_attention(self, embed_dim=None, num_heads=4, dropout=0.0, init_method="xavier_uniform",
                             causal=False, positional_scheme="absolute"):
    """Add a multi-head self-attention layer.

    Input/output shape: (batch, seq_len, embed_dim).
    Stored as a single layer holding its own Q/K/V/output projection weights
    (Wq/bq, Wk/bk, Wv/bv, Wo/bo), so the shared input is used for all three
    projections rather than chaining separate dense layers.

    causal: if True, position i can only attend to positions <= i (for
        autoregressive / language-model style decoders).
    positional_scheme: "absolute" (default -- no positional info is added
        here; pair with add_positional_encoding() as usual), "rope"
        (rotary position embedding -- rotates Q/K per head before the score
        dot product; requires an even head_dim), or "alibi" (a static
        per-head linear-distance bias added to the scores, no extra
        parameters or positional_encoding layer needed).
    """
    if embed_dim is None:
        embed_dim = _require_width(self, "add_multihead_attention")
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
    head_dim = embed_dim // num_heads
    if positional_scheme not in ("absolute", "rope", "alibi"):
        raise ValueError(f"Unknown positional_scheme: {positional_scheme!r}. Expected 'absolute', 'rope', or 'alibi'.")
    if positional_scheme == "rope" and head_dim % 2 != 0:
        raise ValueError(f"positional_scheme='rope' requires an even head_dim, got {head_dim}")
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
        "positional_scheme": positional_scheme,
        "Wq": Wq, "bq": bq,
        "Wk": Wk, "bk": bk,
        "Wv": Wv, "bv": bv,
        "Wo": Wo, "bo": bo,
    })
    self._last_width = embed_dim
    self._last_spatial = None
    self._last_spatial_1d = None

def add_cross_attention(self, kv_source_index, embed_dim=None, num_heads=4, dropout=0.0,
                         init_method="xavier_uniform"):
    """Add a cross-attention layer (encoder-decoder style): queries come
    from the normal sequential `x`; keys/values come from an earlier
    layer's output.

    kv_source_index: directly names "the layer whose output is the KV
        source" -- self.outputs[kv_source_index + 1] is the target
        directly, NO -1 adjustment (same direct-indexing convention as
        "goto"/"concat_at", NOT residual's save_index convention, which
        points AT a marker layer and needs -1).

    Input/output shape: (batch, seq_len_q, embed_dim) for the sequential
    `x`; the KV source must be (batch, seq_len_kv, embed_dim) with the
    SAME embed_dim (e.g. an encoder stack's final layer). Stores its own
    Wq/bq/Wk/bk/Wv/bv/Wo/bo projection weights directly on the layer, same
    as add_multihead_attention. No causal masking (cross-attention
    conventionally attends freely over the full KV source) and no
    positional_scheme (RoPE/ALiBi are self-attention concepts here).
    """
    if embed_dim is None:
        embed_dim = _require_width(self, "add_cross_attention")
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
    head_dim = embed_dim // num_heads
    Wq, bq = init_weights(embed_dim, embed_dim, method=init_method)
    Wk, bk = init_weights(embed_dim, embed_dim, method=init_method)
    Wv, bv = init_weights(embed_dim, embed_dim, method=init_method)
    Wo, bo = init_weights(embed_dim, embed_dim, method=init_method)
    self.layers.append({
        "type": "cross_attention",
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "dropout": dropout,
        "kv_source_index": kv_source_index,
        "Wq": Wq, "bq": bq,
        "Wk": Wk, "bk": bk,
        "Wv": Wv, "bv": bv,
        "Wo": Wo, "bo": bo,
    })
    self._last_width = embed_dim
    self._last_spatial = None
    self._last_spatial_1d = None

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
                    in_ch=None, stride=1, batchnorm=False, pool=None, input_size=None,
                    padding="valid"):
    """Convenience: one conv2d (with its activation), optionally followed by
    batchnorm and/or pooling: conv2d(activation) -> [batchnorm] -> [pool].

    pool: None, "max", or "avg" (pool_size=2).
    """
    self.add_conv2d(in_ch, out_ch, k, activation=activation,
                    init_method=init_method, stride=stride, input_size=input_size,
                    padding=padding)
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
    self._last_spatial_1d = None

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
    self._last_spatial_1d = None

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
    self._last_spatial_1d = None

def _add_bidirectional(self, add_fn, n_in, hidden_dim, return_sequences, init_method):
    """Shared composition for add_bidirectional_rnn/_lstm/_gru: run
    `add_fn` (one of self.add_rnn/add_lstm/add_gru) once forward and once
    on the time-reversed input (via the "goto"/"reverse_sequence"
    primitives), then concatenate both directions' outputs along the
    feature axis via "concat_at".

    n_in must be resolved to a concrete int (not left None) before the
    SECOND add_fn call: after the forward direction runs, self._last_width
    is hidden_dim (the forward RNN's own output width), not the original
    input width -- the "goto" jump back to source_index doesn't update
    _last_width bookkeeping (only add_* layer-builder calls do), so
    auto-inference would silently use the wrong width for the backward
    direction if n_in weren't already a concrete number by then.
    """
    if n_in is None:
        n_in = _require_width(self, add_fn.__name__)
    source_index = len(self.layers) - 1
    add_fn(n_in, hidden_dim, return_sequences=return_sequences, init_method=init_method)  # forward direction
    fwd_index = len(self.layers) - 1
    self.layers.append({"type": "goto", "stored_index": source_index})
    self.layers.append({"type": "reverse_sequence"})
    add_fn(n_in, hidden_dim, return_sequences=return_sequences, init_method=init_method)  # backward direction (reversed input)
    if return_sequences:
        # Un-reverse before concatenating so timestep t aligns correctly --
        # the backward direction's raw output is in reversed-time order.
        # When return_sequences=False, each direction's output is already
        # a single collapsed (batch, hidden) summary (no time axis left to
        # un-reverse): the forward direction's summary is of the sequence
        # read left-to-right, the backward direction's is of the sequence
        # read right-to-left, and concatenating them directly is exactly
        # the standard bidirectional-summary construction.
        self.layers.append({"type": "reverse_sequence"})
    bwd_index = len(self.layers) - 1
    self.layers.append({"type": "concat_at", "idx_a": fwd_index, "idx_b": bwd_index})
    self._last_width = hidden_dim * 2
    self._last_spatial = None
    self._last_spatial_1d = None

def add_bidirectional_rnn(self, n_in=None, hidden_dim=128, return_sequences=True, init_method="xavier_uniform"):
    """Bidirectional vanilla RNN: runs add_rnn once forward and once over
    the time-reversed input, concatenating both directions' hidden states
    along the feature axis -- output width is hidden_dim * 2.
    return_sequences=True: (batch, seq_len, hidden_dim*2).
    return_sequences=False: (batch, hidden_dim*2) (forward's final state
    concatenated with backward's final state -- see _add_bidirectional's
    docstring for why no un-reversing is needed in that case)."""
    _add_bidirectional(self, self.add_rnn, n_in, hidden_dim, return_sequences, init_method)

def add_bidirectional_lstm(self, n_in=None, hidden_dim=128, return_sequences=True, init_method="xavier_uniform"):
    """Bidirectional LSTM -- see add_bidirectional_rnn for the composition
    and shape convention (output width hidden_dim * 2)."""
    _add_bidirectional(self, self.add_lstm, n_in, hidden_dim, return_sequences, init_method)

def add_bidirectional_gru(self, n_in=None, hidden_dim=128, return_sequences=True, init_method="xavier_uniform"):
    """Bidirectional GRU -- see add_bidirectional_rnn for the composition
    and shape convention (output width hidden_dim * 2)."""
    _add_bidirectional(self, self.add_gru, n_in, hidden_dim, return_sequences, init_method)

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
