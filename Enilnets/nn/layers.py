"""NeuralNet.add_*: the layer-builder methods that append a new layer dict
to self.layers, inferring shapes from the previous layer where possible."""
import math
from typing import Any, Dict, List, Optional

from ..core.backend import np
from ..core import backend
from .weight_init import init_weights, init_conv_weights, init_conv1d_weights, init_embedding_weights

def _require_width(self: Any, what: str) -> int:
    if self._last_width is None:
        raise ValueError(
            f"Cannot auto-infer input size for {what}: no previous layer to infer it from. "
            f"Pass the size explicitly for the first layer in the model."
        )
    return self._last_width

def add_dense(self: Any, n_in: Optional[int] = None, n_out: int = 128, activation: str = "relu",
              init_method: str = "xavier_uniform", use_bias: bool = True,
              activation_params: Optional[Dict[str, Any]] = None) -> None:
    if n_in is None:
        n_in = _require_width(self, "add_dense")
    w, b = init_weights(n_in, n_out, method=init_method)
    if not use_bias:
        b = np.zeros(n_out, dtype=backend.default_dtype())
    self.layers.append({"type": "dense", "weights": w, "bias": b, "activation": activation,
                        "use_bias": use_bias, "activation_params": activation_params or {}})
    self._last_width = n_out
    self._last_spatial = None
    self._last_spatial_1d = None

def add_sparse(self: Any, n_in: Optional[int] = None, n_out: int = 128, connectivity: float = 0.5,
               activation: str = "relu", init_method: str = "xavier_uniform",
               activation_params: Optional[Dict[str, Any]] = None) -> None:
    if n_in is None:
        n_in = _require_width(self, "add_sparse")
    w, b = init_weights(n_in, n_out, method=init_method)
    mask = (np.random.rand(n_out, n_in) < connectivity).astype(backend.default_dtype())
    self.layers.append({"type": "sparse", "weights": w * mask, "bias": b, "mask": mask,
                        "activation": activation, "activation_params": activation_params or {}})
    self._last_width = n_out
    self._last_spatial = None
    self._last_spatial_1d = None

def add_conv2d(self: Any, in_ch: Optional[int] = None, out_ch: int = 32, k: int = 3,
               activation: str = "relu", init_method: str = "he_normal",
               stride: int = 1, activation_params: Optional[Dict[str, Any]] = None,
               input_size: Optional[tuple] = None, padding: str = "valid") -> None:
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

def add_conv1d(self: Any, in_ch: Optional[int] = None, out_ch: int = 32, k: int = 3,
               activation: str = "relu", init_method: str = "he_normal",
               stride: int = 1, activation_params: Optional[Dict[str, Any]] = None,
               input_size: Optional[int] = None, padding: str = "valid") -> None:
    """1D convolution over `(batch, channels, length)` data, mirroring
    add_conv2d throughout.

    input_size: length L, needed only on the first conv1d call so a later
        add_flatten() can infer its width; later calls propagate it.
    padding: "valid" (default) or "same" (stride=1 and odd k only, same
        restriction as add_conv2d)."""
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

def add_flatten(self: Any) -> None:
    if self._last_spatial_1d is not None:
        C, L = self._last_spatial_1d
        self._last_width = C * L
    elif self._last_spatial is not None:
        C, H, W = self._last_spatial
        self._last_width = C * H * W
    self._last_spatial = None
    self._last_spatial_1d = None
    self.layers.append({"type": "flatten"})

def add_maxpool2d(self: Any, pool_size: int = 2) -> None:
    self.layers.append({"type": "maxpool2d", "p": pool_size})
    if self._last_spatial is not None:
        C, H, W = self._last_spatial
        self._last_spatial = (C, H // pool_size, W // pool_size)

def add_avgpool2d(self: Any, pool_size: int = 2) -> None:
    self.layers.append({"type": "avgpool2d", "p": pool_size})
    if self._last_spatial is not None:
        C, H, W = self._last_spatial
        self._last_spatial = (C, H // pool_size, W // pool_size)

def add_global_avgpool2d(self: Any) -> None:
    self.layers.append({"type": "globalavgpool2d"})
    if self._last_spatial is not None:
        C = self._last_spatial[0]
        self._last_spatial = (C, 1, 1)

def add_upsample2d(self: Any, scale_factor: int = 2) -> None:
    self.layers.append({"type": "upsample2d", "scale_factor": scale_factor})
    if self._last_spatial is not None:
        C, H, W = self._last_spatial
        self._last_spatial = (C, H * scale_factor, W * scale_factor)

def add_batchnorm(self: Any, num_features: Optional[int] = None, epsilon: float = 1e-5,
                   momentum: float = 0.1) -> None:
    if num_features is None:
        num_features = _require_width(self, "add_batchnorm")
    self.layers.append({"type": "batchnorm", "num_features": num_features, "epsilon": epsilon, "momentum": momentum,
                        "running_mean": np.zeros(num_features, dtype=backend.default_dtype()),
                        "running_var": np.ones(num_features, dtype=backend.default_dtype()),
                        "gamma": np.ones(num_features, dtype=backend.default_dtype()),
                        "beta": np.zeros(num_features, dtype=backend.default_dtype())})

def add_layernorm(self: Any, normalized_shape: Optional[Any] = None, epsilon: float = 1e-5) -> None:
    """Add Layer Normalization layer.
    normalized_shape: int or tuple (e.g., 256 or (C,H,W))
    """
    if normalized_shape is None:
        normalized_shape = _require_width(self, "add_layernorm")
    if isinstance(normalized_shape, int):
        num_features = normalized_shape
    else:
        num_features = math.prod(normalized_shape)
    self.layers.append({"type": "layernorm", "normalized_shape": normalized_shape,
                        "epsilon": epsilon,
                        "gamma": np.ones(num_features, dtype=backend.default_dtype()),
                        "beta": np.zeros(num_features, dtype=backend.default_dtype())})

def add_dropout(self: Any, rate: float = 0.5) -> None:
    self.layers.append({"type": "dropout", "rate": rate})

def add_embedding(self: Any, vocab_size: int, embed_dim: int, init_method: str = "normal") -> None:
    """Add embedding layer for sequence/token data.
    vocab_size: number of unique tokens
    embed_dim: dimension of embedding vectors
    """
    w = init_embedding_weights(vocab_size, embed_dim, method=init_method)
    self.layers.append({"type": "embedding", "weights": w, "vocab_size": vocab_size, "embed_dim": embed_dim})
    self._last_width = embed_dim
    self._last_spatial = None
    self._last_spatial_1d = None

def _validate_num_kv_heads(num_heads: int, num_kv_heads: Optional[int], where: str) -> int:
    """Resolve/validate `num_kv_heads` (MQA/GQA). None -> num_heads."""
    if num_kv_heads is None:
        return num_heads
    if num_kv_heads < 1 or num_heads % num_kv_heads != 0:
        raise ValueError(
            f"{where}: num_kv_heads={num_kv_heads} must be >= 1 and divide "
            f"num_heads={num_heads} evenly (each K/V head is shared by exactly "
            f"num_heads // num_kv_heads query heads)."
        )
    return num_kv_heads


def _validate_attention_kernel(kernel: str, positional_scheme: str,
                               window_size: Optional[int], dropout: float,
                               head_dim: int, num_features: Optional[int]) -> Optional[Any]:
    """Validate an attention_kernel choice and, for "performer", draw its
    fixed random-feature matrix. Returns the omega matrix or None.

    The linearized kernels never materialize an S x S score matrix, so
    anything defined as an additive bias on the scores -- ALiBi, a sliding
    window -- or as a mask over attention weights -- attention dropout --
    has nowhere to be applied. Those are rejected rather than silently
    ignored."""
    from .attention_kernels import KERNELS, make_projection
    if kernel not in KERNELS:
        raise ValueError(
            f"Unknown attention_kernel {kernel!r}. Expected one of {KERNELS}.")
    if kernel == "softmax":
        return None
    for name, value, why in (
        ("positional_scheme='alibi'", positional_scheme == "alibi",
         "ALiBi is an additive bias on the score matrix"),
        ("window_size", window_size is not None,
         "a sliding window is a mask on the score matrix"),
        ("dropout", dropout > 0,
         "attention dropout is a mask on the softmax weights"),
    ):
        if value:
            raise ValueError(
                f"attention_kernel={kernel!r} is incompatible with {name}: {why}, "
                "and linearized attention never materializes one. Use "
                "attention_kernel='softmax' if you need it.")
    return make_projection(num_features or head_dim * 4, head_dim)


def add_multihead_attention(self: Any, embed_dim: Optional[int] = None, num_heads: int = 4,
                             dropout: float = 0.0, init_method: str = "xavier_uniform",
                             causal: bool = False, positional_scheme: str = "absolute",
                             num_kv_heads: Optional[int] = None,
                             window_size: Optional[int] = None,
                             attention_kernel: str = "softmax",
                             num_features: Optional[int] = None,
                             sparse_pattern: Optional[Dict[str, int]] = None,
                             tiled_block_size: Optional[int] = None) -> None:
    """Add a multi-head self-attention layer. Input/output shape
    (batch, seq_len, embed_dim); holds its own Wq/bq..Wo/bo projections.
    See the README's "Multi-head attention" section for what each variant
    is for and how they trade off.

    causal: position i may attend only to positions <= i.
    positional_scheme: "absolute" (nothing added here -- pair with
        add_positional_encoding), "rope" (needs an even head_dim), "alibi".
    num_kv_heads: K/V heads shared across the query heads. None = num_heads
        (MHA), 1 = MQA, any divisor of num_heads = GQA.
    window_size: attend only where |i - j| <= window_size. None = unbounded.
    attention_kernel: "softmax" (exact, O(S^2)), "linear" (elu+1) or
        "performer" (FAVOR+). The linearized kernels build no score matrix,
        so alibi / window_size / attention dropout are rejected with them.
    num_features: Performer feature count; defaults to head_dim * 4.
    sparse_pattern: block-sparse attention, e.g. {"block_size": 16,
        "local": 1, "global": 1, "random": 2, "seed": 0}. Selected key
        blocks are gathered, so no S x S matrix is built. Excludes
        window_size and the linearized kernels.
    tiled_block_size: streaming ("Flash") softmax -- same numbers as the
        default path, O(S * block) memory instead of O(S^2). None = off."""
    if embed_dim is None:
        embed_dim = _require_width(self, "add_multihead_attention")
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
    head_dim = embed_dim // num_heads
    num_kv_heads = _validate_num_kv_heads(num_heads, num_kv_heads, "add_multihead_attention")
    kv_dim = num_kv_heads * head_dim
    if window_size is not None and window_size < 0:
        raise ValueError(f"window_size must be >= 0 or None, got {window_size}")
    omega = _validate_attention_kernel(attention_kernel, positional_scheme, window_size,
                                       dropout, head_dim, num_features)
    if sparse_pattern is not None:
        from .sparse_attention import normalize_pattern
        if attention_kernel != "softmax":
            raise ValueError(
                f"sparse_pattern is incompatible with attention_kernel="
                f"{attention_kernel!r}: a linearized kernel never builds a score "
                "matrix, so there is nothing to sparsify.")
        if window_size is not None:
            raise ValueError(
                "sparse_pattern and window_size both restrict which keys a query "
                "sees; use sparse_pattern's 'local' blocks for a windowed pattern "
                "rather than combining the two.")
        sparse_pattern = normalize_pattern(sparse_pattern)
    if tiled_block_size is not None:
        if int(tiled_block_size) < 1:
            raise ValueError(
                f"tiled_block_size must be >= 1 or None, got {tiled_block_size}")
        if attention_kernel != "softmax" or sparse_pattern is not None:
            raise ValueError(
                "tiled_block_size only applies to the plain softmax path; the "
                "linearized kernels and sparse_pattern already avoid building a "
                "score matrix, so there is nothing to tile.")
        if dropout > 0:
            raise ValueError(
                "tiled_block_size is incompatible with attention dropout: the "
                "streaming path never holds the full attention matrix, so there "
                "is nothing to apply a single consistent dropout mask to.")
        tiled_block_size = int(tiled_block_size)
    if positional_scheme not in ("absolute", "rope", "alibi"):
        raise ValueError(f"Unknown positional_scheme: {positional_scheme!r}. Expected 'absolute', 'rope', or 'alibi'.")
    if positional_scheme == "rope" and head_dim % 2 != 0:
        raise ValueError(f"positional_scheme='rope' requires an even head_dim, got {head_dim}")
    Wq, bq = init_weights(embed_dim, embed_dim, method=init_method)
    Wk, bk = init_weights(embed_dim, kv_dim, method=init_method)
    Wv, bv = init_weights(embed_dim, kv_dim, method=init_method)
    Wo, bo = init_weights(embed_dim, embed_dim, method=init_method)
    self.layers.append({
        "type": "multihead_attention",
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dropout": dropout,
        "causal": causal,
        "window_size": window_size,
        "attention_kernel": attention_kernel,
        "sparse_pattern": sparse_pattern,
        "tiled_block_size": tiled_block_size,
        "positional_scheme": positional_scheme,
        "Wq": Wq, "bq": bq,
        "Wk": Wk, "bk": bk,
        "Wv": Wv, "bv": bv,
        "Wo": Wo, "bo": bo,
    })
    if omega is not None:
        self.layers[-1]["omega"] = omega
    self._last_width = embed_dim
    self._last_spatial = None
    self._last_spatial_1d = None

def add_moe(self: Any, embed_dim: Optional[int] = None, num_experts: int = 4,
            hidden_dim: Optional[int] = None, top_k: int = 1,
            activation: str = "gelu", aux_loss_weight: float = 0.01,
            init_method: str = "xavier_uniform") -> None:
    """Add a Mixture-of-Experts feed-forward layer: a router picks the top_k
    of `num_experts` MLPs per token, and only those run for that token.
    Input/output shape (..., embed_dim) -- a drop-in replacement for the MLP
    half of a Transformer block.

    hidden_dim: each expert's inner width. Defaults to embed_dim * 4.
    top_k: experts per token (1 = Switch Transformer, 2 = Mixtral-style).
    aux_loss_weight: strength of the load-balancing loss that keeps the
        router from collapsing onto a few experts. 0 disables it.
        `model.moe_aux_loss()` reports its current value.
    """
    if embed_dim is None:
        embed_dim = _require_width(self, "add_moe")
    if num_experts < 1:
        raise ValueError(f"num_experts must be >= 1, got {num_experts}")
    if not (1 <= top_k <= num_experts):
        raise ValueError(
            f"top_k must be between 1 and num_experts ({num_experts}), got {top_k}")
    hidden_dim = hidden_dim or embed_dim * 4

    W1, b1, W2, b2 = [], [], [], []
    for _ in range(num_experts):
        w1, bb1 = init_weights(embed_dim, hidden_dim, method=init_method)
        w2, bb2 = init_weights(hidden_dim, embed_dim, method=init_method)
        W1.append(w1); b1.append(bb1); W2.append(w2); b2.append(bb2)
    Wr, br = init_weights(embed_dim, num_experts, method=init_method)

    self.layers.append({
        "type": "moe",
        "embed_dim": embed_dim,
        "num_experts": num_experts,
        "hidden_dim": hidden_dim,
        "top_k": top_k,
        "activation": activation,
        "aux_loss_weight": aux_loss_weight,
        "W1": np.stack(W1), "b1": np.stack(b1),
        "W2": np.stack(W2), "b2": np.stack(b2),
        "Wr": Wr, "br": br,
    })
    self._last_width = embed_dim
    self._last_spatial = None
    self._last_spatial_1d = None


def add_cross_attention(self: Any, kv_source_index: int, embed_dim: Optional[int] = None,
                         num_heads: int = 4, dropout: float = 0.0,
                         init_method: str = "xavier_uniform",
                         num_kv_heads: Optional[int] = None) -> None:
    """Add a cross-attention layer: queries come from the sequential `x`
    (batch, seq_len_q, embed_dim); keys/values from an earlier layer's
    output, which must be (batch, seq_len_kv, embed_dim) with the SAME
    embed_dim. Stores its own Wq/bq..Wo/bo, like add_multihead_attention.

    kv_source_index indexes the KV-source layer DIRECTLY --
    self.outputs[kv_source_index + 1] -- the same convention as
    "goto"/"concat_at", not residual's save_index (which points at a marker
    layer and needs -1).

    num_kv_heads selects MHA/MQA/GQA exactly as in add_multihead_attention.
    There is no causal masking (cross-attention conventionally attends over
    the whole KV source) and no positional_scheme (RoPE/ALiBi are
    self-attention concepts)."""
    if not (-1 <= kv_source_index < len(self.layers)):
        raise ValueError(
            f"kv_source_index={kv_source_index} is out of range: must be -1 "
            f"(the raw network input) or a valid index into the {len(self.layers)} "
            "layer(s) already added before this cross_attention call."
        )
    if embed_dim is None:
        embed_dim = _require_width(self, "add_cross_attention")
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
    head_dim = embed_dim // num_heads
    num_kv_heads = _validate_num_kv_heads(num_heads, num_kv_heads, "add_cross_attention")
    kv_dim = num_kv_heads * head_dim
    Wq, bq = init_weights(embed_dim, embed_dim, method=init_method)
    Wk, bk = init_weights(embed_dim, kv_dim, method=init_method)
    Wv, bv = init_weights(embed_dim, kv_dim, method=init_method)
    Wo, bo = init_weights(embed_dim, embed_dim, method=init_method)
    self.layers.append({
        "type": "cross_attention",
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
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

def add_mlp_block(self: Any, hidden_dims: List[int], in_dim: Optional[int] = None,
                   out_dim: Optional[int] = None, activation: str = "relu",
                   out_activation: str = "linear", init_method: str = "xavier_uniform") -> None:
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

def add_conv_block(self: Any, out_ch: int, k: int = 3, activation: str = "relu",
                    init_method: str = "he_normal", in_ch: Optional[int] = None, stride: int = 1,
                    batchnorm: bool = False, pool: Optional[str] = None,
                    input_size: Optional[tuple] = None, padding: str = "valid") -> None:
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

def add_rnn(self: Any, n_in: Optional[int] = None, hidden_dim: int = 128,
            return_sequences: bool = True, init_method: str = "xavier_uniform",
            stateful: bool = False) -> None:
    """Add a vanilla (tanh) recurrent layer. Input/output: (batch, seq_len,
    features); if return_sequences=False, output is just the last timestep's
    hidden state, (batch, hidden_dim)."""
    if n_in is None:
        n_in = _require_width(self, "add_rnn")
    Wx, b = init_weights(n_in, hidden_dim, method=init_method)
    Wh, _ = init_weights(hidden_dim, hidden_dim, method=init_method)
    self.layers.append({"type": "rnn", "Wx": Wx, "Wh": Wh, "b": b,
                        "n_in": n_in, "hidden_dim": hidden_dim,
                        "return_sequences": return_sequences,
                        "stateful": stateful})
    self._last_width = hidden_dim
    self._last_spatial = None
    self._last_spatial_1d = None

def add_lstm(self: Any, n_in: Optional[int] = None, hidden_dim: int = 128,
             return_sequences: bool = True, init_method: str = "xavier_uniform",
             stateful: bool = False) -> None:
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
                        "return_sequences": return_sequences,
                        "stateful": stateful})
    self._last_width = hidden_dim
    self._last_spatial = None
    self._last_spatial_1d = None

def add_gru(self: Any, n_in: Optional[int] = None, hidden_dim: int = 128,
            return_sequences: bool = True, init_method: str = "xavier_uniform",
            stateful: bool = False) -> None:
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
                        "return_sequences": return_sequences,
                        "stateful": stateful})
    self._last_width = hidden_dim
    self._last_spatial = None
    self._last_spatial_1d = None

def _add_bidirectional(self: Any, add_fn: Any, n_in: Optional[int], hidden_dim: int,
                        return_sequences: bool, init_method: str) -> None:
    """Shared composition for add_bidirectional_rnn/_lstm/_gru: run `add_fn`
    forward, again on the time-reversed input (via "goto"/
    "reverse_sequence"), then join both directions with "concat_at"."""

    # n_in must already be a concrete int here. After the forward direction
    # runs, self._last_width is hidden_dim, not the input width -- the "goto"
    # back to source_index does not update that bookkeeping (only add_* calls
    # do), so auto-inference would size the backward direction wrongly.
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

def add_bidirectional_rnn(self: Any, n_in: Optional[int] = None, hidden_dim: int = 128,
                           return_sequences: bool = True, init_method: str = "xavier_uniform") -> None:
    """Bidirectional vanilla RNN: runs add_rnn once forward and once over
    the time-reversed input, concatenating both directions' hidden states
    along the feature axis -- output width is hidden_dim * 2.
    return_sequences=True: (batch, seq_len, hidden_dim*2).
    return_sequences=False: (batch, hidden_dim*2) (forward's final state
    concatenated with backward's final state -- see _add_bidirectional's
    docstring for why no un-reversing is needed in that case)."""
    _add_bidirectional(self, self.add_rnn, n_in, hidden_dim, return_sequences, init_method)

def add_bidirectional_lstm(self: Any, n_in: Optional[int] = None, hidden_dim: int = 128,
                            return_sequences: bool = True, init_method: str = "xavier_uniform") -> None:
    """Bidirectional LSTM -- see add_bidirectional_rnn for the composition
    and shape convention (output width hidden_dim * 2)."""
    _add_bidirectional(self, self.add_lstm, n_in, hidden_dim, return_sequences, init_method)

def add_bidirectional_gru(self: Any, n_in: Optional[int] = None, hidden_dim: int = 128,
                           return_sequences: bool = True, init_method: str = "xavier_uniform") -> None:
    """Bidirectional GRU -- see add_bidirectional_rnn for the composition
    and shape convention (output width hidden_dim * 2)."""
    _add_bidirectional(self, self.add_gru, n_in, hidden_dim, return_sequences, init_method)

def add_residual_start(self: Any) -> None:
    """Mark the start of a residual/skip-connection block. Pair with a later
    add_residual_end() to compute `x = x + saved_x` (e.g. around attention or
    an MLP sub-block, as in a standard Transformer block). Nestable."""
    self._residual_stack.append(len(self.layers))
    # Record the shape bookkeeping as it stands here. A residual branch
    # returns the same shape it started with, but a GATE branch does not --
    # it pools and flattens down to a per-channel vector -- so closing the
    # block has to restore what the saved tensor's shape was, or the next
    # auto-inferring layer sizes itself from the branch instead.
    self.layers.append({
        "type": "residual_save",
        "shape_at_save": [self._last_width, self._last_spatial,
                          self._last_spatial_1d],
    })

def add_residual_end(self: Any) -> None:
    """Add the input saved at the matching add_residual_start() back to the
    current activations: x = x + saved_x."""
    if not self._residual_stack:
        raise ValueError("add_residual_end() called without a matching add_residual_start()")
    save_index = self._residual_stack.pop()
    self.layers.append({"type": "residual_add", "save_index": save_index})
    _restore_shape_at_save(self, save_index)


def _restore_shape_at_save(self: Any, save_index: int) -> None:
    """Put the shape bookkeeping back to what it was at the matching
    add_residual_start(), since the block's OUTPUT has the saved tensor's
    shape however much the branch changed it in between."""
    saved = self.layers[save_index].get("shape_at_save")
    if saved is None:
        return                    # a model built before this was recorded
    width, spatial, spatial_1d = saved
    self._last_width = width
    self._last_spatial = tuple(spatial) if spatial is not None else None
    self._last_spatial_1d = tuple(spatial_1d) if spatial_1d is not None else None


def add_multiply_end(self: Any) -> None:
    """Close a block opened by add_residual_start() by MULTIPLYING instead of
    adding: x = saved * x, with x broadcast up to saved's rank by appending
    trailing singleton axes. The gating primitive behind Squeeze-and-
    Excitation, CBAM and EfficientNet's SE stage."""
    if not self._residual_stack:
        raise ValueError("add_multiply_end() called without a matching add_residual_start()")
    save_index = self._residual_stack.pop()
    self.layers.append({"type": "residual_mul", "save_index": save_index})
    _restore_shape_at_save(self, save_index)


def reset_rnn_state(self: Any) -> None:
    """Clear the carried hidden/cell state of every stateful RNN/LSTM/GRU
    layer (start a fresh stream). No-op for non-stateful layers."""
    for layer in self.layers:
        if layer.get("type") in ("rnn", "lstm", "gru"):
            layer.pop("_state_h", None)
            layer.pop("_state_c", None)
