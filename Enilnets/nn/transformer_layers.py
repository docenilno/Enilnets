#!/usr/bin/env python3
"""
Transformer (self-attention) layers for Enilnets.
Enables modern NLP, vision transformer (ViT), and cross-modal models.
"""
from typing import Any, Optional

from ..core.backend import np
from ..core import backend
from ..core import constants
from .weight_init import init_embedding_weights

def add_transformer_block(self: Any, embed_dim: Optional[int] = None, num_heads: int = 4,
                           mlp_ratio: float = 4.0, dropout: float = 0.0, activation: str = "swish",
                           causal: bool = False, positional_scheme: str = "absolute",
                           num_kv_heads: Optional[int] = None,
                           window_size: Optional[int] = None,
                           attention_kernel: str = "softmax",
                           num_features: Optional[int] = None,
                           sparse_pattern: Optional[Any] = None,
                           tiled_block_size: Optional[int] = None) -> None:
    """Add a full pre-norm Transformer block: LayerNorm -> Attention ->
    LayerNorm -> MLP, with residual connections around each half.
    Input/output shape (batch, seq_len, embed_dim); embed_dim is inferred
    from the previous layer when None. MLP hidden dim = embed_dim *
    mlp_ratio. `causal`, `positional_scheme` and `num_kv_heads` (MHA / MQA
    / GQA), `window_size` (sliding-window attention) and
    `attention_kernel`/`num_features` (softmax / linear / Performer) and
    `sparse_pattern` (block-sparse attention) and `tiled_block_size`
    (streaming/Flash softmax) are forwarded to add_multihead_attention."""
    if embed_dim is None:
        if self._last_width is None:
            raise ValueError("Cannot auto-infer embed_dim for add_transformer_block: no previous layer.")
        embed_dim = self._last_width

    # Pre-norm: x = x + Attention(LayerNorm(x))
    self.add_residual_start()
    self.add_layernorm((embed_dim,))
    self.add_multihead_attention(embed_dim, num_heads, dropout, causal=causal,
                                 positional_scheme=positional_scheme,
                                 num_kv_heads=num_kv_heads,
                                 window_size=window_size,
                                 attention_kernel=attention_kernel,
                                 num_features=num_features,
                                 sparse_pattern=sparse_pattern,
                                 tiled_block_size=tiled_block_size)
    self.add_residual_end()

    # Pre-norm: x = x + MLP(LayerNorm(x))
    self.add_residual_start()
    self.add_layernorm((embed_dim,))
    mlp_hidden = int(embed_dim * mlp_ratio)
    self.add_dense(embed_dim, mlp_hidden, activation=activation, init_method="xavier_uniform")
    self.add_dense(mlp_hidden, embed_dim, activation="linear", init_method="xavier_uniform")
    if dropout > 0:
        self.add_dropout(dropout)
    self.add_residual_end()

def add_positional_encoding(self: Any, max_seq_len: int, embed_dim: Optional[int] = None,
                             learnable: bool = True, base: Optional[float] = None) -> None:
    """Add positional encoding, to be called after the embedding layer.

    embed_dim is inferred from the previous layer when None. `learnable`
    picks learned embeddings (BERT/GPT style) over fixed sinusoidal ones
    (original Transformer). `base` sets the sinusoidal base frequency
    (default constants.SINUSOIDAL_BASE) and is ignored when learnable."""
    if embed_dim is None:
        if self._last_width is None:
            raise ValueError("Cannot auto-infer embed_dim for add_positional_encoding: no previous layer.")
        embed_dim = self._last_width

    if learnable:
        # A trainable (max_seq_len, embed_dim) table, ADDED to the input at
        # each position -- same additive convention as the sinusoidal branch
        # below, just with learned instead of fixed values. This is its own
        # "positional_encoding" layer (not an "embedding" layer): an
        # embedding layer's forward *replaces* x with a lookup into its
        # table (indexed by integer token ids), which is the wrong op here
        # since x at this point is already the float token-embedding output
        # of the preceding add_embedding call, not an index array.
        w = init_embedding_weights(max_seq_len, embed_dim, method="normal")
        self.layers.append({
            "type": "positional_encoding",
            "weights": w,
            "max_seq_len": max_seq_len,
            "embed_dim": embed_dim,
            "_is_positional": True,
            "_pos_type": "learnable",
        })
        self._last_width = embed_dim
        self._last_spatial = None
        self._last_spatial_1d = None
    else:
        # Fixed sinusoidal encoding stored as a constant
        freq_base = constants.SINUSOIDAL_BASE if base is None else base
        pe = np.zeros((max_seq_len, embed_dim), dtype=backend.default_dtype())
        position = np.arange(max_seq_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(freq_base) / embed_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        # Store as a special layer
        self.layers.append({
            "type": "positional_encoding",
            "pe": pe,
            "max_seq_len": max_seq_len,
            "embed_dim": embed_dim,
            "_is_positional": True,
            "_pos_type": "sinusoidal"
        })
        self._last_width = embed_dim
        self._last_spatial = None
        self._last_spatial_1d = None

def add_vision_transformer_patch_embed(self: Any, img_size: int, patch_size: int,
                                        in_channels: Optional[int] = None, embed_dim: int = 768) -> None:
    """Add a Vision Transformer patch embedding:
    (B, C, H, W) -> (B, (img_size/patch_size)**2, embed_dim).

    `img_size` assumes a square image. `in_channels` is inferred from the
    previous conv/pool layer when None, defaulting to 3 for a first layer."""
    assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
    num_patches = (img_size // patch_size) ** 2
    if in_channels is None:
        in_channels = self._last_spatial[0] if self._last_spatial is not None else 3

    # Use a conv layer as patch embedding (kernel=patch_size, stride=patch_size)
    self.add_conv2d(in_channels, embed_dim, k=patch_size, stride=patch_size, activation="linear",
                    input_size=(img_size, img_size))
    self.add_flatten()  # (B, embed_dim, H//p, W//p) -> (B, embed_dim, num_patches)

    # Store config
    self.layers[-2]["_is_vit_patch"] = True
    self.layers[-2]["vit_config"] = {
        "img_size": img_size,
        "patch_size": patch_size,
        "num_patches": num_patches,
        "embed_dim": embed_dim
    }
