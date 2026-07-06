#!/usr/bin/env python3
"""
Transformer (self-attention) layers for Enilnets.
Enables modern NLP, vision transformer (ViT), and cross-modal models.
"""
from .backend import np
from . import backend
from . import constants

def add_transformer_block(self, embed_dim=None, num_heads=4, mlp_ratio=4.0, dropout=0.0, activation="swish",
                           causal=False, positional_scheme="absolute"):
    """
    Add a full Transformer block: LayerNorm -> Attention -> LayerNorm -> MLP.

    Parameters
    ----------
    embed_dim : int or None
        Embedding dimension. If None, inferred from the previous layer.
    num_heads : int
    mlp_ratio : float
        MLP hidden dim = embed_dim * mlp_ratio
    dropout : float
    activation : str
    causal : bool
        If True, use a causal (autoregressive) self-attention mask -- for
        GPT-style decoders / language models.
    positional_scheme : str
        Forwarded to add_multihead_attention: "absolute" (default), "rope",
        or "alibi".

    Notes
    -----
    Input/output shape: (batch, seq_len, embed_dim)
    """
    if embed_dim is None:
        if self._last_width is None:
            raise ValueError("Cannot auto-infer embed_dim for add_transformer_block: no previous layer.")
        embed_dim = self._last_width

    # Pre-norm: x = x + Attention(LayerNorm(x))
    self.add_residual_start()
    self.add_layernorm((embed_dim,))
    self.add_multihead_attention(embed_dim, num_heads, dropout, causal=causal, positional_scheme=positional_scheme)
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

def add_positional_encoding(self, max_seq_len, embed_dim=None, learnable=True, base=None):
    """
    Add positional encoding to the model.

    Parameters
    ----------
    max_seq_len : int
        Maximum sequence length
    embed_dim : int or None
        Embedding dimension. If None, inferred from the previous layer
        (e.g. a preceding add_embedding call).
    learnable : bool
        If True, use learnable positional embeddings (like BERT/GPT)
        If False, use fixed sinusoidal encodings (like original Transformer)
    base : float or None
        Base frequency for the sinusoidal encoding (default: constants.SINUSOIDAL_BASE).
        Ignored when learnable=True.

    Notes
    -----
    This should be called after the embedding layer.
    The positional encoding is added to the input embeddings.
    """
    if embed_dim is None:
        if self._last_width is None:
            raise ValueError("Cannot auto-infer embed_dim for add_positional_encoding: no previous layer.")
        embed_dim = self._last_width

    if learnable:
        # Add as an embedding layer that will be added to input
        self.add_embedding(max_seq_len, embed_dim, init_method="normal")
        self.layers[-1]["_is_positional"] = True
        self.layers[-1]["_pos_type"] = "learnable"
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

def add_vision_transformer_patch_embed(self, img_size, patch_size, in_channels=None, embed_dim=768):
    """
    Add Vision Transformer (ViT) patch embedding.

    Parameters
    ----------
    img_size : int
        Image size (assumes square, e.g., 224)
    patch_size : int
        Patch size (e.g., 16)
    in_channels : int or None
        Number of input channels (3 for RGB). If None, inferred from the
        previous conv/pool layer, defaulting to 3 if this is the first layer.
    embed_dim : int
        Embedding dimension for each patch

    Notes
    -----
    Converts (B, C, H, W) -> (B, num_patches, embed_dim)
    where num_patches = (img_size / patch_size) ** 2
    """
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
