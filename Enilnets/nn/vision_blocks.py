"""Named computer-vision blocks (roadmap items 54-56).

Each is a specific arrangement of primitives the library already has, so a
caller writes one line instead of eight. The genuinely new pieces are the
primitives they needed -- ``add_global_maxpool2d``, ``add_channel_pool``,
``add_spp``, and the multiplicative gate ``add_multiply_end`` in
``layers.py``.

Builders live here rather than in ``vision/``, which is image I/O and array
utilities: anything bound onto ``NeuralNet`` belongs with its siblings."""

from typing import Any, Optional, Sequence


def add_global_maxpool2d(self: Any) -> None:
    """Global max pool: ``(B, C, H, W) -> (B, C, 1, 1)``, the max over each
    channel's whole spatial extent. The counterpart to
    ``add_global_avgpool2d``, and half of CBAM's channel attention."""
    self.layers.append({"type": "globalmaxpool2d"})
    if self._last_spatial is not None:
        self._last_spatial = (self._last_spatial[0], 1, 1)


def add_channel_pool(self: Any) -> None:
    """Pool ACROSS channels: ``(B, C, H, W) -> (B, 2, H, W)``, stacking the
    per-position mean and max. CBAM's spatial-attention front end."""
    self.layers.append({"type": "channel_pool"})
    self._last_width = 2
    if self._last_spatial is not None:
        self._last_spatial = (2, self._last_spatial[1], self._last_spatial[2])


def add_spp(self: Any, pool_sizes: Sequence[int] = (1, 2, 4)) -> None:
    """Spatial Pyramid Pooling: max-pool to each grid size in `pool_sizes`
    and concatenate the flattened results, giving ``(B, C * sum(n**2))``.

    The point is that the output length depends only on `pool_sizes` and the
    channel count -- never on the input resolution -- so a convolutional
    stack can accept variably sized images and still feed a fixed-width
    dense head."""
    pool_sizes = [int(n) for n in pool_sizes]
    if not pool_sizes or any(n < 1 for n in pool_sizes):
        raise ValueError(f"pool_sizes must be positive ints, got {pool_sizes}")
    self.layers.append({"type": "spp", "pool_sizes": pool_sizes})
    if self._last_spatial is not None:
        channels = self._last_spatial[0]
        self._last_width = channels * sum(n * n for n in pool_sizes)
    self._last_spatial = None
    self._last_spatial_1d = None


def _require_channels(self: Any, channels: Optional[int], what: str) -> int:
    if channels is not None:
        return int(channels)
    if self._last_spatial is not None:
        return int(self._last_spatial[0])
    if self._last_width is not None:
        return int(self._last_width)
    raise ValueError(
        f"Cannot infer the channel count for {what}: no previous conv/pool "
        "layer. Pass `channels` explicitly.")


def add_se_block(self: Any, channels: Optional[int] = None, reduction: int = 16,
                 activation: str = "relu") -> None:
    """Squeeze-and-Excitation (Hu et al. 2018): learn one scalar weight per
    channel from global context and rescale the feature map by it.

    Squeeze the ``(B, C, H, W)`` input to ``(B, C)`` by global average
    pooling, pass it through a bottleneck MLP (``C -> C/reduction -> C``,
    sigmoid at the end), and multiply the original feature map by the
    result. Shape is unchanged; the parameter cost is ``2 * C^2 / reduction``.

    `reduction` is clamped so the hidden width is at least 1, which matters
    for the small channel counts a from-scratch model actually uses."""
    channels = _require_channels(self, channels, "add_se_block")
    hidden = max(1, channels // max(1, int(reduction)))
    self.add_residual_start()
    self.add_global_avgpool2d()
    self.add_flatten()
    self.add_dense(channels, hidden, activation=activation)
    self.add_dense(hidden, channels, activation="sigmoid")
    self.add_multiply_end()


def add_cbam_channel(self: Any, channels: Optional[int] = None, reduction: int = 16,
                     activation: str = "relu",
                     init_method: str = "xavier_uniform") -> None:
    """CBAM's channel attention as a single layer: gate each channel by
    ``sigmoid(MLP(avgpool(x)) + MLP(maxpool(x)))``, shape unchanged.

    One layer rather than a composition because the MLP is SHARED between
    the two pooled paths, and a list of layer dicts has no way to tie two
    dense layers' weights together. Summing before the sigmoid is also what
    distinguishes this from running SE twice."""
    from .weight_init import init_weights
    channels = _require_channels(self, channels, "add_cbam_channel")
    hidden = max(1, channels // max(1, int(reduction)))
    W1, b1 = init_weights(channels, hidden, method=init_method)
    W2, b2 = init_weights(hidden, channels, method=init_method)
    self.layers.append({
        "type": "cbam_channel", "channels": channels, "hidden": hidden,
        "activation": activation,
        "W1": W1, "b1": b1, "W2": W2, "b2": b2,
    })


def add_cbam_block(self: Any, channels: Optional[int] = None, reduction: int = 16,
                   kernel_size: int = 7, activation: str = "relu") -> None:
    """CBAM (Woo et al. 2018): channel attention, then spatial attention,
    each applied as a multiplicative gate. Shape unchanged throughout.

    Channel attention (:func:`add_cbam_channel`) reweights channels from
    global context. Spatial attention then pools ACROSS channels to
    ``(B, 2, H, W)``, convolves that to a one-channel map, and gates every
    position by it -- "what to attend to" after "which features matter".

    `kernel_size` must be odd so the spatial conv can use "same" padding and
    preserve resolution."""
    channels = _require_channels(self, channels, "add_cbam_block")
    if kernel_size % 2 == 0:
        raise ValueError(
            f"cbam kernel_size must be odd so the spatial attention conv can "
            f'use padding="same"; got {kernel_size}')
    self.add_cbam_channel(channels, reduction=reduction, activation=activation)
    self.add_residual_start()
    self.add_channel_pool()
    self.add_conv2d(2, 1, k=kernel_size, activation="sigmoid", padding="same")
    self.add_multiply_end()


def add_convnext_block(self: Any, channels: Optional[int] = None,
                       mlp_ratio: float = 4.0, kernel_size: int = 7,
                       activation: str = "gelu") -> None:
    """A ConvNeXt block (Liu et al. 2022): depthwise conv -> LayerNorm ->
    pointwise expand -> activation -> pointwise project, wrapped in a
    residual connection.

    The characteristic choices are a large depthwise kernel (7x7), a single
    normalization rather than one per conv, and an inverted bottleneck that
    widens by `mlp_ratio` in the middle -- a convolutional stack arranged
    the way a transformer block is."""
    channels = _require_channels(self, channels, "add_convnext_block")
    if kernel_size % 2 == 0:
        raise ValueError(f"convnext kernel_size must be odd, got {kernel_size}")
    hidden = max(1, int(channels * mlp_ratio))
    self.add_residual_start()
    # Depthwise via a grouped conv with groups == channels is the intent;
    # this library's conv2d is dense, so the block uses a full conv of the
    # same kernel size. Documented in the README rather than silently
    # differing -- the arrangement is what the block is about.
    self.add_conv2d(channels, channels, k=kernel_size, activation="linear",
                    padding="same")
    self.add_layernorm((channels,))
    self.add_conv2d(channels, hidden, k=1, activation=activation, padding="same")
    self.add_conv2d(hidden, channels, k=1, activation="linear", padding="same")
    self.add_residual_end()


def add_efficientnet_block(self: Any, channels: Optional[int] = None,
                           expand_ratio: float = 4.0, kernel_size: int = 3,
                           reduction: int = 16, activation: str = "swish") -> None:
    """An EfficientNet MBConv block (Tan & Le 2019): expand pointwise,
    convolve, squeeze-and-excite, project back, add the residual.

    The inverted residual is the point -- it widens in the middle and the
    skip connects the two NARROW ends, the opposite of a ResNet
    bottleneck -- and the SE stage (#54) sits between the conv and the
    projection."""
    channels = _require_channels(self, channels, "add_efficientnet_block")
    if kernel_size % 2 == 0:
        raise ValueError(f"efficientnet kernel_size must be odd, got {kernel_size}")
    hidden = max(1, int(channels * expand_ratio))
    self.add_residual_start()
    self.add_conv2d(channels, hidden, k=1, activation=activation, padding="same")
    self.add_conv2d(hidden, hidden, k=kernel_size, activation=activation,
                    padding="same")
    self.add_se_block(hidden, reduction=reduction)
    self.add_conv2d(hidden, channels, k=1, activation="linear", padding="same")
    self.add_residual_end()
