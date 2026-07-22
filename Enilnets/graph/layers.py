"""Graph-based layers: compose ops from ``ops.py`` in a ``forward()`` and
gradients come for free via autograd -- the ``graph/`` equivalent of
``nn/layers.py``'s ``add_*`` builders, coexisting with (never replacing)
them.

Interop with ``NeuralNet`` is free in both directions: a Tensor's ``.data``
is the raw backend array (no copy, no host<->device transfer), so a graph
layer's output feeds straight into ``model.Forward(t.data)`` and an
``nn/``-trained array can seed a ``Parameter`` without duplication.

See the README's "Custom layers" section for a worked example."""

from typing import Any, Iterator, List, Optional

from ..core.backend import np
from ..core import backend
from ..nn.weight_init import init_weights
from .tensor import Tensor, as_tensor
from . import ops


class Parameter(Tensor):
    """A Tensor that is a trainable weight: ``requires_grad=True`` by
    definition, and discovered by ``Layer.parameters()``. Wraps existing
    arrays by reference, so sharing weights with an ``nn/`` layer dict is
    free."""

    def __init__(self, data: Any, name: Optional[str] = None) -> None:
        super().__init__(data, requires_grad=True, name=name)


class Layer:
    """Base class for graph-based layers.

    Subclasses define ``forward(self, *inputs)`` using ops/Tensors; calling
    the layer instance runs it. ``parameters()`` finds every Parameter
    attached to the layer (directly, in sub-layers, or in lists of either),
    so optimizers and ``zero_grad`` need no manual registration."""

    training: bool = True

    def forward(self, *inputs: Any) -> Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward()")

    def __call__(self, *inputs: Any, **kwargs: Any) -> Tensor:
        # Positional args are tensor inputs (auto-wrapped); keyword args
        # (masks, flags, ...) pass through untouched.
        return self.forward(*(as_tensor(x) for x in inputs), **kwargs)

    def parameters(self) -> List[Parameter]:
        """Every Parameter reachable from this layer, depth-first, in
        attribute-definition order."""
        return list(self._iter_parameters())

    def _iter_parameters(self) -> Iterator[Parameter]:
        for value in self.__dict__.values():
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Layer):
                yield from value._iter_parameters()
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Parameter):
                        yield item
                    elif isinstance(item, Layer):
                        yield from item._iter_parameters()

    def zero_grad(self) -> None:
        """Clear accumulated gradients on every parameter."""
        for p in self.parameters():
            p.zero_grad()

    def train(self) -> "Layer":
        """Set training mode on this layer and every sub-layer (affects
        Dropout-style layers only). Returns self, chainable."""
        self._set_training(True)
        return self

    def eval(self) -> "Layer":
        """Set inference mode (see ``train()``). Returns self, chainable."""
        self._set_training(False)
        return self

    def _set_training(self, flag: bool) -> None:
        self.training = flag
        for value in self.__dict__.values():
            if isinstance(value, Layer):
                value._set_training(flag)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Layer):
                        item._set_training(flag)


class Linear(Layer):
    """A fully-connected layer ``y = x @ W.T + b``, matching ``nn/``'s
    dense-layer weight convention (``weights`` shaped ``(n_out, n_in)``) so
    weights can be shared with an ``add_dense`` layer dict by reference."""

    def __init__(self, n_in: int, n_out: int, use_bias: bool = True,
                 init_method: str = "xavier_uniform") -> None:
        super().__init__()
        w, b = init_weights(n_in, n_out, method=init_method)
        self.weight = Parameter(w, name="weight")
        self.bias = Parameter(b, name="bias") if use_bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = ops.matmul(x, ops.transpose(self.weight))
        if self.bias is not None:
            out = ops.add(out, self.bias)
        return out


class LazyLinear(Layer):
    """A :class:`Linear` that infers ``n_in`` from the first input it sees
    (à la PyTorch's ``LazyLinear``) -- the shape-inference convenience the
    ``add_*`` builders already give the ``nn/`` path, for graph layers.

    ``parameters()`` is empty until the first call materializes the
    weights; build the layer by running one (real or dummy) batch through
    it before handing parameters to an optimizer."""

    def __init__(self, n_out: int, use_bias: bool = True,
                 init_method: str = "xavier_uniform") -> None:
        super().__init__()
        self.n_out = n_out
        self.use_bias = use_bias
        self.init_method = init_method
        self.weight: Optional[Parameter] = None
        self.bias: Optional[Parameter] = None

    def forward(self, x: Tensor) -> Tensor:
        if self.weight is None:
            n_in = int(x.shape[-1])
            w, b = init_weights(n_in, self.n_out, method=self.init_method)
            self.weight = Parameter(w, name="weight")
            self.bias = Parameter(b, name="bias") if self.use_bias else None
        out = ops.matmul(x, ops.transpose(self.weight))
        if self.bias is not None:
            out = ops.add(out, self.bias)
        return out


class ReLU(Layer):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Tanh(Layer):
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)


class Sigmoid(Layer):
    def forward(self, x: Tensor) -> Tensor:
        return ops.sigmoid(x)


class Pad(Layer):
    """Padding layer over every axis of its input (roadmap item 30).

    ``pad_width``: an int (same before/after on all axes), one
    ``(before, after)`` pair (applied to all axes), or one pair per axis --
    for the common NCHW "pad spatial dims only" case pass e.g.
    ``((0, 0), (0, 0), (1, 1), (1, 1))``.
    ``mode``: ``"constant"`` (zero/value), ``"reflect"``, ``"edge"``
    (replication), or ``"wrap"`` (circular)."""

    def __init__(self, pad_width: Any, mode: str = "constant",
                 constant_value: float = 0.0) -> None:
        super().__init__()
        self.pad_width = pad_width
        self.mode = mode
        self.constant_value = constant_value

    def forward(self, x: Tensor) -> Tensor:
        return ops.pad(x, pad_width=self.pad_width, mode=self.mode,
                       constant_value=self.constant_value)


class Dropout(Layer):
    """Inverted dropout (zero `rate` of activations at train time, rescale
    survivors) -- same semantics as ``nn/``'s ``add_dropout``. Identity in
    eval mode."""

    def __init__(self, rate: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"dropout rate must be in [0, 1); got {rate}")
        self.rate = rate

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.rate == 0.0:
            return x
        keep = 1.0 - self.rate
        mask = (np.random.rand(*x.shape) < keep).astype(x.dtype) / keep
        return ops.mul(x, Tensor(mask))


class PixelShuffle(Layer):
    """Layer form of ``functional.pixel_shuffle`` (sub-pixel upsampling)."""

    def __init__(self, upscale_factor: int) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x: Tensor) -> Tensor:
        from .functional import pixel_shuffle
        return pixel_shuffle(x, self.upscale_factor)


class PixelUnshuffle(Layer):
    """Layer form of ``functional.pixel_unshuffle``."""

    def __init__(self, downscale_factor: int) -> None:
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x: Tensor) -> Tensor:
        from .functional import pixel_unshuffle
        return pixel_unshuffle(x, self.downscale_factor)


class GaussianDropout(Layer):
    """Multiplicative Gaussian noise dropout (roadmap item 31): at train
    time multiply by ``N(1, rate/(1-rate))`` noise -- same first moment as
    ordinary inverted dropout (E[out] = x), but smooth instead of hard
    zeroing. Identity in eval mode."""

    def __init__(self, rate: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"dropout rate must be in [0, 1); got {rate}")
        self.rate = rate

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.rate == 0.0:
            return x
        std = (self.rate / (1.0 - self.rate)) ** 0.5
        noise = (1.0 + std * np.random.randn(*x.shape)).astype(x.dtype)
        return ops.mul(x, Tensor(noise))


class AlphaDropout(Layer):
    """SELU-compatible dropout (roadmap item 31): dropped units are set to
    SELU's negative saturation value (not zero) and the result is affinely
    rescaled so a self-normalized input (mean 0, variance 1) keeps mean 0 /
    variance 1 through the layer (Klambauer et al. 2017). Use with "selu"
    networks where ordinary dropout would break self-normalization.
    Identity in eval mode."""

    # -scale * alpha for SELU's standard (alpha, scale) constants.
    _ALPHA_PRIME = -1.7580993408473766

    def __init__(self, rate: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"dropout rate must be in [0, 1); got {rate}")
        self.rate = rate

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.rate == 0.0:
            return x
        keep = 1.0 - self.rate
        ap = self._ALPHA_PRIME
        # Affine correction restoring mean 0 / variance 1 after the mask.
        a = (keep + ap ** 2 * keep * self.rate) ** -0.5
        b = -a * self.rate * ap
        mask = (np.random.rand(*x.shape) < keep).astype(x.dtype)
        kept = ops.mul(x, Tensor(mask))
        dropped_fill = Tensor((ap * (1.0 - mask)).astype(x.dtype))
        return ops.add(ops.mul(ops.add(kept, dropped_fill),
                               Tensor(np.asarray(a, dtype=x.dtype))),
                       Tensor(np.asarray(b, dtype=x.dtype)))


class DropBlock2D(Layer):
    """DropBlock (Ghiasi et al. 2018, roadmap item 34): drop contiguous
    ``block_size x block_size`` spatial regions of a ``(B, C, H, W)`` map
    instead of independent pixels -- ordinary dropout barely regularizes
    conv features because neighbors are correlated. Survivors are rescaled
    to preserve the expectation; identity in eval mode.

    ``seeds``: optional pre-drawn Bernoulli seed mask (the layer's
    reproducibility hook, like fractional pooling's ``random_u``) of shape
    ``(B, C, H-block_size+1, W-block_size+1)``; fresh draws per call
    otherwise."""

    def __init__(self, rate: float = 0.1, block_size: int = 3) -> None:
        super().__init__()
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"drop rate must be in [0, 1); got {rate}")
        self.rate = rate
        self.block_size = int(block_size)

    def forward(self, x: Tensor, seeds: Optional[Any] = None) -> Tensor:
        if not self.training or self.rate == 0.0:
            return x
        B, C, H, W = (int(s) for s in x.shape)
        b = self.block_size
        if H < b or W < b:
            raise ValueError(
                f"block_size {b} exceeds the spatial size {(H, W)}")
        Hv, Wv = H - b + 1, W - b + 1
        if seeds is None:
            # gamma: seed probability chosen so the expected fraction of
            # dropped cells matches `rate` (the paper's formula).
            gamma = self.rate * H * W / (b * b * Hv * Wv)
            seeds = (np.random.rand(B, C, Hv, Wv) < gamma)
        seeds = np.asarray(seeds)
        dropped = np.zeros((B, C, H, W), dtype=x.dtype)
        for di in range(b):          # dilate each seed to its b x b block
            for dj in range(b):
                region = dropped[:, :, di:di + Hv, dj:dj + Wv]
                np.maximum(region, seeds.astype(x.dtype), out=region)
        keep = 1.0 - dropped
        total = keep.size
        kept = float(keep.sum())
        if kept == 0.0:
            return ops.mul(x, Tensor(keep))          # everything dropped
        return ops.mul(x, Tensor(keep * (total / kept)))


class StochasticDepth(Layer):
    """Stochastic depth (Huang et al. 2016, roadmap item 34): wraps a residual
    *branch* so that, per example, the whole branch is dropped with
    probability ``1 - survival_prob`` at train time (survivors rescaled by
    ``1/survival_prob``). The branch always runs in eval mode. Compose as
    ``y = x + StochasticDepth(block)(x)``."""

    def __init__(self, layer: Layer, survival_prob: float = 0.8) -> None:
        super().__init__()
        if not 0.0 < survival_prob <= 1.0:
            raise ValueError(
                f"survival_prob must be in (0, 1]; got {survival_prob}")
        self.layer = layer
        self.survival_prob = survival_prob

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        out = self.layer(x, **kwargs)
        if not self.training or self.survival_prob == 1.0:
            return out
        B = int(out.shape[0])
        keep = (np.random.rand(B) < self.survival_prob).astype(out.dtype)
        keep = keep / self.survival_prob                  # inverted rescale
        shape = (B,) + (1,) * (out.ndim - 1)
        return ops.mul(out, Tensor(keep.reshape(shape)))


class Sequential(Layer):
    """Chain layers in order: ``Sequential(Linear(4, 8), ReLU(), Linear(8, 1))``."""

    def __init__(self, *layers: Layer) -> None:
        super().__init__()
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
