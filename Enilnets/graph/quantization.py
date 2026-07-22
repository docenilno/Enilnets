"""Quantization-aware training (roadmap item 66).

Post-training quantization rounds a finished model and measures the cost;
QAT makes the model train THROUGH the rounding so it learns weights that
survive it.

The difficulty is the gradient: rounding has a derivative of zero almost
everywhere, so backpropagating it honestly stops training dead. The answer
is the straight-through estimator, implemented here as one ``custom_op``
-- no hand-derived backward. It is CLIPPED: gradient passes through inside
the representable range and is zeroed outside, since a value that
saturated the grid cannot be improved by being pushed further out."""

from typing import Any, Optional

from ..core.backend import np
from ..compression.quantization import (compute_scale, fake_quantize,
                                        quant_range)
from .tensor import Tensor, as_tensor
from .layers import Layer, Parameter
from . import ops


def _fake_quant_forward(x, scale=None, zero_point=None, bits=8,
                        scheme="symmetric"):
    return fake_quantize(x, scale, zero_point, bits, scheme)


def _fake_quant_backward(g, out, x, scale=None, zero_point=None, bits=8,
                         scheme="symmetric"):
    # Straight-through: identity inside the range, zero outside it. `out` is
    # the already-rounded value, so comparing against the range in float
    # space is the same test an integer kernel's clamp would apply.
    qmin, qmax = quant_range(bits, scheme)
    low = (qmin - zero_point) * scale
    high = (qmax - zero_point) * scale
    inside = (x >= low) & (x <= high)
    return (g * inside,)


#: ``fake_quant(x, scale=, zero_point=, bits=, scheme=)`` -- rounds onto the
#: integer grid forward, straight-through backward.
fake_quant = ops.custom_op(
    "fake_quant",
    forward=_fake_quant_forward,
    backward=_fake_quant_backward,
    elementwise=True,
)


def quantize_symmetric(x: Any, bits: int = 8) -> Tensor:
    """Fake-quantize a tensor with a symmetric scale read from its own
    current range. The scale is treated as a constant, which is what makes
    the straight-through estimator well defined."""
    x = as_tensor(x)
    data = x.data
    scale, zero_point = compute_scale(np.min(data), np.max(data), bits,
                                      "symmetric")
    return fake_quant(x, scale=scale, zero_point=zero_point, bits=bits,
                      scheme="symmetric")


class MovingRangeObserver:
    """Exponential moving average of a tensor's min/max.

    Activation ranges jump around between batches; using the current
    batch's range directly makes the quantization grid jitter and the
    training signal with it. An EMA is what every QAT implementation uses
    for the same reason."""

    def __init__(self, momentum: float = 0.9) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        self.momentum = momentum
        self.low: Optional[float] = None
        self.high: Optional[float] = None

    def observe(self, x: Any) -> None:
        low, high = float(np.min(x)), float(np.max(x))
        if self.low is None:
            self.low, self.high = low, high
        else:
            m = self.momentum
            self.low = m * self.low + (1 - m) * low
            self.high = m * self.high + (1 - m) * high

    def scale_and_zero_point(self, bits: int, scheme: str):
        if self.low is None:
            raise RuntimeError("observer has seen no data yet")
        return compute_scale(np.asarray(self.low), np.asarray(self.high),
                             bits, scheme)


class QATLinear(Layer):
    """A ``Linear`` that trains through quantization.

        Weights are fake-quantized per output channel on every forward pass, so
        the loss already includes the rounding error and the weights adapt to
        it. With `quantize_activations`, activations are quantized too via a
        moving-average observer updated in training mode and only read in eval
        mode -- the same split batchnorm's running statistics use.

        Same weight convention as ``graph.Linear`` and ``add_dense``, so
        weights move freely between a QAT model and a plain one."""

    def __init__(self, n_in: int, n_out: int, bits: int = 8,
                 per_channel: bool = True, quantize_activations: bool = False,
                 act_bits: int = 8, init_method: str = "xavier_uniform") -> None:
        super().__init__()
        from ..nn.weight_init import init_weights
        w, b = init_weights(n_in, n_out, method=init_method)
        self.weight = Parameter(w, name="weight")
        self.bias = Parameter(b, name="bias")
        self.bits, self.act_bits = int(bits), int(act_bits)
        self.per_channel = per_channel
        self.quantize_activations = quantize_activations
        self.observer = MovingRangeObserver()

    def quantized_weight(self) -> Tensor:
        w = self.weight.data
        if self.per_channel and w.ndim > 1:
            axes = tuple(range(1, w.ndim))
            low = np.min(w, axis=axes, keepdims=True)
            high = np.max(w, axis=axes, keepdims=True)
        else:
            low, high = np.min(w), np.max(w)
        scale, zero_point = compute_scale(low, high, self.bits, "symmetric")
        return fake_quant(self.weight, scale=scale, zero_point=zero_point,
                          bits=self.bits, scheme="symmetric")

    def forward(self, x: Tensor, training: bool = True) -> Tensor:
        out = ops.add(ops.matmul(x, ops.transpose(self.quantized_weight())),
                      self.bias)
        if not self.quantize_activations:
            return out
        if training:
            self.observer.observe(out.data)
        scale, zero_point = self.observer.scale_and_zero_point(
            self.act_bits, "asymmetric")
        return fake_quant(out, scale=scale, zero_point=zero_point,
                          bits=self.act_bits, scheme="asymmetric")
