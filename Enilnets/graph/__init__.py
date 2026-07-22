"""Enilnets.graph -- the autograd engine (roadmap Phase 1).

An **additive** package: it coexists with the manual ``nn``/``NeuralNet``
machinery and never replaces it. Reverse-mode automatic differentiation
over a dynamic computation graph built by tracing actual Python execution:

- ``Tensor`` / ``no_grad`` (tensor.py) -- the array-plus-graph object and
  ``backward()``.
- elementary ops + ``custom_op`` (ops.py) -- every op is a forward formula
  plus one local-gradient rule; user-defined ops use the identical API.
- ``Layer`` / ``Parameter`` / ``Linear`` / ``Sequential`` (layers.py) --
  compose ops in a ``forward()``; gradients come for free.
- ``trace`` / ``symbolic_trace`` / ``Graph`` (tracing.py) -- export the
  traced op graph, introspect it, re-run it on new inputs.
- ``optimize`` (optimize.py) -- dead-node elimination, constant folding,
  elementwise-chain fusion over traced graphs.

See the README's "Autograd" section for a quickstart."""

from .tensor import Tensor, as_tensor, no_grad, is_grad_enabled
from .ops import (
    Op, custom_op, cast, autocast,
    add, sub, mul, div, neg, power, matmul,
    exp, log, sqrt, tanh, sigmoid, relu, conj, real, imag, absolute,
    reshape, transpose, getitem, concatenate, pad,
    sum_, mean, max_, softmax, log_softmax,
)
from .layers import Parameter, Layer, Linear, LazyLinear, ReLU, Tanh, Sigmoid, Dropout, GaussianDropout, AlphaDropout, DropBlock2D, StochasticDepth, Pad, PixelShuffle, PixelUnshuffle, Sequential
from .tracing import Graph, Node, trace, symbolic_trace
from .checkpoint import checkpoint
from .conv import (
    conv1d, conv2d, conv3d, causal_conv1d,
    conv_transpose1d, conv_transpose2d, conv_transpose3d,
    Conv1D, Conv2D, SeparableConv2D,
)
from .pooling import (
    adaptive_avg_pool2d, adaptive_max_pool2d, fractional_max_pool2d,
    max_pool2d, max_pool2d_with_indices, max_unpool2d,
)
from .sequence import (
    lengths_to_mask, PackedSequence, pack_padded, pad_packed,
    MultiHeadAttention, masked_mean,
)
from .optimize import (
    optimize, eliminate_dead_nodes, fold_constants, fuse_elementwise,
)
from . import ops
from . import functional
from .quantization import (fake_quant, quantize_symmetric, QATLinear,
                           MovingRangeObserver)
from .audio import (stft as audio_stft, spectrogram, mel_spectrogram,
                    log_mel_spectrogram, mel_filterbank, window_function)

__all__ = [
    "Tensor", "as_tensor", "no_grad", "is_grad_enabled",
    "Op", "custom_op", "cast", "autocast",
    "add", "sub", "mul", "div", "neg", "power", "matmul",
    "exp", "log", "sqrt", "tanh", "sigmoid", "relu",
    "conj", "real", "imag", "absolute",
    "reshape", "transpose", "getitem", "concatenate", "pad",
    "sum_", "mean", "max_", "softmax", "log_softmax",
    "Parameter", "Layer", "Linear", "LazyLinear", "ReLU", "Tanh", "Sigmoid", "Dropout", "GaussianDropout", "AlphaDropout", "DropBlock2D", "StochasticDepth", "Pad", "PixelShuffle", "PixelUnshuffle",
    "Sequential",
    "Graph", "Node", "trace", "symbolic_trace",
    "optimize", "eliminate_dead_nodes", "fold_constants", "fuse_elementwise",
    "checkpoint",
    "conv1d", "conv2d", "conv3d", "causal_conv1d",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    "Conv1D", "Conv2D", "SeparableConv2D",
    "adaptive_avg_pool2d", "adaptive_max_pool2d", "fractional_max_pool2d",
    "max_pool2d", "max_pool2d_with_indices", "max_unpool2d",
    "lengths_to_mask", "PackedSequence", "pack_padded", "pad_packed",
    "MultiHeadAttention", "masked_mean",
    "ops", "functional",
    "audio_stft", "spectrogram", "mel_spectrogram", "log_mel_spectrogram",
    "mel_filterbank", "window_function",
    "fake_quant", "quantize_symmetric", "QATLinear", "MovingRangeObserver",
]
