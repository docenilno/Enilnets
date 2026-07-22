"""Quantization (roadmap items 63 and 66).

**What this does and does not buy you.** NumPy has no int8 matmul -- it
would upcast immediately -- so quantizing here does not make inference
faster. What it does do is reproduce the accuracy effect *exactly*, so you
can measure whether a model survives 8-bit (or 4-bit) before committing to
a deployment target, and it stores the integer representation plus scales
so a saved model can be a quarter the size. Said plainly rather than
implied, because "quantization" usually promises speed.

Weights are stored **fake-quantized**: rounded onto the integer grid and
mapped back to float. The values are exactly those an integer kernel would
reconstruct, so the arithmetic below them is unchanged.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.backend import np


def quant_range(bits: int, scheme: str) -> Tuple[int, int]:
    """The integer interval `bits` at `scheme` maps onto."""
    if not 2 <= bits <= 16:
        raise ValueError(f"bits must be between 2 and 16, got {bits}")
    if scheme == "symmetric":
        return -(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1
    if scheme == "asymmetric":
        return 0, 2 ** bits - 1
    raise ValueError(
        f"scheme must be 'symmetric' or 'asymmetric', got {scheme!r}")


def compute_scale(low: Any, high: Any, bits: int, scheme: str
                  ) -> Tuple[Any, Any]:
    """Scale and zero-point mapping ``[low, high]`` onto the integer grid.

    Symmetric centres the grid on zero, so an exact zero stays exactly zero
    -- which matters for a pruned or ReLU'd tensor, where most values ARE
    zero and a rounding offset would spread them. Asymmetric uses the full
    range, which is worth more when the data is one-sided."""
    qmin, qmax = quant_range(bits, scheme)
    if scheme == "symmetric":
        amax = np.maximum(np.abs(low), np.abs(high))
        scale = np.where(amax > 0, amax / qmax, 1.0)
        zero_point = np.zeros_like(scale)
    else:
        span = high - low
        scale = np.where(span > 0, span / (qmax - qmin), 1.0)
        zero_point = np.round(qmin - low / scale)
    return scale, zero_point


def quantize(x: Any, scale: Any, zero_point: Any, bits: int, scheme: str) -> Any:
    """Float -> integer codes, clipped to the representable range."""
    qmin, qmax = quant_range(bits, scheme)
    return np.clip(np.round(x / scale + zero_point), qmin, qmax)


def dequantize(q: Any, scale: Any, zero_point: Any) -> Any:
    """Integer codes -> the float values they stand for."""
    return (q - zero_point) * scale


def fake_quantize(x: Any, scale: Any, zero_point: Any, bits: int,
                  scheme: str) -> Any:
    """Round `x` onto the integer grid and back. The result is exactly what
    an integer kernel would reconstruct, at float dtype."""
    return dequantize(quantize(x, scale, zero_point, bits, scheme),
                      scale, zero_point)


def _axes_except(ndim: int, axis: int) -> Tuple[int, ...]:
    return tuple(a for a in range(ndim) if a != axis)


def quantize_tensor(w: Any, bits: int = 8, scheme: str = "symmetric",
                    per_channel: bool = False, axis: int = 0
                    ) -> Tuple[Any, Any, Any]:
    """Fake-quantize one tensor. Returns ``(quantized, scale, zero_point)``.

    `per_channel` computes a separate scale per slice along `axis` (output
    channels, by this library's `(n_out, n_in)` convention). One outlier
    channel then cannot force a coarse grid on every other channel, which
    is usually the difference between 8-bit being free and being lossy."""
    w = np.asarray(w)
    if per_channel and w.ndim > 1:
        axes = _axes_except(w.ndim, axis)
        low = np.min(w, axis=axes, keepdims=True)
        high = np.max(w, axis=axes, keepdims=True)
    else:
        low, high = np.min(w), np.max(w)
    scale, zero_point = compute_scale(low, high, bits, scheme)
    return fake_quantize(w, scale, zero_point, bits, scheme), scale, zero_point


#: Which parameters of each layer type are worth quantizing. Biases are
#: excluded: they are a rounding error of the parameter count and are the
#: most sensitive thing in the layer.
QUANTIZABLE = {
    "dense": ("weights",),
    "sparse": ("weights",),
    "conv2d": ("weights",),
    "conv1d": ("weights",),
    "embedding": ("weights",),
    "multihead_attention": ("Wq", "Wk", "Wv", "Wo"),
    "cross_attention": ("Wq", "Wk", "Wv", "Wo"),
    "rnn": ("Wx", "Wh"),
    "lstm": ("Wx", "Wh"),
    "gru": ("Wx", "Wh"),
    "moe": ("W1", "W2", "Wr"),
    "cbam_channel": ("W1", "W2"),
}


def quantize_weights(model: Any, bits: int = 8, scheme: str = "symmetric",
                     per_channel: bool = False,
                     layer_types: Optional[Sequence[str]] = None
                     ) -> Dict[str, Any]:
    """Post-training quantization of a model's weights, in place.

    Returns a report with the mean absolute error introduced per parameter
    and the storage the integer form would take. The weights are left
    fake-quantized, so `Forward` afterwards gives exactly the accuracy the
    quantized model would have."""
    report: Dict[str, Any] = {"bits": bits, "scheme": scheme,
                              "per_channel": per_channel, "layers": {}}
    total_err, total_n, float_bytes, int_bits = 0.0, 0, 0, 0
    for i, layer in enumerate(model.layers):
        if layer_types is not None and layer["type"] not in layer_types:
            continue
        for name in QUANTIZABLE.get(layer["type"], ()):
            if name not in layer:
                continue
            original = layer[name]
            q, scale, zero_point = quantize_tensor(original, bits, scheme,
                                                   per_channel)
            err = float(np.mean(np.abs(q - original)))
            layer[name] = q
            layer.setdefault("quant", {})[name] = {
                "bits": bits, "scheme": scheme,
                "scale": scale, "zero_point": zero_point}
            report["layers"][f"{i}.{name}"] = {
                "mean_abs_error": err, "size": int(original.size)}
            total_err += err * original.size
            total_n += int(original.size)
            float_bytes += int(original.size) * original.dtype.itemsize
            int_bits += int(original.size) * bits
    report["mean_abs_error"] = total_err / total_n if total_n else 0.0
    report["parameters"] = total_n
    report["float_bytes"] = float_bytes
    report["quantized_bytes"] = int_bits // 8
    report["compression"] = (float_bytes / (int_bits / 8)) if int_bits else 1.0
    return report


class ActivationCalibrator:
    """Records the range each layer's activations actually occupy, so they
    can be quantized too (roadmap item 63's calibration pass).

    Weight-only quantization is the easy half; activations are what an
    integer kernel would also have to represent, and their range cannot be
    read off the weights -- it depends on the data. Run a few
    representative batches through :meth:`observe`, then :meth:`apply` to
    install the ranges on the model."""

    def __init__(self, model: Any, bits: int = 8, scheme: str = "asymmetric",
                 percentile: Optional[float] = None) -> None:
        self.model = model
        self.bits, self.scheme = bits, scheme
        # A percentile clips outliers, which usually costs a little on the
        # tails and buys a much finer grid everywhere else.
        self.percentile = percentile
        self.ranges: Dict[int, List[float]] = {}

    def observe(self, X: Any) -> None:
        """Run one calibration batch and widen the recorded ranges."""
        self.model.Forward(X, training=False)
        # outputs[0] is the network input; outputs[i + 1] is layer i's output.
        for i in range(len(self.model.layers)):
            out = self.model.outputs[i + 1]
            if not hasattr(out, "dtype") or out.dtype.kind != "f":
                continue
            if self.percentile is None:
                low, high = float(np.min(out)), float(np.max(out))
            else:
                p = self.percentile
                low = float(np.percentile(out, 100.0 - p))
                high = float(np.percentile(out, p))
            if i in self.ranges:
                self.ranges[i] = [min(self.ranges[i][0], low),
                                  max(self.ranges[i][1], high)]
            else:
                self.ranges[i] = [low, high]

    def apply(self) -> Dict[int, Dict[str, Any]]:
        """Install the observed ranges so ``Forward`` fake-quantizes each
        layer's output. Returns what was installed."""
        if not self.ranges:
            raise RuntimeError("no calibration data observed -- call observe(X) first")
        installed = {}
        for i, (low, high) in self.ranges.items():
            scale, zero_point = compute_scale(
                np.asarray(low), np.asarray(high), self.bits, self.scheme)
            spec = {"bits": self.bits, "scheme": self.scheme,
                    "scale": scale, "zero_point": zero_point,
                    "low": low, "high": high}
            self.model.layers[i]["act_quant"] = spec
            installed[i] = spec
        return installed


def remove_activation_quantization(model: Any) -> None:
    """Drop every activation-quantization spec, restoring float activations."""
    for layer in model.layers:
        layer.pop("act_quant", None)


def quantization_error(reference: Sequence[Any], candidate: Sequence[Any]
                       ) -> Dict[str, float]:
    """Compare two sets of model outputs. `reference` is the float model's,
    `candidate` the quantized one's."""
    a = np.concatenate([np.asarray(v).reshape(-1) for v in reference])
    b = np.concatenate([np.asarray(v).reshape(-1) for v in candidate])
    diff = b - a
    denom = float(np.sqrt(np.sum(a * a)))
    return {"mean_abs_error": float(np.mean(np.abs(diff))),
            "max_abs_error": float(np.max(np.abs(diff))),
            "relative_error": float(np.sqrt(np.sum(diff * diff)) / denom)
            if denom > 0 else 0.0}
