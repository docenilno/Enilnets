import numpy as np
from . import constants

def activate(name, x, alpha=None, sigmoid_clip=None):
    """Apply an activation function.

    alpha: overrides the leakyrelu negative-slope / elu alpha constant
        (defaults: constants.LEAKYRELU_ALPHA / constants.ELU_ALPHA).
    sigmoid_clip: overrides the sigmoid/softplus overflow-safe clip bound
        (default: constants.SIGMOID_CLIP).
    """
    clip = constants.SIGMOID_CLIP if sigmoid_clip is None else sigmoid_clip
    if name == "relu": return np.maximum(0, x)
    if name == "leakyrelu":
        a = constants.LEAKYRELU_ALPHA if alpha is None else alpha
        return np.where(x > 0, x, a * x)
    if name == "elu":
        a = constants.ELU_ALPHA if alpha is None else alpha
        return np.where(x > 0, x, a * (np.exp(x) - 1))
    if name == "selu":
        selu_alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * np.where(x > 0, x, selu_alpha * (np.exp(x) - 1))
    if name == "gelu": return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    if name == "swish": return x * 1.0 / (1.0 + np.exp(-x))
    if name == "mish": return x * np.tanh(np.log(1 + np.exp(x)))
    if name == "sigmoid": return 1.0 / (1.0 + np.exp(-np.clip(x, -clip, clip)))
    if name == "tanh": return np.tanh(x)
    if name == "softmax":
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    if name == "softplus": return np.log(1 + np.exp(np.clip(x, -clip, clip)))
    if name == "linear": return x
    raise ValueError(f"Unknown activation: {name!r}")

def derivative(name, x, alpha=None, sigmoid_clip=None, cached_output=None):
    """Derivative of an activation function w.r.t. its pre-activation input x.

    alpha / sigmoid_clip: same overrides as activate().
    cached_output: if the caller already has activate(name, x) on hand (e.g.
        Backward() has it cached in self.outputs), pass it here to skip
        recomputing exp/tanh for sigmoid/tanh/swish/mish/gelu. Ignored (and
        safe to omit) for activations whose derivative doesn't need it.
    """
    clip = constants.SIGMOID_CLIP if sigmoid_clip is None else sigmoid_clip
    if name == "relu": return (x > 0).astype(np.float64)
    if name == "leakyrelu":
        a = constants.LEAKYRELU_ALPHA if alpha is None else alpha
        return np.where(x > 0, 1.0, a)
    if name == "elu":
        a = constants.ELU_ALPHA if alpha is None else alpha
        return np.where(x > 0, 1.0, a * np.exp(x))
    if name == "selu":
        selu_alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * np.where(x > 0, 1.0, selu_alpha * np.exp(x))
    if name == "gelu":
        c = np.sqrt(2 / np.pi)
        g = c * (x + 0.044715 * x**3)
        t = np.tanh(g)
        g_prime = c * (1 + 3 * 0.044715 * x**2)
        return 0.5 * (1 + t) + 0.5 * x * (1 - t**2) * g_prime
    if name == "swish":
        s = 1.0 / (1.0 + np.exp(-x))
        return s + x * s * (1 - s)
    if name == "mish":
        sp = np.tanh(np.log(1 + np.exp(x)))
        return sp + x * (1.0 / (1.0 + np.exp(-x))) * (1 - sp**2)
    if name == "sigmoid":
        s = cached_output if cached_output is not None else 1.0 / (1.0 + np.exp(-np.clip(x, -clip, clip)))
        return s * (1 - s)
    if name == "tanh":
        t = cached_output if cached_output is not None else np.tanh(x)
        return 1 - t ** 2
    if name == "softplus": return 1.0 / (1.0 + np.exp(-np.clip(x, -clip, clip)))
    if name == "linear": return np.ones_like(x)
    raise ValueError(f"Unknown activation: {name!r}")
