import numpy as np

def activate(name, x):
    if name == "relu": return np.maximum(0, x)
    if name == "leakyrelu": return np.where(x > 0, x, 0.01 * x)
    if name == "elu": return np.where(x > 0, x, np.exp(x) - 1)
    if name == "selu":
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
    if name == "gelu": return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    if name == "swish": return x * 1.0 / (1.0 + np.exp(-x))
    if name == "sigmoid": return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    if name == "tanh": return np.tanh(x)
    if name == "softmax":
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    return x

def derivative(name, x):
    if name == "relu": return (x > 0).astype(np.float64)
    if name == "leakyrelu": return np.where(x > 0, 1.0, 0.01)
    if name == "elu": return np.where(x > 0, 1.0, np.exp(x))
    if name == "selu":
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * np.where(x > 0, 1.0, alpha * np.exp(x))
    if name == "gelu":
        cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
        return cdf + x * pdf
    if name == "swish":
        s = 1.0 / (1.0 + np.exp(-x))
        return s + x * s * (1 - s)
    if name == "sigmoid":
        s = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return s * (1 - s)
    if name == "tanh": return 1 - np.tanh(x) ** 2
    return np.ones_like(x)
