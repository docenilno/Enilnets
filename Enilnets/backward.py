import numpy as np
from .activations import derivative
from .forward import im2col

def maxpool2d_backward(delta, x, p):
    B, C, H, W = x.shape
    H_trim = (H // p) * p
    W_trim = (W // p) * p
    x_trim = x[:, :, :H_trim, :W_trim]

    H_b, W_b = H_trim // p, W_trim // p
    strides = x_trim.strides
    new_shape = (B, C, H_b, p, W_b, p)
    new_strides = (strides[0], strides[1], strides[2]*p, strides[2], strides[3]*p, strides[3])
    x_blocks = np.lib.stride_tricks.as_strided(x_trim, shape=new_shape, strides=new_strides)

    x_max = x_blocks.max(axis=(3, 5), keepdims=True)
    mask = (x_blocks == x_max).astype(np.float64)
    mask_sum = mask.sum(axis=(3, 5), keepdims=True)
    mask = mask / np.maximum(mask_sum, 1e-12)

    delta_expanded = delta[:, :, :H_b, :W_b][:, :, :, None, :, None]

    dx = np.zeros_like(x)
    dx_view = np.lib.stride_tricks.as_strided(dx[:, :, :H_trim, :W_trim], 
                                                shape=new_shape, strides=new_strides)
    dx_view[:] = mask * delta_expanded
    return dx

def avgpool2d_backward(delta, x, p):
    B, C, H, W = x.shape
    H_trim = (H // p) * p
    W_trim = (W // p) * p
    H_b, W_b = H_trim // p, W_trim // p

    dx = np.zeros_like(x)
    strides = dx[:, :, :H_trim, :W_trim].strides
    new_shape = (B, C, H_b, p, W_b, p)
    new_strides = (strides[0], strides[1], strides[2]*p, strides[2], strides[3]*p, strides[3])
    dx_view = np.lib.stride_tricks.as_strided(dx[:, :, :H_trim, :W_trim], 
                                               shape=new_shape, strides=new_strides)
    dx_view[:] = delta[:, :, :H_b, :W_b][:, :, :, None, :, None] / (p * p)
    return dx

def batchnorm_backward(dout, cache):
    x, x_norm, mean, var, gamma, epsilon = cache
    N = x.shape[0]
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)
    dx_norm = dout * gamma
    dvar = np.sum(dx_norm * (x - mean) * -0.5 * (var + epsilon) ** (-1.5), axis=0)
    dmean = np.sum(dx_norm * -1.0 / np.sqrt(var + epsilon), axis=0)
    dx = dx_norm / np.sqrt(var + epsilon) + dvar * 2.0 * (x - mean) / N + dmean / N
    return dx, dgamma, dbeta

def conv2d_backward_input(delta, weights, input_shape):
    B, F, out_h, out_w = delta.shape
    F, C, K, _ = weights.shape
    H, W = input_shape[2], input_shape[3]

    padded_delta = np.pad(delta, [(0, 0), (0, 0), (K - 1, K - 1), (K - 1, K - 1)], mode="constant")
    col = im2col(padded_delta, K, K)
    weights_flat = weights[:, :, ::-1, ::-1].transpose(1, 0, 2, 3).reshape(C, -1)
    grad = np.dot(col, weights_flat.T)
    grad = grad.reshape(B, H, W, C).transpose(0, 3, 1, 2)
    return grad

def Backward(self, targets=None, output_delta=None):
    if output_delta is not None:
        output_delta = np.asarray(output_delta, dtype=np.float64)
        if output_delta.ndim == 1:
            output_delta = output_delta.reshape(1, -1)
        self.deltas = [None] * len(self.layers)
        self.deltas[-1] = output_delta
    else:
        if targets is None:
            raise ValueError("targets must be provided if output_delta is not given")
        targets = np.asarray(targets, dtype=np.float64)
        if targets.ndim == 1:
            targets = targets.reshape(1, -1)
        batch_size = targets.shape[0]
        self.deltas = [None] * len(self.layers)
        out = self.outputs[-1]
        last = self.layers[-1]
        if last.get("activation") == "softmax":
            delta = (out - targets) / batch_size
        else:
            activation_input = self.pre_activations[-1] if self.pre_activations[-1] is not None else out
            delta = (out - targets) * derivative(last.get("activation", "linear"), activation_input) / batch_size
        self.deltas[-1] = delta

    for l in reversed(range(len(self.layers) - 1)):
        curr = self.layers[l]
        nxt = self.layers[l + 1]
        next_delta = self.deltas[l + 1]

        if nxt["type"] in ("dense", "sparse"):
            err = np.dot(next_delta, nxt["weights"])
        elif nxt["type"] == "flatten":
            err = next_delta.reshape(self.outputs[l + 1].shape)
        elif nxt["type"] == "conv2d":
            err = conv2d_backward_input(next_delta, nxt["weights"], self.outputs[l + 1].shape)
        elif nxt["type"] == "maxpool2d":
            err = maxpool2d_backward(next_delta, self.outputs[l + 1], nxt["p"])
        elif nxt["type"] == "avgpool2d":
            err = avgpool2d_backward(next_delta, self.outputs[l + 1], nxt["p"])
        elif nxt["type"] == "dropout":
            mask = nxt.get("mask")
            rate = nxt.get("rate", 0.0)
            if mask is None or rate == 0.0:
                err = next_delta
            else:
                err = next_delta * mask / (1.0 - rate)
        elif nxt["type"] == "batchnorm":
            flat = next_delta.reshape(self.outputs[l + 1].shape[0], -1)
            cache = self.batchnorm_cache[l + 1]
            if cache is None:
                raise ValueError("BatchNorm cache is None. Ensure Forward(training=True) was called before Backward.")
            err_flat, dgamma, dbeta = batchnorm_backward(flat, cache)
            nxt["d_gamma"] = dgamma
            nxt["d_beta"] = dbeta
            err = err_flat.reshape(self.outputs[l + 1].shape)
        else:
            err = np.zeros_like(self.outputs[l + 1])

        if curr["type"] in ("dense", "sparse", "conv2d"):
            activation_input = self.pre_activations[l+1] if self.pre_activations[l+1] is not None else self.outputs[l + 1]
            self.deltas[l] = err * derivative(curr.get("activation", "linear"), activation_input)
        else:
            self.deltas[l] = err
