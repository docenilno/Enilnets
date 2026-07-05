import numpy as np
from .activations import activate

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant')

    N_stride, C_stride, H_stride, W_stride = img.strides
    shape = (N, C, filter_h, filter_w, out_h, out_w)
    strides = (N_stride, C_stride, H_stride, W_stride, H_stride * stride, W_stride * stride)

    col = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    return col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

def batchnorm_forward(x, layer, training):
    epsilon = layer.get("epsilon", 1e-5)
    momentum = layer.get("momentum", 0.1)
    if training:
        mean = np.mean(x, axis=0)
        variance = np.var(x, axis=0)
        x_norm = (x - mean) / np.sqrt(variance + epsilon)
        out = layer["gamma"] * x_norm + layer["beta"]
        layer["running_mean"] = (1 - momentum) * layer["running_mean"] + momentum * mean
        layer["running_var"] = (1 - momentum) * layer["running_var"] + momentum * variance
        cache = (x, x_norm, mean, variance, layer["gamma"], epsilon)
    else:
        x_norm = (x - layer["running_mean"]) / np.sqrt(layer["running_var"] + epsilon)
        out = layer["gamma"] * x_norm + layer["beta"]
        cache = None
    return out, cache

def Forward(self, inputs, training=False, dropout_rate=0.0):
    x = np.asarray(inputs, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim == 3:
        x = x.reshape(1, *x.shape)

    self.outputs = [x]
    self.pre_activations = [None]
    self.batchnorm_cache = []

    for layer in self.layers:
        x = self.outputs[-1]
        if layer["type"] in ("dense", "sparse"):
            z = np.dot(x, layer["weights"].T) + layer["bias"]
            x = activate(layer["activation"], z)
            self.pre_activations.append(z)
            self.batchnorm_cache.append(None)
        elif layer["type"] == "conv2d":
            B, C, H, W = x.shape
            F, _, K, _ = layer["weights"].shape
            out_h, out_w = H - K + 1, W - K + 1
            col = im2col(x, K, K)
            weights_flat = layer["weights"].reshape(F, -1)
            out = np.dot(col, weights_flat.T).reshape(B, out_h, out_w, F).transpose(0, 3, 1, 2)
            z = out + layer["bias"][None, :, None, None]
            x = activate(layer["activation"], z)
            self.pre_activations.append(z)
            self.batchnorm_cache.append(None)
        elif layer["type"] == "flatten":
            x = x.reshape(x.shape[0], -1)
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
        elif layer["type"] == "maxpool2d":
            B, C, H, W, p = *x.shape, layer["p"]
            x = x[:, :, : H // p * p, : W // p * p].reshape(B, C, H // p, p, W // p, p).max(axis=(3, 5))
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
        elif layer["type"] == "avgpool2d":
            B, C, H, W, p = *x.shape, layer["p"]
            x = x[:, :, : H // p * p, : W // p * p].reshape(B, C, H // p, p, W // p, p).mean(axis=(3, 5))
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
        elif layer["type"] == "batchnorm":
            flat = x.reshape(x.shape[0], -1)
            normalized, cache = batchnorm_forward(flat, layer, training)
            x = normalized.reshape(x.shape)
            self.pre_activations.append(None)
            self.batchnorm_cache.append(cache)
        elif layer["type"] == "dropout":
            rate = layer.get("rate", dropout_rate)
            if training and rate > 0:
                if rate >= 1.0:
                    mask = np.zeros_like(x, dtype=np.float64)
                    x = np.zeros_like(x)
                else:
                    mask = (np.random.rand(*x.shape) > rate).astype(np.float64)
                    x = x * mask / (1.0 - rate)
                layer["mask"] = mask
            else:
                layer["mask"] = None
                x = x
            self.pre_activations.append(None)
            self.batchnorm_cache.append(None)
        else:
            raise ValueError(f"Unknown layer type: {layer['type']}")
        self.outputs.append(x)
    return self.outputs[-1]
