import numpy as np
from .activations import derivative
from .forward import im2col, im2col1d, _rope_cos_sin, _rope_rotate
from . import constants

def maxpool2d_backward(delta, x, p):
    B, C, H, W = x.shape
    H_trim = (H // p) * p
    W_trim = (W // p) * p
    if H_trim == 0 or W_trim == 0:
        return np.zeros_like(x)
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
    if H_trim == 0 or W_trim == 0:
        return np.zeros_like(x)
    H_b, W_b = H_trim // p, W_trim // p

    dx = np.zeros_like(x)
    strides = dx[:, :, :H_trim, :W_trim].strides
    new_shape = (B, C, H_b, p, W_b, p)
    new_strides = (strides[0], strides[1], strides[2]*p, strides[2], strides[3]*p, strides[3])
    dx_view = np.lib.stride_tricks.as_strided(dx[:, :, :H_trim, :W_trim], 
                                               shape=new_shape, strides=new_strides)
    dx_view[:] = delta[:, :, :H_b, :W_b][:, :, :, None, :, None] / (p * p)
    return dx

def globalavgpool2d_backward(delta, x):
    B, C, H, W = x.shape
    dx = np.ones_like(x) * delta / (H * W)
    return dx

def upsample2d_backward(delta, x, scale):
    B, C, H, W = x.shape
    # x is the input to upsample with shape (B, C, H, W)
    # upsample output is (B, C, H*scale, W*scale)
    # delta has shape (B, C, H*scale, W*scale)
    # Each position (h, w) in x contributes to (h*scale+i, w*scale+j) in output,
    # i.e. dx is the block-sum of delta over each scale x scale tile.
    delta = delta[:, :, :H * scale, :W * scale]
    return delta.reshape(B, C, H, scale, W, scale).sum(axis=(3, 5))

def batchnorm_backward(dout, cache):
    x, x_norm, mean, var, gamma, epsilon, axes = cache
    if isinstance(axes, int):
        axes = (axes,)

    if x.ndim == 2:
        N = x.shape[0]
        shape = (1, -1)
    elif x.ndim == 4:
        N = x.shape[0] * x.shape[2] * x.shape[3]
        shape = (1, -1, 1, 1)
    else:
        raise ValueError(f"BatchNorm backward only supports 2D or 4D, got {x.ndim}D")

    dbeta = np.sum(dout, axis=axes)
    dgamma = np.sum(dout * x_norm, axis=axes)
    dx_norm = dout * gamma.reshape(shape)

    dvar = np.sum(dx_norm * (x - mean.reshape(shape)) * -0.5 * (var.reshape(shape) + epsilon) ** (-1.5), axis=axes)
    dmean = np.sum(dx_norm * -1.0 / np.sqrt(var.reshape(shape) + epsilon), axis=axes)

    dx = (dx_norm / np.sqrt(var.reshape(shape) + epsilon) + 
          dvar.reshape(shape) * 2.0 * (x - mean.reshape(shape)) / N + 
          dmean.reshape(shape) / N)
    return dx, dgamma, dbeta

def layernorm_backward(dout, cache):
    x, x_norm, mean, var, gamma, epsilon, axes = cache
    if isinstance(axes, int):
        axes = (axes,)

    ndim = x.ndim
    if ndim == 2:
        N = x.shape[1]
        # gamma is stored as 1D; broadcast shape for computation
        gamma_bc = gamma.reshape(1, -1)
        # dgamma/dbeta: sum over batch dimension (axis 0)
        dgamma = np.sum(dout * x_norm, axis=0)
        dbeta = np.sum(dout, axis=0)
    elif ndim == 4:
        N = x.shape[1] * x.shape[2] * x.shape[3]
        gamma_bc = gamma.reshape(1, -1, 1, 1)
        # dgamma/dbeta: sum over batch, H, W dimensions (axes 0, 2, 3)
        dgamma = np.sum(dout * x_norm, axis=(0, 2, 3))
        dbeta = np.sum(dout, axis=(0, 2, 3))
    elif ndim == 3:
        N = x.shape[2]
        gamma_bc = gamma.reshape(1, 1, -1)
        # dgamma/dbeta: sum over batch, seq_len dimensions (axes 0, 1)
        dgamma = np.sum(dout * x_norm, axis=(0, 1))
        dbeta = np.sum(dout, axis=(0, 1))
    else:
        raise ValueError(f"LayerNorm backward only supports 2D, 3D or 4D, got {ndim}D")

    dx_norm = dout * gamma_bc

    dvar = np.sum(dx_norm * (x - mean) * -0.5 * (var + epsilon) ** (-1.5), axis=axes, keepdims=True)
    dmean = np.sum(dx_norm * -1.0 / np.sqrt(var + epsilon), axis=axes, keepdims=True)

    dx = (dx_norm / np.sqrt(var + epsilon) + 
          dvar * 2.0 * (x - mean) / N + 
          dmean / N)
    return dx, dgamma, dbeta

def embedding_backward(dout, layer):
    """Sparse gradient for embedding layer."""
    x_int = np.asarray(layer.get("_last_input"), dtype=np.int32)
    if x_int.ndim == 1:
        x_int = x_int.reshape(-1, 1)
    grad = np.zeros_like(layer["weights"])
    # Scatter-add every (row, token) gradient into its vocab index in one call
    # instead of a Python double loop over (batch, seq_len).
    np.add.at(grad, x_int.reshape(-1), dout.reshape(-1, dout.shape[-1]))
    return grad

def multihead_attention_backward(dout, layer, cache):
    """Backprop through scaled dot-product multi-head self-attention.

    dout: gradient w.r.t. this layer's output, shape (B, S, E)
    cache: (Q, K, V, Qh, Kh, Vh, attn, context, x_in) saved by Forward()
    Stores per-parameter gradients on `layer` (d_Wq, d_bq, ...) and returns
    dx, the gradient w.r.t. this layer's input (accumulated across Q/K/V since
    all three share the same input).
    """
    Q, K, V, Qh, Kh, Vh, attn, context, x_in = cache
    B, S, E = dout.shape
    Dh = layer["head_dim"]

    dout_flat = dout.reshape(-1, E)
    context_flat = context.reshape(-1, E)
    d_Wo = np.dot(dout_flat.T, context_flat)
    d_bo = np.sum(dout_flat, axis=0)
    dcontext = np.dot(dout, layer["Wo"])

    dcontext_h = dcontext.reshape(B, S, -1, Dh).transpose(0, 2, 1, 3)
    dVh = np.matmul(attn.transpose(0, 1, 3, 2), dcontext_h)
    dattn = np.matmul(dcontext_h, Vh.transpose(0, 1, 3, 2))

    # softmax Jacobian-vector product
    dscores = attn * (dattn - np.sum(dattn * attn, axis=-1, keepdims=True))
    dscores = dscores / np.sqrt(Dh)

    # Note: Qh/Kh here are whatever was actually fed into the score matmul in
    # Forward() -- i.e. already RoPE-rotated if positional_scheme="rope" --
    # so dQh/dKh below are gradients w.r.t. the ROTATED tensors.
    dQh = np.matmul(dscores, Kh)
    dKh = np.matmul(dscores.transpose(0, 1, 3, 2), Qh)

    if layer.get("positional_scheme") == "rope":
        # Un-rotate before using these for d_Wq/d_Wk/dx: Q/K (pre-rotation)
        # are what the Wq/Wk projections actually produced. Rotation is
        # orthogonal (a per-pair 2D rotation), so its inverse is the same
        # rotation with the angle negated (cos(-t)=cos(t), sin(-t)=-sin(t)).
        rope_cos, rope_sin = _rope_cos_sin(S, Dh, constants.SINUSOIDAL_BASE)
        dQh = _rope_rotate(dQh, rope_cos, -rope_sin)
        dKh = _rope_rotate(dKh, rope_cos, -rope_sin)

    dQ = dQh.transpose(0, 2, 1, 3).reshape(B, S, E)
    dK = dKh.transpose(0, 2, 1, 3).reshape(B, S, E)
    dV = dVh.transpose(0, 2, 1, 3).reshape(B, S, E)

    x_in_flat = x_in.reshape(-1, E)
    d_Wq = np.dot(dQ.reshape(-1, E).T, x_in_flat)
    d_bq = np.sum(dQ.reshape(-1, E), axis=0)
    d_Wk = np.dot(dK.reshape(-1, E).T, x_in_flat)
    d_bk = np.sum(dK.reshape(-1, E), axis=0)
    d_Wv = np.dot(dV.reshape(-1, E).T, x_in_flat)
    d_bv = np.sum(dV.reshape(-1, E), axis=0)

    layer["d_Wq"], layer["d_bq"] = d_Wq, d_bq
    layer["d_Wk"], layer["d_bk"] = d_Wk, d_bk
    layer["d_Wv"], layer["d_bv"] = d_Wv, d_bv
    layer["d_Wo"], layer["d_bo"] = d_Wo, d_bo

    dx = np.dot(dQ, layer["Wq"]) + np.dot(dK, layer["Wk"]) + np.dot(dV, layer["Wv"])
    return dx

def cross_attention_backward(dout, layer, cache):
    """Backprop through cross-attention (Q from the sequential input, K/V
    from an earlier layer's output). Adapted from
    multihead_attention_backward -- the math is identical except Q's and
    K/V's sources differ, which matters in exactly two places: `dQ` flows
    normally into the caller's `dx` (returned as the first element, same
    as self-attention), while `dK`/`dV`'s COMBINED contribution has ZERO
    path back through `x_in` at all -- it must be deferred entirely to the
    KV-source layer (`d_kv`, returned as the second element; the caller
    routes it via `_defer_grad(layer["kv_source_index"], d_kv)`, no -1
    offset). `Wq`'s param gradient uses `x_in`; `Wk`/`Wv`'s use `kv_source`
    -- the other correctness-critical divergence from self-attention.

    dout: gradient w.r.t. this layer's output, shape (B, Sq, E)
    cache: (Q, K, V, Qh, Kh, Vh, attn, context, x_in, kv_source)
    Returns (dx, d_kv).
    """
    Q, K, V, Qh, Kh, Vh, attn, context, x_in, kv_source = cache
    B, Sq, E = dout.shape
    Skv = kv_source.shape[1]
    Dh = layer["head_dim"]

    dout_flat = dout.reshape(-1, E)
    context_flat = context.reshape(-1, E)
    d_Wo = np.dot(dout_flat.T, context_flat)
    d_bo = np.sum(dout_flat, axis=0)
    dcontext = np.dot(dout, layer["Wo"])

    dcontext_h = dcontext.reshape(B, Sq, -1, Dh).transpose(0, 2, 1, 3)
    dVh = np.matmul(attn.transpose(0, 1, 3, 2), dcontext_h)
    dattn = np.matmul(dcontext_h, Vh.transpose(0, 1, 3, 2))

    # softmax Jacobian-vector product (over the Skv axis, the last axis of attn)
    dscores = attn * (dattn - np.sum(dattn * attn, axis=-1, keepdims=True))
    dscores = dscores / np.sqrt(Dh)

    dQh = np.matmul(dscores, Kh)
    dKh = np.matmul(dscores.transpose(0, 1, 3, 2), Qh)

    dQ = dQh.transpose(0, 2, 1, 3).reshape(B, Sq, E)
    dK = dKh.transpose(0, 2, 1, 3).reshape(B, Skv, E)
    dV = dVh.transpose(0, 2, 1, 3).reshape(B, Skv, E)

    x_in_flat = x_in.reshape(-1, E)
    kv_source_flat = kv_source.reshape(-1, E)
    d_Wq = np.dot(dQ.reshape(-1, E).T, x_in_flat)
    d_bq = np.sum(dQ.reshape(-1, E), axis=0)
    d_Wk = np.dot(dK.reshape(-1, E).T, kv_source_flat)
    d_bk = np.sum(dK.reshape(-1, E), axis=0)
    d_Wv = np.dot(dV.reshape(-1, E).T, kv_source_flat)
    d_bv = np.sum(dV.reshape(-1, E), axis=0)

    layer["d_Wq"], layer["d_bq"] = d_Wq, d_bq
    layer["d_Wk"], layer["d_bk"] = d_Wk, d_bk
    layer["d_Wv"], layer["d_bv"] = d_Wv, d_bv
    layer["d_Wo"], layer["d_bo"] = d_Wo, d_bo

    dx = np.dot(dQ, layer["Wq"])
    d_kv = np.dot(dK, layer["Wk"]) + np.dot(dV, layer["Wv"])
    return dx, d_kv

def rnn_backward(dout, layer, cache):
    """BPTT for a vanilla tanh RNN. dout is the gradient w.r.t. this layer's
    output: (B,S,hidden) if return_sequences else (B,hidden) (applies only at
    the last timestep). Stores d_Wx/d_Wh/d_b on layer; returns dx (B,S,n_in)."""
    x, h_all = cache["x"], cache["h_all"]
    B, S, n_in = x.shape
    H = layer["hidden_dim"]
    Wx, Wh = layer["Wx"], layer["Wh"]
    return_sequences = layer["return_sequences"]

    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros_like(layer["b"])
    dx = np.zeros((B, S, n_in), dtype=np.float64)
    dh_next = np.zeros((B, H), dtype=np.float64)

    for t in reversed(range(S)):
        dh = (dout[:, t, :] if return_sequences else (dout if t == S - 1 else 0.0)) + dh_next
        h_t = h_all[:, t + 1, :]
        h_prev = h_all[:, t, :]
        dz = dh * (1 - h_t ** 2)
        dWx += np.dot(dz.T, x[:, t, :])
        dWh += np.dot(dz.T, h_prev)
        db += np.sum(dz, axis=0)
        dx[:, t, :] = np.dot(dz, Wx)
        dh_next = np.dot(dz, Wh)

    layer["d_Wx"], layer["d_Wh"], layer["d_b"] = dWx, dWh, db
    return dx

def lstm_backward(dout, layer, cache):
    """BPTT for LSTM (gates stacked in [i,f,g,o] order). dout is the gradient
    w.r.t. output: (B,S,hidden) if return_sequences else (B,hidden). Stores
    d_Wx/d_Wh/d_b on layer; returns dx (B,S,n_in)."""
    x, h_all, c_all = cache["x"], cache["h_all"], cache["c_all"]
    i_all, f_all, g_all, o_all = cache["i"], cache["f"], cache["g"], cache["o"]
    tanh_c_all = cache["tanh_c"]
    B, S, n_in = x.shape
    H = layer["hidden_dim"]
    Wx, Wh = layer["Wx"], layer["Wh"]
    return_sequences = layer["return_sequences"]

    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros_like(layer["b"])
    dx = np.zeros((B, S, n_in), dtype=np.float64)
    dh_next = np.zeros((B, H), dtype=np.float64)
    dc_next = np.zeros((B, H), dtype=np.float64)

    for t in reversed(range(S)):
        dh = (dout[:, t, :] if return_sequences else (dout if t == S - 1 else 0.0)) + dh_next
        i_t, f_t, g_t, o_t = i_all[:, t, :], f_all[:, t, :], g_all[:, t, :], o_all[:, t, :]
        tanh_c = tanh_c_all[:, t, :]
        c_prev = c_all[:, t, :]
        h_prev = h_all[:, t, :]

        dc = dc_next + dh * o_t * (1 - tanh_c ** 2)
        do = dh * tanh_c
        di = dc * g_t
        df = dc * c_prev
        dg = dc * i_t

        dgate_i = di * i_t * (1 - i_t)
        dgate_f = df * f_t * (1 - f_t)
        dgate_g = dg * (1 - g_t ** 2)
        dgate_o = do * o_t * (1 - o_t)

        dgates = np.concatenate([dgate_i, dgate_f, dgate_g, dgate_o], axis=-1)

        dWx += np.dot(dgates.T, x[:, t, :])
        dWh += np.dot(dgates.T, h_prev)
        db += np.sum(dgates, axis=0)
        dx[:, t, :] = np.dot(dgates, Wx)
        dh_next = np.dot(dgates, Wh)
        dc_next = dc * f_t

    layer["d_Wx"], layer["d_Wh"], layer["d_b"] = dWx, dWh, db
    return dx

def gru_backward(dout, layer, cache):
    """BPTT for GRU (gates stacked in [r,z,n] order, separate input/hidden
    biases since the reset gate only multiplies the hidden contribution to
    the candidate gate). dout is the gradient w.r.t. output: (B,S,hidden) if
    return_sequences else (B,hidden). Stores d_Wx/d_Wh/d_bx/d_bh on layer;
    returns dx (B,S,n_in)."""
    x, h_all = cache["x"], cache["h_all"]
    r_all, z_all, n_all, gh_n_all = cache["r"], cache["z"], cache["n"], cache["gh_n"]
    B, S, n_in = x.shape
    H = layer["hidden_dim"]
    Wx, Wh = layer["Wx"], layer["Wh"]
    return_sequences = layer["return_sequences"]

    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    dbx = np.zeros_like(layer["bx"])
    dbh = np.zeros_like(layer["bh"])
    dx = np.zeros((B, S, n_in), dtype=np.float64)
    dh_next = np.zeros((B, H), dtype=np.float64)

    for t in reversed(range(S)):
        dh = (dout[:, t, :] if return_sequences else (dout if t == S - 1 else 0.0)) + dh_next
        r_t, z_t, n_t = r_all[:, t, :], z_all[:, t, :], n_all[:, t, :]
        gh_n = gh_n_all[:, t, :]
        h_prev = h_all[:, t, :]

        dz_postact = dh * (h_prev - n_t)
        dn = dh * (1 - z_t)
        dh_prev_direct = dh * z_t

        dn_pre = dn * (1 - n_t ** 2)
        dr_postact = dn_pre * gh_n
        dgh_n = dn_pre * r_t

        dz_pre = dz_postact * z_t * (1 - z_t)
        dr_pre = dr_postact * r_t * (1 - r_t)

        dgi = np.concatenate([dr_pre, dz_pre, dn_pre], axis=-1)
        dgh = np.concatenate([dr_pre, dz_pre, dgh_n], axis=-1)

        dWx += np.dot(dgi.T, x[:, t, :])
        dbx += np.sum(dgi, axis=0)
        dWh += np.dot(dgh.T, h_prev)
        dbh += np.sum(dgh, axis=0)

        dx[:, t, :] = np.dot(dgi, Wx)
        dh_next = np.dot(dgh, Wh) + dh_prev_direct

    layer["d_Wx"], layer["d_Wh"], layer["d_bx"], layer["d_bh"] = dWx, dWh, dbx, dbh
    return dx

def conv2d_backward_input(delta, weights, input_shape, stride=1, pad=0):
    B, F, out_h, out_w = delta.shape
    F, C, K, _ = weights.shape
    H, W = input_shape[2], input_shape[3]

    if stride > 1:
        # Dilate delta: insert (stride-1) zeros between spatial elements so
        # the stride-1 transposed-convolution below lands on the right pixels.
        dilated_h = (out_h - 1) * stride + 1
        dilated_w = (out_w - 1) * stride + 1
        dilated = np.zeros((B, F, dilated_h, dilated_w), dtype=delta.dtype)
        dilated[:, :, ::stride, ::stride] = delta
    else:
        dilated = delta

    padded_delta = np.pad(dilated, [(0, 0), (0, 0), (K - 1, K - 1), (K - 1, K - 1)], mode="constant")
    col = im2col(padded_delta, K, K)
    weights_flat = weights[:, :, ::-1, ::-1].transpose(1, 0, 2, 3).reshape(C, -1)
    grad = np.dot(col, weights_flat.T)
    out_h2 = padded_delta.shape[2] - K + 1
    out_w2 = padded_delta.shape[3] - K + 1
    grad = grad.reshape(B, out_h2, out_w2, C).transpose(0, 3, 1, 2)
    # The transposed-conv above computes the gradient w.r.t. the *padded*
    # input (spatial size H+2*pad); slice out the pad-sized border to get
    # the gradient w.r.t. the original unpadded input. pad=0 (default,
    # "valid" padding) makes this a no-op slice, byte-identical to before.
    return grad[:, :, pad:pad + H, pad:pad + W]

def conv1d_backward_input(delta, weights, input_shape, stride=1, pad=0):
    B, F, out_l = delta.shape
    F, C, K = weights.shape
    L = input_shape[2]

    if stride > 1:
        dilated_l = (out_l - 1) * stride + 1
        dilated = np.zeros((B, F, dilated_l), dtype=delta.dtype)
        dilated[:, :, ::stride] = delta
    else:
        dilated = delta

    padded_delta = np.pad(dilated, [(0, 0), (0, 0), (K - 1, K - 1)], mode="constant")
    col = im2col1d(padded_delta, K)
    weights_flat = weights[:, :, ::-1].transpose(1, 0, 2).reshape(C, -1)
    grad = np.dot(col, weights_flat.T)
    out_l2 = padded_delta.shape[2] - K + 1
    grad = grad.reshape(B, out_l2, C).transpose(0, 2, 1)
    return grad[:, :, pad:pad + L]

def _loss_divisor(o, batch_size, loss_function, loss_kwargs):
    """Matches loss.py's ComputeLoss reduction convention: losses whose forward
    formula already reduces over the feature axis (cross_entropy sums over
    classes then divides by batch_size; cosine_similarity/triplet/ntxent sum
    over the feature axis into one scalar per example) only divide by
    batch_size under mean reduction. Purely elementwise losses (mse, mae,
    bce, focal, hinge, bce_logits, wasserstein, ...) fall through to
    ComputeLoss's generic `np.mean(loss)`, which averages over every element
    (batch * features), so their gradient must divide by o.size, not just
    batch_size, to stay consistent with the reported loss value."""
    reduction = loss_kwargs.get("reduction", "mean")
    if reduction == "sum":
        return 1.0
    feature_reduced = loss_function in (
        "cross_entropy", "categorical_cross_entropy", "sparse_cross_entropy",
        "cosine_similarity", "triplet", "ntxent",
    )
    return float(batch_size) if feature_reduced else float(o.size)

def _loss_output_delta(o, t, z, activation, loss_function, batch_size, loss_kwargs, activation_params=None):
    """Compute dL/dz (gradient w.r.t. the output layer's pre-activation) for
    whichever loss_function was actually used to train, matching loss.py's
    ComputeLoss formulas. Where the loss+activation combination has a canonical
    simplified form (softmax+CE, sigmoid+BCE, linear+bce_logits/wasserstein),
    that form is used directly instead of chaining through derivative()."""
    activation_params = activation_params or {}
    divisor = _loss_divisor(o, batch_size, loss_function, loss_kwargs)
    sigmoid_clip = activation_params.get("sigmoid_clip", constants.SIGMOID_CLIP)

    if activation == "softmax" and loss_function in (None, "cross_entropy", "categorical_cross_entropy"):
        return (o - t) / divisor
    if activation == "softmax" and loss_function == "sparse_cross_entropy":
        idx = t.astype(np.int64)
        onehot = np.zeros_like(o)
        np.put_along_axis(onehot, idx[..., None], 1.0, axis=-1)
        return (o - onehot) / divisor
    if activation == "sigmoid" and loss_function == "binary_cross_entropy":
        return (o - t) / divisor
    if loss_function == "bce_logits":
        sig = 1.0 / (1.0 + np.exp(-np.clip(z, -sigmoid_clip, sigmoid_clip)))
        return (sig - t) / divisor
    if loss_function == "wasserstein":
        return -t / divisor
    if loss_function == "kl_divergence":
        raise ValueError(
            "kl_divergence is a mu/logvar loss, not an output-layer loss -- "
            "compute its gradient manually (see generative/vae.py) rather than via Backward()."
        )

    eps = loss_kwargs.get("eps", constants.EPS_LOG)
    eps_div = loss_kwargs.get("eps_div", constants.EPS_DIV)
    if loss_function is None or loss_function == "mse":
        dL_do = 2 * (o - t)
    elif loss_function == "mae":
        dL_do = np.sign(o - t)
    elif loss_function == "huber":
        delta_h = loss_kwargs.get("delta", 1.0)
        diff = o - t
        dL_do = np.where(np.abs(diff) < delta_h, diff, delta_h * np.sign(diff))
    elif loss_function == "smooth_l1":
        beta = loss_kwargs.get("beta", 1.0)
        diff = o - t
        dL_do = np.where(np.abs(diff) < beta, diff / beta, np.sign(diff))
    elif loss_function == "binary_cross_entropy":
        oc = np.clip(o, eps, 1 - eps)
        dL_do = -(t / oc) + (1 - t) / (1 - oc)
    elif loss_function in ("cross_entropy", "categorical_cross_entropy"):
        oc = np.clip(o, eps, 1.0)
        dL_do = -t / oc
    elif loss_function == "sparse_cross_entropy":
        oc = np.clip(o, eps, 1.0)
        idx = t.astype(np.int64)
        picked = np.take_along_axis(oc, idx[..., None], axis=-1)
        dL_do = np.zeros_like(o)
        np.put_along_axis(dL_do, idx[..., None], -1.0 / picked, axis=-1)
    elif loss_function == "focal":
        alpha = loss_kwargs.get("alpha", 0.25)
        gamma = loss_kwargs.get("gamma", 2.0)
        oc = np.clip(o, eps, 1.0 - eps)
        pt = oc * t + (1 - oc) * (1 - t)
        q = 1 - pt  # (1-pt) == o when t==0, == (1-o) when t==1
        dq_do = 1 - 2 * t
        A = q ** gamma
        dA_do = gamma * (q ** (gamma - 1)) * dq_do
        B = alpha * t * np.log(oc) + (1 - alpha) * (1 - t) * np.log(1 - oc)
        dB_do = alpha * t / oc - (1 - alpha) * (1 - t) / (1 - oc)
        dL_do = -(dA_do * B + A * dB_do)
    elif loss_function == "hinge":
        active = (t * o) < 1
        dL_do = np.where(active, -t, 0.0)
    elif loss_function == "cosine_similarity":
        norm_o = np.linalg.norm(o, axis=-1, keepdims=True) + eps_div
        norm_t = np.linalg.norm(t, axis=-1, keepdims=True) + eps_div
        o_norm = o / norm_o
        t_norm = t / norm_t
        cos = np.sum(o_norm * t_norm, axis=-1, keepdims=True)
        dL_do = (cos * o_norm - t_norm) / norm_o
    elif loss_function == "triplet":
        margin = loss_kwargs.get("margin", 1.0)
        neg = loss_kwargs.get("negative")
        if neg is None:
            raise ValueError("triplet loss requires 'negative' kwarg")
        d_pos = np.sum((o - t) ** 2, axis=-1, keepdims=True)
        d_neg = np.sum((o - neg) ** 2, axis=-1, keepdims=True)
        active = (d_pos - d_neg + margin) > 0
        dL_do = np.where(active, 2 * (o - t) - 2 * (o - neg), 0.0)
    elif loss_function == "ntxent":
        temperature = loss_kwargs.get("temperature", 0.5)
        norm_o = np.linalg.norm(o, axis=-1, keepdims=True) + eps_div
        norm_t = np.linalg.norm(t, axis=-1, keepdims=True) + eps_div
        o_norm = o / norm_o
        t_norm = t / norm_t
        sim = np.dot(o_norm, t_norm.T) / temperature
        sim_max = np.max(sim, axis=-1, keepdims=True)
        exp_sim = np.exp(sim - sim_max)
        probs = exp_sim / np.sum(exp_sim, axis=-1, keepdims=True)
        n = o.shape[0]
        # row-softmax cross-entropy gradient w.r.t. the similarity row, matching
        # loss.py's row-wise (o vs t) formulation of ntxent.
        dsim = (probs - np.eye(n)) / temperature
        weighted_t = np.dot(dsim, t_norm)
        proj = np.sum(weighted_t * o_norm, axis=-1, keepdims=True)
        dL_do = (weighted_t - proj * o_norm) / norm_o
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")

    return dL_do * derivative(activation, z, cached_output=o, **activation_params) / divisor

def Backward(self, targets=None, output_delta=None, loss_function=None, **loss_kwargs):
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
        if loss_function == "sparse_cross_entropy":
            # targets are integer class indices shaped like out.shape[:-1]
            # (e.g. (B,) or (B,S)) -- not a one-hot array, so the generic
            # "1D -> (1, features)" reshape below doesn't apply here.
            batch_size = int(np.prod(targets.shape))
        else:
            if targets.ndim == 1:
                targets = targets.reshape(1, -1)
            batch_size = int(np.prod(targets.shape[:-1]))
        self.deltas = [None] * len(self.layers)
        out = self.outputs[-1]
        last = self.layers[-1]
        activation = last.get("activation", "linear")
        activation_input = self.pre_activations[-1] if self.pre_activations[-1] is not None else out
        delta = _loss_output_delta(out, targets, activation_input, activation, loss_function, batch_size,
                                   loss_kwargs, activation_params=last.get("activation_params", {}))
        self.deltas[-1] = delta

    # Gradient normally flows strictly sequentially (deltas[l] built only from
    # deltas[l+1]), but some layer types send their incoming gradient to
    # ADDITIONAL places beyond the normal chain: residual_add sends a copy
    # directly to the layer *before* the matching residual_save (the shared
    # tensor x0 that both the block and the skip path started from); goto/
    # concat_at (multi-source layers -- used by bidirectional RNN and
    # cross-attention) send their ENTIRE incoming gradient elsewhere
    # instead of continuing through the normal chain at all (their own
    # sequential predecessor gets zero gradient, since forward() discarded
    # its value entirely -- see the default `else: err = zeros` branch
    # below, which already produces this for free). deferred_grad holds
    # these extra contributions until the loop reaches their target index.
    deferred_grad = [None] * len(self.layers)

    def _defer_grad(target_index, grad):
        """Add `grad` into deferred_grad[target_index] (creating it on
        first use). Shared primitive for residual/goto/concat_at -- all
        three ultimately just need "route this gradient to some other
        layer's delta slot, additively.\""""
        if deferred_grad[target_index] is None:
            deferred_grad[target_index] = grad.copy()
        else:
            deferred_grad[target_index] = deferred_grad[target_index] + grad

    def _record_residual_grad(residual_add_layer, grad):
        # residual's save_index points AT the residual_save marker layer
        # itself, requiring a -1 adjustment to reach the shared tensor's
        # actual producing layer. This is the ONLY one of these deferral
        # helpers with that offset -- goto/concat_at name "the layer whose
        # output is referenced" directly (no -1), see their own docstrings.
        save_index = residual_add_layer["save_index"]
        if save_index == 0:
            return  # skip connects back to the raw network input; not tracked
        _defer_grad(save_index - 1, grad)

    def _record_goto_grad(goto_layer, grad):
        # No -1 offset: stored_index directly names the layer whose output
        # was jumped to (self.outputs[stored_index + 1] in Forward()).
        # Guard stored_index < 0 (e.g. "jump to the raw network input,
        # there's no preceding layer yet") explicitly: Python's negative
        # indexing would otherwise silently wrap to deferred_grad[-1] (the
        # LAST layer) instead of correctly doing nothing -- the same class
        # of silent-corruption risk residual's `save_index == 0` guard
        # exists to prevent.
        stored_index = goto_layer["stored_index"]
        if stored_index < 0:
            return
        _defer_grad(stored_index, grad)

    def _record_concat_at_grad(concat_layer, grad):
        # Split the incoming gradient at the recorded channel boundary
        # (idx_a's output width) and defer each half to its source layer,
        # no -1 offset (same direct convention as goto). Same negative-index
        # guard as goto, per-source.
        idx_a, idx_b = concat_layer["idx_a"], concat_layer["idx_b"]
        width_a = self.outputs[idx_a + 1].shape[-1]
        if idx_a >= 0:
            _defer_grad(idx_a, grad[..., :width_a])
        if idx_b >= 0:
            _defer_grad(idx_b, grad[..., width_a:])

    def _apply_deferral(layer, grad):
        t = layer["type"]
        if t == "residual_add":
            _record_residual_grad(layer, grad)
        elif t == "goto":
            _record_goto_grad(layer, grad)
        elif t == "concat_at":
            _record_concat_at_grad(layer, grad)

    # Note: cross_attention deliberately has NO special "last layer" case
    # here (unlike residual_add/goto/concat_at below). Its KV-source
    # deferral is driven by being `nxt` in the main loop, which already
    # covers cross_attention as the network's very last layer too -- the
    # main loop's l = len(self.layers) - 2 iteration always runs (as long
    # as there are >= 2 layers) and treats nxt = self.layers[-1] normally
    # regardless of whether -1 is "actually" the final index. Adding a
    # special case here as well double-counts the deferral (found via FD
    # verification: layer1's gradient came out exactly 2x too large).
    if self.layers:
        _apply_deferral(self.layers[-1], self.deltas[-1])

    for l in reversed(range(len(self.layers) - 1)):
        curr = self.layers[l]
        nxt = self.layers[l + 1]
        next_delta = self.deltas[l + 1]

        if nxt["type"] in ("dense", "sparse"):
            err = np.dot(next_delta, nxt["weights"])
        elif nxt["type"] == "flatten":
            err = next_delta.reshape(self.outputs[l + 1].shape)
        elif nxt["type"] == "conv2d":
            err = conv2d_backward_input(next_delta, nxt["weights"], self.outputs[l + 1].shape,
                                        stride=nxt.get("stride", 1), pad=nxt.get("pad", 0))
        elif nxt["type"] == "conv1d":
            err = conv1d_backward_input(next_delta, nxt["weights"], self.outputs[l + 1].shape,
                                        stride=nxt.get("stride", 1), pad=nxt.get("pad", 0))
        elif nxt["type"] == "maxpool2d":
            err = maxpool2d_backward(next_delta, self.outputs[l + 1], nxt["p"])
        elif nxt["type"] == "avgpool2d":
            err = avgpool2d_backward(next_delta, self.outputs[l + 1], nxt["p"])
        elif nxt["type"] == "globalavgpool2d":
            err = globalavgpool2d_backward(next_delta, self.outputs[l + 1])
        elif nxt["type"] == "upsample2d":
            err = upsample2d_backward(next_delta, self.outputs[l + 1], nxt.get("scale_factor", 2))
        elif nxt["type"] == "dropout":
            mask = nxt.get("mask")
            rate = nxt.get("rate", 0.0)
            if mask is None or rate == 0.0:
                err = next_delta
            else:
                err = next_delta * mask / (1.0 - rate)
        elif nxt["type"] == "batchnorm":
            cache = self.batchnorm_cache[l + 1]
            if cache is None:
                raise ValueError("BatchNorm cache is None. Ensure Forward(training=True) was called before Backward.")
            err, dgamma, dbeta = batchnorm_backward(next_delta, cache)
            nxt["d_gamma"] = dgamma
            nxt["d_beta"] = dbeta
            err = err
        elif nxt["type"] == "layernorm":
            cache = self.layernorm_cache[l + 1]
            if cache is None:
                raise ValueError("LayerNorm cache is None. Ensure Forward(training=True) was called before Backward.")
            err, dgamma, dbeta = layernorm_backward(next_delta, cache)
            nxt["d_gamma"] = dgamma
            nxt["d_beta"] = dbeta
            err = err
        elif nxt["type"] == "embedding":
            err = next_delta  # gradient flows back to previous layer if any
        elif nxt["type"] == "multihead_attention":
            cache = self.attention_cache[l + 1]
            if cache is None:
                raise ValueError("Attention cache is None. Ensure Forward(training=True) was called before Backward.")
            err = multihead_attention_backward(next_delta, nxt, cache)
        elif nxt["type"] == "cross_attention":
            cache = self.attention_cache[l + 1]
            if cache is None:
                raise ValueError("Cross-attention cache is None. Ensure Forward(training=True) was called before Backward.")
            err, d_kv = cross_attention_backward(next_delta, nxt, cache)
            # dK/dV's combined contribution has zero path back through the
            # normal sequential chain (`err`, above, is Q's path only) --
            # defer it entirely to the KV-source layer, no -1 offset.
            _defer_grad(nxt["kv_source_index"], d_kv)
        elif nxt["type"] == "positional_encoding":
            err = next_delta
        elif nxt["type"] == "rnn":
            err = rnn_backward(next_delta, nxt, self.rnn_cache[l + 1])
        elif nxt["type"] == "lstm":
            err = lstm_backward(next_delta, nxt, self.rnn_cache[l + 1])
        elif nxt["type"] == "gru":
            err = gru_backward(next_delta, nxt, self.rnn_cache[l + 1])
        elif nxt["type"] in ("residual_save", "residual_add"):
            err = next_delta  # both are identity w.r.t. their own input
        elif nxt["type"] == "reverse_sequence":
            err = next_delta[:, ::-1, :]  # self-inverse, same op as forward
        else:
            # Also correctly covers "goto"/"concat_at" as `nxt`: their
            # forward() discarded whatever x was at this point entirely
            # (replaced it with a jumped-to/concatenated value), so this
            # layer's own sequential output has zero influence on nxt's
            # output -- err is correctly zero, no special-casing needed.
            err = np.zeros_like(self.outputs[l + 1])

        if curr["type"] in ("dense", "sparse", "conv2d", "conv1d"):
            activation_input = self.pre_activations[l+1] if self.pre_activations[l+1] is not None else self.outputs[l + 1]
            self.deltas[l] = err * derivative(curr.get("activation", "linear"), activation_input,
                                              cached_output=self.outputs[l + 1],
                                              **curr.get("activation_params", {}))
        else:
            self.deltas[l] = err

        # Send this layer's gradient to wherever it needs to be deferred to
        # (residual_add/goto/concat_at -- see _apply_deferral above; a
        # no-op for every other layer type).
        _apply_deferral(curr, self.deltas[l])
        if deferred_grad[l] is not None:
            # deferred_grad[l] is always a gradient w.r.t. this layer's
            # ACTIVATED output (that's what residual_add/goto/concat_at
            # actually route -- e.g. goto's stored_index names the layer
            # whose *output* value was jumped to). For dense/conv/conv1d
            # layers, self.deltas[l] itself is dL/dz (PRE-activation, needed
            # for that layer's own weight gradient) -- so the deferred
            # contribution needs the same activation-derivative conversion
            # applied to it before being added, or it's added in the wrong
            # units. Matters whenever a residual/goto/concat_at target has a
            # non-linear activation (e.g. a custom ResNet-style block whose
            # skip connection starts at a `tanh` dense layer, not just the
            # `linear` targets every in-repo transformer block happens to use).
            if curr["type"] in ("dense", "sparse", "conv2d", "conv1d"):
                activation_input = self.pre_activations[l+1] if self.pre_activations[l+1] is not None else self.outputs[l + 1]
                deferred_contribution = deferred_grad[l] * derivative(
                    curr.get("activation", "linear"), activation_input,
                    cached_output=self.outputs[l + 1], **curr.get("activation_params", {})
                )
            else:
                deferred_contribution = deferred_grad[l]
            self.deltas[l] = self.deltas[l] + deferred_contribution
