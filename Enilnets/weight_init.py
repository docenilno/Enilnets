import numpy as np

def init_weights(n_in, n_out, method="xavier_uniform"):
    if method == "xavier_uniform":
        limit = np.sqrt(6 / (n_in + n_out))
        w = np.random.uniform(-limit, limit, (n_out, n_in)).astype(np.float64)
    elif method == "xavier_normal":
        std = np.sqrt(2 / (n_in + n_out))
        w = np.random.normal(0, std, (n_out, n_in)).astype(np.float64)
    elif method == "he_uniform":
        limit = np.sqrt(6 / n_in)
        w = np.random.uniform(-limit, limit, (n_out, n_in)).astype(np.float64)
    elif method == "he_normal":
        std = np.sqrt(2 / n_in)
        w = np.random.normal(0, std, (n_out, n_in)).astype(np.float64)
    elif method == "normal":
        w = np.random.normal(0, 0.1, (n_out, n_in)).astype(np.float64)
    elif method == "orthogonal":
        w = np.random.normal(0, 1, (n_out, n_in)).astype(np.float64)
        u, _, vt = np.linalg.svd(w, full_matrices=False)
        w = u @ vt
    else:
        raise ValueError(f"Unknown initialization method: {method}")
    b = np.zeros(n_out, dtype=np.float64)
    return w, b

def init_conv_weights(in_ch, out_ch, k, method="he_normal"):
    if method == "xavier_uniform":
        limit = np.sqrt(6 / (in_ch * k * k + out_ch))
        w = np.random.uniform(-limit, limit, (out_ch, in_ch, k, k)).astype(np.float64)
    elif method == "xavier_normal":
        std = np.sqrt(2 / (in_ch * k * k + out_ch))
        w = np.random.normal(0, std, (out_ch, in_ch, k, k)).astype(np.float64)
    elif method == "he_uniform":
        limit = np.sqrt(6 / (in_ch * k * k))
        w = np.random.uniform(-limit, limit, (out_ch, in_ch, k, k)).astype(np.float64)
    elif method == "he_normal":
        std = np.sqrt(2 / (in_ch * k * k))
        w = np.random.normal(0, std, (out_ch, in_ch, k, k)).astype(np.float64)
    elif method == "normal":
        w = np.random.normal(0, 0.1, (out_ch, in_ch, k, k)).astype(np.float64)
    elif method == "orthogonal":
        w = np.random.normal(0, 1, (out_ch, in_ch * k * k)).astype(np.float64)
        u, _, vt = np.linalg.svd(w, full_matrices=False)
        w = (u @ vt).reshape(out_ch, in_ch, k, k)
    else:
        raise ValueError(f"Unknown initialization method: {method}")
    b = np.zeros(out_ch, dtype=np.float64)
    return w, b
