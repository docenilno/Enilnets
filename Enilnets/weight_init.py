import numpy as np
from . import constants

def init_weights(n_in, n_out, method="xavier_uniform", std=None):
    if method == "xavier_uniform":
        limit = np.sqrt(6 / (n_in + n_out))
        w = np.random.uniform(-limit, limit, (n_out, n_in)).astype(np.float64)
    elif method == "xavier_normal":
        s = np.sqrt(2 / (n_in + n_out))
        w = np.random.normal(0, s, (n_out, n_in)).astype(np.float64)
    elif method == "he_uniform":
        limit = np.sqrt(6 / n_in)
        w = np.random.uniform(-limit, limit, (n_out, n_in)).astype(np.float64)
    elif method == "he_normal":
        s = np.sqrt(2 / n_in)
        w = np.random.normal(0, s, (n_out, n_in)).astype(np.float64)
    elif method == "normal":
        s = constants.NORMAL_INIT_STD if std is None else std
        w = np.random.normal(0, s, (n_out, n_in)).astype(np.float64)
    elif method == "orthogonal":
        w = np.random.normal(0, 1, (n_out, n_in)).astype(np.float64)
        u, _, vt = np.linalg.svd(w, full_matrices=False)
        w = u @ vt
    elif method == "zeros":
        w = np.zeros((n_out, n_in), dtype=np.float64)
    elif method == "ones":
        w = np.ones((n_out, n_in), dtype=np.float64)
    else:
        raise ValueError(f"Unknown initialization method: {method}")
    b = np.zeros(n_out, dtype=np.float64)
    return w, b

def init_conv_weights(in_ch, out_ch, k, method="he_normal", std=None):
    if method == "xavier_uniform":
        limit = np.sqrt(6 / (in_ch * k * k + out_ch * k * k))
        w = np.random.uniform(-limit, limit, (out_ch, in_ch, k, k)).astype(np.float64)
    elif method == "xavier_normal":
        s = np.sqrt(2 / (in_ch * k * k + out_ch * k * k))
        w = np.random.normal(0, s, (out_ch, in_ch, k, k)).astype(np.float64)
    elif method == "he_uniform":
        limit = np.sqrt(6 / (in_ch * k * k))
        w = np.random.uniform(-limit, limit, (out_ch, in_ch, k, k)).astype(np.float64)
    elif method == "he_normal":
        s = np.sqrt(2 / (in_ch * k * k))
        w = np.random.normal(0, s, (out_ch, in_ch, k, k)).astype(np.float64)
    elif method == "normal":
        s = constants.NORMAL_INIT_STD if std is None else std
        w = np.random.normal(0, s, (out_ch, in_ch, k, k)).astype(np.float64)
    elif method == "orthogonal":
        w = np.random.normal(0, 1, (out_ch, in_ch * k * k)).astype(np.float64)
        u, _, vt = np.linalg.svd(w, full_matrices=False)
        w = (u @ vt).reshape(out_ch, in_ch, k, k)
    elif method == "zeros":
        w = np.zeros((out_ch, in_ch, k, k), dtype=np.float64)
    elif method == "ones":
        w = np.ones((out_ch, in_ch, k, k), dtype=np.float64)
    else:
        raise ValueError(f"Unknown initialization method: {method}")
    b = np.zeros(out_ch, dtype=np.float64)
    return w, b

def init_conv1d_weights(in_ch, out_ch, k, method="he_normal", std=None):
    if method == "xavier_uniform":
        limit = np.sqrt(6 / (in_ch * k + out_ch * k))
        w = np.random.uniform(-limit, limit, (out_ch, in_ch, k)).astype(np.float64)
    elif method == "xavier_normal":
        s = np.sqrt(2 / (in_ch * k + out_ch * k))
        w = np.random.normal(0, s, (out_ch, in_ch, k)).astype(np.float64)
    elif method == "he_uniform":
        limit = np.sqrt(6 / (in_ch * k))
        w = np.random.uniform(-limit, limit, (out_ch, in_ch, k)).astype(np.float64)
    elif method == "he_normal":
        s = np.sqrt(2 / (in_ch * k))
        w = np.random.normal(0, s, (out_ch, in_ch, k)).astype(np.float64)
    elif method == "normal":
        s = constants.NORMAL_INIT_STD if std is None else std
        w = np.random.normal(0, s, (out_ch, in_ch, k)).astype(np.float64)
    elif method == "orthogonal":
        w = np.random.normal(0, 1, (out_ch, in_ch * k)).astype(np.float64)
        u, _, vt = np.linalg.svd(w, full_matrices=False)
        w = (u @ vt).reshape(out_ch, in_ch, k)
    elif method == "zeros":
        w = np.zeros((out_ch, in_ch, k), dtype=np.float64)
    elif method == "ones":
        w = np.ones((out_ch, in_ch, k), dtype=np.float64)
    else:
        raise ValueError(f"Unknown initialization method: {method}")
    b = np.zeros(out_ch, dtype=np.float64)
    return w, b

def init_embedding_weights(vocab_size, embed_dim, method="normal", std=None):
    if method == "normal":
        s = constants.NORMAL_INIT_STD if std is None else std
        w = np.random.normal(0, s, (vocab_size, embed_dim)).astype(np.float64)
    elif method == "xavier_uniform":
        limit = np.sqrt(6 / (vocab_size + embed_dim))
        w = np.random.uniform(-limit, limit, (vocab_size, embed_dim)).astype(np.float64)
    elif method == "xavier_normal":
        s = np.sqrt(2 / (vocab_size + embed_dim))
        w = np.random.normal(0, s, (vocab_size, embed_dim)).astype(np.float64)
    elif method == "zeros":
        w = np.zeros((vocab_size, embed_dim), dtype=np.float64)
    else:
        raise ValueError(f"Unknown embedding init method: {method}")
    return w
