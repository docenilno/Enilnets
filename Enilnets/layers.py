import numpy as np
from .weight_init import init_weights, init_conv_weights

def add_dense(self, n_in, n_out, activation="relu", init_method="xavier_uniform"):
    w, b = init_weights(n_in, n_out, method=init_method)
    self.layers.append({"type": "dense", "weights": w, "bias": b, "activation": activation})

def add_sparse(self, n_in, n_out, connectivity=0.5, activation="relu", init_method="xavier_uniform"):
    w, b = init_weights(n_in, n_out, method=init_method)
    mask = (np.random.rand(n_out, n_in) < connectivity).astype(np.float64)
    self.layers.append({"type": "sparse", "weights": w * mask, "bias": b, "mask": mask, "activation": activation})

def add_conv2d(self, in_ch, out_ch, k, activation="relu", init_method="he_normal"):
    w, b = init_conv_weights(in_ch, out_ch, k, method=init_method)
    self.layers.append({"type": "conv2d", "weights": w, "bias": b, "in_ch": in_ch, "out_ch": out_ch, "k": k, "activation": activation})

def add_flatten(self):
    self.layers.append({"type": "flatten"})

def add_maxpool2d(self, pool_size=2):
    self.layers.append({"type": "maxpool2d", "p": pool_size})

def add_avgpool2d(self, pool_size=2):
    self.layers.append({"type": "avgpool2d", "p": pool_size})

def add_batchnorm(self, num_features, epsilon=1e-5, momentum=0.1):
    self.layers.append({"type": "batchnorm", "num_features": num_features, "epsilon": epsilon, "momentum": momentum,
                        "running_mean": np.zeros(num_features, dtype=np.float64),
                        "running_var": np.ones(num_features, dtype=np.float64),
                        "gamma": np.ones(num_features, dtype=np.float64),
                        "beta": np.zeros(num_features, dtype=np.float64)})

def add_dropout(self, rate=0.5):
    self.layers.append({"type": "dropout", "rate": rate})
