import json
import pickle
import os
import numpy as np

def _numpy_encoder(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def Save(self, file):
    payload = {
        "version": 2,
        "layers": self.layers,
        "optimizer": self.optimizer_type,
        "learning_rate": self.learning_rate,
        "l2_lambda": self.l2_lambda,
        "momentum": self.momentum,
        "t": self.t,
    }
    ext = os.path.splitext(file)[1].lower()
    if ext == ".pkl":
        with open(file, "wb") as f:
            pickle.dump(payload, f)
    else:
        with open(file, "w") as f:
            json.dump(payload, f, default=_numpy_encoder)

def Load(self, file):
    ext = os.path.splitext(file)[1].lower()
    if ext == ".pkl":
        with open(file, "rb") as f:
            raw = pickle.load(f)
    else:
        with open(file, "r") as f:
            raw = json.load(f)
    self.layers = []
    for l in raw.get("layers", []):
        for k in ["weights", "bias", "mask", "gamma", "beta", "running_mean", "running_var"]:
            if k in l:
                l[k] = np.array(l[k], dtype=np.float64)
        self.layers.append(l)
    self.opt_state = []
    self.t = raw.get("t", 0)
    self.learning_rate = raw.get("learning_rate", self.learning_rate)
    self.optimizer_type = raw.get("optimizer", self.optimizer_type)
    self.l2_lambda = raw.get("l2_lambda", self.l2_lambda)
    self.momentum = raw.get("momentum", self.momentum)
