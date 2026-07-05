import numpy as np
from .forward import im2col

def update(self):
    self.t += 1
    b1, b2, eps = 0.9, 0.999, 1e-8

    if not self.opt_state:
        for layer in self.layers:
            if layer["type"] in ("dense", "sparse", "conv2d"):
                self.opt_state.append({
                    "mw": np.zeros_like(layer["weights"]),
                    "vw": np.zeros_like(layer["weights"]),
                    "mb": np.zeros_like(layer["bias"]),
                    "vb": np.zeros_like(layer["bias"]),
                    "vgw": np.zeros_like(layer["weights"]),
                    "vgb": np.zeros_like(layer["bias"]),
                })
            elif layer["type"] == "batchnorm":
                self.opt_state.append({
                    "mg": np.zeros_like(layer["gamma"]),
                    "vg": np.zeros_like(layer["gamma"]),
                    "mb": np.zeros_like(layer["beta"]),
                    "vb": np.zeros_like(layer["beta"]),
                })
            else:
                self.opt_state.append(None)

    for l, layer in enumerate(self.layers):
        state = self.opt_state[l]
        if layer["type"] in ("dense", "sparse"):
            grad_w = np.dot(self.deltas[l].T, self.outputs[l])
            grad_b = np.sum(self.deltas[l], axis=0)
            if layer["type"] == "sparse":
                grad_w *= layer["mask"]
        elif layer["type"] == "conv2d":
            K = layer["k"]
            col = im2col(self.outputs[l], K, K)
            delta_flat = self.deltas[l].transpose(0, 2, 3, 1).reshape(-1, layer["weights"].shape[0])
            grad_w_flat = np.dot(delta_flat.T, col)
            grad_w = grad_w_flat.reshape(layer["weights"].shape)
            grad_b = np.sum(self.deltas[l], axis=(0, 2, 3))
        elif layer["type"] == "batchnorm":
            grad_gamma = layer.get("d_gamma", np.zeros_like(layer["gamma"]))
            grad_beta = layer.get("d_beta", np.zeros_like(layer["beta"]))

            if self.optimizer_type == "sgd":
                state["mg"] = self.momentum * state["mg"] - self.learning_rate * grad_gamma
                state["mb"] = self.momentum * state["mb"] - self.learning_rate * grad_beta
                layer["gamma"] += state["mg"]
                layer["beta"] += state["mb"]
            elif self.optimizer_type == "rmsprop":
                state["vg"] = b2 * state["vg"] + (1 - b2) * (grad_gamma ** 2)
                state["vb"] = b2 * state["vb"] + (1 - b2) * (grad_beta ** 2)
                layer["gamma"] -= self.learning_rate * grad_gamma / (np.sqrt(state["vg"]) + eps)
                layer["beta"] -= self.learning_rate * grad_beta / (np.sqrt(state["vb"]) + eps)
            elif self.optimizer_type == "adagrad":
                state["vg"] += grad_gamma ** 2
                state["vb"] += grad_beta ** 2
                layer["gamma"] -= self.learning_rate * grad_gamma / (np.sqrt(state["vg"]) + eps)
                layer["beta"] -= self.learning_rate * grad_beta / (np.sqrt(state["vb"]) + eps)
            else:  # adam
                state["mg"] = b1 * state["mg"] + (1 - b1) * grad_gamma
                state["vg"] = b2 * state["vg"] + (1 - b2) * (grad_gamma ** 2)
                layer["gamma"] -= self.learning_rate * (state["mg"] / (1 - b1 ** self.t)) / (np.sqrt(state["vg"] / (1 - b2 ** self.t)) + eps)
                state["mb"] = b1 * state["mb"] + (1 - b1) * grad_beta
                state["vb"] = b2 * state["vb"] + (1 - b2) * (grad_beta ** 2)
                layer["beta"] -= self.learning_rate * (state["mb"] / (1 - b1 ** self.t)) / (np.sqrt(state["vb"] / (1 - b2 ** self.t)) + eps)
            continue
        else:
            continue

        grad_w = grad_w + self.l2_lambda * layer["weights"] * layer.get("mask", 1.0)

        if self.optimizer_type == "sgd":
            state["vgw"] = self.momentum * state["vgw"] - self.learning_rate * grad_w
            state["vgb"] = self.momentum * state["vgb"] - self.learning_rate * grad_b
            layer["weights"] += state["vgw"]
            layer["bias"] += state["vgb"]
        elif self.optimizer_type == "rmsprop":
            state["vw"] = b2 * state["vw"] + (1 - b2) * (grad_w ** 2)
            state["vb"] = b2 * state["vb"] + (1 - b2) * (grad_b ** 2)
            layer["weights"] -= self.learning_rate * grad_w / (np.sqrt(state["vw"]) + eps)
            layer["bias"] -= self.learning_rate * grad_b / (np.sqrt(state["vb"]) + eps)
        elif self.optimizer_type == "adagrad":
            state["vw"] += grad_w ** 2
            state["vb"] += grad_b ** 2
            layer["weights"] -= self.learning_rate * grad_w / (np.sqrt(state["vw"]) + eps)
            layer["bias"] -= self.learning_rate * grad_b / (np.sqrt(state["vb"]) + eps)
        else:
            state["mw"] = b1 * state["mw"] + (1 - b1) * grad_w
            state["vw"] = b2 * state["vw"] + (1 - b2) * (grad_w ** 2)
            layer["weights"] -= self.learning_rate * (state["mw"] / (1 - b1 ** self.t)) / (np.sqrt(state["vw"] / (1 - b2 ** self.t)) + eps)
            state["mb"] = b1 * state["mb"] + (1 - b1) * grad_b
            state["vb"] = b2 * state["vb"] + (1 - b2) * (grad_b ** 2)
            layer["bias"] -= self.learning_rate * (state["mb"] / (1 - b1 ** self.t)) / (np.sqrt(state["vb"] / (1 - b2 ** self.t)) + eps)
