import numpy as np
from .forward import im2col

# Which of each layer type's trainable params get weight decay (never biases,
# never batchnorm/layernorm gamma/beta).
_WEIGHT_DECAY_PARAMS = {
    "dense": ("weights",),
    "sparse": ("weights",),
    "conv2d": ("weights",),
    "embedding": ("weights",),
    "multihead_attention": ("Wq", "Wk", "Wv", "Wo"),
    "rnn": ("Wx", "Wh"),
    "lstm": ("Wx", "Wh"),
    "gru": ("Wx", "Wh"),
}

def compute_gradients(self):
    """Compute per-layer raw parameter gradients without mutating any weights
    or optimizer state. Returns a list aligned with self.layers; entries are
    {param_name: grad_array} dicts, or None for layers with no trainable
    parameters. Call Backward() first so self.deltas is populated."""
    from .backward import (embedding_backward, multihead_attention_backward,
                            rnn_backward, lstm_backward, gru_backward)
    grads = [None] * len(self.layers)
    for l, layer in enumerate(self.layers):
        t = layer["type"]
        if t in ("dense", "sparse"):
            d, o = self.deltas[l], self.outputs[l]
            if d.ndim == 3:
                # (batch, seq_len, features) -- flatten leading dims for the
                # weight-gradient reduction (transformer MLP/attention sub-layers).
                d = d.reshape(-1, d.shape[-1])
                o = o.reshape(-1, o.shape[-1])
            grad_w = np.dot(d.T, o)
            grad_b = np.sum(d, axis=0)
            if t == "sparse":
                grad_w = grad_w * layer["mask"]
            grads[l] = {"weights": grad_w, "bias": grad_b}
        elif t == "conv2d":
            K = layer["k"]
            stride = layer.get("stride", 1)
            col = self.conv_cache[l] if l < len(self.conv_cache) and self.conv_cache[l] is not None \
                else im2col(self.outputs[l], K, K, stride=stride)
            delta_flat = self.deltas[l].transpose(0, 2, 3, 1).reshape(-1, layer["weights"].shape[0])
            grad_w = np.dot(delta_flat.T, col).reshape(layer["weights"].shape)
            grad_b = np.sum(self.deltas[l], axis=(0, 2, 3))
            grads[l] = {"weights": grad_w, "bias": grad_b}
        elif t in ("batchnorm", "layernorm"):
            grad_gamma = layer.get("d_gamma", np.zeros_like(layer["gamma"]))
            grad_beta = layer.get("d_beta", np.zeros_like(layer["beta"]))
            grads[l] = {"gamma": grad_gamma, "beta": grad_beta}
        elif t == "embedding":
            layer["_last_input"] = self.outputs[l]
            grad_w = embedding_backward(self.deltas[l], layer)
            grads[l] = {"weights": grad_w}
        elif t == "multihead_attention":
            # Own param grads are only computed by Backward() as a side effect
            # when this layer has a preceding layer to propagate error to, so
            # recompute unconditionally here to also cover layer 0.
            multihead_attention_backward(self.deltas[l], layer, self.attention_cache[l])
            grads[l] = {p: layer[f"d_{p}"] for p in ("Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo")}
        elif t in ("rnn", "lstm"):
            fn = rnn_backward if t == "rnn" else lstm_backward
            fn(self.deltas[l], layer, self.rnn_cache[l])
            grads[l] = {p: layer[f"d_{p}"] for p in ("Wx", "Wh", "b")}
        elif t == "gru":
            gru_backward(self.deltas[l], layer, self.rnn_cache[l])
            grads[l] = {p: layer[f"d_{p}"] for p in ("Wx", "Wh", "bx", "bh")}
    return grads

def apply_gradients(self, grads):
    """Apply a list of per-layer gradient dicts (as returned by
    compute_gradients) using the configured optimizer formula, mutating layer
    weights and optimizer state (self.opt_state, self.t)."""
    self.t += 1
    b1, b2, eps = self.adam_beta1, self.adam_beta2, self.adam_epsilon
    rms_decay, rms_eps = self.rmsprop_decay, self.rmsprop_epsilon
    ada_eps = self.adagrad_epsilon
    wd_type = "adam" if self.optimizer_type == "adamw" else self.optimizer_type

    if not self.opt_state:
        for g in grads:
            if g is None:
                self.opt_state.append(None)
            else:
                state = {}
                for p, val in g.items():
                    state[f"m_{p}"] = np.zeros_like(val)
                    state[f"v_{p}"] = np.zeros_like(val)
                self.opt_state.append(state)

    for l, layer in enumerate(self.layers):
        if layer.get("_frozen", False):
            continue
        g = grads[l]
        if g is None:
            continue
        state = self.opt_state[l]
        wd_params = _WEIGHT_DECAY_PARAMS.get(layer["type"], ())
        mask = layer.get("mask", 1.0)

        for p, grad in g.items():
            decayed = p in wd_params
            if decayed and self.optimizer_type != "adamw":
                grad = grad + self.l2_lambda * layer[p] * mask
            mkey, vkey = f"m_{p}", f"v_{p}"

            if self.optimizer_type == "sgd":
                state[mkey] = self.momentum * state[mkey] - self.learning_rate * grad
                layer[p] += state[mkey]
            elif self.optimizer_type == "rmsprop":
                state[vkey] = rms_decay * state[vkey] + (1 - rms_decay) * (grad ** 2)
                layer[p] -= self.learning_rate * grad / (np.sqrt(state[vkey]) + rms_eps)
            elif self.optimizer_type == "adagrad":
                state[vkey] += grad ** 2
                layer[p] -= self.learning_rate * grad / (np.sqrt(state[vkey]) + ada_eps)
            else:  # adam / adamw share the same moment update
                state[mkey] = b1 * state[mkey] + (1 - b1) * grad
                state[vkey] = b2 * state[vkey] + (1 - b2) * (grad ** 2)
                layer[p] -= self.learning_rate * (state[mkey] / (1 - b1 ** self.t)) / \
                    (np.sqrt(state[vkey] / (1 - b2 ** self.t)) + eps)

            if decayed and self.optimizer_type == "adamw":
                # Decoupled weight decay: applied directly to the weights,
                # not folded into the gradient before the Adam moment update.
                layer[p] -= self.learning_rate * self.l2_lambda * layer[p]

def accumulate_gradients(self):
    """Compute gradients for the current batch and add them into
    self._grad_accum (creating it on the first call), incrementing
    self._accum_steps. Call apply_accumulated_gradients() once enough steps
    have been accumulated to actually update the weights."""
    grads = compute_gradients(self)
    if not self._grad_accum:
        self._grad_accum = grads
    else:
        for l, g in enumerate(grads):
            if g is None:
                continue
            for p, val in g.items():
                self._grad_accum[l][p] = self._grad_accum[l][p] + val
    self._accum_steps += 1

def apply_accumulated_gradients(self):
    """Average the accumulated gradients over self._accum_steps and apply
    them via apply_gradients, then reset the accumulator."""
    if self._accum_steps == 0:
        return
    averaged = [
        None if g is None else {p: v / self._accum_steps for p, v in g.items()}
        for g in self._grad_accum
    ]
    apply_gradients(self, averaged)
    self._grad_accum = []
    self._accum_steps = 0

def update(self):
    apply_gradients(self, compute_gradients(self))
