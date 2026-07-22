"""NeuralNet.update/compute_gradients/apply_gradients/accumulate_gradients/
apply_accumulated_gradients: turns the deltas Backward() populated into
actual per-parameter gradients, and applies them via the configured
optimizer."""
import math
from typing import Any, Dict, List, Optional

from ..core.backend import np
from ..core import constants
from ..nn.forward import im2col, im2col1d

#: Every optimizer `NeuralNet(optimizer=...)` accepts. The order is the one
#: used in the error message for an unknown name.
OPTIMIZERS = (
    "sgd", "rmsprop", "adagrad", "adadelta",
    "adam", "adamw", "adamax", "nadam", "radam",
    "lion", "lamb", "adafactor",
)

#: Which accumulator slots each optimizer actually needs, so state is not
#: allocated for buffers a rule never reads. This is what makes Lion's
#: half-of-Adam memory and AdaFactor's O(R + C) state real rather than
#: nominal. Slots created on demand (AdaDelta's u_, AdaFactor's vr_/vc_/vf_)
#: are deliberately absent here.
_STATE_SLOTS = {
    "sgd": ("m",),
    "rmsprop": ("v",),
    "adagrad": ("v",),
    "adadelta": ("v",),
    "adam": ("m", "v"),
    "adamw": ("m", "v"),
    "adamax": ("m", "v"),
    "nadam": ("m", "v"),
    "radam": ("m", "v"),
    "lamb": ("m", "v"),
    "lion": ("m",),
    "adafactor": (),
}

#: Optimizers whose weight decay is DECOUPLED -- applied to the weights
#: directly rather than folded into the gradient before the moment update.
_DECOUPLED_DECAY = frozenset({"adamw", "lion", "lamb"})

# Which of each layer type's trainable params get weight decay (never biases,
# never batchnorm/layernorm gamma/beta). MoE stores its experts as one
# stacked array per parameter, so decay applies to the whole stack at once.
_WEIGHT_DECAY_PARAMS = {
    "dense": ("weights",),
    "sparse": ("weights",),
    "conv2d": ("weights",),
    "conv1d": ("weights",),
    "embedding": ("weights",),
    "positional_encoding": ("weights",),
    "multihead_attention": ("Wq", "Wk", "Wv", "Wo"),
    "cross_attention": ("Wq", "Wk", "Wv", "Wo"),
    "rnn": ("Wx", "Wh"),
    "lstm": ("Wx", "Wh"),
    "gru": ("Wx", "Wh"),
    "moe": ("W1", "W2", "Wr"),
    "cbam_channel": ("W1", "W2"),
}

def compute_gradients(self: Any) -> List[Optional[Dict[str, Any]]]:
    """Compute per-layer raw parameter gradients without mutating any weights
    or optimizer state. Returns a list aligned with self.layers; entries are
    {param_name: grad_array} dicts, or None for layers with no trainable
    parameters. Call Backward() first so self.deltas is populated."""
    from ..nn.backward import (embedding_backward, multihead_attention_backward,
                            cross_attention_backward, rnn_backward, lstm_backward, gru_backward)
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
            if t == "sparse":
                grad_w = grad_w * layer["mask"]
            grads[l] = {"weights": grad_w}
            if layer.get("use_bias", True):
                grads[l]["bias"] = np.sum(d, axis=0)
        elif t == "conv2d":
            K = layer["k"]
            stride = layer.get("stride", 1)
            pad = layer.get("pad", 0)
            col = self.conv_cache[l] if l < len(self.conv_cache) and self.conv_cache[l] is not None \
                else im2col(self.outputs[l], K, K, stride=stride, pad=pad)
            delta_flat = self.deltas[l].transpose(0, 2, 3, 1).reshape(-1, layer["weights"].shape[0])
            grad_w = np.dot(delta_flat.T, col).reshape(layer["weights"].shape)
            grad_b = np.sum(self.deltas[l], axis=(0, 2, 3))
            grads[l] = {"weights": grad_w, "bias": grad_b}
        elif t == "conv1d":
            K = layer["k"]
            stride = layer.get("stride", 1)
            pad = layer.get("pad", 0)
            col = self.conv_cache[l] if l < len(self.conv_cache) and self.conv_cache[l] is not None \
                else im2col1d(self.outputs[l], K, stride=stride, pad=pad)
            delta_flat = self.deltas[l].transpose(0, 2, 1).reshape(-1, layer["weights"].shape[0])
            grad_w = np.dot(delta_flat.T, col).reshape(layer["weights"].shape)
            grad_b = np.sum(self.deltas[l], axis=(0, 2))
            grads[l] = {"weights": grad_w, "bias": grad_b}
        elif t in ("batchnorm", "layernorm"):
            grad_gamma = layer.get("d_gamma", np.zeros_like(layer["gamma"]))
            grad_beta = layer.get("d_beta", np.zeros_like(layer["beta"]))
            grads[l] = {"gamma": grad_gamma, "beta": grad_beta}
        elif t == "embedding":
            layer["_last_input"] = self.outputs[l]
            grad_w = embedding_backward(self.deltas[l], layer)
            grads[l] = {"weights": grad_w}
        elif t == "positional_encoding":
            # Sinusoidal encodings have no trainable params (grads[l] stays
            # None). Learnable ones are a plain additive table -- the
            # gradient w.r.t. the table is just this layer's own output
            # delta summed over the batch axis (d(x+table)/d(table) == 1,
            # broadcast over batch), scattered into the table's first S
            # rows (the only ones this call actually used).
            if layer.get("_pos_type") == "learnable":
                d = self.deltas[l]  # (B, S, E)
                grad_w = np.zeros_like(layer["weights"])
                grad_w[:d.shape[1]] = np.sum(d, axis=0)
                grads[l] = {"weights": grad_w}
        elif t == "multihead_attention":
            # Backward()'s main loop already computes and stores this layer's
            # own d_Wq/etc as a side effect for every layer EXCEPT index 0
            # (it's driven by `nxt = layers[l+1]`, so layer 0 is never reached
            # that way) -- recomputing here for every other layer redoes an
            # already-done backward pass for nothing. Only layer 0 needs it.
            if l == 0:
                multihead_attention_backward(self.deltas[l], layer, self.attention_cache[l])
            grads[l] = {p: layer[f"d_{p}"] for p in ("Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo")}
        elif t == "cross_attention":
            # Same rationale as multihead_attention above.
            if l == 0:
                cross_attention_backward(self.deltas[l], layer, self.attention_cache[l])
            grads[l] = {p: layer[f"d_{p}"] for p in ("Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo")}
        elif t in ("rnn", "lstm"):
            # Same rationale: Backward()'s main loop already ran this exact
            # backward function (and its O(seq_len) BPTT cost) for every
            # layer except index 0.
            if l == 0:
                fn = rnn_backward if t == "rnn" else lstm_backward
                fn(self.deltas[l], layer, self.rnn_cache[l])
            grads[l] = {p: layer[f"d_{p}"] for p in ("Wx", "Wh", "b")}
        elif t == "cbam_channel":
            # Same rationale as multihead_attention: Backward()'s main loop
            # already stored these for every layer except index 0.
            if l == 0:
                from ..nn.backward import cbam_channel_backward
                cbam_channel_backward(self.deltas[l], layer, self.conv_cache[l])
            grads[l] = {p: layer[f"d_{p}"] for p in ("W1", "b1", "W2", "b2")}
        elif t == "moe":
            # Same rationale as multihead_attention: Backward()'s main loop
            # already stored these for every layer except index 0.
            if l == 0:
                from ..nn.moe import moe_backward
                moe_backward(self.deltas[l], layer, self.moe_cache[l])
            grads[l] = {p: layer[f"d_{p}"]
                        for p in ("W1", "b1", "W2", "b2", "Wr", "br")}
        elif t == "gru":
            if l == 0:
                gru_backward(self.deltas[l], layer, self.rnn_cache[l])
            grads[l] = {p: layer[f"d_{p}"] for p in ("Wx", "Wh", "bx", "bh")}
    return grads

def _adafactor_update(self: Any, state: Dict[str, Any], p: str, grad: Any) -> Any:
    """AdaFactor's normalized update for one parameter (Shazeer & Stern 2018).

    For a matrix, the second moment is stored as a ROW vector plus a COLUMN
    vector and reconstructed as their rank-1 outer product -- O(R + C) state
    instead of O(R x C), which is the entire point of the optimizer. Tensors
    with more than two axes are viewed as (-1, last_axis) first. Vectors have
    nothing to factor, so they keep a full accumulator."""
    eps1 = self.adafactor_eps1
    # beta2_t rises towards 1 as training proceeds, so early steps forget
    # quickly and later ones average over a long horizon.
    beta2_t = 1.0 - self.t ** self.adafactor_decay_rate
    sq = grad ** 2 + eps1

    if grad.ndim >= 2:
        shape = grad.shape
        mat = sq.reshape(-1, shape[-1])
        rkey, ckey = f"vr_{p}", f"vc_{p}"
        if rkey not in state:
            state[rkey] = np.zeros(mat.shape[0], dtype=grad.dtype)
            state[ckey] = np.zeros(mat.shape[1], dtype=grad.dtype)
        state[rkey] = beta2_t * state[rkey] + (1 - beta2_t) * np.mean(mat, axis=1)
        state[ckey] = beta2_t * state[ckey] + (1 - beta2_t) * np.mean(mat, axis=0)
        # Normalizing the row factor by its mean is what makes the outer
        # product reconstruct the right scale rather than its square.
        row = state[rkey] / np.maximum(np.mean(state[rkey]), eps1)
        v_hat = np.outer(row, state[ckey]).reshape(shape)
    else:
        fkey = f"vf_{p}"
        if fkey not in state:
            state[fkey] = np.zeros_like(grad)
        state[fkey] = beta2_t * state[fkey] + (1 - beta2_t) * sq
        v_hat = state[fkey]

    update = grad / np.sqrt(np.maximum(v_hat, eps1))
    # Update clipping by RMS, not by norm: keeps the typical element's step
    # bounded regardless of how many elements the tensor has.
    rms = float(np.sqrt(np.mean(update ** 2)))
    denom = max(1.0, rms / self.adafactor_clip_threshold)
    return update / denom


def apply_gradients(self: Any, grads: List[Optional[Dict[str, Any]]]) -> None:
    """Apply a list of per-layer gradient dicts (as returned by
    compute_gradients) using the configured optimizer formula, mutating layer
    weights and optimizer state (self.opt_state, self.t)."""
    self.t += 1
    b1, b2, eps = self.adam_beta1, self.adam_beta2, self.adam_epsilon
    rms_decay, rms_eps = self.rmsprop_decay, self.rmsprop_epsilon
    ada_eps = self.adagrad_epsilon

    if not self.opt_state:
        slots = _STATE_SLOTS.get(self.optimizer_type, ("m", "v"))
        for g in grads:
            if g is None:
                self.opt_state.append(None)
            else:
                self.opt_state.append({
                    f"{slot}_{p}": np.zeros_like(val)
                    for p, val in g.items() for slot in slots
                })

    for l, layer in enumerate(self.layers):
        if layer.get("_frozen", False):
            continue
        g = grads[l]
        if g is None:
            continue
        state = self.opt_state[l]
        wd_params = _WEIGHT_DECAY_PARAMS.get(layer["type"], ())
        mask = layer.get("mask", 1.0)

        prune_mask = layer.get("prune_mask", {})
        for p, grad in g.items():
            # A pruned weight must receive no update at all. Masking the
            # GRADIENT as well as the weight keeps momentum and the adaptive
            # denominators from accumulating for a parameter that is then
            # immediately re-zeroed.
            pruned = prune_mask.get(p)
            if pruned is not None:
                grad = grad * pruned
            decayed = p in wd_params
            if decayed and self.optimizer_type not in _DECOUPLED_DECAY:
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
            elif self.optimizer_type == "adadelta":
                # No learning rate in the original formulation: the step size
                # is the ratio of the RMS of past UPDATES to the RMS of past
                # gradients, which is why it is unit-consistent. `u_` is the
                # extra accumulator that carries; learning_rate multiplies
                # the result and should normally be left at 1.0.
                rho, d_eps = self.adadelta_rho, self.adadelta_epsilon
                ukey = f"u_{p}"
                if ukey not in state:
                    state[ukey] = np.zeros_like(grad)
                state[vkey] = rho * state[vkey] + (1 - rho) * (grad ** 2)
                step = -(np.sqrt(state[ukey] + d_eps) /
                         np.sqrt(state[vkey] + d_eps)) * grad
                state[ukey] = rho * state[ukey] + (1 - rho) * (step ** 2)
                layer[p] += self.learning_rate * step
            elif self.optimizer_type == "adamax":
                # Adam with the L2 norm of past gradients replaced by the
                # L-infinity norm, which needs no bias correction of its own.
                state[mkey] = b1 * state[mkey] + (1 - b1) * grad
                state[vkey] = np.maximum(b2 * state[vkey], np.abs(grad))
                layer[p] -= (self.learning_rate / (1 - b1 ** self.t)) * \
                    state[mkey] / (state[vkey] + eps)
            elif self.optimizer_type == "nadam":
                # Nesterov-accelerated Adam: look ahead by applying the
                # momentum step to the CURRENT gradient rather than the
                # stored moment (Dozat 2016, the practical formulation).
                state[mkey] = b1 * state[mkey] + (1 - b1) * grad
                state[vkey] = b2 * state[vkey] + (1 - b2) * (grad ** 2)
                m_hat = state[mkey] / (1 - b1 ** self.t)
                v_hat = state[vkey] / (1 - b2 ** self.t)
                lookahead = b1 * m_hat + (1 - b1) * grad / (1 - b1 ** self.t)
                layer[p] -= self.learning_rate * lookahead / (np.sqrt(v_hat) + eps)
            elif self.optimizer_type == "radam":
                # Rectified Adam: early in training the second-moment
                # estimate is built from too few samples for its variance to
                # be trustworthy, so the adaptive denominator is switched off
                # entirely until rho_t clears the threshold -- a plain
                # momentum step until then, no warmup schedule needed.
                state[mkey] = b1 * state[mkey] + (1 - b1) * grad
                state[vkey] = b2 * state[vkey] + (1 - b2) * (grad ** 2)
                m_hat = state[mkey] / (1 - b1 ** self.t)
                rho_inf = 2.0 / (1 - b2) - 1
                b2t = b2 ** self.t
                rho_t = rho_inf - 2 * self.t * b2t / (1 - b2t)
                if rho_t > constants.RADAM_RHO_THRESHOLD:
                    rect = math.sqrt(
                        ((rho_t - 4) * (rho_t - 2) * rho_inf) /
                        ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                    v_hat = np.sqrt(state[vkey] / (1 - b2t))
                    layer[p] -= self.learning_rate * rect * m_hat / (v_hat + eps)
                else:
                    layer[p] -= self.learning_rate * m_hat
            elif self.optimizer_type == "lion":
                # Lion (Chen et al. 2023): the update is the SIGN of an
                # interpolated momentum, so every parameter moves by exactly
                # +/- lr. Only one accumulator is kept -- half of Adam's
                # optimizer memory -- and it is updated with a different beta
                # than the one used for the step, which is the whole trick.
                lb1, lb2 = self.lion_beta1, self.lion_beta2
                update = np.sign(lb1 * state[mkey] + (1 - lb1) * grad)
                if decayed:
                    update = update + self.l2_lambda * layer[p] * mask
                layer[p] -= self.learning_rate * update
                state[mkey] = lb2 * state[mkey] + (1 - lb2) * grad
            elif self.optimizer_type == "lamb":
                # LAMB (You et al. 2020): Adam, then rescale the whole
                # parameter tensor's update by the layer-wise trust ratio
                # ||w|| / ||r||, so every tensor moves a distance
                # proportional to its own norm. That is what makes very large
                # batches trainable without per-layer LR tuning.
                state[mkey] = b1 * state[mkey] + (1 - b1) * grad
                state[vkey] = b2 * state[vkey] + (1 - b2) * (grad ** 2)
                m_hat = state[mkey] / (1 - b1 ** self.t)
                v_hat = state[vkey] / (1 - b2 ** self.t)
                r = m_hat / (np.sqrt(v_hat) + eps)
                if decayed:
                    r = r + self.l2_lambda * layer[p] * mask
                w_norm = float(np.sqrt(np.sum(layer[p] ** 2)))
                r_norm = float(np.sqrt(np.sum(r ** 2)))
                # A zero norm on either side leaves the ratio undefined; 1.0
                # degrades cleanly to a plain Adam step there.
                trust = 1.0
                if w_norm > 0.0 and r_norm > 0.0:
                    trust = min(w_norm / r_norm, constants.LAMB_MAX_TRUST_RATIO)
                layer[p] -= self.learning_rate * trust * r
            elif self.optimizer_type == "adafactor":
                layer[p] -= self.learning_rate * _adafactor_update(self, state, p, grad)
            else:  # adam / adamw share the same moment update
                state[mkey] = b1 * state[mkey] + (1 - b1) * grad
                state[vkey] = b2 * state[vkey] + (1 - b2) * (grad ** 2)
                layer[p] -= self.learning_rate * (state[mkey] / (1 - b1 ** self.t)) / \
                    (np.sqrt(state[vkey] / (1 - b2 ** self.t)) + eps)

            if pruned is not None:
                layer[p] = layer[p] * pruned
            if decayed and self.optimizer_type == "adamw":
                # Decoupled weight decay: applied directly to the weights,
                # not folded into the gradient before the Adam moment update.
                # (Lion and LAMB also decouple, but fold the decay into their
                # own update below rather than as a separate step.)
                layer[p] -= self.learning_rate * self.l2_lambda * layer[p]

def accumulate_gradients(self: Any) -> None:
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

def apply_accumulated_gradients(self: Any) -> None:
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

def update(self: Any) -> None:
    """Compute and apply one optimizer step in one call (the common case;
    use compute_gradients/apply_gradients separately if you need to
    inspect/modify gradients in between)."""
    apply_gradients(self, compute_gradients(self))
