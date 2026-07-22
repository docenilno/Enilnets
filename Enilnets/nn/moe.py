"""Mixture-of-Experts (roadmap item 42).

A router scores every token over `num_experts` small MLPs and sends it to
its top-k. Each expert only sees the tokens routed to it -- gathered, run,
scattered back -- so compute per token stays at k experts however many
exist.

Gates are the RAW router softmax probabilities, not renormalized over the
top-k: renormalizing would make every gate exactly 1.0 at k=1 and cut the
router's gradient entirely.

The load-balancing auxiliary loss is Switch Transformer's
``num_experts * sum_e f_e * P_e``, where f_e is the fraction of tokens
whose top-1 choice is expert e and P_e its mean router probability. Only
P_e carries gradient."""

from typing import Any, Dict, Tuple

from ..core.backend import np
from .activations import activate, derivative


def router_probabilities(x_flat: Any, layer: Dict[str, Any]) -> Tuple[Any, Any]:
    """Router logits and softmax probabilities for ``(N, E)`` tokens."""
    logits = np.dot(x_flat, layer["Wr"].T) + layer["br"]
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return logits, exp / np.sum(exp, axis=-1, keepdims=True)


def _top_k(probs: Any, k: int) -> Any:
    """Indices of the k largest probabilities per row, ``(N, k)``."""
    # argpartition would be cheaper, but a full argsort keeps the selection
    # deterministic under ties, which the cached-decode tests rely on.
    return np.argsort(-probs, axis=-1)[:, :k]


def load_balancing_loss(probs: Any, top_idx: Any, num_experts: int) -> Tuple[float, Any]:
    """Switch-Transformer auxiliary loss and its gradient w.r.t. `probs`.
    Returns ``(loss, dprobs)``; `dprobs` is zero unless the loss is used."""
    N = probs.shape[0]
    counts = np.zeros(num_experts, dtype=probs.dtype)
    np.add.at(counts, top_idx[:, 0], 1.0)
    f = counts / max(N, 1)                                   # dispatch fraction
    P = np.mean(probs, axis=0)                               # mean router prob
    loss = float(num_experts * np.sum(f * P))
    # f is a hard count with no gradient; P is the differentiable half.
    dprobs = np.broadcast_to(num_experts * f / max(N, 1), probs.shape)
    return loss, dprobs


def moe_forward(x: Any, layer: Dict[str, Any], training: bool) -> Tuple[Any, Tuple]:
    """Route ``(..., E)`` through the experts. Returns ``(output, cache)``
    with the same leading shape as the input."""
    E = layer["embed_dim"]
    k = layer["top_k"]
    n_exp = layer["num_experts"]
    lead = x.shape[:-1]
    x_flat = x.reshape(-1, E)

    logits, probs = router_probabilities(x_flat, layer)
    top_idx = _top_k(probs, k)                               # (N, k)
    # Gate for each (token, slot) pair; raw probability, see module docstring.
    rows = np.arange(x_flat.shape[0])[:, None]
    gates = probs[rows, top_idx]                             # (N, k)

    aux_loss, _ = load_balancing_loss(probs, top_idx, n_exp)
    layer["_state_aux_loss"] = aux_loss

    out = np.zeros_like(x_flat)
    per_expert = []
    for e in range(n_exp):
        hit = np.asarray(np.nonzero(top_idx == e))           # (2, n_e): rows, slots
        if hit.shape[1] == 0:
            per_expert.append(None)
            continue
        tok, slot = hit[0], hit[1]
        xe = x_flat[tok]
        z = np.dot(xe, layer["W1"][e].T) + layer["b1"][e]
        h = activate(layer["activation"], z)
        y = np.dot(h, layer["W2"][e].T) + layer["b2"][e]
        g = gates[tok, slot][:, None]
        np.add.at(out, tok, g * y)
        per_expert.append((tok, slot, xe, z, h, y))

    cache = (x_flat, logits, probs, top_idx, gates, per_expert, lead)
    return out.reshape(lead + (E,)), cache


def moe_backward(dout: Any, layer: Dict[str, Any], cache: Tuple) -> Any:
    """Backprop through the router and the experts. Stores parameter
    gradients on `layer` and returns dx with the input's shape."""
    x_flat, logits, probs, top_idx, gates, per_expert, lead = cache
    E = layer["embed_dim"]
    n_exp = layer["num_experts"]
    d_flat = dout.reshape(-1, E)

    dx = np.zeros_like(x_flat)
    dgates = np.zeros_like(gates)
    dW1 = np.zeros_like(layer["W1"])
    db1 = np.zeros_like(layer["b1"])
    dW2 = np.zeros_like(layer["W2"])
    db2 = np.zeros_like(layer["b2"])

    for e in range(n_exp):
        entry = per_expert[e]
        if entry is None:
            continue
        tok, slot, xe, z, h, y = entry
        dy_out = d_flat[tok]
        g = gates[tok, slot][:, None]
        # out = sum over slots of gate * y, so both factors take a gradient.
        np.add.at(dgates, (tok, slot), np.sum(dy_out * y, axis=-1))
        dy = dy_out * g

        dW2[e] = np.dot(dy.T, h)
        db2[e] = np.sum(dy, axis=0)
        dh = np.dot(dy, layer["W2"][e])
        dz = dh * derivative(layer["activation"], z)
        dW1[e] = np.dot(dz.T, xe)
        db1[e] = np.sum(dz, axis=0)
        np.add.at(dx, tok, np.dot(dz, layer["W1"][e]))

    # Scatter the per-slot gate gradients back onto the full probability
    # matrix, then add the load-balancing loss's own gradient.
    dprobs = np.zeros_like(probs)
    rows = np.arange(x_flat.shape[0])[:, None]
    np.add.at(dprobs, (rows, top_idx), dgates)
    alpha = layer.get("aux_loss_weight", 0.0)
    if alpha:
        _, daux = load_balancing_loss(probs, top_idx, n_exp)
        dprobs = dprobs + alpha * daux

    dlogits = probs * (dprobs - np.sum(dprobs * probs, axis=-1, keepdims=True))
    layer["d_Wr"] = np.dot(dlogits.T, x_flat)
    layer["d_br"] = np.sum(dlogits, axis=0)
    dx = dx + np.dot(dlogits, layer["Wr"])

    layer["d_W1"], layer["d_b1"] = dW1, db1
    layer["d_W2"], layer["d_b2"] = dW2, db2
    return dx.reshape(lead + (E,))
