"""Linearized attention kernels (roadmap item 40): linear attention and
Performer/FAVOR+.

Both replace ``softmax(q.k)`` with a feature map phi such that
``phi(q).phi(k)`` stands in for the score, letting the sum reassociate as
``phi(Q) @ (phi(K)^T V)`` -- O(S * d^2) instead of O(S^2 * d).

  "linear"    phi(x) = elu(x) + 1. A positive similarity, not a softmax
              approximation.
  "performer" phi(x) = exp(w.x' - ||x'||^2/2) / sqrt(m) with fixed
              Gaussian w. An unbiased estimator of the softmax kernel.

Causal masking uses a prefix-sum formulation costing O(S * d^2) memory."""

from typing import Any, Optional, Tuple

from ..core.backend import np
from ..core import backend

KERNELS = ("softmax", "linear", "performer")


def make_projection(num_features: int, head_dim: int) -> Any:
    """Draw a Performer random-feature matrix ``(num_features, head_dim)``.
    Rows are orthogonalized in blocks of `head_dim` (FAVOR+'s variance
    reduction) and rescaled to chi-distributed norms."""
    blocks = []
    for _ in range((num_features + head_dim - 1) // head_dim):
        q, _ = np.linalg.qr(np.random.randn(head_dim, head_dim))
        blocks.append(q.T)
    omega = np.concatenate(blocks, axis=0)[:num_features]
    # QR gives unit-norm rows; restore the norms an i.i.d. Gaussian would
    # have, which is what keeps the estimator unbiased.
    norms = np.linalg.norm(np.random.randn(num_features, head_dim), axis=1)
    return (omega * norms[:, None]).astype(backend.default_dtype())


def _stab_axes(stabilize: str) -> Any:
    """Which axes the Performer overflow stabilizer maxes over.

    The stabilizer is only legitimate where it factors out of the attention
    ratio. Numerator and denominator are both linear in phi(Q) ROW-WISE, so
    a per-query-row constant cancels ("row"); they are linear in phi(K) only
    as a whole, so a key-side constant must be shared across every key
    ("global"). Using "row" on the key side would silently rescale
    individual keys and change the kernel."""
    return -1 if stabilize == "row" else (-2, -1)


def performer_stab(x: Any, omega: Any, stabilize: str = "global") -> Any:
    """The stabilizer :func:`feature_map` would pick for `x`. Exposed so an
    incremental decoder can pin one shared scale across steps."""
    scale = x.shape[-1] ** 0.25
    proj = np.matmul(x / scale, omega.T)
    return np.max(proj, axis=_stab_axes(stabilize), keepdims=True)


def feature_map(x: Any, kernel: str, omega: Optional[Any] = None,
                stabilize: str = "row", stab: Optional[Any] = None) -> Any:
    """Apply the feature map to ``(B, H, S, Dh)``. Returns ``(B, H, S, Df)``
    (Df == Dh for "linear", num_features for "performer"). `stabilize` is
    "row" for queries and "global" for keys (see :func:`_stab_axes`); `stab`
    overrides the computed stabilizer with an explicit one."""
    if kernel == "linear":
        return np.where(x > 0, x + 1.0, np.exp(np.minimum(x, 0.0)))
    if kernel == "performer":
        scale = x.shape[-1] ** 0.25
        xs = x / scale
        proj = np.matmul(xs, omega.T)                      # (B, H, S, Df)
        sq = np.sum(xs * xs, axis=-1, keepdims=True) / 2.0
        if stab is None:
            stab = np.max(proj, axis=_stab_axes(stabilize), keepdims=True)
        return np.exp(proj - sq - stab) / np.sqrt(omega.shape[0])
    raise ValueError(f"Unknown attention feature map: {kernel!r}")


def feature_map_backward(dphi: Any, x: Any, phi: Any, kernel: str,
                         omega: Optional[Any] = None,
                         stabilize: str = "row") -> Any:
    """Gradient w.r.t. `x` given the gradient w.r.t. ``phi(x)``."""
    if kernel == "linear":
        # elu+1 has a diagonal Jacobian: 1 where x > 0, phi(x) elsewhere.
        return dphi * np.where(x > 0, 1.0, phi)
    if kernel == "performer":
        # phi_j = exp(w_j.x' - ||x'||^2/2 - stab)/sqrt(m), so
        #   d(phi_j)/d(x_k) = phi_j * (w_jk - x'_k - d(stab)/d(x'_k)) / d^(1/4).
        # stab is max_(s,j) of a LINEAR form, so it is not locally constant:
        # its gradient is w_(j*) placed at the argmax row s*. Dropping that
        # term would be invisible in the composed attention gradient (it
        # cancels with the ratio) but wrong for phi on its own -- and this
        # function is FD-checked on its own.
        scale = x.shape[-1] ** 0.25
        g = dphi * phi                                     # (B, H, S, Df)
        dx = (np.matmul(g, omega)
              - np.sum(g, axis=-1, keepdims=True) * (x / scale)) / scale

        B, H, S, Df = phi.shape
        proj = np.matmul(x / scale, omega.T)
        if stabilize == "row":
            j_star = np.argmax(proj, axis=-1)                  # (B, H, S)
            total = np.sum(g, axis=-1, keepdims=True)          # (B, H, S, 1)
            return dx - total * omega[j_star] / scale
        flat_argmax = np.argmax(proj.reshape(B, H, S * Df), axis=-1)
        s_star, j_star = flat_argmax // Df, flat_argmax % Df
        total = np.sum(g, axis=(-2, -1))                       # (B, H)
        dstab = -(total[:, :, None] * omega[j_star]) / scale    # (B, H, Dh)
        b_idx = np.arange(B)[:, None]
        h_idx = np.arange(H)[None, :]
        dx[b_idx, h_idx, s_star] = dx[b_idx, h_idx, s_star] + dstab
        return dx
    raise ValueError(f"Unknown attention feature map: {kernel!r}")


def _reverse_cumsum(x: Any, axis: int) -> Any:
    """Suffix sums: out[t] = sum over s >= t. The adjoint of ``cumsum``."""
    return np.flip(np.cumsum(np.flip(x, axis=axis), axis=axis), axis=axis)


def linear_attention_forward(Qp: Any, Kp: Any, Vh: Any, causal: bool,
                             eps: float = 1e-20) -> Tuple[Any, Tuple]:
    """Linearized attention over feature-mapped Q/K. `Qp`/`Kp` are
    ``(B, H, S, Df)``, `Vh` is ``(B, H, S, Dh)``. Returns
    ``(context (B, H, S, Dh), cache)``.

    Non-causal reassociates into two matmuls. Causal replaces the two global
    sums with prefix sums, which costs O(S * Df * Dh) memory."""
    if causal:
        kv = np.cumsum(Kp[..., :, None] * Vh[..., None, :], axis=2)  # (B,H,S,Df,Dh)
        ksum = np.cumsum(Kp, axis=2)                                 # (B,H,S,Df)
        num = np.einsum("bhsf,bhsfd->bhsd", Qp, kv)
    else:
        kv = np.matmul(Kp.transpose(0, 1, 3, 2), Vh)                 # (B,H,Df,Dh)
        ksum = np.sum(Kp, axis=2)                                    # (B,H,Df)
        num = np.matmul(Qp, kv)
    # eps only guards against underflow to exactly zero: phi > 0 for both
    # kernels, so the true denominator is strictly positive. It has to stay
    # far below any real denominator -- Performer's stabilized features can
    # be very small, and an eps comparable to them would quietly destroy the
    # approximation rather than protect it.
    # ksum is per-position (B,H,S,Df) when causal, global (B,H,Df) otherwise.
    ksum_b = ksum if causal else ksum[:, :, None, :]
    den = np.sum(Qp * ksum_b, axis=-1, keepdims=True) + eps
    return num / den, (Qp, Kp, Vh, kv, ksum, num, den, causal)


def linear_attention_backward(dout: Any, cache: Tuple) -> Tuple[Any, Any, Any]:
    """Gradients (dQp, dKp, dVh) for :func:`linear_attention_forward`."""
    Qp, Kp, Vh, kv, ksum, num, den, causal = cache
    dnum = dout / den
    # out = num / den, so den picks up -num/den^2.
    dden = -np.sum(dout * num / (den * den), axis=-1, keepdims=True)

    if causal:
        dkv = Qp[..., :, None] * dnum[..., None, :]        # (B,H,S,Df,Dh)
        dQp = np.einsum("bhsd,bhsfd->bhsf", dnum, kv)
        dksum = dden * Qp                                  # (B,H,S,Df)
        dQp = dQp + dden * ksum
        # cumsum's adjoint is the suffix sum.
        dkv_step = _reverse_cumsum(dkv, axis=2)            # (B,H,S,Df,Dh)
        dKp = np.einsum("bhsfd,bhsd->bhsf", dkv_step, Vh) + _reverse_cumsum(dksum, axis=2)
        dVh = np.einsum("bhsfd,bhsf->bhsd", dkv_step, Kp)
    else:
        dkv = np.matmul(Qp.transpose(0, 1, 3, 2), dnum)    # (B,H,Df,Dh)
        dQp = np.matmul(dnum, kv.transpose(0, 1, 3, 2)) + dden * ksum[:, :, None, :]
        dksum = np.sum(dden * Qp, axis=2)                  # (B,H,Df)
        dKp = np.matmul(Vh, dkv.transpose(0, 1, 3, 2)) + dksum[:, :, None, :]
        dVh = np.matmul(Kp, dkv)
    return dQp, dKp, dVh
