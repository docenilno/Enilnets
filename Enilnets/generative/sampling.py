from typing import Any, Optional, Tuple

from ..core.backend import np
from ..core import backend
from ..reinforcement.reinforce import compute_returns, gae

def reparameterize(mu: Any, logvar: Any) -> Any:
    """
    VAE reparameterization trick: z = mu + sigma * eps
    """
    eps = np.random.randn(*mu.shape).astype(backend.default_dtype())
    return mu + np.exp(0.5 * logvar) * eps

def langevin_dynamics(energy_fn: Any, x_init: Any, n_steps: int = 20, step_size: float = 0.1,
                       noise_scale: float = 0.005) -> Any:
    """
    Langevin Monte Carlo sampling for energy-based models.
    energy_fn: callable that takes x and returns (energy, grad_energy)
    """
    x = x_init.copy()
    for _ in range(n_steps):
        energy, grad = energy_fn(x)
        noise = np.random.randn(*x.shape).astype(backend.default_dtype())
        x = x - step_size * grad + noise * noise_scale
    return x

def gaussian_sample(mean: Any, std: Any, shape: Optional[Tuple[int, ...]] = None) -> Any:
    """Sample from N(mean, std^2)."""
    if shape is None:
        shape = mean.shape
    return mean + std * np.random.randn(*shape).astype(backend.default_dtype())

def uniform_sample(low: float, high: float, shape: Tuple[int, ...]) -> Any:
    """Sample from Uniform(low, high)."""
    return np.random.uniform(low, high, shape).astype(backend.default_dtype())

def gumbel_softmax_sample(logits: Any, temperature: float = 1.0, hard: bool = False) -> Any:
    """
    Gumbel-Softmax sampling for discrete latent variables.
    logits: (batch, n_classes)
    """
    u = np.random.uniform(1e-12, 1.0, logits.shape).astype(backend.default_dtype())
    gumbel = -np.log(-np.log(u))
    y = logits + gumbel
    y_soft = np.exp(y / temperature) / np.sum(np.exp(y / temperature), axis=-1, keepdims=True)
    if hard:
        # Straight-through estimator
        y_hard = np.zeros_like(y_soft)
        y_hard[np.arange(y_soft.shape[0]), np.argmax(y_soft, axis=-1)] = 1.0
        return y_hard - y_soft + y_soft  # straight-through
    return y_soft

def random_mask(shape: Tuple[int, ...], ratio: float) -> Any:
    """Generate a random boolean mask with given keep ratio."""
    return (np.random.rand(*shape) < ratio).astype(backend.default_dtype())

def nucleus_renormalize(probs: Any, top_p: float) -> Any:
    """Given a (batch, vocab_size) probability array (each row already
    summing to 1), zero out every entry outside that row's top-p nucleus and
    renormalize what's left. The single shared implementation of the actual
    top-p masking logic -- both top_p_sampling (below) and
    TextGenerator._sample_token in text_generation.py need exactly this same
    masking step, just with different final sampling mechanics on top of it
    (a vectorized batched draw here vs. a single-distribution
    temperature/top-k/top-p combo there), so this is factored out rather
    than each keeping its own copy of the sort/cumsum/cutoff logic."""
    sorted_probs = np.sort(probs, axis=-1)[..., ::-1]
    sorted_indices = np.argsort(probs, axis=-1)[..., ::-1]
    cumsum = np.cumsum(sorted_probs, axis=-1)
    # Nucleus definition (Holtzman et al.): the SMALLEST set whose
    # cumulative probability reaches top_p -- so the token that crosses the
    # threshold is kept. "Cumulative mass BEFORE this token < top_p" keeps
    # exactly that set (and always keeps the top token, since its
    # before-mass is 0). The previous `cumsum <= top_p` form dropped the
    # crossing token, leaving a set summing to less than top_p.
    mask = (cumsum - sorted_probs) < top_p

    masked_sorted = sorted_probs * mask
    out = np.zeros_like(probs)
    row_idx = np.arange(probs.shape[0])[:, None]
    out[row_idx, sorted_indices] = masked_sorted
    return out / np.sum(out, axis=-1, keepdims=True)

def top_p_sampling(logits: Any, p: float = 0.9, temperature: float = 1.0) -> Any:
    """
    Nucleus (top-p) sampling for discrete distributions.
    logits: (batch, vocab_size)
    """
    # Subtract the row max before exp (softmax identity): mathematically a
    # no-op, but avoids overflow to inf for large logits (top_k_sampling
    # already did this; top_p was the one place that didn't).
    probs = np.exp((logits - np.max(logits, axis=-1, keepdims=True)) / temperature)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)

    # Renormalize over just the nucleus, then draw one categorical sample per
    # row via vectorized inverse-CDF sampling (a uniform threshold per row
    # against the renormalized cumsum) -- avoids a Python loop over the batch
    # dimension entirely (the earlier per-row np.random.choice loop).
    masked_probs = nucleus_renormalize(probs, p)
    masked_cumsum = np.cumsum(masked_probs, axis=-1)
    u = np.random.rand(logits.shape[0], 1)
    chosen = np.argmax(masked_cumsum >= u, axis=-1)

    result = np.zeros_like(probs)
    result[np.arange(logits.shape[0]), chosen] = 1.0
    return result

def top_k_renormalize(probs: Any, k: int) -> Any:
    """Given a (vocab_size,) probability array (already summing to 1),
    zero out every entry outside the top-k and renormalize what's left.
    Single-distribution only (unlike top_p_sampling/nucleus_renormalize,
    which are batched over (batch, vocab_size)) -- the shared
    implementation of the actual top-k masking step, used by both
    top_k_sampling (below) and TextGenerator._sample_token in
    text_generation.py."""
    k = min(k, probs.shape[-1])
    keep = np.argpartition(probs, -k)[-k:]
    mask = np.zeros_like(probs)
    mask[keep] = probs[keep]
    return mask / mask.sum()

def top_k_sampling(logits: Any, k: int = 10, temperature: float = 1.0) -> int:
    """Top-k sampling for a single distribution: keep only the k highest
    logits, renormalize, and sample. logits: (vocab_size,)."""
    logits = np.asarray(logits, dtype=backend.default_dtype())
    probs = np.exp((logits - np.max(logits)) / temperature)
    probs /= probs.sum()
    probs = top_k_renormalize(probs, k)
    return int(np.random.choice(len(probs), size=1, p=probs)[0])
