from ..backend import np
from .. import backend

def reparameterize(mu, logvar):
    """
    VAE reparameterization trick: z = mu + sigma * eps
    """
    eps = np.random.randn(*mu.shape)
    return mu + np.exp(0.5 * logvar) * eps

def langevin_dynamics(energy_fn, x_init, n_steps=20, step_size=0.1, noise_scale=0.005):
    """
    Langevin Monte Carlo sampling for energy-based models.
    energy_fn: callable that takes x and returns (energy, grad_energy)
    """
    x = x_init.copy()
    for _ in range(n_steps):
        energy, grad = energy_fn(x)
        x = x - step_size * grad + np.random.randn(*x.shape) * noise_scale
    return x

def gaussian_sample(mean, std, shape=None):
    """Sample from N(mean, std^2)."""
    if shape is None:
        shape = mean.shape
    return mean + std * np.random.randn(*shape)

def uniform_sample(low, high, shape):
    """Sample from Uniform(low, high)."""
    return np.random.uniform(low, high, shape)

def gumbel_softmax_sample(logits, temperature=1.0, hard=False):
    """
    Gumbel-Softmax sampling for discrete latent variables.
    logits: (batch, n_classes)
    """
    gumbel = -np.log(-np.log(np.random.uniform(1e-12, 1.0, logits.shape)))
    y = logits + gumbel
    y_soft = np.exp(y / temperature) / np.sum(np.exp(y / temperature), axis=-1, keepdims=True)
    if hard:
        # Straight-through estimator
        y_hard = np.zeros_like(y_soft)
        y_hard[np.arange(y_soft.shape[0]), np.argmax(y_soft, axis=-1)] = 1.0
        return y_hard - y_soft + y_soft  # straight-through
    return y_soft

def random_mask(shape, ratio):
    """Generate a random boolean mask with given keep ratio."""
    return (np.random.rand(*shape) < ratio).astype(backend.default_dtype())

def top_p_sampling(logits, p=0.9, temperature=1.0):
    """
    Nucleus (top-p) sampling for discrete distributions.
    logits: (batch, vocab_size)
    """
    probs = np.exp(logits / temperature)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    sorted_probs = np.sort(probs, axis=-1)[:, ::-1]
    sorted_indices = np.argsort(probs, axis=-1)[:, ::-1]
    cumsum = np.cumsum(sorted_probs, axis=-1)
    mask = cumsum <= p
    mask[:, 0] = True  # always keep at least the top token

    result = np.zeros_like(probs)
    for i in range(probs.shape[0]):
        valid_indices = sorted_indices[i, mask[i]]
        valid_probs = sorted_probs[i, mask[i]]
        valid_probs = valid_probs / np.sum(valid_probs)
        choice = np.random.choice(valid_indices, size=1, p=valid_probs)[0]
        result[i, choice] = 1.0
    return result

def top_k_sampling(logits, k=10, temperature=1.0):
    """Top-k sampling for a single distribution: keep only the k highest
    logits, renormalize, and sample. logits: (vocab_size,)."""
    logits = np.asarray(logits, dtype=backend.default_dtype())
    k = min(k, logits.shape[-1])
    top_idx = np.argpartition(logits, -k)[-k:]
    top_logits = logits[top_idx] / temperature
    top_logits -= np.max(top_logits)
    probs = np.exp(top_logits)
    probs /= probs.sum()
    return int(np.random.choice(top_idx, size=1, p=probs)[0])

def compute_returns(rewards, gamma=0.99):
    """
    Compute discounted returns for a single episode.
    """
    rewards = np.asarray(rewards, dtype=backend.default_dtype())
    returns = np.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns

def gae(rewards, values, gamma=0.99, lambda_=0.95):
    """
    Generalized Advantage Estimation (GAE).
    rewards: (T,) array of step rewards
    values: (T+1,) array of value estimates (includes V(s_T+1))
    gamma: discount factor
    lambda_: GAE lambda parameter
    Returns advantages: (T,) and returns: (T,)
    """
    rewards = np.asarray(rewards, dtype=backend.default_dtype())
    values = np.asarray(values, dtype=backend.default_dtype())
    T = len(rewards)
    advantages = np.zeros(T, dtype=backend.default_dtype())
    gae_t = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae_t = delta + gamma * lambda_ * gae_t
        advantages[t] = gae_t
    returns = advantages + values[:T]
    return advantages, returns
