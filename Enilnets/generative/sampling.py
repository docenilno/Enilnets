import numpy as np

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
    return (np.random.rand(*shape) < ratio).astype(np.float64)


def compute_returns(rewards, gamma=0.99):
    """
    Compute discounted returns for a single episode.
    (Re-exported from reinforce module for convenience)
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    returns = np.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns
