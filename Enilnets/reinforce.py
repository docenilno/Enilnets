import numpy as np

def _copy_layers(layers):
    """Copy only the ndarray values in each layer dict (fast) instead of
    copy.deepcopy'ing the whole nested Python structure (slow) -- every value
    that matters here is either an ndarray (needs its own buffer) or a
    plain/immutable value (safe to share)."""
    return [
        {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in layer.items()}
        for layer in layers
    ]

def Evolve(self, inputs, score_fn, noise=0.05, tries=10, sigma=1.0):
    """
    Evolutionary Strategy (ES). Perturbs network weights with Gaussian noise
    and keeps the best performing variant.
    """
    inputs = np.asarray(inputs, dtype=np.float64)
    best_score = score_fn(self.Forward(inputs))
    best_layers = _copy_layers(self.layers)
    base_layers = _copy_layers(self.layers)

    for _ in range(max(1, tries)):
        candidate = _copy_layers(base_layers)
        for layer in candidate:
            if "weights" in layer:
                layer["weights"] += np.random.normal(0, sigma * noise, layer["weights"].shape)
                if layer["type"] == "sparse":
                    layer["weights"] *= layer["mask"]
                layer["bias"] += np.random.normal(0, sigma * noise, layer["bias"].shape)
        self.layers = candidate
        score = score_fn(self.Forward(inputs))
        if score > best_score:
            best_score = score
            best_layers = _copy_layers(candidate)

    self.layers = best_layers
    return best_score

def compute_returns(rewards, gamma=0.99):
    """
    Compute discounted returns for a single episode.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    returns = np.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns

def Reinforce(self, states, actions, returns, action_type="discrete", std=1.0, normalize_returns=True):
    """
    Real REINFORCE (Monte-Carlo Policy Gradient).
    """
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions)
    returns_raw = np.asarray(returns, dtype=np.float64).reshape(-1, 1)
    returns = returns_raw.copy()

    if normalize_returns:
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

    out = self.Forward(states, training=True)
    batch_size = states.shape[0]

    if action_type == "discrete":
        actions = actions.astype(int)
        num_actions = out.shape[-1]
        one_hot = np.zeros((batch_size, num_actions), dtype=np.float64)
        one_hot[np.arange(batch_size), actions] = 1.0
        output_delta = (out - one_hot) * returns / batch_size
    elif action_type == "continuous":
        means = out
        actions = actions.reshape(means.shape)
        output_delta = -(actions - means) / (std ** 2) * returns / batch_size
    else:
        raise ValueError(f"Unknown action_type: {action_type}. Use 'discrete' or 'continuous'.")

    self.Backward(None, output_delta=output_delta)
    self.update()
    return float(np.mean(returns_raw))

def PPO(self, states, actions, old_log_probs, advantages, action_type="discrete",
        epsilon=0.2, std=1.0, value_targets=None, value_coeff=0.5, entropy_coeff=0.01,
        value_network=None):
    """
    Proximal Policy Optimization (PPO) update.

    Parameters
    ----------
    states : ndarray
        Observed states, shape (N, features)
    actions : ndarray
        Discrete: (N,) integer action indices.
        Continuous: (N, action_dim) continuous actions.
    old_log_probs : ndarray
        Log probabilities of actions under old policy, shape (N, 1)
    advantages : ndarray
        Advantage estimates, shape (N, 1)
    action_type : str
        "discrete" or "continuous"
    epsilon : float
        PPO clipping parameter
    std : float
        Fixed standard deviation for continuous Gaussian policy
    value_targets : ndarray or None
        Target values for the value function. Only used if `value_network`
        is also given -- otherwise accepted but ignored (no value head on
        `self` to train against it).
    value_coeff : float
        Coefficient for value loss
    entropy_coeff : float
        Coefficient for entropy bonus
    value_network : NeuralNet or None
        Optional separate value-function network (same input dim as the
        policy `self`, output dim 1, linear activation). When given together
        with `value_targets`, it's trained via MSE*value_coeff right after
        the policy update.
    """
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions)
    advantages = np.asarray(advantages, dtype=np.float64).reshape(-1, 1)
    old_log_probs = np.asarray(old_log_probs, dtype=np.float64).reshape(-1, 1)
    batch_size = states.shape[0]

    out = self.Forward(states, training=True)

    if action_type == "discrete":
        actions = actions.astype(int)
        num_actions = out.shape[-1]
        probs = out
        log_probs = np.log(np.clip(probs, 1e-12, 1.0))
        action_log_probs = log_probs[np.arange(batch_size), actions].reshape(-1, 1)

        # Entropy
        entropy = -np.sum(probs * log_probs, axis=-1, keepdims=True)

        ratio = np.exp(action_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1 - epsilon, 1 + epsilon) * advantages
        policy_loss = -np.minimum(surr1, surr2)

        # Approximate gradient for policy loss (vectorized: no per-sample loop)
        ratio_flat = ratio.reshape(-1)
        adv_flat = advantages.reshape(-1)
        rows = np.arange(batch_size)
        clipped = (ratio_flat < 1 - epsilon) & (adv_flat < 0)
        vals = np.where(clipped, 0.0, -adv_flat / (probs[rows, actions] + 1e-12) / batch_size)
        output_delta = np.zeros_like(out)
        output_delta[rows, actions] = vals

        # Add entropy gradient
        output_delta += entropy_coeff * (1 + log_probs) / batch_size

    elif action_type == "continuous":
        means = out
        actions = actions.reshape(means.shape)
        log_prob = -0.5 * ((actions - means) / std) ** 2 - 0.5 * np.log(2 * np.pi * std ** 2)
        action_log_probs = np.sum(log_prob, axis=-1, keepdims=True)

        ratio = np.exp(action_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1 - epsilon, 1 + epsilon) * advantages
        policy_loss = -np.minimum(surr1, surr2)

        output_delta = -(actions - means) / (std ** 2) * advantages / batch_size
    else:
        raise ValueError(f"Unknown action_type: {action_type}")

    self.Backward(None, output_delta=output_delta)
    self.update()

    if value_network is not None and value_targets is not None:
        value_targets = np.asarray(value_targets, dtype=np.float64).reshape(-1, 1)
        value_pred = value_network.Forward(states, training=True)
        value_delta = value_coeff * 2 * (value_pred - value_targets) / batch_size
        value_network.Backward(None, output_delta=value_delta)
        value_network.update()

    return float(np.mean(policy_loss))

def ActorCritic(self, states, actions, returns, values, action_type="discrete", std=1.0):
    """
    Actor-Critic with advantage estimation.

    Parameters
    ----------
    states : ndarray
        (N, features)
    actions : ndarray
        Discrete: (N,) or Continuous: (N, action_dim)
    returns : ndarray
        (N, 1) discounted returns
    values : ndarray
        (N, 1) predicted values from value network
    action_type : str
        "discrete" or "continuous"
    std : float
        Standard deviation for continuous actions
    """
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions)
    returns = np.asarray(returns, dtype=np.float64).reshape(-1, 1)
    values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    batch_size = states.shape[0]

    advantages = returns - values

    out = self.Forward(states, training=True)

    if action_type == "discrete":
        actions = actions.astype(int)
        num_actions = out.shape[-1]
        one_hot = np.zeros((batch_size, num_actions), dtype=np.float64)
        one_hot[np.arange(batch_size), actions] = 1.0
        output_delta = (out - one_hot) * advantages / batch_size
    elif action_type == "continuous":
        means = out
        actions = actions.reshape(means.shape)
        output_delta = -(actions - means) / (std ** 2) * advantages / batch_size
    else:
        raise ValueError(f"Unknown action_type: {action_type}")

    self.Backward(None, output_delta=output_delta)
    self.update()
    return float(np.mean(advantages ** 2))
