import copy
import numpy as np

def Evolve(self, inputs, score_fn, noise=0.05, tries=10, sigma=1.0):
    """
    Evolutionary Strategy (ES). Perturbs network weights with Gaussian noise
    and keeps the best performing variant. This was previously misnamed Reinforce.
    """
    inputs = np.asarray(inputs, dtype=np.float64)
    best_score = score_fn(self.Forward(inputs))
    best_layers = copy.deepcopy(self.layers)
    base_layers = copy.deepcopy(self.layers)

    for _ in range(max(1, tries)):
        candidate = copy.deepcopy(base_layers)
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
            best_layers = copy.deepcopy(candidate)

    self.layers = best_layers
    return best_score

def compute_returns(rewards, gamma=0.99):
    """
    Compute discounted returns for a single episode.

    Parameters
    ----------
    rewards : array-like
        1-D array of step rewards.
    gamma : float
        Discount factor (0.0 to 1.0).

    Returns
    -------
    returns : ndarray
        Discounted returns, same shape as rewards.
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

    Updates the policy network by following the gradient of expected reward.
    Works with the model's existing optimizer (Adam, SGD, RMSprop, Adagrad).

    Parameters
    ----------
    states : ndarray
        Observed states, shape (N, features) or (N, C, H, W).
    actions : ndarray
        Discrete: shape (N,) integer action indices.
        Continuous: shape (N, action_dim) continuous actions.
    returns : ndarray
        Discounted returns (rewards-to-go), shape (N,) or (N, 1).
    action_type : str
        "discrete" (softmax policy) or "continuous" (Gaussian policy).
    std : float
        Fixed standard deviation for continuous Gaussian policy.
    normalize_returns : bool
        Normalize returns to zero mean / unit variance for lower variance.

    Returns
    -------
    avg_return : float
        Average raw return for the batch.

    Notes
    -----
    - Discrete: final layer should use "softmax" activation.
    - Continuous: final layer should use "linear" activation.
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
        # Gradient of loss = -log_prob * R  w.r.t. logits
        # d_loss/d_z = (prob - one_hot) * R
        output_delta = (out - one_hot) * returns / batch_size
    elif action_type == "continuous":
        means = out
        actions = actions.reshape(means.shape)
        # Gradient of loss = -log_prob * R  w.r.t. mean (linear activation)
        # d_loss/d_mean = -(a - mean) / std^2 * R
        output_delta = -(actions - means) / (std ** 2) * returns / batch_size
    else:
        raise ValueError(f"Unknown action_type: {action_type}. Use 'discrete' or 'continuous'.")

    self.Backward(None, output_delta=output_delta)
    self.update()
    return float(np.mean(returns_raw))
