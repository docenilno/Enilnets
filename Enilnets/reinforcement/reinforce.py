"""NeuralNet.Evolve/Reinforce/PPO/ActorCritic: gradient-free and
policy-gradient reinforcement-learning training methods."""
from typing import Any, Dict, List, Optional, Tuple

from ..core.backend import np
from ..core import backend

def _copy_layers(layers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Copy only the ndarray values in each layer dict (fast) instead of
    copy.deepcopy'ing the whole nested Python structure (slow) -- every value
    that matters here is either an ndarray (needs its own buffer) or a
    plain/immutable value (safe to share)."""
    return [
        {k: (v.copy() if backend.is_array(v) else v) for k, v in layer.items()}
        for layer in layers
    ]

def Evolve(self: Any, inputs: Any, score_fn: Any, noise: float = 0.05, tries: int = 10,
           sigma: float = 1.0) -> float:
    """
    Gradient-free weight perturbation search: draws `tries` independent
    Gaussian-noise perturbations of the CURRENT (fixed) weights and keeps
    whichever single variant scores best. Named/documented as "Evolutionary
    Strategy (ES)" in earlier versions, but it's closer to repeated-
    perturbation hill-climbing from a fixed base point than canonical ES
    (which typically draws a population of perturbations each generation
    and applies a sample-averaged update in the direction of higher-scoring
    ones, rather than just keeping the single best raw sample). Use when
    you don't have (or don't trust) a gradient signal at all.
    """
    inputs = np.asarray(inputs, dtype=backend.default_dtype())
    best_score = score_fn(self.Forward(inputs))
    best_layers = _copy_layers(self.layers)
    base_layers = _copy_layers(self.layers)

    for _ in range(max(1, tries)):
        candidate = _copy_layers(base_layers)
        for layer in candidate:
            if "weights" in layer:
                layer["weights"] += np.random.normal(0, sigma * noise, layer["weights"].shape).astype(backend.default_dtype())
                if layer["type"] == "sparse":
                    layer["weights"] *= layer["mask"]
                layer["bias"] += np.random.normal(0, sigma * noise, layer["bias"].shape).astype(backend.default_dtype())
        self.layers = candidate
        score = score_fn(self.Forward(inputs))
        if score > best_score:
            best_score = score
            best_layers = _copy_layers(candidate)

    self.layers = best_layers
    return best_score

def compute_returns(rewards: Any, gamma: float = 0.99) -> Any:
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

def gae(rewards: Any, values: Any, gamma: float = 0.99, lambda_: float = 0.95) -> Tuple[Any, Any]:
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

def Reinforce(self: Any, states: Any, actions: Any, returns: Any, action_type: str = "discrete",
              std: float = 1.0, normalize_returns: bool = True) -> float:
    """
    Real REINFORCE (Monte-Carlo Policy Gradient).
    """
    states = np.asarray(states, dtype=backend.default_dtype())
    actions = np.asarray(actions)
    returns_raw = np.asarray(returns, dtype=backend.default_dtype()).reshape(-1, 1)
    returns = returns_raw.copy()

    if normalize_returns:
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

    out = self.Forward(states, training=True)
    batch_size = states.shape[0]

    if action_type == "discrete":
        actions = actions.astype(int)
        num_actions = out.shape[-1]
        one_hot = np.zeros((batch_size, num_actions), dtype=backend.default_dtype())
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

def PPO(self: Any, states: Any, actions: Any, old_log_probs: Any, advantages: Any,
        action_type: str = "discrete", epsilon: float = 0.2, std: float = 1.0,
        value_targets: Optional[Any] = None, value_coeff: float = 0.5, entropy_coeff: float = 0.01,
        value_network: Optional[Any] = None) -> float:
    """Proximal Policy Optimization update.

    `states` (N, features); `actions` (N,) integer indices for
    action_type="discrete" or (N, action_dim) for "continuous";
    `old_log_probs` and `advantages` both (N, 1). `std` is the fixed
    Gaussian spread for continuous policies.

    `value_network` is an optional separate value head (same input dim,
    output dim 1, linear); given together with `value_targets` it is trained
    by MSE * value_coeff right after the policy update. `value_targets` alone
    is accepted but ignored -- there is no value head on `self` to train."""
    states = np.asarray(states, dtype=backend.default_dtype())
    actions = np.asarray(actions)
    advantages = np.asarray(advantages, dtype=backend.default_dtype()).reshape(-1, 1)
    old_log_probs = np.asarray(old_log_probs, dtype=backend.default_dtype()).reshape(-1, 1)
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

        # The clipped surrogate's min() picks the constant (ratio-
        # independent) clip(ratio)*A branch -- zero local gradient --
        # whenever clipping is actually binding, which happens on EITHER
        # side depending on the advantage's sign: A>0 & ratio>1+epsilon,
        # or A<0 & ratio<1-epsilon.
        ratio_flat = ratio.reshape(-1)
        adv_flat = advantages.reshape(-1)
        rows = np.arange(batch_size)
        clipped = ((ratio_flat > 1 + epsilon) & (adv_flat > 0)) | \
                  ((ratio_flat < 1 - epsilon) & (adv_flat < 0))

        # In the unclipped region, ratio_i = p_i / p_old_i (p_old_i fixed,
        # from old_log_probs), so d(-ratio_i*A_i)/dp_i = -A_i/p_old_i -- a
        # genuine constant w.r.t. p_i itself (the ratio_i/p_i factor a
        # naive product rule would suggest cancels exactly).
        p_old_flat = np.exp(old_log_probs.reshape(-1))
        dL_dp = np.zeros_like(out)
        dL_dp[rows, actions] = np.where(clipped, 0.0, -adv_flat / (p_old_flat + 1e-12))
        # Entropy bonus depends on every class's probability, not just the
        # taken action: d(entropy_coeff*(1+log p_k))/dp_k for every k.
        dL_dp += entropy_coeff * (1 + log_probs)

        # dL_dp above is the gradient w.r.t. the softmax OUTPUT (post-
        # activation probabilities), but Backward(output_delta=...) expects
        # the gradient w.r.t. the PRE-activation logits -- convert via the
        # standard softmax Jacobian-vector product (same identity used for
        # attention's softmax backward): dz_j = p_j*(g_j - sum_k p_k*g_k).
        # (REINFORCE's simpler (probs - one_hot)*scale shortcut only works
        # because that loss is a pure -log(p_i) form, where the softmax
        # Jacobian's p_i factor exactly cancels the 1/p_i from d(-log p_i)/
        # dp_i -- PPO's surrogate objective isn't log-shaped, so no such
        # shortcut applies here and the full Jacobian is needed.)
        output_delta = out * (dL_dp - np.sum(dL_dp * out, axis=-1, keepdims=True))
        output_delta = output_delta / batch_size

    elif action_type == "continuous":
        means = out
        actions = actions.reshape(means.shape)
        log_prob = -0.5 * ((actions - means) / std) ** 2 - 0.5 * np.log(2 * np.pi * std ** 2)
        action_log_probs = np.sum(log_prob, axis=-1, keepdims=True)

        ratio = np.exp(action_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1 - epsilon, 1 + epsilon) * advantages
        policy_loss = -np.minimum(surr1, surr2)

        # d(-ratio*A)/d(means_d) = -A * d(ratio)/d(means_d)
        #                        = -A * ratio * (actions_d - means_d) / std^2
        # (chain rule through ratio = exp(sum_d log_prob_d - old_log_probs),
        # d(log_prob_d)/d(means_d) = (actions_d - means_d)/std^2).
        clipped = ((ratio > 1 + epsilon) & (advantages > 0)) | \
                  ((ratio < 1 - epsilon) & (advantages < 0))
        grad_unclipped = -advantages * ratio * (actions - means) / (std ** 2)
        output_delta = np.where(clipped, 0.0, grad_unclipped) / batch_size
    else:
        raise ValueError(f"Unknown action_type: {action_type}")

    self.Backward(None, output_delta=output_delta)
    self.update()

    if value_network is not None and value_targets is not None:
        value_targets = np.asarray(value_targets, dtype=backend.default_dtype()).reshape(-1, 1)
        value_pred = value_network.Forward(states, training=True)
        value_delta = value_coeff * 2 * (value_pred - value_targets) / batch_size
        value_network.Backward(None, output_delta=value_delta)
        value_network.update()

    return float(np.mean(policy_loss))

def ActorCritic(self: Any, states: Any, actions: Any, returns: Any, values: Any,
                action_type: str = "discrete", std: float = 1.0) -> float:
    """Actor-Critic update with advantage estimation.

    `states` (N, features); `actions` (N,) integer indices when
    action_type="discrete", (N, action_dim) when "continuous"; `returns` and
    `values` both (N, 1). `std` is the fixed Gaussian spread for continuous
    policies."""
    states = np.asarray(states, dtype=backend.default_dtype())
    actions = np.asarray(actions)
    returns = np.asarray(returns, dtype=backend.default_dtype()).reshape(-1, 1)
    values = np.asarray(values, dtype=backend.default_dtype()).reshape(-1, 1)
    batch_size = states.shape[0]

    advantages = returns - values

    out = self.Forward(states, training=True)

    if action_type == "discrete":
        actions = actions.astype(int)
        num_actions = out.shape[-1]
        one_hot = np.zeros((batch_size, num_actions), dtype=backend.default_dtype())
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
