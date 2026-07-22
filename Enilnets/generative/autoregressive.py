from typing import Any, List, Optional, Tuple

from ..core.backend import np
from ..core import backend
from ..core.base import NeuralNet
from ..nn.weight_init import init_weights

class AutoregressiveModel:
    def __init__(self, data_dim: int, hidden_dims: List[int] = [512, 512],
                 data_shape: Optional[Tuple[int, ...]] = None,
                 activation: str = "swish", learning_rate: float = 0.001, optimizer: str = "adam",
                 l2_lambda: float = 0.0,
                 # used when discrete=True: the number of classes each dimension
                 # is quantized into (e.g. 256 for 8-bit pixel values). Inputs to
                 # loss/train_step/log_prob must be scaled to [0, 1] regardless
                 # of num_classes -- e.g. raw 0-255 pixel values must be divided
                 # by 255 first, NOT passed as raw integers -- they're rescaled
                 # internally to 0..num_classes-1 (and validated; see
                 # _check_discrete_range below).
                 num_classes: int = 256,
                 discrete: bool = False) -> None:  # discrete (classification-style) vs continuous output
        self.data_dim = data_dim
        self.data_shape = data_shape
        self.activation = activation
        self.num_classes = num_classes
        self.discrete = discrete

        # MADE (Germain et al. 2015): instead of masking the *input* (which
        # leaks -- a cumulative sum fed whole into an ordinary unmasked MLP
        # lets later outputs recover earlier x_i values from the differences
        # between adjacent masked-input entries), each layer gets its own
        # connectivity mask on its WEIGHTS, so the autoregressive property
        # ("output i depends only on inputs < i") holds by construction
        # regardless of what the hidden layers compute.
        #
        # Each unit (input, hidden, or output) is assigned a "degree" in
        # 0..data_dim-1. A connection unit_a -> unit_b is only allowed if
        # degree(a) <= degree(b) for hidden-to-hidden connections, or
        # degree(a) < degree(b) for the final hidden-to-output connection
        # (strict, so output d can see input d-1 but never input d itself).
        in_degrees = np.arange(data_dim)
        # Cycle 0..data_dim-2 across each hidden layer's units so every valid
        # degree is represented (data_dim-1 is excluded: a hidden unit of
        # degree data_dim-1 could only ever feed the -- nonexistent -- output
        # beyond the last dimension, so it'd be dead weight).
        modulus = max(data_dim - 1, 1)
        hidden_degrees = [np.arange(h) % modulus for h in hidden_dims]

        # Build each layer's own weights directly (rather than via add_sparse,
        # which applies its OWN random connectivity mask by default -- letting
        # that compose with the MADE mask below would randomly zero out some
        # otherwise-valid MADE connections, silently breaking the "output i
        # depends on ALL inputs < i" guarantee for whichever paths happened to
        # be hit).
        self.network = NeuralNet(learning_rate=learning_rate, optimizer=optimizer, l2_lambda=l2_lambda)
        prev = data_dim
        prev_degrees = in_degrees
        for h, degrees in zip(hidden_dims, hidden_degrees):
            mask = (degrees[:, None] >= prev_degrees[None, :]).astype(backend.default_dtype())
            w, b = init_weights(prev, h, method="xavier_uniform")
            self.network.layers.append({
                "type": "sparse", "weights": w * mask, "bias": b, "mask": mask,
                "activation": activation, "activation_params": {},
            })
            self.network._last_width = h
            prev = h
            prev_degrees = degrees

        out_degrees = in_degrees  # output d has the same degree as input d
        out_mask = (out_degrees[:, None] > prev_degrees[None, :]).astype(backend.default_dtype())  # strict
        out_width = data_dim * num_classes if discrete else data_dim
        if discrete:
            # Output logits for each dimension and each class -- every class
            # column for dimension d shares dimension d's connectivity row.
            out_mask = np.repeat(out_mask, num_classes, axis=0)
        w, b = init_weights(prev, out_width, method="xavier_uniform")
        self.network.layers.append({
            "type": "sparse", "weights": w * out_mask, "bias": b, "mask": out_mask,
            "activation": "linear", "activation_params": {},
        })
        self.network._last_width = out_width

    def _check_discrete_range(self, x: Any) -> None:
        """loss/train_step/log_prob rescale x (assumed in [0,1]) to integer
        class indices via round(x * (num_classes-1)) -- silently assumed,
        never validated. Passing raw un-normalized data (e.g. actual 0-255
        pixel values instead of pixel/255) doesn't error: x_int just clips
        to num_classes-1 for nearly every value, silently training a
        degenerate model with a plausible-looking loss curve instead of a
        revealing crash. Only called when self.discrete."""
        x_min, x_max = float(np.min(x)), float(np.max(x))
        if x_min < -1e-6 or x_max > 1 + 1e-6:
            raise ValueError(
                f"AutoregressiveModel(discrete=True) expects inputs scaled to "
                f"[0, 1] (rescaled internally to 0..num_classes-1={self.num_classes - 1}), "
                f"but got values in [{x_min}, {x_max}]. If this is e.g. raw 0-255 "
                f"pixel data, divide by 255 first."
            )

    def forward(self, x: Any, training: bool = True) -> Any:
        x = np.asarray(x, dtype=backend.default_dtype())
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        elif x.ndim == 1:
            x = x.reshape(1, -1)

        logits = self.network.Forward(x, training=training)

        if self.discrete:
            logits = logits.reshape(x.shape[0], self.data_dim, self.num_classes)

        return logits

    def loss(self, x: Any) -> float:
        x = np.asarray(x, dtype=backend.default_dtype())
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        logits = self.forward(x, training=True)

        if self.discrete:
            self._check_discrete_range(x)
            # Cross-entropy loss for discrete data
            x_int = np.clip(np.round(x * (self.num_classes - 1)).astype(np.int32), 0, self.num_classes - 1)
            batch_size = x.shape[0]
            # Gather logits for true classes
            logit_true = logits[np.arange(batch_size)[:, None], np.arange(self.data_dim), x_int]
            # Log-sum-exp for numerical stability
            logit_max = np.max(logits, axis=-1, keepdims=True)
            log_sum_exp = logit_max.squeeze(-1) + np.log(np.sum(np.exp(logits - logit_max), axis=-1))
            loss = -np.mean(logit_true - log_sum_exp)
            return float(loss)
        else:
            # MSE for continuous data (original behavior)
            return float(np.mean((logits - x) ** 2))

    def train_step(self, x: Any) -> float:
        x = np.asarray(x, dtype=backend.default_dtype())
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        batch_size = x.shape[0]

        logits = self.forward(x, training=True)

        if self.discrete:
            self._check_discrete_range(x)
            x_int = np.clip(np.round(x * (self.num_classes - 1)).astype(np.int32), 0, self.num_classes - 1)
            # Gradient of cross-entropy w.r.t logits
            logit_max = np.max(logits, axis=-1, keepdims=True)
            exp_logits = np.exp(logits - logit_max)
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            # One-hot encoding of true classes
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(batch_size)[:, None], np.arange(self.data_dim), x_int] = 1.0
            # loss() averages -mean(logit_true - log_sum_exp) over BOTH batch
            # and data_dim (batch_size*data_dim positions total, one
            # cross-entropy term per dimension), so the gradient must divide
            # by batch_size*data_dim, not batch_size alone -- dividing by
            # batch_size only makes the gradient data_dim times too large
            # relative to the reported loss scale.
            delta = (probs - one_hot) / (batch_size * self.data_dim)
            delta = delta.reshape(batch_size, -1)
        else:
            # loss = mean((logits - x)**2) averages over ALL elements
            # (batch_size*data_dim), so dL/dlogits = 2*(logits - x)/x.size,
            # not /batch_size alone (same class of bug as VAE/DiffusionModel).
            delta = 2 * (logits - x) / x.size

        # Dummy call only to populate self.deltas/self.outputs state; the manual
        # reversed loop below overwrites deltas[-1] and recomputes everything.
        # Use output_delta (not targets) so the placeholder shape always matches
        # the network's actual output (discrete mode outputs data_dim*num_classes
        # units, which does not broadcast against x's shape).
        self.network.Backward(None, output_delta=np.zeros_like(self.network.outputs[-1]))
        self.network.deltas[-1] = delta

        for l in range(len(self.network.layers) - 2, -1, -1):
            nxt = self.network.layers[l + 1]
            next_delta = self.network.deltas[l + 1]
            if nxt["type"] in ("dense", "sparse"):
                err = np.dot(next_delta, nxt["weights"])
            else:
                err = next_delta
            curr = self.network.layers[l]
            if curr["type"] in ("dense", "sparse", "conv2d"):
                from ..nn.activations import derivative
                activation_input = self.network.pre_activations[l+1] if self.network.pre_activations[l+1] is not None else self.network.outputs[l+1]
                self.network.deltas[l] = err * derivative(curr.get("activation", "linear"), activation_input)
            else:
                self.network.deltas[l] = err

        self.network.update()
        return self.loss(x)

    def Train(self, X_train: Any, epochs: int = 10, batch_size: int = 64, verbose: bool = True) -> List[float]:
        X = np.asarray(X_train, dtype=backend.default_dtype())
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        n_samples = X.shape[0]
        history = []
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            for i in range(0, n_samples, batch_size):
                batch = X[indices[i:i+batch_size]]
                loss = self.train_step(batch)
                epoch_loss += loss * batch.shape[0]
            avg_loss = epoch_loss / n_samples
            history.append(avg_loss)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - AR loss: {avg_loss:.4f}")
        return history

    def generate(self, n_samples: int = 1, shape: Optional[Tuple[int, ...]] = None,
                 temperature: float = 1.0) -> Any:
        """Generate `n_samples` samples autoregressively. `temperature` scales the
        sampling randomness; `shape` reshapes the output."""
        if shape is None:
            shape = (self.data_dim,)

        samples = np.zeros((n_samples, self.data_dim), dtype=backend.default_dtype())

        for i in range(self.data_dim):
            logits = self.forward(samples, training=False)

            if self.discrete:
                # Sample from categorical distribution
                logit_i = logits[:, i, :] / temperature
                logit_max = np.max(logit_i, axis=-1, keepdims=True)
                exp_logits = np.exp(logit_i - logit_max)
                probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                # Vectorized categorical sampling via inverse-CDF (cumsum +
                # a per-row uniform threshold) instead of a per-row Python
                # loop + np.random.choice -- same pattern already used by
                # sampling.top_p_sampling.
                cumsum = np.cumsum(probs, axis=-1)
                u = np.random.rand(n_samples, 1)
                chosen = np.argmax(cumsum >= u, axis=-1)
                samples[:, i] = chosen / (self.num_classes - 1)
            else:
                # For continuous: sample from Gaussian with mean = logits and std = temperature
                noise = np.random.randn(n_samples).astype(backend.default_dtype())
                samples[:, i] = logits[:, i] + noise * temperature

        if self.data_shape is not None:
            samples = samples.reshape(n_samples, *self.data_shape)
        return samples

    def sample(self, n_samples: int = 1, shape: Optional[Tuple[int, ...]] = None,
               temperature: float = 1.0) -> Any:
        """Alias for generate(), matching the sample()/generate() naming
        used by DiffusionModel/RealNVP/EnergyBasedModel/UNetDenoiser (and
        GAN, which offers both)."""
        return self.generate(n_samples, shape=shape, temperature=temperature)

    def complete(self, partial_x: Any, n_dims: Optional[int] = None, temperature: float = 1.0) -> Any:
        """Fill in the remaining dimensions of a partial sample autoregressively.
        `n_dims` is how many leading dimensions are already filled."""
        partial_x = np.asarray(partial_x, dtype=backend.default_dtype()).copy()
        if partial_x.ndim == 1:
            partial_x = partial_x.reshape(1, -1)

        if n_dims is None:
            n_dims = self.data_dim

        for i in range(n_dims, self.data_dim):
            logits = self.forward(partial_x, training=False)

            if self.discrete:
                logit_i = logits[:, i, :] / temperature
                logit_max = np.max(logit_i, axis=-1, keepdims=True)
                exp_logits = np.exp(logit_i - logit_max)
                probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                cumsum = np.cumsum(probs, axis=-1)
                u = np.random.rand(partial_x.shape[0], 1)
                chosen = np.argmax(cumsum >= u, axis=-1)
                partial_x[:, i] = chosen / (self.num_classes - 1)
            else:
                noise = np.random.randn(partial_x.shape[0]).astype(backend.default_dtype())
                partial_x[:, i] = logits[:, i] + noise * temperature

        if self.data_shape is not None:
            partial_x = partial_x.reshape(partial_x.shape[0], *self.data_shape)
        return partial_x

    def log_prob(self, x: Any) -> Any:
        """Log probability of data under the model."""
        x = np.asarray(x, dtype=backend.default_dtype())
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        logits = self.forward(x, training=False)

        if self.discrete:
            self._check_discrete_range(x)
            x_int = np.clip(np.round(x * (self.num_classes - 1)).astype(np.int32), 0, self.num_classes - 1)
            batch_size = x.shape[0]
            logit_true = logits[np.arange(batch_size)[:, None], np.arange(self.data_dim), x_int]
            logit_max = np.max(logits, axis=-1, keepdims=True)
            log_sum_exp = logit_max.squeeze(-1) + np.log(np.sum(np.exp(logits - logit_max), axis=-1))
            log_prob = np.sum(logit_true - log_sum_exp, axis=-1)
            return log_prob
        else:
            # Gaussian log-likelihood
            return -0.5 * np.sum((x - logits) ** 2, axis=-1)
