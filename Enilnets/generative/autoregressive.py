import numpy as np
from ..base import NeuralNet

class AutoregressiveModel:
    def __init__(self, data_dim, hidden_dims=[512, 512], data_shape=None,
                 activation="swish", learning_rate=0.001, optimizer="adam", l2_lambda=0.0,
                 num_classes=256,  # used when discrete=True, e.g. pixel values 0-255
                 discrete=False):  # discrete (classification-style) vs continuous output
        self.data_dim = data_dim
        self.data_shape = data_shape
        self.activation = activation
        self.num_classes = num_classes
        self.discrete = discrete

        self.network = NeuralNet(learning_rate=learning_rate, optimizer=optimizer, l2_lambda=l2_lambda)
        prev = data_dim
        for h in hidden_dims:
            self.network.add_dense(prev, h, activation=activation)
            prev = h

        if discrete:
            # Output logits for each dimension and each class
            self.network.add_dense(prev, data_dim * num_classes, activation="linear")
        else:
            self.network.add_dense(prev, data_dim, activation="linear")

        # Precomputed once (it never changes): rebuilding + tiling this per
        # batch on every forward() call was pure overhead.
        self._mask = np.tril(np.ones((self.data_dim, self.data_dim), dtype=np.float64), k=-1)

    def _create_masks(self, batch_size):
        """Kept for backward compatibility; prefer the cached self._mask
        (einsum broadcasts it over the batch dimension for free)."""
        return np.tile(self._mask[None, :, :], (batch_size, 1, 1))

    def forward(self, x, training=True):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        elif x.ndim == 1:
            x = x.reshape(1, -1)

        x_masked = np.einsum('ij,bj->bi', self._mask, x)

        logits = self.network.Forward(x_masked, training=training)

        if self.discrete:
            logits = logits.reshape(x.shape[0], self.data_dim, self.num_classes)

        return logits

    def loss(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        logits = self.forward(x, training=True)

        if self.discrete:
            # Cross-entropy loss for discrete data
            # x should be integer class indices (0 to num_classes-1)
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

    def train_step(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        batch_size = x.shape[0]

        logits = self.forward(x, training=True)

        if self.discrete:
            x_int = np.clip(np.round(x * (self.num_classes - 1)).astype(np.int32), 0, self.num_classes - 1)
            # Gradient of cross-entropy w.r.t logits
            logit_max = np.max(logits, axis=-1, keepdims=True)
            exp_logits = np.exp(logits - logit_max)
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            # One-hot encoding of true classes
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(batch_size)[:, None], np.arange(self.data_dim), x_int] = 1.0
            delta = (probs - one_hot) / batch_size
            delta = delta.reshape(batch_size, -1)
        else:
            # loss = mean((logits - x)**2) => dL/dlogits = 2*(logits - x)/N
            delta = 2 * (logits - x) / batch_size

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
                from ..activations import derivative
                activation_input = self.network.pre_activations[l+1] if self.network.pre_activations[l+1] is not None else self.network.outputs[l+1]
                self.network.deltas[l] = err * derivative(curr.get("activation", "linear"), activation_input)
            else:
                self.network.deltas[l] = err

        self.network.update()
        return self.loss(x)

    def Train(self, X_train, epochs=10, batch_size=64, verbose=True):
        X = np.asarray(X_train, dtype=np.float64)
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

    def generate(self, n_samples=1, shape=None, temperature=1.0):
        """
        Generate samples autoregressively.

        Parameters
        ----------
        n_samples : int
        shape : tuple or None
        temperature : float
            Sampling temperature (higher = more random)
        """
        if shape is None:
            shape = (self.data_dim,)

        samples = np.zeros((n_samples, self.data_dim), dtype=np.float64)

        for i in range(self.data_dim):
            logits = self.forward(samples, training=False)

            if self.discrete:
                # Sample from categorical distribution
                logit_i = logits[:, i, :] / temperature
                logit_max = np.max(logit_i, axis=-1, keepdims=True)
                exp_logits = np.exp(logit_i - logit_max)
                probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                # Sample
                samples[:, i] = np.array([np.random.choice(self.num_classes, p=p) for p in probs]) / (self.num_classes - 1)
            else:
                # For continuous: sample from Gaussian with mean = logits and std = temperature
                samples[:, i] = logits[:, i] + np.random.randn(n_samples) * temperature

        if self.data_shape is not None:
            samples = samples.reshape(n_samples, *self.data_shape)
        return samples

    def complete(self, partial_x, n_dims=None, temperature=1.0):
        """
        Complete a partial sample autoregressively.

        Parameters
        ----------
        partial_x : ndarray
            Partial sample with some dimensions filled.
        n_dims : int or None
            Number of dimensions already filled.
        temperature : float
            Sampling temperature.
        """
        partial_x = np.asarray(partial_x, dtype=np.float64).copy()
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
                partial_x[:, i] = np.array([np.random.choice(self.num_classes, p=p) for p in probs]) / (self.num_classes - 1)
            else:
                partial_x[:, i] = logits[:, i] + np.random.randn(partial_x.shape[0]) * temperature

        if self.data_shape is not None:
            partial_x = partial_x.reshape(partial_x.shape[0], *self.data_shape)
        return partial_x

    def log_prob(self, x):
        """Log probability of data under the model."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        logits = self.forward(x, training=False)

        if self.discrete:
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
