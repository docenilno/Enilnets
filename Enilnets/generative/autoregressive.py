import numpy as np
from ..base import NeuralNet

class AutoregressiveModel:
    """
    Autoregressive generative model (MADE-like) built on Enilnets.
    Generates data one dimension at a time, conditioning on previous dimensions.

    For images, generates pixel-by-pixel in raster scan order.
    For vectors, generates dimension-by-dimension.

    Parameters
    ----------
    data_dim : int
        Total number of dimensions (e.g., 784 for 28x28 MNIST)
    hidden_dims : list
        Hidden layer sizes
    data_shape : tuple or None
        If provided, reshapes output to this shape (e.g., (28, 28))
    activation : str
    learning_rate : float
    optimizer : str
    l2_lambda : float
    """
    def __init__(self, data_dim, hidden_dims=[512, 512], data_shape=None,
                 activation="swish", learning_rate=0.001, optimizer="adam", l2_lambda=0.0):
        self.data_dim = data_dim
        self.data_shape = data_shape
        self.activation = activation

        # Build autoregressive network: input -> hidden -> output (data_dim)
        # During training, we mask the output so each dim only sees previous dims
        self.network = NeuralNet(learning_rate=learning_rate, optimizer=optimizer, l2_lambda=l2_lambda)
        prev = data_dim
        for h in hidden_dims:
            self.network.add_dense(prev, h, activation=activation)
            prev = h
        self.network.add_dense(prev, data_dim, activation="linear")

    def _create_masks(self, batch_size):
        """
        Create causal masks for autoregressive property.
        mask[i,j] = 1 if j < i (dim i can see dims 0..i-1), else 0.
        """
        mask = np.tril(np.ones((self.data_dim, self.data_dim), dtype=np.float64), k=-1)
        return np.tile(mask[None, :, :], (batch_size, 1, 1))

    def forward(self, x, training=True):
        """
        Forward pass with causal masking.
        x: (batch, data_dim) or (batch, C, H, W)
        Returns logits for each dimension: (batch, data_dim)
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        elif x.ndim == 1:
            x = x.reshape(1, -1)

        # Causal masking: zero out future dimensions in input
        batch_size = x.shape[0]
        masks = self._create_masks(batch_size)
        x_masked = np.einsum('bij,bj->bi', masks, x)

        logits = self.network.Forward(x_masked, training=training)
        return logits

    def loss(self, x):
        """
        Compute autoregressive loss (MSE between predicted and actual values).
        Each dimension i predicts x_i given x_0...x_{i-1}.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        logits = self.forward(x, training=True)
        return float(np.mean((logits - x) ** 2))

    def train_step(self, x):
        """
        Single training step.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        batch_size = x.shape[0]

        logits = self.forward(x, training=True)
        delta = (logits - x) / batch_size

        self.network.Backward(np.zeros_like(x))
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
        """
        Train autoregressive model.
        """
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

    def generate(self, n_samples=1, shape=None):
        """
        Generate samples sequentially, one dimension at a time.
        """
        if shape is None:
            shape = (self.data_dim,)

        samples = np.zeros((n_samples, self.data_dim), dtype=np.float64)

        for i in range(self.data_dim):
            # Predict dimension i given all previous
            logits = self.forward(samples, training=False)
            # Sample from predicted distribution (Gaussian with mean=logits, std=1)
            samples[:, i] = logits[:, i] + np.random.randn(n_samples) * 0.1

        if self.data_shape is not None:
            samples = samples.reshape(n_samples, *self.data_shape)
        return samples

    def complete(self, partial_x, n_dims=None):
        """
        Complete a partial sample. partial_x has some known dimensions,
        the rest are filled in autoregressively.
        """
        partial_x = np.asarray(partial_x, dtype=np.float64).copy()
        if partial_x.ndim == 1:
            partial_x = partial_x.reshape(1, -1)

        if n_dims is None:
            # Find first unknown dimension
            n_dims = self.data_dim

        for i in range(n_dims, self.data_dim):
            logits = self.forward(partial_x, training=False)
            partial_x[:, i] = logits[:, i] + np.random.randn(partial_x.shape[0]) * 0.1

        if self.data_shape is not None:
            partial_x = partial_x.reshape(partial_x.shape[0], *self.data_shape)
        return partial_x
