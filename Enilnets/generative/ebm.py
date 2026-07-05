import numpy as np
from ..base import NeuralNet
from .sampling import langevin_dynamics

class EnergyBasedModel:
    """
    Energy-Based Model (EBM) built on Enilnets.
    Learns an energy function E(x) such that low energy = high probability.
    Uses contrastive divergence with Langevin dynamics for sampling.

    Parameters
    ----------
    data_dim : int
        Dimension of data
    hidden_dims : list
        Hidden layer sizes for energy network
    activation : str
    learning_rate : float
    optimizer : str
    l2_lambda : float
    """
    def __init__(self, data_dim, hidden_dims=[512, 512], activation="swish",
                 learning_rate=0.001, optimizer="adam", l2_lambda=0.0):
        self.data_dim = data_dim

        # Energy network: maps data -> scalar energy
        # We use a network that outputs 1 value
        self.energy_net = NeuralNet(learning_rate=learning_rate, optimizer=optimizer, l2_lambda=l2_lambda)
        prev = data_dim
        for h in hidden_dims:
            self.energy_net.add_dense(prev, h, activation=activation)
            prev = h
        self.energy_net.add_dense(prev, 1, activation="linear")

    def energy(self, x):
        """
        Compute energy E(x). Lower energy = higher probability.
        x: (batch, data_dim) or (batch, C, H, W)
        Returns: (batch, 1)
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        elif x.ndim == 1:
            x = x.reshape(1, -1)
        return self.energy_net.Forward(x, training=True)

    def _energy_grad(self, x):
        """
        Compute gradient of energy w.r.t. x using finite differences.
        Returns energy and gradient.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        eps = 1e-4
        grad = np.zeros_like(x)
        e = self.energy(x)

        for i in range(x.shape[1]):
            x_plus = x.copy()
            x_plus[:, i] += eps
            e_plus = self.energy(x_plus)
            grad[:, i] = (e_plus - e)[:, 0] / eps

        return e, grad

    def train_step(self, x_data, n_cd_steps=10, step_size=0.1, noise_scale=0.005):
        """
        Contrastive divergence training step.

        1. Sample negative samples using Langevin dynamics starting from random noise
        2. Push down energy on data, push up energy on samples
        """
        x_data = np.asarray(x_data, dtype=np.float64)
        if x_data.ndim > 2:
            x_data = x_data.reshape(x_data.shape[0], -1)
        elif x_data.ndim == 1:
            x_data = x_data.reshape(1, -1)
        batch_size = x_data.shape[0]

        # Generate negative samples via Langevin dynamics
        x_neg = np.random.randn(*x_data.shape) * 0.5
        for _ in range(n_cd_steps):
            e, grad = self._energy_grad(x_neg)
            x_neg = x_neg - step_size * grad + np.random.randn(*x_neg.shape) * noise_scale

        # Energy on data and samples
        e_data = self.energy(x_data)
        e_neg = self.energy(x_neg)

        # Loss: push down data energy, push up sample energy
        # L = E(x_data) - E(x_neg)  (we want to minimize this)
        loss = float(np.mean(e_data - e_neg))

        # Backward for data (gradient = +1, push down)
        self.energy_net.Backward(np.ones((batch_size, 1)))
        self.energy_net.update()

        # Backward for negative samples (gradient = -1, push up)
        self.energy_net.Backward(-np.ones((batch_size, 1)))
        self.energy_net.update()

        return loss

    def Train(self, X_train, epochs=10, batch_size=64, n_cd_steps=10,
              step_size=0.1, noise_scale=0.005, verbose=True):
        """
        Train EBM with contrastive divergence.
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
                loss = self.train_step(batch, n_cd_steps, step_size, noise_scale)
                epoch_loss += loss * batch.shape[0]
            avg_loss = epoch_loss / n_samples
            history.append(avg_loss)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - EBM loss: {avg_loss:.4f}")
        return history

    def sample(self, n_samples=1, n_steps=100, step_size=0.1, noise_scale=0.005):
        """
        Generate samples using Langevin dynamics from random initialization.
        """
        x = np.random.randn(n_samples, self.data_dim) * 0.5
        for _ in range(n_steps):
            e, grad = self._energy_grad(x)
            x = x - step_size * grad + np.random.randn(*x.shape) * noise_scale
        return x

    def score(self, x):
        """
        Compute score function: -grad_x log p(x) = grad_x E(x)
        """
        _, grad = self._energy_grad(x)
        return grad
