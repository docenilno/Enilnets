from ..backend import np
from .. import backend
from ..base import NeuralNet
from .sampling import langevin_dynamics

class EnergyBasedModel:
    def __init__(self, data_dim, hidden_dims=[512, 512], activation="swish",
                 learning_rate=0.001, optimizer="adam", l2_lambda=0.0,
                 persistent_cd=True, persistent_buffer_size=1000,
                 init_noise_scale=0.5):
        self.data_dim = data_dim
        self.persistent_cd = persistent_cd
        self.persistent_buffer_size = persistent_buffer_size
        self.init_noise_scale = init_noise_scale

        self.energy_net = NeuralNet(learning_rate=learning_rate, optimizer=optimizer, l2_lambda=l2_lambda)
        prev = data_dim
        for h in hidden_dims:
            self.energy_net.add_dense(prev, h, activation=activation)
            prev = h
        self.energy_net.add_dense(prev, 1, activation="linear")

        if self.persistent_cd:
            self.persistent_buffer = np.random.randn(persistent_buffer_size, data_dim) * init_noise_scale
            self.buffer_ptr = 0

    def energy(self, x):
        x = np.asarray(x, dtype=backend.default_dtype())
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        elif x.ndim == 1:
            x = x.reshape(1, -1)
        return self.energy_net.Forward(x, training=True)

    def _energy_grad(self, x):
        """Gradient of the scalar energy w.r.t. its input, via backprop."""
        x = np.asarray(x, dtype=backend.default_dtype())
        if x.ndim == 1:
            x = x.reshape(1, -1)

        e = self.energy(x)
        # Backprop with dL/de=1 per-sample to get d(energy)/dx as deltas[0]
        self.energy_net.Backward(None, output_delta=np.ones((x.shape[0], 1)))
        # The gradient of energy w.r.t input is in the first layer's delta
        first_layer = self.energy_net.layers[0]
        grad = np.dot(self.energy_net.deltas[0], first_layer["weights"])
        return e, grad

    def _sample_negative(self, batch_size, n_cd_steps=10, step_size=0.1, noise_scale=0.005):
        """Sample negative examples via Langevin dynamics, using persistent
        contrastive divergence (PCD) if enabled."""
        if self.persistent_cd:
            # Initialize from persistent buffer
            if batch_size <= self.persistent_buffer_size:
                idx = np.random.choice(self.persistent_buffer_size, batch_size, replace=False)
                x_neg = self.persistent_buffer[idx].copy()
            else:
                x_neg = self.persistent_buffer.copy()
                # Supplement with random if needed
                extra = batch_size - self.persistent_buffer_size
                x_neg = np.concatenate([x_neg, np.random.randn(extra, self.data_dim) * self.init_noise_scale], axis=0)
        else:
            x_neg = np.random.randn(batch_size, self.data_dim) * self.init_noise_scale

        for _ in range(n_cd_steps):
            e, grad = self._energy_grad(x_neg)
            x_neg = x_neg - step_size * grad + np.random.randn(*x_neg.shape) * noise_scale

        # Update persistent buffer
        if self.persistent_cd:
            n = min(batch_size, self.persistent_buffer_size)
            self.persistent_buffer[self.buffer_ptr:self.buffer_ptr + n] = x_neg[:n]
            self.buffer_ptr = (self.buffer_ptr + n) % self.persistent_buffer_size

        return x_neg

    def train_step(self, x_data, n_cd_steps=10, step_size=0.1, noise_scale=0.005):
        x_data = np.asarray(x_data, dtype=backend.default_dtype())
        if x_data.ndim > 2:
            x_data = x_data.reshape(x_data.shape[0], -1)
        elif x_data.ndim == 1:
            x_data = x_data.reshape(1, -1)
        batch_size = x_data.shape[0]

        # Sample negative examples
        x_neg = self._sample_negative(batch_size, n_cd_steps, step_size, noise_scale)

        e_data = self.energy(x_data)
        # Push down energy on data: dL/de_data = +1
        self.energy_net.Backward(None, output_delta=np.ones((batch_size, 1)) / batch_size)
        self.energy_net.update()

        e_neg = self.energy(x_neg)
        # Push up energy on negatives: dL/de_neg = -1
        self.energy_net.Backward(None, output_delta=-np.ones((batch_size, 1)) / batch_size)
        self.energy_net.update()

        loss = float(np.mean(e_data - e_neg))
        return loss

    def Train(self, X_train, epochs=10, batch_size=64, n_cd_steps=10,
              step_size=0.1, noise_scale=0.005, verbose=True):
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
                loss = self.train_step(batch, n_cd_steps, step_size, noise_scale)
                epoch_loss += loss * batch.shape[0]
            avg_loss = epoch_loss / n_samples
            history.append(avg_loss)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - EBM loss: {avg_loss:.4f}")
        return history

    def sample(self, n_samples=1, n_steps=100, step_size=0.1, noise_scale=0.005):
        x = np.random.randn(n_samples, self.data_dim) * self.init_noise_scale
        for _ in range(n_steps):
            e, grad = self._energy_grad(x)
            x = x - step_size * grad + np.random.randn(*x.shape) * noise_scale
        return x

    def score(self, x):
        _, grad = self._energy_grad(x)
        return grad
