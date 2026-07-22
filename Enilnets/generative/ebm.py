from typing import Any, List, Optional, Tuple

from ..core.backend import np
from ..core import backend
from ..core.base import NeuralNet
from .sampling import langevin_dynamics

class EnergyBasedModel:
    def __init__(self, data_dim: int, hidden_dims: List[int] = [512, 512], activation: str = "swish",
                 learning_rate: float = 0.001, optimizer: str = "adam", l2_lambda: float = 0.0,
                 persistent_cd: bool = True, persistent_buffer_size: int = 1000,
                 init_noise_scale: float = 0.5) -> None:
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
            self.persistent_buffer = np.random.randn(persistent_buffer_size, data_dim).astype(
                backend.default_dtype()) * init_noise_scale
            self.buffer_ptr = 0

    def energy(self, x: Any) -> Any:
        x = np.asarray(x, dtype=backend.default_dtype())
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        elif x.ndim == 1:
            x = x.reshape(1, -1)
        return self.energy_net.Forward(x, training=True)

    def _energy_grad(self, x: Any) -> Tuple[Any, Any]:
        """Gradient of the scalar energy w.r.t. its input, via backprop."""
        x = np.asarray(x, dtype=backend.default_dtype())
        if x.ndim == 1:
            x = x.reshape(1, -1)

        e = self.energy(x)
        # Backprop with dL/de=1 per-sample to get d(energy)/dx as deltas[0]
        self.energy_net.Backward(None, output_delta=np.ones((x.shape[0], 1), dtype=backend.default_dtype()))
        # The gradient of energy w.r.t input is in the first layer's delta
        first_layer = self.energy_net.layers[0]
        grad = np.dot(self.energy_net.deltas[0], first_layer["weights"])
        return e, grad

    def _sample_negative(self, batch_size: int, n_cd_steps: int = 10, step_size: float = 0.1,
                          noise_scale: float = 0.005) -> Any:
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
                extra_noise = np.random.randn(extra, self.data_dim).astype(backend.default_dtype())
                x_neg = np.concatenate([x_neg, extra_noise * self.init_noise_scale], axis=0)
        else:
            x_neg = np.random.randn(batch_size, self.data_dim).astype(backend.default_dtype()) * self.init_noise_scale

        for _ in range(n_cd_steps):
            e, grad = self._energy_grad(x_neg)
            step_noise = np.random.randn(*x_neg.shape).astype(backend.default_dtype())
            x_neg = x_neg - step_size * grad + step_noise * noise_scale

        # Update persistent buffer
        if self.persistent_cd:
            n = min(batch_size, self.persistent_buffer_size)
            self.persistent_buffer[self.buffer_ptr:self.buffer_ptr + n] = x_neg[:n]
            self.buffer_ptr = (self.buffer_ptr + n) % self.persistent_buffer_size

        return x_neg

    def train_step(self, x_data: Any, n_cd_steps: int = 10, step_size: float = 0.1,
                   noise_scale: float = 0.005) -> float:
        x_data = np.asarray(x_data, dtype=backend.default_dtype())
        if x_data.ndim > 2:
            x_data = x_data.reshape(x_data.shape[0], -1)
        elif x_data.ndim == 1:
            x_data = x_data.reshape(1, -1)
        batch_size = x_data.shape[0]

        # Sample negative examples
        x_neg = self._sample_negative(batch_size, n_cd_steps, step_size, noise_scale)

        # Standard contrastive divergence needs both terms' gradients
        # computed against the SAME weight snapshot and applied as one
        # combined update, so the positive and negative phases are
        # concatenated into a single batch (same pattern as
        # GAN._train_discriminator) rather than run as two separate
        # Backward+update() calls -- one forward pass, one gradient, one
        # update, both terms evaluated against identical weights.
        total = 2 * batch_size
        combined = np.concatenate([x_data, x_neg], axis=0)
        e_combined = self.energy_net.Forward(combined, training=True)
        e_data = e_combined[:batch_size]
        e_neg = e_combined[batch_size:]

        # Push down energy on data (dL/de_data = +1), push up on negatives
        # (dL/de_neg = -1), both scaled by the combined batch size.
        delta = np.concatenate([np.ones((batch_size, 1)), -np.ones((batch_size, 1))], axis=0) / total
        self.energy_net.Backward(None, output_delta=delta)
        self.energy_net.update()

        loss = float(np.mean(e_data - e_neg))
        return loss

    def Train(self, X_train: Any, epochs: int = 10, batch_size: int = 64, n_cd_steps: int = 10,
              step_size: float = 0.1, noise_scale: float = 0.005, verbose: bool = True,
              callbacks: Optional[List[Any]] = None) -> List[float]:
        """callbacks: optional list of duck-typed callback objects (same
        convention as TextGenerator.Train/NeuralNet.Train). Supported hooks:
          on_batch_end(epoch, batch_idx, loss, model=self) -- after every
            minibatch's train_step.
          on_epoch_end(epoch, logs, model=self) -- once per epoch, with
            logs={"loss": avg_loss}.
          on_train_end(history) -- once after the epoch loop.
        Missing methods are skipped (no error)."""
        X = np.asarray(X_train, dtype=backend.default_dtype())
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        n_samples = X.shape[0]
        history = []
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            for batch_idx_num, i in enumerate(range(0, n_samples, batch_size)):
                batch = X[indices[i:i+batch_size]]
                loss = self.train_step(batch, n_cd_steps, step_size, noise_scale)
                epoch_loss += loss * batch.shape[0]
                for cb in (callbacks or []):
                    getattr(cb, "on_batch_end", lambda *a, **k: None)(epoch, batch_idx_num, loss, model=self)
            avg_loss = epoch_loss / n_samples
            history.append(avg_loss)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - EBM loss: {avg_loss:.4f}")
            for cb in (callbacks or []):
                getattr(cb, "on_epoch_end", lambda *a, **k: None)(epoch, {"loss": avg_loss}, model=self)
        for cb in (callbacks or []):
            getattr(cb, "on_train_end", lambda *a, **k: None)(history)
        return history

    def sample(self, n_samples: int = 1, n_steps: int = 100, step_size: float = 0.1,
               noise_scale: float = 0.005) -> Any:
        x = np.random.randn(n_samples, self.data_dim).astype(backend.default_dtype()) * self.init_noise_scale
        for _ in range(n_steps):
            e, grad = self._energy_grad(x)
            step_noise = np.random.randn(*x.shape).astype(backend.default_dtype())
            x = x - step_size * grad + step_noise * noise_scale
        return x

    def score(self, x: Any) -> Any:
        _, grad = self._energy_grad(x)
        return grad
