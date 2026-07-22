from typing import Any, List, Optional, Tuple

from ..core.backend import np
from ..core import backend
from ..core.base import NeuralNet

class RealNVP:
    def __init__(self, data_dim: int, n_coupling: int = 4, hidden_dim: int = 256,
                 activation: str = "swish", learning_rate: float = 0.001, optimizer: str = "adam",
                 l2_lambda: float = 0.0) -> None:
        self.data_dim = data_dim
        self.n_coupling = n_coupling
        self.hidden_dim = hidden_dim

        # _split(x, mask=0) yields (x1, x2) = (x[:,:d1], x[:,d1:]); mask=1
        # swaps their roles, (x[:,d1:], x[:,:d1]) -- x1 is always what's fed
        # into s_net/t_net (the conditioning half) and x2 is always what
        # gets transformed. When data_dim is odd, d1 != data_dim - d1, so
        # the conditioning/transformed widths actually swap too between the
        # two mask values -- each coupling's s_net/t_net must be sized for
        # *its own* mask's (conditioning, transformed) widths, not a single
        # fixed (data_dim//2, data_dim - data_dim//2) pair shared by all
        # couplings regardless of mask.
        d1 = data_dim // 2
        self.couplings = []
        for i in range(n_coupling):
            mask = i % 2
            cond_dim, trans_dim = (d1, data_dim - d1) if mask == 0 else (data_dim - d1, d1)

            s_net = NeuralNet(learning_rate=learning_rate, optimizer=optimizer, l2_lambda=l2_lambda)
            s_net.add_dense(cond_dim, hidden_dim, activation=activation)
            s_net.add_dense(hidden_dim, hidden_dim, activation=activation)
            s_net.add_dense(hidden_dim, trans_dim, activation="tanh")

            t_net = NeuralNet(learning_rate=learning_rate, optimizer=optimizer, l2_lambda=l2_lambda)
            t_net.add_dense(cond_dim, hidden_dim, activation=activation)
            t_net.add_dense(hidden_dim, hidden_dim, activation=activation)
            t_net.add_dense(hidden_dim, trans_dim, activation="linear")

            self.couplings.append({"s_net": s_net, "t_net": t_net, "mask": mask})

    def _split(self, x: Any, mask_type: int) -> Tuple[Any, Any]:
        d = self.data_dim
        d1 = d // 2
        if mask_type == 0:
            return x[:, :d1], x[:, d1:]
        else:
            return x[:, d1:], x[:, :d1]

    def _concat(self, x1: Any, x2: Any, mask_type: int) -> Any:
        if mask_type == 0:
            return np.concatenate([x1, x2], axis=1)
        else:
            return np.concatenate([x2, x1], axis=1)

    def forward(self, x: Any) -> Tuple[Any, Any]:
        x = np.asarray(x, dtype=backend.default_dtype())
        if x.ndim == 1:
            x = x.reshape(1, -1)

        log_det = 0.0
        z = x

        for coupling in self.couplings:
            x1, x2 = self._split(z, coupling["mask"])
            s = coupling["s_net"].Forward(x1, training=True)
            t = coupling["t_net"].Forward(x1, training=True)

            y2 = x2 * np.exp(s) + t
            z = self._concat(x1, y2, coupling["mask"])
            log_det += np.sum(s, axis=1)

        return z, log_det

    def inverse(self, z: Any) -> Any:
        z = np.asarray(z, dtype=backend.default_dtype())
        if z.ndim == 1:
            z = z.reshape(1, -1)

        x = z
        for coupling in reversed(self.couplings):
            x1, y2 = self._split(x, coupling["mask"])
            s = coupling["s_net"].Forward(x1, training=False)
            t = coupling["t_net"].Forward(x1, training=False)

            x2 = (y2 - t) * np.exp(-s)
            x = self._concat(x1, x2, coupling["mask"])

        return x

    def log_prob(self, x: Any) -> Any:
        z, log_det = self.forward(x)
        log_pz = -0.5 * np.sum(z ** 2 + np.log(2 * np.pi), axis=1)
        return log_pz + log_det

    def loss(self, x: Any) -> float:
        return float(-np.mean(self.log_prob(x)))

    def train_step(self, x: Any) -> float:
        """One gradient step minimizing negative log-likelihood, backpropagating
        through all coupling layers."""
        x = np.asarray(x, dtype=backend.default_dtype())
        if x.ndim == 1:
            x = x.reshape(1, -1)
        batch_size = x.shape[0]

        # Single forward pass through the couplings, capturing each one's
        # intermediate (x1, x2, s, t, y2) as we go -- these are exactly what
        # the backward pass below needs, so there's no separate call to
        # self.forward(x) here (that would silently redo this same
        # per-coupling computation from scratch, doubling the forward-pass
        # cost of every training step for no reason).
        activations = []
        z_fwd = x
        log_det = np.zeros(batch_size, dtype=backend.default_dtype())
        for coupling in self.couplings:
            x1, x2 = self._split(z_fwd, coupling["mask"])
            s = coupling["s_net"].Forward(x1, training=True)
            t = coupling["t_net"].Forward(x1, training=True)
            y2 = x2 * np.exp(s) + t
            z_fwd = self._concat(x1, y2, coupling["mask"])
            log_det = log_det + np.sum(s, axis=1)
            activations.append((x1, x2, s, t, y2))
        z = z_fwd

        log_pz = -0.5 * np.sum(z ** 2 + np.log(2 * np.pi), axis=1)
        log_prob = log_pz + log_det
        loss = -np.mean(log_prob)

        # Backward: dL/dz and dL/dlogdet
        # d(log_pz)/dz = -z, and loss = -mean(log_prob), so dL/dz = -(-z)/N = z/N.
        dL_dz = z / batch_size
        dL_dlogdet = -np.ones(batch_size) / batch_size  # gradient of -logdet w.r.t logdet

        # Reverse backprop
        for ci in reversed(range(len(self.couplings))):
            coupling = self.couplings[ci]
            mask = coupling["mask"]
            x1, x2, s, t, y2 = activations[ci]

            # Split dL_dz
            if mask == 0:
                dL_dx1_part = dL_dz[:, :x1.shape[1]]
                dL_dy2 = dL_dz[:, x1.shape[1]:]
            else:
                dL_dy2 = dL_dz[:, :x2.shape[1]]
                dL_dx1_part = dL_dz[:, x2.shape[1]:]

            # Gradients for s and t (w.r.t. s_net/t_net's POST-activation
            # output values -- s = tanh(z_s), t = z_t since t_net's last
            # layer is linear).
            dL_ds = dL_dlogdet.reshape(-1, 1) + dL_dy2 * x2 * np.exp(s)
            dL_dt = dL_dy2

            # Backward(output_delta=...) sets self.deltas[-1] directly,
            # i.e. it expects the gradient w.r.t. the PRE-activation output
            # (dz), not the post-activation value -- see _loss_output_delta's
            # convention in backward.py. t_net's last activation is linear
            # (dt/dz == 1), so dL_dt needs no conversion, but s_net's last
            # activation is tanh: passing dL_ds directly (a POST-tanh
            # gradient) as if it were PRE-activation silently dropped the
            # tanh derivative factor (1 - s**2) entirely.
            #
            # dL_ds/dL_dt are already correctly batch-scaled (dL_dz and
            # dL_dlogdet above both already carry the /batch_size factor,
            # applied once per sample-row, matching loss's mean-over-batch
            # convention), so no further division by batch_size is needed
            # here.
            dL_dz_s = dL_ds * (1 - s ** 2)
            coupling["s_net"].Backward(None, output_delta=dL_dz_s)
            coupling["s_net"].update()
            s_first = coupling["s_net"].layers[0]
            dL_dx1_s = np.dot(coupling["s_net"].deltas[0], s_first["weights"])

            # Backprop through t_net
            coupling["t_net"].Backward(None, output_delta=dL_dt)
            coupling["t_net"].update()
            t_first = coupling["t_net"].layers[0]
            dL_dx1_t = np.dot(coupling["t_net"].deltas[0], t_first["weights"])

            # Reconstruct dL/dx for previous coupling
            dL_dx1 = dL_dx1_part + dL_dx1_s + dL_dx1_t
            dL_dx2 = dL_dy2 * np.exp(s)
            dL_dz = self._concat(dL_dx1, dL_dx2, mask)
            dL_dlogdet = dL_dlogdet  # already accounted for in dL_ds

        return float(loss)

    def Train(self, X_train: Any, epochs: int = 10, batch_size: int = 64, verbose: bool = True,
              callbacks: Optional[List[Any]] = None) -> List[float]:
        """Gradient-based training loop over epochs of minibatches.

        callbacks: optional list of duck-typed callback objects (same
        convention as TextGenerator.Train/NeuralNet.Train). Supported hooks:
          on_batch_end(epoch, batch_idx, loss, model=self) -- after every
            minibatch's train_step.
          on_epoch_end(epoch, logs, model=self) -- once per epoch, with
            logs={"loss": avg_loss}.
          on_train_end(history) -- once after the epoch loop.
        Missing methods are skipped (no error)."""
        X = np.asarray(X_train, dtype=backend.default_dtype())
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_samples = X.shape[0]

        history = []
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            total = 0
            for batch_idx_num, i in enumerate(range(0, n_samples, batch_size)):
                batch = X[indices[i:i+batch_size]]
                loss = self.train_step(batch)
                epoch_loss += loss * batch.shape[0]
                total += batch.shape[0]
                for cb in (callbacks or []):
                    getattr(cb, "on_batch_end", lambda *a, **k: None)(epoch, batch_idx_num, loss, model=self)
            avg_loss = epoch_loss / total
            history.append(avg_loss)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Flow NLL: {avg_loss:.4f}")
            for cb in (callbacks or []):
                getattr(cb, "on_epoch_end", lambda *a, **k: None)(epoch, {"loss": avg_loss}, model=self)
        for cb in (callbacks or []):
            getattr(cb, "on_train_end", lambda *a, **k: None)(history)
        return history

    def sample(self, n_samples: int = 1) -> Any:
        z = np.random.randn(n_samples, self.data_dim).astype(backend.default_dtype())
        return self.inverse(z)

    def interpolate(self, x1: Any, x2: Any, n_steps: int = 10) -> Any:
        z1, _ = self.forward(x1)
        z2, _ = self.forward(x2)
        alphas = np.linspace(0, 1, n_steps).reshape(-1, 1)
        z_interp = alphas * z2 + (1 - alphas) * z1
        return self.inverse(z_interp)
