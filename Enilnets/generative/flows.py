import numpy as np
from ..base import NeuralNet

class RealNVP:
    def __init__(self, data_dim, n_coupling=4, hidden_dim=256,
                 activation="swish", learning_rate=0.001, optimizer="adam", l2_lambda=0.0):
        self.data_dim = data_dim
        self.n_coupling = n_coupling
        self.hidden_dim = hidden_dim

        self.couplings = []
        for i in range(n_coupling):
            s_net = NeuralNet(learning_rate=learning_rate, optimizer=optimizer, l2_lambda=l2_lambda)
            s_net.add_dense(data_dim // 2, hidden_dim, activation=activation)
            s_net.add_dense(hidden_dim, hidden_dim, activation=activation)
            s_net.add_dense(hidden_dim, data_dim - data_dim // 2, activation="tanh")

            t_net = NeuralNet(learning_rate=learning_rate, optimizer=optimizer, l2_lambda=l2_lambda)
            t_net.add_dense(data_dim // 2, hidden_dim, activation=activation)
            t_net.add_dense(hidden_dim, hidden_dim, activation=activation)
            t_net.add_dense(hidden_dim, data_dim - data_dim // 2, activation="linear")

            self.couplings.append({"s_net": s_net, "t_net": t_net, "mask": i % 2})

    def _split(self, x, mask_type):
        d = self.data_dim
        d1 = d // 2
        if mask_type == 0:
            return x[:, :d1], x[:, d1:]
        else:
            return x[:, d1:], x[:, :d1]

    def _concat(self, x1, x2, mask_type):
        if mask_type == 0:
            return np.concatenate([x1, x2], axis=1)
        else:
            return np.concatenate([x2, x1], axis=1)

    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
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

    def inverse(self, z):
        z = np.asarray(z, dtype=np.float64)
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

    def log_prob(self, x):
        z, log_det = self.forward(x)
        log_pz = -0.5 * np.sum(z ** 2 + np.log(2 * np.pi), axis=1)
        return log_pz + log_det

    def loss(self, x):
        return float(-np.mean(self.log_prob(x)))

    def _backward_through_coupling(self, x, coupling_idx, dL_dz, dL_dlogdet):
        """Backpropagate through a single coupling layer, computing gradients
        for its s_net and t_net parameters."""
        coupling = self.couplings[coupling_idx]
        mask = coupling["mask"]

        # Forward through this coupling
        x1, x2 = self._split(x, mask)
        s = coupling["s_net"].Forward(x1, training=True)
        t = coupling["t_net"].Forward(x1, training=True)
        y2 = x2 * np.exp(s) + t
        z = self._concat(x1, y2, mask)

        # Backward: dL/dz -> dL/dy2, dL/dx1
        dL_dy2 = dL_dz[:, self.data_dim//2:] if mask == 0 else dL_dz[:, :self.data_dim//2]
        dL_dx1_part = dL_dz[:, :self.data_dim//2] if mask == 0 else dL_dz[:, self.data_dim//2:]

        # dL/ds = dL/dlogdet * 1 + dL/dy2 * x2 * exp(s)
        dL_ds = dL_dlogdet.reshape(-1, 1) + dL_dy2 * x2 * np.exp(s)
        # dL/dt = dL/dy2 * 1
        dL_dt = dL_dy2
        # dL/dx2 = dL/dy2 * exp(s)
        dL_dx2 = dL_dy2 * np.exp(s)
        # dL/dx1 = dL/dx1_part + dL/ds * ds/dx1 + dL/dt * dt/dx1
        # But we need to backprop through s_net and t_net first

        # Backprop through s_net
        coupling["s_net"].Backward(None, output_delta=dL_ds / x.shape[0])
        coupling["s_net"].update()
        # Get gradient w.r.t s_net input (x1)
        s_first = coupling["s_net"].layers[0]
        dL_dx1_via_s = np.dot(coupling["s_net"].deltas[0], s_first["weights"])

        # Backprop through t_net
        coupling["t_net"].Backward(None, output_delta=dL_dt / x.shape[0])
        coupling["t_net"].update()
        # Get gradient w.r.t t_net input (x1)
        t_first = coupling["t_net"].layers[0]
        dL_dx1_via_t = np.dot(coupling["t_net"].deltas[0], t_first["weights"])

        # Total gradient w.r.t x1
        dL_dx1 = dL_dx1_part + dL_dx1_via_s + dL_dx1_via_t

        # Reconstruct dL/dx for previous layer
        dL_dx = self._concat(dL_dx1, dL_dx2, mask)
        return dL_dx

    def train_step(self, x):
        """One gradient step minimizing negative log-likelihood, backpropagating
        through all coupling layers."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        batch_size = x.shape[0]

        # Forward pass
        z, log_det = self.forward(x)
        log_pz = -0.5 * np.sum(z ** 2 + np.log(2 * np.pi), axis=1)
        log_prob = log_pz + log_det
        loss = -np.mean(log_prob)

        # Backward: dL/dz and dL/dlogdet
        # d(log_pz)/dz = -z, and loss = -mean(log_prob), so dL/dz = -(-z)/N = z/N.
        dL_dz = z / batch_size
        dL_dlogdet = -np.ones(batch_size) / batch_size  # gradient of -logdet w.r.t logdet

        # Backpropagate through couplings in reverse order
        current_x = x.copy()
        # We need to store intermediate activations for each coupling
        activations = []
        z_fwd = x.copy()
        for coupling in self.couplings:
            x1, x2 = self._split(z_fwd, coupling["mask"])
            s = coupling["s_net"].Forward(x1, training=True)
            t = coupling["t_net"].Forward(x1, training=True)
            y2 = x2 * np.exp(s) + t
            z_fwd = self._concat(x1, y2, coupling["mask"])
            activations.append((x1, x2, s, t, y2))

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

            # Gradients for s and t
            dL_ds = dL_dlogdet.reshape(-1, 1) + dL_dy2 * x2 * np.exp(s)
            dL_dt = dL_dy2

            # Backprop through s_net
            coupling["s_net"].Backward(None, output_delta=dL_ds / batch_size)
            coupling["s_net"].update()
            s_first = coupling["s_net"].layers[0]
            dL_dx1_s = np.dot(coupling["s_net"].deltas[0], s_first["weights"])

            # Backprop through t_net
            coupling["t_net"].Backward(None, output_delta=dL_dt / batch_size)
            coupling["t_net"].update()
            t_first = coupling["t_net"].layers[0]
            dL_dx1_t = np.dot(coupling["t_net"].deltas[0], t_first["weights"])

            # Reconstruct dL/dx for previous coupling
            dL_dx1 = dL_dx1_part + dL_dx1_s + dL_dx1_t
            dL_dx2 = dL_dy2 * np.exp(s)
            dL_dz = self._concat(dL_dx1, dL_dx2, mask)
            dL_dlogdet = dL_dlogdet  # already accounted for in dL_ds

        return float(loss)

    def Train(self, X_train, epochs=10, batch_size=64, verbose=True):
        """Gradient-based training loop over epochs of minibatches."""
        X = np.asarray(X_train, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_samples = X.shape[0]

        history = []
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            total = 0
            for i in range(0, n_samples, batch_size):
                batch = X[indices[i:i+batch_size]]
                loss = self.train_step(batch)
                epoch_loss += loss * batch.shape[0]
                total += batch.shape[0]
            avg_loss = epoch_loss / total
            history.append(avg_loss)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Flow NLL: {avg_loss:.4f}")
        return history

    def sample(self, n_samples=1):
        z = np.random.randn(n_samples, self.data_dim)
        return self.inverse(z)

    def interpolate(self, x1, x2, n_steps=10):
        z1, _ = self.forward(x1)
        z2, _ = self.forward(x2)
        alphas = np.linspace(0, 1, n_steps).reshape(-1, 1)
        z_interp = alphas * z2 + (1 - alphas) * z1
        return self.inverse(z_interp)
