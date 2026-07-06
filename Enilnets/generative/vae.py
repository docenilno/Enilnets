from ..backend import np
from .. import backend
from ..base import NeuralNet
from .sampling import reparameterize
from .generative_loss import kl_divergence_gaussian

class VAE:
    def __init__(self, input_dim, latent_dim, encoder_hidden=[512, 256],
                 decoder_hidden=[256, 512], activation="swish",
                 learning_rate=0.001, optimizer="adam", l2_lambda=0.0,
                 num_classes=None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation = activation
        self.num_classes = num_classes
        cond_dim = num_classes if num_classes is not None else 0

        # Build Encoder
        self.encoder = NeuralNet(learning_rate=learning_rate, optimizer=optimizer, l2_lambda=l2_lambda)
        prev = input_dim + cond_dim
        for h in encoder_hidden:
            self.encoder.add_dense(prev, h, activation=activation)
            prev = h
        self.encoder.add_dense(prev, latent_dim * 2, activation="linear")

        # Build Decoder
        self.decoder = NeuralNet(learning_rate=learning_rate, optimizer=optimizer, l2_lambda=l2_lambda)
        prev = latent_dim + cond_dim
        for h in decoder_hidden:
            self.decoder.add_dense(prev, h, activation=activation)
            prev = h
        self.decoder.add_dense(prev, input_dim, activation="sigmoid")

    def _onehot(self, y, n):
        """Validate `y` against `self.num_classes` (raises if one is given
        without the other) and return a one-hot (n, num_classes) array, or
        None if this VAE is unconditional."""
        if self.num_classes is None:
            if y is not None:
                raise ValueError("This VAE was built without num_classes -- don't pass y/labels.")
            return None
        if y is None:
            raise ValueError("This VAE was built with num_classes=... -- y/labels is required.")
        y = np.asarray(y).reshape(-1)
        if y.shape[0] == 1 and n > 1:
            y = np.repeat(y, n)
        if y.shape[0] != n:
            raise ValueError(f"y has {y.shape[0]} labels but batch size is {n}")
        return np.eye(self.num_classes, dtype=backend.default_dtype())[y]

    def encode(self, x, y=None):
        x = np.asarray(x, dtype=backend.default_dtype())
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        elif x.ndim == 1:
            x = x.reshape(1, -1)
        onehot = self._onehot(y, x.shape[0])
        if onehot is not None:
            x = np.concatenate([x, onehot], axis=1)
        h = self.encoder.Forward(x, training=True)
        mu = h[:, :self.latent_dim]
        logvar = h[:, self.latent_dim:]
        return mu, logvar

    def decode(self, z, y=None):
        z = np.asarray(z, dtype=backend.default_dtype())
        if z.ndim == 1:
            z = z.reshape(1, -1)
        onehot = self._onehot(y, z.shape[0])
        if onehot is not None:
            z = np.concatenate([z, onehot], axis=1)
        return self.decoder.Forward(z, training=True)

    def forward(self, x, y=None):
        mu, logvar = self.encode(x, y=y)
        z = reparameterize(mu, logvar)
        recon = self.decode(z, y=y)
        return recon, mu, logvar, z

    def loss(self, x, recon=None, mu=None, logvar=None, kl_weight=1.0, y=None):
        if recon is None:
            recon, mu, logvar, _ = self.forward(x, y=y)
        x = np.asarray(x, dtype=backend.default_dtype())
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        recon = np.clip(recon, 1e-12, 1 - 1e-12)
        recon_loss = -np.mean(x * np.log(recon) + (1 - x) * np.log(1 - recon))
        kl = kl_divergence_gaussian(mu, logvar, reduction="mean", kl_weight=kl_weight)
        return float(recon_loss + kl)

    def train_step(self, x, kl_weight=1.0, y=None):
        x = np.asarray(x, dtype=backend.default_dtype())
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        elif x.ndim == 1:
            x = x.reshape(1, -1)
        batch_size = x.shape[0]

        # Forward
        mu, logvar = self.encode(x, y=y)
        z = reparameterize(mu, logvar)
        recon = self.decode(z, y=y)

        # === Decoder backward ===
        recon = np.clip(recon, 1e-12, 1 - 1e-12)
        # Canonical sigmoid+BCE gradient w.r.t. the pre-sigmoid logit: (recon - x)/N.
        # No extra sigmoid-derivative factor here -- it already cancels analytically.
        d_recon_pre = (recon - x) / batch_size
        self.decoder.Backward(None, output_delta=d_recon_pre)
        self.decoder.update()

        # === Encoder backward ===
        # decoder's input is [z, onehot(y)] concatenated when class-conditional --
        # only the z slice needs a gradient (the one-hot label isn't learned).
        d_z_full = np.dot(self.decoder.deltas[0], self.decoder.layers[0]["weights"])
        d_z = d_z_full[:, :self.latent_dim]

        eps = (z - mu) / np.exp(0.5 * logvar)
        d_mu = d_z + kl_weight * mu
        d_logvar = d_z * 0.5 * np.exp(0.5 * logvar) * eps + kl_weight * 0.5 * (np.exp(logvar) - 1)

        d_h = np.concatenate([d_mu, d_logvar], axis=1) / batch_size

        self.encoder.Backward(None, output_delta=d_h)
        self.encoder.update()

        return self.loss(x, recon, mu, logvar, kl_weight=kl_weight, y=y)

    def Train(self, X_train, epochs=10, batch_size=64, verbose=True, kl_weight=1.0, y_train=None):
        X = np.asarray(X_train, dtype=backend.default_dtype())
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        n_samples = X.shape[0]
        y_arr = None
        if y_train is not None:
            y_arr = np.asarray(y_train).reshape(-1)
            if y_arr.shape[0] != n_samples:
                raise ValueError(f"y_train has {y_arr.shape[0]} labels but X_train has {n_samples} samples")
        history = []
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                batch = X[batch_idx]
                batch_y = y_arr[batch_idx] if y_arr is not None else None
                loss = self.train_step(batch, kl_weight=kl_weight, y=batch_y)
                epoch_loss += loss * batch.shape[0]
            avg_loss = epoch_loss / n_samples
            history.append(avg_loss)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - VAE loss: {avg_loss:.4f}")
        return history

    def generate(self, n_samples=1, y=None):
        z = np.random.randn(n_samples, self.latent_dim)
        return self.decode(z, y=y)

    def reconstruct(self, x, y=None):
        recon, _, _, _ = self.forward(x, y=y)
        return recon

    def interpolate(self, x1, x2, n_steps=10, y=None):
        mu1, _ = self.encode(x1, y=y)
        mu2, _ = self.encode(x2, y=y)
        alphas = np.linspace(0, 1, n_steps).reshape(-1, 1)
        z_interp = alphas * mu2 + (1 - alphas) * mu1
        interp_y = None
        if y is not None:
            interp_y = np.repeat(np.asarray(y).reshape(-1), n_steps)[:n_steps]
        return self.decode(z_interp, y=interp_y)
