import numpy as np
from ..base import NeuralNet
from .sampling import reparameterize
from .generative_loss import kl_divergence_gaussian

class VAE:
    def __init__(self, input_dim, latent_dim, encoder_hidden=[512, 256],
                 decoder_hidden=[256, 512], activation="swish",
                 learning_rate=0.001, optimizer="adam", l2_lambda=0.0):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation = activation

        # Build Encoder
        self.encoder = NeuralNet(learning_rate=learning_rate, optimizer=optimizer, l2_lambda=l2_lambda)
        prev = input_dim
        for h in encoder_hidden:
            self.encoder.add_dense(prev, h, activation=activation)
            prev = h
        self.encoder.add_dense(prev, latent_dim * 2, activation="linear")

        # Build Decoder
        self.decoder = NeuralNet(learning_rate=learning_rate, optimizer=optimizer, l2_lambda=l2_lambda)
        prev = latent_dim
        for h in decoder_hidden:
            self.decoder.add_dense(prev, h, activation=activation)
            prev = h
        self.decoder.add_dense(prev, input_dim, activation="sigmoid")

    def encode(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        elif x.ndim == 1:
            x = x.reshape(1, -1)
        h = self.encoder.Forward(x, training=True)
        mu = h[:, :self.latent_dim]
        logvar = h[:, self.latent_dim:]
        return mu, logvar

    def decode(self, z):
        z = np.asarray(z, dtype=np.float64)
        if z.ndim == 1:
            z = z.reshape(1, -1)
        return self.decoder.Forward(z, training=True)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

    def loss(self, x, recon=None, mu=None, logvar=None, kl_weight=1.0):
        if recon is None:
            recon, mu, logvar, _ = self.forward(x)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        recon = np.clip(recon, 1e-12, 1 - 1e-12)
        recon_loss = -np.mean(x * np.log(recon) + (1 - x) * np.log(1 - recon))
        kl = kl_divergence_gaussian(mu, logvar, reduction="mean", kl_weight=kl_weight)
        return recon_loss + kl

    def train_step(self, x, kl_weight=1.0):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        elif x.ndim == 1:
            x = x.reshape(1, -1)
        batch_size = x.shape[0]

        # Forward
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        recon = self.decode(z)

        # === Decoder backward ===
        recon = np.clip(recon, 1e-12, 1 - 1e-12)
        # Canonical sigmoid+BCE gradient w.r.t. the pre-sigmoid logit: (recon - x)/N.
        # No extra sigmoid-derivative factor here -- it already cancels analytically.
        d_recon_pre = (recon - x) / batch_size
        self.decoder.Backward(None, output_delta=d_recon_pre)
        self.decoder.update()

        # === Encoder backward ===
        d_z = np.dot(self.decoder.deltas[0], self.decoder.layers[0]["weights"])

        eps = (z - mu) / np.exp(0.5 * logvar)
        d_mu = d_z + kl_weight * mu
        d_logvar = d_z * 0.5 * np.exp(0.5 * logvar) * eps + kl_weight * 0.5 * (np.exp(logvar) - 1)

        d_h = np.concatenate([d_mu, d_logvar], axis=1) / batch_size

        self.encoder.Backward(None, output_delta=d_h)
        self.encoder.update()

        return self.loss(x, recon, mu, logvar, kl_weight=kl_weight)

    def Train(self, X_train, epochs=10, batch_size=64, verbose=True, kl_weight=1.0):
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
                loss = self.train_step(batch, kl_weight=kl_weight)
                epoch_loss += loss * batch.shape[0]
            avg_loss = epoch_loss / n_samples
            history.append(avg_loss)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - VAE loss: {avg_loss:.4f}")
        return history

    def generate(self, n_samples=1):
        z = np.random.randn(n_samples, self.latent_dim)
        return self.decode(z)

    def reconstruct(self, x):
        recon, _, _, _ = self.forward(x)
        return recon

    def interpolate(self, x1, x2, n_steps=10):
        mu1, _ = self.encode(x1)
        mu2, _ = self.encode(x2)
        alphas = np.linspace(0, 1, n_steps).reshape(-1, 1)
        z_interp = alphas * mu2 + (1 - alphas) * mu1
        return self.decode(z_interp)
