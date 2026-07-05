import numpy as np
from ..base import NeuralNet
from .sampling import reparameterize
from .generative_loss import kl_divergence_gaussian

class VAE:
    """
    Variational Autoencoder built on Enilnets NeuralNet.

    Architecture:
      Encoder: input -> hidden -> [mu, logvar] (2*latent_dim outputs)
      Decoder: latent -> hidden -> reconstruction

    Parameters
    ----------
    input_dim : int
        Flattened input dimension (e.g., 784 for MNIST)
    latent_dim : int
        Dimensionality of the latent space
    encoder_hidden : list of int
        Hidden layer sizes for encoder, e.g., [512, 256]
    decoder_hidden : list of int
        Hidden layer sizes for decoder, e.g., [256, 512]
    activation : str
        Activation for hidden layers (default: "swish")
    learning_rate : float
    optimizer : str
    l2_lambda : float
    """
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
        self.encoder.add_dense(prev, latent_dim * 2, activation="linear")  # mu + logvar

        # Build Decoder
        self.decoder = NeuralNet(learning_rate=learning_rate, optimizer=optimizer, l2_lambda=l2_lambda)
        prev = latent_dim
        for h in decoder_hidden:
            self.decoder.add_dense(prev, h, activation=activation)
            prev = h
        self.decoder.add_dense(prev, input_dim, activation="sigmoid")

    def encode(self, x):
        """
        x: (batch, input_dim) or (batch, C, H, W) -> auto-flattened
        Returns mu, logvar: each (batch, latent_dim)
        """
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
        """
        z: (batch, latent_dim)
        Returns reconstruction: (batch, input_dim)
        """
        z = np.asarray(z, dtype=np.float64)
        if z.ndim == 1:
            z = z.reshape(1, -1)
        return self.decoder.Forward(z, training=True)

    def forward(self, x):
        """
        Full forward pass.
        Returns recon, mu, logvar, z
        """
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

    def loss(self, x, recon=None, mu=None, logvar=None):
        """
        Compute ELBO = -reconstruction_loss - KL_divergence
        Returns total loss (scalar, mean over batch)
        """
        if recon is None:
            recon, mu, logvar, _ = self.forward(x)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        # Reconstruction loss (binary cross entropy, assuming sigmoid output)
        recon = np.clip(recon, 1e-12, 1 - 1e-12)
        recon_loss = -np.mean(x * np.log(recon) + (1 - x) * np.log(1 - recon))

        # KL divergence
        kl = kl_divergence_gaussian(mu, logvar, reduction="mean")

        return recon_loss + kl

    def train_step(self, x):
        """
        Single training step. Returns loss.
        """
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
        # Reconstruction loss gradient w.r.t. recon (BCE derivative)
        recon = np.clip(recon, 1e-12, 1 - 1e-12)
        d_recon = (recon - x) / batch_size  # d(BCE)/d(recon_post)
        # Chain through sigmoid: d(BCE)/d(recon_pre) = d_recon * sigmoid * (1 - sigmoid)
        d_recon_pre = d_recon * (recon * (1 - recon))
        # Use decoder Backward with custom output_delta to backprop through all layers
        self.decoder.Backward(None, output_delta=d_recon_pre)
        self.decoder.update()

        # === Encoder backward ===
        # Gradient w.r.t. z is the input gradient of decoder (deltas[0] chained through layer 0)
        # deltas[0] is gradient w.r.t. pre-activation of layer 0
        # dL/dz = deltas[0] @ W0
        d_z = np.dot(self.decoder.deltas[0], self.decoder.layers[0]["weights"])

        # KL gradient w.r.t. mu and logvar
        # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        # d(KL)/dmu = mu
        # d(KL)/dlogvar = 0.5 * (exp(logvar) - 1)
        d_mu = d_z + mu
        d_logvar = d_z * 0.5 * np.exp(0.5 * logvar) * ((z - mu) / np.exp(0.5 * logvar)) + 0.5 * (np.exp(logvar) - 1)
        # Simplify reparam contribution: dz = mu + sigma*eps, so dz/dlogvar = 0.5*sigma*eps
        eps = (z - mu) / np.exp(0.5 * logvar)
        d_logvar = d_z * 0.5 * np.exp(0.5 * logvar) * eps + 0.5 * (np.exp(logvar) - 1)

        # Combine into encoder output delta
        d_h = np.concatenate([d_mu, d_logvar], axis=1) / batch_size

        # Use encoder Backward with custom output_delta
        self.encoder.Backward(None, output_delta=d_h)
        self.encoder.update()

        return self.loss(x, recon, mu, logvar)

    def Train(self, X_train, epochs=10, batch_size=64, verbose=True):
        """
        Full training loop.
        X_train: (N, input_dim) or (N, C, H, W)
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
                print(f"Epoch {epoch+1}/{epochs} - VAE loss: {avg_loss:.4f}")
        return history

    def generate(self, n_samples=1):
        """
        Generate new samples by sampling from prior N(0, I).
        Returns (n_samples, input_dim)
        """
        z = np.random.randn(n_samples, self.latent_dim)
        return self.decode(z)

    def reconstruct(self, x):
        """
        Reconstruct input through VAE.
        """
        recon, _, _, _ = self.forward(x)
        return recon

    def interpolate(self, x1, x2, n_steps=10):
        """
        Linear interpolation in latent space between two inputs.
        Returns array of shape (n_steps, input_dim)
        """
        mu1, _ = self.encode(x1)
        mu2, _ = self.encode(x2)
        alphas = np.linspace(0, 1, n_steps).reshape(-1, 1)
        z_interp = alphas * mu2 + (1 - alphas) * mu1
        return self.decode(z_interp)
