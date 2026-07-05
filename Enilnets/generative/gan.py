import numpy as np
from ..base import NeuralNet

class GAN:
    """
    Generative Adversarial Network built on Enilnets NeuralNet.

    Parameters
    ----------
    latent_dim : int
        Dimension of noise vector z
    data_dim : int
        Dimension of generated data (flattened)
    generator_hidden : list of int
        Hidden sizes for generator MLP
    discriminator_hidden : list of int
        Hidden sizes for discriminator MLP
    g_activation : str
        Generator activation (default: "swish")
    d_activation : str
        Discriminator activation (default: "leakyrelu")
    loss_type : str
        "bce", "bce_logits", or "wasserstein"
    learning_rate : float
    optimizer : str
    l2_lambda : float
    """
    def __init__(self, latent_dim, data_dim, generator_hidden=[256, 512],
                 discriminator_hidden=[512, 256], g_activation="swish",
                 d_activation="leakyrelu", loss_type="bce",
                 learning_rate=0.0002, optimizer="adam", l2_lambda=0.0):
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.loss_type = loss_type
        self.g_activation = g_activation
        self.d_activation = d_activation

        # Build Generator: z -> data
        self.generator = NeuralNet(learning_rate=learning_rate, optimizer=optimizer, l2_lambda=l2_lambda)
        prev = latent_dim
        for h in generator_hidden:
            self.generator.add_dense(prev, h, activation=g_activation)
            prev = h
        # Output: tanh for [-1, 1] range (common for images)
        self.generator.add_dense(prev, data_dim, activation="tanh")

        # Build Discriminator: data -> [0, 1] (or logit)
        self.discriminator = NeuralNet(learning_rate=learning_rate, optimizer=optimizer, l2_lambda=l2_lambda)
        prev = data_dim
        for h in discriminator_hidden:
            self.discriminator.add_dense(prev, h, activation=d_activation)
            prev = h
        # Output: sigmoid for BCE, linear for Wasserstein/BCE_logits
        out_activation = "sigmoid" if loss_type == "bce" else "linear"
        self.discriminator.add_dense(prev, 1, activation=out_activation)

    def generate(self, n_samples):
        """Generate fake samples from noise."""
        z = np.random.randn(n_samples, self.latent_dim)
        return self.generator.Forward(z, training=True)

    def discriminate(self, x):
        """Discriminator output for input x."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        return self.discriminator.Forward(x, training=True)

    def _train_discriminator(self, real_data, fake_data):
        """
        Train discriminator on one batch of real and fake data.
        Returns discriminator loss.
        """
        batch_size = real_data.shape[0]
        fake_bs = fake_data.shape[0]
        total_bs = batch_size + fake_bs

        # Single Forward on concatenated data so outputs[-1] matches targets
        combined_data = np.concatenate([real_data, fake_data], axis=0)
        d_combined = self.discriminate(combined_data)
        d_real = d_combined[:batch_size]
        d_fake = d_combined[batch_size:]

        if self.loss_type == "bce":
            combined_targets = np.concatenate([
                np.ones((batch_size, 1)),
                np.zeros((fake_bs, 1))
            ], axis=0)

            self.discriminator.Backward(combined_targets)
            self.discriminator.update()

            real_loss = -np.mean(np.log(np.clip(d_real, 1e-12, 1.0)))
            fake_loss = -np.mean(np.log(1 - np.clip(d_fake, 1e-12, 1.0)))
            return float(real_loss + fake_loss)

        elif self.loss_type == "bce_logits":
            combined_targets = np.concatenate([
                np.ones((batch_size, 1)),
                np.zeros((fake_bs, 1))
            ], axis=0)
            self.discriminator.Backward(combined_targets)
            self.discriminator.update()

            real_loss = np.mean(np.maximum(d_real, 0) - d_real + np.log(1 + np.exp(-np.abs(d_real))))
            fake_loss = np.mean(np.maximum(d_fake, 0) + np.log(1 + np.exp(-np.abs(d_fake))))
            return float(real_loss + fake_loss)

        elif self.loss_type == "wasserstein":
            combined_targets = np.concatenate([
                np.ones((batch_size, 1)),
                -np.ones((fake_bs, 1))
            ], axis=0)
            self.discriminator.Backward(combined_targets)
            self.discriminator.update()
            return float(-np.mean(d_real) + np.mean(d_fake))

        return 0.0

    def _train_generator(self, fake_data):
        """
        Train generator to fool discriminator.
        Computes gradient of generator loss w.r.t. generator parameters
        by chaining through discriminator.
        """
        batch_size = fake_data.shape[0]

        # Forward fake through discriminator
        d_fake = self.discriminate(fake_data)

        if self.loss_type == "bce":
            # Loss = -log(D(fake))
            # dL/d(D) = -1/D(fake)
            dL_d_dfake = -1.0 / np.clip(d_fake, 1e-12, 1.0)
        elif self.loss_type == "bce_logits":
            # d_fake is pre-sigmoid logit
            # dL/dz = sigmoid(z) - 1 = D(fake) - 1
            # But our d_fake could be post-sigmoid... let's handle both
            if np.all((d_fake >= 0) & (d_fake <= 1)):
                # Post-sigmoid: dL/d(D) = -1/D
                dL_d_dfake = -1.0 / np.clip(d_fake, 1e-12, 1.0)
            else:
                # Pre-sigmoid: dL/dz = sigmoid(z) - 1
                from ..activations import activate
                dL_d_dfake = activate("sigmoid", d_fake) - 1.0
        elif self.loss_type == "wasserstein":
            dL_d_dfake = -np.ones_like(d_fake)
        else:
            return 0.0

        # Backprop through discriminator to get gradient w.r.t. its input
        self.discriminator.Backward(None, output_delta=dL_d_dfake / batch_size)
        # The gradient w.r.t. discriminator input is: deltas[0] @ W0^T
        # where deltas[0] is grad w.r.t. pre-activation of layer 0
        first_layer = self.discriminator.layers[0]
        d_input = np.dot(self.discriminator.deltas[0], first_layer["weights"])

        # This is the gradient w.r.t. generator output (fake_data)
        # Now backprop through generator
        self.generator.Backward(None, output_delta=d_input)
        self.generator.update()

        if self.loss_type == "bce":
            return float(-np.mean(np.log(np.clip(d_fake, 1e-12, 1.0))))
        elif self.loss_type == "bce_logits":
            return float(np.mean(np.maximum(d_fake, 0) - d_fake + np.log(1 + np.exp(-np.abs(d_fake)))))
        elif self.loss_type == "wasserstein":
            return float(-np.mean(d_fake))
        return 0.0

    def Train(self, X_train, epochs=10, batch_size=64, d_steps=1, g_steps=1, verbose=True):
        """
        Train GAN.

        Parameters
        ----------
        X_train : ndarray
            Training data, shape (N, data_dim) or (N, C, H, W)
        epochs : int
        batch_size : int
        d_steps : int
            Number of discriminator updates per epoch
        g_steps : int
            Number of generator updates per epoch
        verbose : bool
        """
        X = np.asarray(X_train, dtype=np.float64)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        n_samples = X.shape[0]

        history = {"d_loss": [], "g_loss": []}

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                real_batch = X[indices[i:i+batch_size]]
                bs = real_batch.shape[0]

                # Train Discriminator
                for _ in range(d_steps):
                    fake_batch = self.generate(bs)
                    d_loss = self._train_discriminator(real_batch, fake_batch)

                # Train Generator
                for _ in range(g_steps):
                    fake_batch = self.generate(bs)
                    g_loss = self._train_generator(fake_batch)

                epoch_d_loss += d_loss * bs
                epoch_g_loss += g_loss * bs
                n_batches += bs

            history["d_loss"].append(epoch_d_loss / n_batches)
            history["g_loss"].append(epoch_g_loss / n_batches)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - D_loss: {history['d_loss'][-1]:.4f} - G_loss: {history['g_loss'][-1]:.4f}")

        return history

    def sample(self, n_samples=16):
        """Generate samples."""
        return self.generate(n_samples)
