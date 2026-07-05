import numpy as np
from ..base import NeuralNet

class GAN:
    def __init__(self, latent_dim, data_dim, generator_hidden=[256, 512],
                 discriminator_hidden=[512, 256], g_activation="swish",
                 d_activation="leakyrelu", loss_type="bce",
                 learning_rate=0.0002, optimizer="adam", l2_lambda=0.0,
                 label_smoothing=0.9,
                 g_lr_factor=1.0,
                 d_lr_factor=1.0,
                 wgan_clip_value=0.01,
                 num_classes=None):
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.loss_type = loss_type
        self.g_activation = g_activation
        self.d_activation = d_activation
        self.label_smoothing = label_smoothing
        self.g_lr_factor = g_lr_factor
        self.d_lr_factor = d_lr_factor
        self.wgan_clip_value = wgan_clip_value
        self.num_classes = num_classes
        cond_dim = num_classes if num_classes is not None else 0

        g_lr = learning_rate * g_lr_factor
        d_lr = learning_rate * d_lr_factor

        # Build Generator: [z, onehot(y)] -> data
        self.generator = NeuralNet(learning_rate=g_lr, optimizer=optimizer, l2_lambda=l2_lambda)
        prev = latent_dim + cond_dim
        for h in generator_hidden:
            self.generator.add_dense(prev, h, activation=g_activation)
            prev = h
        self.generator.add_dense(prev, data_dim, activation="tanh")

        # Build Discriminator: [data, onehot(y)] -> real/fake
        self.discriminator = NeuralNet(learning_rate=d_lr, optimizer=optimizer, l2_lambda=l2_lambda)
        prev = data_dim + cond_dim
        for h in discriminator_hidden:
            self.discriminator.add_dense(prev, h, activation=d_activation)
            prev = h
        out_activation = "sigmoid" if loss_type == "bce" else "linear"
        self.discriminator.add_dense(prev, 1, activation=out_activation)

    def _onehot(self, y, n):
        """Validate `y` against `self.num_classes` (raises if one is given
        without the other) and return a one-hot (n, num_classes) array, or
        None if this GAN is unconditional."""
        if self.num_classes is None:
            if y is not None:
                raise ValueError("This GAN was built without num_classes -- don't pass y/labels.")
            return None
        if y is None:
            raise ValueError("This GAN was built with num_classes=... -- y/labels is required.")
        y = np.asarray(y).reshape(-1)
        if y.shape[0] == 1 and n > 1:
            y = np.repeat(y, n)
        if y.shape[0] != n:
            raise ValueError(f"y has {y.shape[0]} labels but batch size is {n}")
        return np.eye(self.num_classes, dtype=np.float64)[y]

    def generate(self, n_samples, y=None):
        z = np.random.randn(n_samples, self.latent_dim)
        onehot = self._onehot(y, n_samples)
        if onehot is not None:
            z = np.concatenate([z, onehot], axis=1)
        return self.generator.Forward(z, training=True)

    def discriminate(self, x, y=None):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        onehot = self._onehot(y, x.shape[0])
        if onehot is not None:
            x = np.concatenate([x, onehot], axis=1)
        return self.discriminator.Forward(x, training=True)

    def _train_discriminator(self, real_data, fake_data, real_y=None, fake_y=None):
        batch_size = real_data.shape[0]
        fake_bs = fake_data.shape[0]
        total = batch_size + fake_bs

        combined_data = np.concatenate([real_data, fake_data], axis=0)
        if self.num_classes is not None:
            real_oh = self._onehot(real_y, batch_size)
            fake_oh = self._onehot(fake_y, fake_bs)
            combined_data = np.concatenate([combined_data, np.concatenate([real_oh, fake_oh], axis=0)], axis=1)
        d_combined = self.discriminator.Forward(combined_data, training=True)
        d_real = d_combined[:batch_size]
        d_fake = d_combined[batch_size:]

        if self.loss_type == "bce":
            # Label smoothing (real labels = label_smoothing, not 1.0) keeps D
            # from getting overconfident, preserving a useful gradient for G.
            real_targets = np.full((batch_size, 1), self.label_smoothing, dtype=np.float64)
            fake_targets = np.zeros((fake_bs, 1), dtype=np.float64)
            combined_targets = np.concatenate([real_targets, fake_targets], axis=0)
            # Canonical sigmoid+BCE gradient w.r.t. the pre-activation logit:
            # (out - target)/N, without the extra sigmoid-derivative factor.
            delta = (d_combined - combined_targets) / total
            self.discriminator.Backward(None, output_delta=delta)
            self.discriminator.update()
            real_loss = -np.mean(np.log(np.clip(d_real, 1e-12, 1.0)))
            fake_loss = -np.mean(np.log(1 - np.clip(d_fake, 1e-12, 1.0)))
            return float(real_loss + fake_loss)

        elif self.loss_type == "bce_logits":
            real_targets = np.full((batch_size, 1), self.label_smoothing, dtype=np.float64)
            fake_targets = np.zeros((fake_bs, 1), dtype=np.float64)
            combined_targets = np.concatenate([real_targets, fake_targets], axis=0)
            # d_combined holds the raw logit; correct BCEWithLogits gradient is
            # sigmoid(logit) - target.
            sigmoid_d = 1.0 / (1.0 + np.exp(-d_combined))
            delta = (sigmoid_d - combined_targets) / total
            self.discriminator.Backward(None, output_delta=delta)
            self.discriminator.update()
            real_loss = np.mean(np.maximum(d_real, 0) - d_real + np.log(1 + np.exp(-np.abs(d_real))))
            fake_loss = np.mean(np.maximum(d_fake, 0) + np.log(1 + np.exp(-np.abs(d_fake))))
            return float(real_loss + fake_loss)

        elif self.loss_type == "wasserstein":
            combined_targets = np.concatenate([
                np.ones((batch_size, 1)),
                -np.ones((fake_bs, 1))
            ], axis=0)
            # loss = mean(-target * out) => dL/dout = -target/N (constant).
            delta = -combined_targets / total
            self.discriminator.Backward(None, output_delta=delta)
            self.discriminator.update()
            # Weight clipping keeps the discriminator K-Lipschitz, required
            # for the Wasserstein loss's gradient to be meaningful.
            self._clip_discriminator_weights(self.wgan_clip_value)
            return float(-np.mean(d_real) + np.mean(d_fake))

        return 0.0

    def _clip_discriminator_weights(self, clip_value):
        """Clip discriminator weights to [-clip_value, clip_value] (Wasserstein GAN)."""
        for layer in self.discriminator.layers:
            if "weights" in layer:
                layer["weights"] = np.clip(layer["weights"], -clip_value, clip_value)
            if "bias" in layer:
                layer["bias"] = np.clip(layer["bias"], -clip_value, clip_value)

    def _train_generator(self, fake_data, fake_y=None):
        batch_size = fake_data.shape[0]
        d_fake = self.discriminate(fake_data, y=fake_y)

        if self.loss_type == "bce":
            # G wants D(fake) -> 1. Canonical sigmoid+BCE gradient: (out - target)/N.
            delta = (d_fake - 1.0) / batch_size
            self.discriminator.Backward(None, output_delta=delta)
            # Chain gradient through discriminator to generator input (drop
            # the onehot(y) columns -- the label isn't a learned generator output).
            first_layer = self.discriminator.layers[0]
            d_input = np.dot(self.discriminator.deltas[0], first_layer["weights"])[:, :self.data_dim]
            self.generator.Backward(None, output_delta=d_input)
            self.generator.update()
            return float(-np.mean(np.log(np.clip(d_fake, 1e-12, 1.0))))

        elif self.loss_type == "bce_logits":
            sigmoid_d = 1.0 / (1.0 + np.exp(-d_fake))
            delta = (sigmoid_d - 1.0) / batch_size
            self.discriminator.Backward(None, output_delta=delta)
            first_layer = self.discriminator.layers[0]
            d_input = np.dot(self.discriminator.deltas[0], first_layer["weights"])[:, :self.data_dim]
            self.generator.Backward(None, output_delta=d_input)
            self.generator.update()
            return float(np.mean(np.maximum(d_fake, 0) - d_fake + np.log(1 + np.exp(-np.abs(d_fake)))))

        elif self.loss_type == "wasserstein":
            # G wants D(fake) to be high: loss = -mean(D(fake)) => dL/dD(fake) = -1/N (constant).
            delta = -np.ones((batch_size, 1), dtype=np.float64) / batch_size
            self.discriminator.Backward(None, output_delta=delta)
            first_layer = self.discriminator.layers[0]
            d_input = np.dot(self.discriminator.deltas[0], first_layer["weights"])[:, :self.data_dim]
            self.generator.Backward(None, output_delta=d_input)
            self.generator.update()
            return float(-np.mean(d_fake))

        return 0.0

    def Train(self, X_train, epochs=10, batch_size=64, d_steps=1, g_steps=1, verbose=True, y_train=None):
        X = np.asarray(X_train, dtype=np.float64)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        n_samples = X.shape[0]
        y_arr = None
        if y_train is not None:
            y_arr = np.asarray(y_train).reshape(-1)
            if y_arr.shape[0] != n_samples:
                raise ValueError(f"y_train has {y_arr.shape[0]} labels but X_train has {n_samples} samples")

        history = {"d_loss": [], "g_loss": []}

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                real_batch = X[batch_idx]
                real_y = y_arr[batch_idx] if y_arr is not None else None
                bs = real_batch.shape[0]

                for _ in range(d_steps):
                    fake_y = real_y  # match the discriminator's real batch's class distribution
                    fake_batch = self.generate(bs, y=fake_y)
                    d_loss = self._train_discriminator(real_batch, fake_batch, real_y=real_y, fake_y=fake_y)

                for _ in range(g_steps):
                    fake_y = real_y
                    fake_batch = self.generate(bs, y=fake_y)
                    g_loss = self._train_generator(fake_batch, fake_y=fake_y)

                epoch_d_loss += d_loss * bs
                epoch_g_loss += g_loss * bs
                n_batches += bs

            history["d_loss"].append(epoch_d_loss / n_batches)
            history["g_loss"].append(epoch_g_loss / n_batches)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - D_loss: {history['d_loss'][-1]:.4f} - G_loss: {history['g_loss'][-1]:.4f}")

        return history

    def sample(self, n_samples=16, y=None):
        return self.generate(n_samples, y=y)

    def mode_collapse_score(self, n_samples=1000, n_clusters=10, y=None):
        """Detect mode collapse by measuring sample diversity.
        Returns a score between 0 (all identical) and 1 (fully diverse)."""
        samples = self.generate(n_samples, y=y)
        # Simple diversity metric: average pairwise distance within each
        # `step`-sized window, vectorized instead of a nested Python loop.
        step = max(1, n_samples // 100)
        i_idx = np.arange(0, n_samples, step)
        offsets = np.arange(1, step)
        if len(i_idx) == 0 or len(offsets) == 0:
            avg_dist = 0.0
        else:
            j_idx = i_idx[:, None] + offsets[None, :]
            valid = (j_idx < n_samples).reshape(-1)
            i_valid = np.repeat(i_idx, len(offsets))[valid]
            j_valid = j_idx.reshape(-1)[valid]
            avg_dist = np.mean(np.linalg.norm(samples[i_valid] - samples[j_valid], axis=-1)) if len(i_valid) else 0.0
        max_possible = np.sqrt(self.data_dim) * 2.0  # tanh output range [-1,1]
        score = min(1.0, avg_dist / (max_possible * 0.3))  # normalize
        return float(score)
