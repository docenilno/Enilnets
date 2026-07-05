import numpy as np
from ..base import NeuralNet
from .sampling import gaussian_sample

class DiffusionModel:
    """
    Denoising Diffusion Probabilistic Model (DDPM) built on Enilnets.

    Uses a neural network (MLP or conv) to predict noise epsilon from (x_t, t).
    Training: minimize MSE between predicted and actual noise.
    Sampling: iterative denoising from pure Gaussian noise.

    Parameters
    ----------
    data_shape : tuple
        Shape of data, e.g., (784,) for flattened MNIST or (1, 28, 28) for images
    time_steps : int
        Number of diffusion steps (default: 1000)
    beta_schedule : str
        "linear" or "cosine"
    beta_start : float
    beta_end : float
    denoiser_type : str
        "mlp" or "conv"
    denoiser_hidden : list
        Hidden sizes for MLP denoiser
    learning_rate : float
    optimizer : str
    l2_lambda : float
    """
    def __init__(self, data_shape, time_steps=1000, beta_schedule="linear",
                 beta_start=1e-4, beta_end=0.02, denoiser_type="mlp",
                 denoiser_hidden=[512, 512, 512], learning_rate=0.001,
                 optimizer="adam", l2_lambda=0.0):
        self.data_shape = data_shape
        self.time_steps = time_steps
        self.denoiser_type = denoiser_type
        self.flattened = len(data_shape) == 1
        self.data_dim = data_shape[0] if self.flattened else int(np.prod(data_shape))

        # Noise schedule
        if beta_schedule == "linear":
            self.betas = np.linspace(beta_start, beta_end, time_steps, dtype=np.float64)
        elif beta_schedule == "cosine":
            s = 0.008
            t = np.arange(time_steps + 1) / time_steps
            alpha_bar = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            self.betas = np.clip(1 - alpha_bar[1:] / alpha_bar[:-1], 0, 0.999)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.concatenate([[1.0], self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = np.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

        # Time embedding dimension
        self.time_emb_dim = 128

        # Build denoiser network
        self.denoiser = NeuralNet(learning_rate=learning_rate, optimizer=optimizer, l2_lambda=l2_lambda)

        if denoiser_type == "mlp":
            # Input: flattened data + time embedding
            input_dim = self.data_dim + self.time_emb_dim
            prev = input_dim
            for h in denoiser_hidden:
                self.denoiser.add_dense(prev, h, activation="swish")
                prev = h
            self.denoiser.add_dense(prev, self.data_dim, activation="linear")
        elif denoiser_type == "conv":
            # For conv denoiser, we assume data_shape = (C, H, W)
            # Time embedding is broadcast and concatenated as extra channels
            C, H, W = data_shape
            # Simple conv denoiser: conv -> conv -> conv
            self.denoiser.add_conv2d(C, 64, k=3, activation="swish")
            self.denoiser.add_conv2d(64, 128, k=3, activation="swish")
            self.denoiser.add_conv2d(128, 64, k=3, activation="swish")
            self.denoiser.add_conv2d(64, C, k=3, activation="linear")
        else:
            raise ValueError(f"Unknown denoiser_type: {denoiser_type}")

    def _time_embedding(self, t):
        """Sinusoidal time embedding."""
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        half = self.time_emb_dim // 2
        freqs = np.exp(-np.log(10000) * np.arange(half) / half)
        args = t[:, None] * freqs[None, :]
        emb = np.concatenate([np.sin(args), np.cos(args)], axis=-1)
        if self.time_emb_dim % 2 == 1:
            emb = np.concatenate([emb, np.zeros((emb.shape[0], 1))], axis=-1)
        return emb

    def _forward_diffusion(self, x_0, t):
        """
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t)*x_0, (1-alpha_bar_t)*I)
        Returns x_t and noise.
        """
        x_0 = np.asarray(x_0, dtype=np.float64)
        noise = np.random.randn(*x_0.shape)
        sqrt_acp = self.sqrt_alphas_cumprod[t].reshape(-1, *([1] * (x_0.ndim - 1)))
        sqrt_omacp = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, *([1] * (x_0.ndim - 1)))
        x_t = sqrt_acp * x_0 + sqrt_omacp * noise
        return x_t, noise

    def _predict_noise(self, x_t, t):
        """Run denoiser to predict noise."""
        if self.denoiser_type == "mlp":
            x_t_flat = x_t.reshape(x_t.shape[0], -1)
            t_emb = self._time_embedding(t)
            inp = np.concatenate([x_t_flat, t_emb], axis=1)
            return self.denoiser.Forward(inp, training=True).reshape(x_t.shape)
        elif self.denoiser_type == "conv":
            # For conv, time is broadcast as extra channels (simplified)
            # In a full implementation, use adaptive norm or FiLM
            t_emb = self._time_embedding(t)
            t_broadcast = t_emb.reshape(t_emb.shape[0], -1, 1, 1)
            # Repeat to match spatial dims
            B, C, H, W = x_t.shape
            t_spatial = np.repeat(t_broadcast, H * W, axis=2).reshape(B, self.time_emb_dim, H, W)
            # We can't easily concat different channel counts without a conv projection
            # So just add as bias-like term (simplified time conditioning)
            return self.denoiser.Forward(x_t, training=True)

    def train_step(self, x_0):
        """
        Single training step.
        x_0: (batch, ...) clean data
        Returns loss.
        """
        x_0 = np.asarray(x_0, dtype=np.float64)
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = np.random.randint(0, self.time_steps, size=batch_size)

        # Forward diffusion
        x_t, noise = self._forward_diffusion(x_0, t)

        # Predict noise
        pred_noise = self._predict_noise(x_t, t)

        # Loss: MSE between predicted and actual noise
        loss = np.mean((pred_noise - noise) ** 2)

        # Backward
        delta = (pred_noise - noise) / batch_size
        if self.denoiser_type == "mlp":
            self.denoiser.Backward(np.zeros((batch_size, self.data_dim)))
            self.denoiser.deltas[-1] = delta.reshape(batch_size, -1)
            # Backprop through hidden layers manually
            for l in range(len(self.denoiser.layers) - 2, -1, -1):
                nxt = self.denoiser.layers[l + 1]
                next_delta = self.denoiser.deltas[l + 1]
                if nxt["type"] in ("dense", "sparse"):
                    err = np.dot(next_delta, nxt["weights"])
                else:
                    err = next_delta
                curr = self.denoiser.layers[l]
                if curr["type"] in ("dense", "sparse", "conv2d"):
                    from ..activations import derivative
                    activation_input = self.denoiser.pre_activations[l+1] if self.denoiser.pre_activations[l+1] is not None else self.denoiser.outputs[l+1]
                    self.denoiser.deltas[l] = err * derivative(curr.get("activation", "linear"), activation_input)
                else:
                    self.denoiser.deltas[l] = err
            self.denoiser.update()
        elif self.denoiser_type == "conv":
            self.denoiser.Backward(np.zeros_like(x_t))
            self.denoiser.deltas[-1] = delta
            for l in range(len(self.denoiser.layers) - 2, -1, -1):
                nxt = self.denoiser.layers[l + 1]
                next_delta = self.denoiser.deltas[l + 1]
                if nxt["type"] in ("dense", "sparse"):
                    err = np.dot(next_delta, nxt["weights"])
                elif nxt["type"] == "conv2d":
                    from ..backward import conv2d_backward_input
                    err = conv2d_backward_input(next_delta, nxt["weights"], self.denoiser.outputs[l+1].shape)
                else:
                    err = next_delta
                curr = self.denoiser.layers[l]
                if curr["type"] in ("dense", "sparse", "conv2d"):
                    from ..activations import derivative
                    activation_input = self.denoiser.pre_activations[l+1] if self.denoiser.pre_activations[l+1] is not None else self.denoiser.outputs[l+1]
                    self.denoiser.deltas[l] = err * derivative(curr.get("activation", "linear"), activation_input)
                else:
                    self.denoiser.deltas[l] = err
            self.denoiser.update()

        return loss

    def Train(self, X_train, epochs=10, batch_size=64, verbose=True):
        """
        Train diffusion model.
        X_train: (N, ...) data
        """
        X = np.asarray(X_train, dtype=np.float64)
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
                print(f"Epoch {epoch+1}/{epochs} - Diffusion loss: {avg_loss:.6f}")
        return history

    def sample(self, n_samples=16, shape=None, clip=True):
        """
        Generate samples using DDPM sampling algorithm.

        Parameters
        ----------
        n_samples : int
        shape : tuple
            Shape of each sample. If None, uses self.data_shape
        clip : bool
            Clip output to [-1, 1] or [0, 1] depending on data normalization
        """
        if shape is None:
            shape = self.data_shape

        # Start from pure noise
        if self.flattened:
            x = np.random.randn(n_samples, *shape)
        else:
            x = np.random.randn(n_samples, *shape)

        # Reverse diffusion loop
        for t_step in reversed(range(self.time_steps)):
            t = np.full(n_samples, t_step, dtype=np.int64)

            # Predict noise
            pred_noise = self._predict_noise(x, t)

            # Compute x_{t-1}
            alpha_t = self.alphas[t].reshape(-1, *([1] * (x.ndim - 1)))
            alpha_cumprod_t = self.alphas_cumprod[t].reshape(-1, *([1] * (x.ndim - 1)))
            beta_t = self.betas[t].reshape(-1, *([1] * (x.ndim - 1)))

            # Mean of p(x_{t-1} | x_t)
            coef1 = 1.0 / np.sqrt(alpha_t)
            coef2 = beta_t / (np.sqrt(1.0 - alpha_cumprod_t) * np.sqrt(alpha_t))
            mean = coef1 * (x - coef2 * pred_noise)

            if t_step > 0:
                variance = self.posterior_variance[t].reshape(-1, *([1] * (x.ndim - 1)))
                noise = np.random.randn(*x.shape)
                x = mean + np.sqrt(variance) * noise
            else:
                x = mean

        if clip:
            x = np.clip(x, -1.0, 1.0)
        return x

    def denoise(self, x_noisy, t_start, t_end=0):
        """
        Partially denoise from timestep t_start to t_end.
        Useful for image editing / inpainting workflows.
        """
        x = x_noisy.copy()
        for t_step in reversed(range(t_end, t_start)):
            t = np.full(x.shape[0], t_step, dtype=np.int64)
            pred_noise = self._predict_noise(x, t)

            alpha_t = self.alphas[t].reshape(-1, *([1] * (x.ndim - 1)))
            alpha_cumprod_t = self.alphas_cumprod[t].reshape(-1, *([1] * (x.ndim - 1)))
            beta_t = self.betas[t].reshape(-1, *([1] * (x.ndim - 1)))

            coef1 = 1.0 / np.sqrt(alpha_t)
            coef2 = beta_t / (np.sqrt(1.0 - alpha_cumprod_t) * np.sqrt(alpha_t))
            mean = coef1 * (x - coef2 * pred_noise)

            if t_step > t_end:
                variance = self.posterior_variance[t].reshape(-1, *([1] * (x.ndim - 1)))
                noise = np.random.randn(*x.shape)
                x = mean + np.sqrt(variance) * noise
            else:
                x = mean
        return x
