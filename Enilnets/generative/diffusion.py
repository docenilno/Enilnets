import numpy as np
from ..base import NeuralNet
from .sampling import gaussian_sample

class DiffusionModel:
    def __init__(self, data_shape, time_steps=1000, beta_schedule="linear",
                 beta_start=1e-4, beta_end=0.02, denoiser_type="mlp",
                 denoiser_hidden=[512, 512, 512], learning_rate=0.001,
                 optimizer="adam", l2_lambda=0.0,
                 use_ema=True, ema_decay=0.999,
                 cosine_schedule_s=0.008, beta_clip=(0, 0.999),
                 time_emb_dim=128, sample_clip_range=(-1.0, 1.0)):
        self.data_shape = data_shape
        self.time_steps = time_steps
        self.denoiser_type = denoiser_type
        self.flattened = len(data_shape) == 1
        self.data_dim = data_shape[0] if self.flattened else int(np.prod(data_shape))
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.sample_clip_range = sample_clip_range

        # Noise schedule
        if beta_schedule == "linear":
            self.betas = np.linspace(beta_start, beta_end, time_steps, dtype=np.float64)
        elif beta_schedule == "cosine":
            s = cosine_schedule_s
            t = np.arange(time_steps + 1) / time_steps
            alpha_bar = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            self.betas = np.clip(1 - alpha_bar[1:] / alpha_bar[:-1], beta_clip[0], beta_clip[1])
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.concatenate([[1.0], self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = np.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

        self.time_emb_dim = time_emb_dim

        # Build denoiser network
        self.denoiser = NeuralNet(learning_rate=learning_rate, optimizer=optimizer, l2_lambda=l2_lambda)

        if denoiser_type == "mlp":
            input_dim = self.data_dim + self.time_emb_dim
            prev = input_dim
            for h in denoiser_hidden:
                self.denoiser.add_dense(prev, h, activation="swish")
                prev = h
            self.denoiser.add_dense(prev, self.data_dim, activation="linear")
        elif denoiser_type == "conv":
            # Time embedding is broadcast spatially and concatenated as extra
            # input channels (see _predict_noise), hence the +time_emb_dim.
            C, H, W = data_shape
            self.denoiser.add_conv2d(C + self.time_emb_dim, 64, k=3, activation="swish")
            self.denoiser.add_conv2d(64, 128, k=3, activation="swish")
            self.denoiser.add_conv2d(128, 64, k=3, activation="swish")
            self.denoiser.add_conv2d(64, C, k=3, activation="linear")
        else:
            raise ValueError(f"Unknown denoiser_type: {denoiser_type}")

        if self.use_ema:
            self.ema_weights = None
            self._update_ema(init=True)

    def _update_ema(self, init=False):
        """Update the exponential moving average of denoiser weights."""
        if not self.use_ema:
            return
        current = self.denoiser.get_weights()
        if self.ema_weights is None or init:
            self.ema_weights = current
        else:
            for i in range(len(current)):
                for k in current[i]:
                    self.ema_weights[i][k] = self.ema_decay * self.ema_weights[i][k] + (1 - self.ema_decay) * current[i][k]

    def _use_ema_weights(self, use=True):
        """Temporarily swap to EMA weights for sampling."""
        if not self.use_ema or self.ema_weights is None:
            return
        if use:
            self._original_weights = self.denoiser.get_weights()
            self.denoiser.set_weights(self.ema_weights)
        else:
            self.denoiser.set_weights(self._original_weights)
            del self._original_weights

    def _time_embedding(self, t):
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        half = self.time_emb_dim // 2
        freqs = np.exp(-np.log(10000) * np.arange(half) / half)
        args = t[:, None] * freqs[None, :]
        emb = np.concatenate([np.sin(args), np.cos(args)], axis=-1)
        if self.time_emb_dim % 2 == 1:
            emb = np.concatenate([emb, np.zeros((emb.shape[0], 1))], axis=-1)
        return emb

    def _forward_diffusion(self, x_0, t):
        x_0 = np.asarray(x_0, dtype=np.float64)
        noise = np.random.randn(*x_0.shape)
        sqrt_acp = self.sqrt_alphas_cumprod[t].reshape(-1, *([1] * (x_0.ndim - 1)))
        sqrt_omacp = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, *([1] * (x_0.ndim - 1)))
        x_t = sqrt_acp * x_0 + sqrt_omacp * noise
        return x_t, noise

    def _predict_noise(self, x_t, t, use_ema=True):
        """Predict the noise added to x_t at timestep t. use_ema=True during
        sampling/denoising; False during training (uses live weights)."""
        if use_ema and self.use_ema:
            self._use_ema_weights(True)
        try:
            if self.denoiser_type == "mlp":
                x_t_flat = x_t.reshape(x_t.shape[0], -1)
                t_emb = self._time_embedding(t)
                inp = np.concatenate([x_t_flat, t_emb], axis=1)
                result = self.denoiser.Forward(inp, training=True).reshape(x_t.shape)
            elif self.denoiser_type == "conv":
                t_emb = self._time_embedding(t)  # (B, time_emb_dim)
                B, C, H, W = x_t.shape
                # Broadcast time embedding to spatial dimensions
                t_spatial = t_emb.reshape(B, self.time_emb_dim, 1, 1)
                t_spatial = np.repeat(np.repeat(t_spatial, H, axis=2), W, axis=3)
                # Concatenate time channels with image channels
                inp = np.concatenate([x_t, t_spatial], axis=1)  # (B, C+time_emb_dim, H, W)
                result = self.denoiser.Forward(inp, training=True)
        finally:
            if use_ema and self.use_ema:
                self._use_ema_weights(False)
        return result

    def train_step(self, x_0):
        x_0 = np.asarray(x_0, dtype=np.float64)
        batch_size = x_0.shape[0]

        t = np.random.randint(0, self.time_steps, size=batch_size)
        x_t, noise = self._forward_diffusion(x_0, t)
        pred_noise = self._predict_noise(x_t, t, use_ema=False)  # train on current weights

        loss = np.mean((pred_noise - noise) ** 2)

        delta = 2 * (pred_noise - noise) / batch_size
        if self.denoiser_type == "mlp":
            self.denoiser.Backward(np.zeros((batch_size, self.data_dim)))
            self.denoiser.deltas[-1] = delta.reshape(batch_size, -1)
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

        self._update_ema()

        return loss

    def Train(self, X_train, epochs=10, batch_size=64, verbose=True):
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
        if shape is None:
            shape = self.data_shape

        if self.flattened:
            x = np.random.randn(n_samples, *shape)
        else:
            x = np.random.randn(n_samples, *shape)

        for t_step in reversed(range(self.time_steps)):
            t = np.full(n_samples, t_step, dtype=np.int64)

            pred_noise = self._predict_noise(x, t, use_ema=True)

            alpha_t = self.alphas[t].reshape(-1, *([1] * (x.ndim - 1)))
            alpha_cumprod_t = self.alphas_cumprod[t].reshape(-1, *([1] * (x.ndim - 1)))
            beta_t = self.betas[t].reshape(-1, *([1] * (x.ndim - 1)))

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
            x = np.clip(x, self.sample_clip_range[0], self.sample_clip_range[1])
        return x

    def denoise(self, x_noisy, t_start, t_end=0):
        x = x_noisy.copy()
        for t_step in reversed(range(t_end, t_start)):
            t = np.full(x.shape[0], t_step, dtype=np.int64)
            pred_noise = self._predict_noise(x, t, use_ema=True)

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
