import math
from ..backend import np
from .. import backend
from ..base import NeuralNet
from .sampling import gaussian_sample
from ._shared import _manual_sequential_backward

class DiffusionModel:
    def __init__(self, data_shape, time_steps=1000, beta_schedule="linear",
                 beta_start=1e-4, beta_end=0.02, denoiser_type="mlp",
                 denoiser_hidden=[512, 512, 512], learning_rate=0.001,
                 optimizer="adam", l2_lambda=0.0,
                 use_ema=True, ema_decay=0.999,
                 cosine_schedule_s=0.008, beta_clip=(0, 0.999),
                 time_emb_dim=128, sample_clip_range=(-1.0, 1.0),
                 num_classes=None):
        self.data_shape = data_shape
        self.time_steps = time_steps
        self.denoiser_type = denoiser_type
        self.flattened = len(data_shape) == 1
        self.data_dim = data_shape[0] if self.flattened else math.prod(data_shape)
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.sample_clip_range = sample_clip_range
        self.num_classes = num_classes
        cond_dim = num_classes if num_classes is not None else 0

        # Noise schedule
        if beta_schedule == "linear":
            self.betas = np.linspace(beta_start, beta_end, time_steps, dtype=backend.default_dtype())
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
        self.alphas_cumprod_prev = np.concatenate([np.array([1.0]), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = np.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

        self.time_emb_dim = time_emb_dim

        # Build denoiser network
        self.denoiser = NeuralNet(learning_rate=learning_rate, optimizer=optimizer, l2_lambda=l2_lambda)

        if denoiser_type == "mlp":
            input_dim = self.data_dim + self.time_emb_dim + cond_dim
            prev = input_dim
            for h in denoiser_hidden:
                self.denoiser.add_dense(prev, h, activation="swish")
                prev = h
            self.denoiser.add_dense(prev, self.data_dim, activation="linear")
        elif denoiser_type == "conv":
            # Time embedding (and, if class-conditional, a one-hot label) is
            # broadcast spatially and concatenated as extra input channels
            # (see _predict_noise), hence the +time_emb_dim (+cond_dim).
            C, H, W = data_shape
            self.denoiser.add_conv2d(C + self.time_emb_dim + cond_dim, 64, k=3, activation="swish")
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

    def _onehot(self, y, n):
        """Validate `y` against `self.num_classes` (raises if one is given
        without the other) and return a one-hot (n, num_classes) array, or
        None if this DiffusionModel is unconditional."""
        if self.num_classes is None:
            if y is not None:
                raise ValueError("This DiffusionModel was built without num_classes -- don't pass y/labels.")
            return None
        if y is None:
            raise ValueError("This DiffusionModel was built with num_classes=... -- y/labels is required.")
        y = np.asarray(y).reshape(-1)
        if y.shape[0] == 1 and n > 1:
            y = np.repeat(y, n)
        if y.shape[0] != n:
            raise ValueError(f"y has {y.shape[0]} labels but batch size is {n}")
        return np.eye(self.num_classes, dtype=backend.default_dtype())[y]

    def _time_embedding(self, t):
        t = np.asarray(t, dtype=backend.default_dtype()).reshape(-1)
        half = self.time_emb_dim // 2
        freqs = np.exp(-np.log(10000) * np.arange(half) / half)
        args = t[:, None] * freqs[None, :]
        emb = np.concatenate([np.sin(args), np.cos(args)], axis=-1)
        if self.time_emb_dim % 2 == 1:
            emb = np.concatenate([emb, np.zeros((emb.shape[0], 1))], axis=-1)
        return emb

    def _forward_diffusion(self, x_0, t):
        x_0 = np.asarray(x_0, dtype=backend.default_dtype())
        noise = np.random.randn(*x_0.shape)
        sqrt_acp = self.sqrt_alphas_cumprod[t].reshape(-1, *([1] * (x_0.ndim - 1)))
        sqrt_omacp = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, *([1] * (x_0.ndim - 1)))
        x_t = sqrt_acp * x_0 + sqrt_omacp * noise
        return x_t, noise

    def _predict_noise(self, x_t, t, use_ema=True, y=None):
        """Predict the noise added to x_t at timestep t. use_ema=True during
        sampling/denoising; False during training (uses live weights)."""
        if use_ema and self.use_ema:
            self._use_ema_weights(True)
        try:
            onehot = self._onehot(y, x_t.shape[0])
            if self.denoiser_type == "mlp":
                x_t_flat = x_t.reshape(x_t.shape[0], -1)
                t_emb = self._time_embedding(t)
                parts = [x_t_flat, t_emb] + ([onehot] if onehot is not None else [])
                inp = np.concatenate(parts, axis=1)
                result = self.denoiser.Forward(inp, training=True).reshape(x_t.shape)
            elif self.denoiser_type == "conv":
                t_emb = self._time_embedding(t)  # (B, time_emb_dim)
                B, C, H, W = x_t.shape
                # Broadcast time embedding to spatial dimensions
                t_spatial = t_emb.reshape(B, self.time_emb_dim, 1, 1)
                t_spatial = np.repeat(np.repeat(t_spatial, H, axis=2), W, axis=3)
                channel_parts = [x_t, t_spatial]
                if onehot is not None:
                    # Broadcast the one-hot label spatially, same as the
                    # time embedding above.
                    y_spatial = onehot.reshape(B, self.num_classes, 1, 1)
                    y_spatial = np.repeat(np.repeat(y_spatial, H, axis=2), W, axis=3)
                    channel_parts.append(y_spatial)
                inp = np.concatenate(channel_parts, axis=1)  # (B, C+time_emb_dim(+num_classes), H, W)
                result = self.denoiser.Forward(inp, training=True)
        finally:
            if use_ema and self.use_ema:
                self._use_ema_weights(False)
        return result

    def train_step(self, x_0, y=None):
        x_0 = np.asarray(x_0, dtype=backend.default_dtype())
        batch_size = x_0.shape[0]

        t = np.random.randint(0, self.time_steps, size=batch_size)
        x_t, noise = self._forward_diffusion(x_0, t)
        pred_noise = self._predict_noise(x_t, t, use_ema=False, y=y)  # train on current weights

        loss = float(np.mean((pred_noise - noise) ** 2))

        delta = 2 * (pred_noise - noise) / batch_size
        if self.denoiser_type == "mlp":
            _manual_sequential_backward(self.denoiser, delta.reshape(batch_size, -1))
        elif self.denoiser_type == "conv":
            _manual_sequential_backward(self.denoiser, delta)
        self.denoiser.update()

        self._update_ema()

        return loss

    def Train(self, X_train, epochs=10, batch_size=64, verbose=True, y_train=None):
        X = np.asarray(X_train, dtype=backend.default_dtype())
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
                loss = self.train_step(batch, y=batch_y)
                epoch_loss += loss * batch.shape[0]
            avg_loss = epoch_loss / n_samples
            history.append(avg_loss)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Diffusion loss: {avg_loss:.6f}")
        return history

    def sample(self, n_samples=16, shape=None, clip=True, y=None):
        if shape is None:
            shape = self.data_shape

        if self.flattened:
            x = np.random.randn(n_samples, *shape)
        else:
            x = np.random.randn(n_samples, *shape)

        for t_step in reversed(range(self.time_steps)):
            t = np.full(n_samples, t_step, dtype=np.int64)

            pred_noise = self._predict_noise(x, t, use_ema=True, y=y)

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

    def sample_ddim(self, n_samples=16, n_steps=50, eta=0.0, shape=None, clip=True, y=None):
        """DDIM fast sampling: `n_steps` denoiser forward-passes over a
        strided subsequence of the full `time_steps` schedule, instead of
        `sample()`'s `time_steps` full ancestral-sampling steps.
        `eta=0.0` (default) is fully deterministic; `eta=1.0` reproduces
        `sample()`'s DDPM-like stochastic behavior at each subsequence step.
        """
        if shape is None:
            shape = self.data_shape
        x = np.random.randn(n_samples, *shape)

        # Strided descending timestep subsequence, e.g. n_steps=50 out of
        # time_steps=1000 -> roughly every 20th timestep.
        steps = np.unique(np.linspace(0, self.time_steps - 1, n_steps).astype(int))[::-1]

        for i, t_step in enumerate(steps):
            t = np.full(n_samples, t_step, dtype=np.int64)
            pred_noise = self._predict_noise(x, t, use_ema=True, y=y)

            alpha_cumprod_t = self.alphas_cumprod[t_step]
            alpha_cumprod_prev = self.alphas_cumprod[steps[i + 1]] if i < len(steps) - 1 else 1.0

            x0_pred = (x - np.sqrt(1.0 - alpha_cumprod_t) * pred_noise) / np.sqrt(alpha_cumprod_t)
            if clip:
                x0_pred = np.clip(x0_pred, self.sample_clip_range[0], self.sample_clip_range[1])

            sigma_t = eta * np.sqrt(
                (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t)
            ) * np.sqrt(1.0 - alpha_cumprod_t / alpha_cumprod_prev) if alpha_cumprod_prev < 1.0 else 0.0
            dir_coeff = np.sqrt(max(0.0, 1.0 - alpha_cumprod_prev - sigma_t ** 2))

            x = np.sqrt(alpha_cumprod_prev) * x0_pred + dir_coeff * pred_noise
            if sigma_t > 0:
                x = x + sigma_t * np.random.randn(*x.shape)

        if clip:
            x = np.clip(x, self.sample_clip_range[0], self.sample_clip_range[1])
        return x

    def denoise(self, x_noisy, t_start, t_end=0, y=None):
        x = x_noisy.copy()
        for t_step in reversed(range(t_end, t_start)):
            t = np.full(x.shape[0], t_step, dtype=np.int64)
            pred_noise = self._predict_noise(x, t, use_ema=True, y=y)

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
