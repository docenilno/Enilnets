import math
from typing import Any, List, Optional, Tuple

from ..core.backend import np
from ..core import backend
from ..core.base import NeuralNet
from .sampling import gaussian_sample
from ._shared import _manual_sequential_backward

class DiffusionModel:
    def __init__(self, data_shape: Tuple[int, ...], time_steps: int = 1000, beta_schedule: str = "linear",
                 beta_start: float = 1e-4, beta_end: float = 0.02, denoiser_type: str = "mlp",
                 denoiser_hidden: List[int] = [512, 512, 512], learning_rate: float = 0.001,
                 optimizer: str = "adam", l2_lambda: float = 0.0,
                 use_ema: bool = True, ema_decay: float = 0.999,
                 cosine_schedule_s: float = 0.008, beta_clip: Tuple[float, float] = (0, 0.999),
                 time_emb_dim: int = 128, sample_clip_range: Tuple[float, float] = (-1.0, 1.0),
                 num_classes: Optional[int] = None) -> None:
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
            t = np.arange(time_steps + 1, dtype=backend.default_dtype()) / time_steps
            alpha_bar = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = np.clip(1 - alpha_bar[1:] / alpha_bar[:-1], beta_clip[0], beta_clip[1])
            self.betas = np.asarray(betas, dtype=backend.default_dtype())
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        # np.array([1.0]) defaults to float64; concatenating it with the
        # (float32-by-default) alphas_cumprod array would upcast the WHOLE
        # result to float64 -- which then contaminates posterior_variance
        # below, and from there every sample()/denoise() call that reads
        # it (array-array ops promote to the higher precision, unlike
        # scalar-array ops, which don't).
        self.alphas_cumprod_prev = np.concatenate([
            np.array([1.0], dtype=backend.default_dtype()), self.alphas_cumprod[:-1]
        ])
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

    def _update_ema(self, init: bool = False) -> None:
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

    def _use_ema_weights(self, use: bool = True) -> None:
        """Temporarily swap to EMA weights for sampling."""
        if not self.use_ema or self.ema_weights is None:
            return
        if use:
            self._original_weights = self.denoiser.get_weights()
            self.denoiser.set_weights(self.ema_weights)
        else:
            self.denoiser.set_weights(self._original_weights)
            del self._original_weights

    def _onehot(self, y: Optional[Any], n: int) -> Optional[Any]:
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

    def _time_embedding(self, t: Any) -> Any:
        t = np.asarray(t, dtype=backend.default_dtype()).reshape(-1)
        half = self.time_emb_dim // 2
        # np.log(10000) (a bare Python int) returns a float64 numpy scalar,
        # which upcasts the whole expression to float64 regardless of
        # np.arange's own dtype -- explicit .astype at the end guards
        # against this rather than relying on every intermediate op to
        # preserve float32.
        freqs = np.exp(-np.log(10000) * np.arange(half, dtype=backend.default_dtype()) / half)
        freqs = freqs.astype(backend.default_dtype())
        args = t[:, None] * freqs[None, :]
        emb = np.concatenate([np.sin(args), np.cos(args)], axis=-1)
        if self.time_emb_dim % 2 == 1:
            emb = np.concatenate([emb, np.zeros((emb.shape[0], 1), dtype=backend.default_dtype())], axis=-1)
        return emb

    def _forward_diffusion(self, x_0: Any, t: Any) -> Tuple[Any, Any]:
        x_0 = np.asarray(x_0, dtype=backend.default_dtype())
        noise = np.random.randn(*x_0.shape).astype(backend.default_dtype())
        sqrt_acp = self.sqrt_alphas_cumprod[t].reshape(-1, *([1] * (x_0.ndim - 1)))
        sqrt_omacp = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, *([1] * (x_0.ndim - 1)))
        x_t = sqrt_acp * x_0 + sqrt_omacp * noise
        return x_t, noise

    def _predict_noise(self, x_t: Any, t: Any, use_ema: bool = True, y: Optional[Any] = None) -> Any:
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

    def train_step(self, x_0: Any, y: Optional[Any] = None) -> float:
        x_0 = np.asarray(x_0, dtype=backend.default_dtype())
        batch_size = x_0.shape[0]

        t = np.random.randint(0, self.time_steps, size=batch_size)
        x_t, noise = self._forward_diffusion(x_0, t)
        pred_noise = self._predict_noise(x_t, t, use_ema=False, y=y)  # train on current weights

        loss = float(np.mean((pred_noise - noise) ** 2))

        # loss is mean over ALL elements (batch * data_dim), so the gradient
        # must divide by pred_noise.size, not just batch_size, or it ends up
        # data_dim times too large relative to the reported loss scale.
        delta = 2 * (pred_noise - noise) / pred_noise.size
        if self.denoiser_type == "mlp":
            _manual_sequential_backward(self.denoiser, delta.reshape(batch_size, -1))
        elif self.denoiser_type == "conv":
            _manual_sequential_backward(self.denoiser, delta)
        self.denoiser.update()

        self._update_ema()

        return loss

    def Train(self, X_train: Any, epochs: int = 10, batch_size: int = 64, verbose: bool = True,
              y_train: Optional[Any] = None, callbacks: Optional[List[Any]] = None) -> List[float]:
        """callbacks: optional list of duck-typed callback objects (same
        convention as TextGenerator.Train/NeuralNet.Train). Supported hooks:
          on_batch_end(epoch, batch_idx, loss, model=self) -- after every
            minibatch's train_step.
          on_epoch_end(epoch, logs, model=self) -- once per epoch, with
            logs={"loss": avg_loss}.
          on_train_end(history) -- once after the epoch loop.
        Missing methods are skipped (no error)."""
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
            for batch_idx_num, i in enumerate(range(0, n_samples, batch_size)):
                batch_idx = indices[i:i+batch_size]
                batch = X[batch_idx]
                batch_y = y_arr[batch_idx] if y_arr is not None else None
                loss = self.train_step(batch, y=batch_y)
                epoch_loss += loss * batch.shape[0]
                for cb in (callbacks or []):
                    getattr(cb, "on_batch_end", lambda *a, **k: None)(epoch, batch_idx_num, loss, model=self)
            avg_loss = epoch_loss / n_samples
            history.append(avg_loss)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Diffusion loss: {avg_loss:.6f}")
            for cb in (callbacks or []):
                getattr(cb, "on_epoch_end", lambda *a, **k: None)(epoch, {"loss": avg_loss}, model=self)
        for cb in (callbacks or []):
            getattr(cb, "on_train_end", lambda *a, **k: None)(history)
        return history

    def sample(self, n_samples: int = 16, shape: Optional[Tuple[int, ...]] = None, clip: bool = True,
               y: Optional[Any] = None) -> Any:
        if shape is None:
            shape = self.data_shape

        x = np.random.randn(n_samples, *shape).astype(backend.default_dtype())

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
                noise = np.random.randn(*x.shape).astype(backend.default_dtype())
                x = mean + np.sqrt(variance) * noise
            else:
                x = mean

        if clip:
            x = np.clip(x, self.sample_clip_range[0], self.sample_clip_range[1])
        return x

    def sample_ddim(self, n_samples: int = 16, n_steps: int = 50, eta: float = 0.0,
                     shape: Optional[Tuple[int, ...]] = None, clip: bool = True,
                     y: Optional[Any] = None) -> Any:
        """DDIM fast sampling: `n_steps` denoiser forward-passes over a
        strided subsequence of the full `time_steps` schedule, instead of
        `sample()`'s `time_steps` full ancestral-sampling steps.
        `eta=0.0` (default) is fully deterministic; `eta=1.0` reproduces
        `sample()`'s DDPM-like stochastic behavior at each subsequence step.
        """
        if shape is None:
            shape = self.data_shape
        x = np.random.randn(n_samples, *shape).astype(backend.default_dtype())

        # Strided descending timestep subsequence, e.g. n_steps=50 out of
        # time_steps=1000 -> roughly every 20th timestep.
        steps = np.unique(np.linspace(0, self.time_steps - 1, n_steps).astype(int))[::-1]

        for i, t_step in enumerate(steps):
            t = np.full(n_samples, t_step, dtype=np.int64)
            pred_noise = self._predict_noise(x, t, use_ema=True, y=y)

            alpha_cumprod_t = self.alphas_cumprod[t_step]
            # The final step's alpha_cumprod_prev is a bare Python float
            # (1.0), not an array-derived value like every other step --
            # np.sqrt() of a plain Python float returns a float64 numpy
            # scalar regardless of the array dtype elsewhere, which then
            # upcasts `x` to float64 for the rest of the function (the
            # last iteration's `x = np.sqrt(alpha_cumprod_prev) * ...`
            # below). Cast explicitly so this step is dtype-consistent
            # with every other one.
            alpha_cumprod_prev = self.alphas_cumprod[steps[i + 1]] if i < len(steps) - 1 \
                else np.asarray(1.0, dtype=backend.default_dtype())

            x0_pred = (x - np.sqrt(1.0 - alpha_cumprod_t) * pred_noise) / np.sqrt(alpha_cumprod_t)
            if clip:
                x0_pred = np.clip(x0_pred, self.sample_clip_range[0], self.sample_clip_range[1])

            sigma_t = eta * np.sqrt(
                (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t)
            ) * np.sqrt(1.0 - alpha_cumprod_t / alpha_cumprod_prev) if alpha_cumprod_prev < 1.0 else 0.0
            # Python's builtin max(0.0, ...) returns the bare Python float
            # 0.0 whenever that branch is picked, not a same-dtype numpy
            # scalar -- np.sqrt() of a plain Python float then returns a
            # float64 numpy scalar regardless of every other array's
            # dtype, silently upcasting `x` from that point on. np.maximum
            # (the backend-proxied ufunc, not Python's builtin) plus
            # np.asarray(..., dtype=...) casts both derived scalars
            # explicitly and correctly under both NumPy and CuPy (unlike
            # calling the raw dtype constructor directly on a value that
            # might already be a CuPy array, which CuPy rejects as an
            # implicit device->host conversion).
            sigma_t = np.asarray(sigma_t, dtype=backend.default_dtype())
            zero = np.asarray(0.0, dtype=backend.default_dtype())
            dir_coeff = np.sqrt(np.maximum(zero, 1.0 - alpha_cumprod_prev - sigma_t ** 2))
            dir_coeff = np.asarray(dir_coeff, dtype=backend.default_dtype())

            x = np.sqrt(alpha_cumprod_prev) * x0_pred + dir_coeff * pred_noise
            if sigma_t > 0:
                noise = np.random.randn(*x.shape).astype(backend.default_dtype())
                x = x + sigma_t * noise

        if clip:
            x = np.clip(x, self.sample_clip_range[0], self.sample_clip_range[1])
        return x

    def denoise(self, x_noisy: Any, t_start: int, t_end: int = 0, y: Optional[Any] = None) -> Any:
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
                noise = np.random.randn(*x.shape).astype(backend.default_dtype())
                x = mean + np.sqrt(variance) * noise
            else:
                x = mean
        return x
