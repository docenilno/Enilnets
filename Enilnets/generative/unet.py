from ..backend import np
from .. import backend
from ..base import NeuralNet

def time_embedding(t, dim, max_period=10000):
    """
    Sinusoidal time embedding for diffusion models.
    t: (batch,) array of timestep indices or (batch, 1)
    dim: embedding dimension (must be even)
    Returns: (batch, dim) embedding
    """
    t = np.asarray(t, dtype=backend.default_dtype()).reshape(-1)
    half = dim // 2
    freqs = np.exp(-np.log(max_period) * np.arange(half) / half)
    args = t[:, None] * freqs[None, :]
    emb = np.concatenate([np.sin(args), np.cos(args)], axis=-1)
    if dim % 2 == 1:
        emb = np.concatenate([emb, np.zeros((emb.shape[0], 1))], axis=-1)
    return emb

class UNetDenoiser:
    def __init__(self, in_ch, base_ch=64, time_emb_dim=128, ch_mult=(1, 2, 4),
                 learning_rate=0.001, optimizer="adam", l2_lambda=0.0, init_method="he_normal",
                 pool_factor=2, time_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.in_ch = in_ch
        self.base_ch = base_ch
        self.time_emb_dim = time_emb_dim
        self.ch_mult = ch_mult
        self.levels = len(ch_mult)
        self.pool_factor = pool_factor
        net_kwargs = dict(learning_rate=learning_rate, optimizer=optimizer, l2_lambda=l2_lambda)

        # Simple linear DDPM noise schedule (mirrors DiffusionModel's
        # "linear" schedule) so this class is directly trainable/sample-able
        # on its own via train_step/Train/sample, without needing a separate
        # DiffusionModel wrapper.
        self.time_steps = time_steps
        self.betas = np.linspace(beta_start, beta_end, time_steps, dtype=backend.default_dtype())
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.concatenate([np.array([1.0]), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

        # Time embedding MLP
        self.time_net = NeuralNet(**net_kwargs)
        self.time_net.add_dense(time_emb_dim, time_emb_dim * 4, activation="swish")
        self.time_net.add_dense(time_emb_dim * 4, time_emb_dim * 4, activation="swish")

        # Encoder blocks
        self.encoders = []
        prev_ch = in_ch
        for i, mult in enumerate(ch_mult):
            out_ch = base_ch * mult
            net = NeuralNet(**net_kwargs)
            net.add_conv2d(prev_ch, out_ch, k=3, activation="swish", init_method=init_method, padding="same")
            net.add_conv2d(out_ch, out_ch, k=3, activation="swish", init_method=init_method, padding="same")
            self.encoders.append(net)
            prev_ch = out_ch

        # Bottleneck
        self.bottleneck = NeuralNet(**net_kwargs)
        bot_ch = base_ch * ch_mult[-1]
        self.bottleneck.add_conv2d(bot_ch, bot_ch, k=3, activation="swish", init_method=init_method, padding="same")
        self.bottleneck.add_conv2d(bot_ch, bot_ch, k=3, activation="swish", init_method=init_method, padding="same")

        # Decoder blocks
        self.decoders = []
        for i in reversed(range(len(ch_mult))):
            out_ch = base_ch * ch_mult[i]
            in_ch_dec = base_ch * ch_mult[min(i+1, len(ch_mult)-1)] if i < len(ch_mult)-1 else bot_ch
            net = NeuralNet(**net_kwargs)
            net.add_conv2d(in_ch_dec + out_ch, out_ch, k=3, activation="swish", init_method=init_method, padding="same")
            net.add_conv2d(out_ch, out_ch, k=3, activation="swish", init_method=init_method, padding="same")
            self.decoders.append(net)

        # Output conv
        self.out_net = NeuralNet(**net_kwargs)
        self.out_net.add_conv2d(base_ch * ch_mult[0], in_ch, k=3, activation="linear", init_method=init_method, padding="same")

    def _add_time_to_feature(self, x, t_emb):
        B, C, H, W = x.shape
        if t_emb.shape[1] == C:
            return x + t_emb.reshape(B, C, 1, 1)
        return x

    def _downsample(self, x):
        B, C, H, W = x.shape
        p = self.pool_factor
        if H < p or W < p:
            return x
        return x[:, :, :H//p*p, :W//p*p].reshape(B, C, H//p, p, W//p, p).mean(axis=(3, 5))

    def _upsample(self, x, target_H, target_W):
        B, C, H, W = x.shape
        if H == target_H and W == target_W:
            return x
        if H == 0 or W == 0:
            return np.zeros((B, C, target_H, target_W), dtype=backend.default_dtype())
        scale_h = max(1, target_H // H)
        scale_w = max(1, target_W // W)
        x = x.repeat(scale_h, axis=2).repeat(scale_w, axis=3)
        if x.shape[2] > target_H or x.shape[3] > target_W:
            x = x[:, :, :target_H, :target_W]
        if x.shape[2] < target_H or x.shape[3] < target_W:
            pad_h = target_H - x.shape[2]
            pad_w = target_W - x.shape[3]
            x = np.pad(x, ((0,0), (0,0), (0, pad_h), (0, pad_w)), mode='constant')
        return x

    def forward(self, x, t):
        x = np.asarray(x, dtype=backend.default_dtype())
        if x.ndim == 3:
            x = x.reshape(1, *x.shape)
        if x.ndim == 2:
            B = x.shape[0]
            side = int(np.sqrt(x.shape[1]))
            if side * side == x.shape[1]:
                x = x.reshape(B, 1, side, side)
            else:
                x = x.reshape(B, 1, x.shape[1], 1)

        orig_shape = x.shape
        t = np.asarray(t, dtype=backend.default_dtype()).reshape(-1)

        # Time embedding
        t_emb = time_embedding(t, self.time_emb_dim)
        t_emb = self.time_net.Forward(t_emb, training=True)

        # Encoder path with downsampling
        skips = []
        h = x
        for i, enc in enumerate(self.encoders):
            h = enc.Forward(h, training=True)
            h = self._add_time_to_feature(h, t_emb)
            skips.append(h)
            if i < len(self.encoders) - 1:
                h = self._downsample(h)

        # Bottleneck
        h = self.bottleneck.Forward(h, training=True)

        # Decoder path with upsampling and skip connections
        for i, dec in enumerate(self.decoders):
            skip = skips[-(i+1)]
            target_H, target_W = skip.shape[2], skip.shape[3]
            h = self._upsample(h, target_H, target_W)
            h = np.concatenate([h, skip], axis=1)
            h = dec.Forward(h, training=True)
            h = self._add_time_to_feature(h, t_emb)

        # Output - ensure exact original shape
        out = self.out_net.Forward(h, training=True)
        if out.shape != orig_shape:
            B, C, H, W = orig_shape
            if out.shape[2] > H or out.shape[3] > W:
                out = out[:, :, :H, :W]
            if out.shape[2] < H or out.shape[3] < W:
                pad_h = H - out.shape[2]
                pad_w = W - out.shape[3]
                out = np.pad(out, ((0,0), (0,0), (0, pad_h), (0, pad_w)), mode='constant')
        return out

    def _resize_backward(self, delta, H, W, target_H, target_W):
        """Invert a forward crop-then-zero-pad from (H, W) to (target_H,
        target_W) (no repeat/scale involved -- see `_upsample_backward` for
        that). Cropped regions simply get zero gradient (they were dropped
        from the output entirely); padded regions' gradient is discarded
        (the padding was constant zero, disconnected from the input)."""
        B, C = delta.shape[0], delta.shape[1]
        copy_H, copy_W = min(H, target_H), min(W, target_W)
        grad = np.zeros((B, C, H, W), dtype=backend.default_dtype())
        grad[:, :, :copy_H, :copy_W] = delta[:, :, :copy_H, :copy_W]
        return grad

    def _upsample_backward(self, delta, H, W, target_H, target_W):
        """Exact inverse of `_upsample(x, target_H, target_W)` for an input
        of spatial size (H, W): nearest-neighbor repeat by (scale_h,
        scale_w), then crop-or-pad to (target_H, target_W)."""
        if H == target_H and W == target_W:
            return delta
        if H == 0 or W == 0:
            return np.zeros((delta.shape[0], delta.shape[1], H, W), dtype=backend.default_dtype())
        scale_h = max(1, target_H // H)
        scale_w = max(1, target_W // W)
        rep_H, rep_W = H * scale_h, W * scale_w
        grad_rep = self._resize_backward(delta, rep_H, rep_W, target_H, target_W)
        B, C = delta.shape[0], delta.shape[1]
        return grad_rep.reshape(B, C, H, scale_h, W, scale_w).sum(axis=(3, 5))

    def backward(self, grad_output):
        """Backprop a gradient w.r.t. `forward()`'s output through every
        subnetwork (time_net, encoders, bottleneck, decoders, out_net) in
        exact reverse of `forward()`'s order, populating each subnetwork's
        `.deltas`/per-layer gradients so `.update()` can be called on each
        (see `train_step`). Does not return a gradient w.r.t. the raw input
        `x` -- nothing downstream of `forward()`'s input needs it.

        Mirrors `forward()`'s structure precisely: `out_net` -> reverse over
        `decoders` -> `bottleneck` -> reverse over `encoders` -> `time_net`.
        Every skip connection is used in exactly two places in `forward()`
        (its matching decoder's concat, and -- except at the deepest level --
        the next encoder's downsample input / the bottleneck), so its
        gradient is the *sum* of both paths' contributions before it's
        routed on backward through the encoder that produced it.
        """
        from ..backward import avgpool2d_backward
        from ._shared import _manual_sequential_backward, _conv_stack_input_gradient

        grad_output = np.asarray(grad_output, dtype=backend.default_dtype())
        levels = self.levels
        t_emb_dim = self.time_net.outputs[-1].shape[1]
        d_t_emb = np.zeros((grad_output.shape[0], t_emb_dim), dtype=backend.default_dtype())

        def consume_time_add(delta, net):
            # `delta`: gradient w.r.t. a tensor that (if channel counts
            # match) had t_emb broadcast-added to it. Broadcast-add's
            # backward passes `delta` through unchanged to both operands --
            # the harvested sum-over-space contribution goes to t_emb, and
            # the caller keeps using `delta` itself for the conv side. Safe
            # to call once per distinct contribution path (not once on
            # their sum) since sum-over-space is linear.
            C = net.outputs[-1].shape[1]
            if C == t_emb_dim:
                d_t_emb[...] += delta.sum(axis=(2, 3))
            return delta

        # --- out_net: undo the final crop/pad-to-orig_shape, then backprop ---
        orig_shape = self.encoders[0].outputs[0].shape
        raw_out_shape = self.out_net.outputs[-1].shape
        if raw_out_shape[2:] != orig_shape[2:]:
            d_out_raw = self._resize_backward(grad_output, raw_out_shape[2], raw_out_shape[3],
                                               orig_shape[2], orig_shape[3])
        else:
            d_out_raw = grad_output
        _manual_sequential_backward(self.out_net, d_out_raw)
        d_h = consume_time_add(_conv_stack_input_gradient(self.out_net), self.decoders[-1])

        # --- decoders, in reverse of forward()'s enumerate() order ---
        # decoders[pos] handles level = levels-1-pos (see __init__'s
        # `reversed(range(len(ch_mult)))` build order).
        d_skip_accum = [None] * levels
        d_bottleneck_out = None
        for pos in reversed(range(levels)):
            dec = self.decoders[pos]
            level = levels - 1 - pos
            _manual_sequential_backward(dec, d_h)
            d_concat = _conv_stack_input_gradient(dec)

            out_ch = self.base_ch * self.ch_mult[level]
            total_in_ch = dec.layers[0]["in_ch"]
            in_ch_dec = total_in_ch - out_ch
            d_upsampled = d_concat[:, :in_ch_dec]
            d_skip = d_concat[:, in_ch_dec:]
            d_skip_accum[level] = d_skip if d_skip_accum[level] is None else d_skip_accum[level] + d_skip

            if pos == 0:
                prev_H, prev_W = self.bottleneck.outputs[-1].shape[2:]
            else:
                prev_H, prev_W = self.decoders[pos - 1].outputs[-1].shape[2:]
            skip_H, skip_W = self.encoders[level].outputs[-1].shape[2:]
            d_prev_out = self._upsample_backward(d_upsampled, prev_H, prev_W, skip_H, skip_W)

            if pos == 0:
                d_bottleneck_out = d_prev_out
            else:
                d_h = consume_time_add(d_prev_out, self.decoders[pos - 1])

        # --- bottleneck (no time-add applied to it in forward()) ---
        _manual_sequential_backward(self.bottleneck, d_bottleneck_out)
        d_deepest_skip = consume_time_add(_conv_stack_input_gradient(self.bottleneck), self.encoders[levels - 1])
        d_skip_accum[levels - 1] = (
            d_deepest_skip if d_skip_accum[levels - 1] is None else d_skip_accum[levels - 1] + d_deepest_skip
        )

        # --- encoders, in reverse (deepest level first) ---
        for i in reversed(range(levels)):
            if i < levels - 1:
                d_next_input = _conv_stack_input_gradient(self.encoders[i + 1])
                pre_downsample = self.encoders[i].outputs[-1]
                H, W = pre_downsample.shape[2], pre_downsample.shape[3]
                if H < self.pool_factor or W < self.pool_factor:
                    d_from_downsample = d_next_input  # _downsample was an identity no-op
                else:
                    d_from_downsample = avgpool2d_backward(d_next_input, pre_downsample, self.pool_factor)
                total = d_skip_accum[i] + d_from_downsample if d_skip_accum[i] is not None else d_from_downsample
            else:
                total = d_skip_accum[i]
            total = consume_time_add(total, self.encoders[i])
            _manual_sequential_backward(self.encoders[i], total)

        # --- time embedding MLP ---
        _manual_sequential_backward(self.time_net, d_t_emb)

    def _predict_noise(self, x_t, t):
        return self.forward(x_t, t)

    def train_step(self, x_0):
        """One DDPM training step: sample a random timestep per example,
        noise `x_0` accordingly, predict the noise via `forward()`, MSE
        loss against the true noise, backprop + update every subnetwork."""
        x_0 = np.asarray(x_0, dtype=backend.default_dtype())
        batch_size = x_0.shape[0]
        t = np.random.randint(0, self.time_steps, size=batch_size)
        noise = np.random.randn(*x_0.shape)
        sqrt_acp = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_omacp = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        x_t = sqrt_acp * x_0 + sqrt_omacp * noise

        pred_noise = self._predict_noise(x_t, t)
        loss = float(np.mean((pred_noise - noise) ** 2))

        grad_output = 2 * (pred_noise - noise) / pred_noise.size
        self.backward(grad_output)
        for net in self.get_params():
            net.update()
        return loss

    def Train(self, X_train, epochs=10, batch_size=16, verbose=True):
        X = np.asarray(X_train, dtype=backend.default_dtype())
        n_samples = X.shape[0]
        history = []
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            for i in range(0, n_samples, batch_size):
                batch = X[indices[i:i + batch_size]]
                loss = self.train_step(batch)
                epoch_loss += loss * batch.shape[0]
            avg_loss = epoch_loss / n_samples
            history.append(avg_loss)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - UNet denoising loss: {avg_loss:.6f}")
        return history

    def sample(self, n_samples=16, shape=None, clip=None):
        """Standard DDPM ancestral sampling (same formula as
        `DiffusionModel.sample`), driven by this class's own noise
        schedule."""
        if shape is None:
            raise ValueError("sample() requires shape=(C, H, W) -- UNetDenoiser has no fixed data_shape.")
        x = np.random.randn(n_samples, *shape)
        for t_step in reversed(range(self.time_steps)):
            t = np.full(n_samples, t_step, dtype=np.int64)
            pred_noise = self._predict_noise(x, t)
            alpha_t = self.alphas[t].reshape(-1, 1, 1, 1)
            alpha_cumprod_t = self.alphas_cumprod[t].reshape(-1, 1, 1, 1)
            beta_t = self.betas[t].reshape(-1, 1, 1, 1)
            coef1 = 1.0 / np.sqrt(alpha_t)
            coef2 = beta_t / (np.sqrt(1.0 - alpha_cumprod_t) * np.sqrt(alpha_t))
            mean = coef1 * (x - coef2 * pred_noise)
            if t_step > 0:
                variance = self.posterior_variance[t].reshape(-1, 1, 1, 1)
                x = mean + np.sqrt(variance) * np.random.randn(*x.shape)
            else:
                x = mean
        if clip is not None:
            x = np.clip(x, clip[0], clip[1])
        return x

    def get_params(self):
        return [self.time_net] + self.encoders + [self.bottleneck] + self.decoders + [self.out_net]
