import numpy as np
from ..base import NeuralNet

def time_embedding(t, dim, max_period=10000):
    """
    Sinusoidal time embedding for diffusion models.
    t: (batch,) array of timestep indices or (batch, 1)
    dim: embedding dimension (must be even)
    Returns: (batch, dim) embedding
    """
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    half = dim // 2
    freqs = np.exp(-np.log(max_period) * np.arange(half) / half)
    args = t[:, None] * freqs[None, :]
    emb = np.concatenate([np.sin(args), np.cos(args)], axis=-1)
    if dim % 2 == 1:
        emb = np.concatenate([emb, np.zeros((emb.shape[0], 1))], axis=-1)
    return emb

class UNetDenoiser:
    """
    Simple UNet-like denoiser for diffusion models.
    Uses k=1 convolutions to avoid spatial shrinking (since base conv2d has no padding).
    Time is embedded and broadcast as extra channels.

    Parameters
    ----------
    in_ch : int
        Number of input channels
    base_ch : int
        Base number of channels
    time_emb_dim : int
        Time embedding dimension
    ch_mult : tuple
        Channel multipliers for each level
    """
    def __init__(self, in_ch, base_ch=64, time_emb_dim=128, ch_mult=(1, 2, 4)):
        self.in_ch = in_ch
        self.base_ch = base_ch
        self.time_emb_dim = time_emb_dim
        self.ch_mult = ch_mult
        self.levels = len(ch_mult)

        # Time embedding MLP
        self.time_net = NeuralNet(learning_rate=0.001, optimizer="adam", l2_lambda=0.0)
        self.time_net.add_dense(time_emb_dim, time_emb_dim * 4, activation="swish")
        self.time_net.add_dense(time_emb_dim * 4, time_emb_dim * 4, activation="swish")

        # Encoder blocks: each level has [Conv -> Conv] with k=1 (no spatial shrink)
        self.encoders = []
        prev_ch = in_ch
        for i, mult in enumerate(ch_mult):
            out_ch = base_ch * mult
            net = NeuralNet(learning_rate=0.001, optimizer="adam", l2_lambda=0.0)
            net.add_conv2d(prev_ch, out_ch, k=1, activation="swish", init_method="he_normal")
            net.add_conv2d(out_ch, out_ch, k=1, activation="swish", init_method="he_normal")
            self.encoders.append(net)
            prev_ch = out_ch

        # Bottleneck
        self.bottleneck = NeuralNet(learning_rate=0.001, optimizer="adam", l2_lambda=0.0)
        bot_ch = base_ch * ch_mult[-1]
        self.bottleneck.add_conv2d(bot_ch, bot_ch, k=1, activation="swish", init_method="he_normal")
        self.bottleneck.add_conv2d(bot_ch, bot_ch, k=1, activation="swish", init_method="he_normal")

        # Decoder blocks
        self.decoders = []
        for i in reversed(range(len(ch_mult))):
            out_ch = base_ch * ch_mult[i]
            in_ch_dec = base_ch * ch_mult[min(i+1, len(ch_mult)-1)] if i < len(ch_mult)-1 else bot_ch
            net = NeuralNet(learning_rate=0.001, optimizer="adam", l2_lambda=0.0)
            net.add_conv2d(in_ch_dec + out_ch, out_ch, k=1, activation="swish", init_method="he_normal")
            net.add_conv2d(out_ch, out_ch, k=1, activation="swish", init_method="he_normal")
            self.decoders.append(net)

        # Output conv
        self.out_net = NeuralNet(learning_rate=0.001, optimizer="adam", l2_lambda=0.0)
        self.out_net.add_conv2d(base_ch * ch_mult[0], in_ch, k=1, activation="linear", init_method="he_normal")

        # Store original shapes for skip connections
        self._skip_shapes = []

    def _add_time_to_feature(self, x, t_emb):
        """Broadcast time embedding and add to spatial features."""
        B, C, H, W = x.shape
        if t_emb.shape[1] == C:
            return x + t_emb.reshape(B, C, 1, 1)
        return x

    def _downsample(self, x):
        """Average pool by factor of 2."""
        B, C, H, W = x.shape
        p = 2
        if H < p or W < p:
            return x
        return x[:, :, :H//p*p, :W//p*p].reshape(B, C, H//p, p, W//p, p).mean(axis=(3, 5))

    def _upsample(self, x, target_H, target_W):
        """Nearest neighbor upsample."""
        B, C, H, W = x.shape
        if H == target_H and W == target_W:
            return x
        if H == 0 or W == 0:
            return np.zeros((B, C, target_H, target_W), dtype=np.float64)
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
        """
        x: (B, C, H, W) noisy image
        t: (B,) or (B, 1) timestep indices
        Returns: (B, C, H, W) predicted noise
        """
        x = np.asarray(x, dtype=np.float64)
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
        t = np.asarray(t, dtype=np.float64).reshape(-1)

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
            # Crop or pad to match
            B, C, H, W = orig_shape
            if out.shape[2] > H or out.shape[3] > W:
                out = out[:, :, :H, :W]
            if out.shape[2] < H or out.shape[3] < W:
                pad_h = H - out.shape[2]
                pad_w = W - out.shape[3]
                out = np.pad(out, ((0,0), (0,0), (0, pad_h), (0, pad_w)), mode='constant')
        return out

    def backward(self, grad_output):
        """
        Backpropagate through the UNet.
        """
        raise NotImplementedError(
            "UNetDenoiser.backward requires custom implementation. "
            "Use DiffusionModel with mlp_denoiser=True for a fully trainable diffusion model."
        )

    def get_params(self):
        """Return all parameter-containing NeuralNet instances."""
        return [self.time_net] + self.encoders + [self.bottleneck] + self.decoders + [self.out_net]
