"""Composable transforms (roadmap item 51).

A transform is any callable ``sample -> sample``; :class:`Compose` chains
them. The classes here wrap the existing per-modality functions so they
compose without each caller re-deriving argument orders.

Two places to attach one, and the difference matters: ``dataset.map(t)``
runs per SAMPLE, so a random transform draws fresh randomness for each --
what augmentation wants. ``DataLoader(transform=t)`` runs on the collated
BATCH, so one vectorized call covers it but a random draw is shared across
it -- cheaper, and correct for deterministic transforms."""

from typing import Any, Callable, Optional, Sequence

from ..core.backend import np
from ..core import backend


class Transform:
    """Base class: subclasses implement ``__call__``. Only exists so
    transforms have a uniform repr and a place for shared helpers."""

    def __call__(self, sample: Any) -> Any:
        raise NotImplementedError

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items()
                         if not k.startswith("_"))
        return f"{type(self).__name__}({args})"


class Compose(Transform):
    """Apply transforms left to right. Empty is a valid identity."""

    def __init__(self, transforms: Sequence[Callable[[Any], Any]]) -> None:
        self.transforms = list(transforms)

    def __call__(self, sample: Any) -> Any:
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __len__(self) -> int:
        return len(self.transforms)

    def __repr__(self) -> str:
        inner = ", ".join(repr(t) for t in self.transforms)
        return f"Compose([{inner}])"


class OnX(Transform):
    """Apply an inner transform to only the x half of an ``(x, y)`` sample,
    passing y through. Bare (unlabelled) samples are transformed directly.

    This is what lets an image augmentation sit in the same Compose as a
    label transform without either needing to know the sample's shape."""

    def __init__(self, transform: Callable[[Any], Any]) -> None:
        self.transform = transform

    def __call__(self, sample: Any) -> Any:
        if isinstance(sample, tuple):
            return (self.transform(sample[0]),) + tuple(sample[1:])
        return self.transform(sample)


class OnY(Transform):
    """Apply an inner transform to only the y half of an ``(x, y)`` sample."""

    def __init__(self, transform: Callable[[Any], Any]) -> None:
        self.transform = transform

    def __call__(self, sample: Any) -> Any:
        if not isinstance(sample, tuple) or len(sample) < 2:
            raise ValueError("OnY needs an (x, y) sample")
        return (sample[0], self.transform(sample[1])) + tuple(sample[2:])


class Lambda(Transform):
    """Wrap an arbitrary callable so it reprs like the rest."""

    def __init__(self, fn: Callable[[Any], Any], name: str = "") -> None:
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "fn")

    def __call__(self, sample: Any) -> Any:
        return self.fn(sample)


class RandomApply(Transform):
    """Apply `transform` with probability `p`, per call."""

    def __init__(self, transform: Callable[[Any], Any], p: float = 0.5,
                 seed: Optional[int] = None) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.transform, self.p = transform, p
        self._rng = _rng(seed)

    def __call__(self, sample: Any) -> Any:
        return self.transform(sample) if self._rng.rand() < self.p else sample


class OneOf(Transform):
    """Pick one of `transforms` uniformly at random, per call."""

    def __init__(self, transforms: Sequence[Callable[[Any], Any]],
                 seed: Optional[int] = None) -> None:
        if not transforms:
            raise ValueError("OneOf needs at least one transform")
        self.transforms = list(transforms)
        self._rng = _rng(seed)

    def __call__(self, sample: Any) -> Any:
        return self.transforms[int(self._rng.randint(len(self.transforms)))](sample)


def _rng(seed: Optional[int]):
    import numpy as _host_np
    return _host_np.random.RandomState(seed) if seed is not None else _host_np.random


# --------------------------------------------------------------- numeric

class ToDtype(Transform):
    """Cast to a dtype; defaults to the backend's current working dtype, so
    a float64 dataset does not silently drag a float32 model up."""

    def __init__(self, dtype: Any = None) -> None:
        self.dtype = dtype

    def __call__(self, x: Any) -> Any:
        return np.asarray(x).astype(self.dtype or backend.default_dtype())


class Scale(Transform):
    """Multiply by `factor` -- e.g. ``Scale(1 / 255)`` for uint8 pixels."""

    def __init__(self, factor: float) -> None:
        self.factor = float(factor)

    def __call__(self, x: Any) -> Any:
        return np.asarray(x) * self.factor


class Normalize(Transform):
    """Subtract `mean`, divide by `std`. Scalars or per-channel arrays.

    Unlike ``preprocessing.normalize_images``, which *computes* statistics
    from the data it is given, this applies FIXED ones -- which is what you
    want per batch, where recomputing would make each batch normalized
    differently."""

    def __init__(self, mean: Any = 0.0, std: Any = 1.0, epsilon: float = 1e-8) -> None:
        self.mean, self.std, self.epsilon = mean, std, epsilon

    def __call__(self, x: Any) -> Any:
        x = np.asarray(x)
        # Cast the constants to x's dtype first. Left alone they are float64
        # arrays, and float32 - float64 promotes the whole batch to float64 --
        # which then silently feeds a float64 batch into a float32 model.
        mean = np.asarray(self.mean, dtype=x.dtype)
        std = np.asarray(self.std, dtype=x.dtype)
        return (x - mean) / (std + self.epsilon)


class Clip(Transform):
    def __init__(self, low: float = 0.0, high: float = 1.0) -> None:
        self.low, self.high = float(low), float(high)

    def __call__(self, x: Any) -> Any:
        x = np.asarray(x)
        return np.clip(x, np.asarray(self.low, dtype=x.dtype),
                       np.asarray(self.high, dtype=x.dtype))


class Reshape(Transform):
    """Reshape, leaving the leading axis alone if `keep_leading` is set --
    which is what makes one transform work on both a sample and a batch."""

    def __init__(self, shape: Sequence[int], keep_leading: bool = False) -> None:
        self.shape = tuple(shape)
        self.keep_leading = keep_leading

    def __call__(self, x: Any) -> Any:
        x = np.asarray(x)
        return x.reshape((x.shape[0],) + self.shape) if self.keep_leading \
            else x.reshape(self.shape)


class OneHot(Transform):
    """Integer class indices -> one-hot rows."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = int(num_classes)

    def __call__(self, y: Any) -> Any:
        idx = np.asarray(y).astype(np.int64)
        out = np.zeros(idx.shape + (self.num_classes,), dtype=backend.default_dtype())
        flat = out.reshape(-1, self.num_classes)
        flat[np.arange(flat.shape[0]), idx.reshape(-1)] = 1.0
        return out


# ----------------------------------------------------------------- image

class RandomFlip(Transform):
    """Flip images horizontally and/or vertically at random.

    NCHW or NHW, matching this library's conv2d convention -- the spatial
    axes are the last two either way, so this works on a single image and a
    batch alike."""

    def __init__(self, horizontal: bool = True, vertical: bool = False,
                 p: float = 0.5, seed: Optional[int] = None) -> None:
        self.horizontal, self.vertical, self.p = horizontal, vertical, float(p)
        self._rng = _rng(seed)

    def __call__(self, x: Any) -> Any:
        x = np.asarray(x)
        if self.horizontal and self._rng.rand() < self.p:
            x = x[..., ::-1]
        if self.vertical and self._rng.rand() < self.p:
            x = x[..., ::-1, :]
        # A reversed view is not contiguous; copying keeps downstream
        # im2col's stride tricks valid.
        return np.ascontiguousarray(x)


class RandomCrop(Transform):
    """Crop a random `size` window out of the last two axes, optionally
    zero-padding first (the standard CIFAR-style pad-4-then-crop)."""

    def __init__(self, size: Any, padding: int = 0, seed: Optional[int] = None) -> None:
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.padding = int(padding)
        self._rng = _rng(seed)

    def __call__(self, x: Any) -> Any:
        x = np.asarray(x)
        if self.padding:
            pad = [(0, 0)] * (x.ndim - 2) + [(self.padding, self.padding)] * 2
            x = np.pad(x, pad, mode="constant")
        h, w = x.shape[-2], x.shape[-1]
        th, tw = self.size
        if th > h or tw > w:
            raise ValueError(
                f"crop size {self.size} exceeds image size {(h, w)} "
                f"(after padding={self.padding})")
        top = int(self._rng.randint(h - th + 1))
        left = int(self._rng.randint(w - tw + 1))
        return np.ascontiguousarray(x[..., top:top + th, left:left + tw])


class CenterCrop(Transform):
    def __init__(self, size: Any) -> None:
        self.size = (size, size) if isinstance(size, int) else tuple(size)

    def __call__(self, x: Any) -> Any:
        x = np.asarray(x)
        h, w = x.shape[-2], x.shape[-1]
        th, tw = self.size
        top, left = (h - th) // 2, (w - tw) // 2
        return np.ascontiguousarray(x[..., top:top + th, left:left + tw])


class Resize(Transform):
    """Resize the last two axes. `mode` is "bilinear" or "nearest"."""

    def __init__(self, size: Any, mode: str = "bilinear") -> None:
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        if mode not in ("bilinear", "nearest"):
            raise ValueError(f"mode must be 'bilinear' or 'nearest', got {mode!r}")
        self.mode = mode

    def __call__(self, x: Any) -> Any:
        from ..vision.image_utils import resize_bilinear, resize_nearest_neighbor
        fn = resize_bilinear if self.mode == "bilinear" else resize_nearest_neighbor
        x = np.asarray(x)
        h, w = self.size
        # image_utils works on (H, W[, C]); map over any leading axes so the
        # same transform handles a lone image, a channel stack and a batch.
        # image_utils' resamplers compute in float64; cast back so a
        # transform never silently changes the pipeline's working precision.
        if x.ndim == 2:
            return np.asarray(fn(x, h, w)).astype(x.dtype)
        lead, spatial = x.shape[:-2], x.shape[-2:]
        flat = x.reshape((-1,) + spatial)
        out = np.stack([fn(flat[i], h, w) for i in range(flat.shape[0])])
        return out.reshape(lead + (h, w)).astype(x.dtype)


class RandomNoise(Transform):
    def __init__(self, std: float = 0.05, seed: Optional[int] = None) -> None:
        self.std = float(std)
        self._rng = _rng(seed)

    def __call__(self, x: Any) -> Any:
        x = np.asarray(x)
        noise = np.asarray(self._rng.randn(*x.shape)).astype(x.dtype)
        return x + noise * self.std


class Augment(Transform):
    """The existing ``preprocessing.image_augmentation`` as a transform.

    Batch-only: that function indexes a leading sample axis, so give it a
    batch (via ``DataLoader(transform=...)``), not a single image.
    Expects float images already scaled to [0, 1] -- it clips to that range."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __call__(self, x: Any) -> Any:
        from .image import image_augmentation
        return image_augmentation(np.asarray(x), **self.kwargs)


# ----------------------------------------------------------------- audio

class AugmentAudio(Transform):
    """The existing ``preprocessing.augment_audio`` as a transform."""

    def __init__(self, sr: int, **kwargs: Any) -> None:
        self.sr, self.kwargs = int(sr), kwargs

    def __call__(self, x: Any) -> Any:
        from .audio import augment_audio
        return augment_audio(np.asarray(x), self.sr, **self.kwargs)


class LoadAudio(Transform):
    """Path -> waveform. Returns just the samples; `with_rate=True` returns
    ``(samples, sample_rate)`` instead."""

    def __init__(self, with_rate: bool = False) -> None:
        self.with_rate = with_rate

    def __call__(self, path: Any) -> Any:
        from ..audio.audio_utils import load_wav
        audio, sr = load_wav(str(path))
        return (audio, sr) if self.with_rate else audio


class ToSpectrogram(Transform):
    """Waveform -> magnitude spectrogram ``(n_freq, n_frames)``."""

    def __init__(self, n_fft: int = 512, hop_length: int = 128,
                 window: str = "hann") -> None:
        self.n_fft, self.hop_length, self.window = n_fft, hop_length, window

    def __call__(self, audio: Any) -> Any:
        from ..audio.audio_utils import stft
        audio = np.asarray(audio)
        mag = np.abs(stft(audio, n_fft=self.n_fft, hop_length=self.hop_length,
                          window=self.window))
        # audio_utils computes in float64; cast back so a transform never
        # silently changes the pipeline's working precision.
        return mag.astype(audio.dtype) if audio.dtype.kind == "f" else mag


class ToMelSpectrogram(Transform):
    """Waveform -> mel spectrogram ``(n_mels, n_frames)``."""

    def __init__(self, sr: int, n_fft: int = 512, hop_length: int = 128,
                 n_mels: int = 40, window: str = "hann") -> None:
        self.sr, self.n_fft, self.hop_length = int(sr), n_fft, hop_length
        self.n_mels, self.window = n_mels, window

    def __call__(self, audio: Any) -> Any:
        from ..audio.audio_utils import stft, spectrogram_to_mel
        audio = np.asarray(audio)
        mag = np.abs(stft(audio, n_fft=self.n_fft, hop_length=self.hop_length,
                          window=self.window))
        mel = spectrogram_to_mel(mag, self.sr, n_mels=self.n_mels)
        return mel.astype(audio.dtype) if audio.dtype.kind == "f" else mel


class LogCompress(Transform):
    """``log(x + epsilon)`` -- the usual last step before a spectrogram
    reaches a model. `epsilon` keeps silent bins finite."""

    def __init__(self, epsilon: float = 1e-10) -> None:
        self.epsilon = float(epsilon)

    def __call__(self, x: Any) -> Any:
        x = np.asarray(x)
        return np.log(x + np.asarray(self.epsilon, dtype=x.dtype))


class TimeMask(Transform):
    """SpecAugment time masking: zero a random run of up to `max_width`
    consecutive frames. Operates on the LAST axis of a spectrogram."""

    def __init__(self, max_width: int = 10, n_masks: int = 1,
                 seed: Optional[int] = None) -> None:
        self.max_width, self.n_masks = int(max_width), int(n_masks)
        self._rng = _rng(seed)

    def __call__(self, spec: Any) -> Any:
        spec = np.asarray(spec).copy()
        frames = spec.shape[-1]
        for _ in range(self.n_masks):
            width = int(self._rng.randint(0, min(self.max_width, frames) + 1))
            if width == 0:
                continue
            start = int(self._rng.randint(0, frames - width + 1))
            spec[..., start:start + width] = 0.0
        return spec


class FreqMask(Transform):
    """SpecAugment frequency masking: zero a random run of up to
    `max_width` consecutive frequency bins (the second-to-last axis)."""

    def __init__(self, max_width: int = 8, n_masks: int = 1,
                 seed: Optional[int] = None) -> None:
        self.max_width, self.n_masks = int(max_width), int(n_masks)
        self._rng = _rng(seed)

    def __call__(self, spec: Any) -> Any:
        spec = np.asarray(spec).copy()
        bins = spec.shape[-2]
        for _ in range(self.n_masks):
            width = int(self._rng.randint(0, min(self.max_width, bins) + 1))
            if width == 0:
                continue
            start = int(self._rng.randint(0, bins - width + 1))
            spec[..., start:start + width, :] = 0.0
        return spec


# ------------------------------------------------------------------ text

class PadSequence(Transform):
    """The existing ``preprocessing.pad_sequences`` as a transform, for a
    list of variable-length sequences."""

    def __init__(self, max_length: Optional[int] = None, pad_value: Any = 0,
                 **kwargs: Any) -> None:
        self.max_length, self.pad_value, self.kwargs = max_length, pad_value, kwargs

    def __call__(self, sequences: Any) -> Any:
        from .text import pad_sequences
        return pad_sequences(sequences, max_length=self.max_length,
                             pad_value=self.pad_value, **self.kwargs)


class Tokenize(Transform):
    """Encode a string with a fitted ``text.Tokenizer``."""

    def __init__(self, tokenizer: Any, **kwargs: Any) -> None:
        self.tokenizer, self.kwargs = tokenizer, kwargs

    def __call__(self, text: Any) -> Any:
        return self.tokenizer.encode(text, **self.kwargs)
