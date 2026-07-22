"""Differentiable audio front ends (roadmap item 60): STFT, spectrogram and
mel spectrogram as graph ops, so a model can backprop through them.

Built entirely from ops that already exist -- framing is a gather, the DFT
a matmul against a precomputed complex matrix, the magnitude the complex
``absolute`` -- so there are no new gradient rules here, and complex-tensor
support (item 26) is what makes the DFT expressible.

An FFT would be O(N log N) where a DFT matmul is O(N^2) per frame; for the
frame sizes this library's models use, that buys a differentiable path with
no bespoke backward."""

import math
from typing import Any, Optional

from ..core.backend import np
from ..core import backend
from .tensor import Tensor, as_tensor
from . import ops


def window_function(name: str, length: int) -> Any:
    """A plain (non-differentiable) analysis window. Constant w.r.t. the
    signal, so it needs no gradient of its own."""
    if name == "rectangular":
        return np.ones(length, dtype=backend.default_dtype())
    n = np.arange(length, dtype=backend.default_dtype())
    if name == "hann":
        return 0.5 - 0.5 * np.cos(2.0 * math.pi * n / length)
    if name == "hamming":
        return 0.54 - 0.46 * np.cos(2.0 * math.pi * n / length)
    raise ValueError(
        f"Unknown window {name!r}; expected 'hann', 'hamming' or 'rectangular'")


def dft_matrix(n_fft: int) -> Any:
    """The one-sided DFT matrix, ``(n_fft, n_fft // 2 + 1)`` complex.
    Multiplying a frame by this is exactly its rfft."""
    n = np.arange(n_fft).reshape(-1, 1)
    k = np.arange(n_fft // 2 + 1).reshape(1, -1)
    angle = -2.0 * math.pi * n * k / n_fft
    return (np.cos(angle) + 1j * np.sin(angle)).astype(
        np.complex128 if backend.is_float64_enabled() else np.complex64)


def frame_indices(length: int, n_fft: int, hop_length: int) -> Any:
    """``(n_frames, n_fft)`` integer index array for the framing gather."""
    if length < n_fft:
        raise ValueError(
            f"signal of length {length} is shorter than n_fft={n_fft}; pad it "
            "first or use a smaller window")
    starts = np.arange(0, length - n_fft + 1, hop_length).reshape(-1, 1)
    return starts + np.arange(n_fft).reshape(1, -1)


def stft(x: Any, n_fft: int = 512, hop_length: Optional[int] = None,
         window: str = "hann") -> Tensor:
    """Differentiable short-time Fourier transform of a 1-D signal.

    Returns a complex Tensor ``(n_frames, n_fft // 2 + 1)``. `hop_length`
    defaults to ``n_fft // 4``."""
    x = as_tensor(x)
    if len(x.shape) != 1:
        raise ValueError(f"stft expects a 1-D signal, got shape {tuple(x.shape)}")
    hop_length = hop_length or n_fft // 4
    # The window, index and DFT matrix all come from the active backend, so
    # they live wherever the signal does -- the same contract every other
    # graph op follows (a host array in GPU mode fails everywhere, not just
    # here).
    idx = frame_indices(int(x.shape[0]), n_fft, hop_length)
    frames = x[idx]                                   # differentiable gather
    win = Tensor(window_function(window, n_fft))
    return ops.matmul(ops.mul(frames, win), Tensor(dft_matrix(n_fft)))


def spectrogram(x: Any, n_fft: int = 512, hop_length: Optional[int] = None,
                window: str = "hann", power: float = 1.0) -> Tensor:
    """Magnitude spectrogram, ``(n_frames, n_fft // 2 + 1)``. `power` is 1
    for magnitude and 2 for energy."""
    mag = ops.absolute(stft(x, n_fft, hop_length, window))
    if power == 1.0:
        return mag
    return ops.power(mag, exponent=power)


def mel_filterbank(sr: int, n_freq: int, n_mels: int = 40, fmin: float = 0.0,
                   fmax: Optional[float] = None) -> Any:
    """Triangular mel filters, ``(n_mels, n_freq)``. Constant w.r.t. the
    signal, so like the window it needs no gradient."""
    fmax = fmax if fmax is not None else sr / 2.0

    def to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def from_mel(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    edges = from_mel(np.linspace(to_mel(fmin), to_mel(fmax), n_mels + 2))
    bins = np.floor((n_freq - 1) * 2 * edges / sr).astype(np.int64)
    fb = np.zeros((n_mels, n_freq), dtype=backend.default_dtype())
    for m in range(n_mels):
        left, centre, right = int(bins[m]), int(bins[m + 1]), int(bins[m + 2])
        # A filter can collapse to zero width when n_mels is large relative
        # to n_freq; leaving it empty is correct and better than dividing by
        # zero to produce NaNs downstream.
        for k in range(left, min(centre, n_freq)):
            if centre > left:
                fb[m, k] = (k - left) / (centre - left)
        for k in range(centre, min(right, n_freq)):
            if right > centre:
                fb[m, k] = (right - k) / (right - centre)
    return fb


def mel_spectrogram(x: Any, sr: int, n_fft: int = 512,
                    hop_length: Optional[int] = None, n_mels: int = 40,
                    window: str = "hann", fmin: float = 0.0,
                    fmax: Optional[float] = None, power: float = 2.0) -> Tensor:
    """Differentiable mel spectrogram, ``(n_frames, n_mels)``."""
    spec = spectrogram(x, n_fft, hop_length, window, power=power)
    fb = mel_filterbank(sr, int(spec.shape[1]), n_mels, fmin, fmax)
    return ops.matmul(spec, Tensor(np.asarray(fb).T))


def log_mel_spectrogram(x: Any, sr: int, epsilon: float = 1e-10,
                        **kwargs: Any) -> Tensor:
    """``log(mel_spectrogram + epsilon)`` -- the usual model input. `epsilon`
    keeps the log finite where a mel band is exactly zero, which happens
    whenever a filter is empty or the signal is silent."""
    mel = mel_spectrogram(x, sr, **kwargs)
    return ops.log(ops.add(mel, Tensor(np.asarray(epsilon, dtype=mel.dtype))))
