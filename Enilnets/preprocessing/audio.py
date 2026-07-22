"""Audio augmentation transforms (split out of audio/audio_utils.py)."""
from typing import Any

from ..core.backend import np
from ..core import backend

def augment_audio(audio: Any, sr: int, pitch_shift: float = 0, time_stretch: float = 1.0,
                   noise_std: float = 0.0) -> Any:
    """
    Audio augmentation (pure numpy).

    Note: pitch_shift and time_stretch are approximate.
    For exact results, use external libraries.
    """
    aug = audio.copy()

    if pitch_shift != 0:
        # Approximate pitch shift via resampling (crude but numpy-only)
        # This is a rough approximation
        factor = 2 ** (pitch_shift / 12)
        n = int(len(aug) / factor)
        indices = np.linspace(0, len(aug) - 1, n)
        indices_int = indices.astype(int)
        frac = indices - indices_int
        aug = aug[indices_int] * (1 - frac) + aug[np.minimum(indices_int + 1, len(aug) - 1)] * frac
        aug = aug.astype(backend.default_dtype())

    if time_stretch != 1.0:
        # Linear interpolation for time stretching
        n = int(len(aug) / time_stretch)
        indices = np.linspace(0, len(aug) - 1, n)
        indices_int = indices.astype(int)
        frac = indices - indices_int
        aug = aug[indices_int] * (1 - frac) + aug[np.minimum(indices_int + 1, len(aug) - 1)] * frac
        aug = aug.astype(backend.default_dtype())

    if noise_std > 0:
        # np.random.normal is float64; adding it straight to a float32 signal
        # promotes the whole thing, which then reaches a float32 model as
        # float64. Same fix as image_augmentation's.
        noise = np.random.normal(0, noise_std, aug.shape)
        aug = aug + (noise.astype(aug.dtype)
                     if np.issubdtype(aug.dtype, np.floating) else noise)

    return aug
