"""Audio utilities: WAV I/O, STFT/mel spectrograms, framing, augmentation.
Phase 7's differentiable spectrogram layers land here."""

from .audio_utils import (
    load_wav, save_wav, stft, istft, spectrogram_to_mel, mel_to_spectrogram,
    audio_to_spectrogram, spectrogram_to_audio, audio_to_frames,
    frames_to_audio, augment_audio,
)

__all__ = ["load_wav", "save_wav", "stft", "istft", "spectrogram_to_mel",
           "mel_to_spectrogram", "audio_to_spectrogram", "spectrogram_to_audio",
           "audio_to_frames", "frames_to_audio", "augment_audio"]
