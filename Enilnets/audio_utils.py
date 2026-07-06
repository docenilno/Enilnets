#!/usr/bin/env python3
"""
Pure NumPy audio utilities for Enilnets.
No external dependencies — reads/writes raw WAV and computes spectrograms.
"""
from .backend import np
from . import backend
import os
import struct

def load_wav(path):
    """
    Load a WAV file using pure numpy (PCM 16-bit or 32-bit float).

    Parameters
    ----------
    path : str

    Returns
    -------
    audio : ndarray
        Audio samples, shape (samples,) or (samples, channels)
    sr : int
        Sample rate
    """
    with open(path, 'rb') as f:
        # Read RIFF header
        riff = f.read(4)
        if riff != b'RIFF':
            raise ValueError(f"Not a WAV file: {path}")

        f.read(4)  # file size
        wave = f.read(4)
        if wave != b'WAVE':
            raise ValueError(f"Not a WAV file: {path}")

        # Read chunks
        fmt_chunk = None
        data_chunk = None

        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack('<I', f.read(4))[0]

            if chunk_id == b'fmt ':
                fmt_chunk = f.read(chunk_size)
            elif chunk_id == b'data':
                data_chunk = f.read(chunk_size)
            else:
                f.seek(chunk_size, 1)  # skip unknown chunk

        if fmt_chunk is None or data_chunk is None:
            raise ValueError("Invalid WAV file: missing fmt or data chunk")

        # Parse fmt chunk
        audio_format = struct.unpack('<H', fmt_chunk[0:2])[0]
        num_channels = struct.unpack('<H', fmt_chunk[2:4])[0]
        sample_rate = struct.unpack('<I', fmt_chunk[4:8])[0]
        bits_per_sample = struct.unpack('<H', fmt_chunk[14:16])[0]

        # Parse data
        if audio_format == 1:  # PCM
            if bits_per_sample == 16:
                dtype = np.int16
                max_val = 32768.0
            elif bits_per_sample == 24:
                # 24-bit: read as bytes and convert
                n_samples = len(data_chunk) // (3 * num_channels)
                raw = np.frombuffer(data_chunk, dtype=np.uint8)
                # Reshape to (n_samples, num_channels, 3)
                raw = raw.reshape(n_samples, num_channels, 3)
                # Convert 3 bytes to signed 24-bit int
                samples = raw[:, :, 0].astype(np.int32) | (raw[:, :, 1].astype(np.int32) << 8) | (raw[:, :, 2].astype(np.int32) << 16)
                # Sign extend
                samples = np.where(samples > 8388607, samples - 16777216, samples)
                audio = samples.astype(backend.default_dtype()) / 8388608.0
                if num_channels == 1:
                    audio = audio.squeeze(-1)
                return audio, sample_rate
            elif bits_per_sample == 32:
                dtype = np.int32
                max_val = 2147483648.0
            else:
                raise ValueError(f"Unsupported bits per sample: {bits_per_sample}")

            samples = np.frombuffer(data_chunk, dtype=dtype)
            audio = samples.astype(backend.default_dtype()) / max_val

            if num_channels > 1:
                audio = audio.reshape(-1, num_channels)

            return audio, sample_rate

        elif audio_format == 3:  # IEEE float
            if bits_per_sample == 32:
                samples = np.frombuffer(data_chunk, dtype=np.float32)
                audio = samples.astype(backend.default_dtype())
                if num_channels > 1:
                    audio = audio.reshape(-1, num_channels)
                return audio, sample_rate
            else:
                raise ValueError(f"Unsupported float bits: {bits_per_sample}")
        else:
            raise ValueError(f"Unsupported audio format: {audio_format}")

def save_wav(audio, path, sr, bits_per_sample=16):
    """
    Save audio as WAV file using pure numpy.

    Parameters
    ----------
    audio : ndarray
        Shape (samples,) or (samples, channels)
    path : str
    sr : int
        Sample rate
    bits_per_sample : int
        16 or 32
    """
    audio = np.asarray(audio, dtype=backend.default_dtype())

    if audio.ndim == 1:
        num_channels = 1
        audio = audio.reshape(-1, 1)
    else:
        num_channels = audio.shape[1]

    # Normalize to prevent clipping
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val

    if bits_per_sample == 16:
        samples = (audio * 32767).astype(np.int16)
        byte_rate = sr * num_channels * 2
        block_align = num_channels * 2
    elif bits_per_sample == 32:
        samples = (audio * 2147483647).astype(np.int32)
        byte_rate = sr * num_channels * 4
        block_align = num_channels * 4
    else:
        raise ValueError(f"Unsupported bits per sample: {bits_per_sample}")

    data = samples.tobytes()
    data_size = len(data)

    with open(path, 'wb') as f:
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))  # subchunk size
        f.write(struct.pack('<H', 1))   # audio format (PCM)
        f.write(struct.pack('<H', num_channels))
        f.write(struct.pack('<I', sr))
        f.write(struct.pack('<I', byte_rate))
        f.write(struct.pack('<H', block_align))
        f.write(struct.pack('<H', bits_per_sample))
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        f.write(data)

def stft(audio, n_fft=2048, hop_length=512, window='hann'):
    """
    Short-Time Fourier Transform (pure numpy).

    Parameters
    ----------
    audio : ndarray
        1D audio signal
    n_fft : int
        FFT window size
    hop_length : int
        Step between frames
    window : str
        'hann', 'hamming', or 'rectangular'

    Returns
    -------
    spectrogram : ndarray
        Complex spectrogram, shape (n_fft//2+1, n_frames)
    """
    if window == 'hann':
        win = np.hanning(n_fft)
    elif window == 'hamming':
        win = np.hamming(n_fft)
    else:
        win = np.ones(n_fft)

    n_samples = len(audio)
    n_frames = 1 + (n_samples - n_fft) // hop_length

    # Extract all frames as one strided view (no copy) and batch the FFT
    # instead of looping n_frames times with one np.fft.rfft call each.
    frame_starts = np.arange(n_frames) * hop_length
    frames = np.lib.stride_tricks.sliding_window_view(audio, n_fft)[frame_starts] * win
    spec = np.fft.rfft(frames, axis=-1).T  # (n_fft//2+1, n_frames)

    return spec

def istft(spectrogram, n_fft=2048, hop_length=512, window='hann', length=None):
    """
    Inverse STFT (pure numpy).

    Parameters
    ----------
    spectrogram : ndarray
        Complex spectrogram
    n_fft : int
    hop_length : int
    window : str
    length : int or None
        Expected output length

    Returns
    -------
    audio : ndarray
    """
    if window == 'hann':
        win = np.hanning(n_fft)
    elif window == 'hamming':
        win = np.hamming(n_fft)
    else:
        win = np.ones(n_fft)

    n_frames = spectrogram.shape[1]
    expected_length = hop_length * (n_frames - 1) + n_fft

    audio = np.zeros(expected_length, dtype=backend.default_dtype())
    window_sum = np.zeros(expected_length, dtype=backend.default_dtype())

    for i in range(n_frames):
        frame = np.fft.irfft(spectrogram[:, i], n=n_fft) * win
        start = i * hop_length
        audio[start:start + n_fft] += frame
        window_sum[start:start + n_fft] += win ** 2

    # Normalize by window overlap
    audio = audio / np.maximum(window_sum, 1e-8)

    if length is not None:
        audio = audio[:length]

    return audio

def _mel_filterbank(sr, n_freq, n_mels, fmin=0, fmax=None):
    """Build a triangular mel filterbank, shape (n_mels, n_freq). Fully
    vectorized (broadcasting) instead of a per-band/per-bin Python loop."""
    if fmax is None:
        fmax = sr / 2
    mel_min = 2595 * np.log10(1 + fmin / 700)
    mel_max = 2595 * np.log10(1 + fmax / 700)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    freq_points = 700 * (10 ** (mel_points / 2595) - 1)
    fft_freqs = np.linspace(0, sr / 2, n_freq)

    f_left = freq_points[:-2][:, None]
    f_center = freq_points[1:-1][:, None]
    f_right = freq_points[2:][:, None]
    f = fft_freqs[None, :]

    rising = (f - f_left) / (f_center - f_left)
    falling = (f_right - f) / (f_right - f_center)
    return np.maximum(0.0, np.minimum(rising, falling))

def spectrogram_to_mel(spectrogram, sr, n_mels=128, fmin=0, fmax=None):
    """
    Convert linear spectrogram to mel spectrogram (pure numpy).

    Parameters
    ----------
    spectrogram : ndarray
        Linear magnitude spectrogram (n_freq, n_frames)
    sr : int
        Sample rate
    n_mels : int
        Number of mel bands
    fmin : float
        Minimum frequency
    fmax : float or None
        Maximum frequency (default = sr/2)

    Returns
    -------
    mel_spec : ndarray
        Mel spectrogram (n_mels, n_frames)
    """
    n_freq = spectrogram.shape[0]
    mel_filterbank = _mel_filterbank(sr, n_freq, n_mels, fmin, fmax)
    mel_spec = np.dot(mel_filterbank, spectrogram)
    return mel_spec

def mel_to_spectrogram(mel_spec, sr, n_freq, fmin=0, fmax=None):
    """
    Approximate inverse: mel spectrogram to linear spectrogram.
    This is a pseudo-inverse (not exact reconstruction).
    """
    n_mels = mel_spec.shape[0]
    mel_filterbank = _mel_filterbank(sr, n_freq, n_mels, fmin, fmax)

    # Pseudo-inverse
    mel_filterbank_pinv = np.linalg.pinv(mel_filterbank.T).T
    linear_spec = np.dot(mel_filterbank_pinv, mel_spec)
    return np.maximum(linear_spec, 0)  # Ensure non-negative

def audio_to_spectrogram(audio, sr, n_fft=2048, hop_length=512, n_mels=128):
    """
    Full pipeline: audio -> mel spectrogram.
    """
    spec = stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag_spec = np.abs(spec)
    mel_spec = spectrogram_to_mel(mag_spec, sr, n_mels=n_mels)
    # Log scale
    log_mel = np.log(np.maximum(mel_spec, 1e-10))
    return log_mel

def spectrogram_to_audio(mel_spec, sr, n_fft=2048, hop_length=512, n_iter=32):
    """
    Full pipeline: mel spectrogram -> audio (approximate via Griffin-Lim).
    """
    # Inverse log
    mel_spec = np.exp(mel_spec)
    # Mel to linear
    n_freq = n_fft // 2 + 1
    linear_mag = mel_to_spectrogram(mel_spec, sr, n_freq)
    # Griffin-Lim phase reconstruction
    angles = np.exp(2j * np.pi * np.random.rand(*linear_mag.shape))
    for _ in range(n_iter):
        complex_spec = linear_mag * angles
        audio = istft(complex_spec, n_fft=n_fft, hop_length=hop_length)
        spec = stft(audio, n_fft=n_fft, hop_length=hop_length)
        angles = np.exp(1j * np.angle(spec))

    complex_spec = linear_mag * angles
    audio = istft(complex_spec, n_fft=n_fft, hop_length=hop_length)
    return audio

def audio_to_frames(audio, frame_length, hop_length=None):
    """Split audio into overlapping frames."""
    if hop_length is None:
        hop_length = frame_length // 2

    n_samples = len(audio)
    n_frames = 1 + (n_samples - frame_length) // hop_length
    frames = np.zeros((n_frames, frame_length), dtype=backend.default_dtype())

    for i in range(n_frames):
        start = i * hop_length
        frames[i] = audio[start:start + frame_length]

    return frames

def frames_to_audio(frames, hop_length, window='hann'):
    """Reconstruct audio from overlapping frames (OLA)."""
    n_frames, frame_length = frames.shape
    output_length = (n_frames - 1) * hop_length + frame_length
    audio = np.zeros(output_length, dtype=backend.default_dtype())
    window_sum = np.zeros(output_length, dtype=backend.default_dtype())

    if window == 'hann':
        win = np.hanning(frame_length)
    elif window == 'hamming':
        win = np.hamming(frame_length)
    else:
        win = np.ones(frame_length)

    for i in range(n_frames):
        start = i * hop_length
        audio[start:start + frame_length] += frames[i] * win
        window_sum[start:start + frame_length] += win

    audio = audio / np.maximum(window_sum, 1e-8)
    return audio

def augment_audio(audio, sr, pitch_shift=0, time_stretch=1.0, noise_std=0.0):
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
        aug = aug + np.random.normal(0, noise_std, aug.shape)

    return aug
