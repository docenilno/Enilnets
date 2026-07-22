#!/usr/bin/env python3
"""
Pure NumPy image utilities for Enilnets.
No external dependencies — reads/writes raw pixel arrays.
Supports PPM format (simplest pure-numpy image format) and raw binary.
"""
from typing import Any, Optional, Tuple

import numpy as _host_np
from ..core.backend import np
from ..core import backend

def load_ppm(path: str) -> Any:
    """Load a binary (P6) PPM image. Returns (H, W, 3) with values in [0, 1]."""
    with open(path, 'rb') as f:
        header = f.readline().strip()
        if header != b'P6':
            raise ValueError(f"Only P6 (binary) PPM supported, got: {header}")

        # Skip comments
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()

        width, height = map(int, line.split())
        maxval = int(f.readline().strip())

        if maxval != 255:
            raise ValueError(f"Only maxval=255 supported, got {maxval}")

        # CuPy has no frombuffer (no host-buffer-backed array constructor) --
        # this is inherently host-side byte parsing, so always use plain
        # NumPy here and only move the result onto the active backend
        # (np.asarray, below) once it's a real array.
        raw = _host_np.frombuffer(f.read(), dtype=_host_np.uint8)
        img = np.asarray(raw.reshape((height, width, 3)), dtype=backend.default_dtype()) / 255.0
        return img

def save_ppm(arr: Any, path: str) -> None:
    """Write an (H, W, 3) array with values in [0, 1] as a binary (P6) PPM."""
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    height, width = arr.shape[:2]

    with open(path, 'wb') as f:
        f.write(b'P6\n')
        f.write(f'{width} {height}\n'.encode())
        f.write(b'255\n')
        f.write(arr.tobytes())

def load_pgm(path: str) -> Any:
    """Load a PGM grayscale image (P5 binary format)."""
    with open(path, 'rb') as f:
        header = f.readline().strip()
        if header != b'P5':
            raise ValueError(f"Only P5 (binary) PGM supported, got: {header}")

        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()

        width, height = map(int, line.split())
        maxval = int(f.readline().strip())

        raw = _host_np.frombuffer(f.read(), dtype=_host_np.uint8)
        img = np.asarray(raw.reshape((height, width)), dtype=backend.default_dtype()) / 255.0
        return img

def save_pgm(arr: Any, path: str) -> None:
    """Save a numpy array as PGM (P5 binary format)."""
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    height, width = arr.shape[:2]

    with open(path, 'wb') as f:
        f.write(b'P5\n')
        f.write(f'{width} {height}\n'.encode())
        f.write(b'255\n')
        f.write(arr.tobytes())

def load_raw_binary(path: str, shape: Tuple[int, ...], dtype: Optional[Any] = None) -> Any:
    """Load raw binary data into an array of `shape`. `dtype` defaults to the
    current working precision (float32 unless use_float64(True))."""
    if dtype is None:
        dtype = backend.default_dtype()
    arr = np.fromfile(path, dtype=dtype)
    return arr.reshape(shape)

def save_raw_binary(arr: Any, path: str) -> None:
    """Save numpy array as raw binary."""
    arr.astype(backend.default_dtype()).tofile(path)

def rgb_to_grayscale(rgb: Any) -> Any:
    """Convert RGB to grayscale using standard weights."""
    return 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]

def grayscale_to_rgb(gray: Any) -> Any:
    """Convert grayscale to RGB by replicating channels."""
    return np.stack([gray, gray, gray], axis=-1)

def resize_nearest_neighbor(img: Any, new_height: int, new_width: int) -> Any:
    """Resize an (H, W) or (H, W, C) image by nearest-neighbor interpolation."""
    h, w = img.shape[:2]
    row_scale = h / new_height
    col_scale = w / new_width

    row_idx = (np.arange(new_height) * row_scale).astype(int)
    col_idx = (np.arange(new_width) * col_scale).astype(int)
    row_idx = np.clip(row_idx, 0, h - 1)
    col_idx = np.clip(col_idx, 0, w - 1)

    if img.ndim == 2:
        return img[row_idx[:, None], col_idx[None, :]]
    else:
        return img[row_idx[:, None], col_idx[None, :], :]

def resize_bilinear(img: Any, new_height: int, new_width: int) -> Any:
    """
    Resize image using bilinear interpolation (pure numpy, fully vectorized).
    """
    h, w = img.shape[:2]

    # Create coordinate grids
    row_coords = np.linspace(0, h - 1, new_height)
    col_coords = np.linspace(0, w - 1, new_width)

    row_floor = np.floor(row_coords).astype(int)
    col_floor = np.floor(col_coords).astype(int)
    row_ceil = np.minimum(row_floor + 1, h - 1)
    col_ceil = np.minimum(col_floor + 1, w - 1)

    row_frac = row_coords - row_floor
    col_frac = col_coords - col_floor

    if img.ndim == 2:
        fy = row_frac[:, None]
        fx = col_frac[None, :]
        top = img[row_floor[:, None], col_floor[None, :]] * (1 - fx) + img[row_floor[:, None], col_ceil[None, :]] * fx
        bot = img[row_ceil[:, None], col_floor[None, :]] * (1 - fx) + img[row_ceil[:, None], col_ceil[None, :]] * fx
        return top * (1 - fy) + bot * fy
    else:
        fy = row_frac[:, None, None]
        fx = col_frac[None, :, None]
        top = img[row_floor[:, None], col_floor[None, :], :] * (1 - fx) + img[row_floor[:, None], col_ceil[None, :], :] * fx
        bot = img[row_ceil[:, None], col_floor[None, :], :] * (1 - fx) + img[row_ceil[:, None], col_ceil[None, :], :] * fx
        return top * (1 - fy) + bot * fy

# Augmentation/normalization transforms live in Enilnets.preprocessing now;
# re-exported here for backward compatibility.
from ..preprocessing.image import image_augmentation, normalize_images, denormalize_images

def images_to_patches(images: Any, patch_size: int, stride: Optional[int] = None) -> Any:
    """Cut a batch of images into patches, returning
    (N * n_y * n_x, C, patch_size, patch_size).

    `images` is (N, C, H, W) -- NCHW, this library's conv2d convention
    everywhere, NOT (N, H, W, C)."""
    if stride is None:
        stride = patch_size
    N, C, H, W = images.shape
    # (N, C, n_y, n_x, patch_size, patch_size)
    windows = np.lib.stride_tricks.sliding_window_view(images, (patch_size, patch_size), axis=(2, 3))
    windows = windows[:, :, ::stride, ::stride]
    n_y, n_x = windows.shape[2], windows.shape[3]
    patches = windows.transpose(0, 2, 3, 1, 4, 5).reshape(N * n_y * n_x, C, patch_size, patch_size)
    return np.ascontiguousarray(patches, dtype=backend.default_dtype())

def pad_image(img: Any, pad_h: int, pad_w: int, mode: str = 'constant', constant_value: float = 0) -> Any:
    """Pad an (H, W) or (H, W, C) image by `pad_h` rows and `pad_w` columns on
    each side. `mode` is 'constant' (using `constant_value`) or 'edge'."""
    if img.ndim == 2:
        h, w = img.shape
        if mode == 'constant':
            return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=constant_value)
        else:
            return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    else:
        h, w, c = img.shape
        if mode == 'constant':
            return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant', constant_values=constant_value)
        else:
            return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')
