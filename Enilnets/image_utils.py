#!/usr/bin/env python3
"""
Pure NumPy image utilities for Enilnets.
No external dependencies — reads/writes raw pixel arrays.
Supports PPM format (simplest pure-numpy image format) and raw binary.
"""
import numpy as np
import os
import struct

def load_ppm(path):
    """
    Load a PPM image (P6 binary format) using pure numpy.
    PPM is a simple format: header + raw RGB bytes.

    Parameters
    ----------
    path : str
        Path to .ppm file

    Returns
    -------
    ndarray : shape (H, W, 3), values in [0, 1]
    """
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

        raw = np.frombuffer(f.read(), dtype=np.uint8)
        img = raw.reshape((height, width, 3)).astype(np.float64) / 255.0
        return img

def save_ppm(arr, path):
    """
    Save a numpy array as PPM (P6 binary format).

    Parameters
    ----------
    arr : ndarray
        Shape (H, W, 3), values in [0, 1]
    path : str
    """
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    height, width = arr.shape[:2]

    with open(path, 'wb') as f:
        f.write(b'P6\n')
        f.write(f'{width} {height}\n'.encode())
        f.write(b'255\n')
        f.write(arr.tobytes())

def load_pgm(path):
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

        raw = np.frombuffer(f.read(), dtype=np.uint8)
        img = raw.reshape((height, width)).astype(np.float64) / 255.0
        return img

def save_pgm(arr, path):
    """Save a numpy array as PGM (P5 binary format)."""
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    height, width = arr.shape[:2]

    with open(path, 'wb') as f:
        f.write(b'P5\n')
        f.write(f'{width} {height}\n'.encode())
        f.write(b'255\n')
        f.write(arr.tobytes())

def load_raw_binary(path, shape, dtype=np.float64):
    """
    Load raw binary data as numpy array.

    Parameters
    ----------
    path : str
    shape : tuple
    dtype : numpy dtype

    Returns
    -------
    ndarray
    """
    arr = np.fromfile(path, dtype=dtype)
    return arr.reshape(shape)

def save_raw_binary(arr, path):
    """Save numpy array as raw binary."""
    arr.astype(np.float64).tofile(path)

def rgb_to_grayscale(rgb):
    """Convert RGB to grayscale using standard weights."""
    return 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]

def grayscale_to_rgb(gray):
    """Convert grayscale to RGB by replicating channels."""
    return np.stack([gray, gray, gray], axis=-1)

def resize_nearest_neighbor(img, new_height, new_width):
    """
    Resize image using nearest-neighbor interpolation (pure numpy).

    Parameters
    ----------
    img : ndarray
        (H, W) or (H, W, C)
    new_height : int
    new_width : int

    Returns
    -------
    ndarray
    """
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

def resize_bilinear(img, new_height, new_width):
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

def image_augmentation(images, flip_h=True, flip_v=False, rotate=0,
                       brightness=0.0, contrast=0.0, noise_std=0.0):
    """
    Apply random augmentations to a batch of images (pure numpy).
    """
    aug = images.copy()
    N = aug.shape[0]

    if flip_h:
        mask = np.random.rand(N) < 0.5
        aug[mask] = aug[mask, :, ::-1, ...] if aug.ndim == 4 else aug[mask, :, ::-1]

    if flip_v:
        mask = np.random.rand(N) < 0.5
        aug[mask] = aug[mask, ::-1, :, ...] if aug.ndim == 4 else aug[mask, ::-1, :]

    if rotate > 0:
        angles = [0, 90, 180, 270]
        valid_angles = [a for a in angles if a <= rotate]
        for i in range(N):
            angle = np.random.choice(valid_angles)
            if angle > 0:
                k = angle // 90
                aug[i] = np.rot90(aug[i], k=k, axes=(0, 1))

    if brightness > 0:
        factors = np.random.uniform(1 - brightness, 1 + brightness,
                                    size=(N, 1, 1, 1) if aug.ndim == 4 else (N, 1, 1))
        aug = aug * factors

    if contrast > 0:
        factors = np.random.uniform(1 - contrast, 1 + contrast,
                                    size=(N, 1, 1, 1) if aug.ndim == 4 else (N, 1, 1))
        means = np.mean(aug, axis=(1, 2), keepdims=True)
        aug = means + (aug - means) * factors

    if noise_std > 0:
        aug = aug + np.random.normal(0, noise_std, aug.shape)

    return np.clip(aug, 0.0, 1.0)

def normalize_images(images, mean=None, std=None):
    """Normalize images to zero mean and unit variance."""
    if mean is None:
        mean = np.mean(images, axis=0)
    if std is None:
        std = np.std(images, axis=0) + 1e-8
    return (images - mean) / std, mean, std

def denormalize_images(images, mean, std):
    """Reverse normalization."""
    return images * std + mean

def images_to_patches(images, patch_size, stride=None):
    """Extract patches from images (fully vectorized via sliding_window_view)."""
    if stride is None:
        stride = patch_size
    N, H, W, C = images.shape
    # (N, n_y, n_x, C, patch_size, patch_size)
    windows = np.lib.stride_tricks.sliding_window_view(images, (patch_size, patch_size), axis=(1, 2))
    windows = windows[:, ::stride, ::stride]
    n_y, n_x = windows.shape[1], windows.shape[2]
    patches = windows.transpose(0, 1, 2, 4, 5, 3).reshape(N * n_y * n_x, patch_size, patch_size, C)
    return np.ascontiguousarray(patches, dtype=np.float64)

def pad_image(img, pad_h, pad_w, mode='constant', constant_value=0):
    """
    Pad image with zeros or edge values.

    Parameters
    ----------
    img : ndarray
        (H, W) or (H, W, C)
    pad_h : int
        Padding on each side (top/bottom)
    pad_w : int
        Padding on each side (left/right)
    mode : str
        'constant' or 'edge'
    constant_value : float
        Value for constant padding

    Returns
    -------
    ndarray
    """
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
