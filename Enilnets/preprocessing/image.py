"""Image augmentation / normalization transforms (split out of
vision/image_utils.py so Phase 6's Compose pipeline has one place to pull
transforms from regardless of modality)."""
from typing import Any, Optional, Tuple

from ..core.backend import np

def _at(arr: Any, dtype: Any) -> Any:
    """Cast `arr` to `dtype`, or leave it alone when dtype is None."""
    return arr if dtype is None else arr.astype(dtype)


def image_augmentation(images: Any, flip_h: bool = True, flip_v: bool = False, rotate: float = 0,
                       brightness: float = 0.0, contrast: float = 0.0, noise_std: float = 0.0) -> Any:
    """Apply random augmentations to a batch of images.

    `images` is (N, C, H, W) or (N, H, W) for grayscale -- NCHW, this
    library's conv2d convention everywhere, NOT (N, H, W, C).

    `rotate` is a CAP, not an angle: each image is rotated by one of
    {0, 90, 180, 270} degrees, whichever of those are <= rotate. So
    rotate=45 never rotates at all; pass 90/180/270/360 to enable it."""
    aug = images.copy()
    N = aug.shape[0]
    # The random factors below are float64 by default, and float32 + float64
    # promotes the whole batch -- silently handing a float64 batch to a
    # float32 model. Draw them at the input's own precision instead.
    rand_dtype = aug.dtype if np.issubdtype(aug.dtype, np.floating) else None

    if flip_h:
        # Horizontal flip = mirror along the W axis: axis 3 for (N,C,H,W),
        # axis 2 for (N,H,W).
        mask = np.random.rand(N) < 0.5
        aug[mask] = aug[mask, :, :, ::-1] if aug.ndim == 4 else aug[mask, :, ::-1]

    if flip_v:
        # Vertical flip = mirror along the H axis: axis 2 for (N,C,H,W),
        # axis 1 for (N,H,W).
        mask = np.random.rand(N) < 0.5
        aug[mask] = aug[mask, :, ::-1, :] if aug.ndim == 4 else aug[mask, ::-1, :]

    if rotate > 0:
        angles = [0, 90, 180, 270]
        valid_angles = [a for a in angles if a <= rotate]
        # Spatial (H,W) plane is axes (1,2) of a single sample for (N,C,H,W)
        # (sample shape (C,H,W)), or (0,1) for (N,H,W) (sample shape (H,W)).
        sample_spatial_axes = (1, 2) if aug.ndim == 4 else (0, 1)
        for i in range(N):
            angle = int(np.random.choice(valid_angles, size=1)[0])
            if angle > 0:
                k = angle // 90
                aug[i] = np.rot90(aug[i], k=k, axes=sample_spatial_axes)

    if brightness > 0:
        factors = _at(np.random.uniform(1 - brightness, 1 + brightness,
                                        size=(N, 1, 1, 1) if aug.ndim == 4 else (N, 1, 1)),
                      rand_dtype)
        aug = aug * factors

    if contrast > 0:
        factors = _at(np.random.uniform(1 - contrast, 1 + contrast,
                                        size=(N, 1, 1, 1) if aug.ndim == 4 else (N, 1, 1)),
                      rand_dtype)
        # Per-sample (per-channel, when there is one) mean over the spatial
        # dims only: (2,3) for (N,C,H,W), (1,2) for (N,H,W).
        spatial_axes = (2, 3) if aug.ndim == 4 else (1, 2)
        means = np.mean(aug, axis=spatial_axes, keepdims=True)
        aug = means + (aug - means) * factors

    if noise_std > 0:
        aug = aug + _at(np.random.normal(0, noise_std, aug.shape), rand_dtype)

    return np.clip(aug, 0.0, 1.0)

def normalize_images(images: Any, mean: Optional[Any] = None, std: Optional[Any] = None) -> Tuple[Any, Any, Any]:
    """Normalize images to zero mean and unit variance."""
    if mean is None:
        mean = np.mean(images, axis=0)
    if std is None:
        std = np.std(images, axis=0) + 1e-8
    return (images - mean) / std, mean, std

def denormalize_images(images: Any, mean: Any, std: Any) -> Any:
    """Reverse normalization."""
    return images * std + mean
