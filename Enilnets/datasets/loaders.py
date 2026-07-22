"""Local-file dataset loaders for standard benchmark formats. Zero network
fetch of any kind -- you provide the files (downloaded separately from
their usual sources), these just parse the on-disk byte layout.
"""
from typing import Any, List, Tuple, Union

import struct
import pickle
import numpy as _host_np
from ..core.backend import np
from ..core import backend


def load_mnist(images_path: str, labels_path: str, normalize: bool = False) -> Tuple[Any, Any]:
    """Parse MNIST/Fashion-MNIST IDX binary files. Returns (X, y): X is
    (N, 1, 28, 28) at the backend's default dtype, y is (N,) int64.

    `normalize` only does the /255 scaling to [0, 1]; for z-scoring, call
    `image_utils.normalize_images(X)` afterwards."""
    with open(images_path, "rb") as f:
        magic, num_images, num_rows, num_cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected magic number {magic} in images file (expected 2051 for IDX images).")
        # CuPy has no frombuffer (no host-buffer-backed array constructor) --
        # this is inherently host-side byte parsing, so always use plain
        # NumPy here and only move the result onto the active backend
        # (np.asarray, below) once it's a real array.
        image_data = _host_np.frombuffer(f.read(), dtype=_host_np.uint8)

    with open(labels_path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected magic number {magic} in labels file (expected 2049 for IDX labels).")
        y = np.asarray(_host_np.frombuffer(f.read(), dtype=_host_np.uint8), dtype=np.int64)

    if num_images != num_labels:
        raise ValueError(f"images file has {num_images} images but labels file has {num_labels} labels.")

    X = np.asarray(image_data.reshape(num_images, 1, num_rows, num_cols), dtype=backend.default_dtype())
    if normalize:
        X = X / 255.0
    return X, y


def load_cifar10(batch_paths: Union[str, List[str]], normalize: bool = False) -> Tuple[Any, Any]:
    """Unpickle one or more CIFAR-10 batch files, concatenated across paths.
    Returns (X, y): X is (N, 3, 32, 32) at the backend's default dtype, y is
    (N,) int64. `normalize` scales pixels to [0, 1].

    X is channel-major, matching CIFAR-10's on-disk layout (1024 red, then
    green, then blue, each row-major) -- not interleaved RGB, not NHWC."""
    if isinstance(batch_paths, str):
        batch_paths = [batch_paths]
    X_list, y_list = [], []
    for path in batch_paths:
        with open(path, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        X_list.append(np.asarray(batch[b"data"], dtype=np.uint8))
        y_list.append(np.asarray(batch[b"labels"], dtype=np.int64))

    X = np.concatenate(X_list, axis=0).astype(backend.default_dtype()).reshape(-1, 3, 32, 32)
    y = np.concatenate(y_list, axis=0)
    if normalize:
        X = X / 255.0
    return X, y
