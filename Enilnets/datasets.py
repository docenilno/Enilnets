"""Local-file dataset loaders for standard benchmark formats. Zero network
fetch of any kind -- you provide the files (downloaded separately from
their usual sources), these just parse the on-disk byte layout.
"""
import struct
import pickle
from .backend import np
from . import backend


def load_mnist(images_path, labels_path, normalize=False):
    """Parse standard MNIST/Fashion-MNIST IDX binary format files.

    Parameters
    ----------
    images_path : str
        Path to the IDX images file (e.g. "train-images-idx3-ubyte").
    labels_path : str
        Path to the IDX labels file (e.g. "train-labels-idx1-ubyte").
    normalize : bool
        If True, divide pixel values by 255.0 (scaling to [0, 1]). If you
        want z-score normalization instead, call
        `image_utils.normalize_images(X)` yourself after loading -- this
        loader only does the simple /255 scaling.

    Returns
    -------
    X : ndarray, float64, shape (N, 1, 28, 28)
    y : ndarray, int64, shape (N,)
    """
    with open(images_path, "rb") as f:
        magic, num_images, num_rows, num_cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected magic number {magic} in images file (expected 2051 for IDX images).")
        image_data = np.frombuffer(f.read(), dtype=np.uint8)

    with open(labels_path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected magic number {magic} in labels file (expected 2049 for IDX labels).")
        y = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)

    if num_images != num_labels:
        raise ValueError(f"images file has {num_images} images but labels file has {num_labels} labels.")

    X = image_data.reshape(num_images, 1, num_rows, num_cols).astype(backend.default_dtype())
    if normalize:
        X = X / 255.0
    return X, y


def load_cifar10(batch_paths, normalize=False):
    """Unpickle one or more standard CIFAR-10 batch files (e.g.
    "data_batch_1".."data_batch_5", "test_batch"), concatenating across all
    given paths.

    Parameters
    ----------
    batch_paths : str or list of str
        Path(s) to CIFAR-10 batch file(s).
    normalize : bool
        If True, divide pixel values by 255.0 (scaling to [0, 1]).

    Returns
    -------
    X : ndarray, float64, shape (N, 3, 32, 32)
        Channel-major (confirmed CIFAR-10's actual on-disk layout: each
        3072-byte row is 1024 red, then 1024 green, then 1024 blue bytes,
        each channel's 1024 bytes in row-major 32x32 order) -- NOT
        interleaved-per-pixel RGB, and NOT (N, 32, 32, 3).
    y : ndarray, int64, shape (N,)
    """
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
