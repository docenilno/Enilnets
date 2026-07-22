"""Local-file dataset loaders (MNIST, CIFAR-10) plus the Phase 6 data
pipeline: the Dataset abstraction and the DataLoader that batches it."""

from .loaders import load_mnist, load_cifar10
from .dataset import (Dataset, IterableDataset, ArrayDataset, MemmapDataset,
                      StreamingDataset, Subset, ConcatDataset, random_split,
                      as_dataset)
from .loader import DataLoader, default_collate

__all__ = [
    "load_mnist", "load_cifar10",
    "Dataset", "IterableDataset", "ArrayDataset", "MemmapDataset",
    "StreamingDataset", "Subset", "ConcatDataset", "random_split", "as_dataset",
    "DataLoader", "default_collate",
]
