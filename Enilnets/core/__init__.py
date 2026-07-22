"""Core of the library: the ``NeuralNet`` skeleton, the NumPy/CuPy backend
switch, shared numeric constants, and general-purpose utilities — the
absolute minimum every other subpackage depends on."""

from .base import NeuralNet
from .backend import (
    np, use_gpu, gpu_available, is_gpu_enabled,
    use_float64, is_float64_enabled, default_dtype,
)
from .utils import (
    set_seed, train_test_split, iterate_minibatches, count_parameters,
    EarlyStopping, one_hot, k_fold_split,
    ModelCheckpoint, CSVLogger, JSONLogger,
)
from . import constants

__all__ = [
    "NeuralNet",
    "np", "use_gpu", "gpu_available", "is_gpu_enabled",
    "use_float64", "is_float64_enabled", "default_dtype",
    "set_seed", "train_test_split", "iterate_minibatches", "count_parameters",
    "EarlyStopping", "one_hot", "k_fold_split",
    "ModelCheckpoint", "CSVLogger", "JSONLogger",
    "constants",
]
