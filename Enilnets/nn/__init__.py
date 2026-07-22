"""The manual (non-autograd) neural-network machinery behind ``NeuralNet``:
layer builders, forward/backward dispatch, activations, weight
initialization, the training loop, and model serialization. Phase 1's
``graph/`` autograd package sits *next to* this, never inside it."""

from .activations import activate, derivative
from .weight_init import init_weights, init_conv_weights, init_conv1d_weights, init_embedding_weights
from .train import LRScheduler, TrainBatch, Train, compute_accuracy
from .io import Save, Load
from .forward import Forward
from .backward import Backward
from .kvcache import KVCache, cached_forward_step

__all__ = [
    "activate", "derivative",
    "init_weights", "init_conv_weights", "init_conv1d_weights", "init_embedding_weights",
    "LRScheduler", "TrainBatch", "Train", "compute_accuracy",
    "Save", "Load", "Forward", "Backward",
    "KVCache", "cached_forward_step",
]
