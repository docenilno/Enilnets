"""Optimizer update rules and gradient handling for ``NeuralNet``.
Phase 5's optimizer/scheduler family (Lion, LAMB, NAdam, ...) lands here."""

from .optimizer import (update, compute_gradients, apply_gradients,
                        accumulate_gradients, apply_accumulated_gradients,
                        OPTIMIZERS)
from .averaging import EMA, SWA
from .lr_finder import find_learning_rate

__all__ = [
    "update", "compute_gradients", "apply_gradients",
    "accumulate_gradients", "apply_accumulated_gradients", "OPTIMIZERS",
    "EMA", "SWA", "find_learning_rate",
]
