"""Model compression (roadmap Phase 8): pruning and quantization.

Both operate on an already-trained model. Pruning zeroes or removes
weights; quantization reduces the precision they are stored at. Neither
needs autograd, except quantization-aware training, which uses ``graph/``'s
custom-op mechanism for its straight-through estimator.
"""

from .quantization import (quantize_weights, ActivationCalibrator,
                           quantize_tensor, fake_quantize, quantize, dequantize,
                           compute_scale, quant_range, quantization_error,
                           remove_activation_quantization, QUANTIZABLE)
from .pruning import (prune_magnitude, prune_channels, PruningSchedule,
                      sparsity, apply_masks, clear_masks, prunable_parameters,
                      channel_importance, PRUNABLE)

__all__ = [
    "prune_magnitude", "prune_channels", "PruningSchedule", "sparsity",
    "apply_masks", "clear_masks", "prunable_parameters", "channel_importance",
    "PRUNABLE",
    "quantize_weights", "ActivationCalibrator", "quantize_tensor",
    "fake_quantize", "quantize", "dequantize", "compute_scale", "quant_range",
    "quantization_error", "remove_activation_quantization", "QUANTIZABLE",
]
