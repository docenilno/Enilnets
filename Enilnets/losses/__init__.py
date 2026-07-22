"""Loss functions shared across model families. Generative-model-specific
losses live in ``generative/generative_loss.py`` next to their models."""

from .loss import ComputeLoss

__all__ = ["ComputeLoss"]
