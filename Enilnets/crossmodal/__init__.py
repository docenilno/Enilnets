"""Cross-modal (image-text/audio-text) utilities: contrastive loss,
CLIP-style normalization, multimodal fusion."""

from .crossmodal_utils import (
    contrastive_loss, clip_normalize, multimodal_fusion,
    create_text_conditioned_image,
)

__all__ = ["contrastive_loss", "clip_normalize", "multimodal_fusion",
           "create_text_conditioned_image"]
