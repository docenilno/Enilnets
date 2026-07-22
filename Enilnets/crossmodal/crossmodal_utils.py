#!/usr/bin/env python3
"""
Cross-modal model utilities for Enilnets.
Supports image-text alignment, audio-text alignment, and multimodal generation.
"""
from typing import Any, Dict, List, Optional, Tuple

from ..core.backend import np

def contrastive_loss(image_embeds: Any, text_embeds: Any, temperature: float = 0.07) -> float:
    """Symmetric InfoNCE loss for CLIP-style image-text alignment. Both
    embedding arrays are (N, embed_dim) and expected already normalized."""
    N = image_embeds.shape[0]

    # Cosine similarity matrix
    logits = np.dot(image_embeds, text_embeds.T) / temperature

    # Labels: diagonal is positive pairs
    labels = np.arange(N)

    # Image-to-text loss
    i2t_exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    i2t_probs = i2t_exp / np.sum(i2t_exp, axis=1, keepdims=True)
    i2t_loss = -np.mean(np.log(i2t_probs[np.arange(N), labels] + 1e-12))

    # Text-to-image loss
    t2i_exp = np.exp(logits.T - np.max(logits.T, axis=1, keepdims=True))
    t2i_probs = t2i_exp / np.sum(t2i_exp, axis=1, keepdims=True)
    t2i_loss = -np.mean(np.log(t2i_probs[np.arange(N), labels] + 1e-12))

    return float((i2t_loss + t2i_loss) / 2.0)

def clip_normalize(embeddings: Any) -> Any:
    """L2-normalize embeddings to unit sphere."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    return embeddings / norms

def multimodal_fusion(embeddings_list: List[Any], fusion_type: str = "concat",
                       weights: Optional[List[float]] = None) -> Any:
    """Fuse a list of (N, embed_dim_i) modality embeddings. `fusion_type` is
    "concat", "sum", "attention", or "gated".

    `weights` applies to "sum"/"gated" only (and is validated against the
    list length there); "concat" ignores it and "attention" computes its own
    data-dependent per-sample weighting instead."""
    if fusion_type == "concat":
        return np.concatenate(embeddings_list, axis=1)

    elif fusion_type == "sum":
        if weights is None:
            weights = [1.0 / len(embeddings_list)] * len(embeddings_list)
        elif len(weights) != len(embeddings_list):
            raise ValueError(
                f"weights has {len(weights)} entries but embeddings_list has "
                f"{len(embeddings_list)} -- zip() would otherwise silently "
                "truncate to the shorter one instead of raising."
            )
        fused = np.zeros_like(embeddings_list[0])
        for emb, w in zip(embeddings_list, weights):
            fused += w * emb
        return fused

    elif fusion_type == "attention":
        # Data-dependent attention over modalities: each sample gets its own
        # weighting (unlike "gated"'s static per-modality weights), based on
        # how well each modality's embedding agrees with the cross-modality
        # mean for that sample.
        stacked = np.stack(embeddings_list, axis=0)  # (M, N, D)
        query = stacked.mean(axis=0, keepdims=True)
        scores = np.sum(stacked * query, axis=-1) / np.sqrt(stacked.shape[-1])
        scores -= scores.max(axis=0, keepdims=True)
        attn = np.exp(scores) / np.sum(np.exp(scores), axis=0, keepdims=True)
        fused = np.sum(stacked * attn[:, :, None], axis=0)
        return fused

    elif fusion_type == "gated":
        if weights is None:
            weights = [1.0 / len(embeddings_list)] * len(embeddings_list)
        elif len(weights) != len(embeddings_list):
            raise ValueError(
                f"weights has {len(weights)} entries but embeddings_list has "
                f"{len(embeddings_list)} -- a mismatched length breaks the "
                "gate_logits reshape below with a confusing shape-mismatch "
                "error instead of this clear one."
            )
        # Simple gating: weighted sum with learned importance
        stacked = np.stack(embeddings_list, axis=0)  # (M, N, D)
        # Compute gate values (simple softmax over modalities)
        gate_logits = np.array(weights).reshape(-1, 1, 1)
        gates = np.exp(gate_logits) / np.sum(np.exp(gate_logits), axis=0)
        fused = np.sum(stacked * gates, axis=0)
        return fused

    else:
        raise ValueError(f"Unknown fusion_type: {fusion_type}")

def create_text_conditioned_image(image_shape: Tuple[int, ...], text_embed_dim: int,
                                   num_classes: int = 10) -> Dict[str, Any]:
    """Build a configuration dict for a text-conditioned image generator.
    `image_shape` is (H, W, C) or (C, H, W)."""
    return {
        "image_shape": image_shape,
        "text_embed_dim": text_embed_dim,
        "num_classes": num_classes,
        "conditioning_type": "concat",  # or "cross_attention"
        "latent_dim": 128,
        "encoder_hidden": [512, 256],
        "decoder_hidden": [256, 512]
    }
