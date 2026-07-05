#!/usr/bin/env python3
"""
Cross-modal model utilities for Enilnets.
Supports image-text alignment, audio-text alignment, and multimodal generation.
"""
import numpy as np

def contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    """
    Compute InfoNCE contrastive loss for image-text alignment (CLIP-style).

    Parameters
    ----------
    image_embeds : ndarray
        (N, embed_dim) normalized image embeddings
    text_embeds : ndarray
        (N, embed_dim) normalized text embeddings
    temperature : float
        Temperature scaling factor

    Returns
    -------
    loss : float
        Symmetric contrastive loss
    """
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

def clip_normalize(embeddings):
    """L2-normalize embeddings to unit sphere."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    return embeddings / norms

def multimodal_fusion(embeddings_list, fusion_type="concat", weights=None):
    """
    Fuse embeddings from multiple modalities.

    Parameters
    ----------
    embeddings_list : list of ndarray
        List of (N, embed_dim_i) arrays
    fusion_type : str
        "concat", "sum", "attention", or "gated"
    weights : list of float or None
        Weights for each modality (for "sum" or "gated")

    Returns
    -------
    fused : ndarray
        Fused embeddings
    """
    if fusion_type == "concat":
        return np.concatenate(embeddings_list, axis=1)

    elif fusion_type == "sum":
        if weights is None:
            weights = [1.0 / len(embeddings_list)] * len(embeddings_list)
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
        # Simple gating: weighted sum with learned importance
        stacked = np.stack(embeddings_list, axis=0)  # (M, N, D)
        # Compute gate values (simple softmax over modalities)
        gate_logits = np.array(weights).reshape(-1, 1, 1)
        gates = np.exp(gate_logits) / np.sum(np.exp(gate_logits), axis=0)
        fused = np.sum(stacked * gates, axis=0)
        return fused

    else:
        raise ValueError(f"Unknown fusion_type: {fusion_type}")

def create_text_conditioned_image(image_shape, text_embed_dim, num_classes=10):
    """
    Create a simple text-conditioned image generation setup.
    Returns a configuration dict for building a conditional model.

    Parameters
    ----------
    image_shape : tuple
        (H, W, C) or (C, H, W)
    text_embed_dim : int
        Dimension of text embeddings
    num_classes : int
        Number of conditioning classes

    Returns
    -------
    config : dict
        Configuration for conditional generation
    """
    return {
        "image_shape": image_shape,
        "text_embed_dim": text_embed_dim,
        "num_classes": num_classes,
        "conditioning_type": "concat",  # or "cross_attention"
        "latent_dim": 128,
        "encoder_hidden": [512, 256],
        "decoder_hidden": [256, 512]
    }
