from typing import Any, Optional

from ..core.backend import np

def kl_divergence_gaussian(mu: Any, logvar: Any, reduction: str = "mean", kl_weight: float = 1.0) -> Any:
    """
    KL(q(z|x) || N(0, I)) for VAE.
    mu, logvar: arrays of shape (batch, latent_dim)
    kl_weight: scalar weight for the KL term (beta-VAE style).
               0.0 = pure autoencoder, 1.0 = standard VAE.
    """
    kl = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar), axis=-1)
    kl = kl * kl_weight
    if reduction == "mean":
        return float(np.mean(kl))
    if reduction == "sum":
        return float(np.sum(kl))
    return kl

def adversarial_loss_discriminator(real_logits: Any, fake_logits: Any, loss_type: str = "bce") -> float:
    """
    Discriminator loss.
    real_logits: D(real) output (before sigmoid if bce_logits)
    fake_logits: D(fake) output
    loss_type: "bce" (binary cross entropy), "bce_logits", or "wasserstein"
    """
    if loss_type == "bce":
        real_loss = -np.mean(np.log(np.clip(real_logits, 1e-12, 1.0)))
        fake_loss = -np.mean(np.log(1 - np.clip(fake_logits, 1e-12, 1.0)))
        return float(real_loss + fake_loss)
    elif loss_type == "bce_logits":
        real_loss = np.mean(np.maximum(real_logits, 0) - real_logits + np.log(1 + np.exp(-np.abs(real_logits))))
        fake_loss = np.mean(np.maximum(fake_logits, 0) + np.log(1 + np.exp(-np.abs(fake_logits))))
        return float(real_loss + fake_loss)
    elif loss_type == "wasserstein":
        return float(-np.mean(real_logits) + np.mean(fake_logits))
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

def adversarial_loss_generator(fake_logits: Any, loss_type: str = "bce") -> float:
    """
    Generator loss (trying to fool discriminator).
    """
    if loss_type == "bce":
        return float(-np.mean(np.log(np.clip(fake_logits, 1e-12, 1.0))))
    elif loss_type == "bce_logits":
        return float(np.mean(np.maximum(fake_logits, 0) - fake_logits + np.log(1 + np.exp(-np.abs(fake_logits)))))
    elif loss_type == "wasserstein":
        return float(-np.mean(fake_logits))
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

def diffusion_loss(predicted_noise: Any, true_noise: Any, reduction: str = "mean") -> Any:
    """
    MSE between predicted and true noise for diffusion models.
    """
    loss = (predicted_noise - true_noise) ** 2
    if reduction == "mean":
        return float(np.mean(loss))
    if reduction == "sum":
        return float(np.sum(loss))
    return loss

def nll_loss(log_px: Any, log_det_jacobian: Any, reduction: str = "mean") -> Any:
    """
    Negative log-likelihood for normalizing flows.
    log_px: log p(z) where z is the transformed variable (base distribution)
    log_det_jacobian: log |det(J)|
    """
    nll = -(log_px + log_det_jacobian)
    if reduction == "mean":
        return float(np.mean(nll))
    if reduction == "sum":
        return float(np.sum(nll))
    return nll

def energy_loss(data_energy: Any, sample_energy: Any, margin: float = 1.0) -> float:
    """
    Contrastive loss for energy-based models.
    Pushes down energy on data, pushes up on samples.
    """
    loss = data_energy + np.maximum(0, margin - sample_energy)
    return float(np.mean(loss))

def perceptual_loss(x: Any, y: Any, feature_extractor: Optional[Any] = None) -> float:
    """
    Perceptual loss using a feature extractor.
    If no feature_extractor provided, falls back to MSE.
    """
    if feature_extractor is None:
        return float(np.mean((x - y) ** 2))
    feat_x = feature_extractor(x)
    feat_y = feature_extractor(y)
    return float(np.mean((feat_x - feat_y) ** 2))

def vgg_loss(x: Any, y: Any, vgg_features: Optional[Any] = None) -> float:
    """VGG-based perceptual loss. `vgg_features` is any callable duck-typing
    to `feature_extractor(x) -> features` (e.g.
    `generative.pretrained.build_vgg16_feature_extractor(...).Forward`, with
    your own converted pretrained weights loaded via `.set_weights()` --
    Enilnets never downloads weights itself). Falls back to plain MSE on the
    raw inputs if not given, mirroring `perceptual_loss`."""
    if vgg_features is None:
        return float(np.mean((x - y) ** 2))
    feat_x = vgg_features(x)
    feat_y = vgg_features(y)
    return float(np.mean((feat_x - feat_y) ** 2))
