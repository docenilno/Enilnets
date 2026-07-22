#!/usr/bin/env python3
"""
Pure NumPy evaluation metrics for generative models.
No external dependencies.
"""
from typing import Any, Optional, Tuple

from ..core.backend import np

def inception_score(samples: Any, classifier: Optional[Any] = None, splits: int = 10) -> Tuple[float, float]:
    """Inception Score for generated `samples` (N, ...), returned as
    (mean, std) over `splits` splits. Without a `classifier`, k-means
    clustering stands in for the class posterior."""
    N = samples.shape[0]
    split_scores = []

    if classifier is None:
        # Fallback: use k-means-like clustering for proxy classes
        for i in range(splits):
            part = samples[i * N // splits:(i + 1) * N // splits]
            flat = part.reshape(part.shape[0], -1)

            # Simple k-means with k=10 (clamped to the split's sample count,
            # since np.random.choice(..., replace=False) raises if asked for
            # more centers than there are samples to draw from -- an easy
            # case to hit with small batches or a large `splits`).
            k = min(10, len(flat))
            # Random initialization
            centers = flat[np.random.choice(len(flat), k, replace=False)]

            for _ in range(5):  # few iterations
                # Assign to nearest center
                dists = np.linalg.norm(flat[:, None, :] - centers[None, :, :], axis=2)
                labels = np.argmin(dists, axis=1)
                # Update centers
                for j in range(k):
                    mask = labels == j
                    if np.sum(mask) > 0:
                        centers[j] = flat[mask].mean(axis=0)

            counts = np.bincount(labels, minlength=k)
            p_y = counts / counts.sum()
            # KL(p(y|x) || p(y)) approximated by entropy difference
            kl = np.sum(p_y * np.log(p_y + 1e-12)) - np.log(1.0 / k)
            split_scores.append(np.exp(-kl))  # Higher = more diverse
    else:
        for i in range(splits):
            part = samples[i * N // splits:(i + 1) * N // splits]
            preds = classifier.Forward(part)
            p_yx = preds / (preds.sum(axis=1, keepdims=True) + 1e-12)
            p_y = p_yx.mean(axis=0)
            kl = p_yx * (np.log(p_yx + 1e-12) - np.log(p_y + 1e-12))
            split_scores.append(np.exp(kl.sum(axis=1).mean()))

    split_scores = np.array(split_scores)
    return float(np.mean(split_scores)), float(np.std(split_scores))

def frechet_distance(mu1: Any, sigma1: Any, mu2: Any, sigma2: Any) -> float:
    """Frechet distance between two Gaussians -- the core of FID."""

    # FID needs only trace(sqrtm(sigma1 @ sigma2)), and that product is
    # generally NOT symmetric (a product of symmetric matrices stays symmetric
    # only if they commute), so eigh would silently be wrong on it.
    #
    # Standard similarity-transform trick instead: for symmetric PSD A, B,
    # C = sqrtm(A) @ B @ sqrtm(A) is genuinely symmetric PSD -- so eigh is
    # valid -- and similar to A @ B (C = A^-1/2 (AB) A^1/2), hence has the same
    # eigenvalues. So sqrtm(C) and sqrtm(AB) share a trace, giving the exact
    # answer with only a symmetric eigendecomposition: no general matrix square
    # root, no Schur decomposition, no new dependency.
    diff = mu1 - mu2
    sigma1_sqrt = _sqrtm_symmetric(sigma1)
    inner = sigma1_sqrt @ sigma2 @ sigma1_sqrt
    inner_eigenvalues = np.maximum(np.linalg.eigvalsh(inner), 0)
    trace_covmean = np.sum(np.sqrt(inner_eigenvalues))
    fid = np.sum(diff ** 2) + np.trace(sigma1) + np.trace(sigma2) - 2 * trace_covmean
    return float(fid)

def _sqrtm_symmetric(matrix: Any) -> Any:
    """Matrix square root of a symmetric positive-semidefinite matrix via
    eigendecomposition. Callers must ensure `matrix` is actually symmetric
    (e.g. a covariance matrix) -- eigh silently only reads one triangular
    half, so passing a non-symmetric matrix here gives a wrong answer with
    no error."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 0)
    return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T

def compute_fid(real_features: Any, fake_features: Any) -> float:
    """Frechet Inception Distance between two (N, feature_dim) feature sets."""
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    return frechet_distance(mu1, sigma1, mu2, sigma2)

def reconstruction_error(original: Any, reconstructed: Any, metric: str = "mse", max_val: float = 1.0) -> float:
    """Reconstruction error between `original` and `reconstructed`.
    `metric` is "mse", "mae", or "psnr".

    `max_val` is used only by "psnr" and must match the data's actual range
    (1.0 for [0,1] data, 255 for raw uint8); a wrong value silently yields a
    meaningless PSNR, since the range cannot be inferred from the data."""
    if metric == "mse":
        return float(np.mean((original - reconstructed) ** 2))
    elif metric == "mae":
        return float(np.mean(np.abs(original - reconstructed)))
    elif metric == "psnr":
        mse = np.mean((original - reconstructed) ** 2)
        if mse == 0:
            return float('inf')
        return float(20 * np.log10(max_val) - 10 * np.log10(mse))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def sample_diversity(samples: Any) -> float:
    """Average pairwise L2 distance between generated samples."""
    N = samples.shape[0]
    flat = samples.reshape(N, -1)
    # Sample subset for efficiency
    subset_size = min(N, 100)
    idx = np.random.choice(N, subset_size, replace=False)
    subset = flat[idx]

    if subset_size < 2:
        return 0.0
    pairwise = np.linalg.norm(subset[:, None, :] - subset[None, :, :], axis=-1)
    iu = np.triu_indices(subset_size, k=1)
    return float(np.mean(pairwise[iu]))

def nearest_neighbor_accuracy(real_features: Any, fake_features: Any, k: int = 5) -> float:
    """Mean Euclidean distance from each fake sample to its `k` nearest real
    neighbors. Lower = fake samples sit close to real data.

    Despite the name this is NOT a [0, 1] accuracy but an unbounded
    distance; the name is kept for backward compatibility. Compare relative
    values, not absolute ones."""
    # Compute pairwise distances
    dists = np.linalg.norm(fake_features[:, None, :] - real_features[None, :, :], axis=2)

    # For each fake sample, find k nearest real samples
    knn = np.argsort(dists, axis=1)[:, :k]

    # Compute average distance to k-NN
    avg_dists = np.mean(np.take_along_axis(dists, knn, axis=1), axis=1)
    return float(np.mean(avg_dists))
