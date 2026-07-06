#!/usr/bin/env python3
"""
Pure NumPy evaluation metrics for generative models.
No external dependencies.
"""
from .backend import np

def inception_score(samples, classifier=None, splits=10):
    """
    Compute Inception Score (IS) for generated samples.

    Parameters
    ----------
    samples : ndarray
        Generated samples, shape (N, ...)
    classifier : NeuralNet or None
        Pre-trained classifier. If None, uses k-means clustering as proxy.
    splits : int
        Number of splits for computing mean/std

    Returns
    -------
    mean : float
    std : float
    """
    N = samples.shape[0]
    split_scores = []

    if classifier is None:
        # Fallback: use k-means-like clustering for proxy classes
        for i in range(splits):
            part = samples[i * N // splits:(i + 1) * N // splits]
            flat = part.reshape(part.shape[0], -1)

            # Simple k-means with k=10
            k = 10
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

def frechet_distance(mu1, sigma1, mu2, sigma2):
    """
    Compute Fréchet distance between two Gaussians.
    Used in Fréchet Inception Distance (FID).
    """
    diff = mu1 - mu2
    # Compute sqrt of product of covariances via eigendecomposition
    covmean = _sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = np.sum(diff ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)

def _sqrtm(matrix):
    """Matrix square root via eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 0)
    return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T

def compute_fid(real_features, fake_features):
    """
    Compute Fréchet Inception Distance (FID).

    Parameters
    ----------
    real_features : ndarray
        (N, feature_dim) from real data
    fake_features : ndarray
        (N, feature_dim) from generated data

    Returns
    -------
    fid : float
    """
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    return frechet_distance(mu1, sigma1, mu2, sigma2)

def reconstruction_error(original, reconstructed, metric="mse"):
    """
    Compute reconstruction error.

    Parameters
    ----------
    original : ndarray
    reconstructed : ndarray
    metric : str
        "mse", "mae", or "psnr"

    Returns
    -------
    error : float
    """
    if metric == "mse":
        return float(np.mean((original - reconstructed) ** 2))
    elif metric == "mae":
        return float(np.mean(np.abs(original - reconstructed)))
    elif metric == "psnr":
        mse = np.mean((original - reconstructed) ** 2)
        if mse == 0:
            return float('inf')
        max_val = 1.0  # assuming normalized [0,1]
        return float(20 * np.log10(max_val) - 10 * np.log10(mse))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def sample_diversity(samples):
    """
    Measure diversity of generated samples using pairwise distances.

    Returns
    -------
    diversity : float
        Average pairwise L2 distance
    """
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

def nearest_neighbor_accuracy(real_features, fake_features, k=5):
    """
    Measure how similar generated samples are to real samples.
    Lower = more novel/diverse, higher = more similar to real data.

    Parameters
    ----------
    real_features : ndarray
    fake_features : ndarray
    k : int
        Number of nearest neighbors

    Returns
    -------
    accuracy : float
        Fraction of fake samples whose k-NN are real samples
    """
    # Compute pairwise distances
    dists = np.linalg.norm(fake_features[:, None, :] - real_features[None, :, :], axis=2)

    # For each fake sample, find k nearest real samples
    knn = np.argsort(dists, axis=1)[:, :k]

    # Compute average distance to k-NN
    avg_dists = np.mean(np.take_along_axis(dists, knn, axis=1), axis=1)
    return float(np.mean(avg_dists))
