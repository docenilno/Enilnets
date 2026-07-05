import numpy as np

def ComputeLoss(self, output, target, function="mse", reduction="mean", **kwargs):
    o = np.asarray(output, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if function == "mse":
        loss = (o - t) ** 2
    elif function == "mae":
        loss = np.abs(o - t)
    elif function == "huber":
        delta = kwargs.get("delta", 1.0)
        diff = np.abs(o - t)
        loss = np.where(diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))
    elif function == "smooth_l1":
        diff = np.abs(o - t)
        loss = np.where(diff < 1, 0.5 * diff**2, diff - 0.5)
    elif function == "binary_cross_entropy":
        o = np.clip(o, 1e-12, 1 - 1e-12)
        loss = -(t * np.log(o) + (1 - t) * np.log(1 - o))
    elif function in ("cross_entropy", "categorical_cross_entropy"):
        o = np.clip(o, 1e-12, 1.0)
        loss = -t * np.log(o)
        if reduction == "mean":
            return float(np.sum(loss) / o.shape[0])
        if reduction == "sum":
            return float(np.sum(loss))
        return loss
    elif function == "focal":
        alpha = kwargs.get("alpha", 0.25)
        gamma = kwargs.get("gamma", 2.0)
        o = np.clip(o, 1e-12, 1.0)
        pt = o * t + (1 - o) * (1 - t)
        loss = - (alpha * t * (1 - pt) ** gamma * np.log(o) + (1 - alpha) * (1 - t) * pt ** gamma * np.log(1 - o))
    elif function == "hinge":
        loss = np.maximum(0, 1 - t * o)
    elif function == "kl_divergence":
        # KL(q||p) where q~N(mu, var) and p~N(0,1)
        mu = kwargs.get("mu", o)
        logvar = kwargs.get("logvar", t)
        loss = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar), axis=-1)
    elif function == "bce_logits":
        # BCE with logits (numerically stable)
        # loss = max(x,0) - x*t + log(1+exp(-|x|))
        loss = np.maximum(o, 0) - o * t + np.log(1 + np.exp(-np.abs(o)))
    elif function == "wasserstein":
        loss = -o * t  # for WGAN: t=1 for real, t=-1 for fake
    else:
        raise ValueError(f"Unknown loss function: {function}")

    if reduction == "mean":
        return float(np.mean(loss))
    if reduction == "sum":
        return float(np.sum(loss))
    return loss
