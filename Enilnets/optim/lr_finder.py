"""Learning-rate range test (roadmap item 47).

Ramps the learning rate exponentially across a few hundred training batches
and records the loss, so the usable range is read off a curve instead of
guessed. Smith (2015): pick a rate near where the loss is falling fastest,
well before the point where it turns up.

This is a probe, not training -- the model's weights, optimizer state, step
counter and learning rate are all restored before returning.
"""

import copy
import math
from typing import Any, Dict, List, Optional

from ..core.utils import iterate_minibatches


def _snapshot(model: Any) -> Dict[str, Any]:
    return {
        "weights": model.get_weights(),
        "opt_state": copy.deepcopy(model.opt_state),
        "t": model.t,
        "lr": model.learning_rate,
        "grad_accum": copy.deepcopy(model._grad_accum),
        "accum_steps": model._accum_steps,
    }


def _restore(model: Any, snap: Dict[str, Any]) -> None:
    model.set_weights(snap["weights"])
    model.opt_state = snap["opt_state"]
    model.t = snap["t"]
    model.learning_rate = snap["lr"]
    model._grad_accum = snap["grad_accum"]
    model._accum_steps = snap["accum_steps"]


def _suggest(lrs: List[float], losses: List[float]) -> Optional[float]:
    """The learning rate at the steepest downward slope of the smoothed
    loss, which is the conventional reading of a range test -- not the
    minimum, which already sits in the unstable region."""
    if len(lrs) < 3:
        return None
    steepest, best = None, 0.0
    for i in range(1, len(lrs) - 1):
        dx = math.log(lrs[i + 1]) - math.log(lrs[i - 1])
        if dx <= 0:
            continue
        slope = (losses[i + 1] - losses[i - 1]) / dx
        if slope < best:
            best, steepest = slope, lrs[i]
    return steepest


def find_learning_rate(model: Any, X: Any, Y: Any, start_lr: float = 1e-7,
                       end_lr: float = 1.0, num_iter: int = 100,
                       batch_size: int = 32, loss_function: Optional[str] = None,
                       diverge_factor: float = 4.0, smooth_beta: float = 0.98
                       ) -> Dict[str, Any]:
    """Run an LR range test. Returns ``{"lrs", "losses", "raw_losses",
    "suggested_lr"}``; `losses` is the EMA-smoothed, bias-corrected curve
    that `suggested_lr` is read from.

    Stops early once the smoothed loss exceeds `diverge_factor` times its
    best value -- past that point the curve carries no information and the
    weights are only getting worse. The model is left exactly as it was."""
    if num_iter < 2:
        raise ValueError(f"num_iter must be >= 2, got {num_iter}")
    if not 0 < start_lr < end_lr:
        raise ValueError(
            f"need 0 < start_lr < end_lr, got start_lr={start_lr}, end_lr={end_lr}")

    snap = _snapshot(model)
    gamma = (end_lr / start_lr) ** (1.0 / (num_iter - 1))
    lrs: List[float] = []
    raw: List[float] = []
    smoothed: List[float] = []
    avg, best = 0.0, math.inf
    step = 0

    try:
        while step < num_iter:
            for xb, yb in iterate_minibatches(X, Y, batch_size, shuffle=True):
                if step >= num_iter:
                    break
                lr = start_lr * gamma ** step
                model.set_lr(lr)
                loss, _ = model.TrainBatch(xb, yb, loss_function=loss_function)
                loss = float(loss)

                # EMA smoothing with bias correction, so the first few points
                # are not dragged towards the zero the average starts at.
                avg = smooth_beta * avg + (1 - smooth_beta) * loss
                value = avg / (1 - smooth_beta ** (step + 1))
                lrs.append(lr)
                raw.append(loss)
                smoothed.append(value)
                step += 1

                if not math.isfinite(value):
                    step = num_iter
                    break
                best = min(best, value)
                if value > diverge_factor * best:
                    step = num_iter
                    break
    finally:
        _restore(model, snap)

    return {"lrs": lrs, "losses": smoothed, "raw_losses": raw,
            "suggested_lr": _suggest(lrs, smoothed)}
