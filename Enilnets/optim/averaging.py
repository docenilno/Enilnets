"""Weight averaging (roadmap items 48 and 49): EMA and SWA.

Both keep a shadow copy of the model's weights, on the observation that an
average of points along the trajectory generalizes better than the final
point. EMA decays the history geometrically and updates every step; SWA
averages epoch-end snapshots equally, while the learning rate is held high
enough to keep moving around the basin.

Neither touches the training loop's gradients."""

from typing import Any, Dict, List, Optional

from ..core.backend import np
from ..core import backend


def _as_arrays(weights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert a list-of-lists state (as it comes back out of a JSON save
    file) into backend arrays at the current working dtype."""
    return [{k: np.asarray(v, dtype=backend.default_dtype()) for k, v in layer.items()}
            for layer in weights]


class _WeightAverager:
    """Shared machinery: hold a shadow copy of the model's weights, and be
    able to swap it in and back out non-destructively."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.shadow: Optional[List[Dict[str, Any]]] = None
        self.num_updates = 0
        self._stashed: Optional[List[Dict[str, Any]]] = None

    def _live(self) -> List[Dict[str, Any]]:
        return self.model.get_weights()

    def apply(self) -> None:
        """Swap the averaged weights into the model, stashing the live ones.
        No-op before the first update()."""
        if self.shadow is None or self._stashed is not None:
            return
        self._stashed = self._live()
        self.model.set_weights(self.shadow)

    def restore(self) -> None:
        """Put the live weights back after :meth:`apply`."""
        if self._stashed is None:
            return
        self.model.set_weights(self._stashed)
        self._stashed = None

    def copy_to(self, other: Any) -> None:
        """Write the averaged weights into another model of the same shape."""
        if self.shadow is None:
            raise RuntimeError("no averaged weights yet -- call update() first")
        other.set_weights(self.shadow)

    def state_dict(self) -> Dict[str, Any]:
        """Serializable state, for `Save(..., extra_state=...)`."""
        return {"num_updates": self.num_updates, "shadow": self.shadow}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.num_updates = int(state["num_updates"])
        shadow = state["shadow"]
        self.shadow = None if shadow is None else _as_arrays(shadow)

    def __enter__(self):
        self.apply()
        return self.model

    def __exit__(self, *exc):
        self.restore()
        return False


class EMA(_WeightAverager):
    """Exponential moving average of a model's weights (roadmap item 48).

    Call update() after every optimizer step. `decay` closer to 1 averages
    over a longer history. With `warmup=True` the effective decay ramps as
    ``min(decay, (1 + n) / (10 + n))``, so the average is not dominated by
    the random initialization for the first few hundred steps.

    Use it as a context manager to evaluate on the averaged weights:
    ``with ema: model.Forward(x, training=False)``.
    """

    def __init__(self, model: Any, decay: float = 0.999,
                 warmup: bool = True) -> None:
        super().__init__(model)
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay = decay
        self.warmup = warmup

    def current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        n = self.num_updates
        return min(self.decay, (1.0 + n) / (10.0 + n))

    def update(self) -> None:
        """Fold the model's current weights into the average."""
        live = self._live()
        if self.shadow is None:
            self.shadow = live
        else:
            d = self.current_decay()
            for layer_avg, layer_new in zip(self.shadow, live):
                for k in layer_avg:
                    layer_avg[k] = d * layer_avg[k] + (1.0 - d) * layer_new[k]
        self.num_updates += 1

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state.update({"decay": self.decay, "warmup": self.warmup})
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.decay = float(state.get("decay", self.decay))
        self.warmup = bool(state.get("warmup", self.warmup))


class SWA(_WeightAverager):
    """Stochastic Weight Averaging (roadmap item 49).

    Call update() once per epoch from `swa_start` onwards, with the LR held
    high (see :meth:`scheduler`). Snapshots are weighted EQUALLY, which is
    the difference from EMA and the reason it wants a flat, high LR.

    Then :meth:`finalize` to install the average, and :meth:`update_bn` if
    the model has batchnorm: those running statistics belong to the
    individual snapshots and are wrong for the average until recomputed.
    """

    def __init__(self, model: Any, swa_start: int = 0, swa_lr: float = 0.05,
                 anneal_epochs: int = 5) -> None:
        super().__init__(model)
        if anneal_epochs < 1:
            raise ValueError(f"anneal_epochs must be >= 1, got {anneal_epochs}")
        self.swa_start = swa_start
        self.swa_lr = swa_lr
        self.anneal_epochs = anneal_epochs

    def should_update(self, epoch: int) -> bool:
        return epoch >= self.swa_start

    def update(self) -> None:
        """Fold the model's current weights into the equal-weight average."""
        live = self._live()
        if self.shadow is None:
            self.shadow = live
        else:
            n = self.num_updates
            for layer_avg, layer_new in zip(self.shadow, live):
                for k in layer_avg:
                    # Running mean, so no snapshot is ever weighted more than
                    # another however long the averaging window runs.
                    layer_avg[k] = layer_avg[k] + (layer_new[k] - layer_avg[k]) / (n + 1)
        self.num_updates += 1

    def finalize(self) -> None:
        """Install the averaged weights permanently."""
        if self.shadow is None:
            raise RuntimeError("no snapshots averaged yet -- call update() first")
        self._stashed = None
        self.model.set_weights(self.shadow)

    def update_bn(self, X: Any, batch_size: int = 32) -> None:
        """Recompute every batchnorm layer's running statistics for the
        current weights by streaming `X` through the model in training mode.

        Necessary after :meth:`finalize`: the stored running mean/var were
        accumulated under the individual snapshots' weights and do not
        describe the average's activations at all."""
        from ..core.utils import iterate_minibatches
        bn = [l for l in self.model.layers if l["type"] == "batchnorm"]
        if not bn:
            return
        saved_momentum = [l.get("momentum", 0.1) for l in bn]
        for layer in bn:
            layer["running_mean"] = np.zeros_like(layer["running_mean"])
            layer["running_var"] = np.ones_like(layer["running_var"])
        try:
            for i, (xb, _) in enumerate(
                    iterate_minibatches(X, X, batch_size, shuffle=False)):
                # momentum = 1/(i+1) turns the running update into an exact
                # cumulative average over the batches seen so far.
                for layer in bn:
                    layer["momentum"] = 1.0 / (i + 1)
                self.model.Forward(xb, training=True)
        finally:
            for layer, m in zip(bn, saved_momentum):
                layer["momentum"] = m

    def scheduler(self, initial_lr: float) -> Any:
        """An :class:`~Enilnets.LRScheduler` in "swa" mode matching this
        SWA's `swa_start` / `swa_lr` / `anneal_epochs`."""
        from ..nn.train import LRScheduler
        return LRScheduler(initial_lr, mode="swa", swa_start=self.swa_start,
                           swa_lr=self.swa_lr, anneal_epochs=self.anneal_epochs)

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state.update({"swa_start": self.swa_start, "swa_lr": self.swa_lr,
                      "anneal_epochs": self.anneal_epochs})
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.swa_start = int(state.get("swa_start", self.swa_start))
        self.swa_lr = float(state.get("swa_lr", self.swa_lr))
        self.anneal_epochs = int(state.get("anneal_epochs", self.anneal_epochs))
