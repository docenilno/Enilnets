"""Weight pruning (roadmap items 62, 64, 65).

Magnitude pruning (#62) zeroes the smallest weights and keeps them zeroed,
leaving shapes untouched. Dynamic pruning (#64) does it gradually during
training. Structured pruning (#65) removes whole channels, which really
does change shapes and must be threaded through every consuming layer.

A pruned parameter is recorded in ``layer["prune_mask"][name]`` -- 1.0
where the weight survives, 0.0 where it does not -- and ``apply_gradients``
enforces it every step, so pruning survives further training."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.backend import np
from ..core import backend


#: Which parameter of each layer type magnitude pruning considers. Biases,
#: normalization scales and router weights are excluded: they are few, and
#: zeroing them damages the model far out of proportion to what it saves.
PRUNABLE = {
    "dense": ("weights",),
    "sparse": ("weights",),
    "conv2d": ("weights",),
    "conv1d": ("weights",),
    "embedding": ("weights",),
    "multihead_attention": ("Wq", "Wk", "Wv", "Wo"),
    "cross_attention": ("Wq", "Wk", "Wv", "Wo"),
    "rnn": ("Wx", "Wh"),
    "lstm": ("Wx", "Wh"),
    "gru": ("Wx", "Wh"),
    "moe": ("W1", "W2"),
    "cbam_channel": ("W1", "W2"),
}


def prunable_parameters(model: Any,
                        layer_types: Optional[Sequence[str]] = None
                        ) -> List[Tuple[int, str]]:
    """``(layer_index, parameter_name)`` for everything pruning may touch."""
    out = []
    for i, layer in enumerate(model.layers):
        t = layer["type"]
        if layer_types is not None and t not in layer_types:
            continue
        for name in PRUNABLE.get(t, ()):
            if name in layer:
                out.append((i, name))
    return out


def sparsity(model: Any) -> Dict[str, float]:
    """Report what fraction of prunable weights are currently zero, overall
    and per layer. Measures the WEIGHTS, not the masks, so it is honest
    about a model pruned by any means."""
    per_layer, zeros, total = {}, 0, 0
    for i, name in prunable_parameters(model):
        w = model.layers[i][name]
        z, n = int(np.count_nonzero(w == 0)), int(w.size)
        per_layer[f"{i}.{name}"] = z / n if n else 0.0
        zeros += z
        total += n
    return {"overall": zeros / total if total else 0.0, **per_layer}


def _threshold(values: Any, fraction: float) -> float:
    """The magnitude below which `fraction` of `values` falls."""
    k = int(round(fraction * values.size))
    if k <= 0:
        return -1.0                     # prune nothing
    if k >= values.size:
        return float(np.max(values)) + 1.0      # prune everything
    flat = np.sort(values.reshape(-1))
    return float(flat[k - 1])


def set_mask(layer: Dict[str, Any], name: str, mask: Any) -> None:
    """Attach (or replace) a pruning mask and apply it immediately."""
    masks = layer.setdefault("prune_mask", {})
    masks[name] = mask
    layer[name] = layer[name] * mask


def clear_masks(model: Any) -> None:
    """Drop every pruning mask. Already-zeroed weights stay zero -- this
    only stops them being *held* at zero, so fine-tuning can recover them."""
    for layer in model.layers:
        layer.pop("prune_mask", None)


def prune_magnitude(model: Any, amount: float = 0.5, scope: str = "global",
                    layer_types: Optional[Sequence[str]] = None,
                    zero_optimizer_state: bool = True) -> Dict[str, float]:
    """Zero the `amount` fraction of smallest-magnitude weights and hold them
        there. Returns the resulting :func:`sparsity` report.

        scope="global" ranks every prunable weight together, so layers that
        matter keep more capacity; "layer" takes the same fraction from each.

        zero_optimizer_state also clears the accumulators of the pruned
        weights: a stale momentum keeps pushing a weight the mask then
        re-zeroes, and corrupts the adaptive denominators of those that
        remain."""
    if not 0.0 <= amount <= 1.0:
        raise ValueError(f"amount must be in [0, 1], got {amount}")
    if scope not in ("global", "layer"):
        raise ValueError(f"scope must be 'global' or 'layer', got {scope!r}")
    targets = prunable_parameters(model, layer_types)
    if not targets:
        raise ValueError(
            "no prunable parameters found -- check `layer_types`, or see "
            "compression.pruning.PRUNABLE for what is eligible")

    if scope == "global":
        pooled = np.concatenate([np.abs(model.layers[i][n]).reshape(-1)
                                 for i, n in targets])
        cutoff = _threshold(pooled, amount)
        thresholds = {(i, n): cutoff for i, n in targets}
    else:
        thresholds = {(i, n): _threshold(np.abs(model.layers[i][n]), amount)
                      for i, n in targets}

    for i, name in targets:
        layer = model.layers[i]
        mask = (np.abs(layer[name]) > thresholds[(i, name)]).astype(
            backend.default_dtype())
        set_mask(layer, name, mask)
        if zero_optimizer_state:
            _zero_state(model, i, name, mask)
    return sparsity(model)


def _zero_state(model: Any, index: int, name: str, mask: Any) -> None:
    """Zero every optimizer accumulator wherever `mask` is 0."""
    if not model.opt_state or index >= len(model.opt_state):
        return
    state = model.opt_state[index]
    if not state:
        return
    for key, buf in state.items():
        # Keys are "<slot>_<param>"; only this parameter's buffers, and only
        # those shaped like it (AdaFactor's factored row/col vectors are not).
        if key.endswith(f"_{name}") and getattr(buf, "shape", None) == mask.shape:
            state[key] = buf * mask


def apply_masks(model: Any) -> None:
    """Re-apply every mask. ``apply_gradients`` calls this per step; call it
    yourself after any manual weight surgery (``set_weights``, ``Load``)."""
    for layer in model.layers:
        for name, mask in layer.get("prune_mask", {}).items():
            if name in layer:
                layer[name] = layer[name] * mask


class PruningSchedule:
    """Gradual magnitude pruning during training (roadmap item 64).

        Sparsity ramps from `initial` to `final` between `start_step` and
        `end_step` on Zhu & Gupta's cubic schedule -- fast early, tapering
        later, so the model recovers between the more damaging increments --
        acting only every `frequency` steps. Call :meth:`step` once per
        optimizer step."""

    def __init__(self, model: Any, final: float = 0.5, initial: float = 0.0,
                 start_step: int = 0, end_step: int = 1000,
                 frequency: int = 50, scope: str = "global",
                 layer_types: Optional[Sequence[str]] = None) -> None:
        if not 0.0 <= initial <= final <= 1.0:
            raise ValueError(
                f"need 0 <= initial <= final <= 1, got initial={initial}, "
                f"final={final}")
        if end_step <= start_step:
            raise ValueError(
                f"end_step ({end_step}) must be after start_step ({start_step})")
        if frequency < 1:
            raise ValueError(f"frequency must be >= 1, got {frequency}")
        self.model = model
        self.initial, self.final = float(initial), float(final)
        self.start_step, self.end_step = int(start_step), int(end_step)
        self.frequency = int(frequency)
        self.scope, self.layer_types = scope, layer_types
        self.current_step = 0
        self.current_sparsity = 0.0

    def target_sparsity(self, step: int) -> float:
        """The cubic ramp's target at `step`."""
        if step < self.start_step:
            return self.initial
        if step >= self.end_step:
            return self.final
        progress = (step - self.start_step) / (self.end_step - self.start_step)
        return self.final + (self.initial - self.final) * (1.0 - progress) ** 3

    def step(self) -> float:
        """Advance one training step, re-pruning if this one is due.
        Returns the sparsity target now in force."""
        target = self.target_sparsity(self.current_step)
        due = (self.current_step % self.frequency == 0
               or self.current_step == self.end_step)
        if due and self.start_step <= self.current_step and target > 0.0:
            prune_magnitude(self.model, amount=target, scope=self.scope,
                            layer_types=self.layer_types)
            self.current_sparsity = target
        self.current_step += 1
        return target


# ------------------------------------------------------- structured pruning

#: Layer types whose OUTPUT channels can be removed, and the axis of each
#: parameter that indexes them.
_OUT_AXIS = {"dense": {"weights": 0, "bias": 0},
             "conv2d": {"weights": 0, "bias": 0},
             "conv1d": {"weights": 0, "bias": 0}}

#: ...and the axis that indexes a layer's INPUT channels, which is what a
#: consumer has to shrink when its producer loses channels.
_IN_AXIS = {"dense": {"weights": 1},
            "conv2d": {"weights": 1},
            "conv1d": {"weights": 1},
            "batchnorm": None,          # handled separately: all params are per-channel
            "layernorm": None}


def channel_importance(layer: Dict[str, Any], norm: str = "l1") -> Any:
    """Per-output-channel importance, used to decide what to remove.
    ``l1``/``l2`` of each filter's weights."""
    w = layer["weights"]
    flat = w.reshape(w.shape[0], -1)
    return np.sum(np.abs(flat), axis=1) if norm == "l1" else \
        np.sqrt(np.sum(flat * flat, axis=1))


def prune_channels(model: Any, layer_index: int, amount: float = 0.5,
                   norm: str = "l1") -> Dict[str, Any]:
    """Remove the least important output channels of one conv/dense layer,
    ACTUALLY shrinking it and every layer that consumes it (roadmap #65).

    Unlike magnitude pruning this changes shapes, so the model really does
    get smaller and faster rather than just sparser. That is also why it is
    restricted: the consumer must be a layer whose input axis can be
    narrowed unambiguously. A flatten between a conv and a dense layer
    remaps channels to a block of columns, which is handled; anything else
    in between raises rather than silently corrupting the model.

    Returns ``{"kept": indices, "removed": n, "consumer": index}``."""
    if not 0.0 < amount < 1.0:
        raise ValueError(f"amount must be strictly between 0 and 1, got {amount}")
    layer = model.layers[layer_index]
    if layer["type"] not in _OUT_AXIS:
        raise ValueError(
            f"structured pruning supports {sorted(_OUT_AXIS)}, not "
            f"{layer['type']!r} (layer {layer_index})")

    n_out = int(layer["weights"].shape[0])
    keep_count = max(1, int(round(n_out * (1.0 - amount))))
    order = np.argsort(-channel_importance(layer, norm))
    # Sorted so the surviving channels keep their original relative order --
    # otherwise every consumer's columns would need permuting too.
    keep = np.sort(order[:keep_count])

    consumer_index, spatial = _find_consumer(model, layer_index)
    layer["weights"] = layer["weights"][keep]
    if "bias" in layer:
        layer["bias"] = layer["bias"][keep]
    _drop_masks(layer)
    if layer["type"] in ("conv2d", "conv1d"):
        layer["out_ch"] = int(keep_count)

    if consumer_index is not None:
        _narrow_consumer(model, consumer_index, keep, spatial)
    _reset_state(model, layer_index)
    return {"kept": keep, "removed": n_out - keep_count,
            "consumer": consumer_index}


def _find_consumer(model: Any, index: int) -> Tuple[Optional[int], int]:
    """The next layer whose input width depends on `index`'s output, plus
    the spatial-block size a flatten introduced (1 if none).

    Normalization layers in between are rewired in place and skipped over,
    which is exactly the pattern conv -> batchnorm -> conv produces."""
    n_out = int(model.layers[index]["weights"].shape[0])
    flattened = False
    for j in range(index + 1, len(model.layers)):
        t = model.layers[j]["type"]
        if t in ("batchnorm", "layernorm", "dropout"):
            continue                    # narrowed by _narrow_consumer as well
        if t in ("flatten", "maxpool2d", "avgpool2d", "globalavgpool2d",
                 "globalmaxpool2d"):
            flattened = flattened or t == "flatten"
            continue
        if t in ("dense", "conv2d", "conv1d"):
            if not flattened:
                return j, 1
            # The flatten laid each channel out as a contiguous block of
            # columns, so the block size follows from the widths rather than
            # from build-time bookkeeping (which a loaded model may not have).
            in_width = int(model.layers[j]["weights"].shape[1])
            if in_width % n_out != 0:
                raise ValueError(
                    f"cannot structurally prune layer {index}: the consuming "
                    f"dense layer at {j} has input width {in_width}, which is "
                    f"not a multiple of the {n_out} channels being pruned, so "
                    "the channel-to-column mapping is ambiguous.")
            return j, in_width // n_out
        raise ValueError(
            f"cannot structurally prune layer {index}: its output is consumed "
            f"by a {t!r} layer at index {j}, whose input width cannot be "
            "narrowed unambiguously. Use magnitude pruning there instead.")
    return None, 1


def _narrow_consumer(model: Any, index: int, keep: Any, spatial: int) -> None:
    """Drop the input columns/channels the producer no longer emits, and
    fix up any normalization layers on the way."""
    for j in range(index):
        layer = model.layers[j]
        if layer["type"] in ("batchnorm", "layernorm") and "gamma" in layer:
            if int(layer["gamma"].shape[0]) != int(keep.shape[0]):
                for key in ("gamma", "beta", "running_mean", "running_var"):
                    if key in layer:
                        layer[key] = layer[key][keep]
                _drop_masks(layer)
    consumer = model.layers[index]
    w = consumer["weights"]
    if consumer["type"] == "dense" and spatial > 1:
        # A flatten mapped each channel to a contiguous block of columns.
        blocks = w.reshape(w.shape[0], -1, spatial)
        consumer["weights"] = blocks[:, keep, :].reshape(w.shape[0], -1)
    else:
        consumer["weights"] = w[:, keep]
    if consumer["type"] in ("conv2d", "conv1d"):
        consumer["in_ch"] = int(keep.shape[0])
    _drop_masks(consumer)
    _reset_state(model, index)


def _drop_masks(layer: Dict[str, Any]) -> None:
    """A pruning mask no longer matches a reshaped parameter."""
    layer.pop("prune_mask", None)


def _reset_state(model: Any, index: int) -> None:
    """Discard a layer's optimizer state after its shapes changed."""
    if model.opt_state and index < len(model.opt_state):
        model.opt_state[index] = None if model.opt_state[index] is None else {}
