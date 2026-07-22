"""Small helpers shared across generative models that manually backprop
through a standalone `NeuralNet` subnetwork given only a gradient w.r.t. its
final output (i.e. outside the normal loss/Backward(targets=...) path)."""
from typing import Any


def _manual_sequential_backward(subnetwork: Any, output_grad: Any) -> None:
    """Backprop through `subnetwork` (already Forward'd with training=True)
    given the gradient w.r.t. its final ACTIVATED output -- i.e. exactly what
    Forward() returned. Populates `subnetwork.deltas` and every layer's
    parameter gradients, ready for `.update()`."""

    # `output_grad` is converted to the pre-activation dL/dz form that
    # NeuralNet.Backward's `output_delta=` actually expects, via the last
    # layer's activation derivative, before delegating to that path. Skipping
    # the conversion is a real bug for any subnetwork whose last layer is not
    # linear -- e.g. UNetDenoiser's swish-activated blocks. DiffusionModel's
    # denoisers always end linear, which is why this never surfaced there.
    from ..nn.activations import derivative
    last = subnetwork.layers[-1]
    z = subnetwork.pre_activations[-1] if subnetwork.pre_activations[-1] is not None else subnetwork.outputs[-1]
    pre_activation_delta = output_grad * derivative(
        last.get("activation", "linear"), z, cached_output=subnetwork.outputs[-1],
        **last.get("activation_params", {})
    )
    subnetwork.Backward(None, output_delta=pre_activation_delta)


def _conv_stack_input_gradient(subnetwork: Any) -> Any:
    """Given a `NeuralNet` built purely from `conv2d` layers whose
    `subnetwork.deltas` has already been populated (via
    `_manual_sequential_backward` or `.Backward(...)`), return the gradient
    w.r.t. the network's own raw input (before its very first layer) --
    i.e. one more backward step past what `.deltas[0]` alone gives you
    (that's the gradient at layer 0's *output*, not its input). Reuses
    `conv2d_backward_input` (already pad/stride-aware) directly, so this is
    the same primitive the main `Backward()` loop uses for every
    interior layer, just applied one layer further.
    """
    from ..nn.backward import conv2d_backward_input
    first_layer = subnetwork.layers[0]
    if first_layer["type"] != "conv2d":
        raise ValueError(
            "_conv_stack_input_gradient only supports networks whose first "
            f"layer is conv2d, got {first_layer['type']!r}"
        )
    input_tensor = subnetwork.outputs[0]
    return conv2d_backward_input(
        subnetwork.deltas[0], first_layer["weights"], input_tensor.shape,
        stride=first_layer.get("stride", 1), pad=first_layer.get("pad", 0),
    )
