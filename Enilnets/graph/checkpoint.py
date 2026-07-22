"""Gradient checkpointing: trade compute for memory by NOT storing a
segment's intermediate activations during the forward pass, and re-running
that segment's forward during ``backward()`` instead.

Checkpoint boundaries are a graph-level concept (roadmap item 24): the
checkpointed segment appears in the recorded graph as one composite node,
and only its inputs/outputs are kept alive between forward and backward --
everything inside is recomputed exactly when the backward pass reaches it.

Caveat (same as every trace-free checkpoint implementation, PyTorch's
included): a segment containing *randomness* (e.g. ``Dropout`` in training
mode) draws fresh random numbers during the recompute, so its backward
would not match the original forward. Keep checkpointed segments
deterministic, or move the randomness outside the boundary.
"""

from typing import Any, Callable

from .tensor import Tensor, as_tensor, no_grad, is_grad_enabled
from .ops import Op


def checkpoint(fn: Callable, *inputs: Any) -> Tensor:
    """Run ``fn(*inputs)`` without storing its internal graph, recomputing it
    during backward -- trading compute for activation memory.

    ``fn`` must be a pure function of its Tensor arguments returning one
    Tensor. Closures over Parameters are fine: their gradients come out of
    the recompute and accumulate as usual."""
    tensors = tuple(as_tensor(x) for x in inputs)
    with no_grad():
        out = fn(*(Tensor(t.data) for t in tensors))
    if not isinstance(out, Tensor):
        raise TypeError(
            f"checkpoint expects fn to return a single Tensor; got {type(out).__name__}")

    # Conservatively record whenever autograd is on: fn may close over
    # Parameters the engine can't see from the positional inputs alone
    # (e.g. a lambda over a Layer), and silently dropping their gradients
    # would be far worse than an occasional unnecessary recompute.
    needs_grad = is_grad_enabled()
    result = Tensor(out.data, requires_grad=needs_grad)
    # Recorded as one composite node, so traces show the boundary and
    # Graph.run can re-execute it.
    result._op = _make_segment_op(fn)
    result._parents = tensors
    if needs_grad:
        def backward_fn(grad_out: Any) -> tuple:
            # Recompute the segment WITH graph recording, then backprop the
            # upstream gradient through the fresh subgraph. Detached input
            # copies collect the per-input gradients; closure Parameters
            # accumulate into their own .grad directly, exactly as if the
            # segment had never been checkpointed.
            detached = [Tensor(t.data, requires_grad=True) for t in tensors]
            reout = fn(*detached)
            reout.backward(grad_out)
            return tuple(d.grad for d in detached)
        result._backward_fn = backward_fn
    return result


def _make_segment_op(fn: Callable) -> Op:
    """A pseudo-op representing the whole checkpointed segment, so tracing
    and Graph.run treat it like any other node."""
    def forward(*arrays: Any) -> Any:
        with no_grad():
            return fn(*(Tensor(a) for a in arrays)).data
    name = getattr(fn, "__name__", None) or type(getattr(fn, "__self__", fn)).__name__
    return Op(f"checkpoint({name})", forward,
              backward=lambda *a, **k: (_ for _ in ()).throw(
                  RuntimeError("checkpoint segments backprop by recompute, not a static rule")))
