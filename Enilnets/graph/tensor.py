"""The autograd tensor: a backend array plus the dynamic computation graph
built by tracing actual Python execution (define-by-run).

A ``Tensor`` wraps exactly one NumPy/CuPy array as ``.data``, always by
reference and never a defensive copy -- that is what makes mixing
``graph/`` code with ``nn/``-style ``NeuralNet`` layers free at the
boundary. Every operation performed through an
:class:`~Enilnets.graph.ops.Op` records itself on its output tensor, and
``Tensor.backward()`` replays those records in reverse topological order,
accumulating into ``.grad``.

This module knows nothing about specific ops; the elementary op set lives
in ``ops.py``, which attaches the operator overloads to ``Tensor`` at
import time."""

import contextlib
from typing import Any, List, Optional, Tuple

from ..core.backend import np
from ..core import backend


class _GradMode:
    """Process-wide autograd switches (mirrors the library's global-switch
    style for backend/precision). ``enabled`` gates gradient recording;
    ``tracing`` forces recording even for no-grad tensors so the symbolic
    tracer (tracing.py) can capture a graph of a pure-inference call."""
    enabled = True
    tracing = False
    autocast = False


def is_grad_enabled() -> bool:
    """Return True if operations are currently being recorded for autograd."""
    return _GradMode.enabled


@contextlib.contextmanager
def no_grad():
    """Context manager disabling gradient recording (inference mode). Ops still
    compute normally but build no graph, so nothing can be backpropagated
    through them and no intermediate arrays are kept alive."""
    prev = _GradMode.enabled
    _GradMode.enabled = False
    try:
        yield
    finally:
        _GradMode.enabled = prev


def _coerce(data: Any) -> Any:
    """Turn arbitrary Python input into a backend array without ever copying
    an array that is already on the active backend.

    Raw NumPy/CuPy arrays pass through untouched (interop requirement:
    wrapping an existing array in a Tensor must be free). Python scalars/
    lists become arrays at the library's default working dtype if they are
    float-like, or stay integer (token ids / indices) if integer-like."""
    if backend.is_array(data):
        return data
    arr = np.asarray(data)
    if arr.dtype.kind == "f":
        arr = arr.astype(backend.default_dtype())
    elif arr.dtype.kind == "c":
        # Complex width follows the float default: complex64 pairs with
        # float32, complex128 with float64.
        arr = arr.astype(np.complex128 if backend.is_float64_enabled() else np.complex64)
    return arr


class Tensor:
    """A backend array plus its position in the traced computation graph.

    `data` wraps existing NumPy/CuPy arrays BY REFERENCE (never copied);
    Python scalars and lists are converted at the default working dtype.
    `requires_grad` makes backward() accumulate into `.grad` for this tensor
    and anything computed from it. `name` is a cosmetic label for the
    tracer/visualizer."""

    def __init__(self, data: Any, requires_grad: bool = False,
                 name: Optional[str] = None,
                 names: Optional[Tuple[Optional[str], ...]] = None) -> None:
        if isinstance(data, Tensor):
            data = data.data
        self.data = _coerce(data)
        self.requires_grad = bool(requires_grad)
        self.grad: Optional[Any] = None
        self.name = name
        # Named tensors: optional per-DIMENSION labels ("batch", "time",
        # ...), a pure metadata layer -- ops propagate/check them (see
        # ops.py's name-propagation rules) but storage and math are
        # untouched. None = unnamed (the default everywhere).
        self.names: Optional[Tuple[Optional[str], ...]] = None
        if names is not None:
            self.set_names(*names)
        # Autograd bookkeeping, populated by Op.__call__ for non-leaf tensors:
        self._op: Optional[Any] = None            # the Op that produced this
        self._parents: Tuple["Tensor", ...] = ()  # its input Tensors
        self._backward_fn: Optional[Any] = None   # grad_out -> per-input grads
        self._kwargs: dict = {}                   # op kwargs (for the tracer)

    # ------------------------------------------------------------------ #
    # Array-like conveniences
    # ------------------------------------------------------------------ #
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> Any:
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    def numpy(self) -> Any:
        """Return the value as a plain host-side NumPy array (transfers off
        the GPU only if the data actually lives there)."""
        return backend.to_numpy(self.data)

    def item(self) -> float:
        """Return the single element of a size-1 tensor as a Python float."""
        return float(self.data.reshape(-1)[0])

    def detach(self) -> "Tensor":
        """Return a new leaf Tensor sharing this tensor's data (no copy) but
        cut off from the graph -- gradients stop here."""
        return Tensor(self.data, names=self.names)

    # ------------------------------------------------------------------ #
    # Named dimensions (metadata only -- see ops.py for propagation rules)
    # ------------------------------------------------------------------ #
    def set_names(self, *names: Optional[str]) -> "Tensor":
        """Label this tensor's dimensions in place (metadata only, no new
        graph node). One entry per dimension; None leaves a dim unnamed.
        Returns self, chainable: ``Tensor(x).set_names("batch", "feature")``.
        Pass a single None to clear all names."""
        if len(names) == 1 and names[0] is None:
            self.names = None
            return self
        if len(names) != self.ndim:
            raise ValueError(
                f"set_names got {len(names)} name(s) for a {self.ndim}-D tensor "
                f"of shape {tuple(self.shape)} -- one entry per dimension "
                "(use None for dims you don't want to name).")
        self.names = tuple(names)
        return self

    def axis(self, dim_name: str) -> int:
        """Return the index of the dimension labeled `dim_name`, so reductions
        can address axes by meaning: ``x.sum(axis=x.axis("time"))``."""
        if self.names is None or dim_name not in self.names:
            raise ValueError(
                f"no dimension named {dim_name!r} on this tensor "
                f"(names={self.names}).")
        return self.names.index(dim_name)

    def zero_grad(self) -> None:
        """Reset the accumulated gradient (call between training steps)."""
        self.grad = None

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        grad_note = ", requires_grad=True" if self.requires_grad else ""
        op_note = f", op={self._op.name}" if self._op is not None else ""
        return f"Tensor(shape={tuple(self.shape)}, dtype={self.dtype}{grad_note}{op_note})"

    # ------------------------------------------------------------------ #
    # Backpropagation
    # ------------------------------------------------------------------ #
    def _topo_order(self) -> List["Tensor"]:
        """Iterative post-order DFS over the recorded graph: returns every
        reachable tensor with parents ordered before children."""
        order: List[Tensor] = []
        visited = set()
        stack: List[Tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._parents:
                if id(parent) not in visited:
                    stack.append((parent, False))
        return order

    def backward(self, grad: Optional[Any] = None) -> None:
        """Backpropagate from this tensor through the recorded graph,
        accumulating into ``.grad`` of every reachable tensor that has
        ``requires_grad=True``.

        grad : array-like, optional
            Gradient of the final objective w.r.t. this tensor. May be
            omitted only for a size-1 tensor (a scalar loss), where it
            defaults to 1 -- same rule as PyTorch's ``backward()``.
        """
        if grad is None:
            if self.size != 1:
                raise ValueError(
                    "backward() without an explicit gradient argument is only "
                    f"defined for a scalar (size-1) tensor; this tensor has shape "
                    f"{tuple(self.shape)}. Pass grad= explicitly (an array of the "
                    "same shape) or reduce to a scalar loss first (e.g. .sum())."
                )
            grad = np.ones_like(self.data)
        grad = _coerce(grad)

        grads = {id(self): grad}
        for node in reversed(self._topo_order()):
            node_grad = grads.pop(id(node), None)
            if node_grad is None:
                continue
            if node.requires_grad:
                node.grad = node_grad if node.grad is None else node.grad + node_grad
            if node._backward_fn is None:
                continue
            parent_grads = node._backward_fn(node_grad)
            for parent, parent_grad in zip(node._parents, parent_grads):
                if parent_grad is None:
                    continue
                # Only route gradient toward subgraphs that can use it.
                if not (parent.requires_grad or parent._backward_fn is not None):
                    continue
                key = id(parent)
                grads[key] = parent_grad if key not in grads else grads[key] + parent_grad


def as_tensor(x: Any) -> Tensor:
    """Return ``x`` as a Tensor, wrapping (by reference, never copying) if
    it isn't one already."""
    return x if isinstance(x, Tensor) else Tensor(x)
