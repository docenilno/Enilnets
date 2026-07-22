"""Elementary differentiable operations, and the public API for defining
new ones.

Every op here is built with :func:`custom_op` -- the same two-piece recipe
(a forward formula plus one local-gradient rule) that user code uses. There
is deliberately no privileged internal mechanism: adding an op never means
editing a central dispatch table the way ``nn/backward.py`` grows one
``elif`` per layer type.

Conventions: ``forward(*input_arrays, **kwargs) -> output_array`` works on
raw backend arrays, never Tensors, and must not mutate its inputs.
``backward(grad_out, out, *input_arrays, **kwargs)`` returns one gradient
per input (``None`` where non-differentiable). Broadcasting is allowed
wherever NumPy allows it; gradients are summed back down to each input's
original shape by ``_unbroadcast``."""

import contextlib
import math
from typing import Any, Callable, Optional, Tuple

from ..core.backend import np
from ..core import backend
from .tensor import Tensor, as_tensor, _GradMode


class Op:
    """A named differentiable operation.

    Calling an Op on Tensors (or anything `as_tensor` accepts) runs the
    forward formula on the raw arrays and -- when autograd is enabled and
    any input requires a gradient -- records itself on the output tensor so
    ``backward()`` can later invoke the local-gradient rule. Created via
    :func:`custom_op`; see that function for the forward/backward contract.

    ``elementwise=True`` marks ops whose output element ``i`` depends only
    on input element ``i`` of a single input -- the property the graph
    optimizer's fusion pass (optimize.py) relies on.
    """

    def __init__(self, name: str, forward: Callable, backward: Callable,
                 elementwise: bool = False,
                 backward_per_input: Optional[Tuple[Callable, ...]] = None) -> None:
        self.name = name
        self.forward = forward
        self.backward = backward
        self.elementwise = elementwise
        self.backward_per_input = backward_per_input

    def __repr__(self) -> str:
        return f"Op({self.name})"

    def __call__(self, *inputs: Any, **kwargs: Any) -> Tensor:
        tensors = tuple(as_tensor(x) for x in inputs)
        datas = tuple(t.data for t in tensors)
        out = self.forward(*datas, **kwargs)
        needs_grad = _GradMode.enabled and any(t.requires_grad for t in tensors)
        result = Tensor(out, requires_grad=needs_grad)
        result.names = _propagate_names(self, tensors, kwargs, out.ndim)
        if needs_grad or _GradMode.tracing:
            result._op = self
            result._parents = tensors
            result._kwargs = kwargs
        if needs_grad:
            op = self
            if op.backward_per_input is not None:
                # Lazy per-input rules: skip the (possibly expensive) work of
                # a gradient no input will ever consume -- e.g. matmul's
                # grad-w.r.t.-x when x is raw input data, which costs a full
                # extra matmul per layer per step.
                needs = tuple(t.requires_grad or t._backward_fn is not None
                              for t in tensors)
                result._backward_fn = lambda g: tuple(
                    fn(g, out, *datas, **kwargs) if needed else None
                    for fn, needed in zip(op.backward_per_input, needs))
            else:
                result._backward_fn = lambda g: op.backward(g, out, *datas, **kwargs)
        return result


def custom_op(name: str, forward: Callable, backward: Callable,
              elementwise: bool = False,
              backward_per_input: Optional[Tuple[Callable, ...]] = None) -> Op:
    """Define a new differentiable op -- the graph engine's extension point.

    `forward(*input_arrays, **kwargs) -> array` gets raw backend arrays and
    must not mutate them. `backward(grad_out, out, *input_arrays, **kwargs)`
    returns one gradient (or None) per input; `out` is the forward result, so
    rules can reuse it. Set `elementwise` only if the op really is
    elementwise-on-one-input (it enables fusion). `backward_per_input` is an
    optional tuple of one rule per input, called only for inputs that need a
    gradient; `backward` must still be given and agree with it."""
    return Op(name, forward, backward, elementwise=elementwise)


# ------------------------------------------------------------------------- #
# Named-dimension propagation (metadata layer, roadmap item 25)
# ------------------------------------------------------------------------- #

# Ops whose output dims correspond 1:1 (right-aligned) to their inputs' dims:
# all elementwise/broadcast arithmetic and nonlinearities.
_NAME_ALIGNED_OPS = {
    "add", "sub", "mul", "div", "neg", "pow", "exp", "log", "sqrt",
    "tanh", "sigmoid", "relu", "cast", "conj", "real", "imag", "abs",
}
_NAME_REDUCTION_OPS = {"sum", "mean", "max"}


def _merge_dim_names(tensors: Tuple[Tensor, ...], out_ndim: int) -> Optional[Tuple]:
    """Right-align every input's names against the (broadcast) output rank
    and merge: matching or one-sided names combine; two DIFFERENT names on
    the same dim is exactly the mistake named tensors exist to catch."""
    merged: list = [None] * out_ndim
    for t in tensors:
        if t.names is None:
            continue
        offset = out_ndim - len(t.names)
        for i, dim_name in enumerate(t.names):
            if dim_name is None:
                continue
            slot = offset + i
            if merged[slot] is None:
                merged[slot] = dim_name
            elif merged[slot] != dim_name:
                raise ValueError(
                    f"named-dimension mismatch: dim {slot} is {merged[slot]!r} "
                    f"on one operand and {dim_name!r} on another.")
    return tuple(merged) if any(m is not None for m in merged) else None


def _propagate_names(op: "Op", tensors: Tuple[Tensor, ...], kwargs: dict,
                     out_ndim: int) -> Optional[Tuple]:
    """Conservative name propagation: keep names only where the output dims
    provably correspond to input dims (elementwise/broadcast ops, reductions,
    transpose); everything else (reshape, matmul, indexing, concat) drops
    them rather than risk silently wrong labels."""
    if all(t.names is None for t in tensors):
        return None
    if op.name in _NAME_ALIGNED_OPS:
        return _merge_dim_names(tensors, out_ndim)
    if op.name in _NAME_REDUCTION_OPS:
        names = tensors[0].names
        axis = kwargs.get("axis")
        if names is None:
            return None
        if axis is None:
            return None  # full reduction: no input dim survives meaningfully
        axes = {ax % len(names) for ax in (axis if isinstance(axis, tuple) else (axis,))}
        if kwargs.get("keepdims"):
            return tuple(n if i not in axes else None for i, n in enumerate(names))
        return tuple(n for i, n in enumerate(names) if i not in axes) or None
    if op.name == "transpose":
        names = tensors[0].names
        if names is None:
            return None
        axes = kwargs.get("axes") or tuple(reversed(range(len(names))))
        return tuple(names[ax] for ax in axes)
    return None


# ------------------------------------------------------------------------- #
# Broadcasting helper
# ------------------------------------------------------------------------- #

def _unbroadcast(grad: Any, shape: Tuple[int, ...]) -> Any:
    """Sum `grad` down to `shape`, undoing NumPy broadcasting: leading
    broadcast axes are summed away, size-1 axes are summed with keepdims."""
    if grad.shape == tuple(shape):
        return grad
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    keep = tuple(i for i, n in enumerate(shape) if n == 1 and grad.shape[i] != 1)
    if keep:
        grad = grad.sum(axis=keep, keepdims=True)
    return grad


# ------------------------------------------------------------------------- #
# Arithmetic
# ------------------------------------------------------------------------- #

add = custom_op(
    "add",
    forward=lambda a, b: a + b,
    backward=lambda g, out, a, b: (_unbroadcast(g, a.shape), _unbroadcast(g, b.shape)),
)

sub = custom_op(
    "sub",
    forward=lambda a, b: a - b,
    backward=lambda g, out, a, b: (_unbroadcast(g, a.shape), _unbroadcast(-g, b.shape)),
)

def _conj(x: Any) -> Any:
    """Conjugate iff complex (identity for real arrays -- no copy).

    Gradient convention for complex tensors (matches PyTorch/JAX): the
    stored gradient of a real-valued loss is dL/dRe(z) + 1j*dL/dIm(z), so
    plain gradient descent works unchanged. For a holomorphic op w = f(z)
    the chain rule under that convention is grad_in = g * conj(f'(z)) --
    hence the _conj calls in mul/div/pow/matmul below, which reduce to the
    familiar real rules when nothing is complex."""
    return np.conj(x) if x.dtype.kind == "c" else x

mul = custom_op(
    "mul",
    forward=lambda a, b: a * b,
    backward=lambda g, out, a, b: (_unbroadcast(g * _conj(b), a.shape),
                                   _unbroadcast(g * _conj(a), b.shape)),
)

div = custom_op(
    "div",
    forward=lambda a, b: a / b,
    backward=lambda g, out, a, b: (_unbroadcast(g * _conj(1.0 / b), a.shape),
                                   _unbroadcast(g * _conj(-a / (b * b)), b.shape)),
)

neg = custom_op(
    "neg",
    forward=lambda a: -a,
    backward=lambda g, out, a: (-g,),
    elementwise=True,
)

def _pow_backward(g: Any, out: Any, a: Any, exponent: float = 2.0) -> Tuple[Any]:
    return (g * _conj(exponent * a ** (exponent - 1)),)

power = custom_op(
    "pow",
    forward=lambda a, exponent=2.0: a ** exponent,
    backward=_pow_backward,
    elementwise=True,
)


def _matmul_forward(a: Any, b: Any) -> Any:
    if a.ndim < 2 or b.ndim < 2:
        raise ValueError(
            "graph.matmul supports 2-D and batched N-D operands only; got "
            f"shapes {a.shape} @ {b.shape}. Reshape 1-D vectors to (1, K) or "
            "(K, 1) explicitly."
        )
    return a @ b

# swapaxes handles both plain 2D matmul and batched (...,M,K)@(...,K,N);
# _unbroadcast folds gradient back down when one operand was broadcast
# across batch dimensions. Split per input so the engine can skip either
# side entirely (each is a full matmul) when its input doesn't need a
# gradient -- the single hottest saving in a typical training step, where
# the first layer's input is raw data.
def _matmul_backward_a(g: Any, out: Any, a: Any, b: Any) -> Any:
    return _unbroadcast(g @ _conj(b).swapaxes(-1, -2), a.shape)

def _matmul_backward_b(g: Any, out: Any, a: Any, b: Any) -> Any:
    return _unbroadcast(_conj(a).swapaxes(-1, -2) @ g, b.shape)

def _matmul_backward(g: Any, out: Any, a: Any, b: Any) -> Tuple[Any, Any]:
    return (_matmul_backward_a(g, out, a, b), _matmul_backward_b(g, out, a, b))

_matmul_op = custom_op(
    "matmul",
    forward=_matmul_forward,
    backward=_matmul_backward,
    backward_per_input=(_matmul_backward_a, _matmul_backward_b),
)


# ------------------------------------------------------------------------- #
# Mixed precision (autograd-aware AMP)
# ------------------------------------------------------------------------- #

cast = custom_op(
    "cast",
    forward=lambda a, dtype=None: a.astype(np.dtype(dtype), copy=False),
    # The gradient of a cast is a cast back to the input's dtype -- this is
    # exactly what makes AMP "a graph-level concern": master float64 weights
    # receive float64 gradients even when the compute between the casts ran
    # at float32.
    backward=lambda g, out, a, dtype=None: (g.astype(a.dtype, copy=False),),
    elementwise=True,
)


@contextlib.contextmanager
def autocast():
    """Autograd-aware mixed precision (the ``graph/`` counterpart of
    ``NeuralNet(use_mixed_precision=True)``).

    Inside the block, ``matmul`` (the BLAS hot path) runs at float32 even
    when the library-wide default is float64, by inserting :data:`cast` ops
    into the recorded graph -- so the downcast is visible when tracing, and
    gradients flow back through the casts to reach master weights at their
    own full precision. A no-op when float32 is already the default
    (nothing to downcast), same rule as the ``nn/`` flag."""
    prev = _GradMode.autocast
    _GradMode.autocast = True
    try:
        yield
    finally:
        _GradMode.autocast = prev


def matmul(a: Any, b: Any) -> Tensor:
    """Matrix multiply ``a @ b`` (2-D or batched N-D).

    Under :func:`autocast` (and a float64 default precision), the operands
    are cast to float32 for the multiply and the result cast back --
    inserted as real graph ops with defined gradients."""
    if _GradMode.autocast and backend.is_float64_enabled():
        a32 = cast(a, dtype="float32")
        b32 = cast(b, dtype="float32")
        return cast(_matmul_op(a32, b32), dtype="float64")
    return _matmul_op(a, b)


# ------------------------------------------------------------------------- #
# Elementwise nonlinearities
# ------------------------------------------------------------------------- #

exp = custom_op(
    "exp",
    forward=lambda a: np.exp(a),
    backward=lambda g, out, a: (g * _conj(out),),
    elementwise=True,
)

log = custom_op(
    "log",
    forward=lambda a: np.log(a),
    backward=lambda g, out, a: (g * _conj(1.0 / a),),
    elementwise=True,
)

sqrt = custom_op(
    "sqrt",
    forward=lambda a: np.sqrt(a),
    backward=lambda g, out, a: (g * _conj(0.5 / out),),
    elementwise=True,
)

tanh = custom_op(
    "tanh",
    forward=lambda a: np.tanh(a),
    backward=lambda g, out, a: (g * _conj(1 - out * out),),
    elementwise=True,
)

def _sigmoid_forward(a: Any) -> Any:
    # Numerically stable split by sign (same approach as nn/activations.py).
    out = np.empty_like(a)
    pos = a >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-a[pos]))
    ea = np.exp(a[~pos])
    out[~pos] = ea / (1.0 + ea)
    return out

sigmoid = custom_op(
    "sigmoid",
    forward=_sigmoid_forward,
    backward=lambda g, out, a: (g * out * (1 - out),),
    elementwise=True,
)

relu = custom_op(
    "relu",
    forward=lambda a: np.maximum(a, 0),
    backward=lambda g, out, a: (g * (a > 0),),
    elementwise=True,
)


# ------------------------------------------------------------------------- #
# Complex-specific ops (roadmap item 26). Gradients follow the stored-
# gradient convention documented on _conj: verified against real/imaginary-
# part finite differences in the test suite.
# ------------------------------------------------------------------------- #

conj = custom_op(
    "conj",
    forward=lambda a: np.conj(a),
    backward=lambda g, out, a: (np.conj(g),),
    elementwise=True,
)

real = custom_op(
    "real",
    # +0 forces a real-dtype COPY (np.real alone returns a view into the
    # complex array's real parts, which the no-mutation convention forbids
    # handing out).
    forward=lambda a: np.real(a) + 0,
    backward=lambda g, out, a: (g.astype(a.dtype) if a.dtype.kind == "c" else g,),
    elementwise=True,
)

imag = custom_op(
    "imag",
    forward=lambda a: np.imag(a) + 0,
    backward=lambda g, out, a: ((1j * g).astype(a.dtype) if a.dtype.kind == "c"
                                else np.zeros_like(a),),
    elementwise=True,
)

absolute = custom_op(
    "abs",
    forward=lambda a: np.abs(a),
    # Real: sign(a). Complex: z/|z| (the gradient of |z| w.r.t. (x, y),
    # packed per the stored-gradient convention).
    backward=lambda g, out, a: ((g * a / np.maximum(out, 1e-30)
                                 if a.dtype.kind == "c" else g * np.sign(a)),),
    elementwise=True,
)


# ------------------------------------------------------------------------- #
# Shape / indexing
# ------------------------------------------------------------------------- #

reshape = custom_op(
    "reshape",
    forward=lambda a, shape=None: a.reshape(shape),
    backward=lambda g, out, a, shape=None: (g.reshape(a.shape),),
)

def _transpose_backward(g: Any, out: Any, a: Any, axes: Optional[Tuple[int, ...]] = None) -> Tuple[Any]:
    if axes is None:
        return (g.transpose(),)
    inverse = [0] * len(axes)
    for i, ax in enumerate(axes):
        inverse[ax] = i
    return (g.transpose(inverse),)

transpose = custom_op(
    "transpose",
    forward=lambda a, axes=None: a.transpose(axes) if axes is not None else a.transpose(),
    backward=_transpose_backward,
)


def _is_int_array(x: Any) -> bool:
    return backend.is_array(x) and x.dtype.kind in "iu"


def _getitem_backward(g: Any, out: Any, a: Any, index: Any = None) -> Tuple[Any]:
    grad = np.zeros_like(a)
    if np.iscomplexobj(g) and not np.iscomplexobj(grad):
        # A REAL tensor feeding a complex-valued computation (a real signal
        # into an STFT, say) still has a purely real gradient: under the
        # conjugate-Wirtinger convention the imaginary parts cancel, since
        # there is no imaginary component to differentiate against. Taking
        # .real is therefore exact -- done explicitly here so it reads as a
        # decision rather than as NumPy's ComplexWarning about a silent cast.
        g = np.real(g)
    gather = _is_int_array(index) or (
        isinstance(index, tuple) and any(_is_int_array(i) for i in index))
    if gather:
        # Integer-array (gather-style) indexing -- plain or tuple/advanced:
        # duplicate indices must ACCUMULATE (the embedding-gradient
        # situation, and im2col-style overlapping-window gathers), so this
        # must be a scatter-add, never fancy-index assignment
        # (last-write-wins).
        np.add.at(grad, index, g)
    else:
        # Basic slicing never repeats an element, so += is safe.
        grad[index] += g
    return (grad,)

getitem = custom_op(
    "getitem",
    forward=lambda a, index=None: a[index],
    backward=_getitem_backward,
)


def _normalize_pad_width(pad_width: Any, ndim: int) -> Tuple[Tuple[int, int], ...]:
    """Accept an int, a single (before, after) pair, or one pair per axis;
    return one (before, after) pair per axis (np.pad's full form)."""
    if isinstance(pad_width, int):
        return ((pad_width, pad_width),) * ndim
    pad_width = tuple(pad_width)
    if len(pad_width) == 2 and all(isinstance(p, int) for p in pad_width):
        return (tuple(pad_width),) * ndim
    if len(pad_width) != ndim:
        raise ValueError(
            f"pad_width has {len(pad_width)} entries for a {ndim}-D input -- "
            "pass an int, one (before, after) pair, or one pair per axis.")
    return tuple((int(b), int(a)) for b, a in pad_width)


def _pad_forward(a: Any, pad_width: Any = 1, mode: str = "constant",
                 constant_value: float = 0.0) -> Any:
    pw = _normalize_pad_width(pad_width, a.ndim)
    if mode == "constant":
        return np.pad(a, pw, mode="constant", constant_values=constant_value)
    if mode not in ("reflect", "edge", "wrap"):
        raise ValueError(
            f"unknown pad mode {mode!r}: use 'constant' (zeros/value), "
            "'reflect', 'edge' (replication), or 'wrap' (circular).")
    return np.pad(a, pw, mode=mode)


def _pad_backward(g: Any, out: Any, a: Any, pad_width: Any = 1,
                  mode: str = "constant", constant_value: float = 0.0) -> Tuple[Any]:
    pw = _normalize_pad_width(pad_width, a.ndim)
    center = tuple(slice(b, b + n) for (b, _), n in zip(pw, a.shape))
    if mode == "constant":
        # Padding cells are constants; only the center passes gradient.
        return (g[center],)
    # reflect/edge/wrap replicate SOURCE elements into the padding, so each
    # source cell's gradient is the sum over every position it was copied
    # to. Padding an index grid with the same mode yields exactly that
    # output->source map; scatter-add finishes the job.
    source = np.pad(np.arange(a.size).reshape(a.shape), pw, mode=mode)
    grad = np.zeros(a.size, dtype=g.dtype)
    np.add.at(grad, source.reshape(-1), g.reshape(-1))
    return (grad.reshape(a.shape),)


pad = custom_op(
    "pad",
    forward=_pad_forward,
    backward=_pad_backward,
)


def _concat_backward(g: Any, out: Any, *arrays: Any, axis: int = 0) -> Tuple[Any, ...]:
    # Split points are plain host-side ints (shapes always are, regardless
    # of backend).
    sizes = [arr.shape[axis] for arr in arrays]
    splits = [sum(sizes[:i + 1]) for i in range(len(sizes) - 1)]
    return tuple(np.split(g, splits, axis=axis))

concatenate = custom_op(
    "concatenate",
    forward=lambda *arrays, axis=0: np.concatenate(arrays, axis=axis),
    backward=_concat_backward,
)


# ------------------------------------------------------------------------- #
# Reductions
# ------------------------------------------------------------------------- #

def _expand_reduced(g: Any, a: Any, axis: Any, keepdims: bool) -> Any:
    """Broadcast a reduced gradient back over the reduced axes."""
    if axis is None:
        return np.broadcast_to(g, a.shape)
    axes = axis if isinstance(axis, tuple) else (axis,)
    if not keepdims:
        for ax in sorted(ax % a.ndim for ax in axes):
            g = np.expand_dims(g, ax)
    return np.broadcast_to(g, a.shape)

sum_ = custom_op(
    "sum",
    forward=lambda a, axis=None, keepdims=False: a.sum(axis=axis, keepdims=keepdims),
    backward=lambda g, out, a, axis=None, keepdims=False:
        (_expand_reduced(g, a, axis, keepdims).copy(),),
)

def _mean_backward(g: Any, out: Any, a: Any, axis: Any = None, keepdims: bool = False) -> Tuple[Any]:
    n = a.size if axis is None else math.prod(
        a.shape[ax % a.ndim] for ax in (axis if isinstance(axis, tuple) else (axis,)))
    return (_expand_reduced(g, a, axis, keepdims) / n,)

mean = custom_op(
    "mean",
    forward=lambda a, axis=None, keepdims=False: a.mean(axis=axis, keepdims=keepdims),
    backward=_mean_backward,
)

def _max_backward(g: Any, out: Any, a: Any, axis: Any = None, keepdims: bool = False) -> Tuple[Any]:
    # Gradient routes to the max position(s); ties split the gradient
    # evenly (matches JAX; simpler and better-behaved than argmax-only).
    expanded_out = _expand_reduced(out, a, axis, keepdims)
    mask = (a == expanded_out).astype(g.dtype)
    counts = mask.sum(axis=axis, keepdims=True) if axis is not None else mask.sum()
    return (_expand_reduced(g, a, axis, keepdims) * mask / counts,)

max_ = custom_op(
    "max",
    forward=lambda a, axis=None, keepdims=False: a.max(axis=axis, keepdims=keepdims),
    backward=_max_backward,
)


# ------------------------------------------------------------------------- #
# Composites (built from primitives -- gradients come for free)
# ------------------------------------------------------------------------- #

def softmax(x: Any, axis: int = -1) -> Tensor:
    """Numerically stable softmax along `axis`, as a composite of primitive
    ops (subtract-max, exp, normalize) so its gradient needs no bespoke rule."""
    x = as_tensor(x)
    shifted = sub(x, max_(x, axis=axis, keepdims=True))
    e = exp(shifted)
    return div(e, sum_(e, axis=axis, keepdims=True))


def log_softmax(x: Any, axis: int = -1) -> Tensor:
    """Numerically stable log-softmax along `axis` (log-sum-exp trick)."""
    x = as_tensor(x)
    shifted = sub(x, max_(x, axis=axis, keepdims=True))
    return sub(shifted, log(sum_(exp(shifted), axis=axis, keepdims=True)))


# ------------------------------------------------------------------------- #
# Operator overloads on Tensor
# ------------------------------------------------------------------------- #

def _install_operators() -> None:
    """Attach arithmetic/indexing sugar to Tensor. Lives here (not in
    tensor.py) so the core engine stays op-agnostic."""
    Tensor.__add__ = lambda self, other: add(self, other)
    Tensor.__radd__ = lambda self, other: add(other, self)
    Tensor.__sub__ = lambda self, other: sub(self, other)
    Tensor.__rsub__ = lambda self, other: sub(other, self)
    Tensor.__mul__ = lambda self, other: mul(self, other)
    Tensor.__rmul__ = lambda self, other: mul(other, self)
    Tensor.__truediv__ = lambda self, other: div(self, other)
    Tensor.__rtruediv__ = lambda self, other: div(other, self)
    Tensor.__neg__ = lambda self: neg(self)
    Tensor.__pow__ = lambda self, exponent: power(self, exponent=float(exponent))
    Tensor.__matmul__ = lambda self, other: matmul(self, other)
    Tensor.__rmatmul__ = lambda self, other: matmul(other, self)
    Tensor.__getitem__ = lambda self, index: getitem(self, index=index)
    Tensor.reshape = lambda self, *shape: reshape(
        self, shape=shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape)
    Tensor.transpose = lambda self, *axes: transpose(
        self, axes=(axes[0] if len(axes) == 1 and isinstance(axes[0], (tuple, list))
                    else (axes or None)))
    Tensor.sum = lambda self, axis=None, keepdims=False: sum_(self, axis=axis, keepdims=keepdims)
    Tensor.mean = lambda self, axis=None, keepdims=False: mean(self, axis=axis, keepdims=keepdims)
    Tensor.max = lambda self, axis=None, keepdims=False: max_(self, axis=axis, keepdims=keepdims)
    Tensor.exp = lambda self: exp(self)
    Tensor.log = lambda self: log(self)
    Tensor.sqrt = lambda self: sqrt(self)
    Tensor.tanh = lambda self: tanh(self)
    Tensor.sigmoid = lambda self: sigmoid(self)
    Tensor.relu = lambda self: relu(self)
    Tensor.conj = lambda self: conj(self)
    Tensor.real = lambda self: real(self)
    Tensor.imag = lambda self: imag(self)
    Tensor.abs = lambda self: absolute(self)
    Tensor.__abs__ = lambda self: absolute(self)


_install_operators()
