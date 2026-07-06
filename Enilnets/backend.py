"""Dual NumPy/CuPy array backend.

NumPy is always the default and is the only required dependency. CuPy is an
optional GPU backend enabled at runtime via `use_gpu(True)`. All other
Enilnets modules import the array module as `from .backend import np` (or
`from ..backend import np` under `generative/`) instead of `import numpy as
np` — `np` here is a proxy that forwards to whichever backend is currently
active, so no other line in those modules needs to change.

Call `use_gpu(True)` once, before constructing any models — all models
built in a process share this one global backend. Mixing backends across
models in the same process isn't supported.

Reproducibility note: seeded RNG sequences are only guaranteed to be
reproducible within one backend. NumPy's legacy Mersenne Twister and
CuPy's cuRAND-backed generator do not produce identical sequences for the
same seed.

float32 is the default working precision (weights, activations, gradients
-- everywhere a dtype used to be hardcoded to float64). Call
`use_float64(True)` once, before constructing any models, to opt into
float64 instead -- same one-global-switch-before-building-models rule as
`use_gpu`. Code that needs a "the model's working dtype" value calls
`default_dtype()` rather than hardcoding either float type.
"""

import numpy as _numpy

try:
    import cupy as _cupy
except ImportError:
    _cupy = None


class _State:
    active = _numpy
    is_gpu = False
    default_dtype = _numpy.float32


class _ArrayModuleProxy:
    def __getattr__(self, name):
        return getattr(_State.active, name)


np = _ArrayModuleProxy()


def gpu_available():
    """Return True if CuPy is importable and a CUDA-capable GPU is visible."""
    if _cupy is None:
        return False
    try:
        return _cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def use_gpu(enabled=True):
    """Switch the active array backend between NumPy (default) and CuPy.

    Call once before building any models. Raises RuntimeError if enabling
    GPU mode but CuPy isn't installed or no CUDA device is visible.
    """
    if enabled:
        if _cupy is None:
            raise RuntimeError(
                "CuPy is not installed; install a cupy-cudaXXx wheel matching "
                "your CUDA version to use GPU mode."
            )
        if not gpu_available():
            raise RuntimeError("No CUDA-capable GPU detected.")
        _State.active = _cupy
        _State.is_gpu = True
    else:
        _State.active = _numpy
        _State.is_gpu = False


def is_gpu_enabled():
    """Return True if GPU (CuPy) mode is currently active."""
    return _State.is_gpu


def use_float64(enabled=True):
    """Switch the default working precision between float32 (default) and
    float64. Call once before building any models -- all models built in a
    process share this one global default dtype."""
    _State.default_dtype = _numpy.float64 if enabled else _numpy.float32


def is_float64_enabled():
    """Return True if float64 is currently the default working precision."""
    return _State.default_dtype == _numpy.float64


def default_dtype():
    """Return the currently active default working precision (float32
    unless use_float64(True) was called)."""
    return _State.default_dtype


def is_array(obj):
    """Return True if obj is a NumPy or CuPy ndarray, regardless of which
    backend is currently active."""
    if isinstance(obj, _numpy.ndarray):
        return True
    return _cupy is not None and isinstance(obj, _cupy.ndarray)


def to_numpy(arr):
    """Return a plain host-side NumPy array, transferring off the GPU if
    needed."""
    if _cupy is not None and isinstance(arr, _cupy.ndarray):
        return arr.get()
    return _numpy.asarray(arr)


def array_module(x):
    """Return the array module (real NumPy or CuPy) that actually owns `x`,
    regardless of which backend is currently globally active.

    Shared elementwise code (activations.py) that's called with arrays from
    a module that deliberately doesn't follow the global switch -- like
    neat.py, which always computes on host NumPy even when GPU mode is on
    elsewhere -- needs this instead of the `np` proxy: `np` would resolve to
    CuPy while `x` is a plain NumPy array, and CuPy's ufuncs reject a bare
    NumPy array argument."""
    if _cupy is not None and isinstance(x, _cupy.ndarray):
        return _cupy
    return _numpy
