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

from typing import Any

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


def gpu_available() -> bool:
    """Return True if CuPy is importable and a CUDA-capable GPU is visible."""
    if _cupy is None:
        return False
    try:
        return _cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _driver_sees_a_gpu() -> bool:
    """Ask the CUDA *driver* API (libcuda) whether any device exists,
    bypassing the CUDA runtime that :func:`gpu_available` goes through."""
    import ctypes
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        if lib.cuInit(0) != 0:
            return False
        count = ctypes.c_int(0)
        return lib.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value > 0
    except Exception:
        return False


def _no_device_message() -> str:
    """Explain WHY no device is usable. The common and confusing case on
    rolling-release distros is a driver package upgraded without a reboot:
    the running kernel module no longer matches the new userspace
    libraries, so the CUDA runtime reports no device even though
    ``nvidia-smi`` still works and the driver API still enumerates the
    GPU. Detecting that split is worth the extra probe -- the bare
    "no GPU detected" reading is actively misleading there."""
    if _driver_sees_a_gpu():
        return (
            "The CUDA driver can see a GPU but the CUDA runtime cannot use it. "
            "This almost always means the NVIDIA driver was upgraded without "
            "rebooting: the running kernel module and the installed userspace "
            "libraries are different versions. `nvidia-smi` keeps working in "
            "this state, so it is not evidence to the contrary. Reboot (or "
            "reload the nvidia kernel modules) and try again."
        )
    return (
        "No CUDA-capable GPU detected: neither the CUDA runtime nor the CUDA "
        "driver reports a device. Check that a GPU is present, the NVIDIA "
        "driver is installed and loaded (`nvidia-smi`), and that "
        "CUDA_VISIBLE_DEVICES is not restricting it."
    )


def use_gpu(enabled: bool = True) -> None:
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
            raise RuntimeError(_no_device_message())
        _State.active = _cupy
        _State.is_gpu = True
    else:
        _State.active = _numpy
        _State.is_gpu = False


def is_gpu_enabled() -> bool:
    """Return True if GPU (CuPy) mode is currently active."""
    return _State.is_gpu


def use_float64(enabled: bool = True) -> None:
    """Switch the default working precision between float32 (default) and
    float64. Call once before building any models -- all models built in a
    process share this one global default dtype."""
    _State.default_dtype = _numpy.float64 if enabled else _numpy.float32


def is_float64_enabled() -> bool:
    """Return True if float64 is currently the default working precision."""
    return _State.default_dtype == _numpy.float64


def default_dtype() -> type:
    """Return the currently active default working precision (float32
    unless use_float64(True) was called)."""
    return _State.default_dtype


def is_array(obj: Any) -> bool:
    """Return True if obj is a NumPy or CuPy ndarray, regardless of which
    backend is currently active."""
    if isinstance(obj, _numpy.ndarray):
        return True
    return _cupy is not None and isinstance(obj, _cupy.ndarray)


def to_numpy(arr: Any) -> "_numpy.ndarray":
    """Return a plain host-side NumPy array, transferring off the GPU if
    needed."""
    if _cupy is not None and isinstance(arr, _cupy.ndarray):
        return arr.get()
    return _numpy.asarray(arr)


def array_module(x: Any) -> Any:
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
