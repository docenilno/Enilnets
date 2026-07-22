"""The dataset abstraction the data pipeline sits on (roadmap items 50/53).

Two kinds, mirroring the only distinction a loader cares about:
**map-style** (:class:`Dataset`) knows its length and can fetch any index,
so it is shufflable and splittable; **iterable-style**
(:class:`IterableDataset`) produces samples in order with no known length,
and is shuffled through a bounded buffer since there is nothing to permute.

A sample is ``(x, y)``, or just ``x`` when unlabelled. Anything with
``__len__`` and ``__getitem__`` works as a map-style dataset."""

from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence

from ..core.backend import np


def _take(arr: Any, indices: Any) -> Any:
    """Gather rows `indices` from `arr`, keeping the index array on the same
    device as the data. A host index array cannot index a device array, and
    the reverse forces a transfer -- so match rather than assume."""
    if type(arr).__module__.split(".")[0] == "cupy":
        import cupy
        return arr[cupy.asarray(indices)]
    import numpy as _host_np
    return arr[_host_np.asarray(indices)]


class Dataset:
    """Map-style dataset: ``__len__`` plus ``__getitem__(i) -> sample``."""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def get_batch(self, indices: Sequence[int]) -> Any:
        """Fetch and stack a whole batch at once, already collated.

        The default is the per-sample loop. Datasets backed by an array
        override it with a single vectorized gather, which is ~17x faster
        than fetching 128 rows individually and stacking them -- the
        difference between the DataLoader being a convenience and being a
        cost. Only used when there is no per-sample transform and no worker
        pool, since both of those are inherently per-sample."""
        from .loader import default_collate
        return default_collate([self[int(i)] for i in indices])

    def map(self, fn: Callable[[Any], Any]) -> "Dataset":
        """A lazy view applying `fn` to each sample as it is fetched."""
        return MappedDataset(self, fn)

    def subset(self, indices: Sequence[int]) -> "Dataset":
        return Subset(self, indices)

    def __add__(self, other: "Dataset") -> "Dataset":
        return ConcatDataset([self, other])


class IterableDataset:
    """Iterable-style dataset: ``__iter__`` yielding samples, no length.

    Implement ``__iter__`` so it can be called more than once (one epoch per
    call) -- returning a single exhausted generator makes the second epoch
    silently empty."""

    def __iter__(self) -> Iterator[Any]:
        raise NotImplementedError

    def map(self, fn: Callable[[Any], Any]) -> "IterableDataset":
        return MappedIterableDataset(self, fn)


class ArrayDataset(Dataset):
    """Wraps in-memory arrays. ``ArrayDataset(X)`` yields ``x``;
    ``ArrayDataset(X, Y)`` yields ``(x, y)``. Arrays are held by reference,
    never copied."""

    def __init__(self, X: Any, Y: Optional[Any] = None) -> None:
        self.X = X
        self.Y = Y
        if Y is not None and len(X) != len(Y):
            raise ValueError(
                f"X and Y must have the same length, got {len(X)} and {len(Y)}")

    def __len__(self) -> int:
        return int(len(self.X))

    def __getitem__(self, index: int) -> Any:
        return self.X[index] if self.Y is None else (self.X[index], self.Y[index])

    def get_batch(self, indices: Sequence[int]) -> Any:
        x = _take(self.X, indices)
        return x if self.Y is None else (x, _take(self.Y, indices))


class MemmapDataset(Dataset):
    """A dataset backed by an on-disk array via ``np.memmap`` -- rows are
    paged in as they are touched, so the file may be far larger than RAM.

    `shape` is the full array shape including the leading sample axis.
    `y_path` (with `y_shape`/`y_dtype`) optionally memory-maps labels too.

    Always reads through host NumPy: memory mapping is a filesystem
    facility, and CuPy has no equivalent. Batches are moved to the active
    backend by the loader, not here."""

    def __init__(self, path: str, shape: Sequence[int], dtype: Any = "float32",
                 y_path: Optional[str] = None, y_shape: Optional[Sequence[int]] = None,
                 y_dtype: Any = "float32", offset: int = 0) -> None:
        import numpy as _host_np
        self.X = _host_np.memmap(path, dtype=dtype, mode="r",
                                 shape=tuple(shape), offset=offset)
        self.Y = None
        if y_path is not None:
            if y_shape is None:
                raise ValueError("y_shape is required when y_path is given")
            self.Y = _host_np.memmap(y_path, dtype=y_dtype, mode="r",
                                     shape=tuple(y_shape))
            if len(self.Y) != len(self.X):
                raise ValueError(
                    f"X and Y memmaps must have the same length, got "
                    f"{len(self.X)} and {len(self.Y)}")

    def __len__(self) -> int:
        return int(len(self.X))

    def __getitem__(self, index: int) -> Any:
        # np.asarray materializes the page(s) actually touched; without it a
        # memmap slice would keep the mapping alive inside every batch.
        x = np.asarray(self.X[index])
        return x if self.Y is None else (x, np.asarray(self.Y[index]))

    def get_batch(self, indices: Sequence[int]) -> Any:
        import numpy as _host_np
        idx = _host_np.asarray(indices)
        # One gather touches each needed page once; the per-sample loop would
        # re-enter the mapping for every row.
        x = np.asarray(self.X[idx])
        return x if self.Y is None else (x, np.asarray(self.Y[idx]))


class StreamingDataset(IterableDataset):
    """Wraps a *factory* returning a fresh iterator over samples.

    The factory, rather than an iterator, is what makes multiple epochs
    work: each ``__iter__`` calls it again. Passing a generator directly
    would give one epoch of data and silently nothing thereafter, so that
    is rejected."""

    def __init__(self, factory: Callable[[], Iterable[Any]]) -> None:
        if not callable(factory):
            raise TypeError(
                "StreamingDataset takes a CALLABLE returning a fresh iterable "
                "(e.g. `lambda: read_records(path)`), not an iterator -- an "
                "iterator would be exhausted after the first epoch.")
        self.factory = factory

    def __iter__(self) -> Iterator[Any]:
        return iter(self.factory())


class MappedDataset(Dataset):
    """Lazy per-sample transform over a map-style dataset."""

    def __init__(self, base: Dataset, fn: Callable[[Any], Any]) -> None:
        self.base, self.fn = base, fn

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Any:
        return self.fn(self.base[index])


class MappedIterableDataset(IterableDataset):
    """Lazy per-sample transform over an iterable-style dataset."""

    def __init__(self, base: IterableDataset, fn: Callable[[Any], Any]) -> None:
        self.base, self.fn = base, fn

    def __iter__(self) -> Iterator[Any]:
        for sample in self.base:
            yield self.fn(sample)


class Subset(Dataset):
    """A view of `base` restricted to `indices`, in that order."""

    def __init__(self, base: Dataset, indices: Sequence[int]) -> None:
        import numpy as _host_np
        self.base = base
        # Held as an array, not a list, so the batch translation below is one
        # vectorized gather rather than a Python loop per sample.
        self.indices = _host_np.asarray(indices, dtype=_host_np.int64)
        n = len(base)
        if self.indices.size and (int(self.indices.min()) < -n
                                  or int(self.indices.max()) >= n):
            raise IndexError(
                f"indices out of range for a dataset of length {n}")

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> Any:
        return self.base[int(self.indices[index])]

    def get_batch(self, indices: Sequence[int]) -> Any:
        import numpy as _host_np
        # Translate through this view, then let the base use its own fast path.
        return self.base.get_batch(self.indices[_host_np.asarray(indices)])


class ConcatDataset(Dataset):
    """Several map-style datasets end to end."""

    def __init__(self, datasets: Sequence[Dataset]) -> None:
        if not datasets:
            raise ValueError("ConcatDataset needs at least one dataset")
        self.datasets = list(datasets)
        self.offsets: List[int] = []
        total = 0
        for d in self.datasets:
            self.offsets.append(total)
            total += len(d)
        self.total = total

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int) -> Any:
        if index < 0:
            index += self.total
        if not 0 <= index < self.total:
            raise IndexError(f"index {index} out of range for {self.total} samples")
        # Linear scan: the dataset count is small (a handful), unlike the
        # sample count, so bisect would be ceremony without a payoff.
        for d, start in zip(reversed(self.datasets), reversed(self.offsets)):
            if index >= start:
                return d[index - start]
        raise AssertionError("unreachable")


def random_split(dataset: Dataset, lengths: Sequence[int],
                 seed: Optional[int] = None) -> List[Dataset]:
    """Split into disjoint :class:`Subset`s of the given lengths, which must
    sum to ``len(dataset)``. Fractions in (0, 1) are accepted instead and
    resolved to counts, with any rounding remainder given to the last part."""
    n = len(dataset)
    if lengths and all(0 < float(v) < 1 for v in lengths):
        counts = [int(n * float(v)) for v in lengths]
        counts[-1] += n - sum(counts)
        lengths = counts
    lengths = [int(v) for v in lengths]
    if sum(lengths) != n:
        raise ValueError(
            f"split lengths {lengths} sum to {sum(lengths)}, but the dataset "
            f"has {n} samples")
    import numpy as _host_np
    rng = _host_np.random.RandomState(seed) if seed is not None else _host_np.random
    order = rng.permutation(n)
    out, start = [], 0
    for size in lengths:
        out.append(Subset(dataset, order[start:start + size]))
        start += size
    return out


def as_dataset(data: Any, Y: Optional[Any] = None) -> Any:
    """Coerce whatever a caller passed into a dataset.

    Already-a-dataset passes through; arrays become an :class:`ArrayDataset`;
    a callable becomes a :class:`StreamingDataset`. This is what lets
    ``DataLoader(X, Y)`` and ``DataLoader(dataset)`` be the same call."""
    if isinstance(data, (Dataset, IterableDataset)):
        if Y is not None:
            raise ValueError("Y cannot be given alongside a Dataset")
        return data
    if callable(data):
        if Y is not None:
            raise ValueError("Y cannot be given alongside a streaming factory")
        return StreamingDataset(data)
    if hasattr(data, "__len__") and hasattr(data, "__getitem__") and Y is None \
            and not hasattr(data, "shape"):
        return data                                   # duck-typed map-style
    return ArrayDataset(data, Y)
