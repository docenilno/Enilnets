"""DataLoader (roadmap items 50 and 52).

Batches a dataset, optionally shuffling, transforming and prefetching. The
iteration contract is deliberately identical to ``iterate_minibatches``'s
-- ``for xb, yb in loader`` -- so it drops straight into existing training
loops, and ``Train()`` accepts either.
"""

import queue
import threading
from typing import Any, Callable, Iterator, List, Optional, Sequence

from ..core.backend import np
from ..core import backend
from .dataset import IterableDataset, as_dataset


def default_collate(samples: Sequence[Any]) -> Any:
    """Stack a list of samples into a batch.

    ``(x, y)`` samples become ``(X_batch, Y_batch)``; bare ``x`` samples
    become one array. Tuples of any width are supported so a dataset can
    carry extra per-sample data (masks, weights, ids) without a custom
    collate."""
    first = samples[0]
    if isinstance(first, tuple):
        width = len(first)
        return tuple(np.stack([np.asarray(s[i]) for s in samples])
                     for i in range(width))
    return np.stack([np.asarray(s) for s in samples])


class DataLoader:
    """Iterate a dataset in batches.

        Accepts a Dataset, an ``(X, Y)`` pair, a bare ``X``, or a callable
        returning a fresh iterator (see ``as_dataset``). Yields whatever
        ``collate`` produces -- by default ``(X_batch, Y_batch)``, or one array
        for unlabelled data. See the README for the worker-backend measurements.

        shuffle: reshuffled every epoch, deterministically from `seed`. Ignored
            for iterable-style datasets; use `shuffle_buffer` there.
        drop_last: discard a final short batch.
        transform: applied to the collated BATCH. Per-SAMPLE transforms belong
            on the dataset via ``.map()``.
        num_workers / worker_backend: "thread" (default) helps whenever
            ``__getitem__`` releases the GIL and hurts when it does not;
            "process" helps either way but needs a picklable dataset and is
            refused under GPU mode.
        prefetch: keep this many batches ready on a background thread.
        shuffle_buffer: reservoir size for shuffling an iterable dataset."""

    def __init__(self, data: Any, Y: Optional[Any] = None, batch_size: int = 32,
                 shuffle: bool = True, seed: Optional[int] = None,
                 drop_last: bool = False,
                 transform: Optional[Callable[[Any], Any]] = None,
                 collate: Optional[Callable[[Sequence[Any]], Any]] = None,
                 num_workers: int = 0, worker_backend: str = "thread",
                 prefetch: int = 0, shuffle_buffer: int = 0) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if num_workers < 0 or prefetch < 0 or shuffle_buffer < 0:
            raise ValueError("num_workers, prefetch and shuffle_buffer must be >= 0")
        self.dataset = as_dataset(data, Y)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.transform = transform
        self.collate = collate or default_collate
        if worker_backend not in ("thread", "process"):
            raise ValueError(
                f"worker_backend must be 'thread' or 'process', got "
                f"{worker_backend!r}")
        self.num_workers = int(num_workers)
        self.worker_backend = worker_backend
        if self.num_workers > 0 and worker_backend == "process":
            self._check_process_backend()
        self.prefetch = int(prefetch)
        self.shuffle_buffer = int(shuffle_buffer)
        self._epoch = 0
        self._pool = None

    # -- introspection ----------------------------------------------------

    @property
    def is_iterable_style(self) -> bool:
        return isinstance(self.dataset, IterableDataset) or \
            not hasattr(self.dataset, "__len__")

    def __len__(self) -> int:
        """Number of batches per epoch. Undefined for iterable datasets."""
        if self.is_iterable_style:
            raise TypeError(
                "an iterable-style dataset has no length, so neither does its "
                "DataLoader -- iterate it instead of asking for len()")
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    # -- iteration --------------------------------------------------------

    def _order(self) -> Any:
        import numpy as _host_np
        n = len(self.dataset)
        if not self.shuffle:
            return _host_np.arange(n)
        if self.seed is None:
            return _host_np.random.permutation(n)
        # Fold the epoch into the seed: reproducible overall, but a different
        # order each epoch rather than the same one forever.
        return _host_np.random.RandomState(self.seed + self._epoch).permutation(n)

    def _batch(self, indices: Sequence[int]) -> Any:
        """Collated batch for `indices`, via the dataset's vectorized gather
        when one applies. Workers and a custom collate both rule it out:
        the first wants per-sample parallelism, the second expects a list."""
        if self.num_workers == 0 and self.collate is default_collate:
            fast = getattr(self.dataset, "get_batch", None)
            if fast is not None:
                return fast(indices)
        return self.collate(self._fetch(indices))

    def _fetch(self, indices: Sequence[int]) -> List[Any]:
        if self.num_workers > 0:
            pool = self._ensure_pool()
            return list(pool.map(self.dataset.__getitem__, indices))
        return [self.dataset[i] for i in indices]

    def _check_process_backend(self) -> None:
        """Fail at construction, not mid-epoch, if processes cannot work here.

        Both conditions are real rather than defensive: device arrays have no
        meaning in another process, and an unpicklable dataset would raise
        from inside a worker with a traceback pointing nowhere useful."""
        if backend.is_gpu_enabled():
            raise ValueError(
                "worker_backend='process' cannot be used in GPU mode: device "
                "arrays do not cross process boundaries. Use "
                "worker_backend='thread', which is what helps for GPU work "
                "anyway since the transfer releases the GIL.")
        import pickle
        try:
            pickle.dumps(self.dataset)
        except Exception as exc:
            raise ValueError(
                "worker_backend='process' needs a picklable dataset, and this "
                f"one is not ({exc}). Lambdas and locally-defined classes are "
                "the usual cause -- use a module-level class, or "
                "worker_backend='thread'.") from exc

    def _ensure_pool(self):
        if self._pool is None:
            if self.worker_backend == "process":
                from concurrent.futures import ProcessPoolExecutor
                self._pool = ProcessPoolExecutor(max_workers=self.num_workers)
            else:
                from concurrent.futures import ThreadPoolExecutor
                self._pool = ThreadPoolExecutor(max_workers=self.num_workers)
        return self._pool

    def _finish(self, samples: List[Any]) -> Any:
        return self._post(self.collate(samples))

    def _post(self, batch: Any) -> Any:
        return self.transform(batch) if self.transform is not None else batch

    def _map_style_batches(self) -> Iterator[Any]:
        order = self._order()
        n = len(order)
        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size]
            if self.drop_last and len(idx) < self.batch_size:
                break
            yield self._post(self._batch(idx))

    def _iterable_batches(self) -> Iterator[Any]:
        import numpy as _host_np
        rng = (_host_np.random.RandomState(self.seed + self._epoch)
               if self.seed is not None else _host_np.random)
        buffer: List[Any] = []
        batch: List[Any] = []

        def emit(sample):
            batch.append(sample)
            if len(batch) == self.batch_size:
                out = self._finish(list(batch))
                batch.clear()
                return out
            return None

        for sample in self.dataset:
            if self.shuffle_buffer > 0:
                # Classic shuffle buffer: fill to capacity, then for every new
                # sample emit a random resident and put the newcomer in its
                # slot. Appending first and then swapping would duplicate the
                # newcomer and drop whatever it displaced.
                if len(buffer) < self.shuffle_buffer:
                    buffer.append(sample)
                    continue
                j = int(rng.randint(len(buffer)))
                sample, buffer[j] = buffer[j], sample
            out = emit(sample)
            if out is not None:
                yield out
        if self.shuffle_buffer > 0:
            rng.shuffle(buffer)
            for sample in buffer:
                out = emit(sample)
                if out is not None:
                    yield out
        if batch and not self.drop_last:
            yield self._finish(batch)

    def __iter__(self) -> Iterator[Any]:
        base = (self._iterable_batches() if self.is_iterable_style
                else self._map_style_batches())
        self._epoch += 1
        if self.prefetch <= 0:
            return base
        return _prefetched(base, self.prefetch)

    def close(self) -> None:
        """Shut the worker pool down. Worth calling explicitly for the
        process backend, whose workers are real OS processes."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


def _prefetched(source: Iterator[Any], depth: int) -> Iterator[Any]:
    """Run `source` on a background thread, keeping `depth` results ready.

    Worth it when producing a batch does real array work (NumPy releases the
    GIL for that) so it overlaps with the caller's compute. A sentinel
    carries both normal exhaustion and any exception, so a failure in the
    producer surfaces in the consumer rather than vanishing into a dead
    thread."""
    done = object()
    q: "queue.Queue[Any]" = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def produce():
        try:
            for item in source:
                while not stop.is_set():
                    try:
                        q.put(item, timeout=0.05)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    break
        except BaseException as exc:                 # noqa: BLE001 -- re-raised below
            q.put(exc)
        else:
            q.put(done)

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    try:
        while True:
            item = q.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
