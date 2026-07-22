"""Symbolic graph tracing: capture the op graph a computation builds,
independent of running it again -- the basis for graph optimization
(optimize.py) and for computational-graph visualization.

- :func:`trace` -- export the graph already recorded behind an output
  Tensor.
- :func:`symbolic_trace` -- run a function once on example inputs, with
  those inputs marked as placeholders, and return a re-runnable
  :class:`Graph`."""

import contextlib
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core import backend
from .tensor import Tensor, _GradMode
from .ops import Op


class Node:
    """One vertex of a traced graph.

    ``kind`` is one of:
      - ``"placeholder"`` -- a graph input, fed at ``Graph.run`` time
      - ``"constant"``    -- a captured array (weights, literals)
      - ``"op"``          -- an operation applied to parent nodes
    """

    def __init__(self, node_id: int, kind: str, op: Optional[Op] = None,
                 kwargs: Optional[dict] = None, parents: Tuple[int, ...] = (),
                 value: Any = None, shape: Optional[Tuple[int, ...]] = None,
                 dtype: Any = None, name: Optional[str] = None) -> None:
        self.id = node_id
        self.kind = kind
        self.op = op
        self.kwargs = kwargs or {}
        self.parents = parents
        self.value = value          # only for constants
        self.shape = shape
        self.dtype = dtype
        self.name = name

    @property
    def op_name(self) -> str:
        if self.kind == "op":
            return self.op.name
        return self.kind

    def __repr__(self) -> str:
        return (f"Node({self.id}, {self.op_name}, parents={list(self.parents)}, "
                f"shape={self.shape})")


class Graph:
    """A traced computation: nodes in topological order plus the output node.

    Purely structural -- introspect it (``nodes``, ``__str__``), transform
    it (``Enilnets.graph.optimize``), or re-execute it on new inputs
    (``run``). Constants hold references to the original arrays (weights),
    so re-running after a training step sees updated weights."""

    def __init__(self, nodes: List[Node], output_id: int,
                 placeholder_ids: List[int]) -> None:
        self.nodes = nodes
        self.output_id = output_id
        self.placeholder_ids = placeholder_ids

    def node(self, node_id: int) -> Node:
        return self._by_id[node_id]

    @property
    def _by_id(self) -> Dict[int, Node]:
        return {n.id: n for n in self.nodes}

    def run(self, *inputs: Any) -> Any:
        """Execute the graph on raw arrays/Tensors, returning a raw array.
        Inputs map positionally onto the graph's placeholders."""
        if len(inputs) != len(self.placeholder_ids):
            raise ValueError(
                f"graph has {len(self.placeholder_ids)} placeholder(s) but "
                f"{len(inputs)} input(s) were given")
        values: Dict[int, Any] = {}
        for pid, x in zip(self.placeholder_ids, inputs):
            values[pid] = x.data if isinstance(x, Tensor) else x
        for node in self.nodes:
            if node.kind == "placeholder":
                if node.id not in values:
                    raise ValueError(f"no input bound for placeholder node {node.id}")
            elif node.kind == "constant":
                values[node.id] = node.value
            else:
                args = [values[p] for p in node.parents]
                values[node.id] = node.op.forward(*args, **node.kwargs)
        return values[self.output_id]

    def __len__(self) -> int:
        return len(self.nodes)

    def __str__(self) -> str:
        lines = ["id   kind         op/name              parents          shape",
                 "-" * 72]
        for n in self.nodes:
            label = n.op_name + (f" ({n.name})" if n.name else "")
            lines.append(f"{n.id:<4} {n.kind:<12} {label:<20} "
                         f"{str(list(n.parents)):<16} {n.shape}")
        return "\n".join(lines)


@contextlib.contextmanager
def _trace_mode():
    """Force op recording even for tensors that don't require gradients, so
    inference-only computations still leave a traceable graph."""
    prev = _GradMode.tracing
    _GradMode.tracing = True
    try:
        yield
    finally:
        _GradMode.tracing = prev


def trace(output: Tensor, placeholders: Tuple[Tensor, ...] = ()) -> Graph:
    """Export the graph recorded behind `output` as a :class:`Graph`.

    Tensors listed in `placeholders` become graph inputs; every other leaf
    (weights, literals) is captured as a constant holding a reference to
    its array. The computation must have been recorded (run under autograd
    or inside :func:`symbolic_trace`)."""
    placeholder_set = {id(t) for t in placeholders}
    nodes: List[Node] = []
    ids: Dict[int, int] = {}         # id(tensor) -> node id
    placeholder_ids: Dict[int, int] = {}  # id(tensor) -> node id, insertion-ordered

    for tensor in output._topo_order():
        node_id = len(nodes)
        ids[id(tensor)] = node_id
        shape, dtype = tuple(tensor.shape), tensor.dtype
        if id(tensor) in placeholder_set:
            nodes.append(Node(node_id, "placeholder", shape=shape, dtype=dtype,
                              name=tensor.name))
            placeholder_ids[id(tensor)] = node_id
        elif tensor._op is None:
            nodes.append(Node(node_id, "constant", value=tensor.data,
                              shape=shape, dtype=dtype, name=tensor.name))
        else:
            nodes.append(Node(node_id, "op", op=tensor._op,
                              kwargs=dict(tensor._kwargs),
                              parents=tuple(ids[id(p)] for p in tensor._parents),
                              shape=shape, dtype=dtype, name=tensor.name))
    # A placeholder the output never consumed still belongs to the graph --
    # dropping it would silently change run()'s input signature.
    for tensor in placeholders:
        if id(tensor) not in placeholder_ids:
            node_id = len(nodes)
            nodes.append(Node(node_id, "placeholder", shape=tuple(tensor.shape),
                              dtype=tensor.dtype, name=tensor.name))
            placeholder_ids[id(tensor)] = node_id
    # Order graph inputs by the caller-supplied placeholder order, not
    # graph-discovery order.
    ordered = [placeholder_ids[id(t)] for t in placeholders]
    return Graph(nodes, ids[id(output)], ordered)


def symbolic_trace(fn: Callable, *example_inputs: Any) -> Graph:
    """Run `fn` once on example inputs and capture its op graph.

    The example inputs (arrays or Tensors) are marked as placeholders; the
    returned Graph re-executes on fresh inputs via ``graph.run(...)``.
    `fn` must be a pure array computation built from graph ops -- Python
    control flow is baked in as traced (one branch, unrolled loops), the
    standard trade-off of trace-based export."""
    inputs = tuple(x if isinstance(x, Tensor) else Tensor(x) for x in example_inputs)
    with _trace_mode():
        output = fn(*inputs)
    if not isinstance(output, Tensor):
        raise TypeError(
            f"symbolic_trace expects fn to return a Tensor; got {type(output).__name__}")
    return trace(output, placeholders=inputs)
