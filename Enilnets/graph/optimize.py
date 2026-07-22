"""Graph optimization passes over traced graphs (tracing.py):

- :func:`eliminate_dead_nodes` -- drop nodes whose value never reaches the
  output (e.g. branches abandoned by traced Python control flow).
- :func:`fold_constants` -- precompute ops whose inputs are all constants
  (weights/literals), storing the result as a new constant. Folded values
  are snapshots: re-running the graph after mutating a folded weight will
  not see the change (unfolded constants stay live references).
- :func:`fuse_elementwise` -- collapse chains of single-input elementwise
  ops (relu(tanh(exp(x))), ...) into one fused node, removing per-op
  Python/array-allocation overhead on re-execution.

``optimize(graph)`` applies all three in that order. Passes return new
Graph objects; the input graph is never mutated. These operate on the
*traced, re-runnable* representation -- eager Tensor math is unaffected.
"""

from typing import Any, Dict, List, Set

from .ops import Op, custom_op
from .tracing import Graph, Node


def _live_ids(graph: Graph) -> Set[int]:
    """IDs of nodes whose value can reach the output."""
    live: Set[int] = set()
    stack = [graph.output_id]
    by_id = {n.id: n for n in graph.nodes}
    while stack:
        nid = stack.pop()
        if nid in live:
            continue
        live.add(nid)
        stack.extend(by_id[nid].parents)
    return live


def eliminate_dead_nodes(graph: Graph) -> Graph:
    """Return a graph containing only nodes that contribute to the output.
    Placeholders are always kept (dropping one would change the run()
    signature) even if traced control flow made them unused."""
    live = _live_ids(graph)
    nodes = [n for n in graph.nodes
             if n.id in live or n.kind == "placeholder"]
    return Graph(nodes, graph.output_id, list(graph.placeholder_ids))


def fold_constants(graph: Graph) -> Graph:
    """Precompute every op whose inputs are all constants, replacing it with
    a constant node holding the computed value."""
    constant_ids: Set[int] = {n.id for n in graph.nodes if n.kind == "constant"}
    values: Dict[int, Any] = {n.id: n.value for n in graph.nodes if n.kind == "constant"}
    nodes: List[Node] = []
    for node in graph.nodes:
        if (node.kind == "op"
                and all(p in constant_ids for p in node.parents)):
            value = node.op.forward(*(values[p] for p in node.parents), **node.kwargs)
            folded = Node(node.id, "constant", value=value,
                          shape=tuple(value.shape), dtype=value.dtype,
                          name=f"folded_{node.op.name}")
            constant_ids.add(node.id)
            values[node.id] = value
            nodes.append(folded)
        else:
            nodes.append(node)
    return Graph(nodes, graph.output_id, list(graph.placeholder_ids))


def _consumer_counts(graph: Graph) -> Dict[int, int]:
    counts: Dict[int, int] = {n.id: 0 for n in graph.nodes}
    for node in graph.nodes:
        for p in node.parents:
            counts[p] += 1
    return counts


def fuse_elementwise(graph: Graph) -> Graph:
    """Collapse maximal chains of elementwise single-input ops into one
    fused op node per chain.

    A node joins the chain below it when it is an elementwise op whose only
    consumer is the next op in the chain -- so no intermediate value that
    something else still needs ever disappears."""
    consumers = _consumer_counts(graph)
    by_id = {n.id: n for n in graph.nodes}

    def chain_start(node: Node) -> Node:
        """Walk upward while the parent is a fusable elementwise op used
        only by `node`."""
        while True:
            parent = by_id[node.parents[0]]
            if (parent.kind == "op" and parent.op.elementwise
                    and consumers[parent.id] == 1
                    and parent.id != graph.output_id):
                node = parent
            else:
                return node

    fused_away: Set[int] = set()
    replacements: Dict[int, Node] = {}
    for node in graph.nodes:
        if node.kind != "op" or not node.op.elementwise or node.id in fused_away:
            continue
        # Only consider ends of chains: a node whose consumer won't absorb it.
        consumer_ids = [n.id for n in graph.nodes
                        if node.id in n.parents and n.kind == "op"
                        and n.op.elementwise and len(n.parents) == 1]
        if consumers[node.id] == 1 and len(consumer_ids) == 1 and node.id != graph.output_id:
            continue  # absorbed into its consumer's chain instead
        start = chain_start(node)
        if start.id == node.id:
            continue  # chain of length 1 -- nothing to fuse
        # Collect the chain start..node (inclusive) top-down.
        chain = [node]
        while chain[-1].id != start.id:
            chain.append(by_id[chain[-1].parents[0]])
        chain.reverse()
        chain_ops = [(n.op, n.kwargs) for n in chain]

        def fused_forward(a: Any, _chain=tuple(chain_ops)) -> Any:
            for op, kwargs in _chain:
                a = op.forward(a, **kwargs)
            return a

        # The fused op is still differentiable: replay the chain's saved
        # intermediates on the fly. (Fused graphs are for re-execution;
        # gradient support keeps the op honest if used eagerly.)
        def fused_backward(g: Any, out: Any, a: Any, _chain=tuple(chain_ops)) -> tuple:
            values = [a]
            for op, kwargs in _chain:
                values.append(op.forward(values[-1], **kwargs))
            for (op, kwargs), inp, outp in zip(reversed(_chain), reversed(values[:-1]),
                                               reversed(values[1:])):
                (g,) = op.backward(g, outp, inp, **kwargs)
            return (g,)

        fused = custom_op("fused(" + "->".join(op.name for op, _ in chain_ops) + ")",
                          fused_forward, fused_backward, elementwise=True)
        replacements[node.id] = Node(node.id, "op", op=fused, kwargs={},
                                     parents=(start.parents[0],),
                                     shape=node.shape, dtype=node.dtype)
        fused_away.update(n.id for n in chain[:-1])

    nodes = [replacements.get(n.id, n) for n in graph.nodes if n.id not in fused_away]
    return Graph(nodes, graph.output_id, list(graph.placeholder_ids))


def optimize(graph: Graph) -> Graph:
    """Apply all passes: dead-node elimination, constant folding, then
    elementwise fusion (with a final elimination sweep, since folding a
    node orphans the constants it consumed)."""
    graph = fold_constants(eliminate_dead_nodes(graph))
    return eliminate_dead_nodes(fuse_elementwise(graph))
