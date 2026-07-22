"""Network visualization as a self-contained SVG string -- pure stdlib, no
matplotlib, keeping the library's NumPy-only dependency footprint.

- plot_network(model) -- a NeuralNet's dense/recurrent layers as node
  columns with real weighted edges; other layer types render as labeled
  blocks, having no natural one-neuron-per-circle form.
- plot_genome(genome) -- a NEAT Genome's variable topology, node positions
  from graph depth, edges colored by weight and dashed if disabled.

Pass `sample_input` to either to color nodes by their live activation
values from an actual forward pass."""
from typing import Any, Dict, List, Optional, Tuple

from ..core.backend import np
from ..core import backend

_NODE_RADIUS = 12
_COL_SPACING = 170
_NODE_SPACING = 36
_MARGIN = 70
_BLOCK_W, _BLOCK_H = 130, 50


def _heat_color(t: float) -> str:
    """t in [0, 1] -> hex color: blue (low) -> white (mid) -> red (high)."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        u = t / 0.5
        r, g, b = 59 + u * (255 - 59), 130 + u * (255 - 130), 246 + u * (255 - 246)
    else:
        u = (t - 0.5) / 0.5
        r, g, b = 255, 255 - u * (255 - 38), 255 - u * (255 - 38)
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def _weight_style(w: float, max_abs: float) -> Tuple[str, float, float]:
    max_abs = max(max_abs, 1e-8)
    t = min(abs(w) / max_abs, 1.0)
    color = "#2563eb" if w >= 0 else "#dc2626"
    return color, 0.5 + 2.5 * t, 0.12 + 0.75 * t


def _display_indices(n: int, max_nodes: int) -> Tuple[List[int], bool]:
    """Which of n node indices to actually draw (all of them, or a capped
    top+bottom sample with a break marker for large layers)."""
    if n <= max_nodes:
        return list(range(n)), False
    half = max(1, max_nodes // 2)
    return list(range(half)) + list(range(n - half, n)), True


def _svg_header(width: int, height: int) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" font-family="sans-serif">'
           f'<rect width="100%" height="100%" fill="white"/>')


def to_html(svg: str, title: str = "Enilnets Network Visualization") -> str:
    """Wrap a raw SVG string (as returned by plot_network/plot_genome) in a
    minimal standalone HTML document -- double-click to open in any browser,
    or serve it directly from a web app (e.g. Flask's
    `Response(to_html(svg), mimetype="text/html")`)."""
    return (f"<!doctype html><html><head><meta charset=\"utf-8\"><title>{title}</title>"
           f"<style>body{{margin:0;padding:24px;background:#fafafa;"
           f"font-family:sans-serif;display:flex;justify-content:center;}}</style></head>"
           f"<body>{svg}</body></html>")


def _write(svg: str, filename: Optional[str]) -> str:
    """Always returns the raw SVG string (for embedding via
    IPython.display.SVG(...), inserting into an existing HTML page, etc.).
    If `filename` is given, also writes it to disk -- as a standalone HTML
    document (via to_html) if the extension is .html/.htm, or as a raw .svg
    file otherwise; either opens directly in a browser. Any OTHER extension
    (e.g. .png) still gets literal SVG/HTML text written to it (there's no
    raster rendering here) -- a warning is emitted so this isn't a silent
    footgun for a filename whose extension doesn't reflect its content."""
    if filename is not None:
        is_html = filename.lower().endswith((".html", ".htm"))
        if not is_html and not filename.lower().endswith(".svg"):
            import warnings
            warnings.warn(
                f"plot_network/plot_genome writes SVG or HTML text, not a raster "
                f"image -- '{filename}' doesn't have a .svg/.html/.htm extension, "
                "but will contain SVG markup regardless of its actual extension.",
                stacklevel=3,
            )
        content = to_html(svg) if is_html else svg
        with open(filename, "w") as f:
            f.write(content)
    return svg


def _describe_layer(layer: Dict[str, Any]) -> str:
    t = layer["type"]
    if t == "conv2d":
        return f"Conv2D {layer['in_ch']}→{layer['out_ch']} k{layer['k']}"
    if t == "embedding":
        return f"Embedding {layer['vocab_size']}x{layer['embed_dim']}"
    if t == "multihead_attention":
        return f"Attention ({layer['num_heads']}h, d={layer['embed_dim']})"
    if t == "cross_attention":
        return f"CrossAttn ({layer['num_heads']}h, d={layer['embed_dim']})"
    if t == "batchnorm":
        return "BatchNorm"
    if t == "layernorm":
        return "LayerNorm"
    if t == "dropout":
        return f"Dropout {layer.get('rate', 0)}"
    if t == "flatten":
        return "Flatten"
    if t == "maxpool2d":
        return f"MaxPool {layer['p']}"
    if t == "avgpool2d":
        return f"AvgPool {layer['p']}"
    if t == "globalavgpool2d":
        return "GlobalAvgPool"
    if t == "upsample2d":
        return f"Upsample x{layer.get('scale_factor', 2)}"
    if t == "positional_encoding":
        return "PosEncoding"
    if t == "residual_save":
        return "Residual start"
    if t == "residual_add":
        return "Residual add"
    return t


# Block layer types that don't change the "channel"/feature count flowing
# through them (only elementwise values, or -- for the pooling/upsample
# entries -- only spatial size, not channel count), so a dense/conv layer
# on the far side of a run of these still has a meaningful edge back to
# the last real node column -- unlike attention/flatten/embedding, where
# no such correspondence exists.
_PASSTHROUGH_BLOCK_TYPES = {
    "batchnorm", "layernorm", "dropout",
    "maxpool2d", "avgpool2d", "globalavgpool2d", "upsample2d",
}

# Layer types rendered as real node columns with per-edge weight styling,
# rather than a single opaque label block.
_NODE_LAYER_TYPES = {"dense", "sparse", "rnn", "lstm", "gru", "conv2d", "conv1d",
                     "multihead_attention", "cross_attention"}


def _find_source_column(columns: List[Dict[str, Any]], model: Any, ci: int) -> Optional[int]:
    """Walk backward from column ci-1 through any run of passthrough blocks
    to find the node column a dense/sparse/conv/attention layer at `ci`
    should draw its real edges from. Returns None if a shape-changing
    layer (or the start of the network) is hit first."""
    j = ci - 1
    while j >= 0:
        col = columns[j]
        if col["kind"] in ("nodes", "conv", "attention"):
            return j
        li = col["layer_index"]
        if li is not None and model.layers[li]["type"] in _PASSTHROUGH_BLOCK_TYPES:
            j -= 1
            continue
        return None
    return None


def _network_columns(model: Any) -> List[Dict[str, Any]]:
    """Walk model.layers into a list of column descriptors:
    - node columns: `{"kind": "nodes"|"conv"|"attention", "size", "label",
      "layer_index"}` (layer_index=None for the synthetic input column).
      "conv" and "attention" are both drawn as real node/edge columns like
      "nodes", just with their own edge-weight source (aggregated kernel
      magnitude / aggregated Q-K-V projection magnitude respectively,
      since neither reduces to one plain weight matrix the way dense does).
    - block columns: `{"kind": "block", "label", "layer_index"}`, for
      every layer type with no natural one-node-per-value representation
      (norm/pooling/dropout/embedding/positional-encoding/residual
      markers/...).
    """
    columns = []
    if model.layers:
        first = model.layers[0]
        if first["type"] in ("dense", "sparse"):
            columns.append({"kind": "nodes", "size": first["weights"].shape[1],
                            "label": "Input", "layer_index": None})
        elif first["type"] in ("conv2d", "conv1d"):
            columns.append({"kind": "conv", "size": first["in_ch"],
                            "label": "Input (channels)", "layer_index": None})
    for i, layer in enumerate(model.layers):
        t = layer["type"]
        if t in ("dense", "sparse"):
            kind_label = "Dense" if t == "dense" else "Sparse"
            columns.append({"kind": "nodes", "size": layer["weights"].shape[0],
                            "label": f"{kind_label} ({layer['activation']})", "layer_index": i})
        elif t in ("rnn", "lstm", "gru"):
            columns.append({"kind": "nodes", "size": layer["hidden_dim"],
                            "label": t.upper(), "layer_index": i})
        elif t in ("conv2d", "conv1d"):
            kind_label = "Conv2D" if t == "conv2d" else "Conv1D"
            columns.append({"kind": "conv", "size": layer["out_ch"],
                            "label": f"{kind_label} ({layer['activation']})", "layer_index": i})
        elif t in ("multihead_attention", "cross_attention"):
            columns.append({"kind": "attention", "size": layer["embed_dim"],
                            "label": _describe_layer(layer), "layer_index": i})
        else:
            columns.append({"kind": "block", "label": _describe_layer(layer), "layer_index": i})
    return columns


def _node_column_activation(out_arr: Any) -> Optional[Any]:
    """Reduce one layer's raw Forward() output (for sample 0) down to one
    scalar per node-column entry, for heat-mapping. Dense/RNN outputs are
    already (batch, features) -- used as-is. Conv outputs are (C,H,W) for
    sample 0 (NCHW) -- mean over spatial dims gives one value per channel.
    Attention outputs are (S,E) for sample 0 -- mean over the sequence
    axis gives one value per embedding dim. Returns None for any other
    shape (no sensible per-node reduction)."""
    if out_arr.ndim == 2:
        return np.asarray(out_arr[0], dtype=backend.default_dtype())
    if out_arr.ndim == 4:
        return np.asarray(np.mean(out_arr[0], axis=(1, 2)), dtype=backend.default_dtype())
    if out_arr.ndim == 3:
        return np.asarray(np.mean(out_arr[0], axis=0), dtype=backend.default_dtype())
    return None


def _conv_edge_matrix(layer: Dict[str, Any]) -> Any:
    """(out_ch, in_ch) matrix of mean(|kernel|) per channel pair, for
    drawing conv2d/conv1d edges the same way dense's weight matrix is
    drawn -- collapses the spatial kernel dims (k,k) or (k,) that dense
    doesn't have, since the node-column model only represents channels,
    not spatial structure."""
    W = layer["weights"]  # conv2d: (out_ch, in_ch, k, k); conv1d: (out_ch, in_ch, k)
    return np.mean(np.abs(W), axis=tuple(range(2, W.ndim)))


def _attention_edge_matrix(layer: Dict[str, Any]) -> Any:
    """(embed_dim, embed_dim) matrix aggregating the three input
    projections (Wq/Wk/Wv) into one edge-weight source for the column
    BEFORE this attention layer -- attention doesn't reduce to one plain
    weight matrix the way dense does, so this is a deliberately
    approximate "how much does output dim b's attention computation
    depend on input dim a" summary, not an exact gradient-flow measure."""
    return (np.abs(layer["Wq"]) + np.abs(layer["Wk"]) + np.abs(layer["Wv"])) / 3.0


def _node_title(col: Dict[str, Any], layer: Optional[Dict[str, Any]]) -> str:
    """<title> tooltip text identifying which layer a node/block column
    came from -- the per-column text label above it names the layer type,
    but once a layer spans multiple sub-elements (e.g. a conv2d's
    channels) there's no other way to trace a specific element back to
    its originating layer index at a glance."""
    if col["layer_index"] is None:
        return "Network input"
    return f"Layer {col['layer_index']}: {col['label']}"


def plot_network(model: Any, sample_input: Optional[Any] = None, max_nodes_per_layer: int = 20,
                  filename: Optional[str] = None) -> str:
    """Render `model` as an SVG node/connection diagram; returns the SVG
    source and writes it to `filename` if given.

    Layers with a weight matrix (dense/sparse/RNN/LSTM/GRU/conv/attention)
    are drawn as columns of nodes with real weighted edges -- blue positive,
    red negative, opacity by magnitude. Conv edges aggregate the spatial
    kernel per channel pair and attention edges aggregate the Q/K/V
    projections, since neither is a single plain matrix. Every other layer
    type is a labeled block; residual_save/residual_add pairs get a curved
    skip edge. Nodes carry hover tooltips naming their layer index.

    With `sample_input`, node fill instead heat-maps that layer's live
    activation values per column (blue low, red high)."""
    columns = _network_columns(model)
    if not columns:
        raise ValueError("Model has no layers to visualize.")

    activations = {}
    if sample_input is not None:
        model.Forward(sample_input, training=False)
        for col in columns:
            li = col["layer_index"]
            out_arr = model.outputs[0] if li is None else model.outputs[li + 1]
            reduced = _node_column_activation(out_arr)
            if reduced is not None:
                key = -1 if li is None else li
                activations[key] = reduced

    node_kinds = ("nodes", "conv", "attention")
    node_cols = [c for c in columns if c["kind"] in node_kinds]
    max_display = max((min(c["size"], max_nodes_per_layer) for c in node_cols), default=1)
    height = _MARGIN * 2 + max_display * _NODE_SPACING
    width = _MARGIN * 2 + max(len(columns) - 1, 0) * _COL_SPACING

    # First pass: layout only (positions needed before edges can be drawn
    # underneath the nodes).
    positions = {}
    for ci, col in enumerate(columns):
        x = _MARGIN + ci * _COL_SPACING
        if col["kind"] == "block":
            positions[ci] = {"x": x, "block": True}
            continue
        idxs, truncated = _display_indices(col["size"], max_nodes_per_layer)
        total_h = len(idxs) * _NODE_SPACING
        y0 = height / 2 - total_h / 2
        pts = [(x, y0 + j * _NODE_SPACING + _NODE_SPACING / 2) for j in range(len(idxs))]
        positions[ci] = {"x": x, "pts": pts, "idxs": idxs, "truncated": truncated, "y0": y0, "total_h": total_h}

    svg = [_svg_header(int(width), int(height))]

    # Edges first, so nodes/labels render on top of them.
    for ci, col in enumerate(columns):
        if col["kind"] not in ("nodes", "conv", "attention") or col["layer_index"] is None:
            continue
        layer = model.layers[col["layer_index"]]
        t = layer["type"]
        if t in ("dense", "sparse"):
            W = layer["weights"]  # (n_out, n_in)
        elif t in ("conv2d", "conv1d"):
            W = _conv_edge_matrix(layer)  # (out_ch, in_ch)
        elif t in ("multihead_attention", "cross_attention"):
            W = _attention_edge_matrix(layer)  # (embed_dim, embed_dim)
        else:
            continue
        src_ci = _find_source_column(columns, model, ci)
        if src_ci is None:
            continue
        max_abs = float(np.max(np.abs(W))) if W.size else 1.0
        prev_pos, curr_pos = positions[src_ci], positions[ci]
        for a, (px, py) in zip(prev_pos["idxs"], prev_pos["pts"]):
            for b, (cx, cy) in zip(curr_pos["idxs"], curr_pos["pts"]):
                color, sw, op = _weight_style(float(W[b, a]), max_abs)
                svg.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{cx:.1f}" y2="{cy:.1f}" '
                          f'stroke="{color}" stroke-width="{sw:.2f}" stroke-opacity="{op:.2f}"/>')

    # Residual skip-connection edges: a curved path from each residual_save
    # block to its matching residual_add block (save_index names the
    # residual_save layer's own index directly -- see add_residual_end).
    block_col_by_layer_index = {col["layer_index"]: ci for ci, col in enumerate(columns)
                                if col["kind"] == "block" and col["layer_index"] is not None}
    for ci, col in enumerate(columns):
        if col["kind"] != "block" or col["layer_index"] is None:
            continue
        layer = model.layers[col["layer_index"]]
        if layer["type"] != "residual_add":
            continue
        save_ci = block_col_by_layer_index.get(layer["save_index"])
        if save_ci is None:
            continue
        x1 = positions[save_ci]["x"]
        x2 = positions[ci]["x"]
        y_top = height / 2 - _BLOCK_H / 2 - 18
        svg.append(f'<path d="M {x1:.1f} {y_top:.1f} '
                  f'C {x1:.1f} {y_top - 24:.1f}, {x2:.1f} {y_top - 24:.1f}, {x2:.1f} {y_top:.1f}" '
                  f'fill="none" stroke="#0d9488" stroke-width="2" stroke-dasharray="6,3" '
                  f'marker-end="url(#arrow)"/>')

    if any(model.layers[col["layer_index"]]["type"] == "residual_add"
           for col in columns if col["kind"] == "block" and col["layer_index"] is not None):
        svg.insert(1, '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" '
                      'orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#0d9488"/></marker></defs>')

    # Nodes/blocks/labels on top.
    for ci, col in enumerate(columns):
        pos = positions[ci]
        x = pos["x"]
        layer = model.layers[col["layer_index"]] if col["layer_index"] is not None else None
        title = _node_title(col, layer)
        if col["kind"] == "block":
            bx, by = x - _BLOCK_W / 2, height / 2 - _BLOCK_H / 2
            svg.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{_BLOCK_W}" height="{_BLOCK_H}" rx="8" '
                      f'fill="#f1f5f9" stroke="#64748b" stroke-width="1.5"><title>{title}</title></rect>')
            if layer is not None and layer["type"] == "embedding":
                # A small table/grid glyph (rows = a few vocab entries,
                # columns = a few embedding dims) instead of plain text,
                # since an embedding table has no natural node-per-value
                # representation (vocab_size is usually far too large).
                gx, gy, gw, gh = x - 22, by + 10, 44, 20
                svg.append(f'<rect x="{gx:.1f}" y="{gy:.1f}" width="{gw}" height="{gh}" '
                          f'fill="none" stroke="#334155" stroke-width="1"/>')
                for r in range(1, 3):
                    ry = gy + gh * r / 3
                    svg.append(f'<line x1="{gx:.1f}" y1="{ry:.1f}" x2="{gx+gw:.1f}" y2="{ry:.1f}" '
                              f'stroke="#334155" stroke-width="0.75"/>')
                for c in range(1, 4):
                    cx_ = gx + gw * c / 4
                    svg.append(f'<line x1="{cx_:.1f}" y1="{gy:.1f}" x2="{cx_:.1f}" y2="{gy+gh:.1f}" '
                              f'stroke="#334155" stroke-width="0.75"/>')
                svg.append(f'<text x="{x:.1f}" y="{by + gh + 22:.1f}" font-size="11" text-anchor="middle" '
                          f'fill="#334155">{col["label"]}</text>')
            else:
                svg.append(f'<text x="{x:.1f}" y="{height / 2:.1f}" font-size="11" text-anchor="middle" '
                          f'dominant-baseline="middle" fill="#334155">{col["label"]}</text>')
            continue

        key = -1 if col["layer_index"] is None else col["layer_index"]
        acts = activations.get(key)
        vmin = vmax = 0.0
        if acts is not None:
            vmin, vmax = float(np.min(acts)), float(np.max(acts))
            if vmax - vmin < 1e-8:
                vmax = vmin + 1e-8
        for node_idx, (px, py) in zip(pos["idxs"], pos["pts"]):
            if acts is not None and node_idx < len(acts):
                fill = _heat_color((float(acts[node_idx]) - vmin) / (vmax - vmin))
            else:
                fill = "#e2e8f0"
            svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{_NODE_RADIUS}" fill="{fill}" '
                      f'stroke="#334155" stroke-width="1"><title>{title}, node {node_idx}</title></circle>')
            if acts is not None and node_idx < len(acts):
                svg.append(f'<text x="{px:.1f}" y="{py:.1f}" font-size="7.5" text-anchor="middle" '
                          f'dominant-baseline="middle" fill="#1e293b">{acts[node_idx]:.2f}</text>')
        svg.append(f'<text x="{x:.1f}" y="{pos["y0"] - 14:.1f}" font-size="11" text-anchor="middle" '
                  f'fill="#0f172a" font-weight="600">{col["label"]}</text>')
        if pos["truncated"]:
            svg.append(f'<text x="{x:.1f}" y="{pos["y0"] + pos["total_h"] / 2:.1f}" font-size="16" '
                      f'text-anchor="middle" fill="#64748b">⋮</text>')

    svg.append("</svg>")
    return _write("".join(svg), filename)


def plot_genome(genome: Any, sample_input: Optional[Any] = None, max_nodes_per_layer: int = 30,
                show_disabled: bool = True, filename: Optional[str] = None) -> str:
    """Render a NEAT `Genome` as an SVG node/connection diagram. Node
    x-position is its graph depth (longest path from an input/bias node
    along enabled connections); nodes are colored by type (input=green,
    bias=amber, hidden=slate, output=blue), or by activation value if
    `sample_input` is given. Enabled connections are solid, colored by
    weight sign/magnitude; disabled connections are dashed light gray
    (omit them with show_disabled=False).
    """
    nodes = genome.nodes
    enabled_adj = {}
    for conn in genome.connections.values():
        if conn.enabled:
            enabled_adj.setdefault(conn.in_node, []).append(conn.out_node)

    depth = {nid: 0 for nid in nodes if nodes[nid].type in ("input", "bias")}
    order = genome._topo_order()
    for nid in order:
        for tgt in enabled_adj.get(nid, []):
            depth[tgt] = max(depth.get(tgt, 0), depth.get(nid, 0) + 1)
    for nid in nodes:
        depth.setdefault(nid, 0)

    by_depth = {}
    for nid, d in depth.items():
        by_depth.setdefault(d, []).append(nid)
    for d in by_depth:
        by_depth[d].sort()

    values = None
    if sample_input is not None:
        # Re-implement Genome.forward's evaluation loop here (rather than
        # calling it) so every intermediate node's value is captured, not
        # just the final outputs.
        x = np.asarray(sample_input, dtype=backend.default_dtype()).reshape(1, -1)
        from ..nn.activations import activate
        incoming = {}
        for conn in genome.connections.values():
            if conn.enabled:
                incoming.setdefault(conn.out_node, []).append((conn.in_node, conn.weight))
        values = {i: float(x[0, i]) for i in range(genome.n_inputs)}
        values[genome.bias_id] = 1.0
        for nid in order:
            node = nodes[nid]
            if node.type in ("input", "bias"):
                continue
            srcs = incoming.get(nid)
            s = 0.0 if not srcs else sum(values[src] * w for src, w in srcs)
            values[nid] = float(activate(node.activation, np.array([s]))[0])

    max_depth = max(by_depth.keys(), default=0)
    max_col_size = max((len(v) for v in by_depth.values()), default=1)
    max_col_size = min(max_col_size, max_nodes_per_layer)
    header_h = 28  # room for the per-column "Depth N" headers
    legend_h = 26  # room for the node-type color legend along the bottom
    width = _MARGIN * 2 + max_depth * _COL_SPACING
    height = _MARGIN * 2 + max_col_size * _NODE_SPACING + header_h + legend_h

    positions = {}
    col_meta = {}  # depth -> {"x", "y0", "total_h", "truncated"}
    for d in range(max_depth + 1):
        ids = by_depth.get(d, [])
        idxs, truncated = _display_indices(len(ids), max_nodes_per_layer)
        total_h = len(idxs) * _NODE_SPACING
        y0 = header_h + (height - header_h - legend_h) / 2 - total_h / 2
        x = _MARGIN + d * _COL_SPACING
        col_meta[d] = {"x": x, "y0": y0, "total_h": total_h, "truncated": truncated}
        for j, i in enumerate(idxs):
            positions[ids[i]] = (x, y0 + j * _NODE_SPACING + _NODE_SPACING / 2)

    svg = [_svg_header(int(width), int(height))]

    # Column headers -- one per depth, mirroring plot_network's per-column
    # label, so a genome reader doesn't have to infer "this column is
    # depth 2" from bare x-position alone.
    for d, meta in col_meta.items():
        svg.append(f'<text x="{meta["x"]:.1f}" y="18" font-size="11" text-anchor="middle" '
                  f'fill="#0f172a" font-weight="600">Depth {d}</text>')
        if meta["truncated"]:
            svg.append(f'<text x="{meta["x"]:.1f}" y="{meta["y0"] + meta["total_h"] / 2:.1f}" '
                      f'font-size="16" text-anchor="middle" fill="#64748b">⋮</text>')

    all_weights = [abs(c.weight) for c in genome.connections.values()]
    max_abs = max(all_weights) if all_weights else 1.0
    for conn in genome.connections.values():
        if conn.in_node not in positions or conn.out_node not in positions:
            continue
        x1, y1 = positions[conn.in_node]
        x2, y2 = positions[conn.out_node]
        if conn.enabled:
            color, sw, op = _weight_style(conn.weight, max_abs)
            svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                      f'stroke="{color}" stroke-width="{sw:.2f}" stroke-opacity="{op:.2f}"/>')
        elif show_disabled:
            svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                      f'stroke="#94a3b8" stroke-width="1" stroke-opacity="0.4" stroke-dasharray="4,3"/>')

    type_color = {"input": "#16a34a", "bias": "#d97706", "hidden": "#475569", "output": "#2563eb"}
    for nid, (x, y) in positions.items():
        node = nodes[nid]
        if values is not None:
            v = values.get(nid, 0.0)
            fill = _heat_color((v + 1) / 2) if node.type not in ("input", "bias") else type_color[node.type]
        else:
            fill = type_color.get(node.type, "#475569")
        title = f"Node {nid} ({node.type}, depth {depth[nid]})"
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{_NODE_RADIUS}" fill="{fill}" '
                  f'stroke="#1e293b" stroke-width="1"><title>{title}</title></circle>')
        svg.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="8" text-anchor="middle" '
                  f'dominant-baseline="middle" fill="white">{nid}</text>')

    # Node-type color legend along the bottom, so the input=green/bias=amber/
    # hidden=slate/output=blue coding is visible on the diagram itself, not
    # just documented in the docstring. Square swatches (not circles), so
    # counting <circle> elements elsewhere still means "count of actual
    # genome nodes".
    legend_items = [("input", "Input"), ("bias", "Bias"), ("hidden", "Hidden"), ("output", "Output")]
    legend_y = height - legend_h / 2
    swatch_s = 11
    item_w = width / len(legend_items)
    for i, (key, label) in enumerate(legend_items):
        lx = item_w * i + item_w / 2 - 24
        svg.append(f'<rect x="{lx - swatch_s / 2:.1f}" y="{legend_y - swatch_s / 2:.1f}" '
                  f'width="{swatch_s}" height="{swatch_s}" '
                  f'fill="{type_color[key]}" stroke="#1e293b" stroke-width="1"/>')
        svg.append(f'<text x="{lx + swatch_s / 2 + 6:.1f}" y="{legend_y:.1f}" font-size="10" '
                  f'dominant-baseline="middle" fill="#334155">{label}</text>')

    svg.append("</svg>")
    return _write("".join(svg), filename)
