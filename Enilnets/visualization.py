"""
Network visualization: the classic node/connection diagram, rendered as a
self-contained SVG string (pure stdlib -- no matplotlib, keeping with the
rest of the library's NumPy-only dependency footprint). SVG opens in any
browser, embeds directly in a Jupyter notebook via IPython.display.SVG(...),
and can be saved to disk.

Two entry points:
- plot_network(model, sample_input=...) -- a NeuralNet's dense/recurrent
  layers as node columns with real weighted edges; other layer types (conv,
  attention, norm, pooling, ...) render as labeled blocks in between, since
  they don't have a natural one-neuron-per-circle representation.
- plot_genome(genome) -- a NEAT Genome's variable topology, node positions
  derived from graph depth, edges colored by weight and dashed if disabled.

Pass sample_input to either to color nodes by their actual activation value
from a live forward pass -- call this from inside your own training loop
whenever you want a snapshot of what the network is doing right now.
"""
from .backend import np
from . import backend

_NODE_RADIUS = 12
_COL_SPACING = 170
_NODE_SPACING = 36
_MARGIN = 70
_BLOCK_W, _BLOCK_H = 130, 50


def _heat_color(t):
    """t in [0, 1] -> hex color: blue (low) -> white (mid) -> red (high)."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        u = t / 0.5
        r, g, b = 59 + u * (255 - 59), 130 + u * (255 - 130), 246 + u * (255 - 246)
    else:
        u = (t - 0.5) / 0.5
        r, g, b = 255, 255 - u * (255 - 38), 255 - u * (255 - 38)
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def _weight_style(w, max_abs):
    max_abs = max(max_abs, 1e-8)
    t = min(abs(w) / max_abs, 1.0)
    color = "#2563eb" if w >= 0 else "#dc2626"
    return color, 0.5 + 2.5 * t, 0.12 + 0.75 * t


def _display_indices(n, max_nodes):
    """Which of n node indices to actually draw (all of them, or a capped
    top+bottom sample with a break marker for large layers)."""
    if n <= max_nodes:
        return list(range(n)), False
    half = max(1, max_nodes // 2)
    return list(range(half)) + list(range(n - half, n)), True


def _svg_header(width, height):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" font-family="sans-serif">'
           f'<rect width="100%" height="100%" fill="white"/>')


def to_html(svg, title="Enilnets Network Visualization"):
    """Wrap a raw SVG string (as returned by plot_network/plot_genome) in a
    minimal standalone HTML document -- double-click to open in any browser,
    or serve it directly from a web app (e.g. Flask's
    `Response(to_html(svg), mimetype="text/html")`)."""
    return (f"<!doctype html><html><head><meta charset=\"utf-8\"><title>{title}</title>"
           f"<style>body{{margin:0;padding:24px;background:#fafafa;"
           f"font-family:sans-serif;display:flex;justify-content:center;}}</style></head>"
           f"<body>{svg}</body></html>")


def _write(svg, filename):
    """Always returns the raw SVG string (for embedding via
    IPython.display.SVG(...), inserting into an existing HTML page, etc.).
    If `filename` is given, also writes it to disk -- as a standalone HTML
    document (via to_html) if the extension is .html/.htm, or as a raw .svg
    file otherwise; either opens directly in a browser."""
    if filename is not None:
        content = to_html(svg) if filename.lower().endswith((".html", ".htm")) else svg
        with open(filename, "w") as f:
            f.write(content)
    return svg


def _describe_layer(layer):
    t = layer["type"]
    if t == "conv2d":
        return f"Conv2D {layer['in_ch']}→{layer['out_ch']} k{layer['k']}"
    if t == "embedding":
        return f"Embedding {layer['vocab_size']}x{layer['embed_dim']}"
    if t == "multihead_attention":
        return f"Attention ({layer['num_heads']}h, d={layer['embed_dim']})"
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


# Block layer types that are dimension-preserving, elementwise transforms
# (output index i depends only on input index i), so a dense layer on the
# far side of a run of these still has a meaningful 1:1 edge back to the
# last real node column -- unlike conv/pooling/attention/flatten/embedding,
# where no such correspondence exists.
_PASSTHROUGH_BLOCK_TYPES = {"batchnorm", "layernorm", "dropout"}


def _find_source_column(columns, model, ci):
    """Walk backward from column ci-1 through any run of passthrough blocks
    to find the node column a dense/sparse layer at `ci` should draw its
    real weight-matrix edges from. Returns None if a shape-changing layer
    (or the start of the network) is hit first."""
    j = ci - 1
    while j >= 0:
        col = columns[j]
        if col["kind"] == "nodes":
            return j
        li = col["layer_index"]
        if li is not None and model.layers[li]["type"] in _PASSTHROUGH_BLOCK_TYPES:
            j -= 1
            continue
        return None
    return None


def _network_columns(model):
    """Walk model.layers into a list of column descriptors: node columns
    (`{"kind": "nodes", "size", "label", "layer_index"}`, layer_index=None
    for the synthetic input column) or block columns
    (`{"kind": "block", "label", "layer_index"}`)."""
    columns = []
    if model.layers and model.layers[0]["type"] in ("dense", "sparse"):
        columns.append({"kind": "nodes", "size": model.layers[0]["weights"].shape[1],
                        "label": "Input", "layer_index": None})
    for i, layer in enumerate(model.layers):
        t = layer["type"]
        if t in ("dense", "sparse"):
            kind_label = "Dense" if t == "dense" else "Sparse"
            columns.append({"kind": "nodes", "size": layer["weights"].shape[0],
                            "label": f"{kind_label} ({layer['activation']})", "layer_index": i})
        elif t in ("rnn", "lstm", "gru"):
            columns.append({"kind": "nodes", "size": layer["hidden_dim"],
                            "label": t.upper(), "layer_index": i})
        else:
            columns.append({"kind": "block", "label": _describe_layer(layer), "layer_index": i})
    return columns


def plot_network(model, sample_input=None, max_nodes_per_layer=20, filename=None):
    """Render `model` (a NeuralNet) as an SVG node/connection diagram.

    Dense/sparse/RNN/LSTM/GRU layers are drawn as columns of circular nodes;
    real dense/sparse weight matrices are drawn as colored edges (blue =
    positive, red = negative, opacity = relative magnitude). Every other
    layer type (conv2d, attention, norm, pooling, dropout, residual
    markers, ...) is drawn as a labeled block, since it doesn't reduce to a
    single weight matrix between two node columns.

    If `sample_input` is given, node fill color instead shows that layer's
    actual activation value from a live forward pass (heat-mapped per
    column, blue=low to red=high) -- call this from inside your training
    loop to snapshot what the network is doing at that point.

    Returns the SVG source as a string (embed via IPython.display.SVG(...)
    in a notebook, or open the saved file in any browser); also writes it to
    `filename` if given.
    """
    columns = _network_columns(model)
    if not columns:
        raise ValueError("Model has no layers to visualize.")

    activations = {}
    if sample_input is not None:
        model.Forward(sample_input, training=False)
        for col in columns:
            li = col["layer_index"]
            out_arr = model.outputs[0] if li is None else model.outputs[li + 1]
            if out_arr.ndim == 2:
                key = -1 if li is None else li
                activations[key] = np.asarray(out_arr[0], dtype=backend.default_dtype())

    node_cols = [c for c in columns if c["kind"] == "nodes"]
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
        if col["kind"] != "nodes" or col["layer_index"] is None:
            continue
        layer = model.layers[col["layer_index"]]
        if layer["type"] not in ("dense", "sparse"):
            continue
        src_ci = _find_source_column(columns, model, ci)
        if src_ci is None:
            continue
        W = layer["weights"]  # (n_out, n_in)
        max_abs = float(np.max(np.abs(W))) if W.size else 1.0
        prev_pos, curr_pos = positions[src_ci], positions[ci]
        for a, (px, py) in zip(prev_pos["idxs"], prev_pos["pts"]):
            for b, (cx, cy) in zip(curr_pos["idxs"], curr_pos["pts"]):
                color, sw, op = _weight_style(float(W[b, a]), max_abs)
                svg.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{cx:.1f}" y2="{cy:.1f}" '
                          f'stroke="{color}" stroke-width="{sw:.2f}" stroke-opacity="{op:.2f}"/>')

    # Nodes/blocks/labels on top.
    for ci, col in enumerate(columns):
        pos = positions[ci]
        x = pos["x"]
        if col["kind"] == "block":
            bx, by = x - _BLOCK_W / 2, height / 2 - _BLOCK_H / 2
            svg.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{_BLOCK_W}" height="{_BLOCK_H}" rx="8" '
                      f'fill="#f1f5f9" stroke="#64748b" stroke-width="1.5"/>')
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
                      f'stroke="#334155" stroke-width="1"/>')
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


def plot_genome(genome, sample_input=None, max_nodes_per_layer=30, show_disabled=True, filename=None):
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
        from .activations import activate
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
    width = _MARGIN * 2 + max_depth * _COL_SPACING
    height = _MARGIN * 2 + max_col_size * _NODE_SPACING

    positions = {}
    for d in range(max_depth + 1):
        ids = by_depth.get(d, [])
        idxs, truncated = _display_indices(len(ids), max_nodes_per_layer)
        total_h = len(idxs) * _NODE_SPACING
        y0 = height / 2 - total_h / 2
        x = _MARGIN + d * _COL_SPACING
        for j, i in enumerate(idxs):
            positions[ids[i]] = (x, y0 + j * _NODE_SPACING + _NODE_SPACING / 2)

    svg = [_svg_header(int(width), int(height))]

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
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{_NODE_RADIUS}" fill="{fill}" '
                  f'stroke="#1e293b" stroke-width="1"/>')
        svg.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="8" text-anchor="middle" '
                  f'dominant-baseline="middle" fill="white">{nid}</text>')

    svg.append("</svg>")
    return _write("".join(svg), filename)
