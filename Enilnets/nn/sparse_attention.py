"""Block-sparse attention (roadmap item 41).

Each query block attends to a small chosen set of key blocks instead of the
whole sequence. Unlike a masking trick, the selected blocks are physically
GATHERED, so the S x S score matrix is never built and the cost is
O(S * blocks_per_query * block_size) -- genuinely sub-quadratic.

Pattern components, combined as a union (Longformer / BigBird family):
  local   the `local` nearest blocks (both sides when non-causal)
  global  the first `global` blocks, attended by everyone
  random  `random` extra blocks per query block, drawn once at build time
"""

from typing import Any, Dict, Optional, Tuple

from ..core.backend import np

#: Additive score for a masked slot. Finite so exp() underflows to zero
#: cleanly rather than producing inf - inf.
MASK_VALUE = -1e9


def normalize_pattern(pattern: Any) -> Dict[str, int]:
    """Validate a sparse-attention pattern dict and fill in its defaults:
    ``block_size`` (required, > 0), ``local`` (1), ``global`` (0),
    ``random`` (0), ``seed`` (0)."""
    if not isinstance(pattern, dict):
        raise ValueError(
            "sparse_pattern must be a dict, e.g. "
            "{'block_size': 16, 'local': 1, 'global': 1, 'random': 2}")
    unknown = set(pattern) - {"block_size", "local", "global", "random", "seed"}
    if unknown:
        raise ValueError(f"Unknown sparse_pattern keys: {sorted(unknown)}")
    spec = {"local": 1, "global": 0, "random": 0, "seed": 0}
    spec.update(pattern)
    if "block_size" not in pattern or int(spec["block_size"]) < 1:
        raise ValueError("sparse_pattern needs a block_size >= 1")
    for key in ("local", "global", "random"):
        if int(spec[key]) < 0:
            raise ValueError(f"sparse_pattern['{key}'] must be >= 0")
    if spec["local"] == 0 and spec["global"] == 0 and spec["random"] == 0:
        raise ValueError(
            "sparse_pattern selects no blocks at all (local/global/random are "
            "all 0), which would leave every query with nothing to attend to.")
    return {k: int(v) for k, v in spec.items()}


def block_selection(spec: Dict[str, int], n_blocks: int,
                    causal: bool) -> Tuple[Any, Any]:
    """Which key blocks each query block reads.

    Returns ``(index, valid)``, both ``(n_blocks, max_selected)``: `index`
    holds key-block numbers, `valid` marks the real entries. Query blocks
    with fewer selections than the maximum are padded by repeating their own
    block (always a legal choice) with `valid` False, so the gather stays a
    single uniform-shape operation."""
    rng = np.random.RandomState(spec["seed"])
    rows = []
    for i in range(n_blocks):
        chosen = {i}                                   # a block always sees itself
        for d in range(1, spec["local"] + 1):
            if i - d >= 0:
                chosen.add(i - d)
            if not causal and i + d < n_blocks:
                chosen.add(i + d)
        chosen.update(range(min(spec["global"], n_blocks)))
        if causal:
            chosen = {b for b in chosen if b <= i}
        if spec["random"]:
            pool = [b for b in range(i + 1 if causal else n_blocks) if b not in chosen]
            if pool:
                picks = rng.choice(len(pool), size=min(spec["random"], len(pool)),
                                   replace=False)
                chosen.update(pool[int(p)] for p in np.asarray(picks).reshape(-1))
        rows.append(sorted(chosen))

    width = max(len(r) for r in rows)
    index = np.zeros((n_blocks, width), dtype=np.int64)
    valid = np.zeros((n_blocks, width), dtype=bool)
    for i, row in enumerate(rows):
        index[i, :len(row)] = np.asarray(row)
        index[i, len(row):] = i                        # harmless filler
        valid[i, :len(row)] = True
    return index, valid


def _to_blocks(t: Any, n_blocks: int, block_size: int, S_pad: int) -> Any:
    """(B, H, S, D) zero-padded to S_pad, then split into blocks."""
    B, H, S, D = t.shape
    if S_pad != S:
        pad = np.zeros((B, H, S_pad - S, D), dtype=t.dtype)
        t = np.concatenate([t, pad], axis=2)
    return t.reshape(B, H, n_blocks, block_size, D)


def sparse_attention_forward(Qh: Any, Kh: Any, Vh: Any, spec: Dict[str, int],
                             causal: bool, alibi_slopes: Optional[Any] = None
                             ) -> Tuple[Any, Tuple]:
    """Block-sparse attention over ``(B, H, S, Dh)`` Q/K/V. Returns
    ``(context (B, H, S, Dh), cache)``. `alibi_slopes` optionally adds the
    per-head ALiBi distance bias, computed from absolute positions."""
    B, H, S, Dh = Qh.shape
    bs = spec["block_size"]
    n_blocks = (S + bs - 1) // bs
    S_pad = n_blocks * bs

    index, valid = block_selection(spec, n_blocks, causal)
    width = index.shape[1]

    Qb = _to_blocks(Qh, n_blocks, bs, S_pad)                      # (B,H,nb,bs,Dh)
    Kb = _to_blocks(Kh, n_blocks, bs, S_pad)
    Vb = _to_blocks(Vh, n_blocks, bs, S_pad)

    # The gather: only the selected key blocks are ever touched.
    Kg = Kb[:, :, index].reshape(B, H, n_blocks, width * bs, Dh)
    Vg = Vb[:, :, index].reshape(B, H, n_blocks, width * bs, Dh)

    scores = np.matmul(Qb, Kg.transpose(0, 1, 2, 4, 3)) / np.sqrt(Dh)

    # Absolute positions of each query slot and each gathered key slot, used
    # for causality, the ALiBi bias, and dropping the tail padding.
    q_pos = (np.arange(n_blocks)[:, None] * bs + np.arange(bs)[None, :])
    k_pos = (index[:, :, None] * bs + np.arange(bs)[None, None, :]
             ).reshape(n_blocks, width * bs)
    if alibi_slopes is not None:
        distance = q_pos[:, :, None] - k_pos[:, None, :]          # (nb,bs,W*bs)
        scores = scores + (-alibi_slopes[:, None, None, None]
                           * (distance if causal else np.abs(distance))[None, :, :, :])

    blocked = np.zeros((n_blocks, bs, width * bs), dtype=bool)
    blocked = blocked | ~np.repeat(valid, bs, axis=1)[:, None, :]  # filler blocks
    blocked = blocked | (k_pos >= S)[:, None, :]                  # tail padding
    if causal:
        blocked = blocked | (k_pos[:, None, :] > q_pos[:, :, None])
    scores = scores + np.where(blocked, MASK_VALUE, 0.0)[None, None, :, :, :]

    scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    context = np.matmul(attn, Vg).reshape(B, H, S_pad, Dh)[:, :, :S]
    return context, (Qb, Kb, Vb, Kg, Vg, attn, index, valid, n_blocks, bs, S, Dh)


def sparse_attention_backward(dcontext: Any, cache: Tuple) -> Tuple[Any, Any, Any]:
    """Gradients (dQh, dKh, dVh) for :func:`sparse_attention_forward`."""
    Qb, Kb, Vb, Kg, Vg, attn, index, valid, n_blocks, bs, S, Dh = cache
    B, H = dcontext.shape[0], dcontext.shape[1]
    S_pad = n_blocks * bs
    width = index.shape[1]

    d = _to_blocks(dcontext, n_blocks, bs, S_pad)                 # (B,H,nb,bs,Dh)
    dVg = np.matmul(attn.transpose(0, 1, 2, 4, 3), d)
    dattn = np.matmul(d, Vg.transpose(0, 1, 2, 4, 3))
    # Softmax JVP. The mask needs no special handling: masked slots have
    # attn == 0, so their gradient vanishes on its own.
    dscores = attn * (dattn - np.sum(dattn * attn, axis=-1, keepdims=True))
    dscores = dscores / np.sqrt(Dh)

    dQb = np.matmul(dscores, Kg)
    dKg = np.matmul(dscores.transpose(0, 1, 2, 4, 3), Qb)

    # Scatter the gathered gradients back. A key block selected by several
    # query blocks accumulates all of their contributions, hence add.at.
    dKb = np.zeros_like(Kb)
    dVb = np.zeros_like(Vb)
    dKg = dKg.reshape(B, H, n_blocks, width, bs, Dh)
    dVg = dVg.reshape(B, H, n_blocks, width, bs, Dh)
    # Zero the filler selections so they cannot leak gradient into a real
    # block (their scores were masked, but add.at does not know that).
    keep = valid[None, None, :, :, None, None]
    np.add.at(dKb, (slice(None), slice(None), index), np.where(keep, dKg, 0.0))
    np.add.at(dVb, (slice(None), slice(None), index), np.where(keep, dVg, 0.0))

    def unblock(t):
        return t.reshape(B, H, S_pad, Dh)[:, :, :S]

    return unblock(dQb), unblock(dKb), unblock(dVb)
