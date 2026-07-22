"""Text/sequence batching transforms (split out of text/text_utils.py)."""
from typing import Any, List, Optional

from ..core.backend import np

def pad_sequences(sequences: List[Any], max_length: Optional[int] = None, pad_value: Any = 0,
                   dtype: Any = np.int32) -> Any:
    """Pad a list of sequences to the same length.

    dtype: defaults to int32 (token-id sequences, this function's primary
        use case) -- pass e.g. backend.default_dtype() to pad batches of
        continuous-feature sequences instead."""
    if max_length is None:
        max_length = max(len(s) for s in sequences)

    padded = np.full((len(sequences), max_length), pad_value, dtype=dtype)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        padded[i, :length] = seq[:length]

    return padded
