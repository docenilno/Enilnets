"""Text utilities: the Tokenizer, sliding windows, padding, one-hot
encoding. Phase 7's BPE/SentencePiece tokenizers land here."""

from .subword import BPETokenizer, SPACE
from .text_utils import (
    Tokenizer, load_text_file, load_texts_from_directory,
    create_sliding_windows, one_hot_encode, pad_sequences,
)

__all__ = [
    "BPETokenizer","Tokenizer", "load_text_file", "load_texts_from_directory",
           "create_sliding_windows", "one_hot_encode", "pad_sequences"]
