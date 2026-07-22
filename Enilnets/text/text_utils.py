#!/usr/bin/env python3
"""
Text preprocessing, tokenization, and dataset utilities for Enilnets.
"""
from typing import Any, List, Optional, Tuple

from ..core.backend import np
from ..core import backend
import re
import os
from collections import Counter

class Tokenizer:
    """
    Simple character-level or word-level tokenizer.
    """
    def __init__(self, vocab_size: int = 256, level: str = "char", oov_token: str = "<OOV>",
                 pad_token: str = "<PAD>", start_token: str = "<START>", end_token: str = "<END>") -> None:
        self.vocab_size = vocab_size
        self.level = level
        self.oov_token = oov_token
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.fitted = False

    def fit(self, texts: List[str]) -> "Tokenizer":
        """Build vocabulary from texts."""
        if self.level == "char":
            all_chars = set()
            for text in texts:
                all_chars.update(text)
            vocab = [self.pad_token, self.start_token, self.end_token, self.oov_token] + sorted(list(all_chars))
        else:  # word
            # vocab_size must leave room for the 4 built-in special tokens
            # added below, or the vocabulary ends up as just those 4 tokens
            # with no actual words at all.
            if self.vocab_size <= 4:
                raise ValueError(
                    f"vocab_size={self.vocab_size} leaves no room for any actual "
                    "words after the 4 built-in special tokens (pad/start/end/oov) "
                    "-- pass a larger vocab_size."
                )
            words = Counter()
            for text in texts:
                words.update(text.lower().split())
            vocab = [self.pad_token, self.start_token, self.end_token, self.oov_token]
            vocab += [w for w, _ in words.most_common(self.vocab_size - len(vocab))]

        self.word_to_idx = {w: i for i, w in enumerate(vocab)}
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}
        self.fitted = True
        return self

    def encode(self, text: str, max_length: Optional[int] = None, add_special_tokens: bool = True) -> Any:
        """Convert text to token indices."""
        if not self.fitted:
            raise RuntimeError("Tokenizer must be fitted before encoding")

        if self.level == "char":
            tokens = list(text)
        else:
            tokens = text.lower().split()

        indices = []
        if add_special_tokens:
            indices.append(self.word_to_idx[self.start_token])

        for token in tokens:
            indices.append(self.word_to_idx.get(token, self.word_to_idx[self.oov_token]))

        if add_special_tokens:
            indices.append(self.word_to_idx[self.end_token])

        if max_length is not None:
            if len(indices) < max_length:
                indices += [self.word_to_idx[self.pad_token]] * (max_length - len(indices))
            else:
                indices = indices[:max_length]

        return np.array(indices, dtype=np.int32)

    def decode(self, indices: Any, skip_special: bool = True) -> str:
        """Convert token indices back to text."""
        tokens = []
        for idx in indices:
            token = self.idx_to_word.get(int(idx), self.oov_token)
            if skip_special and token in [self.pad_token, self.start_token, self.end_token]:
                continue
            tokens.append(token)

        if self.level == "char":
            return "".join(tokens)
        else:
            return " ".join(tokens)

    def save(self, path: str) -> None:
        import json
        with open(path, 'w') as f:
            json.dump({
                'word_to_idx': self.word_to_idx,
                'level': self.level,
                'vocab_size': self.vocab_size,
                'oov_token': self.oov_token,
                'pad_token': self.pad_token,
                'start_token': self.start_token,
                'end_token': self.end_token,
            }, f)

    def load(self, path: str) -> None:
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        self.word_to_idx = data['word_to_idx']
        self.idx_to_word = {int(i): w for w, i in self.word_to_idx.items()}
        self.level = data['level']
        self.vocab_size = data['vocab_size']
        # Older saved files predate these fields -- fall back to this
        # instance's constructor defaults rather than KeyError.
        self.oov_token = data.get('oov_token', self.oov_token)
        self.pad_token = data.get('pad_token', self.pad_token)
        self.start_token = data.get('start_token', self.start_token)
        self.end_token = data.get('end_token', self.end_token)
        self.fitted = True


def load_text_file(path: str, encoding: str = 'utf-8') -> str:
    """Load text from file."""
    with open(path, 'r', encoding=encoding) as f:
        return f.read()

def load_texts_from_directory(directory: str, encoding: str = 'utf-8',
                               max_files: Optional[int] = None) -> List[str]:
    """Load all text files from a directory."""
    texts = []
    files = sorted(os.listdir(directory))
    if max_files is not None:
        files = files[:max_files]
    for fname in files:
        path = os.path.join(directory, fname)
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding=encoding) as f:
                    texts.append(f.read())
            except UnicodeDecodeError:
                pass
    return texts

def create_sliding_windows(data: Any, window_size: int, stride: int = 1) -> Tuple[Any, Any]:
    """Chunk a 1D token/feature array into overlapping windows for next-token
    prediction. Returns (X, y): X drops each window's last element, y drops
    its first, so y is X shifted by one position."""
    if window_size >= len(data):
        raise ValueError(
            f"window_size={window_size} must be < len(data)={len(data)} -- "
            "a too-large window_size silently yields zero windows here, then "
            "a confusing IndexError on the empty result downstream."
        )
    windows = []
    for i in range(0, len(data) - window_size, stride):
        windows.append(data[i:i + window_size + 1])
    windows = np.array(windows)
    return windows[:, :-1], windows[:, 1:]

def one_hot_encode(indices: Any, vocab_size: int) -> Any:
    """One-hot encode token indices."""
    one_hot = np.zeros((len(indices), vocab_size), dtype=backend.default_dtype())
    one_hot[np.arange(len(indices)), indices] = 1.0
    return one_hot

# pad_sequences lives in Enilnets.preprocessing now; re-exported here
# for backward compatibility.
from ..preprocessing.text import pad_sequences
