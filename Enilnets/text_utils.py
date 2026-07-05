#!/usr/bin/env python3
"""
Text preprocessing, tokenization, and dataset utilities for Enilnets.
"""
import numpy as np
import re
import os
from collections import Counter

class Tokenizer:
    """
    Simple character-level or word-level tokenizer.
    """
    def __init__(self, vocab_size=256, level="char", oov_token="<OOV>",
                 pad_token="<PAD>", start_token="<START>", end_token="<END>"):
        self.vocab_size = vocab_size
        self.level = level
        self.oov_token = oov_token
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.fitted = False

    def fit(self, texts):
        """Build vocabulary from texts."""
        if self.level == "char":
            all_chars = set()
            for text in texts:
                all_chars.update(text)
            vocab = [self.pad_token, self.start_token, self.end_token, self.oov_token] + sorted(list(all_chars))
        else:  # word
            words = Counter()
            for text in texts:
                words.update(text.lower().split())
            vocab = [self.pad_token, self.start_token, self.end_token, self.oov_token]
            vocab += [w for w, _ in words.most_common(self.vocab_size - len(vocab))]

        self.word_to_idx = {w: i for i, w in enumerate(vocab)}
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}
        self.fitted = True
        return self

    def encode(self, text, max_length=None, add_special_tokens=True):
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

    def decode(self, indices, skip_special=True):
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

    def save(self, path):
        import json
        with open(path, 'w') as f:
            json.dump({
                'word_to_idx': self.word_to_idx,
                'level': self.level,
                'vocab_size': self.vocab_size
            }, f)

    def load(self, path):
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        self.word_to_idx = data['word_to_idx']
        self.idx_to_word = {int(i): w for w, i in self.word_to_idx.items()}
        self.level = data['level']
        self.vocab_size = data['vocab_size']
        self.fitted = True


def load_text_file(path, encoding='utf-8'):
    """Load text from file."""
    with open(path, 'r', encoding=encoding) as f:
        return f.read()

def load_texts_from_directory(directory, encoding='utf-8', max_files=None):
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

def create_sliding_windows(data, window_size, stride=1):
    """
    Create sliding windows from sequential data.

    Parameters
    ----------
    data : ndarray
        1D array of tokens or features
    window_size : int
        Size of each window
    stride : int
        Step between windows

    Returns
    -------
    X : ndarray
        Input windows (all but last element)
    y : ndarray
        Target windows (all but first element) — for next-token prediction
    """
    windows = []
    for i in range(0, len(data) - window_size, stride):
        windows.append(data[i:i + window_size + 1])
    windows = np.array(windows)
    return windows[:, :-1], windows[:, 1:]

def one_hot_encode(indices, vocab_size):
    """One-hot encode token indices."""
    one_hot = np.zeros((len(indices), vocab_size), dtype=np.float64)
    one_hot[np.arange(len(indices)), indices] = 1.0
    return one_hot

def pad_sequences(sequences, max_length=None, pad_value=0):
    """Pad a list of sequences to the same length."""
    if max_length is None:
        max_length = max(len(s) for s in sequences)

    padded = np.full((len(sequences), max_length), pad_value, dtype=np.int32)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_length)
        padded[i, :length] = seq[:length]

    return padded
