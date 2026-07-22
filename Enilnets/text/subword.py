"""Subword tokenizers (roadmap item 59): BPE, word-level and byte-level.

Same ``fit``/``encode``/``decode``/``save``/``load`` interface as
``text_utils.Tokenizer``, so either drops into ``TextGenerator``.

A character tokenizer has a tiny vocabulary but long sequences; a word
tokenizer has short sequences but a huge vocabulary and no way to spell an
unseen word. BPE learns the tradeoff, merging the most frequent adjacent
pair until it hits the vocabulary budget."""

import json
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from ..core.backend import np


#: Marks a word-initial position, so decoding can restore spaces without a
#: separate word-boundary channel. SentencePiece's convention.
SPACE = "▁"


class BPETokenizer:
    """Byte Pair Encoding, trained from scratch.

    `level` is "word" (default) or "byte":

    - **"word"** splits on whitespace first and learns merges within words,
      so a merge can never span a word boundary. Vocabulary is built from
      the characters actually present.
    - **"byte"** works on raw UTF-8 bytes, so *any* input is encodable and
      there is no out-of-vocabulary case at all -- the SentencePiece-style
      variant. Costs a slightly longer sequence for non-ASCII text.
    """

    def __init__(self, vocab_size: int = 1000, level: str = "word",
                 oov_token: str = "<OOV>", pad_token: str = "<PAD>",
                 start_token: str = "<START>", end_token: str = "<END>",
                 min_frequency: int = 2) -> None:
        if level not in ("word", "byte"):
            raise ValueError(f"level must be 'word' or 'byte', got {level!r}")
        floor = 4 + (256 if level == "byte" else 1)
        if vocab_size < floor:
            raise ValueError(
                f"vocab_size={vocab_size} is below the minimum of {floor} for "
                f"level={level!r}: four special tokens" +
                (" plus all 256 byte values, which byte level always includes"
                 if level == "byte" else " plus at least one symbol"))
        self.vocab_size = int(vocab_size)
        self.level = level
        self.oov_token = oov_token
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
        self.min_frequency = int(min_frequency)
        self.merges: List[Tuple[str, str]] = []
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.fitted = False
        self._cache: Dict[str, List[str]] = {}

    # -- training ---------------------------------------------------------

    def _pretokenize(self, text: str) -> List[str]:
        """Split text into the units merges may run inside. Whitespace is
        folded into the following word as a SPACE marker rather than dropped,
        so decoding is exact."""
        if self.level == "byte":
            return [text] if text else []
        return [SPACE + word for word in text.split()]

    def _symbols(self, unit: str) -> List[str]:
        if self.level == "byte":
            return [f"<{b}>" for b in unit.encode("utf-8")]
        return list(unit)

    def fit(self, texts: List[str]) -> "BPETokenizer":
        """Learn the merge table from `texts`."""
        counts: Counter = Counter()
        for text in texts:
            counts.update(self._pretokenize(text))
        if not counts:
            raise ValueError("no text to train on")

        # Each distinct word is carried once with its frequency, so a merge
        # scan costs O(distinct words) rather than O(corpus length).
        words = {word: self._symbols(word) for word in counts}
        if self.level == "byte":
            # ALL 256 byte values, not just the ones the corpus happened to
            # contain -- covering every byte is precisely what makes byte
            # level free of out-of-vocabulary inputs.
            alphabet = [f"<{b}>" for b in range(256)]
        else:
            alphabet = sorted({sym for syms in words.values() for sym in syms})

        specials = [self.pad_token, self.start_token, self.end_token, self.oov_token]
        budget = self.vocab_size - len(specials) - len(alphabet)
        self.merges = []
        for _ in range(max(0, budget)):
            pairs: Counter = Counter()
            for word, syms in words.items():
                freq = counts[word]
                for a, b in zip(syms, syms[1:]):
                    pairs[(a, b)] += freq
            if not pairs:
                break
            (best, freq) = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
            if freq < self.min_frequency:
                break
            self.merges.append(best)
            merged = best[0] + best[1]
            for word, syms in words.items():
                words[word] = _apply_merge(syms, best, merged)

        vocab = specials + alphabet + [a + b for a, b in self.merges]
        # dict.fromkeys keeps first-seen order while dropping duplicates: a
        # merge can reproduce a symbol already in the alphabet.
        vocab = list(dict.fromkeys(vocab))[:self.vocab_size]
        self.word_to_idx = {tok: i for i, tok in enumerate(vocab)}
        self.idx_to_word = {i: tok for tok, i in self.word_to_idx.items()}
        self._merge_rank = {pair: i for i, pair in enumerate(self.merges)}
        self._cache = {}
        self.fitted = True
        return self

    # -- inference --------------------------------------------------------

    def _tokenize_unit(self, unit: str) -> List[str]:
        cached = self._cache.get(unit)
        if cached is not None:
            return cached
        syms = self._symbols(unit)
        while len(syms) > 1:
            # Always apply the EARLIEST-learned applicable merge, so encoding
            # replays training order exactly; taking any applicable merge
            # would give a different segmentation for the same table.
            ranked = [(self._merge_rank[p], p) for p in zip(syms, syms[1:])
                      if p in self._merge_rank]
            if not ranked:
                break
            _, pair = min(ranked)
            syms = _apply_merge(syms, pair, pair[0] + pair[1])
        self._cache[unit] = syms
        return syms

    def tokenize(self, text: str) -> List[str]:
        """Text -> subword strings, without ids or special tokens."""
        self._require_fitted()
        out: List[str] = []
        for unit in self._pretokenize(text):
            out.extend(self._tokenize_unit(unit))
        return out

    def encode(self, text: str, max_length: Optional[int] = None,
               add_special_tokens: bool = True) -> Any:
        """Text -> an int array of token ids."""
        self._require_fitted()
        oov = self.word_to_idx[self.oov_token]
        ids = [self.word_to_idx.get(tok, oov) for tok in self.tokenize(text)]
        if add_special_tokens:
            ids = ([self.word_to_idx[self.start_token]] + ids
                   + [self.word_to_idx[self.end_token]])
        if max_length is not None:
            ids = ids[:max_length]
            ids += [self.word_to_idx[self.pad_token]] * (max_length - len(ids))
        return np.array(ids, dtype=np.int64)

    def decode(self, indices: Any, skip_special: bool = True) -> str:
        """Token ids -> text. Exact for anything `encode` produced, since
        whitespace is carried in the SPACE marker rather than discarded."""
        self._require_fitted()
        specials = {self.pad_token, self.start_token, self.end_token, self.oov_token}
        pieces = []
        for i in np.asarray(indices).reshape(-1).tolist():
            tok = self.idx_to_word.get(int(i), self.oov_token)
            if skip_special and tok in specials:
                continue
            pieces.append(tok)
        joined = "".join(pieces)
        if self.level == "byte":
            raw = bytes(int(p[1:-1]) for p in _split_byte_tokens(joined))
            return raw.decode("utf-8", errors="replace")
        return joined.replace(SPACE, " ").lstrip()

    # -- persistence ------------------------------------------------------

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise ValueError("tokenizer is not fitted; call fit(texts) first")

    def state_dict(self) -> Dict[str, Any]:
        return {
            "kind": "bpe", "vocab_size": self.vocab_size, "level": self.level,
            "oov_token": self.oov_token, "pad_token": self.pad_token,
            "start_token": self.start_token, "end_token": self.end_token,
            "min_frequency": self.min_frequency,
            "merges": [list(pair) for pair in self.merges],
            "vocab": [self.idx_to_word[i] for i in range(len(self.idx_to_word))],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "BPETokenizer":
        self.vocab_size = int(state["vocab_size"])
        self.level = state["level"]
        self.oov_token = state["oov_token"]
        self.pad_token = state["pad_token"]
        self.start_token = state["start_token"]
        self.end_token = state["end_token"]
        self.min_frequency = int(state.get("min_frequency", 2))
        self.merges = [tuple(pair) for pair in state["merges"]]
        self.word_to_idx = {tok: i for i, tok in enumerate(state["vocab"])}
        self.idx_to_word = {i: tok for tok, i in self.word_to_idx.items()}
        self._merge_rank = {pair: i for i, pair in enumerate(self.merges)}
        self._cache = {}
        self.fitted = True
        return self

    def save(self, path: str) -> None:
        """Write the merge table and vocabulary as JSON."""
        self._require_fitted()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.state_dict(), fh, ensure_ascii=False)

    def load(self, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as fh:
            return self.load_state_dict(json.load(fh))

    def __len__(self) -> int:
        return len(self.word_to_idx)

    def __repr__(self) -> str:
        state = f"{len(self.word_to_idx)} tokens, {len(self.merges)} merges" \
            if self.fitted else "unfitted"
        return f"BPETokenizer(level={self.level!r}, {state})"


def _apply_merge(symbols: List[str], pair: Tuple[str, str], merged: str) -> List[str]:
    """Replace every non-overlapping occurrence of `pair` with `merged`."""
    out, i = [], 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def _split_byte_tokens(joined: str) -> List[str]:
    """Split a concatenation of ``<n>`` byte tokens back into its parts.
    Merged byte tokens are just several ``<n>`` runs stuck together."""
    out, buf = [], ""
    for ch in joined:
        buf += ch
        if ch == ">":
            out.append(buf)
            buf = ""
    return out
