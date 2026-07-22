from typing import Any, List, Optional, Tuple

from ..core.backend import np
from ..core import backend
from ..core.base import NeuralNet
from ..text.text_utils import Tokenizer, create_sliding_windows
from ..nn.kvcache import KVCache, cached_forward_step


class TextGenerator:
    """A small GPT-style causal transformer for character- or word-level text
    generation, built from Enilnets' own layers. Trains by next-token
    prediction and samples autoregressively with temperature / top-k / top-p
    control. See the README's "Text generation" section for a worked
    example."""

    def __init__(self, tokenizer: Tokenizer, embed_dim: int = 64, num_heads: int = 4, num_layers: int = 2,
                 mlp_ratio: float = 4.0, dropout: float = 0.0, activation: str = "gelu",
                 max_seq_len: int = 128, learning_rate: float = 3e-4, optimizer: str = "adam",
                 l2_lambda: float = 0.0) -> None:
        # Duck-typed rather than isinstance: BPETokenizer implements the same
        # fit/encode/decode/word_to_idx interface without inheriting from
        # Tokenizer, and any equivalent tokenizer should work here too.
        required = ("encode", "decode", "word_to_idx", "fitted",
                    "pad_token", "start_token", "end_token", "oov_token")
        missing = [name for name in required if not hasattr(tokenizer, name)]
        if missing:
            raise ValueError(
                f"tokenizer is missing {missing}; it needs the same interface as "
                "Enilnets.Tokenizer or Enilnets.BPETokenizer")
        if not tokenizer.fitted:
            raise ValueError("tokenizer must be fitted -- call fit(texts) first")
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.word_to_idx)
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        self.network = NeuralNet(learning_rate=learning_rate, optimizer=optimizer, l2_lambda=l2_lambda)
        self.network.add_embedding(self.vocab_size, embed_dim)
        self.network.add_positional_encoding(max_seq_len, embed_dim, learnable=False)
        for _ in range(num_layers):
            self.network.add_transformer_block(embed_dim, num_heads, mlp_ratio=mlp_ratio,
                                               dropout=dropout, activation=activation, causal=True)
        self.network.add_layernorm()
        self.network.add_dense(None, self.vocab_size, activation="softmax")

    def prepare_sequences(self, texts: List[str], seq_len: Optional[int] = None,
                           stride: int = 1) -> Tuple[Any, Any]:
        """Tokenize `texts` into one continuous stream and chunk it into
        (X, y) next-token-prediction windows via create_sliding_windows.
        seq_len defaults to max_seq_len - 1 (room for the positional table)."""
        seq_len = seq_len or (self.max_seq_len - 1)
        all_ids = []
        for text in texts:
            all_ids.extend(self.tokenizer.encode(text, add_special_tokens=True).tolist())
        all_ids = np.array(all_ids, dtype=np.int64)
        if len(all_ids) <= seq_len:
            raise ValueError(
                f"Corpus has only {len(all_ids)} tokens, need more than seq_len={seq_len}."
            )
        X, y = create_sliding_windows(all_ids, seq_len, stride)
        return X, y

    def train_step(self, X_batch: Any, y_batch: Any) -> float:
        """X_batch, y_batch: (B, S) integer token-id arrays (y is X shifted by
        one position, as returned by prepare_sequences)."""
        probs = self.network.Forward(X_batch, training=True)  # (B, S, V)
        batch_size, seq_len = X_batch.shape
        one_hot = np.zeros_like(probs)
        rows = np.arange(batch_size)[:, None]
        cols = np.arange(seq_len)[None, :]
        one_hot[rows, cols, y_batch] = 1.0

        n = batch_size * seq_len
        delta = (probs - one_hot) / n
        self.network.Backward(None, output_delta=delta)
        self.network.update()

        probs_clipped = np.clip(probs, 1e-12, 1.0)
        loss = -np.sum(one_hot * np.log(probs_clipped)) / n
        return float(loss)

    def Train(self, texts: List[str], epochs: int = 10, batch_size: int = 32,
              seq_len: Optional[int] = None, stride: int = 1, verbose: bool = True,
              callbacks: Optional[List[Any]] = None) -> List[float]:
        """callbacks: optional list of duck-typed callback objects (same
        convention as NeuralNet.Train's `callbacks`). Supported hooks:
          on_batch_end(epoch, batch_idx, loss, model=self) -- after every
            minibatch's train_step.
          on_epoch_end(epoch, logs, model=self) -- once per epoch, with
            logs={"loss": avg_loss}. `model=self` is this TextGenerator (not
            the underlying NeuralNet), so `model.network`/`model.tokenizer`
            are available for checkpointing/generation-sample callbacks.
          on_train_end(history) -- once after the epoch loop.
        Missing methods are skipped (no error). Use this (plus
        `model.network.get_weights()`/`set_weights()` inside a callback) for
        checkpointing/resuming a run instead of writing a manual training
        loop from scratch."""
        X, y = self.prepare_sequences(texts, seq_len=seq_len, stride=stride)
        n_samples = X.shape[0]
        history = []
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            for batch_idx, start in enumerate(range(0, n_samples, batch_size)):
                idx = indices[start:start + batch_size]
                loss = self.train_step(X[idx], y[idx])
                epoch_loss += loss * len(idx)
                for cb in (callbacks or []):
                    getattr(cb, "on_batch_end", lambda *a, **k: None)(epoch, batch_idx, loss, model=self)
            avg_loss = epoch_loss / n_samples
            history.append(avg_loss)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")
            for cb in (callbacks or []):
                getattr(cb, "on_epoch_end", lambda *a, **k: None)(epoch, {"loss": avg_loss}, model=self)
        for cb in (callbacks or []):
            getattr(cb, "on_train_end", lambda *a, **k: None)(history)
        return history

    def _sample_token(self, probs: Any, temperature: float = 1.0, top_p: Optional[float] = None,
                       top_k: Optional[int] = None, greedy: bool = False) -> int:
        if greedy:
            return int(np.argmax(probs))
        p = np.asarray(probs, dtype=backend.default_dtype()).copy()
        if temperature != 1.0:
            p = np.power(np.maximum(p, 1e-12), 1.0 / temperature)
            p = p / p.sum()
        if top_k is not None:
            from .sampling import top_k_renormalize
            p = top_k_renormalize(p, top_k)
        if top_p is not None:
            from .sampling import nucleus_renormalize
            p = nucleus_renormalize(p.reshape(1, -1), top_p)[0]
        return int(np.random.choice(len(p), size=1, p=p)[0])

    def generate(self, prompt: str = "", max_new_tokens: int = 100, temperature: float = 1.0,
                 top_p: Optional[float] = None, top_k: Optional[int] = None,
                 greedy: bool = False, use_cache: bool = True) -> str:
        """Autoregressively generate text continuing `prompt`.

        use_cache: incrementally decode with a KV-cache (only recomputes the
        new token's Q each step, O(n) over the generation instead of O(n^2)
        from recomputing the full growing context every step). Requires the
        standard embedding/positional-encoding/transformer-block/layernorm/
        dense architecture built by this class; a custom network with an
        unsupported layer type raises -- pass use_cache=False for those.
        """
        start_id = self.tokenizer.word_to_idx[self.tokenizer.start_token]
        end_id = self.tokenizer.word_to_idx[self.tokenizer.end_token]
        if prompt:
            ids = [start_id] + self.tokenizer.encode(prompt, add_special_tokens=False).tolist()
        else:
            ids = [start_id]

        if not use_cache:
            for _ in range(max_new_tokens):
                context = np.array([ids[-self.max_seq_len:]], dtype=np.int64)
                probs = self.network.Forward(context, training=False)  # (1, S, V)
                next_probs = probs[0, -1]
                next_id = self._sample_token(next_probs, temperature=temperature, top_p=top_p,
                                             top_k=top_k, greedy=greedy)
                if next_id == end_id:
                    break
                ids.append(next_id)
            return self.tokenizer.decode(ids, skip_special=True)

        # The non-cached path above truncates the *context it forwards* to
        # the last max_seq_len tokens every step (but keeps the full `ids`
        # list intact for the final decode). The cached path primes the
        # cache once up front instead, so it needs the same truncation on
        # what it primes with -- otherwise a prompt (plus the start token)
        # longer than max_seq_len drives cache.position past the positional
        # table's end and raises a raw IndexError instead of this clean
        # truncation. `ids` itself is left untouched so the returned text
        # still includes the full original prompt.
        prime_ids = ids[-self.max_seq_len:] if len(ids) > self.max_seq_len else ids

        # The whole prompt primes in ONE batched step (cached_forward_step
        # applies the causal mask among the new tokens itself), which is
        # exactly equivalent to feeding it token by token but far fewer
        # matmuls; the generation loop below then steps one token at a time.
        cache = KVCache()
        probs = cached_forward_step(self.network, [prime_ids], cache)
        next_probs = probs[0, -1]

        for _ in range(max_new_tokens):
            next_id = self._sample_token(next_probs, temperature=temperature, top_p=top_p,
                                         top_k=top_k, greedy=greedy)
            if next_id == end_id:
                break
            ids.append(next_id)
            if cache.position >= self.max_seq_len:
                break
            probs = cached_forward_step(self.network, [[next_id]], cache)
            next_probs = probs[0, -1]

        return self.tokenizer.decode(ids, skip_special=True)

    def generate_beam(self, prompt: str = "", beam_width: int = 5,
                       max_new_tokens: int = 50, length_penalty: float = 1.0,
                       top_k: Optional[int] = None, use_cache: bool = True) -> str:
        """Beam search: keep `beam_width` candidates by cumulative
        log-probability, extend each by one token per step, and return the
        best by length-normalized score.

        top_k restricts how many tokens each beam may expand to per step
        (default `beam_width`). A larger top_k explores more of the
        distribution per step at proportionally more scoring work; it does
        not change how many beams survive.
        use_cache decodes incrementally, reordering the KV cache by parent
        beam after each prune. Same result, O(n) instead of O(n^2)."""
        start_id = self.tokenizer.word_to_idx[self.tokenizer.start_token]
        end_id = self.tokenizer.word_to_idx[self.tokenizer.end_token]
        if prompt:
            init_ids = [start_id] + self.tokenizer.encode(prompt, add_special_tokens=False).tolist()
        else:
            init_ids = [start_id]
        # `top_k or beam_width` would read 0 as "unset" and silently fall
        # back rather than rejecting it.
        expand = beam_width if top_k is None else int(top_k)
        if expand < 1:
            raise ValueError(f"top_k must be >= 1 (or None), got {top_k}")
        if beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {beam_width}")

        if use_cache:
            return self._beam_cached(init_ids, beam_width, max_new_tokens,
                                     length_penalty, expand, end_id)

        def score(c: Tuple[List[int], float, bool]) -> float:
            ids, logprob, _ = c
            return logprob / (len(ids) ** length_penalty)

        # Each beam: (token_ids, cumulative log-prob, finished)
        beams = [(init_ids, 0.0, False)]
        for _ in range(max_new_tokens):
            active = [(ids, logprob) for ids, logprob, finished in beams if not finished]
            if not active:
                break

            # All active beams extend by exactly one token per step, so
            # within a single step they're always the same length -- stack
            # them into one batched Forward call instead of one call per
            # beam.
            contexts = np.array([ids[-self.max_seq_len:] for ids, _ in active], dtype=np.int64)
            all_probs = self.network.Forward(contexts, training=False)[:, -1, :]
            all_probs = np.clip(all_probs, 1e-12, 1.0)

            candidates = [(ids, logprob, True) for ids, logprob, finished in beams if finished]
            for (ids, logprob), probs in zip(active, all_probs):
                top_idx = np.argsort(-probs)[:expand]
                for next_id in top_idx:
                    next_id = int(next_id)
                    new_ids = ids + [next_id]
                    new_logprob = logprob + float(np.log(probs[next_id]))
                    candidates.append((new_ids, new_logprob, next_id == end_id))

            candidates.sort(key=score, reverse=True)
            beams = candidates[:beam_width]

        best_ids = max(beams, key=score)[0] if beams else init_ids
        return self.tokenizer.decode(best_ids, skip_special=True)

    def _beam_cached(self, init_ids: List[int], beam_width: int,
                      max_new_tokens: int, length_penalty: float,
                      expand: int, end_id: int) -> str:
        """KV-cached beam search.

        Starts from ONE beam and lets the cache grow to however many
        candidates survive each prune -- `reorder` is a gather, so it widens
        and narrows the batch alike. Every beam advances one token per step,
        so they stay the same length and share one cache; a beam that has
        emitted the end token is pinned to it with an unchanged score, which
        keeps it comparable without letting it accumulate more."""
        prime = init_ids[-self.max_seq_len:]
        cache = KVCache()
        probs = cached_forward_step(self.network, [prime], cache)[:, -1, :]

        sequences = [list(init_ids)]
        logprobs = [0.0]
        finished = [False]

        for _ in range(max_new_tokens):
            if all(finished) or cache.position >= self.max_seq_len:
                break
            candidates = []               # (score, parent, token, logprob, done)
            for b in range(len(sequences)):
                if finished[b]:
                    candidates.append(
                        (logprobs[b] / (len(sequences[b]) ** length_penalty),
                         b, end_id, logprobs[b], True))
                    continue
                row = np.clip(probs[b], 1e-12, 1.0)
                for token in np.argsort(-row)[:expand]:
                    token = int(token)
                    lp = logprobs[b] + float(np.log(row[token]))
                    length = len(sequences[b]) + 1
                    candidates.append((lp / (length ** length_penalty), b, token,
                                       lp, token == end_id))
            candidates.sort(key=lambda c: c[0], reverse=True)
            chosen = candidates[:beam_width]

            cache.reorder([c[1] for c in chosen])
            sequences = [sequences[c[1]] + [c[2]] for c in chosen]
            logprobs = [c[3] for c in chosen]
            finished = [c[4] for c in chosen]
            probs = cached_forward_step(self.network,
                                        [[c[2]] for c in chosen], cache)[:, -1, :]

        best = max(range(len(sequences)),
                   key=lambda b: logprobs[b] / (len(sequences[b]) ** length_penalty))
        return self.tokenizer.decode(sequences[best], skip_special=True)

    def generate_batch(self, prompts: List[str], max_new_tokens: int = 100,
                        temperature: float = 1.0, top_p: Optional[float] = None,
                        top_k: Optional[int] = None, greedy: bool = False) -> List[str]:
        """Generate a continuation for several prompts at once, sharing one
        batched KV cache. Returns one string per prompt, in the order given.

        Prompts are grouped by TOKEN LENGTH and each group decoded as its own
        exact batch. Padding to a common width would be wrong rather than
        merely wasteful here: `nn/` attention has no padding mask, so pad
        tokens would be attended to, and left-padding additionally shifts
        every real token's absolute position. Same-length prompts need
        neither, so grouping keeps the batching benefit and stays exactly
        equal to generating each prompt on its own."""
        if not prompts:
            return []
        start_id = self.tokenizer.word_to_idx[self.tokenizer.start_token]

        encoded = []
        for prompt in prompts:
            ids = [start_id]
            if prompt:
                ids += self.tokenizer.encode(prompt, add_special_tokens=False).tolist()
            encoded.append(ids[-self.max_seq_len:])

        groups: dict = {}
        for i, ids in enumerate(encoded):
            groups.setdefault(len(ids), []).append(i)

        results: List[Optional[str]] = [None] * len(prompts)
        for members in groups.values():
            texts = self._generate_group([encoded[i] for i in members],
                                         max_new_tokens, temperature, top_p,
                                         top_k, greedy)
            for i, text in zip(members, texts):
                results[i] = text
        return [r for r in results]

    def _generate_group(self, encoded: List[List[int]], max_new_tokens: int,
                         temperature: float, top_p: Optional[float],
                         top_k: Optional[int], greedy: bool) -> List[str]:
        """Decode a batch of EQUAL-LENGTH prompts through one shared cache."""
        end_id = self.tokenizer.word_to_idx[self.tokenizer.end_token]
        cache = KVCache()
        probs = cached_forward_step(self.network, np.array(encoded, dtype=np.int64),
                                    cache)[:, -1, :]
        outputs = [list(ids) for ids in encoded]
        done = [False] * len(encoded)

        for _ in range(max_new_tokens):
            if all(done) or cache.position >= self.max_seq_len:
                break
            step = []
            for b in range(len(encoded)):
                if done[b]:
                    step.append(end_id)
                    continue
                token = self._sample_token(probs[b], temperature=temperature,
                                           top_p=top_p, top_k=top_k, greedy=greedy)
                if token == end_id:
                    done[b] = True
                    step.append(end_id)
                else:
                    outputs[b].append(token)
                    step.append(token)
            probs = cached_forward_step(self.network, [[t] for t in step],
                                        cache)[:, -1, :]
        return [self.tokenizer.decode(ids, skip_special=True) for ids in outputs]

    def perplexity(self, text: str) -> float:
        """Average per-token cross-entropy over `text`, exponentiated -- the
        standard language-model held-out evaluation metric."""
        ids = self.tokenizer.encode(text, add_special_tokens=True).tolist()
        if len(ids) < 2:
            raise ValueError("Need at least 2 tokens (after tokenization) to compute perplexity.")
        seq_len = min(self.max_seq_len - 1, len(ids) - 1)
        X, y = create_sliding_windows(np.array(ids, dtype=np.int64), seq_len, stride=seq_len)
        probs = self.network.Forward(X, training=False)  # (N, seq_len, V)
        probs_clipped = np.clip(probs, 1e-12, 1.0)
        rows = np.arange(y.shape[0])[:, None]
        cols = np.arange(y.shape[1])[None, :]
        token_probs = probs_clipped[rows, cols, y]
        avg_nll = -np.mean(np.log(token_probs))
        return float(np.exp(avg_nll))
