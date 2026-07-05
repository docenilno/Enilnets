import numpy as np
from ..base import NeuralNet
from ..text_utils import Tokenizer, create_sliding_windows
from ..activations import activate


class TextGenerator:
    """A small GPT-style causal transformer for character- or word-level text
    generation, built entirely from Enilnets' own layers (embedding,
    sinusoidal positional encoding, causal multi-head attention, transformer
    blocks). Trains by next-token prediction and samples autoregressively
    with temperature / top-p control.

    Example
    -------
    >>> tok = Tokenizer(vocab_size=128, level="char").fit([corpus_text])
    >>> gen = TextGenerator(tok, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=64)
    >>> gen.Train([corpus_text], epochs=20, batch_size=32, seq_len=32, verbose=False)
    >>> gen.generate(prompt="once upon a", max_new_tokens=100, temperature=0.8, top_p=0.9)
    """

    def __init__(self, tokenizer, embed_dim=64, num_heads=4, num_layers=2,
                 mlp_ratio=4.0, dropout=0.0, activation="gelu",
                 max_seq_len=128, learning_rate=3e-4, optimizer="adam", l2_lambda=0.0):
        if not isinstance(tokenizer, Tokenizer) or not tokenizer.fitted:
            raise ValueError("tokenizer must be a fitted Enilnets.text_utils.Tokenizer")
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

    def prepare_sequences(self, texts, seq_len=None, stride=1):
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

    def train_step(self, X_batch, y_batch):
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

    def Train(self, texts, epochs=10, batch_size=32, seq_len=None, stride=1, verbose=True,
              callbacks=None):
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

    def _sample_token(self, probs, temperature=1.0, top_p=None, top_k=None, greedy=False):
        if greedy:
            return int(np.argmax(probs))
        p = np.asarray(probs, dtype=np.float64).copy()
        if temperature != 1.0:
            p = np.power(np.maximum(p, 1e-12), 1.0 / temperature)
            p = p / p.sum()
        if top_k is not None:
            k = min(top_k, len(p))
            keep = np.argpartition(p, -k)[-k:]
            mask = np.zeros_like(p)
            mask[keep] = p[keep]
            p = mask / mask.sum()
        if top_p is not None:
            order = np.argsort(p)[::-1]
            sorted_p = p[order]
            cumsum = np.cumsum(sorted_p)
            cutoff = max(1, int(np.searchsorted(cumsum, top_p) + 1))
            keep = order[:cutoff]
            mask = np.zeros_like(p)
            mask[keep] = p[keep]
            p = mask / mask.sum()
        return int(np.random.choice(len(p), p=p))

    def _kv_step(self, token_id, cache):
        """Run one new token through the network using cached per-layer K/V
        (multihead_attention) and residual-save values, appending this step's
        K/V to the cache. Only valid for a network built the standard way
        (embedding -> positional_encoding -> transformer blocks -> layernorm
        -> dense); any other layer type raises. Returns (1,1,vocab) probs."""
        layers = self.network.layers
        x = np.array([[token_id]], dtype=np.int64)
        position = cache["position"]
        for idx, layer in enumerate(layers):
            t = layer["type"]
            if t == "embedding":
                x = layer["weights"][x]  # (1,1,E)
            elif t == "positional_encoding":
                if layer.get("_pos_type") == "learnable":
                    x = x + layer["weights"][position].reshape(1, 1, -1)
                else:
                    x = x + layer["pe"][position].reshape(1, 1, -1)
            elif t == "layernorm":
                eps = layer.get("epsilon", 1e-5)
                mean = np.mean(x, axis=-1, keepdims=True)
                var = np.var(x, axis=-1, keepdims=True)
                x_norm = (x - mean) / np.sqrt(var + eps)
                x = layer["gamma"].reshape(1, 1, -1) * x_norm + layer["beta"].reshape(1, 1, -1)
            elif t == "dense":
                z = np.dot(x, layer["weights"].T) + layer["bias"]
                x = activate(layer["activation"], z, **layer.get("activation_params", {}))
            elif t == "dropout":
                pass  # inference: no-op
            elif t == "multihead_attention":
                H, Dh, E = layer["num_heads"], layer["head_dim"], layer["embed_dim"]
                Q = np.dot(x, layer["Wq"].T) + layer["bq"]
                K_new = np.dot(x, layer["Wk"].T) + layer["bk"]
                V_new = np.dot(x, layer["Wv"].T) + layer["bv"]
                Qh = Q.reshape(1, 1, H, Dh).transpose(0, 2, 1, 3)
                Kh_new = K_new.reshape(1, 1, H, Dh).transpose(0, 2, 1, 3)
                Vh_new = V_new.reshape(1, 1, H, Dh).transpose(0, 2, 1, 3)
                prev = cache["kv"].get(idx)
                if prev is None:
                    Kh_all, Vh_all = Kh_new, Vh_new
                else:
                    Kh_all = np.concatenate([prev[0], Kh_new], axis=2)
                    Vh_all = np.concatenate([prev[1], Vh_new], axis=2)
                cache["kv"][idx] = (Kh_all, Vh_all)
                scores = np.matmul(Qh, Kh_all.transpose(0, 1, 3, 2)) / np.sqrt(Dh)
                scores_max = np.max(scores, axis=-1, keepdims=True)
                exp_scores = np.exp(scores - scores_max)
                attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
                context = np.matmul(attn, Vh_all).transpose(0, 2, 1, 3).reshape(1, 1, E)
                x = np.dot(context, layer["Wo"].T) + layer["bo"]
            elif t == "residual_save":
                cache["residual"][idx] = x
            elif t == "residual_add":
                x = x + cache["residual"][layer["save_index"]]
            else:
                raise ValueError(
                    f"KV-cache generation doesn't support layer type '{t}'; "
                    f"use generate(..., use_cache=False) for arbitrary networks."
                )
        return x

    def generate(self, prompt="", max_new_tokens=100, temperature=1.0, top_p=None, top_k=None,
                 greedy=False, use_cache=True):
        """Autoregressively generate text continuing `prompt`.

        use_cache: incrementally decode with a KV-cache (only recomputes the
        new token's Q each step, O(n) over the generation instead of O(n^2)
        from recomputing the full growing context every step). Requires the
        standard embedding/positional-encoding/transformer-block/layernorm/
        dense architecture built by this class; falls back automatically is
        not attempted for custom networks -- pass use_cache=False for those.
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

        cache = {"kv": {}, "residual": {}, "position": 0}
        for tid in ids:
            probs = self._kv_step(tid, cache)
            cache["position"] += 1
        next_probs = probs[0, 0]

        for _ in range(max_new_tokens):
            next_id = self._sample_token(next_probs, temperature=temperature, top_p=top_p,
                                         top_k=top_k, greedy=greedy)
            if next_id == end_id:
                break
            ids.append(next_id)
            if cache["position"] >= self.max_seq_len:
                break
            probs = self._kv_step(next_id, cache)
            cache["position"] += 1
            next_probs = probs[0, 0]

        return self.tokenizer.decode(ids, skip_special=True)

    def generate_beam(self, prompt="", beam_width=5, max_new_tokens=50, length_penalty=1.0):
        """Beam search decoding: maintains `beam_width` candidate sequences by
        cumulative log-probability, extending each by one token per step and
        keeping the top `beam_width` by length-normalized score. Returns the
        best completed (or max-length) sequence as decoded text."""
        start_id = self.tokenizer.word_to_idx[self.tokenizer.start_token]
        end_id = self.tokenizer.word_to_idx[self.tokenizer.end_token]
        if prompt:
            init_ids = [start_id] + self.tokenizer.encode(prompt, add_special_tokens=False).tolist()
        else:
            init_ids = [start_id]

        def score(c):
            ids, logprob, _ = c
            return logprob / (len(ids) ** length_penalty)

        # Each beam: (token_ids, cumulative log-prob, finished)
        beams = [(init_ids, 0.0, False)]
        for _ in range(max_new_tokens):
            if all(finished for _, _, finished in beams):
                break
            candidates = []
            for ids, logprob, finished in beams:
                if finished:
                    candidates.append((ids, logprob, finished))
                    continue
                context = np.array([ids[-self.max_seq_len:]], dtype=np.int64)
                probs = self.network.Forward(context, training=False)[0, -1]
                probs = np.clip(probs, 1e-12, 1.0)
                top_idx = np.argpartition(probs, -beam_width)[-beam_width:]
                for next_id in top_idx:
                    next_id = int(next_id)
                    new_ids = ids + [next_id]
                    new_logprob = logprob + float(np.log(probs[next_id]))
                    candidates.append((new_ids, new_logprob, next_id == end_id))

            candidates.sort(key=score, reverse=True)
            beams = candidates[:beam_width]

        best_ids = max(beams, key=score)[0] if beams else init_ids
        return self.tokenizer.decode(best_ids, skip_special=True)

    def perplexity(self, text):
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
