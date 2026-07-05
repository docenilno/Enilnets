# Enilnets

GitHub: https://github.com/docenilno/Enilnets

Enilnets is a neural network library built entirely on NumPy — no PyTorch, no
TensorFlow, no GPU, no C extensions. Every layer, optimizer, loss, generative
model, RL algorithm, evolutionary algorithm, and utility is implemented from
scratch with plain array math, so every line of what happens to your data and
gradients is readable Python + NumPy.

**Version 3.0.0.** This document is exhaustive by design: every layer type,
every function signature, every configuration knob, and — importantly —
every known limitation and rough edge, so you know exactly what you're
building on before you rely on it.

## Table of contents

- [Install](#install)
- [Quickstart](#quickstart)
- [Examples](#examples)
  - [MNIST (image classification)](#mnist-image-classification)
  - [CIFAR-10 (deeper conv classifier)](#cifar-10-deeper-conv-classifier)
  - [Text generation](#example-text-generation)
  - [Reinforcement learning](#example-reinforcement-learning)
  - [GAN](#example-gan)
  - [Diffusion](#example-diffusion)
- [Core concepts](#core-concepts)
  - [The `NeuralNet` object](#the-neuralnet-object)
  - [Auto shape inference](#auto-shape-inference)
  - [The layer dict / `model.layers`](#the-layer-dict--modellayers)
- [Every layer type](#every-layer-type)
  - [Dense](#dense)
  - [Sparse](#sparse)
  - [Conv2D](#conv2d)
  - [Pooling (max / avg / global-avg)](#pooling-max--avg--global-avg)
  - [Upsample2D](#upsample2d)
  - [Flatten](#flatten)
  - [BatchNorm](#batchnorm)
  - [LayerNorm](#layernorm)
  - [Dropout](#dropout)
  - [Embedding](#embedding)
  - [Multi-head attention](#multi-head-attention)
  - [Positional encoding](#positional-encoding)
  - [Transformer block](#transformer-block)
  - [Vision Transformer patch embedding](#vision-transformer-patch-embedding)
  - [RNN / LSTM / GRU](#rnn--lstm--gru)
  - [Residual / skip connections](#residual--skip-connections)
  - [Convenience block builders](#convenience-block-builders)
- [Activations](#activations)
- [Weight initialization](#weight-initialization)
- [Losses](#losses)
- [Optimizers](#optimizers)
- [Training](#training)
  - [`TrainBatch` / `Train`](#trainbatch--train)
  - [Gradient clipping](#gradient-clipping)
  - [Gradient accumulation](#gradient-accumulation)
  - [Mixed precision](#mixed-precision)
  - [Learning rate schedules](#learning-rate-schedules)
  - [Early stopping](#early-stopping)
  - [Accuracy / precision / recall / F1](#accuracy--precision--recall--f1)
- [Text generation (`TextGenerator`)](#text-generation-textgenerator)
- [Generative models](#generative-models)
  - [VAE](#vae)
  - [GAN](#gan)
  - [DiffusionModel](#diffusionmodel)
  - [RealNVP (normalizing flow)](#realnvp-normalizing-flow)
  - [EnergyBasedModel](#energybasedmodel)
  - [AutoregressiveModel](#autoregressivemodel)
  - [UNetDenoiser](#unetdenoiser)
  - [Low-level sampling & loss building blocks](#low-level-sampling--loss-building-blocks)
- [Reinforcement learning](#reinforcement-learning)
- [NEAT (neuroevolution)](#neat-neuroevolution)
- [Visualization](#visualization)
- [Evaluation utilities](#evaluation-utilities)
- [Data utilities](#data-utilities)
  - [General (`utils.py`)](#general-utilspy)
  - [Text (`text_utils.py`)](#text-text_utilspy)
  - [Images (`image_utils.py`)](#images-image_utilspy)
  - [Audio (`audio_utils.py`)](#audio-audio_utilspy)
  - [Cross-modal (`crossmodal_utils.py`)](#cross-modal-crossmodal_utilspy)
- [Model persistence](#model-persistence)
- [Configuration system](#configuration-system)
- [Full API index](#full-api-index)
- [Known limitations](#known-limitations)
- [Running the test suite](#running-the-test-suite)

## Install

Enilnets has no dependencies beyond NumPy.

```bash
pip install enilnets
```

```python
import Enilnets
print(Enilnets.__version__)  # "3.0.0"
```

## Quickstart

```python
import numpy as np
from Enilnets import NeuralNet, set_seed, train_test_split

set_seed(0)

X = np.random.randn(500, 20)
y = (X[:, 0] + X[:, 1] > 0).astype(np.float64).reshape(-1, 1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, seed=0)

model = NeuralNet(learning_rate=0.001, optimizer="adam", l2_lambda=0.001)
model.add_dense(20, 64, activation="relu")
model.add_dense(64, 32, activation="relu")
model.add_dense(32, 1, activation="sigmoid")

history = model.Train(X_train, y_train, epochs=20, batch_size=32,
                       X_val=X_val, Y_val=y_val, loss_function="binary_cross_entropy")

preds = model.Forward(X_val, training=False)
print("val accuracy:", model.compute_accuracy(preds, y_val))
```

## Examples

Six compact, runnable examples covering the most common use cases. Each one
uses synthetic data shaped like the real dataset it names (so you can
copy-paste and run immediately with no downloads) — swap in real
`X_train`/`y_train` for actual results.

### MNIST (image classification)

```python
import numpy as np
from Enilnets import NeuralNet

np.random.seed(0)
# Replace with real MNIST: X_train (N,1,28,28) in [0,1], y_train one-hot (N,10)
X_train = np.random.rand(200, 1, 28, 28)
y_train = np.eye(10)[np.random.randint(0, 10, 200)]
X_val, y_val = X_train[:40], y_train[:40]

model = NeuralNet(learning_rate=0.001, optimizer="adam")
model.add_conv_block(out_ch=16, k=3, in_ch=1, batchnorm=True, pool="max", input_size=(28, 28))
model.add_conv_block(out_ch=32, k=3, batchnorm=True, pool="max")
model.add_flatten()
model.add_dense(n_out=64, activation="relu")
model.add_dense(n_out=10, activation="softmax")

model.Train(X_train, y_train, epochs=3, batch_size=32, X_val=X_val, Y_val=y_val,
            loss_function="cross_entropy", verbose=True)

preds = model.Forward(X_val, training=False)
print("val accuracy:", model.compute_accuracy(preds, y_val))
```

### CIFAR-10 (deeper conv classifier)

```python
import numpy as np
from Enilnets import NeuralNet

np.random.seed(0)
# Replace with real CIFAR-10: X_train (N,3,32,32) in [0,1], y_train one-hot (N,10)
X_train = np.random.rand(200, 3, 32, 32)
y_train = np.eye(10)[np.random.randint(0, 10, 200)]
X_val, y_val = X_train[:40], y_train[:40]

model = NeuralNet(learning_rate=0.001, optimizer="adamw", l2_lambda=0.0001)
model.add_conv_block(out_ch=32, k=3, in_ch=3, batchnorm=True, pool="max", input_size=(32, 32))
model.add_conv_block(out_ch=64, k=3, batchnorm=True, pool="max")
model.add_conv_block(out_ch=128, k=3, batchnorm=True, pool="max")
model.add_global_avgpool2d()
model.add_flatten()
model.add_dense(n_out=10, activation="softmax")

model.Train(X_train, y_train, epochs=3, batch_size=32, X_val=X_val, Y_val=y_val,
            loss_function="cross_entropy", verbose=True)

preds = model.Forward(X_val, training=False)
print("val accuracy:", model.compute_accuracy(preds, y_val))
```

### Example: text generation

```python
import numpy as np
from Enilnets import TextGenerator, Tokenizer

np.random.seed(0)
corpus = "the quick brown fox jumps over the lazy dog. " * 50

tokenizer = Tokenizer(vocab_size=128, level="char").fit([corpus])
gen = TextGenerator(tokenizer, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=64)
gen.Train([corpus], epochs=10, batch_size=32, seq_len=32, verbose=True)

print(gen.generate(prompt="the quick", max_new_tokens=50, temperature=0.8, top_p=0.9))
print("perplexity:", gen.perplexity(corpus))
```

### Example: reinforcement learning

REINFORCE on a toy one-step-memory task (reward is 1 if the chosen action
matches the sign of the current state's first feature):

```python
import numpy as np
from Enilnets import NeuralNet, compute_returns

np.random.seed(0)
state_dim, n_actions = 4, 2
policy = NeuralNet(learning_rate=0.01, optimizer="adam")
policy.add_dense(state_dim, 32, activation="relu")
policy.add_dense(32, n_actions, activation="softmax")

def run_episode():
    states, actions, rewards = [], [], []
    state = np.random.randn(state_dim)
    for _ in range(20):
        probs = policy.Forward(state, training=False)[0]
        action = np.random.choice(n_actions, p=probs)
        reward = 1.0 if (action == 0) == (state[0] > 0) else -1.0
        states.append(state); actions.append(action); rewards.append(reward)
        state = np.random.randn(state_dim)
    return np.array(states), np.array(actions), np.array(rewards)

for episode in range(200):
    states, actions, rewards = run_episode()
    returns = compute_returns(rewards, gamma=0.95)
    policy.Reinforce(states, actions, returns, action_type="discrete")
    if episode % 50 == 0:
        print(f"episode {episode} - total reward: {rewards.sum():.1f}")
```

### Example: GAN

```python
import numpy as np
from Enilnets import GAN

np.random.seed(0)
# Toy 2D "real" data: a ring of points
theta = np.random.uniform(0, 2 * np.pi, 500)
X_train = np.stack([np.cos(theta), np.sin(theta)], axis=1) + np.random.randn(500, 2) * 0.05

gan = GAN(latent_dim=8, data_dim=2, generator_hidden=[32, 32], discriminator_hidden=[32, 32],
          loss_type="wasserstein", learning_rate=0.0005, wgan_clip_value=0.01)

gan.Train(X_train, epochs=30, batch_size=64, d_steps=3, g_steps=1, verbose=True)

samples = gan.sample(10)
print("generated samples:\n", samples)
print("mode collapse score (0=collapsed, 1=diverse):", gan.mode_collapse_score())
```

### Example: diffusion

```python
import numpy as np
from Enilnets import DiffusionModel

np.random.seed(0)
# Toy 2D "real" data: a noisy ring of points
theta = np.random.uniform(0, 2 * np.pi, 500)
X_train = np.stack([np.cos(theta), np.sin(theta)], axis=1) + np.random.randn(500, 2) * 0.05

diffusion = DiffusionModel(data_shape=(2,), time_steps=20, beta_schedule="linear",
                            beta_start=1e-4, beta_end=0.1, denoiser_hidden=[64, 64],
                            learning_rate=0.002, use_ema=True)

diffusion.Train(X_train, epochs=100, batch_size=64, verbose=True)

samples = diffusion.sample(n_samples=10)
print("sample radii (should trend toward 1.0, the ring's radius):")
print(np.linalg.norm(samples, axis=1))
```

## Core concepts

### The `NeuralNet` object

Everything in the "classic layer stack" half of the library revolves around
one class, `NeuralNet`. It holds:

- `model.layers` — an ordered list of plain Python dicts (see below), the
  entire model definition and all its learned parameters.
- `model.opt_state` — per-layer optimizer state (Adam moments, RMSprop
  accumulators, etc.), built lazily on first `update()`/`apply_gradients()`.
- Forward-pass caches (`model.outputs`, `model.pre_activations`,
  `model.attention_cache`, `model.rnn_cache`, `model.conv_cache`,
  `model.batchnorm_cache`, `model.layernorm_cache`) — populated by
  `Forward()`, consumed by `Backward()`. You generally don't touch these
  directly, but they're plain lists/arrays if you need to inspect them.
- `model.deltas` — per-layer gradients w.r.t. each layer's *output*,
  populated by `Backward()`.

Constructor:

```python
NeuralNet(learning_rate=0.001, optimizer="adam", l2_lambda=0.01, momentum=0.9,
          grad_clip_norm=0.0, use_mixed_precision=False,
          adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
          rmsprop_decay=0.9, rmsprop_epsilon=1e-8, adagrad_epsilon=1e-8)
```

Every one of these is a plain attribute you can also read/write after
construction (`model.learning_rate = 0.0001`, etc.) — there are no hidden
private copies. `optimizer` is lower-cased and stored as `optimizer_type`;
**any string other than `"sgd"`, `"rmsprop"`, `"adagrad"` silently behaves as
Adam** (including `"adamw"`, which shares Adam's moment update but adds
decoupled weight decay, and including typos — there's no validation error
for an unrecognized optimizer name).

Other `NeuralNet` methods not tied to a specific layer or training call:

| Method | What it does |
|---|---|
| `train()` / `eval()` | Set `self.training = True`/`False`; both return `self` (chainable). Layers that behave differently in train vs. eval (dropout, batchnorm) read `training` from the `Forward(training=...)` argument, not this flag directly — `train()`/`eval()` just set a convenience default you can check yourself. |
| `set_lr(lr)` / `get_lr()` | Get/set `learning_rate`. |
| `clip_gradients(max_norm)` | Global L2-norm clipping across `self.deltas` in place. No-op if `max_norm <= 0`. Called automatically by `TrainBatch` when `grad_clip_norm > 0`. |
| `freeze(layer_idx=None)` / `unfreeze(layer_idx=None)` | Mark one layer (or all, if `None`) with `_frozen=True/False`; `apply_gradients`/`update` skip frozen layers entirely. |
| `get_weights()` / `set_weights(weights)` | Snapshot/restore just the weight arrays (not optimizer state) as a list of dicts — lighter-weight than full `Save`/`Load` for e.g. EMA-style weight swapping. |
| `copy()` | Deep-copies the entire model (layers, optimizer state, step counter, shape-inference bookkeeping, residual stack) into a new independent `NeuralNet`. |
| `reset_optimizer_state()` | Clears `opt_state`, `t`, and gradient-accumulation buffers (e.g. before fine-tuning with a different optimizer). |
| `check_nan_inf()` | Returns a list of human-readable strings describing any NaN/Inf found in weights/biases/gamma/beta/deltas — empty list means clean. |
| `summary()` | Prints a layer-by-layer shape/parameter-count table to stdout. Returns `None` — for a value you can use in code, see `count_parameters(model)` in [Data utilities](#general-utilspy). |
| `predict(x)` | A plain alias for `Forward(x, training=False, dropout_rate=0.0)` — same signature, same code path, just a familiar name. |

### Auto shape inference

Every `add_*` layer method infers its input size from whatever was added
before it, tracked internally via `model._last_width` (feature width) and
`model._last_spatial` (a `(C, H, W)` tuple, tracked separately for
conv/pool/flatten chains). You only need to specify the input size on the
very first layer:

```python
model = NeuralNet()
model.add_dense(784, 256, activation="relu")     # n_in=784 required (first layer)
model.add_dense(n_out=128, activation="relu")    # n_in auto-inferred as 256
model.add_dense(n_out=10, activation="softmax")  # n_in auto-inferred as 128
```
(`n_in` is always the first positional argument, so pass `n_out` as a
keyword — or `None` positionally, e.g. `add_dense(None, 128, ...)` — to
trigger auto-inference; a bare `add_dense(128, ...)` would set `n_in=128`,
not `n_out`.)

```python
model = NeuralNet()
model.add_conv2d(in_ch=3, out_ch=16, k=3, input_size=(32, 32))
model.add_maxpool2d(2)
model.add_conv2d(out_ch=32, k=3)               # in_ch inferred as 16
model.add_flatten()                             # feature width inferred from (C, H, W)
model.add_dense(n_out=10, activation="softmax") # n_in inferred from flatten
```

Calling an `add_*` method that needs inference with no previous layer (and
no explicit size given) raises `ValueError` with a message telling you which
argument to supply.

### The layer dict / `model.layers`

Each entry in `model.layers` is a plain dict with at minimum a `"type"` key
(`"dense"`, `"conv2d"`, `"lstm"`, etc.) plus whatever weight arrays and
config that layer type needs (e.g. `"weights"`/`"bias"` for dense,
`"Wq"/"bq"/"Wk"/...` for attention, `"Wx"/"Wh"/"b"` for RNN/LSTM,
`"Wx"/"Wh"/"bx"/"bh"` for GRU). This is intentionally simple and inspectable
— there's no hidden object wrapping, so `model.layers[0]["weights"].shape`
always just works, and `Save`/`Load` (JSON or pickle) round-trip these dicts
directly.

## Every layer type

All `add_*` methods are `NeuralNet` instance methods; call them in sequence
to build up `model.layers`.

### Dense

```python
add_dense(n_in=None, n_out=128, activation="relu", init_method="xavier_uniform",
          use_bias=True, activation_params=None)
```
A standard fully-connected layer: `y = activation(x @ W.T + b)`. `n_in=None`
auto-infers. `activation_params` is a dict forwarded to the activation
function (e.g. `{"alpha": 0.2}` for leakyrelu).

- **Pros:** the fastest, best-understood, most numerically stable layer in
  the library; supports 2D `(batch, features)` and 3D `(batch, seq, features)`
  input transparently (weight-gradient reduction flattens leading dims).
- **Cons:** no parameter sharing — doesn't scale to large spatial/sequential
  inputs the way conv/attention/RNN do.

### Sparse

```python
add_sparse(n_in=None, n_out=128, connectivity=0.5, activation="relu",
           init_method="xavier_uniform", activation_params=None)
```
A dense layer with a fixed random binary connectivity mask
(`Bernoulli(connectivity)`), applied once at construction and reapplied to
every weight gradient forever after — the sparsity pattern **never changes**
during training (only reduces effective parameter count / provides a crude
regularization prior, not a real dynamic-sparsity method like NEAT).

- **Pros:** fewer effective parameters than dense at the same width; simple
  way to experiment with fixed-sparsity architectures.
- **Cons:** the mask is random and static — no structured sparsity, no
  pruning-during-training, no guarantee the fixed pattern is a good one.

### Conv2D

```python
add_conv2d(in_ch=None, out_ch=32, k=3, activation="relu", init_method="he_normal",
           stride=1, activation_params=None, input_size=None)
```
2D convolution implemented via `im2col` + matrix multiply (the standard
NumPy-conv trick — turns convolution into one big `dot()` call instead of
nested loops). No padding support (`k>1` shrinks spatial size); `input_size`
is only needed on the very first conv call, to let a later `add_flatten()`
compute its output width.

- **Pros:** vectorized (`im2col` + BLAS matmul), reasonably fast for a pure-
  NumPy conv; supports arbitrary stride.
- **Cons:** no built-in padding (`"same"` convolutions aren't directly
  supported — pad your input manually with `image_utils.pad_image` first);
  `im2col` materializes every patch, so memory scales with
  `batch * out_h * out_w * in_ch * k * k` — large images/kernels can be
  memory-hungry compared to a true sliding-window convolution.

### Pooling (max / avg / global-avg)

```python
add_maxpool2d(pool_size=2)
add_avgpool2d(pool_size=2)
add_global_avgpool2d()
```
Non-overlapping spatial pooling (`pool_size` is both window and stride).
`add_global_avgpool2d()` reduces the full spatial extent to `1x1` (as used
before a classification head in modern CNNs, avoiding a huge flatten+dense).

- **Pros:** cheap, reduces spatial size and overfitting risk; global-avg-pool
  makes the network robust to input spatial size changes.
- **Cons:** only non-overlapping pooling (no arbitrary stride ≠ pool_size);
  max-pool's gradient routes to the argmax position only (standard, but no
  "soft"/differentiable pooling option).

### Upsample2D

```python
add_upsample2d(scale_factor=2)
```
Nearest-neighbor upsampling (repeats pixels), used in decoders/UNet-style
architectures.

- **Pros:** simple, cheap, no learned parameters.
- **Cons:** nearest-neighbor only — no bilinear/transposed-convolution
  upsampling built into the layer API (see `image_utils.resize_bilinear` for
  a non-layer alternative on raw arrays).

### Flatten

```python
add_flatten()
```
`(B, C, H, W) → (B, C*H*W)`. Required to transition from conv/pool layers
into dense layers.

### BatchNorm

```python
add_batchnorm(num_features=None, epsilon=1e-5, momentum=0.1)
```
Standard batch normalization over the batch (+ spatial, for 4D input) axes,
with running mean/variance tracked for inference (`training=False` uses the
running stats instead of the current batch's).

- **Pros:** stabilizes/accelerates training, especially for deeper conv
  stacks; supports both 2D and 4D input.
- **Cons:** behavior depends on batch size (small batches → noisy
  statistics); running stats must be carried consistently through
  save/load (they are — see [Model persistence](#model-persistence)).

### LayerNorm

```python
add_layernorm(normalized_shape=None, epsilon=1e-5)
```
Normalizes over the feature axis per-example (independent of batch size),
supporting 2D `(batch, features)`, 3D `(batch, seq, features)` — normalizing
over the embedding axis only, standard Transformer-style — and 4D
`(batch, C, H, W)` input.

- **Pros:** batch-size-independent (unlike BatchNorm), the standard choice
  for Transformers/RNNs; works cleanly with variable sequence lengths.
- **Cons:** slightly more compute per example than BatchNorm (no shared
  running statistics to amortize at inference).

### Dropout

```python
add_dropout(rate=0.5)
```
Inverted dropout (scales surviving activations by `1/(1-rate)` at train
time, so no rescaling needed at inference). Only active when
`Forward(..., training=True)`.

- **Pros:** simple, well-understood regularizer; `rate=1.0` is handled as a
  full zero-out edge case rather than dividing by zero.
- **Cons:** like all layers, per-layer `rate` overrides the global
  `Forward(dropout_rate=...)` default — easy to forget you set a per-layer
  rate when trying to globally disable dropout for a quick experiment.

### Embedding

```python
add_embedding(vocab_size, embed_dim, init_method="normal")
```
A lookup table `(vocab_size, embed_dim)`; input is integer token
indices, `(batch, seq_len)` or `(batch,)`. Gradient is a sparse scatter-add
(`np.add.at`), not a dense matmul, so it scales with tokens actually seen in
the batch rather than full vocab size.

- **Pros:** efficient sparse gradient; works for both `(batch,)` single-token
  and `(batch, seq)` sequence input.
- **Cons:** no weight-tying helper (if you want tied input/output embeddings
  for a language model, you must manage that manually by sharing the array
  reference yourself).

### Multi-head attention

```python
add_multihead_attention(embed_dim=None, num_heads=4, dropout=0.0,
                        init_method="xavier_uniform", causal=False)
```
Standard scaled dot-product multi-head self-attention. `embed_dim` must be
divisible by `num_heads` (asserted). Stores its own `Wq/bq/Wk/bk/Wv/bv/Wo/bo`
projection weights directly on the layer (not as separate dense layers).
`causal=True` applies an autoregressive mask (position *i* attends only to
`j <= i`) — this is what `TextGenerator` builds on.

- **Pros:** the mask, softmax, and backward pass are all verified against
  finite-difference gradients; supports arbitrary `num_heads`/`embed_dim`
  combinations; causal masking has zero extra backward-pass cost (masked
  softmax entries naturally have zero gradient).
- **Cons:** no built-in support for cross-attention (encoder-decoder
  attention with a separate key/value source) — this layer is self-attention
  only; the `dropout` parameter is accepted but currently informational only
  on the attention weights themselves (no separate attention-dropout layer
  is inserted — use `add_dropout()` after the block if you need it applied
  to the output).

### Positional encoding

```python
add_positional_encoding(max_seq_len, embed_dim=None, learnable=True, base=None)
```
`learnable=True` (default) adds it as an **embedding layer** internally
(reusing `add_embedding`, tagged so it's added to the input rather than
looked up separately) — i.e. it's implemented as a special-cased embedding,
not a distinct layer type. `learnable=False` precomputes fixed sinusoidal
encodings (`base` defaults to `constants.SINUSOIDAL_BASE = 10000.0`) and
stores them as a genuinely separate `"positional_encoding"` layer type.

- **Pros:** both classic Transformer variants (fixed sinusoidal, learned)
  available with one flag.
- **Cons:** no relative positional encoding (RoPE, ALiBi, etc.) — only
  absolute position, added once at the input.

### Transformer block

```python
add_transformer_block(embed_dim=None, num_heads=4, mlp_ratio=4.0, dropout=0.0,
                      activation="swish", causal=False)
```
A full pre-norm Transformer block, expanding into: `residual_start →
layernorm → multihead_attention → residual_end → residual_start → layernorm
→ dense(hidden=embed_dim*mlp_ratio) → dense(embed_dim) → [dropout] →
residual_end`. This is what both `TextGenerator` and manual GPT/ViT-style
models are built from.

- **Pros:** correct residual wiring (verified via finite-difference gradient
  checks after a real bug was found and fixed this session — earlier
  versions of this block had no residual connections at all); one call
  builds 8+ underlying layers correctly wired.
- **Cons:** it's pre-norm only (no post-norm option); no built-in support for
  cross-attention blocks (decoder-only, not encoder-decoder, architectures).

### Vision Transformer patch embedding

```python
add_vision_transformer_patch_embed(img_size, patch_size, in_channels=None, embed_dim=768)
```
Converts `(B, C, H, W)` images into `(B, num_patches, embed_dim)` patch
tokens via a conv2d with `kernel=stride=patch_size` + flatten. Asserts
`img_size % patch_size == 0`.

- **Pros:** the standard, efficient "patchify via strided conv" approach —
  no manual patch-slicing loop.
- **Cons:** square images/patches only (`img_size`/`patch_size` are single
  ints, not `(H, W)` pairs).

### RNN / LSTM / GRU

```python
add_rnn(n_in=None, hidden_dim=128, return_sequences=True, init_method="xavier_uniform")
add_lstm(n_in=None, hidden_dim=128, return_sequences=True, init_method="xavier_uniform")
add_gru(n_in=None, hidden_dim=128, return_sequences=True, init_method="xavier_uniform")
```
All three take `(batch, seq_len, features)` input. `return_sequences=True`
outputs every timestep `(batch, seq_len, hidden)`; `False` outputs just the
last timestep `(batch, hidden)`. Backprop is full backpropagation-through-
time (BPTT), verified against finite-difference gradients over multi-
timestep sequences, for every parameter, in both `return_sequences` modes.

- `add_rnn`: single tanh gate — simplest, most prone to vanishing gradients
  over long sequences.
- `add_lstm`: `[i, f, g, o]` gates stacked into one `(4*hidden, ·)` matrix;
  forget-gate bias initialized to `1.0` (standard trick, encourages
  remembering by default early in training).
- `add_gru`: `[r, z, n]` gates stacked into a `(3*hidden, ·)` matrix, with
  **separate** input/hidden biases (`bx`/`bh`) since the reset gate only
  multiplies the hidden contribution to the candidate state, not the input
  contribution — this is why GRU layers have two bias arrays where RNN/LSTM
  have one.

- **Pros:** real BPTT (not truncated/approximated), all three variants
  available, auto-shape-inference works the same as every other layer.
- **Cons:** the per-timestep Python loop in `Forward`/`Backward` (unavoidable
  for a from-scratch recurrent implementation) means these layers are
  meaningfully slower than dense/conv for long sequences — there's no
  cuDNN-style fused kernel; no bidirectional variant built in (build one
  manually by running two RNN layers over forward/reversed input and
  concatenating).

### Residual / skip connections

```python
add_residual_start()
add_residual_end()
```
Generic, nestable skip connections: `add_residual_end()` computes
`x = x + saved_x`, where `saved_x` is whatever `x` was at the matching
`add_residual_start()`. This is the primitive `add_transformer_block` is
built from — use it directly for custom ResNet-style blocks:

```python
model.add_dense(64, 64, activation="linear")
model.add_residual_start()
model.add_dense(64, 64, activation="tanh")
model.add_dense(64, 64, activation="linear")
model.add_residual_end()          # x = x + tanh_block(x)
```

- **Pros:** generic (works around any sequence of layers, not just
  attention/MLP blocks), nestable, gradient routing verified via finite
  difference (a real off-by-one bug in the gradient routing was found and
  fixed this session).
- **Cons:** `add_residual_end()` with no matching `add_residual_start()`
  raises `ValueError` rather than silently no-op'ing — intentional (fails
  loudly on a construction mistake) but worth knowing if you're
  programmatically building/mutating layer sequences.

### Convenience block builders

```python
add_mlp_block(hidden_dims, in_dim=None, out_dim=None, activation="relu",
              out_activation="linear", init_method="xavier_uniform")
add_conv_block(out_ch, k=3, activation="relu", init_method="he_normal",
              in_ch=None, stride=1, batchnorm=False, pool=None, input_size=None)
```
`add_mlp_block([256, 128, 64], out_dim=10)` = a loop of `add_dense` calls,
saving you the boilerplate. `add_conv_block(...)` = `conv2d → [batchnorm] →
[pool]` in one call (`pool` accepts `None`, `"max"`, or `"avg"`; pool size is
fixed at 2 regardless of what you pass elsewhere).

## Activations

```python
activate(name, x, alpha=None, sigmoid_clip=None)
derivative(name, x, alpha=None, sigmoid_clip=None, cached_output=None)
```
(Used internally by every layer's `activation=` argument; not usually called
directly.) Supported `name` strings: `relu`, `leakyrelu`, `elu`, `selu`,
`gelu`, `swish`, `mish`, `sigmoid`, `tanh`, `softmax`, `softplus`, `linear`.

- `alpha` overrides `LEAKYRELU_ALPHA` (0.01) / `ELU_ALPHA` (1.0) as
  applicable; `sigmoid_clip` overrides `SIGMOID_CLIP` (500.0), the
  overflow-safe clip bound used by `sigmoid`/`softplus`'s `exp()`.
- Pass these per-layer via `activation_params={"alpha": ..., "sigmoid_clip": ...}`.
- **Known gotcha:** an unrecognized activation `name` doesn't raise — it
  silently falls through to `return x` in `activate` and
  `return np.ones_like(x)` in `derivative`. A typo like `"realu"` will run
  without error and silently behave as a linear layer. Double-check spelling.

## Weight initialization

```python
init_weights(n_in, n_out, method="xavier_uniform", std=None)          # dense/attention/RNN weights
init_conv_weights(in_ch, out_ch, k, method="he_normal", std=None)      # conv2d weights
init_embedding_weights(vocab_size, embed_dim, method="normal", std=None)
```
`init_method` strings:
- `init_weights` / `init_conv_weights`: `xavier_uniform`, `xavier_normal`,
  `he_uniform`, `he_normal`, `normal`, `orthogonal`, `zeros`, `ones`.
- `init_embedding_weights`: `normal`, `xavier_uniform`, `xavier_normal`,
  `zeros` (no He/orthogonal/ones options).
- Unknown method string → `ValueError` for both (this one *does* validate).
- `std` (only used by `"normal"`) defaults to `constants.NORMAL_INIT_STD = 0.1`.
- `orthogonal` uses SVD of a Gaussian matrix (reshaped for conv weights).

Pick Xavier for tanh/sigmoid-family activations, He for ReLU-family — the
per-layer `init_method=` default is already sensible for that layer's common
use (`add_dense` defaults to `xavier_uniform`, `add_conv2d` to `he_normal`),
but override it if you're using an unusual activation.

## Losses

```python
ComputeLoss(output, target, function="mse", reduction="mean", **kwargs)
```
and the matching gradient inside `Backward(targets, loss_function=..., **kwargs)`.

| `function` | kwargs | Notes |
|---|---|---|
| `mse` | — | |
| `mae` | — | |
| `huber` | `delta=1.0` | |
| `smooth_l1` | `beta=1.0` | |
| `binary_cross_entropy` | `eps=1e-12` | |
| `cross_entropy` / `categorical_cross_entropy` | `eps` | Aliases for the same implementation |
| `focal` | `alpha=0.25`, `gamma=2.0`, `eps` | |
| `hinge` | — | |
| `bce_logits` | — | Numerically-stable log-sum-exp form; use with a linear output layer |
| `wasserstein` | — | For WGAN-style critics |
| `cosine_similarity` | `eps_div=1e-8` | |
| `triplet` | `margin=1.0`, `negative` (**required**) | Raises `ValueError` if `negative` isn't passed |
| `ntxent` | `temperature=0.5`, `eps_div` | Contrastive loss (SimCLR/CLIP-style) |
| `kl_divergence` | `mu`, `logvar` | **Gotcha:** if you don't pass `mu`/`logvar` explicitly, they silently default to `output`/`target` positionally — this loss is meant for VAE mu/logvar tensors, not a generic output/target pair; always pass both kwargs explicitly. Not usable via `Backward(loss_function="kl_divergence")` for the same reason (see `generative/vae.py` for the manual gradient it actually needs). |

Unknown `function` → `ValueError`. `reduction="mean"`/`"sum"` return a plain
Python `float`; anything else returns the raw elementwise array.

**Reduction convention**, load-bearing if you write your own finite-
difference reference: under `reduction="mean"`, elementwise losses (`mse`,
`mae`, `huber`, `bce`, `focal`, `hinge`, `bce_logits`, `wasserstein`, ...)
average over **every element** (`batch_size * n_features`). Losses that
already reduce over the feature axis in their own formula (`cross_entropy`,
`cosine_similarity`, `triplet`, `ntxent`) divide by `batch_size` only.

Where a loss+activation pair has a canonical simplified gradient (softmax +
cross-entropy, sigmoid + BCE, linear + `bce_logits`/`wasserstein`), Enilnets
uses that closed form directly instead of chaining a separate activation
derivative — slightly faster and more numerically stable than the generic
chain rule path.

## Optimizers

Set via `NeuralNet(optimizer=..., learning_rate=..., l2_lambda=..., momentum=...)`.

| Optimizer | Extra kwargs | Pros | Cons |
|---|---|---|---|
| `"sgd"` | `momentum` | Simple, predictable, cheapest per step (no per-parameter state beyond momentum). Often generalizes best with enough tuning. | Needs careful learning-rate tuning; slow convergence on ill-conditioned losses without momentum tuned well. |
| `"rmsprop"` | `rmsprop_decay`, `rmsprop_epsilon` | Per-parameter adaptive learning rate; handles non-stationary objectives (RL, GANs) reasonably well. | No bias correction (unlike Adam) — early steps can be less stable; one fewer hyperparameter than Adam to tune, which is a pro *and* a con depending on how much control you want. |
| `"adagrad"` | `adagrad_epsilon` | Good for sparse gradients (e.g. embeddings) — features updated rarely get comparatively larger steps. | Accumulated squared gradient only grows, so the effective learning rate monotonically shrinks — can stall on long training runs. |
| `"adam"` | `adam_beta1`, `adam_beta2`, `adam_epsilon` | The default for a reason: adaptive per-parameter rates + momentum + bias correction; a reasonable default across almost every model type in this library. | L2 weight decay is *coupled* into the gradient before the moment update (interacts with the adaptive scaling in ways that can undertune effective regularization — this is exactly what AdamW fixes). |
| `"adamw"` | same as Adam | Decoupled weight decay: applied directly to weights after the Adam step (`w -= lr * l2_lambda * w`), independent of gradient scale — the modern default recommendation over plain Adam+L2 when you're using weight decay. | One more subtlety to be aware of: decay still happens every step regardless of gradient (including when the gradient is exactly zero) — usually desired, but different from Adam+L2's behavior. |

**Known gotcha:** `optimizer_type` is checked against exactly `"sgd"`,
`"rmsprop"`, `"adagrad"` in `apply_gradients`; **any other string — including
typos — silently falls into the Adam/AdamW branch** rather than raising an
error. Double-check your spelling of `optimizer="..."`.

Weight-decay eligibility is hardcoded per layer type: `dense`/`sparse`/
`conv2d`/`embedding` decay `weights` only (never bias); `multihead_attention`
decays `Wq`/`Wk`/`Wv`/`Wo` (never the biases); `rnn`/`lstm`/`gru` decay
`Wx`/`Wh` (never `b`/`bx`/`bh`); batchnorm/layernorm `gamma`/`beta` are never
decayed.

Gradient clipping is automatic whenever `grad_clip_norm > 0`:
```python
model = NeuralNet(optimizer="adam", grad_clip_norm=1.0)
model.TrainBatch(x, y)  # Backward() -> clip_gradients(1.0) -> update(), automatically
```

Lower-level primitives, if you want to inspect/modify gradients before
they're applied, or build your own accumulation/multi-optimizer scheme:
```python
compute_gradients(self)           # -> list aligned with self.layers, None for param-free layers
apply_gradients(self, grads)      # applies the configured optimizer formula, mutates weights + opt_state
update(self)                      # = apply_gradients(self, compute_gradients(self))
```

## Training

### `TrainBatch` / `Train`

```python
TrainBatch(xs, ys, loss_function=None, accumulation_steps=1, **loss_kwargs)
Train(X_train, Y_train, epochs=10, batch_size=32, X_val=None, Y_val=None,
      loss_function=None, verbose=True, scheduler=None, early_stopping=None,
      accumulation_steps=1, **loss_kwargs)
```
`TrainBatch` runs one batch end-to-end: forward → loss → backward → optional
clip → optimizer step, returning `(loss, out)`. If `loss_function=None`, it
auto-picks `"cross_entropy"` when the last layer's activation is
`"softmax"`, else `"mse"`.

`Train` wraps `TrainBatch` in a full epoch/minibatch loop, returning a
`history` dict with `"loss"`, `"accuracy"`, `"lr"` (always populated) and
`"val_loss"`/`"val_accuracy"` (empty lists if no `X_val`/`Y_val` given, not
omitted from the dict).

```python
history = model.Train(X_train, Y_train, epochs=50, batch_size=64,
                      X_val=X_val, Y_val=Y_val, loss_function="mse",
                      scheduler=scheduler, early_stopping=early_stopping)
```

If you need full manual control, drop to the primitives directly:
```python
out = model.Forward(x, training=True)
model.Backward(y, loss_function="mse")
model.update()
```

### Gradient clipping

Covered above under [Optimizers](#optimizers) — set `grad_clip_norm > 0` at
construction; `TrainBatch` applies it automatically after every `Backward()`.

### Gradient accumulation

Simulate a larger batch size without the memory cost:
```python
model.Train(X_train, Y_train, epochs=10, batch_size=16, accumulation_steps=4)
# effective batch size 64, only 16 samples in memory at a time
```
Or manually:
```python
model.Forward(x1, training=True); model.Backward(y1); model.accumulate_gradients()
model.Forward(x2, training=True); model.Backward(y2); model.accumulate_gradients()
model.apply_accumulated_gradients()  # averages and applies both steps at once
```
Verified to match a single full-size batch exactly for SGD; a reasonable
(not necessarily bit-identical) approximation for adaptive optimizers, since
Adam/RMSprop/Adagrad's per-step moment updates are inherently order- and
step-count-sensitive.

### Mixed precision

```python
model = NeuralNet(optimizer="adam", use_mixed_precision=True)
```
Runs the dense/conv2d matmuls (the hottest path per the benchmark suite) in
float32 while keeping master weights at float64, for a real BLAS speedup —
a lightweight CPU approximation of AMP (no tensor-core path or loss scaling,
since neither applies without GPU hardware). Expect outputs close to, not
bit-identical to, the float64 path.

### Learning rate schedules

```python
LRScheduler(initial_lr, mode="step", **kwargs)
scheduler.step(epoch)  # -> new learning rate for this epoch
```
| `mode` | kwargs | Behavior |
|---|---|---|
| `"step"` | `drop=0.5`, `epochs_drop=10` | `lr * drop^(epoch // epochs_drop)` |
| `"exponential"` | `decay=0.95` | `lr * decay^epoch` |
| `"cosine"` | `max_epochs=100` | Cosine decay to 0 over `max_epochs` |
| `"warmup_cosine"` | `max_epochs=100`, `warmup_epochs=5` | Linear warmup then cosine decay |
| `"plateau"` | — | **Not implemented** — this mode is a stub that just returns `initial_lr` unchanged every epoch (the docstring notes it "requires history tracking externally"; there's no actual plateau-detection logic). Any other/unrecognized `mode` string also silently returns `initial_lr` unchanged. |

### Early stopping

```python
EarlyStopping(patience=5, min_delta=0.0, mode="min")
early_stopping.step(metric)  # -> bool, True once training should stop
```
`mode="min"` requires `metric < best - min_delta` to count as improvement;
`"max"` requires `metric > best + min_delta`. Pass the instance to
`model.Train(..., early_stopping=early_stopping)`.

### Accuracy / precision / recall / F1

```python
model.compute_accuracy(predictions, targets)
```
Dispatches on `predictions.shape[-1]`: `>1` → multi-class `argmax` comparison;
`==1` → binary `>0.5` threshold. Purely shape-based, not an explicit flag.

`compute_precision_recall_f1(predictions, targets)` (binary-only, same
shape-based dispatch) exists in `Enilnets/train.py` but is **not** bound onto
`NeuralNet` and **not** in `__all__` — import it explicitly
(`from Enilnets.train import compute_precision_recall_f1`) or use
[`classification_report`](#evaluation-utilities) for the general multi-class
case.

## Text generation (`TextGenerator`)

```python
TextGenerator(tokenizer, embed_dim=64, num_heads=4, num_layers=2,
             mlp_ratio=4.0, dropout=0.0, activation="gelu",
             max_seq_len=128, learning_rate=3e-4, optimizer="adam", l2_lambda=0.0)
```
Builds a GPT-style causal transformer: `embedding → positional_encoding
(fixed sinusoidal) → num_layers × transformer_block(causal=True) →
layernorm → dense(softmax)`. Requires an already-`.fit()`-ted `Tokenizer`
(raises `ValueError` otherwise).

```python
from Enilnets import TextGenerator, Tokenizer

tokenizer = Tokenizer(vocab_size=2000, level="char").fit([corpus])
gen = TextGenerator(tokenizer, embed_dim=128, num_heads=4, num_layers=4, max_seq_len=128)
gen.Train([corpus], epochs=20, batch_size=32, seq_len=64, verbose=True)

print(gen.generate(prompt="once upon a", max_new_tokens=200, greedy=True))
print(gen.generate(prompt="once upon a", max_new_tokens=200, temperature=0.8, top_p=0.9))
print(gen.generate(prompt="once upon a", max_new_tokens=200, temperature=0.8, top_k=40))
print(gen.generate_beam(prompt="once upon a", beam_width=5, max_new_tokens=100))
print("perplexity:", gen.perplexity(held_out_text))
```

- `generate(..., use_cache=True)` (default) decodes with a KV-cache: only
  the new token's query is computed each step, past keys/values cached and
  reused — O(n) over the sequence instead of O(n²). Verified to produce
  identical probabilities to `use_cache=False` (full recompute every step).
  **Known limitation:** the KV-cache fast path only understands the exact
  architecture `TextGenerator` builds (embedding → positional encoding →
  transformer blocks → layernorm → dense) — if you've hand-modified
  `gen.network` to something else, it raises `ValueError`; pass
  `use_cache=False` for custom architectures.
- `generate_beam(...)` does **not** use the KV-cache (always full recompute
  per beam-step) — slower than `generate()` per token, as expected for
  exact beam search.
- `perplexity(text)` requires at least 2 tokens after tokenization, else
  raises `ValueError`.

`Tokenizer(vocab_size=256, level="char", oov_token="<OOV>", pad_token="<PAD>",
start_token="<START>", end_token="<END>")` — `level="char"` ignores
`vocab_size` entirely (vocabulary = every character actually seen, unbounded);
`level="word"` truncates to the `vocab_size - 4` most common words (the 4
special tokens always occupy the first slots). `.fit(texts)` returns `self`
(chainable). `.encode(text, max_length=None, add_special_tokens=True)` →
`ndarray` of `int32`; unknown tokens map to `oov_token`.

## Generative models

All of these live under `Enilnets.generative` (and are re-exported from the
top-level `Enilnets` package). Each follows: build with hyperparameters, call
`.Train(X_train, epochs=..., batch_size=...)`, then `.generate(...)`/`.sample(...)`.

### VAE

```python
VAE(input_dim, latent_dim, encoder_hidden=[512, 256], decoder_hidden=[256, 512],
   activation="swish", learning_rate=0.001, optimizer="adam", l2_lambda=0.0)
```
```python
vae = VAE(input_dim=784, latent_dim=32, encoder_hidden=[256, 128])
vae.Train(X_train, epochs=30, batch_size=64, kl_weight=1.0)
samples = vae.generate(n_samples=16)
recon = vae.reconstruct(X_val)
midpoints = vae.interpolate(x1, x2, n_steps=10)
```
Methods: `encode(x) -> (mu, logvar)`, `decode(z)`, `forward(x) -> (recon, mu,
logvar, z)`, `loss(x, ..., kl_weight=1.0)`, `train_step(x, kl_weight=1.0)`,
`Train(...)`, `generate(n_samples=1)`, `reconstruct(x)`,
`interpolate(x1, x2, n_steps=10)`.

- **Pros:** smooth, well-structured latent space (good for interpolation,
  downstream feature extraction); stable to train, no adversarial dynamics.
- **Cons:** decoder ends in `sigmoid`, implicitly assuming inputs are scaled
  to `[0, 1]` (BCE reconstruction loss) — rescale your data accordingly;
  samples tend to be blurrier than GAN/diffusion output (a well-known VAE
  characteristic, not specific to this implementation).

### GAN

```python
GAN(latent_dim, data_dim, generator_hidden=[256, 512], discriminator_hidden=[512, 256],
   g_activation="swish", d_activation="leakyrelu", loss_type="bce",
   learning_rate=0.0002, optimizer="adam", l2_lambda=0.0,
   label_smoothing=0.9, g_lr_factor=1.0, d_lr_factor=1.0, wgan_clip_value=0.01)
```
```python
gan = GAN(latent_dim=64, data_dim=784, loss_type="wasserstein", wgan_clip_value=0.01)
gan.Train(X_train, epochs=100, batch_size=64, d_steps=5, g_steps=1)
samples = gan.sample(16)
print(gan.mode_collapse_score())  # 0=collapsed, 1=diverse
```
`loss_type` ∈ `"bce"`, `"bce_logits"`, `"wasserstein"`. Generator output is
`tanh` (assumes data scaled to `[-1, 1]`); discriminator output is `sigmoid`
for `"bce"`, else `linear`. `label_smoothing` softens real-label targets
(`bce`/`bce_logits` only). For Wasserstein, `_clip_discriminator_weights`
clips weights directly after each D step (standard WGAN weight clipping).

- **Pros:** sharpest samples among the generative models here; Wasserstein
  variant is more stable to train than vanilla BCE GAN in the classic
  GAN-training-instability sense; `mode_collapse_score()` gives a quick
  diversity diagnostic without external tooling.
- **Cons:** still a GAN — training can require balancing `d_steps`/`g_steps`
  and learning rates (`g_lr_factor`/`d_lr_factor` exist specifically to help
  with this); `mode_collapse_score`'s `n_clusters` parameter is accepted but
  currently unused in the body (dead parameter — the diversity heuristic
  doesn't actually cluster).

### DiffusionModel

```python
DiffusionModel(data_shape, time_steps=1000, beta_schedule="linear",
              beta_start=1e-4, beta_end=0.02, denoiser_type="mlp",
              denoiser_hidden=[512, 512, 512], learning_rate=0.001,
              optimizer="adam", l2_lambda=0.0, use_ema=True, ema_decay=0.999,
              cosine_schedule_s=0.008, beta_clip=(0, 0.999),
              time_emb_dim=128, sample_clip_range=(-1.0, 1.0))
```
```python
diffusion = DiffusionModel(data_shape=(784,), time_steps=1000, beta_schedule="cosine")
diffusion.Train(X_train, epochs=50, batch_size=64)
samples = diffusion.sample(n_samples=16)
partial = diffusion.denoise(x_noisy, t_start=500, t_end=0)
```
`beta_schedule` ∈ `"linear"`, `"cosine"`; `denoiser_type` ∈ `"mlp"`, `"conv"`
(the latter requires `data_shape=(C, H, W)` and broadcasts the time
embedding as extra input channels). `use_ema=True` maintains a separate EMA
copy of denoiser weights, automatically swapped in during `sample()`/
`denoise()` (not during `train_step`).

- **Pros:** DDPM-style training (the well-studied, stable denoising
  objective); EMA weights typically produce noticeably better samples than
  raw training weights, and this is wired in by default; both MLP and
  (basic) conv denoisers available.
- **Cons:** `sample()`/`denoise()` are O(`time_steps`) sequential forward
  passes — the slowest generation path in the library by design (this is
  inherent to standard DDPM ancestral sampling, not an implementation
  shortcut; there's no DDIM/fast-sampling variant here yet); the backward
  pass is manually unrolled layer-by-layer rather than going through
  `NeuralNet.Backward`'s standard loss-based entry point (an implementation
  detail, but means this model doesn't benefit from future generic
  `Backward()` improvements automatically).

### RealNVP (normalizing flow)

```python
RealNVP(data_dim, n_coupling=4, hidden_dim=256, activation="swish",
       learning_rate=0.001, optimizer="adam", l2_lambda=0.0)
```
```python
flow = RealNVP(data_dim=2, n_coupling=6, hidden_dim=128)
flow.Train(X_train, epochs=50, batch_size=128)
samples = flow.sample(n_samples=500)
z, log_det = flow.forward(x)
x_reconstructed = flow.inverse(z)
```
Each of `n_coupling` layers alternates which half of the input dims gets
transformed. `log_prob(x)`/`loss(x)` (negative log-likelihood) are exact,
not a variational bound — the defining advantage of normalizing flows.

- **Pros:** exact likelihood computation (unlike VAE's ELBO or GAN's implicit
  density) and exact invertibility (`forward`/`inverse` are true inverses of
  each other) — useful when you need actual density estimates, not just
  samples.
- **Cons:** `data_dim` odd means the two coupling halves are unequal sizes
  (`data_dim//2` vs. the remainder) — works, but architecturally asymmetric;
  scales less gracefully to very high-dimensional data (e.g. large images)
  than diffusion/GAN in general (a known property of coupling-layer flows,
  not specific to this implementation).

### EnergyBasedModel

```python
EnergyBasedModel(data_dim, hidden_dims=[512, 512], activation="swish",
                 learning_rate=0.001, optimizer="adam", l2_lambda=0.0,
                 persistent_cd=True, persistent_buffer_size=1000, init_noise_scale=0.5)
```
```python
ebm = EnergyBasedModel(data_dim=2, persistent_cd=True)
ebm.Train(X_train, epochs=50, batch_size=64, n_cd_steps=20)
samples = ebm.sample(n_samples=500, n_steps=200)
score = ebm.score(x)  # gradient of energy w.r.t. x
```
Trained via (persistent) contrastive divergence with Langevin dynamics for
negative sampling. `persistent_cd=True` maintains a ring-buffer
(`persistent_buffer_size`) of negative samples carried across calls — i.e.
`sample()`/`train_step()` have hidden state between calls, not pure
functions of their arguments alone.

- **Pros:** conceptually the most flexible generative model here (just
  learns an unnormalized energy function — no need for a tractable
  likelihood or an adversarial pair); `score(x)` directly gives you a
  gradient field useful for other downstream uses (e.g. as a critic/guidance
  signal).
- **Cons:** sampling requires an iterative Langevin chain (`n_steps`,
  slower than a single forward pass); training via contrastive divergence is
  notoriously sensitive to hyperparameters (`n_cd_steps`, `step_size`,
  `noise_scale`) — expect to tune more than with VAE/GAN.

### AutoregressiveModel

```python
AutoregressiveModel(data_dim, hidden_dims=[512, 512], data_shape=None,
                    activation="swish", learning_rate=0.001, optimizer="adam",
                    l2_lambda=0.0, num_classes=256, discrete=False)
```
```python
ar = AutoregressiveModel(data_dim=784, discrete=True, num_classes=256)
ar.Train(X_train, epochs=30, batch_size=64)
samples = ar.generate(n_samples=16)
completed = ar.complete(partial_x, n_dims=400)
print("log-likelihood:", ar.log_prob(X_val).mean())
```
Autoregression is enforced via a lower-triangular masked input (dimension
*i*'s prediction only sees dimensions `< i`), not causal-masked attention.
`discrete=True` treats each dimension as a `num_classes`-way categorical
(pixel-value-style, e.g. 0-255); `discrete=False` uses per-dimension
Gaussian/MSE.

- **Pros:** exact likelihood (like normalizing flows); `complete()` gives a
  natural inpainting/completion API (fill in missing dimensions given known
  ones) that the other generative models don't offer directly.
- **Cons:** `generate()`/`complete()` sample dimension-by-dimension in a
  Python loop, calling `forward()` on the whole growing sample each time —
  O(data_dim²) forward passes, no caching (unlike `TextGenerator`'s
  KV-cache) — generation gets slow for high-dimensional data (full images,
  for instance). Discrete mode assumes continuous inputs are pre-scaled to
  `[0, 1]` before being bucketed into classes.

### UNetDenoiser

```python
UNetDenoiser(in_ch, base_ch=64, time_emb_dim=128, ch_mult=(1, 2, 4),
            learning_rate=0.001, optimizer="adam", l2_lambda=0.0,
            init_method="he_normal", pool_factor=2)
time_embedding(t, dim, max_period=10000)  # standalone sinusoidal time embedding function
```
```python
unet = UNetDenoiser(in_ch=1, base_ch=64, ch_mult=(1, 2, 4))
out = unet.forward(x, t)
```

- **Pros:** encoder-decoder with skip connections at matching resolutions
  (the actual "U" shape), usable as a standalone forward-pass component in a
  custom pipeline (`get_params()` exposes every internal `NeuralNet`
  sub-network for external inspection/training).
- **Cons — read before relying on this for training:** **`backward()` is a
  stub that unconditionally raises `NotImplementedError`.** This class is
  forward-only; it is not independently trainable via the standard
  `Backward()`/`update()` flow. (Its error message points at
  `DiffusionModel(mlp_denoiser=True)`, but that parameter doesn't actually
  exist on `DiffusionModel`'s real signature — a stale message; use
  `DiffusionModel(denoiser_type="conv", ...)` for a genuinely trainable
  conv-based diffusion denoiser instead.) Also, despite the "U-Net" name,
  every convolution uses `k=1` (1×1 convs only) — spatial receptive field
  only grows via the manual downsample/upsample pooling, not from
  conv-kernel spatial extent, unlike a conventional U-Net's 3×3/5×5 convs.

### Low-level sampling & loss building blocks

Used internally by the models above, also directly importable for building
your own generative training loops:

```python
# Enilnets.generative.sampling
reparameterize(mu, logvar)
langevin_dynamics(energy_fn, x_init, n_steps=20, step_size=0.1, noise_scale=0.005)
gaussian_sample(mean, std, shape=None)
uniform_sample(low, high, shape)
gumbel_softmax_sample(logits, temperature=1.0, hard=False)
random_mask(shape, ratio)
top_p_sampling(logits, p=0.9, temperature=1.0)   # batched (batch, vocab) logits -> one-hot array
top_k_sampling(logits, k=10, temperature=1.0)    # single 1D logits vector -> plain int index
gae(rewards, values, gamma=0.99, lambda_=0.95)   # -> (advantages, returns); import via Enilnets.generative.sampling, not re-exported at top level

# Enilnets.generative.generative_loss
kl_divergence_gaussian(mu, logvar, reduction="mean", kl_weight=1.0)
adversarial_loss_discriminator(real_logits, fake_logits, loss_type="bce")  # "bce"|"bce_logits"|"wasserstein"
adversarial_loss_generator(fake_logits, loss_type="bce")
diffusion_loss(predicted_noise, true_noise, reduction="mean")
nll_loss(log_px, log_det_jacobian, reduction="mean")
energy_loss(data_energy, sample_energy, margin=1.0)
perceptual_loss(x, y, feature_extractor=None)  # falls back to plain MSE if no extractor given
vgg_loss(x, y)  # NOTE: literally `mean((x - y) ** 2)` -- an explicit placeholder, not real VGG features
```

**Known gotcha:** `top_p_sampling` and `top_k_sampling` have different call
conventions despite similar names — `top_p_sampling` takes a batch of
logits and returns a one-hot array; `top_k_sampling` takes a single 1D
logits vector and returns a plain integer index. Check which one you need.

## Reinforcement learning

Policy-gradient methods hang directly off any `NeuralNet` used as a policy
(and, for `ActorCritic`, a second `NeuralNet` as a value function). All of
them build the output-layer gradient by hand (`Backward(None,
output_delta=...)`) rather than going through `ComputeLoss`, since policy
gradients aren't a standard supervised loss.

```python
from Enilnets import compute_returns

policy = NeuralNet(learning_rate=0.001, optimizer="adam")
policy.add_dense(state_dim, 64, activation="relu")
policy.add_dense(64, n_actions, activation="softmax")

returns = compute_returns(rewards, gamma=0.99)
policy.Reinforce(states, actions, returns, action_type="discrete")
```

| Method | Signature | Notes |
|---|---|---|
| `Reinforce` | `(states, actions, returns, action_type="discrete", std=1.0, normalize_returns=True)` | Vanilla REINFORCE. `action_type` ∈ `"discrete"` (softmax policy) / `"continuous"` (Gaussian policy, fixed `std`); anything else raises `ValueError`. |
| `PPO` | `(states, actions, old_log_probs, advantages, action_type="discrete", epsilon=0.2, std=1.0, value_targets=None, value_coeff=0.5, entropy_coeff=0.01)` | Clipped-objective PPO. **`value_targets`/`value_coeff` are accepted but never referenced in the function body** — there's no value-loss term implemented despite the docstring; compute your critic loss separately (e.g. via `ActorCritic` or your own `ComputeLoss(..., "mse")` call on the value network) and apply it as its own update. |
| `ActorCritic` | `(states, actions, returns, values, action_type="discrete", std=1.0)` | Advantage actor-critic; `values` comes from a separate critic `NeuralNet` you maintain yourself. |
| `Evolve` | `(inputs, score_fn, noise=0.05, tries=10, sigma=1.0)` | Gradient-free evolution strategy: perturbs every layer's weights/bias with `N(0, (sigma*noise)^2)`, re-masks sparse layers, keeps whichever candidate (including the unperturbed baseline) scores highest under `score_fn(self.Forward(inputs))`. Mutates `self.layers` in place; returns the best score. No gradient step at all — useful when you don't have (or don't trust) a gradient signal. |
| `compute_returns` | `(rewards, gamma=0.99)` | Standalone function (also bound onto `NeuralNet`, though calling it as `policy.compute_returns(...)` will double-pass `self` positionally — **call it as the standalone `Enilnets.compute_returns(rewards, gamma=...)` import, not as a bound method**). Discounted-return computation; duplicated verbatim in `generative/sampling.py`. |

`gae(rewards, values, gamma=0.99, lambda_=0.95)` (Generalized Advantage
Estimation) lives in `Enilnets.generative.sampling`, not re-exported at the
top level — import it from there if you need GAE-based advantages for PPO.

## NEAT (neuroevolution)

`NEATPopulation` implements NeuroEvolution of Augmenting Topologies: a
population of small networks ("genomes") evolves via mutation (perturb
weights, add a connection, add a node) and crossover, guided only by a
fitness function you supply — no gradients required. It doesn't build on
`NeuralNet` at all; a `Genome` is its own small feedforward network with its
own `.forward(x)`, since its topology (which nodes exist, how they're wired)
changes over evolution rather than being fixed up front.

```python
import numpy as np
from Enilnets import NEATPopulation

xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
xor_targets = np.array([0, 1, 1, 0], dtype=np.float64)

def fitness(genome):
    preds = np.array([genome.forward(x)[0] for x in xor_inputs])
    return float(4.0 - np.sum((preds - xor_targets) ** 2))  # higher is better

pop = NEATPopulation(n_inputs=2, n_outputs=1, population_size=150, seed=0)
history = pop.evolve(fitness, generations=100, verbose=True)

best = pop.best_genome
print(best.forward(xor_inputs))          # trending toward [0, 1, 1, 0]
print(len(best.nodes), len(best.connections))
```

```python
NEATPopulation(n_inputs, n_outputs, population_size=150, activation="sigmoid",
              output_activation=None, compatibility_threshold=None, c1=None, c2=None, c3=None,
              weight_mutate_rate=None, weight_perturb_rate=None, weight_perturb_power=None,
              add_connection_rate=None, add_node_rate=None, crossover_rate=None,
              survival_threshold=None, stagnation_limit=None, elitism=1, seed=None)
```
Every `None`-defaulted knob falls back to a `constants.NEAT_*` value:
`compatibility_threshold=3.0`, `c1=1.0`, `c2=1.0`, `c3=0.4` (speciation
distance: excess-gene, disjoint-gene, weight-difference coefficients),
`weight_mutate_rate=0.8`, `weight_perturb_rate=0.9`,
`weight_perturb_power=0.5`, `add_connection_rate=0.05`, `add_node_rate=0.03`,
`crossover_rate=0.75`, `survival_threshold=0.2` (fraction of each species
allowed to reproduce), `stagnation_limit=15` (generations without
improvement before a species stops being prioritized), `elitism=1` (top
genomes carried over unchanged each generation).

XOR is the classic NEAT sanity check: it isn't linearly separable, so a
population that starts fully-connected with no hidden units must actually
*discover* a hidden node (via the add-node mutation) to solve it. Like any
evolutionary search, convergence speed varies by seed/hyperparameters; give
it more generations or a larger population if a given run hasn't fully
separated all four cases yet.

For direct genome manipulation (custom evolutionary loops, single-genome
save/load, etc.), the building blocks are importable from `Enilnets.neat`:

```python
from Enilnets.neat import Genome, InnovationTracker, crossover  # crossover also exported as Enilnets.neat_crossover
```
- `Genome.minimal(n_inputs, n_outputs, innovation_tracker, activation="sigmoid", output_activation=None, weight_range=1.0)` — a fully-connected genome with no hidden nodes.
- `genome.forward(x)` — accepts `(n_inputs,)` or `(batch, n_inputs)`, **returns matching rank** (1D in → 1D out).
- `genome.mutate_weights(...)`, `genome.mutate_add_connection(innovation_tracker, ...)`, `genome.mutate_add_node(innovation_tracker)` — the latter two return `bool` (success), not exceptions, if no valid mutation could be found.
- `genome.distance(other, c1, c2, c3)` — compatibility distance for manual speciation.
- `genome.break_cycles()` — a safety net (see below) that disables the minimum number of connections needed to restore a valid feedforward ordering.
- `crossover(fitter, other)` — combines two genomes, assuming `fitter.fitness >= other.fitness`.

**Correctness note (worth knowing if you build custom NEAT pipelines):**
each individual mutation (`mutate_add_connection`) checks for cycles *within
that one genome*, so it's always locally safe. But `crossover`'s matching-
gene coin flip can independently pick each endpoint of a same-innovation
edge from a *different* parent — and since the two parents are each only
guaranteed acyclic *individually*, not in combination, this can silently
reintroduce a cycle the disabling was specifically there to prevent (this
happened for real during a ~200-generation XOR evolution run while building
this feature). `crossover()` calls `child.break_cycles()` internally as a
safety net before returning, so this is handled automatically — but if you
ever hand-construct a genome by merging connection dicts from two sources
yourself (bypassing `crossover()`), call `genome.break_cycles()` before
`forward()`-ing it.

## Visualization

`plot_network`/`model.plot(...)` renders the classic node/connection diagram
of a `NeuralNet` as a self-contained SVG string — pure stdlib, no matplotlib
dependency. Dense/sparse/RNN/LSTM/GRU layers become columns of circular
nodes with real weighted edges (blue = positive, red = negative, opacity =
relative magnitude); other layer types (conv2d, attention, pooling,
embedding, ...) render as labeled blocks in between, since they don't reduce
to a single weight matrix connecting two node columns. Batchnorm/layernorm/
dropout are transparent to the diagram — edges are drawn straight through
them using the real dense weight matrix on the far side, since they're
dimension-preserving.

```python
model = NeuralNet(learning_rate=0.001, optimizer="adam")
model.add_dense(4, 8, activation="relu")
model.add_batchnorm(8)
model.add_dense(8, 3, activation="softmax")

svg = model.plot()  # structure only, no values

# With a sample input, node fill color shows that layer's actual activation
# value from a live forward pass (heat-mapped blue=low to red=high) --
# call this from inside your own training loop for a live snapshot:
for epoch in range(epochs):
    model.TrainBatch(X_batch, Y_batch)
    if epoch % 10 == 0:
        model.plot(sample_input=X_batch[:1], filename=f"epoch_{epoch}.svg")
```

`plot_genome(genome, sample_input=..., show_disabled=True)` does the same
for a NEAT `Genome`: node x-position is graph depth (topological distance
from the inputs), node color is type (input/bias/hidden/output) or
activation value if `sample_input` is given, disabled connections drawn
dashed.

**Using it in a real project:**
```python
svg = model.plot(sample_input=x)   # always the raw SVG string, embed it directly

from IPython.display import SVG, display
display(SVG(svg))                  # Jupyter

from Enilnets import to_html
from flask import Response
@app.route("/network")
def network_view():
    return Response(to_html(svg, title="Live Network"), mimetype="text/html")  # web app

model.plot(sample_input=x, filename="network.svg")    # raw SVG file
model.plot(sample_input=x, filename="network.html")   # standalone HTML doc, open in any browser
```

Large layers auto-cap (`max_nodes_per_layer=20` default for `plot_network`,
`30` for `plot_genome`), showing first/last half with a `⋮` marker in between
— a 784-neuron input layer won't render 784 circles. `plot_network` raises
`ValueError` on a model with no layers.

## Evaluation utilities

```python
from Enilnets import confusion_matrix, classification_report

preds = model.Forward(X_val, training=False)
y_pred = preds.argmax(axis=1)
y_true = Y_val.argmax(axis=1)

cm = confusion_matrix(y_true, y_pred, num_classes=10)
report = classification_report(y_true, y_pred, num_classes=10)
print(report["weighted_avg"], report["accuracy"])
```
`confusion_matrix(y_true, y_pred, num_classes=None)` expects 1D integer
class-label arrays (argmax first); `num_classes=None` infers from the data.
`classification_report(...)` returns `{0: {...}, 1: {...}, ...,
"macro_avg": {...}, "weighted_avg": {...}, "accuracy": float}` — note the
per-class keys are plain integers mixed with string keys in the same dict.

`Enilnets.eval_utils` (generative-model-focused, import the module
directly — not re-exported at top level): `inception_score(samples,
classifier=None, splits=10)` — **with `classifier=None`, this falls back to
a crude k-means-based diversity proxy, not a real Inception-v3-based score**
despite the name; pass your own `NeuralNet`-like classifier for anything
resembling the standard metric. Also `frechet_distance`/`compute_fid`,
`reconstruction_error(original, reconstructed, metric="mse"|"mae"|"psnr")`
(psnr assumes `[0,1]`-scaled data), `sample_diversity`,
`nearest_neighbor_accuracy`.

## Data utilities

### General (`utils.py`)

```python
set_seed(seed)
train_test_split(X, Y, test_size=0.2, shuffle=True, seed=None)
k_fold_split(X, Y, k=5, shuffle=True, seed=None)         # generator
iterate_minibatches(X, Y, batch_size, shuffle=True)      # generator
count_parameters(model)                                   # -> (total_params, per_layer_dict)
one_hot(indices, vocab_size)                               # = text_utils.one_hot_encode
EarlyStopping(patience=5, min_delta=0.0, mode="min")
```
`train_test_split`'s `seed=None` uses the *global* `np.random` state (not
reproducible across calls unless you called `set_seed` first); pass `seed`
for an isolated, reproducible split. `k_fold_split` distributes remainder
samples (`n % k`) one-per-fold to the first folds.

### Text (`text_utils.py`)

Covered above under [Text generation](#text-generation-textgenerator)
(`Tokenizer`). Also:
```python
load_text_file(path, encoding='utf-8')
load_texts_from_directory(directory, encoding='utf-8', max_files=None)  # silently skips undecodable files
create_sliding_windows(data, window_size, stride=1)   # -> (X, y) next-token pairs
pad_sequences(sequences, max_length=None, pad_value=0)
```

### Images (`image_utils.py`)

```python
load_ppm(path) / save_ppm(arr, path)     # binary P6 only, maxval must be 255
load_pgm(path) / save_pgm(arr, path)     # binary P5 only
load_raw_binary(path, shape, dtype=np.float64) / save_raw_binary(arr, path)
rgb_to_grayscale(rgb) / grayscale_to_rgb(gray)
resize_nearest_neighbor(img, new_height, new_width)
resize_bilinear(img, new_height, new_width)
image_augmentation(images, flip_h=True, flip_v=False, rotate=0, brightness=0.0, contrast=0.0, noise_std=0.0)
normalize_images(images, mean=None, std=None)   # -> (normalized, mean, std) -- 3-tuple, easy to forget when unpacking
denormalize_images(images, mean, std)
images_to_patches(images, patch_size, stride=None)
pad_image(img, pad_h, pad_w, mode='constant', constant_value=0)
```
`load_ppm`/`load_pgm` raise `ValueError` for anything other than binary
P6/P5 with `maxval=255` — no ASCII PPM/PGM support. `image_augmentation`'s
`rotate` is a 90°-multiple ceiling (`[a for a in [0,90,180,270] if a <=
rotate]`), not an arbitrary-angle rotation — `rotate=45` only permits 0°.

### Audio (`audio_utils.py`)

```python
load_wav(path)                          # -> (audio, sr); PCM16/24/32 or IEEE float32
save_wav(audio, path, sr, bits_per_sample=16)
stft(audio, n_fft=2048, hop_length=512, window='hann') / istft(...)
spectrogram_to_mel(spectrogram, sr, n_mels=128, fmin=0, fmax=None)
mel_to_spectrogram(mel_spec, sr, n_freq, fmin=0, fmax=None)   # pseudo-inverse, not exact
audio_to_spectrogram(audio, sr, n_fft=2048, hop_length=512, n_mels=128)   # -> log-mel
spectrogram_to_audio(mel_spec, sr, n_fft=2048, hop_length=512, n_iter=32) # Griffin-Lim
audio_to_frames(audio, frame_length, hop_length=None) / frames_to_audio(frames, hop_length, window='hann')
augment_audio(audio, sr, pitch_shift=0, time_stretch=1.0, noise_std=0.0)
```
`save_wav` silently rescales audio if `max(abs(audio)) > 1.0` (no clipping
surprise, but also no warning). `augment_audio`'s `pitch_shift`/
`time_stretch` use naive resampling with no anti-aliasing — adequate for
data-augmentation purposes, not studio-quality pitch/time manipulation.

### Cross-modal (`crossmodal_utils.py`)

```python
contrastive_loss(image_embeds, text_embeds, temperature=0.07)   # symmetric InfoNCE, CLIP-style
clip_normalize(embeddings)                                        # L2-normalize rows
multimodal_fusion(embeddings_list, fusion_type="concat", weights=None)
create_text_conditioned_image(image_shape, text_embed_dim, num_classes=10)
```
**Known gotcha:** `multimodal_fusion`'s docstring lists `fusion_type=
"attention"` as valid, but only `"concat"`, `"sum"`, and `"gated"` are
actually implemented — passing `"attention"` raises `ValueError` (documented
but not implemented; use `"gated"` if you want a learned-weighting-style
fusion). `create_text_conditioned_image` doesn't build any network — it
returns a plain dict of suggested hyperparameters, mostly ignoring its own
inputs beyond echoing them back; treat it as a config template, not a
working constructor.

## Model persistence

```python
model.Save("model.json")   # or "model.pkl" for pickle

model2 = NeuralNet()
model2.add_dense(20, 64, activation="relu")
model2.add_dense(64, 1, activation="sigmoid")
model2.Load("model.json")
```
```python
Save(file, save_opt_state=True, extra_state=None)
Load(file, load_opt_state=True)   # -> extra_state (dict or None)
```
`Save`/`Load` round-trip everything needed to resume training exactly where
you left off: layer parameters (weights/bias/gamma/beta/running stats/
attention Q-K-V-O/RNN-LSTM-GRU weights), optimizer state (`opt_state`,
unless `save_opt_state=False`), every per-model hyperparameter overridable
at construction (`learning_rate`, `l2_lambda`, `momentum`, `grad_clip_norm`,
`use_mixed_precision`, all Adam/RMSprop/Adagrad betas/epsilons), the
training-step counter `t`, in-progress gradient-accumulation buffers, the
train/eval mode flag, and the auto-shape-inference/residual-connection
bookkeeping needed to keep calling `add_*`/`add_residual_end` on the loaded
model — plus any `extra_state` dict you pass in (e.g. a diffusion model's
EMA weights). The target model's layer *shapes* must already match (build
the identical architecture before calling `Load`) — only values are
restored, not structure. Loading a file that's missing a given key silently
keeps the current model's value for that setting rather than erroring — a
partial/older save file won't crash `Load`, but also won't warn you about
what didn't get restored.

Other introspection/utility methods: `summary()`, `get_weights()`/
`set_weights()`, `freeze(layer_idx)`/`unfreeze(layer_idx)`,
`check_nan_inf()`, `copy()`, `train()`/`eval()`, `reset_optimizer_state()`.

## Configuration system

Every numeric default that would otherwise be a hardcoded "magic number"
lives in `Enilnets.constants` and can be overridden three ways:

```python
import Enilnets

# 1. Globally, for every model created afterward (read at call time, so this affects everything downstream immediately)
Enilnets.constants.EPS_LOG = 1e-10

# 2. Per-model, via constructor kwargs
model = NeuralNet(adam_beta1=0.9, adam_beta2=0.95, adam_epsilon=1e-6,
                  rmsprop_decay=0.95, adagrad_epsilon=1e-6)

# 3. Per-layer/per-call, via activation_params or loss/NEAT kwargs
model.add_dense(10, 10, activation="elu", activation_params={"alpha": 2.0})
model.ComputeLoss(out, y, function="huber", delta=0.5)
```

Full constant list (grouped by subsystem):

| Constant | Default | Subsystem |
|---|---|---|
| `EPS_LOG` | `1e-12` | Loss log guards |
| `EPS_DIV` | `1e-8` | Loss division guards (cosine/ntxent) |
| `SIGMOID_CLIP` | `500.0` | sigmoid/softplus exp overflow guard |
| `LEAKYRELU_ALPHA` | `0.01` | leakyrelu default slope |
| `ELU_ALPHA` | `1.0` | elu default alpha |
| `ADAM_BETA1` | `0.9` | Adam |
| `ADAM_BETA2` | `0.999` | Adam |
| `ADAM_EPSILON` | `1e-8` | Adam |
| `RMSPROP_DECAY` | `0.9` | RMSprop |
| `RMSPROP_EPSILON` | `1e-8` | RMSprop |
| `ADAGRAD_EPSILON` | `1e-8` | Adagrad |
| `NORMAL_INIT_STD` | `0.1` | `"normal"` weight init std |
| `SINUSOIDAL_BASE` | `10000.0` | Positional/time embedding base frequency |
| `NEAT_COMPATIBILITY_THRESHOLD` | `3.0` | NEAT speciation |
| `NEAT_C1` / `NEAT_C2` / `NEAT_C3` | `1.0` / `1.0` / `0.4` | NEAT compatibility distance |
| `NEAT_WEIGHT_MUTATE_RATE` | `0.8` | NEAT mutation |
| `NEAT_WEIGHT_PERTURB_RATE` | `0.9` | NEAT mutation (perturb vs. replace) |
| `NEAT_WEIGHT_PERTURB_POWER` | `0.5` | NEAT mutation magnitude |
| `NEAT_ADD_CONNECTION_RATE` | `0.05` | NEAT structural mutation |
| `NEAT_ADD_NODE_RATE` | `0.03` | NEAT structural mutation |
| `NEAT_CROSSOVER_RATE` | `0.75` | NEAT reproduction |
| `NEAT_SURVIVAL_THRESHOLD` | `0.2` | NEAT reproduction |
| `NEAT_STAGNATION_LIMIT` | `15` | NEAT species stagnation |

## Full API index

Everything importable as `from Enilnets import X`:

```
NeuralNet, LRScheduler,
VAE, GAN, DiffusionModel, AutoregressiveModel, RealNVP, EnergyBasedModel,
UNetDenoiser, time_embedding, TextGenerator, Tokenizer,
reparameterize, langevin_dynamics, gaussian_sample, uniform_sample,
gumbel_softmax_sample, random_mask, top_p_sampling, top_k_sampling,
kl_divergence_gaussian, adversarial_loss_discriminator,
adversarial_loss_generator, diffusion_loss, nll_loss, energy_loss,
compute_returns,
set_seed, train_test_split, iterate_minibatches, count_parameters,
EarlyStopping, one_hot, k_fold_split, constants,
confusion_matrix, classification_report,
NEATPopulation, NEATGenome, neat_crossover,
plot_network, plot_genome, to_html
```
Note the renames at the top level: `NEATGenome` = `neat.Genome`,
`neat_crossover` = `neat.crossover`, `one_hot` = `text_utils.one_hot_encode`.

**Available but not re-exported at the top level** (import the module
directly): `Enilnets.image_utils`, `Enilnets.audio_utils`,
`Enilnets.eval_utils`, `Enilnets.crossmodal_utils`, `Enilnets.weight_init`,
`Enilnets.layers`, `Enilnets.transformer_layers`, `Enilnets.activations`,
`Enilnets.loss`, `Enilnets.optimizer`, `Enilnets.train` (only `LRScheduler`
is re-exported from it — `TrainBatch`/`Train`/`compute_accuracy` are
reachable as bound `NeuralNet` methods; `compute_precision_recall_f1` needs
a direct import), `Enilnets.io`, `Enilnets.reinforce`, `Enilnets.forward`,
`Enilnets.backward`, `Enilnets.visualization` (only `plot_network`/
`plot_genome`/`to_html` re-exported), and within `Enilnets.generative`:
`Enilnets.generative.sampling.gae`.

## Known limitations

A consolidated list of every rough edge documented above, so you can decide
up front whether they matter for your use case:

- **No GPU support, by design.** Everything is NumPy on CPU. Large
  models/datasets will be slow compared to any GPU-backed framework — this
  library trades speed for full transparency/hackability.
- **Activation/optimizer name typos fail silently, not loudly.** An
  unrecognized `activation=` string behaves as linear (no error); an
  unrecognized `optimizer=` string behaves as Adam (no error). Double-check
  spelling — see [Activations](#activations) and [Optimizers](#optimizers).
- **`LRScheduler(mode="plateau")` is an unimplemented stub** — returns the
  initial learning rate unchanged forever; there's no real validation-
  plateau detection. Use `"cosine"`/`"warmup_cosine"`/`"step"` instead, or
  implement your own plateau logic around `model.set_lr(...)`.
- **`PPO`'s `value_targets`/`value_coeff` parameters are dead** — no value-
  loss term is implemented despite being accepted; you must train your
  critic separately.
- **`UNetDenoiser.backward()` unconditionally raises `NotImplementedError`**
  — it's forward-only, not trainable via the standard `Backward()`/
  `update()` flow. Use `DiffusionModel(denoiser_type="conv", ...)` if you
  need a trainable conv-based diffusion denoiser.
- **`multimodal_fusion(fusion_type="attention")` is documented but not
  implemented** — only `"concat"`, `"sum"`, `"gated"` actually work; passing
  `"attention"` raises `ValueError`.
- **`vgg_loss` is a literal placeholder** (`mean((x-y)**2)`, i.e. plain MSE)
  — there's no real VGG feature extraction behind it despite the name.
- **`eval_utils.inception_score(classifier=None)` is a crude k-means-based
  proxy**, not a real Inception-v3-based Inception Score — pass your own
  classifier for anything resembling the standard metric.
- **`kl_divergence` in `ComputeLoss` silently defaults `mu`/`logvar` to
  `output`/`target`** if you don't pass them explicitly as kwargs — always
  pass both explicitly; this loss isn't meant to be used as a generic
  output/target pair.
- **No padding option for `add_conv2d`** — output shrinks with kernel size;
  pad your input manually (`image_utils.pad_image`) for "same"-style
  convolutions.
- **RNN/LSTM/GRU have no fused/vectorized-across-time implementation** — the
  per-timestep Python loop is the unavoidable cost of a from-scratch
  recurrent layer; expect these to be the slowest layer type for long
  sequences. No bidirectional variant is built in.
- **`AutoregressiveModel.generate()`/`.complete()` have no KV-cache** (unlike
  `TextGenerator`) — generation is O(data_dim²) forward passes; slow for
  high-dimensional data.
- **`DiffusionModel.sample()`/`.denoise()` use standard ancestral sampling**
  (O(`time_steps`) sequential steps) — no fast-sampling (DDIM-style)
  variant is implemented yet.
- **`GAN.mode_collapse_score()`'s `n_clusters` parameter is unused** — the
  diversity heuristic doesn't actually cluster despite accepting the param.
- **`image_utils` PPM/PGM I/O is binary-only** (P6/P5, `maxval=255`) — no
  ASCII PPM/PGM support.
- **`image_utils.image_augmentation`'s `rotate` is a 90°-multiple ceiling**,
  not an arbitrary-angle rotation.
- **`audio_utils.augment_audio`'s pitch/time manipulation is naive
  resampling** (no anti-aliasing) — fine for data augmentation, not
  studio-quality processing.
- **`text_utils.load_texts_from_directory` silently skips files it can't
  decode** — no warning is printed for skipped files.
- **`train_test_split`/`iterate_minibatches` with `seed=None` use the global
  NumPy RNG state** — call `set_seed(...)` first if you need reproducibility
  without passing `seed=` explicitly everywhere.
- **`compute_precision_recall_f1` and `generative.sampling.gae` are not
  re-exported anywhere convenient** — import them from their defining module
  directly (see [Full API index](#full-api-index)).
- **`plot_network`'s edge-drawing only bridges through
  batchnorm/layernorm/dropout blocks** — any other intervening layer type
  (conv, attention, pooling, flatten, etc.) breaks the edge chain in the
  diagram, rendering as an opaque block with no drawn connections on either
  side (structurally accurate — there genuinely isn't a single weight matrix
  to draw there — but worth knowing if you expected every layer boundary to
  show connections).

## Running the test suite

`test_enilnets.py` is a single unittest-based suite (303 tests) covering
every layer/model/optimizer/utility above, plus a benchmark harness:

```bash
python test_enilnets.py                      # run all correctness tests
python test_enilnets.py -v                   # verbose
python test_enilnets.py TestRecurrentLayers   # run one class
python test_enilnets.py --benchmark          # timing only (Benchmark* classes)
```

Every gradient-bearing feature (attention, residual connections, RNN/LSTM/
GRU BPTT, KV-cache, NEAT crossover/cycle-safety) is checked against
finite-difference numerical gradients or explicit invariant checks, not just
"does it run" smoke tests.
