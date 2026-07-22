# Enilnets

**A neural network library built entirely on NumPy, with an optional CuPy
GPU backend.** No PyTorch, no TensorFlow, no C extensions — every layer,
optimizer, loss, generative model, reinforcement-learning algorithm, and
evolutionary algorithm is implemented from scratch with plain array math.
If you want to *see* exactly what happens to your data and gradients, line
by line, in readable Python, this is that library. NumPy (CPU) is always
the default and requires nothing extra to install; CuPy (GPU) is a
one-line opt-in for anyone who has it — see
[GPU mode (optional)](#gpu-mode-optional) below.

- **GitHub:** https://github.com/docenilno/Enilnets
- **License:** MIT
- **Version:** 4.11.0
- **Dependencies:** NumPy (required), CuPy (optional, for GPU mode)

This README is a complete guide, not just a reference: it explains what
each piece is *for*, when to reach for it, how to size it for your
problem, and shows a runnable example for essentially every public
function in the library. If you're new to neural networks generally, start
with [Choosing the right model for your task](#choosing-the-right-model-for-your-task)
— it explains the concepts as it goes.

## Table of contents

- [Install](#install)
- [GPU mode (optional)](#gpu-mode-optional)
- [Precision: float32 vs float64](#precision-float32-vs-float64)
- [Quickstart](#quickstart)
- [Choosing the right model for your task](#choosing-the-right-model-for-your-task)
  - [Structured / tabular data](#structured--tabular-data)
  - [Images](#images)
  - [Sequences: text, time series, audio](#sequences-text-time-series-audio)
  - [Generating new data](#generating-new-data)
  - [Sequential decision-making (reinforcement learning)](#sequential-decision-making-reinforcement-learning)
  - [You don't know what architecture you need](#you-dont-know-what-architecture-you-need)
  - [Sizing a model up or down](#sizing-a-model-up-or-down)
- [Core concepts](#core-concepts)
  - [The `NeuralNet` object](#the-neuralnet-object)
  - [Auto shape inference](#auto-shape-inference)
  - [The layer dict / `model.layers`](#the-layer-dict--modellayers)
- [Every layer type](#every-layer-type)
  - [Dense](#dense)
  - [Sparse](#sparse)
  - [Conv2D](#conv2d)
  - [Conv1D](#conv1d)
  - [Pooling (max / avg / global-avg)](#pooling-max--avg--global-avg)
  - [Upsample2D](#upsample2d)
  - [Flatten](#flatten)
  - [BatchNorm](#batchnorm)
  - [LayerNorm](#layernorm)
  - [Dropout](#dropout)
  - [Embedding](#embedding)
  - [Multi-head attention](#multi-head-attention)
  - [Cross-attention](#cross-attention)
  - [Positional encoding](#positional-encoding)
  - [Vision blocks](#vision-blocks)
  - [Multiplicative gating](#multiplicative-gating)
  - [Mixture-of-Experts](#mixture-of-experts)
  - [Transformer block](#transformer-block)
  - [KV-cache decoding](#kv-cache-decoding)
  - [Vision Transformer patch embedding](#vision-transformer-patch-embedding)
  - [RNN / LSTM / GRU](#rnn--lstm--gru)
  - [Bidirectional RNN/LSTM/GRU](#bidirectional-rnnlstmgru)
  - [Residual / skip connections](#residual--skip-connections)
  - [Convenience block builders](#convenience-block-builders)
  - [Layer type summary table](#layer-type-summary-table)
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
  - [Finding a learning rate](#finding-a-learning-rate)
  - [Weight averaging: EMA and SWA](#weight-averaging-ema-and-swa)
  - [Early stopping](#early-stopping)
  - [Callbacks](#callbacks)
  - [Accuracy / precision / recall / F1](#accuracy--precision--recall--f1)
- [Text generation (`TextGenerator`)](#text-generation-textgenerator)
- [Generative models](#generative-models)
  - [Which generative model should I use?](#which-generative-model-should-i-use)
  - [VAE](#vae)
  - [GAN](#gan)
  - [DiffusionModel](#diffusionmodel)
  - [RealNVP (normalizing flow)](#realnvp-normalizing-flow)
  - [EnergyBasedModel](#energybasedmodel)
  - [AutoregressiveModel](#autoregressivemodel)
  - [UNetDenoiser](#unetdenoiser)
  - [Class-conditional generation](#class-conditional-generation)
  - [Low-level sampling & loss building blocks](#low-level-sampling--loss-building-blocks)
  - [Bring your own pretrained weights](#bring-your-own-pretrained-weights)
- [Reinforcement learning](#reinforcement-learning)
- [NEAT (neuroevolution)](#neat-neuroevolution)
- [Visualization](#visualization)
- [Evaluation utilities](#evaluation-utilities)
- [Data utilities](#data-utilities)
  - [General (`core/utils.py`)](#general-coreutilspy)
- [Compression: pruning and quantization](#compression-pruning-and-quantization)
  - [Pruning](#pruning)
  - [Quantization](#quantization)
  - [Quantization-aware training](#quantization-aware-training)
  - [Differentiable audio front ends](#differentiable-audio-front-ends)
  - [Audio transform pipeline](#audio-transform-pipeline)
  - [Datasets and the DataLoader](#datasets-and-the-dataloader)
  - [Transform pipelines (`preprocessing/`)](#transform-pipelines-preprocessing)
  - [Text (`text/text_utils.py`)](#text-texttext_utilspy)
  - [Images (`vision/image_utils.py`)](#images-visionimage_utilspy)
  - [Audio (`audio/audio_utils.py`)](#audio-audioaudio_utilspy)
  - [Cross-modal (`crossmodal/crossmodal_utils.py`)](#cross-modal-crossmodalcrossmodal_utilspy)
  - [Dataset loaders (`datasets/loaders.py`)](#dataset-loaders-datasetsloaderspy)
- [Model persistence](#model-persistence)
- [Configuration system](#configuration-system)
- [Autograd (`Enilnets.graph`)](#autograd-enilnetsgraph)
- [Package layout](#package-layout)
- [Full API index](#full-api-index)
- [Known limitations](#known-limitations)
- [Running the test suite](#running-the-test-suite)
- [Contributing](#contributing)
- [License](#license)

## Install

Enilnets has no dependencies beyond NumPy.

```bash
pip install enilnets
```

```python
import Enilnets
print(Enilnets.__version__)  # "4.11.0"
```

## GPU mode (optional)

Enilnets runs on plain NumPy by default — nothing to configure, nothing
extra to install. If you have an NVIDIA GPU and want to accelerate
training, install a [CuPy](https://cupy.dev/) wheel matching your CUDA
toolkit version (pick one):

```bash
pip install cupy-cuda12x   # CUDA 12.x
pip install cupy-cuda13x   # CUDA 13.x
```

Then, **before building any models**, switch on GPU mode:

```python
import Enilnets

Enilnets.use_gpu(True)   # raises if CuPy isn't installed or no GPU is visible
print(Enilnets.is_gpu_enabled())  # True

net = Enilnets.NeuralNet(learning_rate=0.001)
net.add_dense(784, 128, activation="relu")
net.add_dense(128, 10, activation="softmax")
# weights, forward/backward, and optimizer state all live on the GPU from here on
```

Nothing else about the API changes — every `add_*` call, `Forward`,
`TrainBatch`, `Save`/`Load`, etc. works exactly the same whether the active
backend is NumPy or CuPy. A few things to know:

- **If `use_gpu(True)` fails after a system update, reboot before
  debugging anything else.** Upgrading the NVIDIA driver leaves the
  *running* kernel module mismatched against the *newly installed*
  userspace libraries until the machine restarts. In that window CUDA
  reports no usable device — but `nvidia-smi` keeps working, which makes
  it look like a library problem when it is not. `use_gpu()` probes the
  CUDA driver API separately from the CUDA runtime and, when only the
  former sees the GPU, says so explicitly instead of the misleading
  "no GPU detected".
- **`use_gpu()` is a single global switch**, not a per-model setting. Call
  it once before constructing any models; every model built afterward
  shares the active backend. Mixing backends across models in the same
  process isn't supported.
- **Reproducibility is per-backend.** `set_seed(n)` seeds whichever
  backend is currently active, but NumPy's legacy RNG and CuPy's
  cuRAND-backed generator don't produce identical sequences for the same
  seed — don't expect a CPU run and a GPU run with the same seed to match
  bit-for-bit.
- **Saved models are backend-agnostic.** `Save`/`Load` always round-trip
  through host NumPy arrays regardless of which backend built the model,
  so a `.pkl`/`.json` file saved on GPU loads fine on a CPU-only machine
  and vice versa.
- **GPU mode benefits `NeuralNet`'s batched Forward/Backward/optimizer
  path** (dense/conv/attention/RNN layers, large batches). **NEAT
  (`NEATPopulation`/`Genome`) always runs on host NumPy regardless of
  `use_gpu()`** — it evaluates one node at a time in a Python loop, and
  per-genome topologies vary across a population, so there's no shared
  computational graph to batch even within one generation; dispatching
  each tiny per-node op as its own GPU kernel measured ~17x *slower* than
  plain CPU. `use_gpu(True)` still applies to any `NeuralNet` you build
  elsewhere in the same process — this exception is specific to NEAT.
- Check `Enilnets.gpu_available()` to see if CuPy + a CUDA device are both
  available before calling `use_gpu(True)`, if you want to fail gracefully
  instead of catching the `RuntimeError`.

## Precision: float32 vs float64

**float32 is the default working precision** for every weight, activation,
and gradient in the library — the same default every mainstream deep
learning framework uses, and the one that makes GPU mode (above) actually
fast (consumer GPUs deliberately cripple float64 throughput to a fraction
of their float32 rate). If you need float64 instead — more numerically
sensitive work, matching an existing float64 pipeline, or just wanting the
old default back — switch it on the same way as GPU mode:

```python
import Enilnets

Enilnets.use_float64(True)   # before building any models
print(Enilnets.default_dtype())     # <class 'numpy.float64'>
print(Enilnets.is_float64_enabled())  # True

net = Enilnets.NeuralNet(learning_rate=0.001)
net.add_dense(784, 128, activation="relu")
# weights and activations are float64 from here on
```

A few things to know:

- **`use_float64()` is a single global switch**, same rule as `use_gpu()`:
  call it once before constructing any models; every model built
  afterward uses the active default. Mixing precisions across models in
  the same process isn't supported.
- **`use_mixed_precision=True`** (a `NeuralNet` constructor flag) forces
  the dense/conv matmul down to float32 specifically, regardless of the
  active default — meaningful when running in float64 mode (a real BLAS
  speedup on the hot path, master weights/gradients stay float64), a
  no-op if float32 is already the default.
- **Saved models are precision-faithful.** `Save`/`Load` record which
  dtype was active when a model was saved and always restore at that
  dtype, regardless of the *current* default when you call `Load` — a
  model saved under float32 loads back as float32 even after a later
  `use_float64(True)`, and vice versa. Files saved by versions before this
  existed (always float64) still load correctly.
- **Gradient/precision-sensitive work may want float64 explicitly** —
  finite-difference-style numerical checks, ill-conditioned optimization,
  or anything accumulating many small updates over a long run can lose
  meaningful precision in float32.

## Quickstart

The core workflow is always the same four steps: **build** a model by
chaining `add_*` calls, **train** it, **predict** with it, and optionally
**save** it. Here's the whole loop on a toy binary-classification problem:

```python
import numpy as np
from Enilnets import NeuralNet, set_seed, train_test_split

set_seed(0)  # reproducible random numbers

# 1. Get some data. (500 examples, 20 features each; label is whether the
#    first two features sum to something positive.)
X = np.random.randn(500, 20)
y = (X[:, 0] + X[:, 1] > 0).astype(np.float64).reshape(-1, 1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, seed=0)

# 2. Build the model: a small 3-layer MLP.
model = NeuralNet(learning_rate=0.001, optimizer="adam", l2_lambda=0.001)
model.add_dense(20, 64, activation="relu")   # input layer: 20 features in, 64 out
model.add_dense(64, 32, activation="relu")   # hidden layer
model.add_dense(32, 1, activation="sigmoid") # output layer: 1 probability out

# 3. Train it.
history = model.Train(X_train, y_train, epochs=20, batch_size=32,
                       X_val=X_val, Y_val=y_val, loss_function="binary_cross_entropy")

# 4. Use it.
preds = model.Forward(X_val, training=False)
print("val accuracy:", model.compute_accuracy(preds, y_val))

# 5. (Optional) Save and reload it later.
model.Save("model.json")
```

That's the whole pattern. Everything else in this README is either a
different kind of layer to `add_*` into that stack, a different kind of
model entirely (generative models, RL, NEAT), or a training/evaluation
detail. Every major section below opens with its own short "quickstart" so
you can jump straight to the part you need — see the table of contents.

## Choosing the right model for your task

This section is a map: given what you're trying to do, which class or
layer combination should you reach for, and roughly how big should it be?
Skip to whichever subsection matches your problem.

### Structured / tabular data

**Use:** [`NeuralNet`](#the-neuralnet-object) with [`add_dense`](#dense)
layers (a plain multi-layer perceptron / MLP).

If your input is a fixed-size vector of numbers or categories (spreadsheet
rows, sensor readings, engineered features) with no meaningful spatial or
sequential structure, a stack of dense layers is the right — and simplest
— tool:

```python
model = NeuralNet(optimizer="adam")
model.add_dense(n_features, 64, activation="relu")
model.add_dense(64, 32, activation="relu")
model.add_dense(32, n_outputs, activation="softmax")  # or "sigmoid"/"linear"
```
Or the one-line version: `model.add_mlp_block([64, 32], in_dim=n_features,
out_dim=n_outputs, out_activation="softmax")` (see
[Convenience block builders](#convenience-block-builders)).

**Sizing it:** each entry in the hidden-layer list is a layer's *width*
(how many neurons); the *length* of the list is the *depth* (how many
layers). More width lets a layer represent more distinct patterns at once;
more depth lets the network compose simpler patterns into more complex
ones. For tabular data, 2-4 hidden layers of 32-256 neurons each is a
reasonable starting range — go wider/deeper only if you're clearly
underfitting (training accuracy itself is poor), and pull back (fewer/
smaller layers, more `l2_lambda`, add `add_dropout()`) if you're
overfitting (training accuracy is much higher than validation accuracy).
Use `model.summary()` or `count_parameters(model)` to see exactly how many
learned numbers your choice adds up to.

### Images

**Use:** [`NeuralNet`](#the-neuralnet-object) with
[`add_conv2d`](#conv2d)/[`add_conv_block`](#convenience-block-builders)
layers (a convolutional neural network / CNN), or
[`add_vision_transformer_patch_embed`](#vision-transformer-patch-embedding)
+ [`add_transformer_block`](#transformer-block) for a Vision Transformer.

Images have *spatial* structure — a pattern found in one corner should be
recognized the same way in another corner. Convolution layers exploit this
by sharing the same small set of weights (a "kernel") across every spatial
position, which is both far more parameter-efficient and a better
inductive bias than treating every pixel as an independent feature (what a
dense layer would do). The standard recipe is: a few
`conv → [batchnorm] → pool` blocks that shrink the spatial size while
growing the channel count, then flatten and finish with dense layers:

```python
model = NeuralNet(optimizer="adam")
model.add_conv_block(out_ch=16, k=3, in_ch=1, batchnorm=True, pool="max", input_size=(28, 28))
model.add_conv_block(out_ch=32, k=3, batchnorm=True, pool="max")
model.add_flatten()
model.add_dense(n_out=64, activation="relu")
model.add_dense(n_out=10, activation="softmax")
```

**Sizing it:** each `add_conv_block` roughly doubles the channel count as
spatial size halves (16→32→64→128 channels is a common progression) — more
blocks (depth) let the network recognize increasingly abstract/composite
shapes; more channels per block (width) let it recognize more distinct
low-level patterns at each stage. For small images (MNIST-sized, 28x28),
2-3 blocks is plenty; for larger images, more blocks. Vision Transformers
(patchify the image, then run ordinary transformer blocks) scale better to
very large images and datasets but need much more training data to match a
CNN's built-in spatial inductive bias on small datasets — prefer CNN unless
you have a lot of data or specifically want attention's global receptive
field from layer one.

### Sequences: text, time series, audio

Three tools, pick based on what you need:

| You need... | Use | Why |
|---|---|---|
| Local patterns in raw audio/sensor data, translation-invariant | [`add_conv1d`](#conv1d) | Same weight-sharing idea as Conv2D, along one axis (time) instead of two (space). |
| To process a sequence step-by-step, need it to work on variable lengths, care about "what happened before now" (causal, one direction) | [`add_rnn`/`add_lstm`/`add_gru`](#rnn--lstm--gru) | Explicit hidden state carried forward through time; naturally handles arbitrary-length sequences. LSTM/GRU handle long-range dependencies much better than plain RNN. |
| Context from *both* directions matters (e.g. classifying a token using words that come after it too), not generating new tokens | [`add_bidirectional_rnn`/`_lstm`/`_gru`](#bidirectional-rnnlstmgru) | Runs one RNN forward and one over the reversed sequence, concatenates both — sees the whole sequence at each position. |
| Long-range dependencies, want to attend directly to any other position regardless of distance, generating text | [`add_multihead_attention`](#multi-head-attention) / [`add_transformer_block`](#transformer-block), or the ready-made [`TextGenerator`](#text-generation-textgenerator) | Every position can directly attend to every other position in one step (no need to carry information step-by-step through a hidden state); this is what modern language models are built from. |

Text generation specifically has a ready-made class — see
[TextGenerator](#text-generation-textgenerator) — you usually don't need
to hand-build the transformer stack yourself:

```python
from Enilnets import TextGenerator, Tokenizer

tokenizer = Tokenizer(vocab_size=128, level="char").fit([my_corpus])
gen = TextGenerator(tokenizer, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=64)
gen.Train([my_corpus], epochs=10, batch_size=32, seq_len=32)
print(gen.generate(prompt="once upon a", max_new_tokens=100, temperature=0.8, top_p=0.9))
```

**Sizing it:** for RNN/LSTM/GRU, `hidden_dim` is the width (how much the
network can "remember" at once) and stacking multiple `add_lstm(...)`
calls in sequence (with `return_sequences=True` on all but the last) is
the depth. For attention, `embed_dim` is the width, `num_heads` splits
that width into parallel attention "views" (must divide `embed_dim`
evenly), and the number of `add_transformer_block` calls (or
`num_layers` on `TextGenerator`) is the depth. Bigger/deeper networks fit
more complex language/sequence structure but need proportionally more
training data and time — for character-level toy corpora, `embed_dim=64,
num_heads=4, num_layers=2` is already plenty; scale up by roughly the same
factor GPT-family models do (embed_dim and depth grow together) as your
real dataset grows.

### Generating new data

See the full [generative models](#which-generative-model-should-i-use)
decision table below — short version: VAE for a smooth, fast, "good
enough" generator with a useful latent space; GAN for the sharpest samples
if you can tolerate finicky training; DiffusionModel for the best sample
quality if you can tolerate slow (or DDIM-accelerated) generation;
RealNVP/AutoregressiveModel if you specifically need an exact likelihood
number, not just samples.

### Sequential decision-making (reinforcement learning)

**Use:** any `NeuralNet` as a policy, trained with
[`Reinforce`/`PPO`/`ActorCritic`](#reinforcement-learning), or
[`Evolve`](#reinforcement-learning) if you don't have a differentiable
reward signal at all.

If your problem is "an agent takes actions in an environment and gets
rewarded/penalized over time" (games, control, resource allocation) rather
than "predict a label for a fixed input," you want reinforcement learning,
not supervised training:

```python
from Enilnets import NeuralNet, compute_returns

policy = NeuralNet(optimizer="adam")
policy.add_dense(state_dim, 64, activation="relu")
policy.add_dense(64, n_actions, activation="softmax")

returns = compute_returns(rewards, gamma=0.99)
policy.Reinforce(states, actions, returns, action_type="discrete")
```

**Sizing it:** same width/depth guidance as tabular MLPs above — RL
policies for simple control tasks are often surprisingly small (1-2 hidden
layers of 32-128 units).

### You don't know what architecture you need

**Use:** [`NEATPopulation`](#neat-neuroevolution) — it evolves both the
weights *and* the topology (which neurons exist, how they're wired) from a
minimal starting point, guided only by a fitness function you supply. No
gradients, no architecture decisions up front. Good for small problems
where you want the structure discovered rather than designed; not a
gradient-descent replacement for large-scale training.

### Sizing a model up or down

Across every model type in this library:

- **Width** (neurons per layer / channels / `hidden_dim` / `embed_dim`)
  controls how much a single layer can represent at once. Wider = more
  capacity per layer, more parameters, slower.
- **Depth** (number of layers) controls how many times the network can
  compose/refine what it's learned. Deeper = more abstraction, more
  parameters, slower, and (for architectures without residual connections)
  harder to train.
- **Check your actual parameter count** rather than guessing:
  ```python
  model.summary()                     # prints a layer-by-layer table
  total, per_layer = count_parameters(model)   # -> int, list of dicts (for use in code)
  ```
- **Signs you're too small (underfitting):** training loss/accuracy itself
  is poor, even after many epochs. Fix: more width, more depth, train
  longer, or a better-suited layer type (e.g. conv instead of dense for
  images).
- **Signs you're too big (overfitting):** training accuracy is much better
  than validation accuracy. Fix: fewer/narrower layers, `add_dropout()`,
  raise `l2_lambda`, get more training data, or stop training earlier
  (`EarlyStopping` — see [Training](#early-stopping)).

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
          rmsprop_decay=0.9, rmsprop_epsilon=1e-8, adagrad_epsilon=1e-8,
          adadelta_rho=0.95, adadelta_epsilon=1e-6,
          lion_beta1=0.9, lion_beta2=0.99,
          adafactor_eps1=1e-30, adafactor_clip_threshold=1.0,
          adafactor_decay_rate=-0.8)
```

Every one of these is a plain attribute you can also read/write after
construction (`model.learning_rate = 0.0001`, etc.) — there are no hidden
private copies. `optimizer` must be one of `"sgd"`, `"rmsprop"`,
`"adagrad"`, `"adadelta"`, `"adam"`, `"adamw"`, `"adamax"`, `"nadam"`,
`"radam"`, `"lion"`, `"lamb"`, `"adafactor"` — any other string (including
typos) raises `ValueError` at construction time. See [Optimizers](#optimizers) for
what each one means and when to pick it.

Other `NeuralNet` methods not tied to a specific layer or training call:

| Method | What it does |
|---|---|
| `train()` / `eval()` | Set `self.training = True`/`False`; both return `self` (chainable). Layers that behave differently in train vs. eval (dropout, batchnorm) read `training` from the `Forward(training=...)` argument, not this flag directly — `train()`/`eval()` just set a convenience default you can check yourself. |
| `set_lr(lr)` / `get_lr()` | Get/set `learning_rate`. |
| `clip_gradients(max_norm)` | Clips the global L2 norm across every trainable parameter's actual gradient (computed via `compute_gradients()`) to `max_norm` -- standard `clip_grad_norm_`-style semantics, not just the norm of backprop deltas. No-op if `max_norm <= 0`. Called automatically by `TrainBatch` when `grad_clip_norm > 0`. |
| `freeze(layer_idx=None)` / `unfreeze(layer_idx=None)` | Mark one layer (or all, if `None`) with `_frozen=True/False`; `apply_gradients`/`update` skip frozen layers entirely. Useful for transfer learning: freeze early layers, only train the head. |
| `get_weights()` / `set_weights(weights)` | Snapshot/restore just the weight arrays (not optimizer state) as a list of dicts — lighter-weight than full `Save`/`Load` for e.g. checkpointing the best epoch. |
| `copy()` | Deep-copies the entire model (layers, optimizer state, step counter, shape-inference bookkeeping, residual stack) into a new independent `NeuralNet`. |
| `reset_optimizer_state()` | Clears `opt_state`, `t`, and gradient-accumulation buffers (e.g. before fine-tuning with a different optimizer). |
| `check_nan_inf()` | Returns a list of human-readable strings describing any NaN/Inf found in weights/biases/gamma/beta/deltas — empty list means clean. Useful when training suddenly produces `nan` losses. |
| `summary()` | Prints a layer-by-layer shape/parameter-count table to stdout. Returns `None` — for a value you can use in code, see `count_parameters(model)` in [Data utilities](#general-coreutilspy). |
| `predict(x)` | A plain alias for `Forward(x, training=False, dropout_rate=0.0)` — same signature, same code path, just a familiar name. |

```python
model = NeuralNet(optimizer="adam")
model.add_dense(10, 20, activation="relu")
model.add_dense(20, 5, activation="softmax")
model.summary()
# Model Summary
# ======================================================================
# Optimizer: ADAM | LR: 0.001 | L2: 0.01
# ======================================================================
# Layer 0: DENSE        Input:     10 Output:     20 Params: 220
# Layer 1: DENSE        Input:     20 Output:      5 Params: 105
# Total Parameters: 325
```

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
to build up `model.layers`. For a "which layer for which job" overview, see
[Choosing the right model](#choosing-the-right-model-for-your-task) above;
this section is the complete reference for each one, including a summary
table at the end.

### Dense

```python
add_dense(n_in=None, n_out=128, activation="relu", init_method="xavier_uniform",
          use_bias=True, activation_params=None)
```
A standard fully-connected layer: `y = activation(x @ W.T + b)`. Every
input feature can influence every output — the most general-purpose layer,
and the building block of the "head" at the end of almost every other
architecture. `n_in=None` auto-infers. `activation_params` is a dict
forwarded to the activation function (e.g. `{"alpha": 0.2}` for
leakyrelu).

```python
model.add_dense(20, 64, activation="relu")
model.add_dense(64, 10, activation="softmax")
```

- **Use it for:** tabular/structured data, or as the final classification/
  regression head after conv/RNN/attention layers have extracted features.
- **Pros:** the fastest, best-understood, most numerically stable layer in
  the library; supports 2D `(batch, features)` and 3D `(batch, seq, features)`
  input transparently.
- **Cons:** no parameter sharing — doesn't scale to large spatial/sequential
  inputs the way conv/attention/RNN do (a 1000x1000 image flattened into a
  dense layer would need a million input weights per neuron).

### Sparse

```python
add_sparse(n_in=None, n_out=128, connectivity=0.5, activation="relu",
           init_method="xavier_uniform", activation_params=None)
```
A dense layer with a fixed random binary connectivity mask
(`Bernoulli(connectivity)`), applied once at construction and reapplied to
every weight gradient forever after — the sparsity pattern **never
changes** during training.

```python
model.add_sparse(64, 64, connectivity=0.3, activation="relu")  # ~30% of connections active
```

- **Use it for:** experimenting with fixed-sparsity architectures, or
  reducing parameter count vs. a same-width dense layer.
- **Pros:** fewer effective parameters than dense at the same width.
- **Cons:** the mask is random and static — no structured sparsity, no
  pruning-during-training, no guarantee the fixed pattern is a good one. If
  you want dynamically-evolving structure, see [NEAT](#neat-neuroevolution)
  instead.

### Conv2D

```python
add_conv2d(in_ch=None, out_ch=32, k=3, activation="relu", init_method="he_normal",
           stride=1, activation_params=None, input_size=None, padding="valid")
```
2D convolution implemented via `im2col` + matrix multiply (the standard
NumPy-conv trick — turns convolution into one big `dot()` call instead of
nested loops). `input_size` is only needed on the very first conv call, to
let a later `add_flatten()` compute its output width.

`padding`: `"valid"` (default — no padding, so `k>1` shrinks spatial size)
or `"same"` (output spatial size equals input spatial size). `"same"` only
supports `stride=1` and an odd kernel size `k` (raises `ValueError`
otherwise — avoids asymmetric-padding edge cases); for even `k` or
`stride>1`, pad your input manually with `image_utils.pad_image` instead.
`add_conv_block(...)` also accepts `padding=`, forwarded straight through.

```python
model.add_conv2d(in_ch=3, out_ch=16, k=3, input_size=(32, 32), padding="same")
model.add_maxpool2d(2)   # -> spatial size halves, channels stay 16
model.add_conv2d(out_ch=32, k=3, padding="same")
```

- **Use it for:** images, or any 2D-grid data (spectrograms, sensor grids)
  where a pattern found in one region should be recognized the same way
  elsewhere. See [Images](#images) above.
- **Pros:** vectorized (`im2col` + BLAS matmul), reasonably fast for a pure-
  NumPy conv; supports arbitrary stride; `"same"` padding available for the
  common `stride=1`/odd-`k` case.
- **Cons:** `"same"` padding doesn't cover `stride>1` or even `k`; `im2col`
  materializes every patch, so memory scales with
  `batch * out_h * out_w * in_ch * k * k` — large images/kernels can be
  memory-hungry compared to a true sliding-window convolution.

### Conv1D

```python
add_conv1d(in_ch=None, out_ch=32, k=3, activation="relu", init_method="he_normal",
           stride=1, activation_params=None, input_size=None, padding="valid")
```
1D convolution for `(batch, channels, length)` data (audio, raw waveforms,
time series) — mirrors `add_conv2d` throughout, including the
`padding="valid"|"same"` restriction (`"same"` only for `stride=1` + odd
`k`). `input_size` here is a plain int length `L`, not an `(H, W)` tuple.

```python
model = NeuralNet(optimizer="adam")
model.add_conv1d(in_ch=1, out_ch=32, k=5, input_size=1000, padding="same", activation="relu")
model.add_conv1d(out_ch=64, k=3, padding="same", activation="relu")
model.add_flatten()
model.add_dense(None, 10, activation="softmax")
```

- **Use it for:** raw audio/waveform classification, 1D sensor time series,
  any sequence where local (nearby-in-time) patterns matter more than
  long-range dependencies. See [Sequences](#sequences-text-time-series-audio)
  above.
- **Pros:** same `im2col` + BLAS matmul approach as `add_conv2d`, same
  `stride`/`padding` support.
- **Cons:** same memory-scaling caveat as `add_conv2d`; no dilated/causal-
  conv variant.

### Pooling (max / avg / global-avg)

```python
add_maxpool2d(pool_size=2)
add_avgpool2d(pool_size=2)
add_global_avgpool2d()
```
Non-overlapping spatial pooling (`pool_size` is both window and stride) —
shrinks the spatial size, reducing compute for later layers and providing
a small amount of translation invariance. `add_global_avgpool2d()` reduces
the full spatial extent to `1x1` (as used before a classification head in
modern CNNs, avoiding a huge flatten+dense).

```python
model.add_conv2d(in_ch=1, out_ch=16, k=3, input_size=(28, 28))
model.add_maxpool2d(2)   # (16, 26, 26) -> (16, 13, 13)
```

- **Use it for:** shrinking spatial size between conv blocks (max/avg), or
  replacing the final flatten+dense with something size-independent
  (global-avg).
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
architectures to grow spatial size back after it's been shrunk.

```python
model.add_upsample2d(2)  # (C, 8, 8) -> (C, 16, 16)
```

- **Use it for:** decoder/generator architectures (autoencoders, GANs,
  diffusion U-Nets) that need to grow spatial size back up.
- **Pros:** simple, cheap, no learned parameters.
- **Cons:** nearest-neighbor only — no bilinear/transposed-convolution
  upsampling built into the layer API (see `image_utils.resize_bilinear` for
  a non-layer alternative on raw arrays).

### Flatten

```python
add_flatten()
```
`(B, C, H, W) → (B, C*H*W)`. Required to transition from conv/pool layers
into dense layers — dense layers only understand flat feature vectors.

```python
model.add_conv2d(1, 16, k=3, input_size=(28, 28))
model.add_flatten()
model.add_dense(None, 10, activation="softmax")  # n_in auto-inferred from the flatten
```

### BatchNorm

```python
add_batchnorm(num_features=None, epsilon=1e-5, momentum=0.1)
```
Standard batch normalization: rescales each feature/channel to zero mean,
unit variance across the current batch (+ spatial axes, for 4D input),
then applies a learned scale/shift. Running mean/variance are tracked for
inference (`training=False` uses the running stats instead of the current
batch's).

```python
model.add_conv2d(1, 16, k=3, input_size=(28, 28))
model.add_batchnorm(16)   # one running-stat pair per channel
model.add_maxpool2d(2)
```

- **Use it for:** stabilizing/speeding up training in deeper conv stacks —
  add it after conv layers, before the activation-bearing pooling/next
  conv.
- **Pros:** stabilizes/accelerates training, especially for deeper conv
  stacks; supports both 2D and 4D input.
- **Cons:** behavior depends on batch size (small batches → noisy
  statistics — prefer LayerNorm if your batches are tiny); running stats
  must be carried consistently through save/load (they are — see
  [Model persistence](#model-persistence)).

### LayerNorm

```python
add_layernorm(normalized_shape=None, epsilon=1e-5)
```
Normalizes over the feature axis per-example (independent of batch size),
supporting 2D `(batch, features)`, 3D `(batch, seq, features)` — normalizing
over the embedding axis only, standard Transformer-style — and 4D
`(batch, C, H, W)` input.

```python
model.add_multihead_attention(embed_dim=64, num_heads=4)
model.add_layernorm(64)
```

- **Use it for:** transformers and RNNs (the standard normalization choice
  there), or anywhere batch size is small/variable so BatchNorm's
  batch-dependent statistics would be unreliable.
- **Pros:** batch-size-independent (unlike BatchNorm); works cleanly with
  variable sequence lengths.
- **Cons:** slightly more compute per example than BatchNorm (no shared
  running statistics to amortize at inference).

### Dropout

```python
add_dropout(rate=0.5)
```
Randomly zeroes `rate` fraction of activations at training time (rescaling
survivors so the expected sum stays the same), forcing the network not to
rely too heavily on any single neuron — a standard regularizer against
overfitting. Only active when `Forward(..., training=True)`.

```python
model.add_dense(64, 64, activation="relu")
model.add_dropout(0.3)   # 30% of activations zeroed during training only
model.add_dense(64, 10, activation="softmax")
```

- **Use it for:** reducing overfitting, especially in dense layers with a
  lot of parameters relative to your dataset size.
- **Pros:** simple, well-understood regularizer.
- **Cons:** per-layer `rate` overrides the global `Forward(dropout_rate=...)`
  default — easy to forget you set a per-layer rate when trying to
  globally disable dropout for a quick experiment.

### Embedding

```python
add_embedding(vocab_size, embed_dim, init_method="normal")
```
A lookup table `(vocab_size, embed_dim)` mapping integer token IDs to
dense vectors — the standard first layer for any text/categorical-sequence
model. Input is integer indices, `(batch, seq_len)` or `(batch,)`.

```python
model.add_embedding(vocab_size=10000, embed_dim=128)   # token id -> 128-dim vector
model.add_lstm(128, 256)
```

- **Use it for:** the first layer of any text model, or any model whose
  input is categorical IDs rather than continuous numbers.
- **Pros:** efficient sparse gradient (scales with tokens actually seen in
  the batch, not full vocab size); works for both `(batch,)` single-token
  and `(batch, seq)` sequence input.
- **Cons:** no weight-tying helper (if you want tied input/output embeddings
  for a language model, you must manage that manually by sharing the array
  reference yourself).

### Multi-head attention

```python
add_multihead_attention(embed_dim=None, num_heads=4, dropout=0.0,
                        init_method="xavier_uniform", causal=False,
                        positional_scheme="absolute", num_kv_heads=None,
                        window_size=None, attention_kernel="softmax",
                        num_features=None, sparse_pattern=None,
                        tiled_block_size=None)
```
Standard scaled dot-product multi-head self-attention: every position in
the sequence computes a weighted average of every *other* position,
weighted by how relevant they are to each other (learned via Query/Key/
Value projections). This is what lets transformers model long-range
dependencies in one step, instead of carrying information step-by-step
like an RNN. `embed_dim` must be divisible by `num_heads`.
`causal=True` applies an autoregressive mask (position *i* attends only to
`j <= i`) — this is what `TextGenerator` builds on for text generation.

```python
model.add_embedding(vocab_size, embed_dim=64)
model.add_multihead_attention(embed_dim=64, num_heads=4, causal=True)
```

`positional_scheme` (attention has no inherent sense of position — you need
one of these, or a separate [positional encoding](#positional-encoding)
layer, so the model can tell *where* in the sequence each token is):
- `"absolute"` (default) — no positional info added here; pair with
  `add_positional_encoding()`.
- `"rope"` (rotary position embedding) — rotates Q/K per head based on
  position before the score dot product; encodes *relative* position
  naturally. Requires an even `head_dim`.
- `"alibi"` — a static per-head bias that penalizes attending to distant
  positions, added directly to the scores. No extra parameters.

`num_kv_heads` selects **MHA / MQA / GQA** — how many Key/Value heads to
project, shared across the `num_heads` Query heads:
- `None` (default) → `num_heads` — ordinary **Multi-Head Attention**, every
  query head gets its own K/V head. Byte-identical to the pre-existing
  behavior.
- `1` → **Multi-Query Attention**: all query heads read one shared K/V head.
- any divisor of `num_heads` → **Grouped-Query Attention**: query heads are
  split into `num_kv_heads` groups, each group sharing one K/V head.

Query head *h* reads K/V group `h // (num_heads // num_kv_heads)` —
group-major, the same convention as every reference implementation. This
shrinks `Wk`/`Wv` and — the reason it actually matters — the
[KV cache](#kv-cache-decoding) by `num_heads / num_kv_heads`, which is what
bounds memory during long generations. Quality cost is small in practice,
which is why essentially every modern decoder ships GQA.

```python
model.add_transformer_block(512, num_heads=8, causal=True, num_kv_heads=2)
#  -> 8 query heads, 2 K/V heads: a 4x smaller KV cache
```

`window_size` turns on **sliding-window attention**: position *i* attends
only where `|i - j| <= window_size` (`None`, the default, is unbounded).
It composes with `causal`, which then leaves exactly the `window_size`
preceding positions plus *i* itself — the Mistral-style decoder pattern.
`window_size=0` degenerates to each position seeing only itself.

Note what this does and does not buy you. On its own `window_size` is a
*mask*: the S×S score matrix is still built and then masked, so training
time and memory are unchanged (measured: 954 ms / 824 MB vs. 959 ms / 824 MB
at B=4, S=1024, E=64, H=8). To turn the restriction into an actual saving,
combine it with `tiled_block_size` below — same numbers, 143 MB. Where the
window pays off unconditionally is *generation*, described next.

The modelling trade is that a window only reaches further context indirectly,
through stacked layers (a stack of *L* windowed layers has an effective receptive field of
roughly `L * window_size`). During generation the effect is stronger still:
[the KV cache](#kv-cache-decoding) **evicts** positions that have fallen out
of the window rather than masking them, so decoding runs in constant memory
per layer (`window_size + 1` entries) no matter how long the output gets.

```python
model.add_transformer_block(512, num_heads=8, causal=True, window_size=256)
```

Because `j == i` is always allowed, no query row is ever fully masked — the
softmax stays well-defined for every setting.

`attention_kernel` picks the attention *math* itself. Softmax attention has
to build the full S×S score matrix, which is why cost grows quadratically
with sequence length. The linearized kernels replace `softmax(q·k)` with a
feature map φ where the score is just `φ(q)·φ(k)` — and because that is a
plain inner product, the sum reassociates:

```
       softmax:  (Q K^T) V     ->  O(S^2 * d)
    linearized:  φ(Q) (φ(K)^T V)  ->  O(S * d^2)      linear in S
```

- `"softmax"` (default) — exact, unchanged, byte-identical to before.
- `"linear"` — φ(x) = elu(x) + 1 (Katharopoulos et al. 2020). A positive
  similarity in its own right, *not* an approximation of softmax; models
  trained with it simply learn a different (cheaper) attention.
- `"performer"` — FAVOR+ random features (Choromanski et al. 2021):
  φ(x) = exp(ω·x′ − ‖x′‖²/2)/√m with a fixed random ω drawn at build time.
  This one *is* an unbiased estimator of softmax attention, so accuracy
  improves with `num_features` (default `head_dim * 4`). ω is saved and
  loaded with the model — it defines the layer's function, so redrawing it
  would silently change the model.

```python
model.add_transformer_block(256, num_heads=8, causal=True,
                            attention_kernel="performer", num_features=128)
```

Causal masking still works: the two global sums become *prefix* sums
(`cumsum`), so position *i* only ever aggregates positions ≤ *i*.

**During generation this is the strongest form of caching in the library.**
Causal linear attention is a linear RNN — the entire history compresses into
a running `Σ φ(k)⊗v` and `Σ φ(k)` — so
[the KV cache](#kv-cache-decoding) keeps a **fixed-size** state that never
grows with the generated length, rather than a K/V tensor per position.

- **Limitations, and why:** these kernels never materialize a score matrix,
  so anything defined *as* a score-matrix bias or mask has nowhere to live —
  `positional_scheme="alibi"`, `window_size`, and attention `dropout` are
  rejected with an explicit error rather than silently ignored. RoPE, MQA/GQA
  and causal masking all work normally.
- **Cost note:** the causal path materializes the prefix sums, which is
  O(S · d_feat · head_dim) *memory*. That is the price of the readable
  vectorized formulation here; for very long sequences with many Performer
  features it can dominate, so prefer a modest `num_features`.

`sparse_pattern` turns on **block-sparse attention**. The sequence is cut
into blocks, and each *query block* attends to a small chosen set of *key
blocks*. Crucially this is not a mask over a matrix that got built anyway:
the selected blocks are physically **gathered**, so the S×S score matrix
never exists and the cost is `O(S · blocks_per_query · block_size)`.

```python
model.add_transformer_block(256, num_heads=8, causal=True, sparse_pattern={
    "block_size": 64,   # sequence is cut into blocks of this many positions
    "local": 1,         # ...each block sees its 1 nearest neighbour block(s)
    "global": 1,        # ...plus the first 1 block(s), visible to everyone
    "random": 2,        # ...plus 2 random blocks, fixed at build time
    "seed": 0,          # ...drawn from this seed, so the pattern round-trips
})
```
The components form a union, which is the Longformer/BigBird recipe: local
blocks capture nearby detail, global blocks give every position a shared
channel to route information through, and random blocks keep the graph's
diameter small so information can still cross the sequence in few layers. A
block always sees itself, so no query is ever left with nothing to attend
to. `local`/`global`/`random` all default such that a bare
`{"block_size": n}` gives local-only attention.

The pattern is a property of the model: the random choices come from `seed`
and round-trip through save/load, so a reloaded model attends identically.
For causal patterns the selection for block *i* is independent of how many
blocks follow it, which is what lets [the KV cache](#kv-cache-decoding) step
one token at a time and land on exactly the same numbers as a full forward
pass.

- **Use it for:** long sequences where you want *exact* softmax attention
  over a restricted, structured neighbourhood — as opposed to a linearized
  kernel, which approximates full attention instead of restricting it.
- **Pros:** the gathered result is pinned against a dense masked-attention
  oracle to 1e-12; gradients FD-checked standalone and end-to-end; works with
  RoPE, ALiBi, MQA/GQA, causal or not; any sequence length (the tail is
  padded internally and masked out, so `S` need not divide `block_size`).
- **Cons/limitations, and why:** it cannot combine with `window_size` (both
  restrict which keys a query sees — express the window as `local` blocks
  instead) or with a linearized `attention_kernel` (there is no score matrix
  to sparsify). Cached decoding keeps the **full** K/V history, since a
  pattern may select any earlier block — the saving there is in per-step
  compute, not memory.

`tiled_block_size` opts into the **streaming ("Flash") softmax** path. Be
clear about what that means here. FlashAttention is two things: a fused CUDA
kernel, and the algorithm that makes fusing worth doing. A NumPy/CuPy
library cannot write the kernel — but the algorithm is portable, and it is
where the memory win actually comes from. Instead of building the whole
S×S score matrix and then reducing it, this streams over key blocks while
carrying an **online softmax** (a running max, running normalizer, and
running output that are rescaled as each block arrives):

```python
model.add_transformer_block(512, num_heads=8, causal=True, tiled_block_size=128)
```

Peak memory per head drops from `O(S²)` to `O(S · block)`. The backward
pass does the same thing in reverse — it recomputes each score block from
Q/K/V rather than storing the attention weights, needing only the row sums
`rowsum(dO ∘ O)` — so it is O(S · block) too.

- **It is a backend, not a replacement.** `tiled_block_size=None` (the
  default) leaves the original code path byte-for-byte untouched. When
  enabled, the results are *the same numbers*: a test asserts equality with
  the plain path to 1e-12 across every combination of causal, window,
  positional scheme and block size, and asserts the gradients match the
  plain path's as well as finite differences.
- **Expect memory savings, not speed.** Without kernel fusion there is no
  reduction in arithmetic. Measured on a causal layer at B=4, S=1024, E=64,
  H=8: peak allocation falls from **824 MB to 143 MB** (5.8x) while the
  forward pass runs 959 ms → 819 ms. It buys you sequence lengths that would
  otherwise not fit.
- **Limitations, and why:** incompatible with attention `dropout` (the
  streaming path never holds the full attention matrix, so there is no
  single object to apply one consistent mask to), and with `sparse_pattern`
  or the linearized kernels (both already avoid building a score matrix —
  there is nothing left to tile).

- **Use it for:** text (via `TextGenerator` or manually), or any sequence
  task where long-range dependencies matter more than step-by-step
  recurrence. See [Sequences](#sequences-text-time-series-audio) above.
- **Pros:** the mask, softmax, and backward pass (including all positional
  schemes and every `num_kv_heads` setting) are verified against
  finite-difference gradients; MQA/GQA are additionally pinned against an
  independent oracle (plain MHA with the K/V weights tiled out by hand),
  and against the `graph` implementation under shared weights;
  supports arbitrary `num_heads`/`embed_dim` combinations.
- **Cons:** this layer is self-attention only (Q/K/V all from the same
  input) — see `add_cross_attention` below for encoder-decoder-style
  attention with a separate key/value source.

`dropout` applies standard post-softmax attention-weight dropout (dropped
after the softmax, before the context matmul, at `training=True` only) —
gradient-verified against finite differences with the dropout mask held
fixed across perturbations.

### Cross-attention

```python
add_cross_attention(kv_source_index, embed_dim=None, num_heads=4,
                    dropout=0.0, init_method="xavier_uniform",
                    num_kv_heads=None)
```
Encoder-decoder-style attention: queries come from the normal sequential
`x`; keys/values come from an *earlier* layer's output, named by
`kv_source_index` (the index of that earlier layer in `model.layers`).
This is the mechanism behind translation-style "decoder attends to
encoder" architectures. No causal masking (cross-attention conventionally
attends freely over the full KV source).

Because a `NeuralNet` only ever threads one sequential `x` through
`Forward()`, an encoder-decoder split has to be built as two branches
sharing a common source rather than two truly independent inputs — use the
internal `"goto"` layer type to fork back to an earlier point for the
second branch:

```python
model = NeuralNet(optimizer="adam")
model.add_embedding(vocab_size=50, embed_dim=32)      # layer 0: shared source (embedded tokens)
model.add_transformer_block(32, num_heads=4)          # layer 1..N: "encoder" branch (KV source)
enc_index = len(model.layers) - 1
model.layers.append({"type": "goto", "stored_index": 0})   # fork back to the shared embedded tokens
model.add_dense(32, 32, activation="tanh")            # "decoder" query stream
model.add_cross_attention(kv_source_index=enc_index, embed_dim=32, num_heads=4)
model.add_layernorm(32)
model.add_dense(32, 10, activation="softmax")

X = np.random.randint(0, 50, size=(4, 6))              # (batch, seq_len) token ids
out = model.Forward(X, training=True)                  # (4, 6, 10)
```

- **Use it for:** encoder-decoder architectures (machine translation-style
  tasks, or anything where a "decoder" needs to look up information from a
  separately-processed "encoder" sequence).
- **Pros:** correct gradient split verified via finite difference; works
  whether the KV-source branch or the query branch was built first, and
  even if cross-attention is the network's very last layer.
- **Cons:** requires understanding the `"goto"` layer-index trick above to
  build a genuine two-branch architecture; no causal masking option.

`dropout` works the same way as `add_multihead_attention`'s (post-softmax,
`training=True` only), and `num_kv_heads` selects MHA/MQA/GQA exactly as it
does there. There is deliberately no `window_size` here: a window is a
constraint on the *distance between a query and a key*, which is only
meaningful when both live in the same sequence — across two independently
indexed sequences it would have no defined meaning.

### Positional encoding

```python
add_positional_encoding(max_seq_len, embed_dim=None, learnable=True, base=None)
```
Adds information about each token's position in the sequence — attention
has no built-in sense of order, so without this (or `positional_scheme` on
the attention layer itself) it would treat the sequence as an unordered
set. `learnable=True` (default) adds it as a trained embedding table.
`learnable=False` precomputes fixed sinusoidal encodings (`base` defaults
to `10000.0`).

```python
model.add_embedding(vocab_size, embed_dim=64)
model.add_positional_encoding(max_seq_len=128, embed_dim=64, learnable=False)
model.add_transformer_block(64, num_heads=4)
```

- **Use it for:** classic (non-RoPE/ALiBi) transformer architectures — pair
  right after the embedding layer.
- **Pros:** both classic Transformer variants (fixed sinusoidal, learned)
  available with one flag.
- **Cons:** only absolute position, added once at the input — for
  relative-position schemes, use `add_multihead_attention(positional_scheme=
  "rope"|"alibi")` instead of this layer.

### Vision blocks

```python
add_se_block(channels=None, reduction=16, activation="relu")
add_cbam_channel(channels=None, reduction=16, activation="relu")
add_cbam_block(channels=None, reduction=16, kernel_size=7, activation="relu")
add_convnext_block(channels=None, mlp_ratio=4.0, kernel_size=7, activation="gelu")
add_efficientnet_block(channels=None, expand_ratio=4.0, kernel_size=3,
                       reduction=16, activation="swish")
add_spp(pool_sizes=(1, 2, 4))
add_global_maxpool2d()
add_channel_pool()
```
Named arrangements of primitives the library already has, so a caller
writes one line instead of eight. `channels` is inferred from the previous
conv/pool layer when left `None`.

| Block | What it does |
|---|---|
| **SE** (Hu et al. 2018) | Squeeze `(B, C, H, W)` to `(B, C)` by global average pooling, run a bottleneck MLP (`C → C/reduction → C`, sigmoid), and rescale each channel by the result. Shape unchanged |
| **CBAM channel** (Woo et al. 2018) | SE's idea with average **and** max pooling, summed before the sigmoid, through **one shared MLP** |
| **CBAM block** | Channel attention, then spatial attention: pool across channels to `(B, 2, H, W)`, convolve to a one-channel map, gate every position by it |
| **ConvNeXt** (Liu et al. 2022) | Large-kernel conv → LayerNorm → pointwise expand → activation → pointwise project, in a residual. A conv stack arranged like a transformer block |
| **EfficientNet MBConv** (Tan & Le 2019) | Inverted residual: expand pointwise, convolve, squeeze-and-excite, project back. The skip connects the two *narrow* ends |
| **SPP** | Max-pool to each grid size and concatenate, so the output length depends only on `pool_sizes` and channel count — **never on input resolution** |

```python
model.add_conv2d(3, 64, k=3, padding="same", input_size=(32, 32))
model.add_se_block()             # channels inferred
model.add_efficientnet_block()
model.add_spp((1, 2, 4))         # -> fixed 64*21 width, any input size
model.add_dense(None, 10, activation="softmax")
```

- **Pros:** every block is FD-gradient-checked at every parameter, with the
  check additionally asserting no parameter group has an all-zero gradient —
  a check that passes because everything is zero proves nothing. Shape
  preservation, the SE gate being per-channel and in (0, 1), SPP's
  resolution independence, and CBAM's MLP actually being shared are each
  pinned separately.
- **Cons/deviations, stated rather than hidden:**
  - **CBAM channel attention is one fused layer**, not a composition. Its
    MLP is shared between the two pooled paths, and a list of layer dicts
    has no way to tie two `add_dense` layers' weights together.
  - **ConvNeXt's depthwise conv is a full conv here.** `add_conv2d` is
    dense; the block's *arrangement* is what it is about, but the parameter
    count is higher than the paper's -- and so is the cost, measurably.

**Measured cost** (forward+backward, B=8, 16 channels, 32x32, relative to
the bare conv stack they wrap):

| Block | vs baseline |
|---|---|
| `add_spp` | 1.02x |
| `add_se_block` | 1.71x |
| `add_cbam_block` | 2.37x |
| `add_convnext_block` | 26x |
| `add_efficientnet_block` | 63x |

SE and CBAM are nearly free -- a pooling plus a tiny MLP. ConvNeXt and
EfficientNet are not, because they *are* several full-resolution
convolutions each; ConvNeXt additionally pays for the dense-conv
substitution above, where a true depthwise conv would be `C x` cheaper.
Reach for them when you want the architecture, not as a drop-in speedup.

Two primitives exist mainly to serve these blocks but are useful alone:
`add_global_maxpool2d()` (`(B,C,H,W) → (B,C,1,1)`) and `add_channel_pool()`
(`(B,C,H,W) → (B,2,H,W)`, the per-position mean and max *across* channels).

### Multiplicative gating

```python
model.add_residual_start()
...                              # a branch producing a gate
model.add_multiply_end()         # x = saved * gate
```
The counterpart to `add_residual_end()`'s addition. The gate is broadcast up
to the saved tensor's rank by appending trailing singleton axes, so a
channel gate `(B, C)` and a spatial gate `(B, 1, H, W)` both work against a
`(B, C, H, W)` feature map. This is the primitive behind SE, CBAM and
EfficientNet's SE stage.

Closing either kind of block **restores the shape bookkeeping** to what it
was at `add_residual_start()`, since the block's output has the saved
tensor's shape however much the branch changed it in between — without
that, the next auto-inferring layer would size itself from the gate branch.

### Mixture-of-Experts

```python
add_moe(embed_dim=None, num_experts=4, hidden_dim=None, top_k=1,
        activation="gelu", aux_loss_weight=0.01,
        init_method="xavier_uniform")
```
A feed-forward layer that is really `num_experts` small MLPs plus a router.
For each token the router scores every expert and only the **top-k actually
run** — the tokens routed to an expert are gathered, pushed through it, and
scattered back. So parameter count grows with `num_experts` while the
compute per token stays fixed at `k` experts. That is the entire idea: more
capacity without more work.

Drop it in wherever a Transformer block's MLP would go:

```python
model.add_residual_start()
model.add_layernorm(256)
model.add_moe(256, num_experts=8, top_k=2)   # 8x the MLP capacity, 2x the work
model.add_residual_end()
```

**Gates are the raw router softmax probabilities**, not renormalized across
the chosen k. This matters: renormalizing would make every gate exactly 1.0
at `top_k=1` and cut the router's gradient completely, leaving routing
untrainable. Keeping the raw probability is what lets the router learn.

**Load balancing.** Left alone, routers collapse: a few experts win early,
get all the gradient, and the rest never train. `aux_loss_weight` adds the
Switch Transformer auxiliary loss `num_experts · Σₑ fₑ·Pₑ` — `fₑ` is the
fraction of tokens dispatched to expert *e*, `Pₑ` its mean router
probability. It equals `1.0` for a perfectly balanced router and
`num_experts` for a fully collapsed one, so minimizing it spreads the load.
Only `Pₑ` carries gradient (`fₑ` is a hard count).

The auxiliary loss is **always measured** and readable via
`model.moe_aux_loss()` (summed over every MoE layer, from the last
`Forward`). Its *gradient* is folded into `Backward()` automatically, scaled
by `aux_loss_weight`. It is deliberately not added to the number
`ComputeLoss` reports, so your training curve stays a curve of the task
loss — watch the two separately.

- **Use it for:** scaling a model's capacity when compute, not memory, is
  your constraint.
- **Pros:** genuine conditional compute — a test asserts that perturbing an
  *unselected* expert's weights changes nothing; the whole layer is
  FD-checked (including the auxiliary loss, against `task + α·aux`); the
  forward is pinned against a dense mixture oracle that runs every expert on
  every token; the balancing loss is verified both by formula and
  behaviorally (training with it on ends measurably more balanced).
- **Cons/notes:** there is **no expert-capacity cap and no token dropping** —
  a real gather has no fixed-size buffer to overflow, so this is strictly
  better behaved than the capacity-factor scheme, but it also means a badly
  balanced router costs uneven time rather than silently dropping tokens.
  The per-expert loop is Python (over experts, not tokens), so very large
  `num_experts` adds interpreter overhead.

### Transformer block

```python
add_transformer_block(embed_dim=None, num_heads=4, mlp_ratio=4.0, dropout=0.0,
                      activation="swish", causal=False, positional_scheme="absolute",
                      num_kv_heads=None, window_size=None,
                      attention_kernel="softmax", num_features=None,
                      sparse_pattern=None, tiled_block_size=None)
```
A full pre-norm Transformer block in one call: attention (with its own
residual connection) followed by a small MLP (with its own residual
connection) — the standard building block of GPT/BERT/ViT-style models.
This is what both `TextGenerator` and manual GPT/ViT-style models are
built from; call it multiple times in a row to stack layers (depth).

```python
model.add_embedding(vocab_size, embed_dim=128)
model.add_positional_encoding(max_seq_len=256, embed_dim=128, learnable=False)
for _ in range(4):  # 4 transformer layers deep
    model.add_transformer_block(128, num_heads=4, causal=True)
model.add_layernorm(128)
model.add_dense(128, vocab_size, activation="softmax")
```

- **Use it for:** the standard way to build any transformer stack (text,
  ViT, etc.) — this is the layer you actually stack for depth, not
  `add_multihead_attention` directly.
- **Pros:** correct residual wiring, verified via finite-difference
  gradient checks; one call builds 8+ underlying layers correctly wired.
- **Cons:** pre-norm only (no post-norm option); no built-in cross-attention
  variant (build encoder-decoder architectures via `add_cross_attention`
  directly, as shown above).

### KV-cache decoding

```python
from Enilnets import KVCache, cached_forward_step

cache = KVCache()
logits = cached_forward_step(model, prompt_ids, cache)   # prime, (B, S, V)
logits = cached_forward_step(model, next_ids, cache)     # step,  (B, 1, V)
```
Incremental (autoregressive) decoding for any **causal** attention stack:
instead of re-running the whole growing context at every step — O(n²) over
a generation — each call projects only the *new* tokens and attends over
the keys/values cached from all previous ones, which is O(n) overall.

`token_ids` is an integer array `(batch, n_new)` (a bare `(n_new,)` is
treated as batch 1). Multi-token steps are allowed and get the usual causal
mask *among themselves*, so a whole prompt can prime in one call and the
result is bit-for-bit the same as a full `Forward()` over the same tokens
(pinned by tests for all three positional schemes). `cache.position` tracks
the absolute position of the next token; pass `advance_position=False` to
manage it yourself. Create one `KVCache` per generation stream and discard
it to start over.

Supported layer types: `embedding`, `positional_encoding` (learned and
sinusoidal), `layernorm`, `dense`, `dropout` (inference no-op),
`residual_save`/`residual_add`, `moe`, and **causal** `multihead_attention` in all
three positional schemes (`absolute`, `rope`, `alibi`) and any
`num_kv_heads` (MHA/MQA/GQA — the cache stores the *unexpanded* K/V heads,
so GQA's memory saving is real here, not just in the weights) and any
`window_size` (out-of-window entries are **evicted** from the cache, so a
windowed layer decodes in constant memory), and any `attention_kernel` —
linear/Performer layers keep a fixed-size recurrent state in `cache.linear`
instead of a K/V tensor, so they decode in constant memory at *any* length —
and any `sparse_pattern`, where each step scores only the keys inside its
own selected blocks — i.e. everything `add_transformer_block(causal=True)`
builds. Anything else raises a clear
error naming the offending layer.

- **Use it for:** any hand-built GPT-style stack you want to sample from;
  `TextGenerator.generate()` uses exactly this internally.
- **Pros:** batched and multi-token (unlike a typical one-token-at-a-time
  cache); handles RoPE by caching already-rotated keys and ALiBi by biasing
  at absolute positions, so it stays exact; refuses unsupported
  architectures loudly rather than silently returning wrong numbers.
- **Cons:** inference only (no gradients — it bypasses `Forward`'s bookkeeping
  entirely); requires `causal=True` attention, since a non-causal layer's
  output at earlier positions changes when later tokens arrive, which a cache
  fundamentally cannot represent; no support for arbitrary layer types
  (conv, RNN, pooling) — use a full `Forward()` for those.

### Vision Transformer patch embedding

```python
add_vision_transformer_patch_embed(img_size, patch_size, in_channels=None, embed_dim=768)
```
Converts `(B, C, H, W)` images into `(B, num_patches, embed_dim)` patch
tokens — chops the image into `patch_size x patch_size` squares and
linearly projects each one, the standard first step of a Vision
Transformer (ViT). Asserts `img_size % patch_size == 0`.

```python
model.add_vision_transformer_patch_embed(img_size=32, patch_size=4, in_channels=3, embed_dim=128)
# -> (B, 64, 128): 64 patches (32/4)^2, each a 128-dim token
model.add_transformer_block(128, num_heads=4)
model.add_global_avgpool2d()  # or flatten + dense on the token sequence
```

- **Use it for:** the first layer of a Vision Transformer, as an
  alternative to convolutions for image tasks — see
  [Images](#images) above for when to prefer one over the other.
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
All three take `(batch, seq_len, features)` input and process it one
timestep at a time, carrying a hidden state forward — the classic way to
handle sequences of arbitrary length. `return_sequences=True` outputs
every timestep `(batch, seq_len, hidden)`; `False` outputs just the last
timestep `(batch, hidden)` (typical when you only need a single summary of
the whole sequence, e.g. for classification). Backprop is full
backpropagation-through-time (BPTT), verified against finite-difference
gradients.

```python
model.add_embedding(vocab_size, embed_dim=64)
model.add_lstm(64, 128, return_sequences=False)   # summarize the whole sequence
model.add_dense(128, n_classes, activation="softmax")
```

Which one to pick:
- **`add_rnn`**: single tanh gate — simplest, but prone to vanishing
  gradients over long sequences. Rarely the best choice; included mostly
  as a baseline.
- **`add_lstm`**: the classic choice for sequence tasks — gates that
  explicitly control what to remember/forget, handling long-range
  dependencies far better than plain RNN.
- **`add_gru`**: similar to LSTM but with fewer parameters (3 gates
  instead of 4) — often a good default when you want LSTM-like behavior
  more cheaply.

**Stacking for depth:** chain multiple calls, `return_sequences=True` on
all but the last:
```python
model.add_lstm(64, 128, return_sequences=True)
model.add_lstm(128, 128, return_sequences=True)
model.add_lstm(128, 64, return_sequences=False)  # final layer summarizes
```

**Stateful mode** (`stateful=True` on `add_rnn`/`add_lstm`/`add_gru`):
the layer retains its final hidden (and cell) state across `Forward`
calls, so a long stream can be fed in chunks with identical results to
one full pass — call `model.reset_rnn_state()` to start a new stream.
Carried state is treated as a constant for BPTT (standard truncated-BPTT
semantics), never saved into `Save()` files (a loaded model starts
fresh), and a batch-size change mid-stream raises a clear error. Default
`stateful=False` reproduces the old always-from-zero behavior exactly.

- **Pros:** real BPTT (not truncated/approximated), all three variants
  available, auto-shape-inference works the same as every other layer.
- **Cons:** the per-timestep Python loop in `Forward`/`Backward`
  (unavoidable for a from-scratch recurrent implementation) means these
  layers are meaningfully slower than dense/conv for long sequences; for a
  bidirectional variant, see `add_bidirectional_rnn`/`_lstm`/`_gru` below.

### Bidirectional RNN/LSTM/GRU

```python
add_bidirectional_rnn(n_in=None, hidden_dim=128, return_sequences=True, init_method="xavier_uniform")
add_bidirectional_lstm(n_in=None, hidden_dim=128, return_sequences=True, init_method="xavier_uniform")
add_bidirectional_gru(n_in=None, hidden_dim=128, return_sequences=True, init_method="xavier_uniform")
```
Runs one direction over the input as given, one direction over the
time-reversed input, then concatenates both directions' outputs along the
feature axis. Output width is always `hidden_dim * 2`. Use this when you
need context from *both* directions (e.g. tagging each word in a sentence
using the words both before and after it) — **not** for text generation or
anything autoregressive, since a bidirectional model needs to see the
whole sequence up front (it can't generate token-by-token without already
knowing what comes next).

```python
model = NeuralNet(optimizer="adam")
model.add_bidirectional_lstm(n_in=64, hidden_dim=128, return_sequences=True)
model.add_dense(256, 10, activation="softmax")  # 256 = hidden_dim * 2
```

`return_sequences=True` gives `(batch, seq_len, hidden_dim*2)` (each
timestep combines both directions' information *about* that timestep).
`return_sequences=False` gives `(batch, hidden_dim*2)` (the forward
direction's final state concatenated with the backward direction's final
state — a summary of the whole sequence from both ends).

- **Use it for:** sequence labeling/classification tasks where the whole
  sequence is available up front (not text generation).
- **Pros:** gradient composition verified via finite difference for both
  directions, both `return_sequences` modes, all three RNN variants;
  verified functionally on a task solvable only by looking
  backward-in-time — the bidirectional version reaches ~99% accuracy where
  a plain unidirectional RNN stays at chance.
- **Cons:** roughly 2x the compute/memory of a single-direction RNN.

### Residual / skip connections

```python
add_residual_start()
add_residual_end()
```
Generic, nestable skip connections: `add_residual_end()` computes
`x = x + saved_x`, where `saved_x` is whatever `x` was at the matching
`add_residual_start()`. Skip connections let gradients flow directly
through the "+" without having to pass through every intermediate layer,
which is what makes very deep networks trainable at all — this is the
primitive `add_transformer_block` is built from; use it directly for
custom ResNet-style blocks:

```python
model.add_dense(64, 64, activation="linear")
model.add_residual_start()
model.add_dense(64, 64, activation="tanh")
model.add_dense(64, 64, activation="linear")
model.add_residual_end()          # x = x + tanh_block(x)
```

- **Use it for:** any custom deep architecture (dense, conv, or otherwise)
  where you want gradients to skip past a sub-block — the standard fix
  for vanishing gradients in deep networks.
- **Pros:** generic (works around any sequence of layers, not just
  attention/MLP blocks), nestable, gradient routing verified via finite
  difference.
- **Cons:** `add_residual_end()` with no matching `add_residual_start()`
  raises `ValueError` rather than silently no-op'ing — intentional (fails
  loudly on a construction mistake).

### Convenience block builders

```python
add_mlp_block(hidden_dims, in_dim=None, out_dim=None, activation="relu",
              out_activation="linear", init_method="xavier_uniform")
add_conv_block(out_ch, k=3, activation="relu", init_method="he_normal",
              in_ch=None, stride=1, batchnorm=False, pool=None, input_size=None,
              padding="valid")
```
`add_mlp_block([256, 128, 64], out_dim=10)` = a loop of `add_dense` calls,
saving you the boilerplate:
```python
model.add_mlp_block([256, 128, 64], in_dim=784, out_dim=10, out_activation="softmax")
# equivalent to:
#   model.add_dense(784, 256, activation="relu")
#   model.add_dense(256, 128, activation="relu")
#   model.add_dense(128, 64, activation="relu")
#   model.add_dense(64, 10, activation="softmax")
```
`add_conv_block(...)` = `conv2d → [batchnorm] → [pool]` in one call
(`pool` accepts `None`, `"max"`, or `"avg"`; pool size is fixed at 2
regardless of what you pass elsewhere) — this is what the
[Images](#images) quickstart above uses.

### Layer type summary table

Quick reference for every `add_*` method — see each layer's own section
above for the full explanation, examples, and pros/cons.

| Layer | Input shape | Learns parameters? | One-line purpose |
|---|---|---|---|
| `add_dense` | `(B, F)` or `(B, S, F)` | Yes | Fully-connected layer; general-purpose, classification/regression heads |
| `add_sparse` | `(B, F)` | Yes | Dense layer with a fixed random sparsity mask |
| `add_conv2d` | `(B, C, H, W)` | Yes | 2D convolution; images and spatial-grid data |
| `add_conv1d` | `(B, C, L)` | Yes | 1D convolution; audio/time-series |
| `add_maxpool2d`/`add_avgpool2d` | `(B, C, H, W)` | No | Shrink spatial size |
| `add_global_avgpool2d` | `(B, C, H, W)` | No | Collapse spatial size to 1x1 |
| `add_upsample2d` | `(B, C, H, W)` | No | Grow spatial size back (decoders) |
| `add_flatten` | `(B, C, H, W)` | No | Flatten to `(B, C*H*W)` for dense layers |
| `add_batchnorm` | `(B, F)` or `(B, C, H, W)` | Yes (scale/shift) | Normalize over the batch; stabilizes deep conv training |
| `add_layernorm` | `(B, F)`, `(B, S, F)`, or `(B, C, H, W)` | Yes (scale/shift) | Normalize over features; the transformer/RNN standard |
| `add_dropout` | any | No | Randomly zero activations; reduces overfitting |
| `add_embedding` | integer IDs `(B,)`/`(B, S)` | Yes | Token/category ID -> dense vector |
| `add_multihead_attention` | `(B, S, E)` | Yes | Self-attention; long-range sequence dependencies |
| `add_cross_attention` | `(B, S, E)` + a KV source | Yes | Encoder-decoder attention |
| `add_positional_encoding` | `(B, S, E)` | Maybe (if learnable) | Inject sequence-position information |
| `add_transformer_block` | `(B, S, E)` | Yes | Full attention+MLP block; the transformer building block |
| `add_vision_transformer_patch_embed` | `(B, C, H, W)` | Yes | Image -> patch token sequence |
| `add_rnn`/`add_lstm`/`add_gru` | `(B, S, F)` | Yes | Recurrent sequence processing |
| `add_bidirectional_rnn`/`_lstm`/`_gru` | `(B, S, F)` | Yes | Both-direction recurrent sequence processing |
| `add_residual_start`/`add_residual_end` | any | No | Skip connection around the wrapped layers |
| `add_mlp_block`/`add_conv_block` | varies | Yes | Convenience multi-layer builders |

## Activations

```python
activate(name, x, alpha=None, sigmoid_clip=None)
derivative(name, x, alpha=None, sigmoid_clip=None, cached_output=None)
```
(Used internally by every layer's `activation=` argument; not usually
called directly.) An activation function is what makes a layer non-linear
— without one, stacking dense layers would collapse into a single linear
transformation no matter how many you stack.

| Name | Shape | Typical use |
|---|---|---|
| `relu` | `max(0, x)` | Default for hidden layers in dense/conv nets — cheap, works well in practice |
| `leakyrelu` | like relu but a small negative slope (`alpha`, default `0.01`) instead of a hard zero | Avoids "dead" neurons that relu can produce |
| `elu` | smooth negative-side curve (`alpha`, default `1.0`) | Alternative to leakyrelu with a smoother gradient |
| `selu` | self-normalizing variant of elu | Deep dense nets without explicit normalization layers |
| `gelu` | smooth, S-shaped (tanh approximation) | Standard choice in transformer MLP blocks |
| `swish` | `x * sigmoid(x)` | Common in modern conv/generative-model architectures |
| `mish` | smooth, similar spirit to swish | Alternative smooth activation |
| `sigmoid` | squashes to `(0, 1)` | Binary classification output, gates in LSTM/GRU |
| `tanh` | squashes to `(-1, 1)` | RNN/GAN-generator outputs, zero-centered hidden states |
| `softmax` | outputs sum to 1 across the last axis | Multi-class classification output |
| `softplus` | smooth approximation of relu | Rarely used directly; building block for other functions |
| `linear` | identity, no transformation | Output layers for unbounded regression, or where you want a purely linear layer |

- `alpha` overrides `LEAKYRELU_ALPHA`/`ELU_ALPHA` as applicable;
  `sigmoid_clip` overrides `SIGMOID_CLIP` (500.0), the overflow-safe clip
  bound used by `sigmoid`/`softplus`'s `exp()`.
- Pass these per-layer via `activation_params={"alpha": ..., "sigmoid_clip": ...}`.
- An unrecognized activation `name` raises `ValueError` in both `activate`
  and `derivative` — a typo like `"realu"` fails immediately instead of
  silently behaving as a linear layer.

```python
from Enilnets.activations import activate, derivative
import numpy as np
x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
print(activate("relu", x))                          # [0.  0.  0.  0.5 2. ]
print(activate("leakyrelu", x, alpha=0.1))           # [-0.2 -0.05 0.  0.5 2. ]
print(derivative("relu", x))                         # [0. 0. 0. 1. 1.]
```

## Weight initialization

```python
init_weights(n_in, n_out, method="xavier_uniform", std=None)          # dense/attention/RNN weights
init_conv_weights(in_ch, out_ch, k, method="he_normal", std=None)      # conv2d weights
init_embedding_weights(vocab_size, embed_dim, method="normal", std=None)
```
How a layer's weights are randomly set *before* training starts. This
matters more than it might seem: badly-scaled initial weights can make
gradients explode or vanish before training even gets going.

`init_method` strings:
- `init_weights` / `init_conv_weights`: `xavier_uniform`, `xavier_normal`,
  `he_uniform`, `he_normal`, `normal`, `orthogonal`, `zeros`, `ones`.
- `init_embedding_weights`: `normal`, `xavier_uniform`, `xavier_normal`,
  `zeros` (no He/orthogonal/ones options).
- Unknown method string → `ValueError` for both.
- `std` (only used by `"normal"`) defaults to `constants.NORMAL_INIT_STD = 0.1`.

**Rule of thumb:** Xavier for tanh/sigmoid-family activations, He for
ReLU-family — the per-layer `init_method=` default is already sensible for
that layer's common use (`add_dense` defaults to `xavier_uniform`,
`add_conv2d` to `he_normal`), so you usually don't need to touch this at
all; override it only if you're using an unusual activation.

```python
from Enilnets.weight_init import init_weights
W, b = init_weights(n_in=784, n_out=256, method="he_normal")
print(W.shape, b.shape)  # (256, 784) (256,)
```

## Losses

```python
ComputeLoss(output, target, function="mse", reduction="mean", **kwargs)
```
and the matching gradient inside `Backward(targets, loss_function=..., **kwargs)`.

| `function` | kwargs | Use it for |
|---|---|---|
| `mse` | — | Regression (continuous, unbounded output) |
| `mae` | — | Regression, more robust to outliers than MSE |
| `huber` | `delta=1.0` | Regression, a smooth blend of MSE (small errors) and MAE (large errors) |
| `smooth_l1` | `beta=1.0` | Similar to huber; common in detection/localization tasks |
| `binary_cross_entropy` | `eps=1e-12` | Binary classification (`sigmoid` output) |
| `cross_entropy` / `categorical_cross_entropy` | `eps` | Multi-class classification (`softmax` output); target is one-hot, same shape as output — `(B, V)` or `(B, S, V)` for sequence data |
| `sparse_cross_entropy` | `eps` | Same as `cross_entropy` but `target` is integer class indices `(B,)`/`(B, S)` instead of one-hot — saves memory for large vocabularies (e.g. language modeling) |
| `focal` | `alpha=0.25`, `gamma=2.0`, `eps` | Classification with severe class imbalance (down-weights easy examples) |
| `hinge` | — | SVM-style margin classification |
| `bce_logits` | — | Binary classification directly from an unbounded logit (numerically-stable form; use with a `linear` output layer, not `sigmoid`) |
| `wasserstein` | — | WGAN-style critic training |
| `cosine_similarity` | `eps_div=1e-8` | Embedding/similarity learning |
| `triplet` | `margin=1.0`, `negative` (**required**) | Metric learning (anchor/positive/negative triplets) |
| `ntxent` | `temperature=0.5`, `eps_div` | Contrastive learning (SimCLR/CLIP-style) |
| `kl_divergence` | `mu`, `logvar` (**both required**) | VAE regularization term only — not a general-purpose output/target loss (see `generative/vae.py`) |

Unknown `function` → `ValueError`. `reduction="mean"`/`"sum"` return a plain
Python `float`; anything else returns the raw elementwise array.

```python
from Enilnets import NeuralNet
model = NeuralNet()
out = np.array([[0.7, 0.2, 0.1]])
target = np.array([[1, 0, 0]])
print(model.ComputeLoss(out, target, function="cross_entropy"))  # a small positive float
```

**Reduction convention**, load-bearing if you write your own finite-
difference reference: under `reduction="mean"`, elementwise losses (`mse`,
`mae`, `huber`, `bce`, `focal`, `hinge`, `bce_logits`, `wasserstein`, ...)
average over **every element** (`batch_size * n_features`). Losses that
already reduce over the feature/class axis in their own formula
(`cross_entropy`, `sparse_cross_entropy`, `cosine_similarity`, `triplet`,
`ntxent`) divide by every *other* (non-feature) dimension instead —
`batch_size` for plain `(B, V)` output, `batch_size * seq_len` for
sequence-shaped `(B, S, V)` output.

Where a loss+activation pair has a canonical simplified gradient (softmax +
cross-entropy, sigmoid + BCE, linear + `bce_logits`/`wasserstein`), Enilnets
uses that closed form directly instead of chaining a separate activation
derivative — slightly faster and more numerically stable than the generic
chain rule path.

## Optimizers

Set via `NeuralNet(optimizer=..., learning_rate=..., l2_lambda=..., momentum=...)`.
The optimizer is *how* the network's weights get nudged each training step
once you have a gradient — different optimizers trade off simplicity,
speed of convergence, and how much per-parameter adaptation they do.

| Optimizer | Extra kwargs | Use it when | Pros | Cons |
|---|---|---|---|---|
| `"sgd"` | `momentum` | You want the simplest, most predictable baseline, or are matching a classic paper's setup | Cheapest per step; often generalizes best with enough tuning | Needs careful learning-rate tuning; slow convergence without momentum tuned well |
| `"rmsprop"` | `rmsprop_decay`, `rmsprop_epsilon` | Non-stationary objectives (RL, GANs) | Per-parameter adaptive learning rate | No bias correction (unlike Adam) |
| `"adagrad"` | `adagrad_epsilon` | Sparse gradients (e.g. embeddings with a huge vocabulary) | Rarely-updated features get comparatively larger steps | Effective learning rate only shrinks over time — can stall on long runs |
| `"adam"` | `adam_beta1`, `adam_beta2`, `adam_epsilon` | The default choice for almost everything | Adaptive per-parameter rates + momentum + bias correction | L2 weight decay is coupled into the gradient (AdamW fixes this) |
| `"adamw"` | same as Adam | You're using `l2_lambda` and want the modern, decoupled-decay behavior | Decoupled weight decay, the modern recommendation over Adam+L2 | Decay still happens every step regardless of gradient |
| `"adadelta"` | `adadelta_rho`, `adadelta_epsilon` | You don't want to tune a learning rate at all | Step size is a ratio of RMS(past updates) to RMS(past gradients), so it's unit-consistent and needs no LR — leave `learning_rate=1.0` | Can be slow late in training as the accumulators equilibrate |
| `"adamax"` | same as Adam | Gradients are occasionally huge and you want the denominator to react to the spike rather than average it away | Uses the L-∞ norm of past gradients, which needs no bias correction of its own | Less well studied than Adam; the max decays only geometrically after a spike |
| `"nadam"` | same as Adam | You'd use Adam but want momentum's lookahead | Nesterov-accelerated Adam — applies the momentum step to the *current* gradient | Marginal gains over Adam on many problems |
| `"radam"` | same as Adam | Training is unstable in the first few hundred steps and you'd otherwise add an LR warmup | Rectified Adam: switches the adaptive denominator *off* until the second-moment estimate is statistically trustworthy, so no warmup schedule is needed | Deliberately slower at the very start; that's the mechanism, not a defect |
| `"lion"` | `lion_beta1`, `lion_beta2` | Optimizer memory matters, or you want a very cheap step | Update is the *sign* of an interpolated momentum, so every weight moves by exactly ±`lr`; keeps **one** accumulator instead of Adam's two | Needs a noticeably smaller `learning_rate` than Adam (roughly 3-10×); the fixed step size is unusual to tune |
| `"lamb"` | same as Adam | Very large batches, where a single global LR suits no layer | Rescales each parameter tensor's update by the layer-wise trust ratio ‖w‖/‖r‖, so every tensor moves proportionally to its own norm | The trust ratio is clamped at 10 for stability; on small batches it mostly just reproduces Adam |
| `"adafactor"` | `adafactor_eps1`, `adafactor_clip_threshold`, `adafactor_decay_rate` | The model is big enough that optimizer state is a real memory cost | Stores the second moment **factored** as a row vector plus a column vector — O(R+C) instead of O(R×C) — and clips each update's RMS | The factorization is an approximation; 1-D parameters can't be factored and keep a full accumulator |

`optimizer_type` is validated at `NeuralNet(...)` construction time — any
string other than the twelve above (including typos) raises `ValueError`
immediately, and the message lists the valid names.

**Weight decay is decoupled** (applied to the weights, not folded into the
gradient) for `"adamw"`, `"lion"` and `"lamb"`, and coupled for everything
else — which is the standard behavior for each.

**Optimizer state is allocated per rule**, not one-size-fits-all, so the
memory characteristics are real rather than nominal. Measured on a
512→512→512→10 MLP (~530k parameters), floats of optimizer state after one
step:

| | state | vs Adam |
|---|---|---|
| `sgd`, `rmsprop`, `adagrad`, `lion` | 530,442 | 0.5× |
| `adam`, `adamw`, `adamax`, `nadam`, `radam`, `lamb`, `adadelta` | 1,060,884 | 1.0× |
| `adafactor` | **3,604** | **0.003×** |

```python
model = NeuralNet(optimizer="adamw", learning_rate=0.001, l2_lambda=0.01)
```

Weight-decay eligibility is hardcoded per layer type: `dense`/`sparse`/
`conv2d`/`embedding` decay `weights` only (never bias); `multihead_attention`
decays `Wq`/`Wk`/`Wv`/`Wo` (never the biases); `rnn`/`lstm`/`gru` decay
`Wx`/`Wh` (never `b`/`bx`/`bh`); `moe` decays the expert stacks `W1`/`W2` and
the router `Wr` (never `b1`/`b2`/`br`); batchnorm/layernorm `gamma`/`beta`
are never decayed.

Gradient clipping is automatic whenever `grad_clip_norm > 0`:
```python
model = NeuralNet(optimizer="adam", grad_clip_norm=1.0)
model.TrainBatch(x, y)  # Backward() -> clip_gradients(1.0) -> update(), automatically
```

Lower-level primitives, if you want to inspect/modify gradients before
they're applied, or build your own accumulation/multi-optimizer scheme:
```python
grads = model.compute_gradients()   # -> list aligned with self.layers, None for param-free layers
model.apply_gradients(grads)        # applies the configured optimizer formula, mutates weights + opt_state
model.update()                      # = apply_gradients(compute_gradients())
```

## Training

### `TrainBatch` / `Train`

```python
TrainBatch(xs, ys, loss_function=None, accumulation_steps=1, **loss_kwargs)
Train(X_train, Y_train, epochs=10, batch_size=32, X_val=None, Y_val=None,
      loss_function=None, verbose=True, scheduler=None, early_stopping=None,
      accumulation_steps=1, callbacks=None, **loss_kwargs)
```
`TrainBatch` runs one batch end-to-end: forward → loss → backward →
optional clip → optimizer step, returning `(loss, out)`. If
`loss_function=None`, it auto-picks `"cross_entropy"` when the last
layer's activation is `"softmax"`, else `"mse"`.

`Train` wraps `TrainBatch` in a full epoch/minibatch loop — this is what
the [Quickstart](#quickstart) uses — returning a `history` dict with
`"loss"`, `"accuracy"`, `"lr"` (always populated) and `"val_loss"`/
`"val_accuracy"` (empty lists if no `X_val`/`Y_val` given).

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
Use this if training loss occasionally spikes to `nan`/huge values (a sign
of exploding gradients, common in RNNs/transformers).

### Gradient accumulation

Simulate a larger batch size without the memory cost — useful when your
GPU-less setup can't hold a large batch in memory but the optimizer
benefits from more stable, larger-batch gradient estimates:
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
approximation for adaptive optimizers.

### Mixed precision

```python
model = NeuralNet(optimizer="adam", use_mixed_precision=True)
```
Forces the dense/conv2d matmuls to run in float32 specifically,
regardless of whatever the active default precision is (see
[Precision: float32 vs float64](#precision-float32-vs-float64)) — a real
BLAS speedup on the hot path when running in float64 mode (master
weights/gradients stay float64), and a no-op if float32 is already the
default. Expect outputs close to, not bit-identical to, the float64 path
when this actually changes the compute dtype. Turn this on if you're
running in float64 mode and training speed matters more than bit-exact
reproducibility.

### Learning rate schedules

```python
LRScheduler(initial_lr, mode="step", **kwargs)
scheduler.step(epoch)  # -> new learning rate for this epoch
```
A learning rate schedule changes the step size over the course of
training — typically starting higher (fast early progress) and decaying
(fine-tuning near the end).

| `mode` | kwargs | Behavior |
|---|---|---|
| `"step"` | `drop=0.5`, `epochs_drop=10` | Multiply LR by `drop` every `epochs_drop` epochs |
| `"exponential"` | `decay=0.95` | Multiply LR by `decay` every epoch |
| `"cosine"` | `max_epochs=100` | Smooth cosine decay to 0 over `max_epochs` |
| `"warmup_cosine"` | `max_epochs=100`, `warmup_epochs=5` | Linear ramp-up, then cosine decay — common for transformers |
| `"polynomial"` | `max_epochs=100`, `power=1.0`, `end_lr=0.0` | Decay from `initial_lr` to `end_lr` as `(1 - e/E)**power`. `power=1` is a straight line; higher powers stay high longer, then fall off |
| `"cyclic"` | `base_lr`, `max_lr`, `step_size=10`, `policy="triangular"`, `gamma=0.999` | Cyclical LR (Smith 2017): sweep `base_lr ↔ max_lr` every `2*step_size` epochs. `"triangular2"` halves the amplitude each cycle; `"exp_range"` scales it by `gamma**epoch` |
| `"one_cycle"` | `max_lr`, `max_epochs=100`, `pct_start=0.3`, `div_factor=25`, `final_div_factor=1e4` | The 1cycle policy: cosine-anneal up to `max_lr` over the first `pct_start` of the run, then all the way down to `max_lr/final_div_factor`. One cycle, no repeats |
| `"cosine_warm_restarts"` | `T_0=10`, `T_mult=1`, `eta_min=0.0` | SGDR: cosine-anneal over `T_0` epochs, jump back to full LR, and lengthen each successive cycle by `T_mult` |
| `"lambda"` | `lr_lambda=fn` | Whatever you want: `fn(epoch)` returns a multiplier on `initial_lr` |
| `"swa"` | `swa_start=0`, `swa_lr=0.05`, `anneal_epochs=5` | Stochastic Weight Averaging's schedule: train normally until `swa_start`, linearly anneal to `swa_lr`, then **hold** it — see [Weight averaging](#weight-averaging-ema-and-swa) |
| `"sequential"` | `schedulers=[...]`, `milestones=[...]` | Chain schedules back to back. `milestones` are the epochs where each next schedule takes over; there must be exactly one fewer milestone than scheduler |
| `"plateau"` | `factor=0.5`, `patience=10`, `min_delta=0.0`, `metric_mode="min"` | Real `ReduceLROnPlateau`: cuts the LR by `factor` once `patience` epochs pass with no improvement in the monitored metric. Call `scheduler.step(epoch, metric=...)`; `Train(..., scheduler=...)` does this automatically. |

```python
from Enilnets import LRScheduler
scheduler = LRScheduler(initial_lr=0.001, mode="cosine", max_epochs=50)
history = model.Train(X_train, Y_train, epochs=50, batch_size=32, scheduler=scheduler)
print(history["lr"])  # the actual LR used each epoch
```

`"sequential"` restarts the epoch counter for each schedule it hands over
to, so a chained cosine anneals across its own span instead of resuming
somewhere mid-curve — which is almost always what you meant:

```python
warmup = LRScheduler(1e-3, mode="lambda", lr_lambda=lambda e: (e + 1) / 5)
anneal = LRScheduler(1e-3, mode="cosine", max_epochs=45)
scheduler = LRScheduler(1e-3, mode="sequential",
                        schedulers=[warmup, anneal], milestones=[5])
```

Every schedule is clamped past its horizon rather than allowed to run
negative, and none ever returns a value outside its own floor/peak — both
pinned by tests across 60 epochs of every mode.

### Finding a learning rate

```python
from Enilnets import find_learning_rate
report = find_learning_rate(model, X_train, Y_train,
                            start_lr=1e-7, end_lr=1.0, num_iter=100)
print(report["suggested_lr"])
```
The LR range test (Smith 2015): ramp the learning rate exponentially across
a hundred-odd batches, record the loss, and read the usable range off the
curve instead of guessing. Returns `{"lrs", "losses", "raw_losses",
"suggested_lr"}` — `losses` is EMA-smoothed and bias-corrected, and
`suggested_lr` is the rate at the **steepest downward slope**, which is the
conventional reading (the loss *minimum* already sits in the unstable
region, so it is too high to train at).

It stops early once the smoothed loss exceeds `diverge_factor`× its best,
since past that point the curve carries no information.

**This is a probe, not training.** Weights, optimizer state, step counter,
learning rate and any in-progress gradient accumulation are all snapshotted
and restored before it returns — pinned by a test that trains first, so
there is real optimizer state to disturb, then asserts every array is
unchanged afterwards.

### Weight averaging: EMA and SWA

Both keep a second set of weights alongside the ones being trained, on the
observation that an average of points along the trajectory generalizes
better than the final point. They differ in how they weight the history.

```python
from Enilnets import EMA

ema = EMA(model, decay=0.999)
for epoch in range(epochs):
    for xb, yb in batches:
        model.TrainBatch(xb, yb)
        ema.update()                 # after every optimizer step

with ema:                            # swap the averaged weights in
    val_loss = model.ComputeLoss(model.Forward(X_val, training=False), Y_val)
# live weights are back, even if the block raised
```
`EMA` decays the history geometrically. `warmup=True` (the default) ramps
the effective decay as `min(decay, (1+n)/(10+n))`, so the average is not
anchored to the random initialization for the first few hundred steps.
`apply()`/`restore()` are the explicit form of the context manager, and
`copy_to(other_model)` writes the average elsewhere.

```python
from Enilnets import SWA

swa = SWA(model, swa_start=75, swa_lr=0.05, anneal_epochs=5)
history = model.Train(X, Y, epochs=100, scheduler=swa.scheduler(initial_lr=0.1),
                      callbacks=[...])
# ...calling swa.update() once per epoch from swa_start onwards, then:
swa.finalize()          # install the average
swa.update_bn(X_train)  # only needed if the model has batchnorm layers
```
`SWA` averages epoch-end snapshots with **equal** weight — an unweighted
running mean — which is why it wants a flat, high learning rate: the
snapshots have to be meaningfully different points around the basin.
`swa.scheduler(initial_lr)` produces exactly that schedule (`mode="swa"`):
train normally until `swa_start`, linearly anneal to `swa_lr` over
`anneal_epochs`, then hold it there.

- **`update_bn` is not optional if you use batchnorm.** The stored running
  mean/variance were accumulated under the individual snapshots' weights and
  do not describe the averaged model's activations at all. `update_bn`
  resets them and streams the data through once with `momentum = 1/(i+1)`,
  which makes the running update an exact cumulative average. It is a no-op
  on models without batchnorm.
- Both expose `state_dict()`/`load_state_dict()`, so the averaged weights
  round-trip through `model.Save(..., extra_state={"ema": ema.state_dict()})`.

### Early stopping

```python
EarlyStopping(patience=5, min_delta=0.0, mode="min")
early_stopping.step(metric)  # -> bool, True once training should stop
```
Stops training once a monitored metric hasn't improved for `patience`
epochs in a row — the standard defense against overfitting from training
too long. `mode="min"` requires `metric < best - min_delta` to count as
improvement; `"max"` requires `metric > best + min_delta`.

```python
from Enilnets import EarlyStopping
stopper = EarlyStopping(patience=5, mode="min")
model.Train(X_train, Y_train, epochs=200, batch_size=32,
           X_val=X_val, Y_val=Y_val, early_stopping=stopper)
# training stops early if val_loss hasn't improved for 5 straight epochs
```

### Callbacks

`Train(..., callbacks=[...])` (both `NeuralNet.Train` and
`TextGenerator.Train`) takes a plain list of duck-typed callback objects —
no shared base class, implement whichever hooks you need:

```python
class MyCallback:
    def on_epoch_end(self, epoch, logs, model=None):
        print(f"epoch {epoch}: {logs}")

    def on_train_end(self, history):
        print("training finished")

model.Train(X_train, Y_train, epochs=10, batch_size=32, callbacks=[MyCallback()])
```
- `NeuralNet.Train`: `on_epoch_end(epoch, logs, model=self)` fires once per
  epoch; `logs` has `"loss"`/`"accuracy"`/`"lr"` and, if validation data was
  given, `"val_loss"`/`"val_accuracy"`. `on_train_end(history)` fires once
  at the end.
- `TextGenerator.Train`: same convention (`model=self` is the
  `TextGenerator`), plus `on_batch_end(epoch, batch_idx, loss, model=self)`
  after every minibatch.
- Missing hook methods are simply skipped (no error) — implement only the
  ones you need.

For the two most common needs, use the ready-made callbacks below instead
of writing your own:

#### ModelCheckpoint

```python
ModelCheckpoint(monitor="val_loss", mode="min", min_delta=0.0)
checkpoint.restore(model)   # load the best snapshot back after training
```
```python
ckpt = ModelCheckpoint(monitor="val_loss", mode="min")
model.Train(X_train, Y_train, epochs=50, batch_size=32,
           X_val=X_val, Y_val=Y_val, callbacks=[ckpt])
ckpt.restore(model)  # model now has the best-val_loss epoch's weights
```
Snapshots `model.get_weights()` whenever `monitor` improves. Silently does
nothing on an epoch where `monitor` isn't present in `logs` (e.g.
requesting `"val_loss"` without passing `X_val`/`Y_val`).

#### CSVLogger / JSONLogger

```python
CSVLogger(path)    # appends one row per epoch
JSONLogger(path)   # appends one JSON object per line (JSON-lines, not a single array)
```
```python
model.Train(X_train, Y_train, epochs=50, batch_size=32,
           callbacks=[CSVLogger("history.csv"), JSONLogger("history.jsonl")])
```
Both write incrementally, one epoch at a time, so an interrupted training
run still leaves a valid, parseable partial log.

### Accuracy / precision / recall / F1

```python
model.compute_accuracy(predictions, targets)
```
Dispatches on `predictions.shape[-1]`: `>1` → multi-class `argmax`
comparison; `==1` → binary `>0.5` threshold.

```python
preds = model.Forward(X_val, training=False)
print(model.compute_accuracy(preds, y_val))
```

`compute_precision_recall_f1(predictions, targets)` (binary-only, same
shape-based dispatch) is **not** bound onto `NeuralNet` — import it
explicitly:
```python
from Enilnets.train import compute_precision_recall_f1
print(compute_precision_recall_f1(preds, y_val))  # {"precision":..., "recall":..., "f1":...}
```
For the general multi-class case, use
[`classification_report`](#evaluation-utilities) instead.

## Text generation (`TextGenerator`)

```python
TextGenerator(tokenizer, embed_dim=64, num_heads=4, num_layers=2,
             mlp_ratio=4.0, dropout=0.0, activation="gelu",
             max_seq_len=128, learning_rate=3e-4, optimizer="adam", l2_lambda=0.0)
```
A ready-made GPT-style causal transformer for character- or word-level
text generation — you don't need to hand-build the transformer stack
yourself (though you can, using the layer types above; this is exactly
what `TextGenerator` builds internally: `embedding → positional_encoding →
num_layers × transformer_block(causal=True) → layernorm → dense(softmax)`).
Requires an already-`.fit()`-ted `Tokenizer`.

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
`generate(...)`'s sampling controls, in order of how much randomness they
add: `greedy=True` (always pick the single most likely next token — fully
deterministic, often repetitive); `temperature` (>1.0 = more random, <1.0
= more confident/conservative); `top_p` (nucleus sampling — only consider
the smallest set of tokens whose probabilities sum to `p`); `top_k` (only
consider the `k` most likely tokens). `generate_beam(...)` explores
multiple candidate continuations at once and keeps the best-scoring one —
slower, but often higher-quality than sampling for short, precise outputs.

- `generate(..., use_cache=True)` (default) decodes with a KV-cache: only
  the new token's query is computed each step, past keys/values cached and
  reused — much faster than recomputing the whole sequence at every step.
  It is the general `Enilnets.cached_forward_step` mechanism (see
  [KV-cache decoding](#kv-cache-decoding)), so the whole prompt primes in a
  single batched step. **Limitation:** the supported layer types are the
  ones listed there — if you've hand-modified `gen.network` with something
  else, pass `use_cache=False`.
- `generate_beam(..., top_k=None, use_cache=True)` restricts how many
  tokens each beam may expand to per step (`top_k`, default `beam_width`)
  and decodes incrementally, reordering the KV cache by parent beam after
  each prune. `use_cache` changes nothing about the result — a test asserts
  the cached and uncached paths return identical text across beam widths,
  top-k values and length penalties. Measured on 12 new tokens: 1.6x faster
  at `beam_width=1`, 2.7x at 4, 3.2x at 8 -- the gap widens with beam width,
  since every beam was previously re-running the full context.
- `generate_batch(prompts, ...)` generates for several prompts at once.
  Prompts are grouped by **token length** rather than padded to a common
  width: `nn/` attention has no padding mask, so pad tokens would be
  attended to, and left-padding would additionally shift every real token's
  absolute position. Grouping keeps the batching benefit while staying
  *exactly* equal to generating each prompt on its own, which is pinned by
  a test. Measured 4.3x faster than a Python loop over 8 same-length
  prompts.
- `perplexity(text)` (a standard language-model quality metric — lower is
  better, roughly "how surprised was the model by this text") requires at
  least 2 tokens after tokenization.

- **Pros:** ready-made GPT-style transformer — no manual layer wiring
  needed; greedy/temperature/top-p/top-k sampling and exact beam search all
  built in; KV-cache makes `generate()` linear instead of quadratic in
  output length; `perplexity()` gives a standard quantitative quality check
  without a separate eval script.
- **Cons:** `use_cache=True` (the default) only understands the exact
  architecture this class builds — hand-modifying `gen.network` requires
  `use_cache=False`; there's no dedicated whole-model save/load — persist
  training progress via `gen.network.get_weights()`/`set_weights()`
  (alongside `tokenizer.save()`/`load()` for the vocabulary) rather than a
  single call.

```python
Tokenizer(vocab_size=256, level="char", oov_token="<OOV>", pad_token="<PAD>",
         start_token="<START>", end_token="<END>")
tokenizer.fit(texts)                                          # build the vocabulary, chainable
tokenizer.encode(text, max_length=None, add_special_tokens=True)  # -> ndarray of int32
tokenizer.decode(indices, skip_special=True)                  # -> str
tokenizer.save(path) / tokenizer.load(path)                   # JSON round-trip
```
`level="char"` builds a vocabulary from every character actually seen
(unbounded, ignores `vocab_size`); `level="word"` truncates to the
`vocab_size - 4` most common words. Unknown tokens at encode time map to
`oov_token`.

### Subword tokenizers (BPE)

```python
from Enilnets import BPETokenizer

tok = BPETokenizer(vocab_size=8000, level="word").fit(corpus)
ids = tok.encode("the quick brown fox")
text = tok.decode(ids)                      # exact round trip
tok.save("bpe.json"); BPETokenizer().load("bpe.json")
```
Byte Pair Encoding trained from scratch, with the same
`fit`/`encode`/`decode`/`save`/`load` interface as `Tokenizer` — so it
drops straight into `TextGenerator`.

A character tokenizer has a tiny vocabulary but long sequences; a word
tokenizer has short sequences but a huge vocabulary and no way to spell an
unseen word. BPE learns the tradeoff, merging the most frequent adjacent
pair repeatedly until it hits the vocabulary budget.

- `level="word"` (default) splits on whitespace first, so a merge never
  spans a word boundary. Unseen characters become `<OOV>`.
- `level="byte"` works on raw UTF-8 bytes and seeds the alphabet with **all
  256 byte values**, so there is genuinely no out-of-vocabulary case — any
  input at all encodes and round-trips, including scripts absent from the
  training corpus. Its `vocab_size` floor is therefore 260 (256 bytes plus
  the four special tokens), which is enforced.

Encoding always applies the **earliest-learned** applicable merge, so one
merge table yields exactly one segmentation; whitespace is carried in a
`▁` marker rather than discarded, which is what makes decoding exact.

## Generative models

All of these live under `Enilnets.generative` (and are re-exported from the
top-level `Enilnets` package). Each follows the same basic pattern: build
with hyperparameters, call `.Train(X_train, epochs=..., batch_size=...)`,
then `.generate(...)`/`.sample(...)`.

### Which generative model should I use?

| Model | Sample quality | Training stability | Generation speed | Gives exact likelihood? | Pick it when... |
|---|---|---|---|---|---|
| [VAE](#vae) | Blurrier | Very stable, no adversarial dynamics | Fast (one forward pass) | No (approximate bound) | You want a fast, reliable generator with a useful, smooth latent space (interpolation, downstream features) and can tolerate blurrier output |
| [GAN](#gan) | Sharpest | Can be finicky (balancing generator/discriminator) | Fast (one forward pass) | No | Sample sharpness matters most and you can afford to tune training |
| [DiffusionModel](#diffusionmodel) | Best overall | Very stable (simple denoising objective) | Slow (many steps; use `sample_ddim` to speed up) | No | You want the best sample quality and can tolerate (or accelerate) slower generation |
| [RealNVP](#realnvp-normalizing-flow) | Reasonable | Stable | Fast | **Yes, exact** | You specifically need a real likelihood/density number, not just samples, and want exact invertibility |
| [AutoregressiveModel](#autoregressivemodel) | Reasonable | Stable | Very slow (dimension-by-dimension) | **Yes, exact** | You need exact likelihood and/or want to fill in missing dimensions given known ones (`complete()`) |
| [EnergyBasedModel](#energybasedmodel) | Variable | Sensitive to hyperparameters | Slow (iterative sampling) | Unnormalized only | You want the most flexible model (no architecture constraints) and mainly need a gradient/critic signal (`score()`), not fast sampling |
| [UNetDenoiser](#unetdenoiser) | Best (image-shaped data) | Very stable | Slow (DDPM-style) | No | You want a real U-Net architecture specifically for image-shaped diffusion (vs. `DiffusionModel`'s flatter conv stack) |

All of VAE, GAN, and DiffusionModel also support
[class-conditional generation](#class-conditional-generation) (generate
samples of a specific class, e.g. "generate a 7").

### VAE

```python
VAE(input_dim, latent_dim, encoder_hidden=[512, 256], decoder_hidden=[256, 512],
   activation="swish", learning_rate=0.001, optimizer="adam", l2_lambda=0.0,
   num_classes=None)
```
A Variational Autoencoder: an encoder compresses each input into a
probability distribution over a small latent space, a decoder reconstructs
the input from a sample of that distribution. Trained to both reconstruct
well *and* keep the latent space close to a simple Gaussian (via a KL
divergence penalty), which is what makes the latent space smooth and
useful for interpolation.

```python
vae = VAE(input_dim=784, latent_dim=32, encoder_hidden=[256, 128])
vae.Train(X_train, epochs=30, batch_size=64, kl_weight=1.0)
samples = vae.generate(n_samples=16)
recon = vae.reconstruct(X_val)
midpoints = vae.interpolate(x1, x2, n_steps=10)
```
Methods: `encode(x) -> (mu, logvar)`, `decode(z)`, `forward(x) -> (recon, mu,
logvar, z)`, `loss(x, ..., kl_weight=1.0)`, `train_step(x, kl_weight=1.0)`,
`Train(..., callbacks=None)`, `generate(n_samples=1)` (also available as
`sample(n_samples=1)`, matching the other generative models' naming),
`reconstruct(x)`, `interpolate(x1, x2, n_steps=10)`. `Train` accepts the
same `callbacks=[...]` convention as `NeuralNet.Train`/`TextGenerator.Train`
(see [Callbacks](#callbacks)) — `on_batch_end`/`on_epoch_end`/`on_train_end`.

- **Pros:** smooth, well-structured latent space (good for interpolation,
  downstream feature extraction); stable to train, no adversarial dynamics.
- **Cons:** decoder ends in `sigmoid`, implicitly assuming inputs are scaled
  to `[0, 1]` — rescale your data accordingly; samples tend to be blurrier
  than GAN/diffusion output (a well-known VAE characteristic).

### GAN

```python
GAN(latent_dim, data_dim, generator_hidden=[256, 512], discriminator_hidden=[512, 256],
   g_activation="swish", d_activation="leakyrelu", loss_type="bce",
   learning_rate=0.0002, optimizer="adam", l2_lambda=0.0,
   label_smoothing=0.9, g_lr_factor=1.0, d_lr_factor=1.0, wgan_clip_value=0.01,
   num_classes=None)
```
A Generative Adversarial Network: a generator turns random noise into fake
samples, a discriminator tries to tell real from fake, and the two are
trained against each other — the generator gets better by fooling an
increasingly good discriminator.

```python
gan = GAN(latent_dim=64, data_dim=784, loss_type="wasserstein", wgan_clip_value=0.01)
gan.Train(X_train, epochs=100, batch_size=64, d_steps=5, g_steps=1)
samples = gan.sample(16)
print(gan.mode_collapse_score())  # 0=collapsed, 1=diverse
```
`loss_type` ∈ `"bce"`, `"bce_logits"`, `"wasserstein"` (the Wasserstein
variant is generally the most stable to train). Generator output is
`tanh` (assumes data scaled to `[-1, 1]`); discriminator output is
`sigmoid` for `"bce"`, else `linear`. `label_smoothing` softens real-label
targets. `mode_collapse_score()` is a quick diagnostic for "mode
collapse" — a common GAN failure mode where the generator finds a small
number of outputs that reliably fool the discriminator and stops
exploring, producing low-diversity samples. `Train(..., callbacks=None)`
accepts the same convention as `NeuralNet.Train` (see
[Callbacks](#callbacks)), except `on_batch_end`/`on_epoch_end`'s `logs`
carry both `d_loss` and `g_loss` (GAN tracks two losses, not one).

- **Pros:** sharpest samples among the generative models here; Wasserstein
  variant is more stable than vanilla BCE GAN; `mode_collapse_score()`
  gives a quick diversity diagnostic without external tooling.
- **Cons:** still a GAN — training can require balancing `d_steps`/
  `g_steps` and learning rates (`g_lr_factor`/`d_lr_factor` exist
  specifically to help with this); `mode_collapse_score`'s `n_clusters`
  parameter is accepted but currently unused (dead parameter).

### DiffusionModel

```python
DiffusionModel(data_shape, time_steps=1000, beta_schedule="linear",
              beta_start=1e-4, beta_end=0.02, denoiser_type="mlp",
              denoiser_hidden=[512, 512, 512], learning_rate=0.001,
              optimizer="adam", l2_lambda=0.0, use_ema=True, ema_decay=0.999,
              cosine_schedule_s=0.008, beta_clip=(0, 0.999),
              time_emb_dim=128, sample_clip_range=(-1.0, 1.0),
              num_classes=None)
```
A denoising diffusion model (DDPM): training gradually adds noise to real
data over many steps and teaches a network to predict (and thus reverse)
that noise; generation starts from pure noise and repeatedly denoises it
back into a sample. Currently the best sample-quality generative model in
most published comparisons, at the cost of slower generation than a GAN.

```python
diffusion = DiffusionModel(data_shape=(784,), time_steps=1000, beta_schedule="cosine")
diffusion.Train(X_train, epochs=50, batch_size=64)
samples = diffusion.sample(n_samples=16)
partial = diffusion.denoise(x_noisy, t_start=500, t_end=0)
```
`beta_schedule` ∈ `"linear"`, `"cosine"`; `denoiser_type` ∈ `"mlp"`, `"conv"`
(the latter requires `data_shape=(C, H, W)`). `use_ema=True` maintains a
separate exponential-moving-average copy of denoiser weights, typically
producing noticeably better samples than raw training weights (automatic,
swapped in during `sample()`/`denoise()`). `Train(..., callbacks=None)`
accepts the same convention as `NeuralNet.Train` (see
[Callbacks](#callbacks)).

- **Pros:** DDPM-style training (the well-studied, stable denoising
  objective); EMA weights typically produce noticeably better samples;
  both MLP and (basic) conv denoisers available; `sample_ddim()` (below)
  for fast generation.
- **Cons:** `sample()`/`denoise()` are O(`time_steps`) sequential forward
  passes — the slowest generation path in the library by design (this is
  inherent to standard DDPM ancestral sampling); the backward pass is
  manually unrolled rather than going through `NeuralNet.Backward`'s
  standard entry point.

#### DDIM fast sampling

```python
sample_ddim(n_samples=16, n_steps=50, eta=0.0, shape=None, clip=True)
```
```python
# 1000-step DDPM training as usual, but sample with only 50 forward passes
# instead of 1000:
samples = diffusion.sample_ddim(n_samples=16, n_steps=50)
```
Denoising Diffusion Implicit Models: instead of walking every one of
`time_steps` timesteps, `sample_ddim` builds a strided subsequence of
`n_steps` timesteps and predicts the final result directly at each step —
much faster generation, at a small potential quality cost. `eta=0.0`
(default) is fully deterministic; `eta=1.0` reproduces `sample()`'s
DDPM-like stochastic behavior. Roughly `time_steps / n_steps` times fewer
denoiser forward passes than `sample()`.

### RealNVP (normalizing flow)

```python
RealNVP(data_dim, n_coupling=4, hidden_dim=256, activation="swish",
       learning_rate=0.001, optimizer="adam", l2_lambda=0.0)
```
A normalizing flow: a stack of invertible transformations that turn a
simple distribution (Gaussian noise) into your data distribution and
*back*, exactly. This exact invertibility is what makes flows special —
you get a genuine likelihood number for any input, not just samples.

```python
flow = RealNVP(data_dim=2, n_coupling=6, hidden_dim=128)
flow.Train(X_train, epochs=50, batch_size=128)
samples = flow.sample(n_samples=500)
z, log_det = flow.forward(x)
x_reconstructed = flow.inverse(z)
```
Each of `n_coupling` layers alternates which half of the input dims gets
transformed. `log_prob(x)`/`loss(x)` (negative log-likelihood) are exact,
not a variational bound like a VAE's. `Train(..., callbacks=None)` accepts
the same convention as `NeuralNet.Train` (see [Callbacks](#callbacks)).

- **Pros:** exact likelihood computation and exact invertibility
  (`forward`/`inverse` are true inverses of each other) — useful when you
  need actual density estimates, not just samples.
- **Cons:** `data_dim` odd means the two coupling halves are unequal sizes
  (works, but architecturally asymmetric); scales less gracefully to very
  high-dimensional data than diffusion/GAN (a known property of
  coupling-layer flows in general).

### EnergyBasedModel

```python
EnergyBasedModel(data_dim, hidden_dims=[512, 512], activation="swish",
                 learning_rate=0.001, optimizer="adam", l2_lambda=0.0,
                 persistent_cd=True, persistent_buffer_size=1000, init_noise_scale=0.5)
```
An Energy-Based Model: instead of directly modeling a probability
distribution, it learns an "energy" function that's low for real data and
high for everything else — the most flexible generative modeling approach
here, since you don't need a normalized likelihood or an adversarial pair,
just a scalar energy function.

```python
ebm = EnergyBasedModel(data_dim=2, persistent_cd=True)
ebm.Train(X_train, epochs=50, batch_size=64, n_cd_steps=20)
samples = ebm.sample(n_samples=500, n_steps=200)
score = ebm.score(x)  # gradient of energy w.r.t. x
```
Trained via (persistent) contrastive divergence with Langevin dynamics
(gradient-based sampling with added noise) for negative sampling.
`persistent_cd=True` maintains a buffer of negative samples carried across
calls. `Train(..., callbacks=None)` accepts the same convention as
`NeuralNet.Train` (see [Callbacks](#callbacks)).

- **Pros:** conceptually the most flexible generative model here;
  `score(x)` directly gives you a gradient field useful for other
  downstream uses (e.g. as a critic/guidance signal).
- **Cons:** sampling requires an iterative Langevin chain (slower than a
  single forward pass); training via contrastive divergence is notoriously
  sensitive to hyperparameters — expect to tune more than with VAE/GAN.

### AutoregressiveModel

```python
AutoregressiveModel(data_dim, hidden_dims=[512, 512], data_shape=None,
                    activation="swish", learning_rate=0.001, optimizer="adam",
                    l2_lambda=0.0, num_classes=256, discrete=False)
```
Models the joint probability of all dimensions as a product of
conditionals — dimension *i*'s prediction only sees dimensions `< i` (like
`TextGenerator`, but over arbitrary vector dimensions instead of a token
sequence, and without attention). `discrete=True` treats each dimension as
a `num_classes`-way categorical (pixel-value-style, e.g. 0-255);
`discrete=False` uses per-dimension Gaussian/MSE.

```python
ar = AutoregressiveModel(data_dim=784, discrete=True, num_classes=256)
ar.Train(X_train, epochs=30, batch_size=64)
samples = ar.generate(n_samples=16)  # also available as ar.sample(n_samples=16)
completed = ar.complete(partial_x, n_dims=400)
print("log-likelihood:", ar.log_prob(X_val).mean())
```
`discrete=True` inputs to `loss`/`train_step`/`log_prob` must be scaled to
`[0, 1]` (rescaled internally to `0..num_classes-1`) — passing raw
un-normalized data (e.g. actual 0-255 pixel values instead of pixel/255)
raises a clear `ValueError` rather than silently training on clipped
garbage.

- **Pros:** exact likelihood (like normalizing flows); `complete()` gives a
  natural inpainting/completion API (fill in missing dimensions given
  known ones) that the other generative models don't offer directly.
- **Cons:** `generate()`/`complete()` sample dimension-by-dimension in a
  Python loop — O(data_dim²) forward passes, no caching (unlike
  `TextGenerator`'s KV-cache) — slow for high-dimensional data (full
  images). Discrete mode assumes continuous inputs are pre-scaled to
  `[0, 1]` before being bucketed into classes.

### UNetDenoiser

```python
UNetDenoiser(in_ch, base_ch=64, time_emb_dim=128, ch_mult=(1, 2, 4),
            learning_rate=0.001, optimizer="adam", l2_lambda=0.0,
            init_method="he_normal", pool_factor=2,
            time_steps=1000, beta_start=1e-4, beta_end=0.02)
time_embedding(t, dim, max_period=10000)  # standalone sinusoidal time embedding function
```
A real U-Net architecture (encoder-decoder with skip connections at
matching resolutions, the classic "U" shape) for diffusion on image-shaped
data — an alternative to `DiffusionModel(denoiser_type="conv", ...)`'s
flatter conv stack, specifically when you want proper U-Net-style
multi-resolution processing.

```python
unet = UNetDenoiser(in_ch=1, base_ch=64, ch_mult=(1, 2, 4))
out = unet.forward(x, t)

unet.Train(X_train, epochs=50, batch_size=32)   # full DDPM training, like DiffusionModel.Train
samples = unet.sample(n_samples=16, shape=(1, 28, 28))

# DDIM fast sampling (same idea as DiffusionModel.sample_ddim above, ported
# onto UNetDenoiser's own noise schedule): far fewer forward passes than
# the full time_steps ancestral sample() above.
fast_samples = unet.sample_ddim(n_samples=16, shape=(1, 28, 28), n_steps=50, eta=0.0)
```
Every convolution is `k=3, padding="same"` (real 3x3 convs, not 1x1),
with skip connections at matching resolutions. Fully independently
trainable — `backward(grad_output)` backprops through every subnetwork in
exact reverse of `forward()`'s order, and `train_step`/`Train`/`sample`/
`sample_ddim` give it its own self-contained DDPM training loop (a
linear-only noise schedule — no cosine option, no EMA; use `DiffusionModel`
if you need those).

- **Pros:** encoder-decoder with skip connections at matching resolutions,
  real 3x3 same-padding convs, independently trainable (`get_params()`
  exposes every internal `NeuralNet` sub-network if you want to
  inspect/customize training yourself).
- **Cons:** its own noise schedule is linear-only (no cosine, no EMA);
  the time embedding only reaches a block if that block's channel count
  exactly equals `time_emb_dim * 4` — there's no per-block learned
  projection like a typical U-Net, so in most configurations time
  conditioning silently doesn't reach most blocks (check `time_emb_dim`
  vs `base_ch * ch_mult[i]` if this matters for you).

### Class-conditional generation

`VAE`, `GAN`, and `DiffusionModel` all accept `num_classes=None` (default —
fully unconditional) plus a `labels`/`y` argument threaded through training
and generation, letting you generate a sample of a *specific* class (e.g.
"generate a picture of digit 7") instead of a random one. Passing one
without the other raises `ValueError` — either both `num_classes=...` at
construction and `y=...`/`y_train=...` at every train/generate call, or
neither.

```python
# VAE: 10-class conditional (e.g. MNIST digits)
vae = VAE(input_dim=784, latent_dim=32, num_classes=10)
vae.Train(X_train, epochs=30, batch_size=64, y_train=y_train)
samples_of_digit_7 = vae.generate(n_samples=16, y=7)

# GAN: same convention
gan = GAN(latent_dim=64, data_dim=784, num_classes=10)
gan.Train(X_train, epochs=100, batch_size=64, y_train=y_train)
samples_of_digit_7 = gan.sample(16, y=7)

# DiffusionModel: same convention, including sample_ddim
diffusion = DiffusionModel(data_shape=(784,), num_classes=10)
diffusion.Train(X_train, epochs=50, batch_size=64, y_train=y_train)
samples_of_digit_7 = diffusion.sample_ddim(n_samples=16, n_steps=50, y=7)
```

`y`/`y_train` accepts either a single scalar class index (broadcast across
the whole batch — e.g. `y=7` above generates 16 samples all of class 7) or
a per-sample array of class indices matching the batch size.

- **Pros:** verified end-to-end on a synthetic multi-class toy dataset for
  all three models; fully backward compatible (`num_classes=None` is the
  default and changes nothing about the unconditional API or math).
- **Cons:** no classifier-free guidance or conditioning-strength control —
  the model either was built conditional or wasn't; label validation is
  shape/presence-only (an out-of-range class index fails with a NumPy
  indexing error rather than a clear message here).

### Low-level sampling & loss building blocks

Used internally by the models above, also directly importable for building
your own generative training loops:

```python
# Enilnets.generative.sampling
reparameterize(mu, logvar)                       # VAE reparameterization trick
langevin_dynamics(energy_fn, x_init, n_steps=20, step_size=0.1, noise_scale=0.005)
gaussian_sample(mean, std, shape=None)
uniform_sample(low, high, shape)
gumbel_softmax_sample(logits, temperature=1.0, hard=False)
random_mask(shape, ratio)
top_p_sampling(logits, p=0.9, temperature=1.0)   # batched (batch, vocab) logits -> one-hot array
top_k_sampling(logits, k=10, temperature=1.0)    # single 1D logits vector -> plain int index
gae(rewards, values, gamma=0.99, lambda_=0.95)   # -> (advantages, returns); import from Enilnets.generative.sampling

# Enilnets.generative.generative_loss
kl_divergence_gaussian(mu, logvar, reduction="mean", kl_weight=1.0)
adversarial_loss_discriminator(real_logits, fake_logits, loss_type="bce")  # "bce"|"bce_logits"|"wasserstein"
adversarial_loss_generator(fake_logits, loss_type="bce")
diffusion_loss(predicted_noise, true_noise, reduction="mean")
nll_loss(log_px, log_det_jacobian, reduction="mean")
energy_loss(data_energy, sample_energy, margin=1.0)
perceptual_loss(x, y, feature_extractor=None)  # falls back to plain MSE if no extractor given
vgg_loss(x, y, vgg_features=None)              # falls back to plain MSE if no vgg_features given
```

**Known gotcha:** `top_p_sampling` and `top_k_sampling` have different call
conventions despite similar names — `top_p_sampling` takes a batch of
logits and returns a one-hot array; `top_k_sampling` takes a single 1D
logits vector and returns a plain integer index.

### Bring your own pretrained weights

Enilnets never downloads anything — no network fetches of any kind, by
design. To use `vgg_loss` or `inception_score` with real pretrained
features, convert weights yourself (e.g. from a `.npz`/`.npy` export of
another framework's checkpoint) and load them in:

```python
from Enilnets.generative.generative_loss import vgg_loss
from Enilnets.eval_utils import inception_score

# Any callable duck-typing to feature_extractor(x) -> features works --
# doesn't have to be an Enilnets NeuralNet.
vgg_loss(x, y, vgg_features=my_feature_extractor_fn)

# inception_score's `classifier` duck-types to anything with
# `.Forward(batch) -> (N, num_classes)` -- e.g. your own NeuralNet with
# converted weights loaded via `.set_weights()`.
score = inception_score(samples, classifier=my_classifier_model)
```

A ready-made VGG16 skeleton (not re-exported at top level — import it
directly):

```python
from Enilnets.generative.pretrained import build_vgg16_feature_extractor

vgg = build_vgg16_feature_extractor(up_to_block=5, input_ch=3)  # randomly initialized
vgg.set_weights(my_converted_vgg16_weights)  # your own conversion, e.g. from a .npz export
vgg_loss(x, y, vgg_features=vgg.Forward)
```

`up_to_block` (1-5) controls how many of VGG16's 5 conv blocks to include
(13 conv layers total at `up_to_block=5`, each `k=3` with `padding="same"`,
pooled 2x after each block — a 224x224 input produces the standard
`(N, 512, 7, 7)` VGG16 feature map shape at `up_to_block=5`). No
Inception-v3 skeleton is provided — its branching topology doesn't map
onto this library's sequential layer-list model; use `inception_score`'s
`.Forward(batch) -> (N, num_classes)` duck-typed contract with your own
network instead.

## Reinforcement learning

Policy-gradient methods hang directly off any `NeuralNet` used as a policy
(and, for `ActorCritic`, a second `NeuralNet` as a value function). All of
them build the output-layer gradient directly rather than going through
`ComputeLoss`, since policy gradients aren't a standard supervised loss.
See [Sequential decision-making](#sequential-decision-making-reinforcement-learning)
above for when to reach for RL at all.

```python
from Enilnets import NeuralNet, compute_returns

policy = NeuralNet(learning_rate=0.001, optimizer="adam")
policy.add_dense(state_dim, 64, activation="relu")
policy.add_dense(64, n_actions, activation="softmax")

returns = compute_returns(rewards, gamma=0.99)
policy.Reinforce(states, actions, returns, action_type="discrete")
```

| Method | Signature | Use it for |
|---|---|---|
| `Reinforce` | `(states, actions, returns, action_type="discrete", std=1.0, normalize_returns=True)` | Vanilla REINFORCE, the simplest policy-gradient method. `action_type` ∈ `"discrete"` (softmax policy) / `"continuous"` (Gaussian policy, fixed `std`). |
| `PPO` | `(states, actions, old_log_probs, advantages, action_type="discrete", epsilon=0.2, std=1.0, value_targets=None, value_coeff=0.5, entropy_coeff=0.01, value_network=None)` | Proximal Policy Optimization — the modern standard for more stable policy-gradient training. Pass a separate `value_network` together with `value_targets` to also train a value head. |
| `ActorCritic` | `(states, actions, returns, values, action_type="discrete", std=1.0)` | Advantage actor-critic; `values` comes from a separate critic `NeuralNet` you maintain yourself. |
| `Evolve` | `(inputs, score_fn, noise=0.05, tries=10, sigma=1.0)` | Gradient-free evolution strategy: perturbs weights randomly, keeps whichever variant scores best under `score_fn`. Use when you don't have (or don't trust) a gradient signal at all. |
| `compute_returns` | `(rewards, gamma=0.99)` | Standalone function — discounted-return computation. **Call it as `Enilnets.compute_returns(rewards, gamma=...)`, not as a bound `policy.compute_returns(...)` method** (that would double-pass `self`). |

`gae(rewards, values, gamma=0.99, lambda_=0.95)` (Generalized Advantage
Estimation, a lower-variance alternative to raw returns for advantage
computation) lives in `Enilnets.generative.sampling`:
```python
from Enilnets.generative.sampling import gae
advantages, returns = gae(rewards, values, gamma=0.99, lambda_=0.95)
```

## NEAT (neuroevolution)

`NEATPopulation` implements NeuroEvolution of Augmenting Topologies: a
population of small networks ("genomes") evolves via mutation (perturb
weights, add a connection, add a node) and crossover, guided only by a
fitness function you supply — no gradients required, and the network's
*structure* (not just its weights) is discovered rather than designed. See
[You don't know what architecture you need](#you-dont-know-what-architecture-you-need)
above.

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
XOR is the classic NEAT sanity check: it isn't linearly separable, so a
population that starts fully-connected with no hidden units must actually
*discover* a hidden node to solve it.

```python
NEATPopulation(n_inputs, n_outputs, population_size=150, activation="sigmoid",
              output_activation=None, compatibility_threshold=None, c1=None, c2=None, c3=None,
              weight_mutate_rate=None, weight_perturb_rate=None, weight_perturb_power=None,
              add_connection_rate=None, add_node_rate=None, crossover_rate=None,
              survival_threshold=None, stagnation_limit=None, elitism=1, seed=None)
```
Every `None`-defaulted knob falls back to a `constants.NEAT_*` value (see
[Configuration system](#configuration-system) for the full list) — the
defaults are reasonable starting points; tune `population_size` and
`generations` first before touching the mutation-rate internals.

For direct genome manipulation (custom evolutionary loops, single-genome
save/load, etc.):

```python
from Enilnets.neat import Genome, InnovationTracker, crossover  # crossover also exported as Enilnets.neat_crossover

genome = Genome.minimal(n_inputs=2, n_outputs=1, innovation_tracker=InnovationTracker())
out = genome.forward(np.array([0.0, 1.0]))  # accepts (n_inputs,) or (batch, n_inputs), matching rank out
genome.mutate_weights(...)
genome.mutate_add_connection(innovation_tracker, ...)  # -> bool (success)
genome.mutate_add_node(innovation_tracker)             # -> bool (success)
distance = genome.distance(other_genome, c1=1.0, c2=1.0, c3=0.4)  # compatibility distance, for manual speciation
child = crossover(fitter_genome, other_genome)         # assumes fitter.fitness >= other.fitness
```

**Worth knowing if you build custom NEAT pipelines:** each mutation checks
for cycles within that one genome, so it's always locally safe — but
`crossover`'s gene-matching can (rarely) reintroduce a cycle when combining
two independently-acyclic parents. `crossover()` calls
`child.break_cycles()` internally to guard against this automatically; if
you ever hand-construct a genome by merging connection dicts yourself
(bypassing `crossover()`), call `genome.break_cycles()` before
`forward()`-ing it.

## Visualization

`plot_network`/`model.plot(...)` renders the classic node/connection
diagram of a `NeuralNet` as a self-contained SVG string — pure stdlib, no
matplotlib dependency. Useful for sanity-checking an architecture or
showing a live snapshot during training.

```python
model = NeuralNet(learning_rate=0.001, optimizer="adam")
model.add_dense(4, 8, activation="relu")
model.add_batchnorm(8)
model.add_dense(8, 3, activation="softmax")

svg = model.plot()  # structure only, no values

# With a sample input, node fill color shows that layer's actual activation
# value from a live forward pass (heat-mapped blue=low to red=high):
for epoch in range(epochs):
    model.TrainBatch(X_batch, Y_batch)
    if epoch % 10 == 0:
        model.plot(sample_input=X_batch[:1], filename=f"epoch_{epoch}.svg")
```
Dense/sparse/RNN/LSTM/GRU/conv2d/conv1d/attention layers all become
columns of circular nodes with real weighted edges (blue = positive, red
= negative, thickness/opacity = relative magnitude) — conv edges
aggregate the spatial kernel into a per-channel-pair magnitude, and
attention edges aggregate its Q/K/V input projections, since neither
reduces to one plain weight matrix the way dense does. Every other layer
type (norm, pooling, dropout, embedding, positional encoding, residual
markers, ...) renders as a labeled block; embedding gets a small
table/grid glyph instead of plain text. `residual_save`/`residual_add`
pairs additionally get a curved dashed skip-connection edge between them,
so a ResNet-style block's actual topology is visible instead of two
disconnected-looking blocks. Batchnorm/layernorm/dropout/pooling/upsample
are transparent to the diagram (edges drawn straight through them, since
they don't change channel/feature count). Hover any node or block for a
tooltip naming its originating layer index.

`plot_genome(genome, sample_input=..., show_disabled=True)` does the same
for a NEAT `Genome` — nodes colored by type (a legend is drawn along the
bottom), columns labeled by graph depth, edges colored/weighted by
connection weight and dashed if disabled.

**Using it in a real project:**
```python
svg = model.plot(sample_input=x)   # always the raw SVG string, embed it directly

from IPython.display import SVG, display
display(SVG(svg))                  # Jupyter

from Enilnets import to_html
model.plot(sample_input=x, filename="network.svg")    # raw SVG file
model.plot(sample_input=x, filename="network.html")   # standalone HTML doc, open in any browser
```

Large layers/depth-columns auto-cap (`max_nodes_per_layer=20` default for
`plot_network`, `30` for `plot_genome`), showing first/last half with a
`⋮` marker in between — a 784-neuron input layer won't render 784
circles. `plot_network` raises `ValueError` on a model with no layers.

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
class-label arrays (argmax first); `num_classes=None` infers from the
data. `classification_report(...)` returns `{0: {...}, 1: {...}, ...,
"macro_avg": {...}, "weighted_avg": {...}, "accuracy": float}`.

Generative-model-focused evaluation lives in `Enilnets.eval_utils` (import
the module directly — not re-exported at top level):
```python
from Enilnets import eval_utils

eval_utils.inception_score(samples, classifier=None, splits=10)
# with classifier=None, falls back to a crude k-means-based diversity
# proxy, NOT a real Inception-v3-based score -- pass your own classifier
# for anything resembling the standard metric.

eval_utils.frechet_distance(...) / eval_utils.compute_fid(...)
eval_utils.reconstruction_error(original, reconstructed, metric="mse"|"mae"|"psnr", max_val=1.0)
# max_val only matters for "psnr" -- the data's max possible value (1.0 for
# [0,1]-scaled data, the default; pass 255 for raw uint8-range images).
eval_utils.sample_diversity(samples)
eval_utils.nearest_neighbor_accuracy(real_features, fake_features, k=5)
# returns the average distance from each fake sample to its k nearest real
# neighbors (lower = fake samples land close to real data) -- an unbounded
# distance, not a [0,1] fraction, despite the name.
```

### Differentiable audio front ends

```python
from Enilnets.graph import Tensor, audio_stft, spectrogram, mel_spectrogram, log_mel_spectrogram

x = Tensor(waveform, requires_grad=True)
feats = log_mel_spectrogram(x, sr=16000, n_fft=512, hop_length=128, n_mels=40)
feats.sum().backward()          # gradients reach the raw waveform
```
`audio/audio_utils.py`'s `stft`/`spectrogram_to_mel` are plain array
functions with no gradient — fine for preprocessing, useless if the front
end is part of the model. These graph versions are differentiable.

They are built **entirely from ops that already exist**: framing is a
gather, the DFT is a matmul against a precomputed complex matrix, and the
magnitude is the complex `absolute`. So there are no new gradient rules
here to get wrong, and [complex tensors](#complex-tensors) are what make
the DFT expressible at all. The STFT is checked against `numpy.fft.rfft` to
1e-8, and the gradient against finite differences.

- **Cost:** an FFT would be O(N log N) where a DFT matmul is O(N²) per
  frame. For the frame sizes a from-scratch model actually uses, that buys
  a differentiable path with no bespoke backward.
- A real signal gets a **real** gradient even though the STFT is
  complex-valued in between — under the conjugate-Wirtinger convention the
  imaginary parts cancel, and the gather's backward takes `.real`
  explicitly rather than tripping NumPy's silent-cast warning.

### Audio transform pipeline

The [transform pipeline](#transform-pipelines-preprocessing) covers audio
too: `LoadAudio` (path → waveform), `ToSpectrogram`, `ToMelSpectrogram`,
`LogCompress`, `AugmentAudio`, plus SpecAugment's `TimeMask` and `FreqMask`.

```python
from Enilnets import Compose
from Enilnets.preprocessing import AugmentAudio, ToMelSpectrogram, LogCompress, FreqMask, TimeMask

pipe = Compose([AugmentAudio(sr=16000, noise_std=0.01),
                ToMelSpectrogram(sr=16000, n_mels=40),
                LogCompress(), FreqMask(8), TimeMask(10)])
```
The masks copy rather than mutate their input, and every audio transform
preserves the working dtype (`augment_audio`'s float64 noise used to
promote a float32 signal — fixed at the source).

## Compression: pruning and quantization

Both operate on an already-trained model, and live in
`Enilnets.compression`.

### Pruning

```python
from Enilnets import prune_magnitude, PruningSchedule, prune_channels, sparsity

prune_magnitude(model, amount=0.8)      # zero the smallest 80%, and hold them
model.Train(X, Y, epochs=10)            # fine-tune; the zeros stay zero
print(sparsity(model)["overall"])       # -> 0.8
```

| | What it does |
|---|---|
| `prune_magnitude(model, amount, scope)` | Zero the smallest-magnitude weights. `scope="global"` ranks the whole model together (layers that matter keep more capacity); `scope="layer"` takes the same fraction from each |
| `PruningSchedule(model, final, start_step, end_step)` | Ramp sparsity **during** training on Zhu & Gupta's cubic schedule — fast early, tapering later, so the model can recover between the more damaging increments |
| `prune_channels(model, layer_index, amount)` | **Structured**: remove whole output channels and rewire every consumer, so tensors genuinely shrink |
| `sparsity(model)` / `clear_masks(model)` | Report actual zeros; stop holding them |

A pruned weight is recorded in `layer["prune_mask"]`, and `apply_gradients`
enforces it every step — otherwise the very next update would move each
zeroed weight straight off zero. It also masks the **gradient**, and
`prune_magnitude` zeroes the corresponding optimizer accumulators: a stale
momentum on a pruned weight is wasted work and pollutes the adaptive
denominators of the weights that remain.

**Measured** on a 20→64→4 classifier (cross-entropy; lower is better):

| | loss |
|---|---|
| float baseline | 0.201 |
| pruned 50%, no fine-tune | 0.390 |
| pruned 80%, no fine-tune | 1.118 |
| pruned 80%, one-shot then fine-tuned | 0.722 |
| pruned 90%, one-shot then fine-tuned | 1.166 |
| **pruned 90% *gradually* via `PruningSchedule`** | **0.710** |
| quantized 8-bit per-channel | 0.202 |
| quantized 4-bit per-channel | 0.226 |
| pruned 50% + quantized 8-bit | 0.390 |

Two things worth reading off that table. **Gradual pruning beats one-shot
plus fine-tuning** at high sparsity — 0.710 vs 1.166 at 90%, and it even
beats one-shot at *80%*; that is the whole reason `PruningSchedule` exists,
and it holds on 5/5 seeds. And **8-bit quantization is essentially free**
(0.201 → 0.202) where pruning always costs something, so if you only do one
of the two, quantize.

The masks cost **1.01×** per training step — effectively nothing — and the
one-off passes are milliseconds on ~530k parameters.

**Structured vs magnitude pruning** is the real distinction. Magnitude
pruning leaves shapes untouched, so the model is *sparser* but not smaller
or faster in NumPy. `prune_channels` actually shrinks the tensors — which
is why it is restricted: the consuming layer's input axis has to be
narrowable unambiguously. Conv → batchnorm → conv, conv → pool → flatten →
dense and dense → dense all work (the flatten's channel-to-column block
size is derived from the widths); anything else raises rather than
silently corrupting the model.

### Quantization

```python
from Enilnets import quantize_weights, ActivationCalibrator

report = quantize_weights(model, bits=8, per_channel=True)
print(report["mean_abs_error"], report["compression"])   # -> ~7e-4, 4.0

cal = ActivationCalibrator(model, bits=8)
for batch in calibration_batches:
    cal.observe(batch)
cal.apply()                     # Forward now quantizes activations too
```

**Be clear about what this buys you.** NumPy has no int8 matmul — it would
upcast immediately — so this does **not** make inference faster. It
reproduces the accuracy effect *exactly*, so you can measure whether a
model survives 8-bit (or 4-bit) before committing to a deployment target,
and it stores the integer representation plus scales so a saved model can
be a quarter the size. Weights are left **fake-quantized**: rounded onto
the integer grid and mapped back to float, which is precisely what an
integer kernel would reconstruct.

- `scheme="symmetric"` (default) centres the grid on zero, so an exact zero
  stays exactly zero — which matters a lot for a pruned or ReLU'd tensor.
  `"asymmetric"` uses the full range, better for one-sided data.
- `per_channel=True` gives each output channel its own scale. One outlier
  channel then cannot force a coarse grid on all the others — measured
  **>50× lower error on the remaining channels** in that situation.
- Activation ranges cannot be read off the weights; they depend on the
  data, which is what the calibration pass is for.

### Quantization-aware training

```python
from Enilnets.graph import QATLinear

layer = QATLinear(n_in, n_out, bits=4, per_channel=True)
```
PTQ rounds a finished model and measures the damage. QAT makes the model
*train through* the rounding so it learns weights that survive it.

The whole difficulty is the gradient: rounding has a derivative of zero
almost everywhere, so backpropagating it honestly would stop training dead.
The answer is the **straight-through estimator**, implemented here as a
single `custom_op` — no hand-derived backward anywhere. It is *clipped*:
gradient passes through inside the representable range and is zeroed
outside, since a value that saturated the grid cannot be improved by being
pushed further out.

- **Correctness:** a test asserts the pass-through/clip rule directly, that
  a QAT step is a descent direction on the quantized loss, and — the
  sharpest check — that at 12 and 16 bits QAT training reaches the *same*
  loss as training with no quantization at all, i.e. the estimator is
  transparent when the grid is fine.
- A finite-difference check is **meaningless here by design**: the STE is
  deliberately not the true derivative.
- **How to use it:** QAT is fine-tuning *from* a trained model, with a
  decayed learning rate, stochastic batches, and keeping the best quantized
  checkpoint. Trained from scratch it is strictly harder than plain
  training, because the estimator's noise is pure cost until the model is
  near a solution. In that fine-tuning regime it beats PTQ at 2 and 3 bits,
  where PTQ is genuinely broken; at 8 bits PTQ is already near-free and
  there is nothing to win.

## Data utilities

### General (`core/utils.py`)

```python
from Enilnets import set_seed, train_test_split, k_fold_split, iterate_minibatches, count_parameters, one_hot

set_seed(0)                                              # seed NumPy's global RNG, for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=0)
for fold_X_train, fold_X_val, fold_y_train, fold_y_val in k_fold_split(X, y, k=5, seed=0):
    ...                                                   # generator, one split per fold
for xb, yb in iterate_minibatches(X, y, batch_size=32, shuffle=True, seed=None):
    ...                                                   # generator, one minibatch at a time
total, per_layer = count_parameters(model)                # -> (int, list of dicts)
oh = one_hot(np.array([0, 2, 1]), num_classes=3)           # -> one-hot array
```
`train_test_split`/`iterate_minibatches`'s `seed=None` uses the *global*
`np.random` state (not reproducible across calls unless you called
`set_seed` first); pass `seed` for an isolated, reproducible split/shuffle.
`k_fold_split` distributes remainder samples (`n % k`) one-per-fold to the
first folds.

### Datasets and the DataLoader

`iterate_minibatches` above is fine for arrays that fit in memory. When
they don't, or when you want augmentation, prefetching or worker threads,
use a `Dataset` and a `DataLoader`.

```python
from Enilnets import DataLoader, ArrayDataset, random_split

train, val = random_split(ArrayDataset(X, Y), [0.8, 0.2], seed=0)
loader = DataLoader(train, batch_size=32, shuffle=True, seed=0)

model.Train(loader, epochs=20)          # Train() accepts a loader in place of (X, Y)
for xb, yb in loader:                   # ...or iterate it yourself
    model.TrainBatch(xb, yb)
```

`DataLoader` accepts a `Dataset`, an `(X, Y)` pair, a bare `X`, or a
callable returning a fresh iterator — so it is a drop-in wherever
`iterate_minibatches` was.

| Dataset | What it is |
|---|---|
| `ArrayDataset(X, Y=None)` | In-memory arrays, held **by reference** (never copied) |
| `MemmapDataset(path, shape, dtype, ...)` | An on-disk array via `np.memmap` — rows page in as touched, so the file may be far larger than RAM |
| `StreamingDataset(factory)` | A *factory* returning a fresh iterator; unknown length, yielded in order |
| `Subset(ds, indices)` / `ConcatDataset([...])` | Views: a reordered slice, or several datasets end to end (`a + b`) |
| `ds.map(fn)` | A lazy per-sample transform |
| `random_split(ds, [0.8, 0.2], seed=0)` | Disjoint `Subset`s; counts or fractions |

`StreamingDataset` takes a **callable**, not an iterator, and rejects an
iterator explicitly — an iterator would be exhausted after one epoch and
silently yield nothing thereafter.

**DataLoader options.** `batch_size`, `shuffle` (reshuffled every epoch,
reproducibly from `seed`), `drop_last`, `transform` (applied to the
collated *batch*), `collate` (custom stacking), `shuffle_buffer` (for
streams, which have nothing to permute), `prefetch`, and
`num_workers`/`worker_backend`.

**On workers — what actually helps.** Measured on 64 samples, batch 8:

| Per-sample work | serial | 4 threads | 4 processes |
|---|---|---|---|
| I/O-bound (a real file read) | 0.129 s | 0.034 s (**3.8×**) | — |
| NumPy-heavy | 0.345 s | 0.134 s (**2.6×**) | 0.103 s (**3.3×**) |
| Pure Python | 0.055 s | 0.071 s (**0.8×**) | 0.019 s (**2.9×**) |

Threads help exactly when `__getitem__` releases the GIL — file and memmap
reads, NumPy work — and are *slower* than serial for pure-Python work,
where the pool overhead buys nothing. Processes help in both cases, but
`worker_backend="process"` requires a **picklable** dataset (no lambdas, no
locally-defined classes) and is refused under GPU mode, since device arrays
do not cross process boundaries. Both conditions are checked at
construction, not mid-epoch. Threads remain the default because they work
with any dataset.

`prefetch=n` keeps `n` batches ready on a background thread; an exception
raised while producing surfaces in the consuming loop rather than
disappearing with the thread. It pays off when producing a batch is slow
and *costs* when it is instant — on in-memory arrays it roughly triples a
1 ms epoch, so leave it off unless the dataset does real work per batch.

**The DataLoader is not a tax.** Array-backed datasets gather a whole batch
in one indexing operation rather than fetching rows one at a time, so the
convenience is close to free. One epoch over 20 000 × 64 samples at batch
128:

| | per epoch |
|---|---|
| `iterate_minibatches` | 1.20 ms |
| `DataLoader(X, Y)` | 1.28 ms (1.07×) |
| `DataLoader` over a `random_split` subset | 1.07 ms (0.90×, it is 80% of the data) |

The fast path is skipped where it cannot apply — worker pools, a custom
`collate`, or a per-sample `.map()` are all inherently per-sample — and a
test pins that both paths produce byte-identical batches.

### Transform pipelines (`preprocessing/`)

```python
from Enilnets import Compose, DataLoader, ArrayDataset
from Enilnets.preprocessing import OnX, ToDtype, Scale, Normalize, RandomFlip, RandomCrop

augment = OnX(Compose([
    ToDtype(), Scale(1 / 255), Normalize(0.5, 0.5),
    RandomFlip(), RandomCrop(32, padding=4),
]))
loader = DataLoader(ArrayDataset(X, Y).map(augment), batch_size=64)
```
A transform is any callable `sample -> sample`. `Compose` chains them, and
there are two places to attach one — the difference matters:

- **`dataset.map(t)`** runs per *sample*, so a random transform draws fresh
  randomness for each one. What augmentation wants.
- **`DataLoader(transform=t)`** runs on the collated *batch*, so one
  vectorized call covers it but a random draw is shared across the batch.
  Cheaper; correct for deterministic transforms like normalization.

`OnX`/`OnY` apply an inner transform to only one half of an `(x, y)`
sample, so an image augmentation and a label transform can sit in the same
`Compose` without either knowing the sample's shape.

Available: `ToDtype`, `Scale`, `Normalize`, `Clip`, `Reshape`, `OneHot`,
`RandomFlip`, `RandomCrop`, `CenterCrop`, `Resize`, `RandomNoise`,
`Augment`, `AugmentAudio`, `PadSequence`, `Tokenize`, plus the combinators
`Compose`, `OnX`, `OnY`, `Lambda`, `RandomApply` and `OneOf`.

- **`Normalize` applies fixed statistics**, unlike
  `preprocessing.normalize_images`, which *computes* them from the data it
  is handed — doing that per batch would normalize every batch differently.
- **Every transform preserves the working dtype.** This needed fixing:
  `Normalize`'s constants, `Resize`'s resampler and `image_augmentation`'s
  random draws were each float64, so a float32 batch came out float64 and
  reached a float32 model promoted. Pinned by a test.
- `Augment` (the existing `image_augmentation`) is batch-only and expects
  float images already in [0, 1] — it clips to that range.

### Text (`text/text_utils.py`)

The `Tokenizer` class is covered above under
[Text generation](#text-generation-textgenerator). Also:
```python
from Enilnets.text_utils import load_text_file, load_texts_from_directory, create_sliding_windows, pad_sequences

text = load_text_file("corpus.txt", encoding='utf-8')
texts = load_texts_from_directory("corpus_dir/", max_files=100)   # silently skips undecodable files
X, y = create_sliding_windows(token_ids, window_size=64, stride=1)  # -> next-token-prediction pairs
padded = pad_sequences([[1,2,3], [4,5]], max_length=5, pad_value=0, dtype=np.int32)
# dtype defaults to int32 (token-id sequences); pass e.g. backend.default_dtype()
# to pad batches of continuous-feature sequences instead.
```

### Images (`vision/image_utils.py`)

```python
from Enilnets import image_utils

image_utils.load_ppm(path) / image_utils.save_ppm(arr, path)     # binary P6 only, maxval must be 255
image_utils.load_pgm(path) / image_utils.save_pgm(arr, path)     # binary P5 only
image_utils.load_raw_binary(path, shape, dtype=None) / image_utils.save_raw_binary(arr, path)  # dtype=None uses the active default precision
image_utils.rgb_to_grayscale(rgb) / image_utils.grayscale_to_rgb(gray)
image_utils.resize_nearest_neighbor(img, new_height, new_width)
image_utils.resize_bilinear(img, new_height, new_width)
image_utils.image_augmentation(images, flip_h=True, flip_v=False, rotate=0, brightness=0.0, contrast=0.0, noise_std=0.0)
# images: (N, C, H, W) or (N, H, W) grayscale -- NCHW, matching this
# library's conv2d convention (NOT (N, H, W, C)). rotate is a CAP on
# {0,90,180,270}, not a continuous angle -- rotate=45 allows no rotation
# at all (90 > 45), rotate=90 allows {0,90}, rotate=360 allows all four.
image_utils.normalize_images(images, mean=None, std=None)   # -> (normalized, mean, std) -- 3-tuple
image_utils.denormalize_images(images, mean, std)
image_utils.images_to_patches(images, patch_size, stride=None)  # images: (N, C, H, W) -- NCHW
image_utils.pad_image(img, pad_h, pad_w, mode='constant', constant_value=0)
```
`load_ppm`/`load_pgm` raise `ValueError` for anything other than binary
P6/P5 with `maxval=255` — no ASCII PPM/PGM support. `image_augmentation`'s
`rotate` is a 90°-multiple ceiling, not an arbitrary-angle rotation.

### Audio (`audio/audio_utils.py`)

```python
from Enilnets import audio_utils

audio, sr = audio_utils.load_wav(path)                    # PCM16/24/32 or IEEE float32
audio_utils.save_wav(audio, path, sr, bits_per_sample=16)
spec = audio_utils.stft(audio, n_fft=2048, hop_length=512, window='hann')
audio_back = audio_utils.istft(spec, ...)
mel = audio_utils.spectrogram_to_mel(spec, sr, n_mels=128)
spec_back = audio_utils.mel_to_spectrogram(mel, sr, n_freq, ...)   # pseudo-inverse, not exact
log_mel = audio_utils.audio_to_spectrogram(audio, sr, n_fft=2048, hop_length=512, n_mels=128)
audio_reconstructed = audio_utils.spectrogram_to_audio(log_mel, sr, n_iter=32)  # Griffin-Lim
frames = audio_utils.audio_to_frames(audio, frame_length, hop_length=None)
audio_from_frames = audio_utils.frames_to_audio(frames, hop_length, window='hann')
augmented = audio_utils.augment_audio(audio, sr, pitch_shift=0, time_stretch=1.0, noise_std=0.0)
```
`save_wav` silently rescales audio if `max(abs(audio)) > 1.0`.
`augment_audio`'s `pitch_shift`/`time_stretch` use naive resampling with no
anti-aliasing — adequate for data augmentation, not studio-quality
processing.

### Cross-modal (`crossmodal/crossmodal_utils.py`)

```python
from Enilnets import crossmodal_utils

loss = crossmodal_utils.contrastive_loss(image_embeds, text_embeds, temperature=0.07)  # symmetric InfoNCE, CLIP-style
normalized = crossmodal_utils.clip_normalize(embeddings)   # L2-normalize rows
fused = crossmodal_utils.multimodal_fusion([emb_a, emb_b], fusion_type="attention")  # "concat"|"sum"|"gated"|"attention"
config = crossmodal_utils.create_text_conditioned_image(image_shape, text_embed_dim, num_classes=10)
```
`fusion_type="attention"` computes real data-dependent attention over
modalities (each sample gets its own per-modality weighting), unlike
`"gated"`'s static per-modality weights. `create_text_conditioned_image`
doesn't build any network — it returns a plain dict of suggested
hyperparameters; treat it as a config template, not a working constructor.

### Dataset loaders (`datasets/loaders.py`)

```python
from Enilnets.datasets import load_mnist, load_cifar10

X, y = load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte", normalize=True)
X, y = load_cifar10(["data_batch_1", "data_batch_2"], normalize=True)   # accepts one path or a list
```
Local-file-only parsers for two standard benchmark formats — zero network
fetch of any kind; you download the files yourself (MNIST from
`http://yann.lecun.com/exdb/mnist/`, CIFAR-10 from
`https://www.cs.toronto.edu/~kriz/cifar.html`) and pass their paths.

- **`load_mnist`**: parses the standard IDX binary format. Returns `X` as
  `(N, 1, 28, 28)` at the active default precision (float32 unless
  `use_float64(True)` was called) and `y` as `(N,)` int64 class labels.
  `normalize=True` divides by `255.0` — for z-score normalization instead,
  call `image_utils.normalize_images(X)` yourself after loading.
- **`load_cifar10`**: unpickles standard CIFAR-10 batch files. Returns `X`
  as `(N, 3, 32, 32)` at the active default precision — **channel-major**,
  matching CIFAR-10's actual on-disk layout — and `y` as `(N,)` int64
  labels.
- Both loaders raise `ValueError` on a corrupt/unexpected file.

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
`Save`/`Load` round-trip everything needed to resume training exactly
where you left off: layer parameters, optimizer state (unless
`save_opt_state=False`), every per-model hyperparameter, the training-step
counter, in-progress gradient-accumulation buffers, the train/eval mode
flag, and the auto-shape-inference/residual-connection bookkeeping needed
to keep calling `add_*` on the loaded model — plus any `extra_state` dict
you pass in (e.g. a diffusion model's EMA weights). The target model's
layer *shapes* must already match (build the identical architecture before
calling `Load`) — only values are restored, not structure.

```python
# Round-tripping extra state alongside a model (e.g. a diffusion model's
# EMA weights) via extra_state:
model.Save("model.json", extra_state={"my_extra_data": [1, 2, 3]})

model2 = NeuralNet()
model2.add_dense(20, 64, activation="relu")   # same architecture as `model`
model2.add_dense(64, 1, activation="sigmoid")
extra = model2.Load("model.json")
print(extra)  # {"my_extra_data": [1, 2, 3]}
```
`extra_state` is stored as plain JSON/pickle — arrays nested inside it
(like a diffusion model's `ema_weights`) come back as plain nested Python
lists after a `.json` round-trip, not NumPy arrays; wrap them yourself with
`np.array(...)` if you need them as arrays again (a `.pkl` round-trip
preserves the original Python/NumPy types exactly, no reconversion needed).

Loading a file that's missing a given key silently keeps the current
model's value for that setting rather than erroring — a partial/older save
file won't crash `Load`, but also won't warn you about what didn't get
restored.

Other introspection/utility methods: `summary()`, `get_weights()`/
`set_weights()`, `freeze(layer_idx)`/`unfreeze(layer_idx)`,
`check_nan_inf()`, `copy()`, `train()`/`eval()`, `reset_optimizer_state()`
— all covered under [The `NeuralNet` object](#the-neuralnet-object) above.

## Configuration system

Every numeric default that would otherwise be a hardcoded "magic number"
lives in `Enilnets.constants` and can be overridden three ways:

```python
import Enilnets

# 1. Globally, for every model created afterward
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

## Autograd (`Enilnets.graph`)

An **additive** reverse-mode automatic-differentiation engine (roadmap
Phase 1). It coexists with `NeuralNet` and never replaces it: everything
documented above keeps working unchanged, and nothing about `NeuralNet`
uses `graph/` internally. New code can opt into building models where
gradients are derived automatically instead of hand-written per layer.

The graph is **dynamic**: it's recorded by tracing your actual Python
execution as it happens (define-by-run, like PyTorch) — there is no
separate graph-definition step.

```python
import numpy as np
from Enilnets.graph import Tensor, ops

w = Tensor(np.random.randn(3, 2), requires_grad=True)
x = Tensor(np.random.randn(4, 3))          # wraps by reference, never copies

loss = ops.relu(x @ w).sum()               # ordinary math, recorded as a graph
loss.backward()                            # reverse-mode autodiff
print(w.grad)                              # dloss/dw, shape (3, 2)
```

Highlights:

- **Operator sugar**: `+ - * / ** @`, slicing/int-array indexing, and
  method forms (`x.tanh()`, `x.sum(axis=0)`, `x.reshape(...)`) all record
  gradients. Broadcasting works everywhere NumPy allows it; gradients are
  reduced back to each input's shape automatically.
- **`no_grad()`** context manager for inference (no graph, no memory
  overhead); **`Tensor.detach()`** to cut gradients at a point.
- **Backend/precision transparent**: tensors live on whatever backend the
  global `use_gpu()` / `use_float64()` switches selected, same as every
  other module.
- **Free interop with `NeuralNet`**: `t.data` is the raw NumPy/CuPy array
  (never a copy), so graph outputs feed `model.Forward(t.data)` with zero
  boundary cost, and existing weight arrays can seed graph tensors by
  reference.

### Defining a custom op

Every built-in op is two pieces — a forward formula and one local-gradient
rule — and user ops use the identical public API (`custom_op`), so
extending the engine never means editing a central dispatch table:

```python
from Enilnets.graph import custom_op

sqerr = custom_op(
    "sqerr",
    forward=lambda a, b: (a - b) ** 2,                     # raw-array math
    backward=lambda g, out, a, b: (2*g*(a-b), -2*g*(a-b)), # one grad per input
)
loss = sqerr(pred, target).mean()
loss.backward()          # flows through your op like any built-in
```

`backward` receives the upstream gradient `g`, the forward result `out`
(reusable, e.g. sigmoid's `g * out * (1-out)`), and the raw inputs; return
`None` for non-differentiable inputs. Gradient rules for all built-in ops
are verified against finite-difference numerical gradients in the test
suite (`TestGraphOps`), and new ops should get the same treatment.

### Custom layers with automatic gradients

`Layer` is the `graph/` counterpart of the `add_*` builder API: compose
ops in a `forward()` and every parameter's gradient arrives via
`backward()` — no hand-derived backward pass, no dispatch-table entry.
`Linear`, `ReLU`/`Tanh`/`Sigmoid`, `Dropout`, and `Sequential` are
provided; subclass `Layer` for anything else:

```python
from Enilnets.graph import Layer, Parameter, Linear, ReLU, Sequential, Tensor, ops

class Gated(Layer):                       # a custom layer
    def __init__(self, n_in, n_out):
        super().__init__()
        self.w = Parameter(np.random.randn(n_in, n_out) * 0.1)
        self.g = Parameter(np.random.randn(n_in, n_out) * 0.1)
    def forward(self, x):
        return ops.tanh(x @ self.w) * ops.sigmoid(x @ self.g)

model = Sequential(Linear(2, 16), ReLU(), Gated(16, 8), Linear(8, 1))
for _ in range(100):                      # a manual training loop
    model.zero_grad()
    loss = ((model(X) - Tensor(Y)) ** 2).mean()
    loss.backward()
    for p in model.parameters():          # parameters() finds them all
        p.data -= 0.05 * p.grad
```

`parameters()` discovers every `Parameter` in the layer, its sub-layers,
and lists thereof — no registration calls. `train()`/`eval()` toggle
`Dropout`-style behavior, mirroring `NeuralNet.train()`/`.eval()`.

**Interop is free in both directions**: `Linear` uses the same
`(n_out, n_in)` weight convention as `add_dense`, and `Parameter` wraps
arrays by reference — so `Parameter(model.layers[0]["weights"])` shares
(not copies) an existing `NeuralNet`'s weights, and a graph block's
`.data` output feeds `model.Forward(...)` without any boundary copy.

### Named tensors

Optional per-dimension labels — a pure metadata layer (storage and math
untouched) that catches axis mistakes and lets reductions address axes by
meaning:

```python
x = Tensor(data, names=("batch", "time", "feature"))
x.mean(axis=x.axis("time"))        # reduce by name; result: ("batch", "feature")
x + y_with_different_names          # ValueError — the mistake names exist to catch
```

Names propagate where output dims provably correspond to input dims
(elementwise/broadcast ops merge right-aligned, reductions drop the
reduced name, `transpose` permutes); shape-changing ops (`reshape`,
`matmul`, indexing, `concatenate`) conservatively drop them rather than
guess. `set_names(...)` labels in place; `None` entries leave dims
unnamed.

### Complex tensors

`graph/` tensors support complex dtypes end to end (the groundwork for
future FFT/spectral layers): Python complex input coerces to
`complex64`/`complex128` following the float32/float64 default, and
`conj()`, `real()`, `imag()`, `abs()` are differentiable ops. Gradients
follow the PyTorch/JAX convention — for a real-valued loss the stored
gradient is `dL/dRe(z) + 1j·dL/dIm(z)`, so plain gradient descent works
unchanged — and every rule (including complex `matmul`/`mul`/`div`) is
verified against per-part finite differences in the test suite. Real
tensors are entirely unaffected (the conjugations are identities there).

### Mixed precision in `graph/` (autocast)

The `graph/` counterpart of `NeuralNet(use_mixed_precision=True)`, done
the graph-native way: precision changes are **cast ops with defined
gradients**, visible in traces, not a hidden forward-side detail. Inside
`autocast()`, `matmul` runs at float32 even under a float64 default; the
result is cast back, and gradients flow through the casts so master
weights receive float64 gradients. A no-op when float32 is already the
default:

```python
from Enilnets.graph import autocast, cast

with autocast():
    loss = ((model(X) - Tensor(Y)) ** 2).mean()   # matmuls run at float32
loss.backward()                                    # float64 master grads
```

`cast(x, dtype="float32")` is also available directly as an ordinary
differentiable op (its gradient casts back to the input's dtype).

### Gradient checkpointing

Trade compute for memory: `checkpoint(fn, *inputs)` runs a segment
without storing its internal activations, then re-runs the segment's
forward during `backward()` — gradient-identical to the normal path
(pinned by tests to 1e-12):

```python
from Enilnets.graph import checkpoint

h = checkpoint(block1, x)     # block1's interior is NOT kept alive
y = block2(h)                 # block2 behaves normally
y.sum().backward()            # block1 re-runs here, then backprops
```

The segment appears in traces as one composite `checkpoint(...)` node.
Keep checkpointed segments deterministic — a `Dropout` inside one would
draw fresh randomness on recompute (the standard caveat, same as
PyTorch's checkpoint).

### Functional API and lazy layers

`Enilnets.functional` (also `Enilnets.graph.functional`) offers the
stateless spelling of the graph ops — no layer objects, no stored state,
gradients included: `relu`, `softmax`, `linear(x, w, b)`, `dropout`,
`mse_loss`, `cross_entropy(logits, class_indices)` (numerically stable,
log-sum-exp based), and friends:

```python
import Enilnets.functional as F
loss = F.cross_entropy(F.linear(x, w, b), targets)
loss.backward()
```

`LazyLinear(n_out)` defers weight creation until the first call infers
`n_in` from the input — the graph-side counterpart of the `add_*`
builders' auto shape inference. Run one batch through a lazy model
before handing `parameters()` to an optimizer (it's empty until then).

### DropBlock & Stochastic Depth

`DropBlock2D(rate, block_size)` drops contiguous spatial blocks of a
feature map (per-pixel dropout under-regularizes correlated conv
features); survivors rescale to preserve the expectation, `seeds=` is the
reproducibility hook. `StochasticDepth(branch, survival_prob)` wraps a
residual branch and drops it per-example at train time (inverted
rescale), running it always in eval — compose as `y = x + branch(x)`.

### Dropout variants: Gaussian & Alpha

`GaussianDropout(rate)` multiplies by `N(1, rate/(1-rate))` noise at
train time (same expectation as ordinary dropout, smooth instead of hard
zeros). `AlphaDropout(rate)` is the SELU-compatible variant: dropped
units go to SELU's negative saturation value and an affine correction
keeps a self-normalized input at mean 0 / variance 1 (verified
statistically in the tests). Both are identity in eval mode.

### Convolution variants (graph)

`graph.conv2d` / `conv1d` implement the whole item-35 family as one
gather+matmul composite (zero convolution-specific gradient code):
`stride`, symmetric zero `padding`, `dilation`, and `groups`
(`groups=in_ch` = depthwise). `causal_conv1d` left-pads by
`dilation·(k-1)` so `output[t]` sees only inputs `≤ t` (WaveNet-style).
Layer forms: `Conv2D`, `Conv1D(causal=...)`, and `SeparableConv2D`
(depthwise + pointwise, the MobileNet block). The base case is pinned
numerically equal to `add_conv2d`; dilation/groups/causal are verified
against manual references and finite differences.

`conv3d` generalizes the same composite to `(B, C, D, H, W)` volumes, and
`conv_transpose1d/2d/3d` provide transposed ("de")convolution via
zero-stuffing + flipped-kernel conv — weights use the `(in, out, k, …)`
transposed convention, `output_padding` recovers sizes a strided conv
discarded, and each is the **exact adjoint** of its forward conv
(`⟨conv(x,w), y⟩ = ⟨x, convᵀ(y,w)⟩`, pinned across stride/padding
combinations).

### Adaptive & fractional pooling, MaxUnpool

`adaptive_avg_pool2d` / `adaptive_max_pool2d(x, (oh, ow))` pool any input
size to a fixed output grid (PyTorch-style bin edges).
`fractional_max_pool2d` (Graham 2014) places the bin boundaries
pseudo-randomly (`random_u=(uh, uw)` fixes the draws for
reproducibility). `max_pool2d_with_indices` returns the argmax positions
that `max_unpool2d(values, indices, (H, W))` uses to invert the pooling
(values return to their exact argmax cells, zeros elsewhere). All are
composites of existing differentiable ops, FD-verified; `max_pool2d`
matches `add_maxpool2d`'s non-overlapping semantics (pinned).

### PixelShuffle / PixelUnshuffle

Sub-pixel rearrangement for upsampling decoders:
`functional.pixel_shuffle(x, r)` turns `(B, C·r², H, W)` into
`(B, C, H·r, W·r)` (layer forms `PixelShuffle(r)` / `PixelUnshuffle(r)`).
Pure reshape/transpose composites — exact round trip, gradients are the
inverse rearrangement (pinned in tests).

### Padding modes

`ops.pad` (layer form: `Pad`) pads any-rank tensors with full gradient
support in four modes: `"constant"` (zeros or a value), `"reflect"`,
`"edge"` (replication), `"wrap"` (circular). Reflection/edge/wrap copy
source cells into the padding, and each source cell's gradient correctly
sums over every copy (FD-verified per mode):

```python
from Enilnets.graph import Pad, ops
y = ops.pad(x, pad_width=((0,0), (0,0), (1,1), (1,1)), mode="reflect")  # NCHW spatial pad
```

### Variable-length sequences: padding masks & packed sequences

The graph API settles the variable-length-batch representation (and
closes the "attention has no padding-mask support" gap graph-side; the
`nn/` layers are unchanged):

```python
from Enilnets.graph import (lengths_to_mask, pack_padded, pad_packed,
                            MultiHeadAttention, masked_mean)

mask = lengths_to_mask([5, 3, 2], max_len=5)   # (batch, seq) bool, True = real
attn = MultiHeadAttention(embed_dim=64, num_heads=4)  # causal/num_kv_heads/window_size too
h = attn(x, key_padding_mask=mask)             # padded keys get ~0 attention
pooled = masked_mean(h, mask)                  # padding-aware mean pooling

packed = pack_padded(x, lengths)               # (total_tokens, F), no padding
restored = pad_packed(packed)                  # differentiable round trip
```

`MultiHeadAttention` uses the same `Wq..Wo` weight conventions as
`add_multihead_attention` (output equivalence with shared weights is
pinned in the tests), so weights move freely between the two paths. It
takes the same `num_kv_heads` (MHA/MQA/GQA) and `window_size`
(sliding-window) settings, with matching weight shapes, so those layers are
shareable across the two paths too.
Packing/unpacking are built from differentiable gathers — gradients flow
through, and padding slots receive exactly zero gradient.

### Tracing and exporting the graph

The dynamic graph a computation records can be exported as a symbolic
structure — introspect it, print it, or re-execute it on new inputs
(`symbolic_trace` marks the example inputs as placeholders):

```python
from Enilnets.graph import symbolic_trace, ops

def f(x):
    return ops.relu(x @ w + b)

graph = symbolic_trace(f, example_x)   # runs f once, captures the op graph
print(graph)                            # id / kind / op / parents / shape table
y = graph.run(new_x)                    # re-execute with a fresh input
```

Captured weights become constant nodes holding **references** (not
snapshots) — re-running after a training step sees the updated weights.
Python control flow is baked in as traced (one branch, loops unrolled),
the standard trade-off of trace-based export. `trace(output_tensor)` does
the same for a graph you already have in hand.

### Optimizing a traced graph

Traced graphs can be transformed before re-execution — `optimize` applies
dead-node elimination, constant folding (input-independent subgraphs are
precomputed; note folded values are snapshots of the weights at optimize
time), and elementwise-chain fusion (`relu(tanh(exp(x)))` collapses to one
fused node), each preserving `run()` results exactly:

```python
from Enilnets.graph import symbolic_trace, optimize

graph = symbolic_trace(f, example_x)
fast = optimize(graph)                 # fewer nodes, same outputs
y = fast.run(new_x)
```

The individual passes (`eliminate_dead_nodes`, `fold_constants`,
`fuse_elementwise`) are importable separately. Eager Tensor math is never
affected — these operate only on the exported symbolic representation.

## Package layout

As of the Phase 0 reorganization (see `ROADMAP.md`), the library is split
into topic-scoped subpackages instead of one flat directory:

```
Enilnets/
├── core/           # NeuralNet skeleton (base.py), NumPy/CuPy backend switch
│                   #   (backend.py), constants.py, general utils.py
├── nn/             # layer builders, forward/backward dispatch, activations,
│                   #   weight init, training loop (train.py), Save/Load (io.py)
├── optim/          # optimizer update rules (optimizer.py)
├── losses/         # ComputeLoss (loss.py)
├── metrics/        # eval_metrics.py, eval_utils.py
├── generative/     # VAE/GAN/Diffusion/Flow/EBM/UNet/TextGenerator
├── reinforcement/  # Evolve/Reinforce/PPO/ActorCritic, compute_returns, gae
├── evolving/       # NEAT (neat.py)
├── vision/         # image_utils.py (I/O, resize, patches)
├── text/           # text_utils.py (Tokenizer, windows, one-hot)
├── audio/          # audio_utils.py (WAV I/O, STFT/mel)
├── datasets/       # loaders.py (load_mnist, load_cifar10)
├── preprocessing/  # augmentation/normalization transforms (image/audio/text)
├── visualization/  # plotting.py (plot_network, plot_genome, to_html)
└── crossmodal/     # crossmodal_utils.py (contrastive loss, fusion)
```

**Every pre-reorganization import path still works.** The old flat module
names (`Enilnets.base`, `Enilnets.backend`, `Enilnets.layers`,
`Enilnets.neat`, `Enilnets.image_utils`, ...) are registered as aliases of
their new homes, so both spellings below are the same module object:

```python
from Enilnets.neat import Genome            # old flat path — still works
from Enilnets.evolving.neat import Genome   # new canonical path
```

New code should prefer the subpackage paths; the aliases exist so nothing
existing breaks.

## Full API index

Everything importable as `from Enilnets import X`:

```
NeuralNet, LRScheduler, KVCache, cached_forward_step, BPETokenizer,
prune_magnitude, prune_channels, PruningSchedule, sparsity,
quantize_weights, ActivationCalibrator,
EMA, SWA, find_learning_rate,
DataLoader, Dataset, IterableDataset, ArrayDataset, MemmapDataset,
StreamingDataset, Subset, ConcatDataset, random_split, Compose,
VAE, GAN, DiffusionModel, AutoregressiveModel, RealNVP, EnergyBasedModel,
UNetDenoiser, time_embedding, TextGenerator, Tokenizer,
reparameterize, langevin_dynamics, gaussian_sample, uniform_sample,
gumbel_softmax_sample, random_mask, top_p_sampling, top_k_sampling,
kl_divergence_gaussian, adversarial_loss_discriminator,
adversarial_loss_generator, diffusion_loss, nll_loss, energy_loss,
compute_returns,
set_seed, train_test_split, iterate_minibatches, count_parameters,
EarlyStopping, one_hot, k_fold_split, constants,
ModelCheckpoint, CSVLogger, JSONLogger,
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
`plot_genome`/`to_html` re-exported), `Enilnets.datasets` (`load_mnist`/
`load_cifar10`), and within `Enilnets.generative`:
`Enilnets.generative.sampling.gae`, `Enilnets.generative.pretrained`
(`build_vgg16_feature_extractor`); and the autograd package
`Enilnets.graph` (`Tensor`, `custom_op`, `Layer`/`Linear`/`Sequential`,
`trace`/`symbolic_trace`, `optimize` — see
[Autograd](#autograd-enilnetsgraph)).

## Known limitations

A consolidated list of every rough edge documented above, so you can decide
up front whether they matter for your use case:

- **GPU support is opt-in, not automatic.** NumPy/CPU is the default;
  CuPy/GPU requires installing CuPy yourself and calling
  `Enilnets.use_gpu(True)` — see [GPU mode (optional)](#gpu-mode-optional).
  Even with GPU mode on, this is still a from-scratch, readable
  implementation, not a fused/optimized kernel library — it trades some
  speed for full transparency/hackability either way.
- **float32 is the default working precision, not float64** — call
  `Enilnets.use_float64(True)` if you need the old default back, see
  [Precision: float32 vs float64](#precision-float32-vs-float64). Numerical
  work sensitive to float32's ~7 decimal digits (e.g. hand-rolled
  finite-difference gradient checks against this library) should opt into
  float64 explicitly rather than assume it.
- **`UNetDenoiser`'s time embedding often silently doesn't reach most
  blocks** — see [UNetDenoiser](#unetdenoiser) above.
- **`vgg_loss`/`build_vgg16_feature_extractor` need your own pretrained
  weights** — see [Bring your own pretrained weights](#bring-your-own-pretrained-weights).
- **`eval_utils.inception_score(classifier=None)` is a crude k-means-based
  proxy**, not a real Inception-v3-based Inception Score — pass your own
  classifier for anything resembling the standard metric.
- **`add_conv2d(padding="same")` only covers `stride=1` + odd kernel size**
  — raises `ValueError` for `stride>1` or even `k`; pad your input manually
  (`image_utils.pad_image`) for those cases.
- **RNN/LSTM/GRU have no fused/vectorized-across-time implementation** — the
  per-timestep Python loop is the unavoidable cost of a from-scratch
  recurrent layer; expect these to be the slowest layer type for long
  sequences (applies to the bidirectional variants too — roughly 2x the
  cost of a single direction).
- **`AutoregressiveModel.generate()`/`.complete()` cannot use a KV-cache** —
  it is a MADE-style *masked-dense* model, not an attention stack: each
  dimension's conditional is produced by a full masked forward pass over the
  whole (partially filled) vector, so there is no per-position K/V to reuse.
  Generation is inherently O(data_dim) forward passes; slow for
  high-dimensional data. `Enilnets.cached_forward_step` applies to attention
  stacks only (see [KV-cache decoding](#kv-cache-decoding)).
- **`DiffusionModel.sample()`/`.denoise()` use standard ancestral sampling**
  (O(`time_steps`) sequential steps) — use `sample_ddim()` instead for
  fast generation.
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
  side (structurally accurate — there genuinely isn't a single weight
  matrix to draw there).
- **`TextGenerator.generate()` only supports a single prompt at a time**
  (no batched generation) — the token *sampler* is scalar, so the loop runs
  one stream. The underlying cache mechanism itself is fully batched: call
  `cached_forward_step` directly (see [KV-cache decoding](#kv-cache-decoding))
  if you want to decode several streams at once.
- **The graph `MultiHeadAttention` has no `attention_kernel`** (linear /
  Performer live on the `nn/` path only) — the causal linearized form needs
  a prefix-sum (`cumsum`) op with a gradient rule, which `graph/ops.py`
  does not have yet. Build linearized stacks with `add_transformer_block`.
- **`add_multihead_attention`/`add_cross_attention` have no padding-mask
  support** for batched variable-length sequences — only `causal=True|False`
  exists; pack sequences to a fixed length instead.
- **`activate("gelu", ...)` is the tanh approximation only** — no exact
  erf-based variant.
- **`nn/io.py`'s `.pkl` save/load path uses Python's `pickle`** — only load
  `.pkl` checkpoints you trust or produced yourself (arbitrary code
  execution risk on untrusted files, same as any use of `pickle.load`);
  use `.json` if loading from an untrusted source is a real scenario.

## Running the test suite

`test_enilnets.py` is a single unittest-based suite (393 tests) covering
every layer/model/optimizer/utility above, plus a benchmark harness:

```bash
python test_enilnets.py                      # run all correctness tests
python test_enilnets.py -v                   # verbose
python test_enilnets.py TestRecurrentLayers   # run one class
python test_enilnets.py --benchmark          # timing only (Benchmark* classes)
```

Every gradient-bearing feature (attention, residual connections, RNN/LSTM/
GRU BPTT, KV-cache, NEAT crossover/cycle-safety) is checked against
finite-difference numerical gradients or explicit invariant checks, not
just "does it run" smoke tests.

## Contributing

Issues and pull requests are welcome at
https://github.com/docenilno/Enilnets. This project has one hard rule:
**zero new dependencies, ever** (pure NumPy + Python standard library) —
any contribution that would require installing something else to use this
library will be declined regardless of how useful it is. Any new
gradient-bearing feature should come with a finite-difference numerical
gradient check, following the pattern used throughout `test_enilnets.py`.

## License

MIT — see the [GitHub repository](https://github.com/docenilno/Enilnets)
for the full license text.
