# Enilnets

GitHub: https://github.com/docenilno/Enilnets

Enilnets is a neural network library built entirely on NumPy — no PyTorch,
no TensorFlow, no GPU. Every layer, optimizer, generative model, and training
loop is implemented from scratch with plain array math, so you can read every
line of what's actually happening to your gradients.

It covers a lot of ground for a NumPy-only library: dense/conv/attention/
recurrent layers with a Keras-like sequential API, a full suite of generative
models (VAE, GAN, Diffusion, Normalizing Flows, Energy-Based Models,
autoregressive pixel models), a GPT-style causal transformer for text
generation, policy-gradient reinforcement learning, NEAT neuroevolution, and
a set of NumPy-native data utilities for images, audio, and text.

**Version 3.0.0.**

## Table of contents

- [Install](#install)
- [Quickstart](#quickstart)
- [Building networks: the layer API](#building-networks-the-layer-api)
- [Auto shape inference](#auto-shape-inference)
- [Convenience block builders](#convenience-block-builders)
- [Residual / skip connections](#residual--skip-connections)
- [Recurrent layers: RNN / LSTM / GRU](#recurrent-layers-rnn--lstm--gru)
- [Attention & Transformers](#attention--transformers)
- [Losses](#losses)
- [Optimizers](#optimizers)
- [Training](#training)
- [Gradient accumulation](#gradient-accumulation)
- [Mixed precision](#mixed-precision)
- [Text generation](#text-generation)
- [Generative models](#generative-models)
- [Reinforcement learning](#reinforcement-learning)
- [NEAT (neuroevolution)](#neat-neuroevolution)
- [Evaluation utilities](#evaluation-utilities)
- [Data utilities](#data-utilities)
- [Visualization](#visualization)
- [Model persistence](#model-persistence)
- [Configuration system](#configuration-system)
- [Running the test suite](#running-the-test-suite)

## Install

Enilnets has no dependencies beyond NumPy. Install it from PyPI:

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

# Toy binary classification data
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

`add_dense(n_in, n_out, activation=...)` is the core building block; every
`add_*` method appends a layer dict to `model.layers` and updates internal
book-keeping used for auto shape inference (see below). `Train` handles
minibatching, shuffling, optional validation tracking, learning-rate
scheduling and early stopping in one call; `TrainBatch` is the single-batch
primitive underneath it if you want your own loop.

## Building networks: the layer API

Every layer method lives on `NeuralNet` and returns nothing — it mutates
`model.layers` in place, so you chain calls rather than assign them.

```python
model = NeuralNet(learning_rate=0.001, optimizer="adam")

model.add_dense(n_in=784, n_out=256, activation="relu", init_method="he_normal")
model.add_batchnorm(num_features=256)
model.add_dropout(rate=0.3)
model.add_dense(256, 128, activation="gelu")
model.add_dense(128, 10, activation="softmax")
```

Layer types available:

| Method | Purpose |
|---|---|
| `add_dense(n_in, n_out, activation, init_method, use_bias, activation_params)` | Fully connected layer |
| `add_sparse(n_in, n_out, connectivity, activation, init_method)` | Dense layer with a fixed random sparsity mask |
| `add_conv2d(in_ch, out_ch, k, activation, stride, init_method, input_size)` | 2D convolution (im2col-based) |
| `add_flatten()` | (B, C, H, W) → (B, C\*H\*W) |
| `add_maxpool2d(pool_size)` / `add_avgpool2d(pool_size)` | Spatial pooling |
| `add_global_avgpool2d()` | (B, C, H, W) → (B, C, 1, 1) |
| `add_upsample2d(scale_factor)` | Nearest-neighbor spatial upsampling |
| `add_batchnorm(num_features, epsilon, momentum)` | Batch normalization (2D/4D input) |
| `add_layernorm(normalized_shape, epsilon)` | Layer normalization (2D/3D/4D input) |
| `add_dropout(rate)` | Inverted dropout |
| `add_embedding(vocab_size, embed_dim, init_method)` | Token embedding lookup |
| `add_multihead_attention(embed_dim, num_heads, dropout, causal, init_method)` | Multi-head self-attention |
| `add_positional_encoding(max_seq_len, embed_dim, learnable, base)` | Learnable or sinusoidal positions |
| `add_transformer_block(embed_dim, num_heads, mlp_ratio, dropout, activation, causal)` | Full pre-norm residual transformer block |
| `add_rnn` / `add_lstm` / `add_gru` | Recurrent layers with BPTT (see below) |
| `add_residual_start()` / `add_residual_end()` | Generic skip connections (see below) |
| `add_vision_transformer_patch_embed(img_size, patch_size, in_channels, embed_dim)` | ViT patch embedding |

Activations: `relu`, `leakyrelu`, `elu`, `selu`, `gelu`, `swish`, `mish`,
`sigmoid`, `tanh`, `softmax`, `softplus`, `linear`. Most take extra tunables
via `activation_params={"alpha": ..., "sigmoid_clip": ...}` on the layer that
uses them (defaults come from `Enilnets.constants`).

Weight init methods (`init_method=`): `xavier_uniform`, `xavier_normal`,
`he_uniform`, `he_normal`, `normal`, `orthogonal`, `zeros`, `ones`.

## Auto shape inference

You only need to specify the input size once. Every `add_*` method infers
`n_in`/`in_ch` from the previous layer if you omit it (or leave it `None`):

```python
model = NeuralNet()
model.add_dense(784, 256, activation="relu")   # first layer: n_in required
model.add_dense(None, 128, activation="relu")  # or just add_dense(128, ...)
model.add_dense(64, activation="softmax")      # n_out-only call also works
```

The same applies across conv/pool/flatten boundaries:

```python
model = NeuralNet()
model.add_conv2d(in_ch=3, out_ch=16, k=3, input_size=(32, 32))
model.add_maxpool2d(2)
model.add_conv2d(out_ch=32, k=3)   # in_ch inferred as 16
model.add_flatten()                # width inferred from (C, H, W)
model.add_dense(10, activation="softmax")  # n_in inferred from flatten
```

`input_size=(H, W)` only needs to be given once, on the first conv layer, so
that `add_flatten()` downstream knows the final spatial size.

## Convenience block builders

For stacks of identical layers, two helpers save you the loop:

```python
model = NeuralNet()
model.add_mlp_block([256, 128, 64], in_dim=784, out_dim=10,
                    activation="relu", out_activation="softmax")
# equivalent to: dense(784,256) -> dense(256,128) -> dense(128,64) -> dense(64,10,softmax)

model2 = NeuralNet()
model2.add_conv_block(out_ch=32, k=3, in_ch=3, batchnorm=True, pool="max",
                      input_size=(64, 64))
model2.add_conv_block(out_ch=64, k=3, batchnorm=True, pool="max")
# each call = conv2d -> batchnorm -> maxpool2d
```

## Residual / skip connections

`add_residual_start()`/`add_residual_end()` wrap any block of layers in
`x = x + block(x)`, exactly like a Transformer or ResNet skip connection.
They're nestable, so you can wrap sub-blocks inside larger residual blocks.

```python
model = NeuralNet(learning_rate=0.001, optimizer="adam")
model.add_dense(64, 64, activation="linear")
model.add_residual_start()
model.add_dense(64, 64, activation="tanh")
model.add_dense(64, 64, activation="linear")
model.add_residual_end()          # x = x + tanh_block(x)
model.add_dense(64, 10, activation="softmax")
```

`add_transformer_block` is built entirely out of this primitive
(`residual_start -> layernorm -> attention -> residual_end`, then
`residual_start -> layernorm -> MLP -> residual_end`), so any custom block
you wrap this way gets the same well-conditioned gradient flow.

## Recurrent layers: RNN / LSTM / GRU

All three take `(batch, seq_len, features)` input and support
`return_sequences=True` (output every timestep, shape `(B, S, hidden)`) or
`False` (output only the last timestep, shape `(B, hidden)`). Backprop is
full backpropagation-through-time (BPTT), verified against finite-difference
gradients over multi-timestep sequences.

```python
model = NeuralNet(learning_rate=0.001, optimizer="adam")
model.add_lstm(n_in=32, hidden_dim=128, return_sequences=True)
model.add_lstm(hidden_dim=64, return_sequences=False)  # n_in auto-inferred as 128
model.add_dense(64, 10, activation="softmax")

x = np.random.randn(16, 20, 32)  # (batch, seq_len, features)
out = model.Forward(x, training=True)  # (16, 10)
```

`add_rnn` uses a single tanh gate. `add_lstm` stacks `[i, f, g, o]` gates
into one `(4*hidden, ·)` matrix (forget-gate bias initialized to 1, the
standard trick for better early-training gradient flow). `add_gru` stacks
`[r, z, n]` gates into a `(3*hidden, ·)` matrix with separate input/hidden
biases, since the reset gate only multiplies the hidden contribution to the
candidate state.

## Attention & Transformers

```python
model = NeuralNet(learning_rate=3e-4, optimizer="adam")
model.add_embedding(vocab_size=5000, embed_dim=128)
model.add_positional_encoding(max_seq_len=256, learnable=False)
for _ in range(4):
    model.add_transformer_block(embed_dim=128, num_heads=8, mlp_ratio=4.0,
                                dropout=0.1, activation="gelu", causal=True)
model.add_layernorm()
model.add_dense(None, 5000, activation="softmax")
```

`causal=True` on attention (or on `add_transformer_block`) applies an
autoregressive mask so position *i* can only attend to positions `<= i` —
this is what `TextGenerator` uses under the hood.

## Losses

`ComputeLoss(output, target, function="mse", reduction="mean", **kwargs)`
and the matching gradients in `Backward(targets, loss_function=..., **kwargs)`
support:

`mse`, `mae`, `huber` (`delta=`), `smooth_l1` (`beta=`),
`binary_cross_entropy`, `cross_entropy`/`categorical_cross_entropy`,
`focal` (`alpha=`, `gamma=`), `hinge`, `bce_logits`, `wasserstein`,
`cosine_similarity`, `triplet` (`negative=`, `margin=`), `ntxent`
(`temperature=`), `kl_divergence` (mu/logvar losses — compute this one
manually, see `generative/vae.py`, since it isn't a plain output-layer loss).

```python
out = model.Forward(x, training=True)
loss = model.ComputeLoss(out, y, function="cross_entropy")
model.Backward(y, loss_function="cross_entropy")
model.update()
```

**Reduction convention** (matters if you write your own reference/finite-
difference checks against this library): under `reduction="mean"`,
elementwise losses (`mse`, `mae`, `huber`, `bce`, `focal`, `hinge`,
`bce_logits`, `wasserstein`, ...) average over *every* element
(`batch_size * n_features`). Losses that already reduce over the feature
axis in their own formula (`cross_entropy`, `cosine_similarity`, `triplet`,
`ntxent`) divide by `batch_size` only. `reduction="sum"` skips normalization
entirely for any loss.

Where a loss+activation pair has a canonical simplified gradient (softmax +
cross-entropy, sigmoid + BCE, linear + `bce_logits`/`wasserstein`), Enilnets
uses that closed form directly instead of chaining the activation derivative
through a separate loss derivative.

## Optimizers

Set via `NeuralNet(optimizer=..., learning_rate=..., l2_lambda=..., momentum=...)`:

- `"sgd"` — momentum SGD (`momentum=`)
- `"rmsprop"` — `rmsprop_decay=`, `rmsprop_epsilon=`
- `"adagrad"` — `adagrad_epsilon=`
- `"adam"` — `adam_beta1=`, `adam_beta2=`, `adam_epsilon=`; L2 weight decay is
  coupled into the gradient before the Adam update (classic Adam+L2)
- `"adamw"` — identical moment updates to Adam, but weight decay is
  **decoupled**: applied directly to the weights after the Adam step
  (`w -= lr * l2_lambda * w`), rather than folded into the gradient. This
  means AdamW still decays weights even when the gradient is exactly zero.

All hyperparameters default from `Enilnets.constants` and can be overridden
per-model:

```python
model = NeuralNet(optimizer="adamw", learning_rate=1e-3, l2_lambda=0.01,
                  adam_beta1=0.9, adam_beta2=0.95)
```

Gradient clipping is automatic whenever `grad_clip_norm > 0`:

```python
model = NeuralNet(optimizer="adam", grad_clip_norm=1.0)
model.TrainBatch(x, y)  # Backward() -> clip_gradients(1.0) -> update(), automatically
```

## Training

`TrainBatch(xs, ys, loss_function=None, accumulation_steps=1, **loss_kwargs)`
runs one batch end to end (forward, loss, backward, optional clipping,
optimizer step). `Train(...)` wraps it in an epoch/minibatch loop with
optional validation tracking, an `LRScheduler`, and `EarlyStopping`:

```python
from Enilnets import LRScheduler, EarlyStopping

scheduler = LRScheduler(initial_lr=1e-3, mode="warmup_cosine",
                        warmup_epochs=5, max_epochs=50)
early_stopping = EarlyStopping(patience=5, mode="min")

history = model.Train(X_train, Y_train, epochs=50, batch_size=64,
                      X_val=X_val, Y_val=Y_val, loss_function="mse",
                      scheduler=scheduler, early_stopping=early_stopping)
```

`LRScheduler` modes: `"step"` (`drop=`, `epochs_drop=`), `"exponential"`
(`decay=`), `"cosine"` (`max_epochs=`), `"warmup_cosine"` (`warmup_epochs=`,
`max_epochs=`).

If you need full control, drop to the primitives directly:

```python
out = model.Forward(x, training=True)
model.Backward(y, loss_function="mse")
model.update()
```

## Gradient accumulation

Simulate a larger batch size without the memory cost, by accumulating
gradients over several microbatches before applying them:

```python
model.Train(X_train, Y_train, epochs=10, batch_size=16, accumulation_steps=4)
# effective batch size 64, 16 samples in memory at a time
```

Or drive it manually with the lower-level primitives:

```python
model.Forward(x1, training=True); model.Backward(y1); model.accumulate_gradients()
model.Forward(x2, training=True); model.Backward(y2); model.accumulate_gradients()
model.apply_accumulated_gradients()  # averages and applies both steps at once
```

`compute_gradients()`/`apply_gradients(grads)` are the pure split behind all
of this — `update()` is just `apply_gradients(compute_gradients(self))` —
useful if you want to inspect or modify gradients before they're applied.

## Mixed precision

`use_mixed_precision=True` runs the dense/conv2d matmuls (the hottest path,
per the benchmark suite) in float32 while keeping master weights at
float64, for a real BLAS speedup:

```python
model = NeuralNet(optimizer="adam", use_mixed_precision=True)
```

This is a lightweight CPU approximation of AMP — there's no hardware
tensor-core path or loss scaling here, just a smaller compute dtype on the
matmul itself. Expect outputs close to (not bit-identical to) the float64
path.

## Text generation

`TextGenerator` builds a GPT-style causal transformer (embedding + positional
encoding + causal transformer blocks + final layernorm + softmax) out of the
primitives above, and trains it by next-token prediction.

```python
from Enilnets import TextGenerator, Tokenizer

corpus = open("corpus.txt").read()
tokenizer = Tokenizer(vocab_size=2000, level="char").fit([corpus])

gen = TextGenerator(tokenizer, embed_dim=128, num_heads=4, num_layers=4,
                    max_seq_len=128, learning_rate=3e-4, optimizer="adam")

gen.Train([corpus], epochs=20, batch_size=32, seq_len=64, verbose=True)

# Sampling strategies
print(gen.generate(prompt="once upon a", max_new_tokens=200, greedy=True))
print(gen.generate(prompt="once upon a", max_new_tokens=200,
                   temperature=0.8, top_p=0.9))
print(gen.generate(prompt="once upon a", max_new_tokens=200,
                   temperature=0.8, top_k=40))

# Beam search
print(gen.generate_beam(prompt="once upon a", beam_width=5, max_new_tokens=100))

# Held-out evaluation
print("perplexity:", gen.perplexity(held_out_text))
```

`generate(..., use_cache=True)` (the default) decodes with a KV-cache: only
the new token's query is computed each step, with past keys/values cached
and reused, turning generation from O(n²) into O(n) over the sequence
length. It's verified to produce identical probabilities to the
non-cached `use_cache=False` path (useful as a correctness cross-check, or
if you've hand-modified the network to something the cache's fast path
doesn't understand — it only supports the standard embedding →
positional-encoding → transformer-block(s) → layernorm → dense architecture
this class builds).

`Tokenizer(vocab_size, level="char"|"word")` handles vocabulary building
(`.fit(texts)`), encoding (`.encode(text, add_special_tokens=...)`), and
decoding (`.decode(ids, skip_special=...)`), with automatic start/end tokens.

## Generative models

All of these live under `Enilnets.generative` (also re-exported from the
top-level `Enilnets` package) and follow the same shape: build with
hyperparameters, call `.Train(X_train, epochs=..., batch_size=...)`, then
`.generate(...)`/`.sample(...)`.

**VAE** — variational autoencoder:
```python
from Enilnets import VAE
vae = VAE(input_dim=784, latent_dim=32, encoder_hidden=[256, 128])
vae.Train(X_train, epochs=30, batch_size=64, kl_weight=1.0)
samples = vae.generate(n_samples=16)
```

**GAN** — supports `bce`, `bce_logits`, and Wasserstein losses:
```python
from Enilnets import GAN
gan = GAN(latent_dim=64, data_dim=784, loss_type="wasserstein",
         wgan_clip_value=0.01)
gan.Train(X_train, epochs=100, batch_size=64, d_steps=5, g_steps=1)
samples = gan.sample(16)
print("mode collapse score:", gan.mode_collapse_score())  # 0=collapsed, 1=diverse
```

**DiffusionModel** — DDPM-style denoising diffusion, with EMA-smoothed
sampling weights:
```python
from Enilnets import DiffusionModel
diffusion = DiffusionModel(data_shape=(784,), time_steps=1000,
                           beta_schedule="cosine", use_ema=True)
diffusion.Train(X_train, epochs=50, batch_size=64)
samples = diffusion.sample(n_samples=16)
```
For image data, pass `denoiser_type="conv"` with `data_shape=(C, H, W)`, or
build your own denoiser with `UNetDenoiser(in_ch, base_ch=64, ch_mult=(1,2,4))`
and its matching `time_embedding(t, dim)` helper.

**RealNVP** — normalizing flow (exact likelihood, invertible):
```python
from Enilnets import RealNVP
flow = RealNVP(data_dim=2, n_coupling=6, hidden_dim=128)
flow.Train(X_train, epochs=50, batch_size=128)
samples = flow.sample(n_samples=500)
```

**EnergyBasedModel** — trained via (persistent) contrastive divergence with
Langevin dynamics for both negative sampling and generation:
```python
from Enilnets import EnergyBasedModel
ebm = EnergyBasedModel(data_dim=2, persistent_cd=True)
ebm.Train(X_train, epochs=50, batch_size=64, n_cd_steps=20)
samples = ebm.sample(n_samples=500, n_steps=200)
```

**AutoregressiveModel** — pixel-by-pixel (or feature-by-feature)
autoregressive generation, discrete or continuous:
```python
from Enilnets import AutoregressiveModel
ar = AutoregressiveModel(data_dim=784, discrete=True, num_classes=256)
ar.Train(X_train, epochs=30, batch_size=64)
samples = ar.generate(n_samples=16)
print("log-likelihood:", ar.log_prob(X_val).mean())
```

Lower-level building blocks used by (or alongside) the models above are also
exported: `reparameterize`, `langevin_dynamics`, `gaussian_sample`,
`uniform_sample`, `gumbel_softmax_sample`, `random_mask`, `top_p_sampling`,
`top_k_sampling`, plus the raw losses `kl_divergence_gaussian`,
`adversarial_loss_discriminator`, `adversarial_loss_generator`,
`diffusion_loss`, `nll_loss`, `energy_loss`.

## Reinforcement learning

Policy-gradient methods hang directly off any `NeuralNet` used as a policy
(and, for `ActorCritic`, a second `NeuralNet` used as a value function):

```python
from Enilnets import compute_returns

policy = NeuralNet(learning_rate=0.001, optimizer="adam")
policy.add_dense(state_dim, 64, activation="relu")
policy.add_dense(64, n_actions, activation="softmax")

returns = compute_returns(rewards, gamma=0.99)
policy.Reinforce(states, actions, returns, action_type="discrete")
```

- `Reinforce(states, actions, returns, action_type, std, normalize_returns)` —
  vanilla REINFORCE (discrete softmax actions or continuous Gaussian actions
  with fixed `std`).
- `PPO(states, actions, old_log_probs, advantages, action_type, ...)` —
  clipped-objective proximal policy optimization.
- `ActorCritic(states, actions, returns, values, action_type, std)` —
  advantage actor-critic, `values` from a separate critic `NeuralNet`.
- `Evolve(inputs, score_fn, noise, tries, sigma)` — gradient-free evolution
  strategy: perturbs weights, scores each candidate with `score_fn`, keeps
  the best. Useful when you don't have (or don't trust) a gradient signal.
- `compute_returns(rewards, gamma)` / `gae(rewards, values, gamma, lambda_)`
  (standalone functions, not `NeuralNet` methods; `gae` lives in
  `Enilnets.generative.sampling`) — discounted returns and Generalized
  Advantage Estimation.

## NEAT (neuroevolution)

`NEATPopulation` implements NeuroEvolution of Augmenting Topologies: instead
of gradient descent on a fixed architecture, a population of small networks
("genomes") evolves via mutation (perturb weights, add a connection, add a
node) and crossover, guided only by a fitness function you supply — no
gradients required. It doesn't build on `NeuralNet`, since a NEAT genome's
topology (which nodes exist, how they're wired) changes over evolution
rather than being fixed up front; a `Genome` is its own small feedforward
network with its own `.forward(x)`.

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
print(len(best.nodes), len(best.connections))  # topology grown beyond the minimal start
```

XOR is the classic NEAT sanity check: it isn't linearly separable, so a
population that starts fully-connected with no hidden units must actually
*discover* a hidden node (via the add-node mutation) to solve it — a good
signal that speciation and structural mutation are both doing real work,
not just tuning weights. Like any evolutionary search, convergence speed
varies by seed and hyperparameters; give it more generations, a larger
population, or both if a given run hasn't fully separated all four cases
yet (`test_enilnets.py`'s `TestNEAT.test_population_evolves_and_solves_xor`
solves it exactly within 100 generations at `population_size=150`).

Population-level knobs (all overridable, defaults live in `Enilnets.constants`
as `NEAT_*`): `compatibility_threshold`/`c1`/`c2`/`c3` (speciation distance:
excess-gene, disjoint-gene, and weight-difference coefficients),
`weight_mutate_rate`/`weight_perturb_rate`/`weight_perturb_power`,
`add_connection_rate`/`add_node_rate`, `crossover_rate`,
`survival_threshold` (fraction of each species allowed to reproduce),
`stagnation_limit` (generations without improvement before a species stops
being prioritized), `elitism` (top genomes carried over unchanged each
generation).

For direct genome manipulation (custom evolutionary loops, saving/loading a
single evolved genome, etc.), the building blocks are also importable from
`Enilnets.neat`: `Genome` (`.forward`, `.copy`, `.mutate_weights`,
`.mutate_add_connection`, `.mutate_add_node`, `.distance`),
`crossover(fitter, other)`, and `InnovationTracker` (tracks the historical
markings that keep independently-discovered structural mutations comparable
across the population).

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

`Enilnets.eval_utils` adds generative-model-focused metrics:
`inception_score`, `frechet_distance`/`compute_fid`, `reconstruction_error`,
`sample_diversity`, `nearest_neighbor_accuracy`.

`model.compute_accuracy(preds, targets)` and
`model.compute_precision_recall_f1(preds, targets)` (binary) are also
available directly on any `NeuralNet` for quick checks during training.

## Data utilities

**Splitting & batching** (`Enilnets.utils`):
```python
from Enilnets import train_test_split, iterate_minibatches, k_fold_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, seed=0)

for X_batch, Y_batch in iterate_minibatches(X_train, Y_train, batch_size=32):
    model.TrainBatch(X_batch, Y_batch)

for X_tr, X_val, Y_tr, Y_val in k_fold_split(X, Y, k=5, seed=0):
    ...  # train/evaluate one fold
```

Also in `Enilnets.utils`: `set_seed(seed)`, `count_parameters(model)` (the
programmatic counterpart to `model.summary()`), `one_hot(indices, n_classes)`,
and `EarlyStopping(patience, min_delta, mode)`.

**Images** (`Enilnets.image_utils`): `load_ppm`/`save_ppm`,
`load_pgm`/`save_pgm`, `load_raw_binary`/`save_raw_binary`,
`rgb_to_grayscale`/`grayscale_to_rgb`, `resize_nearest_neighbor`,
`resize_bilinear`, `image_augmentation` (flips/rotation/etc.),
`normalize_images`/`denormalize_images`, `images_to_patches`, `pad_image`.

**Audio** (`Enilnets.audio_utils`): `load_wav`/`save_wav`, `stft`/`istft`,
`spectrogram_to_mel`/`mel_to_spectrogram`, `audio_to_spectrogram`/
`spectrogram_to_audio`, `audio_to_frames`/`frames_to_audio`, `augment_audio`
(pitch shift / time stretch / noise).

**Text** (`Enilnets.text_utils`): `Tokenizer` (see above),
`load_text_file`/`load_texts_from_directory`, `create_sliding_windows`,
`one_hot_encode`, `pad_sequences`.

**Cross-modal** (`Enilnets.crossmodal_utils`): `contrastive_loss` (CLIP-style
image/text contrastive loss — pairs naturally with the `ntxent` loss above),
`clip_normalize`, `multimodal_fusion` (`"concat"`/`"sum"`/`"weighted"`),
`create_text_conditioned_image`.

## Visualization

`plot_network`/`model.plot(...)` renders the classic node/connection diagram
of a `NeuralNet` as a self-contained SVG string — pure stdlib, no matplotlib
dependency, so it fits the same NumPy-only footprint as everything else.
Dense/sparse/RNN/LSTM/GRU layers become columns of circular nodes with real
weighted edges (blue = positive, red = negative, opacity = relative
magnitude); other layer types (conv2d, attention, pooling, embedding, ...)
render as labeled blocks in between, since they don't reduce to a single
weight matrix connecting two node columns. Batchnorm/layernorm/dropout are
transparent to the diagram — edges are drawn straight through them using the
real dense weight matrix on the far side, since they're dimension-preserving.

```python
model = NeuralNet(learning_rate=0.001, optimizer="adam")
model.add_dense(4, 8, activation="relu")
model.add_batchnorm(8)
model.add_dense(8, 3, activation="softmax")

# Structure only, no values:
svg = model.plot()

# With a sample input, node fill color shows that layer's actual activation
# value from a live forward pass (heat-mapped blue=low to red=high) -- call
# this from inside your own training loop for a snapshot of what the
# network is doing right now:
for epoch in range(epochs):
    model.TrainBatch(X_batch, Y_batch)
    if epoch % 10 == 0:
        model.plot(sample_input=X_batch[:1], filename=f"epoch_{epoch}.svg")
```

`plot_genome(genome, sample_input=...)` does the same for a NEAT `Genome`:
node x-position is graph depth (topological distance from the inputs), node
color is type (input/bias/hidden/output) or activation value if
`sample_input` is given, and disabled connections are drawn dashed so you
can see structure NEAT has pruned as well as what's active.

**Using it in a real project:**

```python
# Returned value is always the raw SVG string -- embed it directly.
svg = model.plot(sample_input=x)

# In a Jupyter notebook:
from IPython.display import SVG, display
display(SVG(svg))

# In a web app (Flask, e.g.), serve a full standalone page:
from Enilnets import to_html
from flask import Response

@app.route("/network")
def network_view():
    return Response(to_html(svg, title="Live Network"), mimetype="text/html")

# Or just save it -- .svg writes the raw SVG, .html wraps it in a minimal
# standalone document (open either directly in any browser):
model.plot(sample_input=x, filename="network.svg")
model.plot(sample_input=x, filename="network.html")
```

Large layers are automatically capped (`max_nodes_per_layer=20` by default,
showing the first/last half with a `⋮` marker in between) so a 784-neuron
input layer doesn't render 784 circles.

## Model persistence

```python
model.Save("model.json")

model2 = NeuralNet()
model2.add_dense(20, 64, activation="relu")
model2.add_dense(64, 1, activation="sigmoid")
model2.Load("model.json")
```

`Save`/`Load` round-trip everything needed to resume training exactly where
you left off, not just the weights: layer parameters (including RNN/LSTM/GRU
and attention weights), optimizer state (unless `save_opt_state=False`),
every per-model hyperparameter that can be overridden at construction
(`learning_rate`, `l2_lambda`, `momentum`, `grad_clip_norm`,
`use_mixed_precision`, the Adam/RMSprop/Adagrad betas/epsilons), the
training-step counter, in-progress gradient-accumulation buffers, the
train/eval mode flag, and the auto-shape-inference/residual-connection
bookkeeping needed to keep calling `add_*`/`add_residual_end` on the loaded
model — plus any `extra_state` dict you pass in. The target model's layer
*shapes* must already match (build the same architecture before calling
`Load`) — only the values are restored.

Other `NeuralNet` introspection/utility methods: `summary()` (prints layer
shapes and parameter counts), `get_weights()`/`set_weights()`,
`freeze(layer_idx)`/`unfreeze(layer_idx)` (zero out training for specific
layers), `check_nan_inf()`, `copy()` (deep copy, including optimizer state
and residual-stack bookkeeping), `train()`/`eval()` (toggle the default
`training` flag), `reset_optimizer_state()`.

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

# 3. Per-layer/per-call, via activation_params or loss kwargs
model.add_dense(10, 10, activation="elu", activation_params={"alpha": 2.0})
model.ComputeLoss(out, y, function="huber", delta=0.5)
```

Constants exposed this way include `EPS_LOG`/`EPS_DIV` (log/division
guards), `SIGMOID_CLIP` (overflow-safe exp clipping), `LEAKYRELU_ALPHA`/
`ELU_ALPHA`, and the Adam/RMSprop/Adagrad hyperparameter defaults.

## Running the test suite

`test_enilnets.py` is a single unittest-based suite (281 tests) covering
every layer/model/optimizer/utility above, plus a benchmark harness:

```bash
python test_enilnets.py                # run all correctness tests
python test_enilnets.py -v             # verbose
python test_enilnets.py TestRecurrentLayers  # run one class
python test_enilnets.py --benchmark    # timing only (Benchmark* classes)
```

Every gradient-bearing feature (attention, residual connections, RNN/LSTM/
GRU BPTT, KV-cache) is checked against finite-difference numerical
gradients, not just "does it run" smoke tests.
