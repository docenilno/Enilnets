# Enilnets Library Documentation

A pure NumPy-based deep learning library with support for dense, convolutional, pooling, batch normalization, dropout, and sparse layers. Includes multiple optimizers, loss functions, activation functions, weight initialization methods, and a **full generative AI framework**.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Architecture](#core-architecture)
3. [Model Configuration](#model-configuration)
4. [Layer Types](#layer-types)
5. [Forward Pass](#forward-pass)
6. [Backward Pass](#backward-pass)
7. [Optimizers](#optimizers)
8. [Loss Functions](#loss-functions)
9. [Training](#training)
10. [Activation Functions](#activation-functions)
11. [Weight Initialization](#weight-initialization)
12. [Reinforcement Learning](#reinforcement-learning)
13. [Generative AI Framework](#generative-ai-framework)
    - [VAE](#vae)
    - [GAN](#gan)
    - [Diffusion Models](#diffusion-models)
    - [Autoregressive Models](#autoregressive-models)
    - [Normalizing Flows](#normalizing-flows)
    - [Energy-Based Models](#energy-based-models)
    - [Sampling Utilities](#sampling-utilities)
    - [UNet Denoiser](#unet-denoiser)
14. [Model I/O](#model-io)
15. [Utility Functions](#utility-functions)
16. [Known Limitations](#known-limitations)

---

## Quick Start

### Discriminative Example

```python
from Enilnets import NeuralNet
import numpy as np

model = NeuralNet(learning_rate=0.001, optimizer="adam", l2_lambda=0.01)
model.add_dense(784, 256, activation="relu")
model.add_dropout(0.3)
model.add_dense(256, 10, activation="softmax")

X_train = np.random.randn(1000, 784)
Y_train = np.eye(10)[np.random.randint(0, 10, 1000)]

history = model.Train(X_train, Y_train, epochs=10, batch_size=32)
```

### Generative Example (VAE)

```python
from Enilnets import VAE
import numpy as np

vae = VAE(input_dim=784, latent_dim=32,
          encoder_hidden=[512, 256], decoder_hidden=[256, 512],
          learning_rate=0.001, optimizer="adam")

X_train = np.random.rand(1000, 784)  # Normalized to [0,1]

history = vae.Train(X_train, epochs=20, batch_size=64)
generated = vae.generate(n_samples=16)
reconstructed = vae.reconstruct(X_train[:10])
```

### Generative Example (Diffusion)

```python
from Enilnets import DiffusionModel

diffusion = DiffusionModel(
    data_shape=(784,),
    time_steps=1000,
    beta_schedule="linear",
    denoiser_type="mlp",
    denoiser_hidden=[512, 512, 512],
    learning_rate=0.001
)

# Data should be normalized to roughly [-1, 1] or [0, 1]
X_train = np.random.randn(1000, 784) * 0.5

history = diffusion.Train(X_train, epochs=10, batch_size=64)
samples = diffusion.sample(n_samples=16)
```

---

## Core Architecture

The library is built around the `NeuralNet` class in `base.py`, which is a **unified class**
containing all methods inline. This avoids the method-binding issues that can occur with
monkey-patching approaches.

| Attribute | Type | Description |
|-----------|------|-------------|
| `layers` | `list` | Layer definitions with weights, biases, and hyperparameters |
| `learning_rate` | `float` | Global learning rate |
| `optimizer_type` | `str` | Optimizer name: `"sgd"`, `"rmsprop"`, `"adagrad"`, `"adam"` |
| `l2_lambda` | `float` | L2 regularization coefficient |
| `momentum` | `float` | Momentum coefficient for SGD |
| `outputs` | `list` | Cached layer outputs during forward pass |
| `pre_activations` | `list` | Cached pre-activation values (z) |
| `batchnorm_cache` | `list` | BatchNorm statistics cache |
| `deltas` | `list` | Gradient errors per layer |
| `opt_state` | `list` | Optimizer state (momentum, velocity) |
| `t` | `int` | Global timestep for bias correction (Adam) |

---

## Generative AI Framework

The generative framework lives in `Enilnets.generative` and provides six major model classes, all implemented in pure NumPy.

### VAE

```python
from Enilnets import VAE

vae = VAE(
    input_dim=784,           # Flattened input size
    latent_dim=32,           # Latent space dimension
    encoder_hidden=[512, 256],
    decoder_hidden=[256, 512],
    activation="swish",
    learning_rate=0.001,
    optimizer="adam",
    l2_lambda=0.0
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `encode(x)` | Returns `mu`, `logvar` |
| `decode(z)` | Returns reconstruction |
| `forward(x)` | Returns `recon`, `mu`, `logvar`, `z` |
| `loss(x)` | Computes BCE reconstruction + KL divergence |
| `train_step(x)` | One gradient update (fixed gradient chaining through decoder) |
| `Train(X, epochs, batch_size, verbose)` | Full training loop |
| `generate(n_samples)` | Sample from prior N(0,I) |
| `reconstruct(x)` | Encode then decode |
| `interpolate(x1, x2, n_steps)` | Linear interpolation in latent space |

**Notes:**
- Encoder outputs `2 * latent_dim` (concatenated mu and logvar).
- Decoder uses `sigmoid` output activation; inputs should be normalized to `[0, 1]`.
- Gradients are computed manually through the reparameterization trick with proper decoder backprop via `output_delta`.

---

### GAN

```python
from Enilnets import GAN

gan = GAN(
    latent_dim=100,
    data_dim=784,
    generator_hidden=[256, 512],
    discriminator_hidden=[512, 256],
    g_activation="swish",
    d_activation="leakyrelu",
    loss_type="bce",        # "bce", "bce_logits", or "wasserstein"
    learning_rate=0.0002,
    optimizer="adam"
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `generate(n_samples)` | Generate fake data from noise |
| `discriminate(x)` | Discriminator output |
| `Train(X, epochs, batch_size, d_steps, g_steps, verbose)` | Alternating D/G training |
| `sample(n_samples)` | Alias for generate |

**Notes:**
- `loss_type="wasserstein"` removes sigmoid from discriminator output.
- Generator output uses `tanh` (normalize data to `[-1, 1]` for best results).
- Discriminator training uses a **single Forward pass** on concatenated real+fake data to ensure `outputs[-1]` matches target shape during Backward.
- Generator gradients are properly chained through the discriminator using `d_input = dot(deltas[0], W0)` to get gradient w.r.t. generator output.

---

### Diffusion Models

```python
from Enilnets import DiffusionModel

diffusion = DiffusionModel(
    data_shape=(784,),              # or (1, 28, 28) for conv
    time_steps=1000,
    beta_schedule="linear",         # or "cosine"
    beta_start=1e-4,
    beta_end=0.02,
    denoiser_type="mlp",            # or "conv"
    denoiser_hidden=[512, 512, 512],
    learning_rate=0.001,
    optimizer="adam"
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `train_step(x)` | Sample random t, add noise, predict noise, MSE loss |
| `Train(X, epochs, batch_size, verbose)` | Full training loop |
| `sample(n_samples, shape, clip)` | Generate via iterative denoising |
| `denoise(x_noisy, t_start, t_end)` | Partial denoising for editing |

**Notes:**
- Training loss is MSE between predicted and actual noise.
- `sample()` runs the full reverse diffusion loop (can be slow for many timesteps).
- For images, use `denoiser_type="conv"`; the conv denoiser uses simple time broadcasting.
- Data should be normalized to roughly `[-1, 1]` or `[0, 1]`.

---

### Autoregressive Models

```python
from Enilnets import AutoregressiveModel

ar = AutoregressiveModel(
    data_dim=784,
    hidden_dims=[512, 512],
    data_shape=(28, 28),       # Optional, for reshaping output
    activation="swish",
    learning_rate=0.001
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `forward(x, training)` | Causal masked forward pass |
| `loss(x)` | MSE loss |
| `train_step(x)` | One gradient update |
| `Train(X, epochs, batch_size, verbose)` | Full training loop |
| `generate(n_samples, shape)` | Sequential generation |
| `complete(partial_x, n_dims)` | Complete partial samples |

**Notes:**
- Uses causal masking so dimension `i` only sees dimensions `0..i-1`.
- Generation is sequential and can be slow for high-dimensional data.
- `complete()` is useful for inpainting or partial observation tasks.

---

### Normalizing Flows (RealNVP)

```python
from Enilnets import RealNVP

flow = RealNVP(
    data_dim=784,
    n_coupling=4,
    hidden_dim=256,
    activation="swish",
    learning_rate=0.001
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `forward(x)` | Data -> latent, returns `z`, `log_det_jacobian` |
| `inverse(z)` | Latent -> data |
| `log_prob(x)` | Compute log p(x) under the model |
| `loss(x)` | Negative log-likelihood |
| `Train(X, epochs, batch_size, verbose)` | Training via evolutionary strategy per coupling layer |
| `sample(n_samples)` | Sample from base Gaussian and invert |
| `interpolate(x1, x2, n_steps)` | Latent space interpolation |

**Notes:**
- Uses affine coupling layers with alternating masks.
- `s_net` uses `tanh` output for stability; `t_net` uses linear output.
- Training uses `Evolve` evolutionary strategy per coupling layer. The score function properly runs the forward pass up to each coupling layer to get the correct input shape for `s_net`/`t_net`.

---

### Energy-Based Models

```python
from Enilnets import EnergyBasedModel

ebm = EnergyBasedModel(
    data_dim=784,
    hidden_dims=[512, 512],
    activation="swish",
    learning_rate=0.001
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `energy(x)` | Compute scalar energy E(x) |
| `train_step(x, n_cd_steps, step_size, noise_scale)` | Contrastive divergence update |
| `Train(X, epochs, batch_size, ...)` | Full training loop |
| `sample(n_samples, n_steps, ...)` | Langevin dynamics sampling |
| `score(x)` | Compute grad_x E(x) |

**Notes:**
- Uses contrastive divergence: push down energy on data, push up on negative samples.
- Negative samples are generated via Langevin dynamics.
- Energy gradients w.r.t. input are computed via finite differences.

---

### Sampling Utilities

```python
from Enilnets.generative.sampling import (
    reparameterize, langevin_dynamics,
    gaussian_sample, gumbel_softmax_sample,
    compute_returns  # Also available here for convenience
)
```

| Function | Description |
|----------|-------------|
| `reparameterize(mu, logvar)` | VAE reparameterization: `z = mu + sigma * eps` |
| `langevin_dynamics(energy_fn, x_init, n_steps, step_size, noise_scale)` | MCMC sampling for EBMs |
| `gaussian_sample(mean, std, shape)` | Sample from Gaussian |
| `gumbel_softmax_sample(logits, temperature, hard)` | Gumbel-Softmax for discrete latents |
| `compute_returns(rewards, gamma)` | Discounted returns for RL |

---

### UNet Denoiser

```python
from Enilnets import UNetDenoiser, time_embedding

unet = UNetDenoiser(
    in_ch=1,
    base_ch=64,
    time_emb_dim=128,
    ch_mult=(1, 2, 4)
)

# Time embedding for diffusion
t_emb = time_embedding(t=np.array([0, 50, 100]), dim=128)
```

**Notes:**
- Designed for diffusion models on spatial data.
- Uses **k=1 convolutions** to avoid spatial shrinking (base conv2d has no padding support).
- Uses skip connections between encoder and decoder paths.
- Time conditioning is added via broadcasted embeddings at each level.
- `forward(x, t)` takes noisy images and timestep indices.
- Output shape is preserved to match input shape exactly.

---

## Updated Loss Functions

The `ComputeLoss` method supports generative losses:

| Function | Description | Extra Args |
|----------|-------------|------------|
| `"mse"` | Mean Squared Error | none |
| `"mae"` | Mean Absolute Error | none |
| `"huber"` | Huber Loss | `delta=1.0` |
| `"smooth_l1"` | Smooth L1 Loss | none |
| `"binary_cross_entropy"` | Binary Cross-Entropy | none |
| `"cross_entropy"` | Categorical Cross-Entropy | none |
| `"focal"` | Focal Loss | `alpha=0.25`, `gamma=2.0` |
| `"hinge"` | Hinge Loss | none |
| **`"kl_divergence"`** | KL(q\|\|N(0,I)) | `mu`, `logvar` |
| **`"bce_logits"`** | BCE with logits (stable) | none |
| **`"wasserstein"`** | Wasserstein/GP loss | none |

---

## Complete Example: Generative Models on MNIST

```python
from Enilnets import VAE, GAN, DiffusionModel, AutoregressiveModel
import numpy as np

# Load MNIST (pseudo-code)
# X_train: (60000, 1, 28, 28), normalized to [0, 1] or [-1, 1]

# --- VAE ---
vae = VAE(input_dim=784, latent_dim=32,
          encoder_hidden=[512, 256], decoder_hidden=[256, 512],
          learning_rate=0.001, optimizer="adam")
vae_history = vae.Train(X_train.reshape(-1, 784), epochs=20, batch_size=128)
vae_samples = vae.generate(n_samples=16).reshape(-1, 1, 28, 28)

# --- GAN ---
gan = GAN(latent_dim=100, data_dim=784,
          generator_hidden=[256, 512], discriminator_hidden=[512, 256],
          loss_type="bce", learning_rate=0.0002)
gan_history = gan.Train(X_train.reshape(-1, 784), epochs=50, batch_size=64,
                        d_steps=1, g_steps=1)
gan_samples = gan.sample(n_samples=16).reshape(-1, 1, 28, 28)

# --- Diffusion ---
diffusion = DiffusionModel(
    data_shape=(1, 28, 28),
    time_steps=1000,
    beta_schedule="cosine",
    denoiser_type="conv",
    learning_rate=0.001
)
diff_history = diffusion.Train(X_train, epochs=10, batch_size=64)
diff_samples = diffusion.sample(n_samples=16)

# --- Autoregressive ---
ar = AutoregressiveModel(
    data_dim=784,
    hidden_dims=[512, 512],
    data_shape=(28, 28)
)
ar_history = ar.Train(X_train.reshape(-1, 784), epochs=10, batch_size=64)
ar_samples = ar.generate(n_samples=16, shape=(28, 28))
```

---

## Architecture Notes

### Data Format
- The library uses **channels-first** format for convolutions: `(batch, channels, height, width)`.
- Generative models that accept flattened data will auto-reshape 4D image inputs.

### Training Generative Models
- **VAE**: Uses manual backpropagation through the reparameterization trick with proper `output_delta` chaining.
- **GAN**: Alternates between discriminator and generator updates; discriminator uses single concatenated Forward pass, generator chains gradients through discriminator input.
- **Diffusion**: Predicts noise epsilon; training is essentially MSE regression.
- **Autoregressive**: Uses causal masking; generation is sequential.
- **Flows**: Uses evolutionary strategy (`Evolve`) per coupling layer with proper forward pass up to each layer.
- **EBM**: Uses contrastive divergence with Langevin dynamics for negative sampling.

### Numerical Stability
- All generative models use `float64` dtype.
- Diffusion models clip beta values and use stable cumulative product computations.
- VAE decoder uses `sigmoid` with BCE reconstruction; inputs should be normalized to `[0, 1]`.

---

## Known Limitations

1. **Conv2d has no padding support**: All convolutions use `pad=0`, so spatial dimensions shrink by `k-1` per layer. The UNet uses `k=1` convolutions to work around this.
2. **UNet backward is not implemented**: The UNet denoiser has `backward()` that raises `NotImplementedError`. Use the MLP-based `DiffusionModel` for fully trainable diffusion.
3. **Flows use evolutionary training**: RealNVP uses `Evolve` rather than analytical backprop through the log-determinant Jacobian, which is slower but works in pure NumPy.
4. **GAN training can be unstable**: As with all GANs, training may require tuning of learning rates and architecture sizes.

---

## API Reference Summary

### NeuralNet Methods

| Method | Description |
|--------|-------------|
| `__init__(lr, opt, l2, mom)` | Constructor |
| `summary()` | Print architecture |
| `add_dense(...)` | Add dense layer |
| `add_sparse(...)` | Add sparse layer |
| `add_conv2d(...)` | Add conv layer (no padding) |
| `add_flatten()` | Add flatten layer |
| `add_maxpool2d(p)` | Add max pool |
| `add_avgpool2d(p)` | Add avg pool |
| `add_batchnorm(...)` | Add batch norm |
| `add_dropout(rate)` | Add dropout |
| `Forward(x, training, dropout_rate)` | Forward pass |
| `predict(x)` | Alias for Forward |
| `Backward(targets, output_delta)` | Backpropagation |
| `update()` | Apply gradients |
| `TrainBatch(xs, ys, ...)` | Train one batch |
| `Train(X, Y, epochs, ...)` | Full training loop |
| `ComputeLoss(out, tgt, ...)` | Compute loss |
| `compute_accuracy(pred, tgt)` | Compute accuracy |
| `Reinforce(...)` | Policy gradient |
| `Evolve(...)` | Evolutionary strategy |
| `Save(file)` | Save model to JSON or PKL |
| `Load(file)` | Load model from JSON or PKL |

### Generative Classes

| Class | Module | Description |
|-------|--------|-------------|
| `VAE` | `Enilnets.generative.vae` | Variational Autoencoder |
| `GAN` | `Enilnets.generative.gan` | Generative Adversarial Network |
| `DiffusionModel` | `Enilnets.generative.diffusion` | DDPM diffusion model |
| `AutoregressiveModel` | `Enilnets.generative.autoregressive` | MADE-like AR model |
| `RealNVP` | `Enilnets.generative.flows` | Normalizing flow |
| `EnergyBasedModel` | `Enilnets.generative.ebm` | Energy-based model |
| `UNetDenoiser` | `Enilnets.generative.unet` | UNet for diffusion |
| `time_embedding` | `Enilnets.generative.unet` | Sinusoidal time embedding |

### Generative Loss Functions

| Function | Module | Description |
|----------|--------|-------------|
| `kl_divergence_gaussian(mu, logvar)` | `generative_loss` | KL(q\|\|N(0,I)) |
| `adversarial_loss_discriminator(...)` | `generative_loss` | D loss |
| `adversarial_loss_generator(...)` | `generative_loss` | G loss |
| `diffusion_loss(pred, true)` | `generative_loss` | MSE noise prediction |
| `nll_loss(log_px, log_det)` | `generative_loss` | Flow negative log-likelihood |
| `energy_loss(data_e, sample_e)` | `generative_loss` | EBM contrastive loss |
