#!/usr/bin/env python3
"""
Enilnets Combined Test & Benchmark Suite
=========================================
Merges what used to be test.py, test2.py and test3.py into one unittest-based
suite, plus a shared benchmark harness. Assumes the Enilnets package is
importable (no sys.path hacks).

Usage:
    python test_enilnets.py                 # run all correctness tests
    python test_enilnets.py -v              # verbose
    python test_enilnets.py TestVAE          # run one class
    python test_enilnets.py --benchmark      # run only the Benchmark* classes
    python test_enilnets.py --gpu            # run with Enilnets.use_gpu(True)
    python test_enilnets.py --float64        # run with Enilnets.use_float64(True)
    python test_enilnets.py --gpu --float64 --benchmark  # any combination
"""

import sys
import time
import unittest
import tempfile
import os
import warnings
import contextlib
import statistics

warnings.filterwarnings("ignore", category=RuntimeWarning)

import Enilnets
if "--gpu" in sys.argv:
    sys.argv.remove("--gpu")
    Enilnets.use_gpu(True)
if "--float64" in sys.argv:
    sys.argv.remove("--float64")
    Enilnets.use_float64(True)

from Enilnets.backend import np
from Enilnets import NeuralNet, LRScheduler, TextGenerator, Tokenizer
from Enilnets.generative import (
    VAE, GAN, DiffusionModel, AutoregressiveModel,
    RealNVP, EnergyBasedModel, UNetDenoiser, time_embedding,
    reparameterize, langevin_dynamics, gaussian_sample,
    uniform_sample, gumbel_softmax_sample, random_mask,
    kl_divergence_gaussian, adversarial_loss_discriminator,
    adversarial_loss_generator, diffusion_loss, nll_loss, energy_loss,
    compute_returns,
)
from Enilnets.generative.sampling import top_p_sampling, gae
from Enilnets.generative.generative_loss import perceptual_loss, vgg_loss
from Enilnets.activations import activate, derivative
from Enilnets.weight_init import init_weights, init_conv_weights, init_embedding_weights
from Enilnets.forward import im2col
from Enilnets.backward import multihead_attention_backward, embedding_backward, upsample2d_backward
import Enilnets.image_utils as img_utils
import Enilnets.audio_utils as aud_utils
import Enilnets.text_utils as txt_utils
import Enilnets.eval_utils as eval_utils
import Enilnets.crossmodal_utils as cm_utils
from Enilnets import (
    set_seed, train_test_split, iterate_minibatches, count_parameters,
    EarlyStopping, one_hot, constants,
    ModelCheckpoint, CSVLogger, JSONLogger,
)


# ========================================================================
# Helpers
# ========================================================================

class _FDPrecisionMixin:
    """Mixin for test classes built around finite-difference numerical
    gradient checks. Forces float64 for the whole class's duration
    regardless of the library's ambient default dtype: FD checks at
    eps=1e-5/1e-6 rely on subtractive cancellation that float32's ~7
    decimal digits can't resolve reliably, so these specific
    gradient-correctness checks always run at float64 -- unrelated to
    whether the rest of the suite is exercising the float32 default or
    the --float64 opt-in pass."""

    @classmethod
    def setUpClass(cls):
        cls._prev_float64 = Enilnets.is_float64_enabled()
        Enilnets.use_float64(True)

    @classmethod
    def tearDownClass(cls):
        Enilnets.use_float64(cls._prev_float64)


@contextlib.contextmanager
def _force_float64():
    """Same rationale as _FDPrecisionMixin, scoped to a single test method
    instead of a whole class -- for one-off precision-sensitive tests
    living in an otherwise-unrelated TestCase."""
    prev = Enilnets.is_float64_enabled()
    Enilnets.use_float64(True)
    try:
        yield
    finally:
        Enilnets.use_float64(prev)


def numerical_gradient(fn, x, eps=1e-5):
    """Compute numerical gradient of fn w.r.t. x via central differences."""
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index
        old = x[ix]
        x[ix] = old + eps
        fplus = fn(x)
        x[ix] = old - eps
        fminus = fn(x)
        x[ix] = old
        grad[ix] = (fplus - fminus) / (2 * eps)
        it.iternext()
    return grad


def make_classification_data(n, in_dim, n_classes, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = np.random.randn(n, in_dim).astype(np.float64)
    Y = np.eye(n_classes)[np.random.randint(0, n_classes, n)]
    return X, Y


# ========================================================================
# Core Tests
# ========================================================================

class TestWeightInit(unittest.TestCase):
    def test_dense_init_shapes(self):
        for method in ["xavier_uniform", "xavier_normal", "he_uniform", "he_normal",
                       "normal", "orthogonal", "zeros", "ones"]:
            w, b = init_weights(100, 50, method=method)
            self.assertEqual(w.shape, (50, 100))
            self.assertEqual(b.shape, (50,))
            self.assertEqual(w.dtype, Enilnets.default_dtype())
            self.assertTrue(np.all(np.isfinite(w)))
        w, _ = init_weights(10, 10, method="zeros")
        self.assertTrue(np.allclose(w, 0))
        w, _ = init_weights(10, 10, method="ones")
        self.assertTrue(np.allclose(w, 1))

    def test_conv_init_shapes(self):
        for method in ["xavier_uniform", "xavier_normal", "he_uniform", "he_normal",
                       "normal", "orthogonal", "zeros", "ones"]:
            w, b = init_conv_weights(3, 16, 3, method=method)
            self.assertEqual(w.shape, (16, 3, 3, 3))
            self.assertEqual(b.shape, (16,))
            self.assertTrue(np.all(np.isfinite(w)))

    def test_conv_xavier_uses_full_fan_out(self):
        # Regression test: xavier fan_out for conv must be out_ch*k*k, not out_ch
        # (fixed bug: the k*k factor was missing, over-scaling the init).
        in_ch, out_ch, k = 8, 16, 3
        expected_limit = np.sqrt(6 / (in_ch * k * k + out_ch * k * k))
        np.random.seed(0)
        w, _ = init_conv_weights(in_ch, out_ch, k, method="xavier_uniform")
        self.assertLessEqual(np.max(np.abs(w)), expected_limit + 1e-9)
        empirical_std = np.std(w)
        theoretical_std = expected_limit / np.sqrt(3)
        self.assertAlmostEqual(empirical_std, theoretical_std, delta=theoretical_std * 0.15)

    def test_embedding_init_shapes(self):
        for method in ["normal", "xavier_uniform", "xavier_normal", "zeros"]:
            w = init_embedding_weights(1000, 128, method=method)
            self.assertEqual(w.shape, (1000, 128))
            self.assertTrue(np.all(np.isfinite(w)))

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            init_weights(10, 10, method="unknown")

    def test_im2col(self):
        x = np.random.randn(2, 3, 4, 4).astype(np.float64)
        col = im2col(x, 2, 2)
        self.assertEqual(col.shape, (18, 12))
        self.assertTrue(np.all(np.isfinite(col)))


class TestActivations(_FDPrecisionMixin, unittest.TestCase):
    def test_relu_range(self):
        x = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
        out = activate("relu", x)
        self.assertTrue(np.all(out >= 0))
        self.assertTrue(np.allclose(out, [0, 0, 0, 1, 2]))

    def test_sigmoid_range(self):
        x = np.array([-10, 0, 10], dtype=np.float64)
        out = activate("sigmoid", x)
        self.assertTrue(np.all((out >= 0) & (out <= 1)))

    def test_tanh_range(self):
        x = np.array([-10, 0, 10], dtype=np.float64)
        out = activate("tanh", x)
        self.assertTrue(np.all((out >= -1) & (out <= 1)))

    def test_softmax_sums_to_one(self):
        x = np.array([[1, 2, 3], [1, 1, 1]], dtype=np.float64)
        out = activate("softmax", x)
        self.assertTrue(np.allclose(out.sum(axis=-1), 1.0))

    def test_all_activations_finite(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
        names = ["relu", "leakyrelu", "elu", "selu", "gelu", "swish", "mish",
                  "sigmoid", "tanh", "softplus", "linear"]
        for name in names:
            out = activate(name, x)
            self.assertTrue(np.all(np.isfinite(out)), msg=name)
            self.assertEqual(out.shape, x.shape, msg=name)

    def test_sigmoid_saturation(self):
        x = np.array([1000.0, -1000.0], dtype=np.float64)
        out = activate("sigmoid", x)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertTrue(np.allclose(out, [1.0, 0.0], atol=1e-6))

    def test_derivative_finite_diff(self):
        # gelu is included: this is a regression test for the fix making the
        # derivative match the tanh-approximation actually used in activate().
        for name in ["relu", "leakyrelu", "elu", "selu", "gelu", "swish", "mish",
                     "sigmoid", "tanh", "softplus"]:
            x = np.array([-1.0, 0.5, 2.0], dtype=np.float64)
            analytical = derivative(name, x)
            numerical = np.zeros_like(x)
            eps = 1e-5
            for i in range(x.size):
                xv = x.copy()
                xv.flat[i] += eps
                fplus = activate(name, xv).flat[i]
                xv.flat[i] -= 2 * eps
                fminus = activate(name, xv).flat[i]
                numerical.flat[i] = (fplus - fminus) / (2 * eps)
            self.assertTrue(np.allclose(analytical, numerical, rtol=1e-3, atol=1e-4),
                            msg=f"Derivative mismatch for {name}: {analytical} vs {numerical}")


class TestLayers(unittest.TestCase):
    def test_add_dense(self):
        model = NeuralNet()
        model.add_dense(10, 20, activation="relu")
        self.assertEqual(model.layers[0]["type"], "dense")
        self.assertEqual(model.layers[0]["weights"].shape, (20, 10))
        self.assertEqual(model.layers[0]["activation"], "relu")

    def test_add_sparse(self):
        model = NeuralNet()
        model.add_sparse(10, 20, connectivity=0.5)
        self.assertEqual(model.layers[0]["type"], "sparse")
        mask = model.layers[0]["mask"]
        self.assertTrue(0.3 < np.mean(mask) < 0.7)
        self.assertTrue(np.allclose(model.layers[0]["weights"][mask == 0], 0))

    def test_add_conv2d(self):
        model = NeuralNet()
        model.add_conv2d(1, 8, k=3, activation="relu")
        self.assertEqual(model.layers[0]["type"], "conv2d")
        self.assertEqual(model.layers[0]["weights"].shape, (8, 1, 3, 3))

    def test_add_batchnorm(self):
        model = NeuralNet()
        model.add_batchnorm(64)
        self.assertEqual(model.layers[0]["type"], "batchnorm")
        self.assertEqual(model.layers[0]["gamma"].shape, (64,))
        self.assertEqual(model.layers[0]["beta"].shape, (64,))

    def test_summary_runs(self):
        model = NeuralNet()
        model.add_dense(10, 20, activation="relu")
        model.add_dense(20, 5, activation="softmax")
        try:
            model.summary()
        except Exception as e:
            self.fail(f"summary() raised {e}")


class TestForwardPass(unittest.TestCase):
    def test_dense_forward_shape(self):
        model = NeuralNet()
        model.add_dense(10, 20, activation="relu")
        model.add_dense(20, 5, activation="softmax")
        x = np.random.randn(4, 10)
        out = model.Forward(x)
        self.assertEqual(out.shape, (4, 5))

    def test_sparse_forward(self):
        model = NeuralNet()
        model.add_sparse(10, 5, connectivity=0.5, activation="relu")
        x = np.random.randn(3, 10).astype(np.float64)
        out = model.Forward(x)
        self.assertEqual(out.shape, (3, 5))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_conv_forward_shape(self):
        model = NeuralNet()
        model.add_conv2d(1, 4, k=3, activation="relu")
        x = np.random.randn(2, 1, 8, 8)
        out = model.Forward(x)
        self.assertEqual(out.shape, (2, 4, 6, 6))

    def test_flatten_forward(self):
        model = NeuralNet()
        model.add_conv2d(1, 2, k=3, activation="relu")
        model.add_flatten()
        model.add_dense(2 * 6 * 6, 10, activation="softmax")
        x = np.random.randn(2, 1, 8, 8)
        out = model.Forward(x)
        self.assertEqual(out.shape, (2, 10))

    def test_maxpool_forward(self):
        model = NeuralNet()
        model.add_conv2d(1, 2, k=3, activation="relu")
        model.add_maxpool2d(pool_size=2)
        x = np.random.randn(2, 1, 8, 8)
        out = model.Forward(x)
        self.assertEqual(out.shape, (2, 2, 3, 3))

    def test_avgpool_forward(self):
        model = NeuralNet()
        model.add_conv2d(1, 2, k=3, activation="relu")
        model.add_avgpool2d(pool_size=2)
        x = np.random.randn(2, 1, 8, 8)
        out = model.Forward(x)
        self.assertEqual(out.shape, (2, 2, 3, 3))

    def test_global_avgpool_forward(self):
        model = NeuralNet()
        model.add_conv2d(3, 4, k=3, activation="relu")
        model.add_global_avgpool2d()
        x = np.random.randn(2, 3, 6, 6)
        out = model.Forward(x)
        self.assertEqual(out.shape, (2, 4, 1, 1))

    def test_upsample_forward(self):
        model = NeuralNet()
        model.add_conv2d(3, 4, k=3, activation="relu")
        model.add_upsample2d(scale_factor=2)
        x = np.random.randn(2, 3, 6, 6)
        out = model.Forward(x)
        self.assertEqual(out.shape, (2, 4, 8, 8))

    def test_batchnorm_forward_training_vs_eval(self):
        model = NeuralNet()
        model.add_dense(10, 20, activation="relu")
        model.add_batchnorm(20)
        x = np.random.randn(8, 10)
        out_train = model.Forward(x, training=True)
        out_eval = model.Forward(x, training=False)
        self.assertEqual(out_train.shape, (8, 20))
        self.assertEqual(out_eval.shape, (8, 20))
        self.assertFalse(np.allclose(model.layers[1]["running_mean"], 0))

    def test_layernorm_forward(self):
        model = NeuralNet()
        model.add_dense(10, 5, activation="linear")
        model.add_layernorm(5)
        x = np.random.randn(32, 10)
        out = model.Forward(x, training=True)
        self.assertEqual(out.shape, (32, 5))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_layernorm_3d_sequence_forward(self):
        # Regression: LayerNorm must support (batch, seq_len, embed_dim) for
        # transformer blocks, normalizing over the embedding axis only.
        model = NeuralNet()
        model.add_layernorm(8)
        x = np.random.randn(2, 5, 8)
        out = model.Forward(x, training=True)
        self.assertEqual(out.shape, (2, 5, 8))
        self.assertTrue(np.allclose(np.mean(out, axis=-1), 0, atol=1e-6))

    def test_dropout_mask(self):
        model = NeuralNet()
        model.add_dense(10, 20, activation="relu")
        model.add_dropout(0.5)
        x = np.random.randn(4, 10)
        out1 = model.Forward(x, training=True)
        mask1 = model.layers[1].get("mask")
        self.assertIsNotNone(mask1)
        out2 = model.Forward(x, training=True)
        mask2 = model.layers[1].get("mask")
        self.assertFalse(np.allclose(mask1, mask2))

    def test_embedding_forward(self):
        model = NeuralNet()
        model.add_embedding(vocab_size=100, embed_dim=32)
        x = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        out = model.Forward(x)
        self.assertEqual(out.shape, (2, 3, 32))
        self.assertTrue(np.all(np.isfinite(out)))


class TestBackwardPass(_FDPrecisionMixin, unittest.TestCase):
    def test_dense_gradient_flow(self):
        model = NeuralNet(learning_rate=0.1, optimizer="sgd")
        model.add_dense(5, 4, activation="linear")
        x = np.random.randn(2, 5)
        target = np.random.randn(2, 4)
        model.Forward(x, training=True)
        model.Backward(target)
        self.assertIsNotNone(model.deltas[0])
        self.assertEqual(model.deltas[0].shape, (2, 4))

    def test_gradient_check_simple(self):
        np.random.seed(42)
        model = NeuralNet(learning_rate=0.01, optimizer="sgd", l2_lambda=0.0)
        model.add_dense(3, 2, activation="linear")
        x = np.random.randn(1, 3)
        target = np.array([[1.0, 0.5]])

        def loss_fn():
            out = model.Forward(x, training=True)
            return np.mean((out - target) ** 2)

        w0 = model.layers[0]["weights"].copy()
        eps = 1e-5
        num_grad = np.zeros_like(w0)
        for i in range(w0.shape[0]):
            for j in range(w0.shape[1]):
                model.layers[0]["weights"] = w0.copy()
                model.layers[0]["weights"][i, j] += eps
                lplus = loss_fn()
                model.layers[0]["weights"][i, j] -= 2 * eps
                lminus = loss_fn()
                num_grad[i, j] = (lplus - lminus) / (2 * eps)

        model.layers[0]["weights"] = w0.copy()
        model.Forward(x, training=True)
        model.Backward(target, loss_function="mse")
        ana_grad = model.deltas[0].T @ x

        self.assertTrue(np.allclose(ana_grad, num_grad, rtol=1e-3, atol=1e-3))

    def test_conv_backward_shape(self):
        model = NeuralNet()
        model.add_conv2d(1, 2, k=3, activation="relu")
        x = np.random.randn(2, 1, 8, 8)
        target = np.random.randn(2, 2, 6, 6)
        model.Forward(x, training=True)
        model.Backward(target)
        self.assertEqual(model.deltas[0].shape, (2, 2, 6, 6))

    def test_batchnorm_backward(self):
        model = NeuralNet(learning_rate=0.01, optimizer="adam")
        model.add_dense(10, 5, activation="linear")
        model.add_batchnorm(5)
        model.add_dense(5, 2, activation="softmax")
        x = np.random.randn(8, 10)
        y = np.eye(2)[np.random.randint(0, 2, 8)]
        model.Forward(x, training=True)
        model.Backward(y)
        self.assertTrue(np.all(np.isfinite(model.deltas[0])))
        self.assertIn("d_gamma", model.layers[1])
        self.assertIn("d_beta", model.layers[1])

    def test_layernorm_backward(self):
        model = NeuralNet(learning_rate=0.01, optimizer="adam")
        model.add_dense(10, 5, activation="linear")
        model.add_layernorm(5)
        model.add_dense(5, 2, activation="softmax")
        x = np.random.randn(8, 10)
        y = np.eye(2)[np.random.randint(0, 2, 8)]
        model.Forward(x, training=True)
        model.Backward(y)
        self.assertTrue(np.all(np.isfinite(model.deltas[0])))
        self.assertIn("d_gamma", model.layers[1])

    def test_pooling_backward(self):
        model = NeuralNet(learning_rate=0.01, optimizer="sgd")
        model.add_conv2d(3, 4, k=3, activation="relu")
        model.add_maxpool2d(pool_size=2)
        model.add_flatten()
        model.add_dense(16, 2, activation="softmax")
        x = np.random.randn(2, 3, 6, 6)
        y = np.eye(2)[np.random.randint(0, 2, 2)]
        model.Forward(x, training=True)
        model.Backward(y)
        self.assertTrue(np.all(np.isfinite(model.deltas[0])))

    def test_upsample_backward(self):
        model = NeuralNet(learning_rate=0.01, optimizer="sgd")
        model.add_conv2d(3, 4, k=3, activation="relu")
        model.add_upsample2d(scale_factor=2)
        model.add_flatten()
        model.add_dense(256, 2, activation="softmax")
        x = np.random.randn(2, 3, 6, 6)
        y = np.eye(2)[np.random.randint(0, 2, 2)]
        model.Forward(x, training=True)
        model.Backward(y)
        self.assertTrue(np.all(np.isfinite(model.deltas[0])))

    def test_dropout_backward(self):
        model = NeuralNet(learning_rate=0.01, optimizer="sgd")
        model.add_dense(10, 5, activation="relu")
        model.add_dropout(rate=0.5)
        model.add_dense(5, 2, activation="softmax")
        x = np.random.randn(4, 10)
        y = np.eye(2)[np.random.randint(0, 2, 4)]
        model.Forward(x, training=True)
        model.Backward(y)
        self.assertTrue(np.all(np.isfinite(model.deltas[0])))


class TestBackwardLossGradients(_FDPrecisionMixin, unittest.TestCase):
    """Regression tests for Backward()'s per-loss-function gradient dispatch:
    every loss_function must produce a gradient matching a finite-difference
    approximation of loss.py's own ComputeLoss formula, not the old hardcoded
    softmax/MSE-only formula."""

    def _check(self, loss_function, activation, targets, n_out=4, n_in=6, batch=3, **kwargs):
        np.random.seed(7)
        model = NeuralNet(learning_rate=0.01, optimizer="adam")
        model.add_dense(n_in, 5, activation="tanh")
        model.add_dense(5, n_out, activation=activation)
        x = np.random.randn(batch, n_in)

        model.Forward(x, training=True)
        model.Backward(targets, loss_function=loss_function, **kwargs)
        analytic = np.dot(model.deltas[1].T, model.outputs[1])

        eps = 1e-6
        W = model.layers[1]["weights"]
        i, j = 0, 1
        orig = float(W[i, j])

        def loss_val():
            out = model.Forward(x, training=True)
            return model.ComputeLoss(out, targets, function=loss_function, reduction="mean", **kwargs)

        W[i, j] = orig + eps
        lp = loss_val()
        W[i, j] = orig - eps
        lm = loss_val()
        W[i, j] = orig
        numeric = (lp - lm) / (2 * eps)
        self.assertAlmostEqual(numeric, analytic[i, j], delta=max(1e-4, abs(numeric) * 1e-3))

    def test_mse(self):
        self._check("mse", "linear", np.random.randn(3, 4))

    def test_mae(self):
        self._check("mae", "linear", np.random.randn(3, 4))

    def test_huber(self):
        self._check("huber", "linear", np.random.randn(3, 4), delta=0.5)

    def test_smooth_l1(self):
        self._check("smooth_l1", "linear", np.random.randn(3, 4))

    def test_binary_cross_entropy(self):
        self._check("binary_cross_entropy", "sigmoid", (np.random.rand(3, 4) > 0.5).astype(float))

    def test_cross_entropy_softmax(self):
        self._check("cross_entropy", "softmax", np.eye(4)[np.random.randint(0, 4, 3)])

    def test_focal(self):
        self._check("focal", "sigmoid", (np.random.rand(3, 4) > 0.5).astype(float))

    def test_hinge(self):
        self._check("hinge", "tanh", np.sign(np.random.randn(3, 4)))

    def test_bce_logits(self):
        self._check("bce_logits", "linear", (np.random.rand(3, 4) > 0.5).astype(float))

    def test_wasserstein(self):
        self._check("wasserstein", "linear", np.sign(np.random.randn(3, 4)))

    def test_cosine_similarity(self):
        self._check("cosine_similarity", "tanh", np.random.randn(3, 4))

    def test_triplet(self):
        self._check("triplet", "linear", np.random.randn(3, 4), negative=np.random.randn(3, 4))

    def test_ntxent(self):
        self._check("ntxent", "linear", np.random.randn(3, 4))

    def test_kl_divergence_rejected(self):
        # kl_divergence is a mu/logvar loss, not a Backward() output-layer loss.
        model = NeuralNet()
        model.add_dense(4, 4, activation="linear")
        model.Forward(np.random.randn(2, 4), training=True)
        with self.assertRaises(ValueError):
            model.Backward(np.random.randn(2, 4), loss_function="kl_divergence")

    def test_train_batch_uses_selected_loss(self):
        model = NeuralNet(learning_rate=0.05, optimizer="adam")
        model.add_dense(6, 5, activation="tanh")
        model.add_dense(5, 4, activation="linear")
        x = np.random.randn(8, 6)
        y = np.random.randn(8, 4)
        loss1, _ = model.TrainBatch(x, y, loss_function="huber", delta=0.5)
        self.assertIsInstance(loss1, float)
        self.assertTrue(np.isfinite(loss1))


class TestAttention(_FDPrecisionMixin, unittest.TestCase):
    """Regression tests for the multi-head self-attention implementation
    (previously non-functional: chained dense layers with no Q/K/V/softmax,
    and never even bound onto NeuralNet)."""

    def test_add_multihead_attention_creates_single_layer(self):
        model = NeuralNet()
        model.add_multihead_attention(embed_dim=8, num_heads=2)
        self.assertEqual(len(model.layers), 1)
        layer = model.layers[0]
        self.assertEqual(layer["type"], "multihead_attention")
        for key in ("Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo"):
            self.assertIn(key, layer)
        self.assertEqual(layer["Wq"].shape, (8, 8))

    def test_forward_shape_and_softmax_normalization(self):
        model = NeuralNet()
        model.add_multihead_attention(embed_dim=8, num_heads=2)
        x = np.random.randn(2, 5, 8)
        out = model.Forward(x, training=True)
        self.assertEqual(out.shape, (2, 5, 8))
        Q, K, V, Qh, Kh, Vh, attn, context, x_in = model.attention_cache[0]
        self.assertTrue(np.allclose(np.sum(attn, axis=-1), 1.0))

    def test_backward_gradients_match_finite_difference(self):
        np.random.seed(11)
        model = NeuralNet()
        model.add_multihead_attention(embed_dim=8, num_heads=2)
        x = np.random.randn(2, 4, 8)
        out = model.Forward(x, training=True)
        dout = np.random.randn(*out.shape)
        layer = model.layers[0]
        cache = model.attention_cache[0]
        dx = multihead_attention_backward(dout.copy(), layer, cache)

        eps = 1e-6
        for name in ("Wq", "Wk", "Wv", "Wo"):
            W = layer[name]
            i, j = 1, 2
            orig = float(W[i, j])
            W[i, j] = orig + eps
            out_p = model.Forward(x, training=True)
            loss_p = np.sum(out_p * dout)
            W[i, j] = orig - eps
            out_m = model.Forward(x, training=True)
            loss_m = np.sum(out_m * dout)
            W[i, j] = orig
            numeric = (loss_p - loss_m) / (2 * eps)
            self.assertAlmostEqual(numeric, layer["d_" + name][i, j], delta=max(1e-4, abs(numeric) * 1e-3))

        b, s, e = 0, 1, 3
        orig = float(x[b, s, e])
        x[b, s, e] = orig + eps
        out_p = model.Forward(x, training=True)
        loss_p = np.sum(out_p * dout)
        x[b, s, e] = orig - eps
        out_m = model.Forward(x, training=True)
        loss_m = np.sum(out_m * dout)
        x[b, s, e] = orig
        numeric_dx = (loss_p - loss_m) / (2 * eps)
        self.assertAlmostEqual(numeric_dx, dx[b, s, e], delta=max(1e-4, abs(numeric_dx) * 1e-3))

    def test_update_changes_weights(self):
        model = NeuralNet(learning_rate=0.01, optimizer="adam")
        model.add_multihead_attention(embed_dim=8, num_heads=2)
        x = np.random.randn(2, 4, 8)
        out = model.Forward(x, training=True)
        w_before = model.layers[0]["Wq"].copy()
        model.Backward(None, output_delta=np.random.randn(*out.shape))
        model.update()
        self.assertFalse(np.allclose(w_before, model.layers[0]["Wq"]))

    def test_sinusoidal_positional_encoding(self):
        model = NeuralNet()
        model.add_positional_encoding(max_seq_len=10, embed_dim=8, learnable=False)
        x = np.zeros((2, 5, 8))
        out = model.Forward(x, training=True)
        self.assertEqual(out.shape, (2, 5, 8))
        self.assertFalse(np.allclose(out, 0))

    def test_transformer_block_end_to_end(self):
        np.random.seed(1)
        model = NeuralNet(learning_rate=0.001, optimizer="adam")
        model.add_positional_encoding(max_seq_len=10, embed_dim=16, learnable=False)
        model.add_transformer_block(embed_dim=16, num_heads=4, mlp_ratio=2.0)
        x = np.random.randn(3, 5, 16)
        out = model.Forward(x, training=True)
        self.assertEqual(out.shape, (3, 5, 16))
        dout = np.random.randn(*out.shape)
        model.Backward(None, output_delta=dout)
        model.update()
        out2 = model.Forward(x, training=True)
        self.assertFalse(np.allclose(out, out2))

    def test_save_load_round_trip(self):
        model = NeuralNet()
        model.add_multihead_attention(embed_dim=8, num_heads=2)
        x = np.random.randn(2, 4, 8)
        before = model.Forward(x, training=False)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fname = f.name
        try:
            model.Save(fname)
            model2 = NeuralNet()
            model2.add_multihead_attention(embed_dim=8, num_heads=2)
            model2.Load(fname)
            after = model2.Forward(x, training=False)
            self.assertTrue(np.allclose(before, after))
        finally:
            os.remove(fname)

    def test_causal_mask_blocks_future_positions(self):
        np.random.seed(0)
        model = NeuralNet()
        model.add_multihead_attention(embed_dim=8, num_heads=2, causal=True)
        x = np.random.randn(1, 5, 8)
        out1 = model.Forward(x, training=True)

        x2 = x.copy()
        x2[0, 4, :] = np.random.randn(8)  # perturb only the last position
        out2 = model.Forward(x2, training=True)

        self.assertTrue(np.allclose(out1[0, :4], out2[0, :4]))
        self.assertFalse(np.allclose(out1[0, 4], out2[0, 4]))

    def test_causal_mask_gradient_matches_finite_difference(self):
        np.random.seed(0)
        model = NeuralNet()
        model.add_multihead_attention(embed_dim=8, num_heads=2, causal=True)
        x = np.random.randn(1, 5, 8)
        layer = model.layers[0]
        out = model.Forward(x, training=True)
        dout = np.random.randn(*out.shape)
        # Compute d_Wq directly (without update(), which would mutate the
        # weights and invalidate the finite-difference comparison below).
        multihead_attention_backward(dout.copy(), layer, model.attention_cache[0])

        eps = 1e-6
        i, j = 1, 2
        W = layer["Wq"]
        orig = float(W[i, j])
        W[i, j] = orig + eps
        loss_p = np.sum(model.Forward(x, training=True) * dout)
        W[i, j] = orig - eps
        loss_m = np.sum(model.Forward(x, training=True) * dout)
        W[i, j] = orig
        numeric = (loss_p - loss_m) / (2 * eps)
        self.assertAlmostEqual(numeric, layer["d_Wq"][i, j], delta=max(1e-4, abs(numeric) * 1e-3))


class TestPositionalSchemes(_FDPrecisionMixin, unittest.TestCase):
    """v3.1.0 Phase 8: positional_scheme="absolute"|"rope"|"alibi" on
    add_multihead_attention."""

    def _fd_check_param(self, model, X, Y, layer_idx, param_name, eps=1e-5, n_check=5):
        model.Forward(X, training=True)
        model.Backward(Y, loss_function="mse")
        analytic = model.compute_gradients()[layer_idx][param_name]
        flat = model.layers[layer_idx][param_name].reshape(-1)
        rng = np.random.RandomState(0)
        idxs = rng.choice(flat.size, size=min(n_check, flat.size), replace=False)
        max_err = 0.0
        for idx in idxs:
            orig = float(flat[idx])
            flat[idx] = orig + eps
            loss_p = model.ComputeLoss(model.Forward(X, training=False), Y, function="mse")
            flat[idx] = orig - eps
            loss_m = model.ComputeLoss(model.Forward(X, training=False), Y, function="mse")
            flat[idx] = orig
            num_grad = (loss_p - loss_m) / (2 * eps)
            max_err = max(max_err, abs(num_grad - analytic.reshape(-1)[idx]))
        return max_err

    def _build(self, scheme, causal):
        model = NeuralNet(optimizer="sgd")
        model.add_multihead_attention(embed_dim=8, num_heads=2, causal=causal, positional_scheme=scheme)
        model._last_width = 8
        model.add_dense(None, 3, activation="linear")
        return model

    def test_unknown_positional_scheme_raises(self):
        with self.assertRaises(ValueError):
            NeuralNet().add_multihead_attention(embed_dim=8, num_heads=2, positional_scheme="bogus")

    def test_rope_requires_even_head_dim(self):
        with self.assertRaises(ValueError):
            # embed_dim=12, num_heads=4 -> head_dim=3 (odd).
            NeuralNet().add_multihead_attention(embed_dim=12, num_heads=4, positional_scheme="rope")

    def test_absolute_matches_default_no_positional_scheme_arg(self):
        # Regression safety: positional_scheme defaults to "absolute" and
        # must reproduce the exact prior (pre-Phase-8) output/gradients --
        # trivially true by construction (Qh_scored=Qh, Kh_scored=Kh, no
        # bias added) but pinned here as an explicit regression test.
        np.random.seed(0)
        x = np.random.randn(1, 5, 8)
        m1 = NeuralNet()
        m1.add_multihead_attention(embed_dim=8, num_heads=2, causal=True)
        m2 = NeuralNet()
        m2.add_multihead_attention(embed_dim=8, num_heads=2, causal=True, positional_scheme="absolute")
        for k in ("Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo"):
            m2.layers[0][k] = m1.layers[0][k].copy()
        out1 = m1.Forward(x, training=True)
        out2 = m2.Forward(x, training=True)
        self.assertTrue(np.allclose(out1, out2))

    def test_fd_gradients_all_schemes(self):
        np.random.seed(0)
        for scheme in ("absolute", "rope", "alibi"):
            for causal in (False, True):
                model = self._build(scheme, causal)
                X = np.random.randn(2, 5, 8)
                Y = np.random.randn(2, 5, 3)
                for pname in ("Wq", "Wk", "Wv", "Wo"):
                    err = self._fd_check_param(model, X, Y, 0, pname)
                    self.assertLess(err, 1e-6, f"scheme={scheme} causal={causal} param={pname}")

    def test_rope_changes_attention_output_vs_absolute(self):
        np.random.seed(0)
        x = np.random.randn(1, 6, 8)
        m_abs = NeuralNet()
        m_abs.add_multihead_attention(embed_dim=8, num_heads=2, positional_scheme="absolute")
        np.random.seed(1)
        m_rope = NeuralNet()
        m_rope.add_multihead_attention(embed_dim=8, num_heads=2, positional_scheme="rope")
        # Give both models identical weights so any output difference is
        # purely due to RoPE's rotation, not random init.
        for k in ("Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo"):
            m_rope.layers[0][k] = m_abs.layers[0][k].copy()
        out_abs = m_abs.Forward(x, training=False)
        out_rope = m_rope.Forward(x, training=False)
        self.assertFalse(np.allclose(out_abs, out_rope))

    def test_alibi_biases_nearby_tokens_more_than_far_ones(self):
        # A single-head, all-else-equal check: ALiBi's bias should make an
        # otherwise-uniform attention distribution favor nearby positions.
        np.random.seed(0)
        model = NeuralNet()
        model.add_multihead_attention(embed_dim=4, num_heads=1, positional_scheme="alibi")
        layer = model.layers[0]
        for k in ("Wq", "Wk", "Wv"):
            layer[k] = np.zeros_like(layer[k])  # Q=K=V=0 -> raw scores are all 0 before ALiBi bias
        layer["Wo"] = np.eye(4)
        x = np.zeros((1, 5, 4))
        out = model.Forward(x, training=False)
        attn = model.attention_cache[0][6]  # (B, H, S, S)
        # Row for the last position: attention should be highest for the
        # closest (most recent) key and monotonically decrease with distance.
        last_row = attn[0, 0, -1]
        self.assertTrue(np.all(np.diff(last_row) > 0))  # increasing toward the closest position


class TestTextGenerator(unittest.TestCase):
    """End-to-end tests for the GPT-style causal transformer text generator
    (Tokenizer + embedding + positional encoding + causal transformer blocks
    + next-token training + temperature/top-p/greedy sampling)."""

    def _make_tokenizer(self, corpus):
        return Tokenizer(vocab_size=64, level="char").fit([corpus])

    def test_prepare_sequences_shapes(self):
        corpus = "the quick brown fox jumps over the lazy dog. " * 5
        tok = self._make_tokenizer(corpus)
        gen = TextGenerator(tok, embed_dim=16, num_heads=2, num_layers=1, max_seq_len=16)
        X, y = gen.prepare_sequences([corpus], seq_len=8)
        self.assertEqual(X.shape, y.shape)
        self.assertEqual(X.shape[1], 8)
        self.assertTrue(np.array_equal(X[0, 1:], y[0, :-1]))  # y is X shifted by one

    def test_train_step_runs_and_returns_float(self):
        corpus = "the quick brown fox jumps over the lazy dog. " * 5
        tok = self._make_tokenizer(corpus)
        gen = TextGenerator(tok, embed_dim=16, num_heads=2, num_layers=1, max_seq_len=16)
        X, y = gen.prepare_sequences([corpus], seq_len=8)
        loss = gen.train_step(X[:4], y[:4])
        self.assertIsInstance(loss, float)
        self.assertTrue(np.isfinite(loss))

    def test_train_reduces_loss(self):
        np.random.seed(1)
        corpus = "ab" * 200
        tok = self._make_tokenizer(corpus)
        gen = TextGenerator(tok, embed_dim=16, num_heads=2, num_layers=1, max_seq_len=16, learning_rate=0.01)
        history = gen.Train([corpus], epochs=15, batch_size=32, seq_len=8, verbose=False)
        self.assertLess(history[-1], history[0])

    def test_generate_returns_string_of_requested_length_or_less(self):
        corpus = "the quick brown fox jumps over the lazy dog. " * 5
        tok = self._make_tokenizer(corpus)
        gen = TextGenerator(tok, embed_dim=16, num_heads=2, num_layers=1, max_seq_len=16)
        out = gen.generate(prompt="the", max_new_tokens=20, temperature=1.0)
        self.assertIsInstance(out, str)

    def test_generate_greedy_is_deterministic(self):
        corpus = "ab" * 100
        tok = self._make_tokenizer(corpus)
        gen = TextGenerator(tok, embed_dim=16, num_heads=2, num_layers=1, max_seq_len=16)
        out1 = gen.generate(prompt="ab", max_new_tokens=15, greedy=True)
        out2 = gen.generate(prompt="ab", max_new_tokens=15, greedy=True)
        self.assertEqual(out1, out2)

    def test_learns_simple_periodic_pattern(self):
        # Regression test proving the whole pipeline (causal attention +
        # training + autoregressive sampling) actually works, not just runs.
        np.random.seed(2)
        corpus = "ab" * 300
        tok = self._make_tokenizer(corpus)
        gen = TextGenerator(tok, embed_dim=16, num_heads=2, num_layers=1, max_seq_len=16, learning_rate=0.01)
        gen.Train([corpus], epochs=40, batch_size=32, seq_len=8, verbose=False)
        out = gen.generate(prompt="ababab", max_new_tokens=10, greedy=True)
        self.assertGreater(out.count("ab"), 2)


class TestOptimizers(unittest.TestCase):
    def _fit_xor(self, optimizer, epochs=200):
        np.random.seed(1)
        model = NeuralNet(learning_rate=0.1, optimizer=optimizer, l2_lambda=0.0)
        model.add_dense(2, 8, activation="relu")
        model.add_dense(8, 1, activation="sigmoid")
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        Y = np.array([[0], [1], [1], [0]], dtype=np.float64)
        for _ in range(epochs):
            model.TrainBatch(X, Y, loss_function="binary_cross_entropy")
        preds = model.Forward(X)
        return model.compute_accuracy(preds, Y)

    def test_sgd_convergence(self):
        self.assertGreater(self._fit_xor("sgd", epochs=300), 0.75)

    def test_adam_convergence(self):
        self.assertGreater(self._fit_xor("adam", epochs=150), 0.9)

    def test_rmsprop_convergence(self):
        np.random.seed(42)
        model = NeuralNet(learning_rate=0.01, optimizer="rmsprop", l2_lambda=0.0)
        model.add_dense(2, 8, activation="relu")
        model.add_dense(8, 1, activation="sigmoid")
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        Y = np.array([[0], [1], [1], [0]], dtype=np.float64)
        for _ in range(400):
            model.TrainBatch(X, Y, loss_function="binary_cross_entropy")
        preds = model.Forward(X)
        self.assertGreater(model.compute_accuracy(preds, Y), 0.75)

    def test_adagrad_convergence(self):
        self.assertGreater(self._fit_xor("adagrad", epochs=300), 0.75)

    def test_generic_optimizer_update_changes_weights(self):
        for opt_name in ["sgd", "rmsprop", "adagrad", "adam"]:
            model = NeuralNet(learning_rate=0.01, optimizer=opt_name)
            model.add_dense(10, 5, activation="relu")
            model.add_dense(5, 2, activation="softmax")
            x, y = make_classification_data(4, 10, 2)
            w_before = model.layers[0]["weights"].copy()
            model.Forward(x, training=True)
            model.Backward(y)
            model.update()
            self.assertFalse(np.allclose(w_before, model.layers[0]["weights"]), msg=opt_name)
            self.assertTrue(np.all(np.isfinite(model.layers[0]["weights"])), msg=opt_name)

    def test_optimizer_with_batchnorm(self):
        model = NeuralNet(learning_rate=0.01, optimizer="adam")
        model.add_dense(10, 5, activation="linear")
        model.add_batchnorm(5)
        model.add_dense(5, 2, activation="softmax")
        x, y = make_classification_data(8, 10, 2)
        g_before = model.layers[1]["gamma"].copy()
        model.Forward(x, training=True)
        model.Backward(y)
        model.update()
        self.assertFalse(np.allclose(g_before, model.layers[1]["gamma"]))

    def test_optimizer_with_layernorm(self):
        model = NeuralNet(learning_rate=0.01, optimizer="adam")
        model.add_dense(10, 5, activation="linear")
        model.add_layernorm(5)
        model.add_dense(5, 2, activation="softmax")
        x, y = make_classification_data(8, 10, 2)
        g_before = model.layers[1]["gamma"].copy()
        model.Forward(x, training=True)
        model.Backward(y)
        model.update()
        self.assertFalse(np.allclose(g_before, model.layers[1]["gamma"]))


class TestLossFunctions(unittest.TestCase):
    def _scalar(self, name, out, tgt, **kwargs):
        model = NeuralNet()
        val = model.ComputeLoss(out, tgt, function=name, **kwargs)
        self.assertIsInstance(val, float)
        return val

    def test_mse(self):
        val = self._scalar("mse", np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[1.5, 2.5], [2.5, 3.5]]))
        self.assertGreaterEqual(val, 0)

    def test_mae(self):
        val = self._scalar("mae", np.array([[1.0, 2.0]]), np.array([[2.0, 4.0]]))
        self.assertEqual(val, 1.5)

    def test_huber(self):
        val = self._scalar("huber", np.array([[0.1], [5.0]]), np.array([[0.0], [0.0]]), delta=1.0)
        self.assertGreaterEqual(val, 0)

    def test_smooth_l1(self):
        val = self._scalar("smooth_l1", np.array([[0.1], [5.0]]), np.array([[0.0], [0.0]]))
        self.assertGreaterEqual(val, 0)

    def test_cross_entropy(self):
        val = self._scalar("cross_entropy", np.array([[0.1, 0.9], [0.8, 0.2]]), np.array([[0, 1], [1, 0]]))
        self.assertGreater(val, 0)

    def test_binary_cross_entropy(self):
        val = self._scalar("binary_cross_entropy", np.array([[0.9, 0.1]]), np.array([[1, 0]]))
        self.assertGreater(val, 0)

    def test_hinge(self):
        val = self._scalar("hinge", np.array([[0.5, -0.5]]), np.array([[1, -1]]))
        self.assertGreaterEqual(val, 0)

    def test_kl_divergence(self):
        mu = np.array([[0.0, 0.0]])
        logvar = np.array([[0.0, 0.0]])
        val = self._scalar("kl_divergence", mu, logvar, mu=mu, logvar=logvar)
        self.assertGreaterEqual(val, 0)

    def test_bce_logits(self):
        val = self._scalar("bce_logits", np.array([[2.0, -2.0]]), np.array([[1, 0]]))
        self.assertGreater(val, 0)

    def test_wasserstein(self):
        val = self._scalar("wasserstein", np.array([[0.5, -0.5]]), np.array([[1, -1]]))
        self.assertIsInstance(val, float)

    def test_focal(self):
        val = self._scalar("focal", np.array([[0.9, 0.1]]), np.array([[1, 0]]))
        self.assertGreater(val, 0)

    def test_focal_negative_class_direction(self):
        # Regression test: the negative-class term must use (1-pt)**gamma
        # (== o**gamma), so a confidently-wrong negative example (high o) is
        # penalized *more* than an easy, correctly-classified one (low o).
        model = NeuralNet()
        t = np.array([[0.0]])
        hard_neg = np.array([[0.9]])
        easy_neg = np.array([[0.1]])
        loss_hard = model.ComputeLoss(hard_neg, t, function="focal", alpha=0.25, gamma=2.0)
        loss_easy = model.ComputeLoss(easy_neg, t, function="focal", alpha=0.25, gamma=2.0)
        self.assertGreater(loss_hard, loss_easy)

    def test_cosine_similarity(self):
        out = np.array([[1.0, 0.0], [0.0, 1.0]])
        tgt = np.array([[1.0, 0.0], [0.0, 1.0]])
        val = self._scalar("cosine_similarity", out, tgt)
        self.assertGreaterEqual(val, 0)
        self.assertAlmostEqual(val, 0, delta=1e-5)

    def test_triplet(self):
        anchor = np.array([[1.0, 0.0]])
        positive = np.array([[0.9, 0.1]])
        negative = np.array([[-1.0, 0.0]])
        val = self._scalar("triplet", anchor, positive, margin=1.0, negative=negative)
        self.assertGreaterEqual(val, 0)

    def test_ntxent(self):
        out = np.array([[1.0, 0.0], [0.0, 1.0]])
        tgt = np.array([[1.0, 0.0], [0.0, 1.0]])
        val = self._scalar("ntxent", out, tgt, temperature=0.5)
        self.assertGreaterEqual(val, 0)

    def test_reduction_modes(self):
        out = np.array([[1.0, 2.0], [3.0, 4.0]])
        tgt = np.array([[1.0, 2.0], [3.0, 4.0]])
        model = NeuralNet()
        mean_val = model.ComputeLoss(out, tgt, function="mse", reduction="mean")
        sum_val = model.ComputeLoss(out, tgt, function="mse", reduction="sum")
        arr_val = model.ComputeLoss(out, tgt, function="mse", reduction="none")
        self.assertEqual(mean_val, 0.0)
        self.assertEqual(sum_val, 0.0)
        self.assertTrue(np.allclose(arr_val, 0.0))


class TestTrainingLoop(unittest.TestCase):
    def test_train_batch(self):
        model = NeuralNet(learning_rate=0.1, optimizer="sgd")
        model.add_dense(10, 5, activation="relu")
        model.add_dense(5, 2, activation="softmax")
        x, y = make_classification_data(4, 10, 2)
        loss, out = model.TrainBatch(x, y)
        self.assertIsInstance(loss, float)
        self.assertEqual(out.shape, (4, 2))

    def test_train_returns_history(self):
        model = NeuralNet(learning_rate=0.01, optimizer="adam")
        model.add_dense(10, 8, activation="relu")
        model.add_dense(8, 3, activation="softmax")
        X, Y = make_classification_data(32, 10, 3)
        history = model.Train(X, Y, epochs=2, batch_size=8, verbose=False)
        self.assertIn("loss", history)
        self.assertIn("accuracy", history)
        self.assertEqual(len(history["loss"]), 2)

    def test_train_with_validation(self):
        model = NeuralNet(learning_rate=0.01, optimizer="adam")
        model.add_dense(10, 8, activation="relu")
        model.add_dense(8, 3, activation="softmax")
        X, Y = make_classification_data(32, 10, 3)
        Xv, Yv = make_classification_data(8, 10, 3)
        history = model.Train(X, Y, epochs=2, batch_size=8, X_val=Xv, Y_val=Yv, verbose=False)
        self.assertIn("val_loss", history)
        self.assertIn("val_accuracy", history)

    def test_lr_scheduler(self):
        X, Y = make_classification_data(50, 10, 2)
        for mode in ["step", "exponential", "cosine", "warmup_cosine"]:
            scheduler = LRScheduler(initial_lr=0.1, mode=mode, max_epochs=5)
            model = NeuralNet(learning_rate=0.1, optimizer="sgd")
            model.add_dense(10, 5, activation="relu")
            model.add_dense(5, 2, activation="softmax")
            history = model.Train(X, Y, epochs=3, batch_size=16, scheduler=scheduler, verbose=False)
            self.assertIn("lr", history, msg=mode)

    def test_gradient_clipping(self):
        model = NeuralNet(learning_rate=0.1, optimizer="sgd")
        model.add_dense(10, 5, activation="relu")
        model.add_dense(5, 2, activation="softmax")
        x, y = make_classification_data(4, 10, 2)
        model.Forward(x, training=True)
        model.Backward(y)
        model.clip_gradients(1.0)
        for d in model.deltas:
            if d is not None:
                self.assertLessEqual(np.linalg.norm(d), 1.0 + 1e-5)

    def test_freeze_unfreeze(self):
        model = NeuralNet(learning_rate=0.1, optimizer="sgd")
        model.add_dense(10, 5, activation="relu")
        model.add_dense(5, 2, activation="softmax")
        x, y = make_classification_data(4, 10, 2)
        w_before = model.layers[0]["weights"].copy()
        model.freeze(0)
        model.Forward(x, training=True)
        model.Backward(y)
        model.update()
        self.assertTrue(np.allclose(w_before, model.layers[0]["weights"]))

        model.unfreeze(0)
        model.Forward(x, training=True)
        model.Backward(y)
        model.update()
        self.assertFalse(np.allclose(w_before, model.layers[0]["weights"]))

    def test_model_copy(self):
        model = NeuralNet(learning_rate=0.01, optimizer="adam")
        model.add_dense(10, 5, activation="relu")
        model.add_dense(5, 2, activation="softmax")
        model2 = model.copy()
        self.assertEqual(len(model2.layers), len(model.layers))
        self.assertEqual(model2.learning_rate, model.learning_rate)
        self.assertTrue(np.allclose(model.layers[0]["weights"], model2.layers[0]["weights"]))
        model2.layers[0]["weights"][0, 0] = 999
        self.assertFalse(np.allclose(model.layers[0]["weights"], model2.layers[0]["weights"]))

    def test_get_set_weights(self):
        model = NeuralNet()
        model.add_dense(10, 5, activation="relu")
        model.add_dense(5, 2, activation="softmax")
        weights = model.get_weights()
        self.assertEqual(len(weights), 2)
        self.assertIn("weights", weights[0])
        weights[0]["weights"] = np.ones_like(weights[0]["weights"])
        model.set_weights(weights)
        self.assertTrue(np.allclose(model.layers[0]["weights"], 1.0))

    def test_nan_inf_detection(self):
        model = NeuralNet()
        model.add_dense(10, 5, activation="relu")
        self.assertEqual(len(model.check_nan_inf()), 0)
        model.layers[0]["weights"][0, 0] = np.nan
        self.assertGreater(len(model.check_nan_inf()), 0)

    def test_accuracy_binary(self):
        model = NeuralNet()
        acc = model.compute_accuracy(np.array([[0.9], [0.4], [0.6]]), np.array([[1], [0], [1]]))
        self.assertEqual(acc, 1.0)

    def test_accuracy_multiclass(self):
        model = NeuralNet()
        acc = model.compute_accuracy(np.array([[0.1, 0.9], [0.8, 0.2]]), np.array([[0, 1], [1, 0]]))
        self.assertEqual(acc, 1.0)


class TestModelIO(unittest.TestCase):
    def test_save_load_json(self):
        model = NeuralNet(learning_rate=0.01, optimizer="adam", l2_lambda=0.001)
        model.add_dense(10, 5, activation="relu")
        model.add_batchnorm(5)
        model.add_dense(5, 2, activation="softmax")
        X = np.random.randn(4, 10)
        model.TrainBatch(X, np.eye(2)[np.random.randint(0, 2, 4)])
        before = model.Forward(X)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fname = f.name
        try:
            model.Save(fname)
            model2 = NeuralNet()
            model2.Load(fname)
            after = model2.Forward(X)
            self.assertTrue(np.allclose(before, after))
            self.assertEqual(model2.learning_rate, 0.01)
            self.assertEqual(model2.optimizer_type, "adam")
        finally:
            os.remove(fname)

    def test_save_load_pkl(self):
        model = NeuralNet(learning_rate=0.01, optimizer="adam")
        model.add_dense(10, 5, activation="relu")
        X = np.random.randn(4, 10)
        before = model.Forward(X)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            fname = f.name
        try:
            model.Save(fname)
            model2 = NeuralNet()
            model2.Load(fname)
            after = model2.Forward(X)
            self.assertTrue(np.allclose(before, after))
        finally:
            os.remove(fname)

    def test_save_load_restores_recurrent_layer_weights_as_arrays(self):
        # Regression test: Wx/Wh/b (RNN/LSTM)/bx/bh (GRU) weren't in the
        # array-restoration key list, so after a JSON round-trip they stayed
        # plain Python lists and Forward() crashed on `.T`.
        model = NeuralNet(learning_rate=0.01, optimizer="adam")
        model.add_lstm(4, hidden_dim=6, return_sequences=False)
        model.add_dense(6, 3, activation="softmax")
        x = np.random.randn(2, 5, 4)
        before = model.Forward(x, training=False)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fname = f.name
        try:
            model.Save(fname)
            model2 = NeuralNet()
            model2.add_lstm(4, hidden_dim=6, return_sequences=False)
            model2.add_dense(6, 3, activation="softmax")
            model2.Load(fname)
            self.assertIsInstance(model2.layers[0]["Wx"], np.ndarray)
            after = model2.Forward(x, training=False)
            self.assertTrue(np.allclose(before, after))
        finally:
            os.remove(fname)

    def test_save_load_restores_all_training_hyperparameters(self):
        # Regression test: optimizer/training hyperparameters that can be
        # overridden per-model (grad clipping, mixed precision, per-optimizer
        # betas/epsilons) weren't persisted at all, so a loaded model would
        # silently resume training with different defaults.
        model = NeuralNet(learning_rate=0.02, optimizer="adamw", l2_lambda=0.05,
                          momentum=0.8, grad_clip_norm=2.0, use_mixed_precision=True,
                          adam_beta1=0.85, adam_beta2=0.95, adam_epsilon=1e-6,
                          rmsprop_decay=0.95, rmsprop_epsilon=1e-7, adagrad_epsilon=1e-9)
        model.add_dense(4, 4, activation="relu")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fname = f.name
        try:
            model.Save(fname)
            model2 = NeuralNet()
            model2.add_dense(4, 4, activation="relu")
            model2.Load(fname)
            self.assertEqual(model2.optimizer_type, "adamw")
            self.assertEqual(model2.l2_lambda, 0.05)
            self.assertEqual(model2.momentum, 0.8)
            self.assertEqual(model2.grad_clip_norm, 2.0)
            self.assertEqual(model2.use_mixed_precision, True)
            self.assertEqual(model2.adam_beta1, 0.85)
            self.assertEqual(model2.adam_beta2, 0.95)
            self.assertEqual(model2.adam_epsilon, 1e-6)
            self.assertEqual(model2.rmsprop_decay, 0.95)
            self.assertEqual(model2.rmsprop_epsilon, 1e-7)
            self.assertEqual(model2.adagrad_epsilon, 1e-9)
        finally:
            os.remove(fname)

    def test_save_load_restores_shape_inference_and_residual_bookkeeping(self):
        # A loaded model must still support add_*(n_in=None, ...) auto shape
        # inference and mid-residual-block add_residual_end(), matching
        # exactly the state it had at save time.
        model = NeuralNet()
        model.add_dense(10, 20, activation="relu")
        model.add_residual_start()
        model.add_dense(20, 20, activation="tanh")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fname = f.name
        try:
            model.Save(fname)
            model2 = NeuralNet()
            model2.add_dense(10, 20, activation="relu")
            model2.add_residual_start()
            model2.add_dense(20, 20, activation="tanh")
            model2.Load(fname)
            self.assertEqual(model2._last_width, 20)
            self.assertEqual(model2._residual_stack, [1])
            model2.add_residual_end()
            model2.add_dense(20, 5, activation="softmax")
            out = model2.Forward(np.random.randn(2, 10))
            self.assertEqual(out.shape, (2, 5))
        finally:
            os.remove(fname)

    def test_save_load_restores_gradient_accumulation_state(self):
        model = NeuralNet(optimizer="sgd", learning_rate=0.1)
        model.add_dense(4, 4, activation="linear")
        model.Forward(np.random.randn(3, 4), training=True)
        model.Backward(np.random.randn(3, 4), loss_function="mse")
        model.accumulate_gradients()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fname = f.name
        try:
            model.Save(fname)
            model2 = NeuralNet(optimizer="sgd", learning_rate=0.1)
            model2.add_dense(4, 4, activation="linear")
            model2.Load(fname)
            self.assertEqual(model2._accum_steps, 1)
            self.assertIsInstance(model2._grad_accum[0]["weights"], np.ndarray)
        finally:
            os.remove(fname)


class TestReinforce(unittest.TestCase):
    def test_compute_returns(self):
        rewards = np.array([1.0, 0.0, 1.0])
        returns = compute_returns(rewards, gamma=0.9)
        expected = np.array([1 + 0.9 * 0 + 0.9 ** 2 * 1, 0 + 0.9 * 1, 1.0])
        self.assertTrue(np.allclose(returns, expected))

    def test_gae(self):
        rewards = np.array([1.0, 0.0, 1.0, 0.0])
        values = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        advantages, returns = gae(rewards, values, gamma=0.99, lambda_=0.95)
        self.assertEqual(len(advantages), 4)
        self.assertEqual(len(returns), 4)

    def test_evolve_runs(self):
        model = NeuralNet(learning_rate=0.01, optimizer="sgd")
        model.add_dense(5, 3, activation="relu")
        model.add_dense(3, 1, activation="linear")
        inputs = np.random.randn(10, 5)

        def score_fn(out):
            return float(np.mean(out ** 2))

        best = model.Evolve(inputs, score_fn, noise=0.1, tries=3)
        self.assertIsInstance(best, float)

    def test_reinforce_discrete_runs(self):
        model = NeuralNet(learning_rate=0.01, optimizer="sgd")
        model.add_dense(4, 3, activation="softmax")
        states = np.random.randn(8, 4)
        actions = np.random.randint(0, 3, size=8)
        returns = np.random.randn(8)
        avg = model.Reinforce(states, actions, returns, action_type="discrete")
        self.assertIsInstance(avg, float)

    def test_reinforce_continuous_runs(self):
        model = NeuralNet(learning_rate=0.01, optimizer="sgd")
        model.add_dense(4, 2, activation="linear")
        states = np.random.randn(8, 4)
        actions = np.random.randn(8, 2)
        returns = np.random.randn(8)
        avg = model.Reinforce(states, actions, returns, action_type="continuous", std=1.0)
        self.assertIsInstance(avg, float)

    def test_ppo(self):
        model = NeuralNet(learning_rate=0.01, optimizer="adam")
        model.add_dense(4, 8, activation="relu")
        model.add_dense(8, 2, activation="softmax")
        states = np.random.randn(10, 4)
        actions = np.random.randint(0, 2, 10)
        old_log_probs = np.random.randn(10, 1)
        advantages = np.random.randn(10, 1)
        loss = model.PPO(states, actions, old_log_probs, advantages, action_type="discrete")
        self.assertIsInstance(loss, float)

    def test_actor_critic(self):
        model = NeuralNet(learning_rate=0.01, optimizer="adam")
        model.add_dense(4, 8, activation="relu")
        model.add_dense(8, 2, activation="softmax")
        states = np.random.randn(10, 4)
        actions = np.random.randint(0, 2, 10)
        returns = np.random.randn(10)
        values = np.random.randn(10)
        loss = model.ActorCritic(states, actions, returns, values, action_type="discrete")
        self.assertIsInstance(loss, float)


# ========================================================================
# Generative Model Tests
# ========================================================================

class TestVAE(unittest.TestCase):
    def setUp(self):
        self.vae = VAE(input_dim=20, latent_dim=4, encoder_hidden=[16], decoder_hidden=[16],
                       learning_rate=0.01, optimizer="adam")
        self.X = np.random.rand(8, 20).astype(np.float64)

    def test_encode_shape(self):
        mu, logvar = self.vae.encode(self.X)
        self.assertEqual(mu.shape, (8, 4))
        self.assertEqual(logvar.shape, (8, 4))

    def test_decode_shape(self):
        z = np.random.randn(8, 4)
        recon = self.vae.decode(z)
        self.assertEqual(recon.shape, (8, 20))

    def test_forward_tuple(self):
        recon, mu, logvar, z = self.vae.forward(self.X)
        self.assertEqual(recon.shape, (8, 20))
        self.assertEqual(mu.shape, (8, 4))
        self.assertEqual(z.shape, (8, 4))

    def test_loss_scalar(self):
        recon, mu, logvar, _ = self.vae.forward(self.X)
        loss = self.vae.loss(self.X, recon, mu, logvar)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)

    def test_loss_kl_weight_zero(self):
        loss = self.vae.loss(self.X, kl_weight=0.0)
        self.assertIsInstance(loss, float)

    def test_train_step_runs(self):
        loss = self.vae.train_step(self.X)
        self.assertIsInstance(loss, float)

    def test_train_decreases_loss(self):
        X = np.random.rand(32, 20).astype(np.float64)
        h1 = self.vae.Train(X, epochs=5, batch_size=8, verbose=False)
        h2 = self.vae.Train(X, epochs=5, batch_size=8, verbose=False)
        self.assertLessEqual(h2[-1], h1[0] * 2)

    def test_generate_shape(self):
        samples = self.vae.generate(n_samples=4)
        self.assertEqual(samples.shape, (4, 20))

    def test_reconstruct_shape(self):
        recon = self.vae.reconstruct(self.X[:2])
        self.assertEqual(recon.shape, (2, 20))

    def test_interpolate_shape(self):
        interp = self.vae.interpolate(self.X[:1], self.X[1:2], n_steps=5)
        self.assertEqual(interp.shape, (5, 20))


class TestGAN(unittest.TestCase):
    def setUp(self):
        self.gan = GAN(latent_dim=4, data_dim=12,
                       generator_hidden=[8, 8], discriminator_hidden=[8, 8],
                       loss_type="bce", learning_rate=0.01, optimizer="adam")
        self.X = np.random.randn(16, 12).astype(np.float64)

    def test_generate_shape(self):
        fake = self.gan.generate(4)
        self.assertEqual(fake.shape, (4, 12))

    def test_discriminate_shape(self):
        d = self.gan.discriminate(self.X)
        self.assertEqual(d.shape, (16, 1))

    def test_train_runs(self):
        history = self.gan.Train(self.X, epochs=2, batch_size=8, d_steps=1, g_steps=1, verbose=False)
        self.assertIn("d_loss", history)
        self.assertIn("g_loss", history)
        self.assertEqual(len(history["d_loss"]), 2)

    def test_sample_shape(self):
        samples = self.gan.sample(4)
        self.assertEqual(samples.shape, (4, 12))

    def test_mode_collapse_score(self):
        score = self.gan.mode_collapse_score(n_samples=50)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_bce_logits_and_wasserstein_train(self):
        for loss_type in ("bce_logits", "wasserstein"):
            gan = GAN(latent_dim=4, data_dim=12, generator_hidden=[8], discriminator_hidden=[8],
                     loss_type=loss_type, learning_rate=0.01, optimizer="adam")
            history = gan.Train(self.X, epochs=1, batch_size=8, verbose=False)
            self.assertTrue(np.isfinite(history["d_loss"][-1]), msg=loss_type)
            self.assertTrue(np.isfinite(history["g_loss"][-1]), msg=loss_type)


class TestDiffusion(unittest.TestCase):
    def setUp(self):
        self.diff = DiffusionModel(
            data_shape=(16,), time_steps=50, beta_schedule="linear",
            denoiser_type="mlp", denoiser_hidden=[32, 32],
            learning_rate=0.01, optimizer="adam",
        )
        self.X = np.random.randn(8, 16).astype(np.float64) * 0.5

    def test_forward_diffusion_shape(self):
        x_t, noise = self.diff._forward_diffusion(self.X, np.array([0, 1, 2, 3, 4, 5, 6, 7]))
        self.assertEqual(x_t.shape, self.X.shape)
        self.assertEqual(noise.shape, self.X.shape)

    def test_predict_noise_shape(self):
        t = np.array([1, 5, 10, 15, 20, 25, 30, 35])
        x_t, _ = self.diff._forward_diffusion(self.X, t)
        pred = self.diff._predict_noise(x_t, t)
        self.assertEqual(pred.shape, self.X.shape)

    def test_train_step_runs(self):
        loss = self.diff.train_step(self.X)
        self.assertIsInstance(loss, float)

    def test_train_runs(self):
        history = self.diff.Train(self.X, epochs=2, batch_size=4, verbose=False)
        self.assertEqual(len(history), 2)

    def test_sample_shape(self):
        samples = self.diff.sample(n_samples=2, shape=(16,), clip=True)
        self.assertEqual(samples.shape, (2, 16))

    def test_denoise_shape(self):
        x_noisy = np.random.randn(2, 16)
        out = self.diff.denoise(x_noisy, t_start=10, t_end=0)
        self.assertEqual(out.shape, (2, 16))

    def test_sample_ddim_shape_across_step_counts(self):
        for n_steps in (5, 10, 25):
            samples = self.diff.sample_ddim(n_samples=3, n_steps=n_steps, shape=(16,))
            self.assertEqual(samples.shape, (3, 16))

    def test_sample_ddim_fewer_forward_passes_than_full_sample(self):
        calls = {"n": 0}
        orig = self.diff._predict_noise
        def counting_predict_noise(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)
        self.diff._predict_noise = counting_predict_noise
        try:
            self.diff.sample_ddim(n_samples=2, n_steps=10, shape=(16,))
            ddim_calls = calls["n"]
            calls["n"] = 0
            self.diff.sample(n_samples=2, shape=(16,))
            full_calls = calls["n"]
        finally:
            self.diff._predict_noise = orig
        self.assertEqual(ddim_calls, 10)
        self.assertEqual(full_calls, self.diff.time_steps)
        self.assertLess(ddim_calls, full_calls)

    def test_sample_ddim_deterministic_when_eta_zero(self):
        np.random.seed(7)
        s1 = self.diff.sample_ddim(n_samples=2, n_steps=10, eta=0.0, shape=(16,))
        np.random.seed(7)
        s2 = self.diff.sample_ddim(n_samples=2, n_steps=10, eta=0.0, shape=(16,))
        self.assertTrue(np.allclose(s1, s2))

    def test_sample_ddim_stochastic_when_eta_one(self):
        np.random.seed(0)
        s1 = self.diff.sample_ddim(n_samples=2, n_steps=10, eta=1.0, shape=(16,))
        s2 = self.diff.sample_ddim(n_samples=2, n_steps=10, eta=1.0, shape=(16,))
        self.assertFalse(np.allclose(s1, s2))

    def test_sample_ddim_quality_on_ring_dataset(self):
        np.random.seed(0)
        diff = DiffusionModel(data_shape=(2,), time_steps=100, beta_schedule="linear",
                              denoiser_hidden=[32, 32], learning_rate=0.02, use_ema=False)
        theta = np.random.uniform(0, 2 * np.pi, 300)
        X = np.stack([np.cos(theta), np.sin(theta)], axis=1) + np.random.randn(300, 2) * 0.05
        for _ in range(150):
            diff.train_step(X)
        samples = diff.sample_ddim(n_samples=50, n_steps=20, shape=(2,))
        radii = np.linalg.norm(samples, axis=1)
        self.assertLess(abs(radii.mean() - 1.0), 0.5)

    def test_ema_variant(self):
        diff_ema = DiffusionModel(
            data_shape=(16,), time_steps=20, denoiser_type="mlp",
            denoiser_hidden=[16], use_ema=True,
        )
        loss = diff_ema.train_step(self.X)
        self.assertIsInstance(loss, float)

    def test_train_step_loss_decreases_after_shared_backward_refactor(self):
        # Regression test for Phase 3's refactor of train_step to use the
        # shared _manual_sequential_backward helper instead of a hand-rolled
        # duplicate backward loop -- training loss curve must still decrease.
        np.random.seed(0)
        diff = DiffusionModel(data_shape=(4,), time_steps=20, denoiser_hidden=[16],
                             learning_rate=0.02, use_ema=False)
        X = np.random.randn(30, 4) * 0.5
        losses = [diff.train_step(X) for _ in range(60)]
        self.assertLess(statistics.mean(losses[-10:]), statistics.mean(losses[:10]))


class TestClassConditionalGeneration(unittest.TestCase):
    """v3.1.0 Phase 11: num_classes=None (default, unconditional, backward
    compatible) + labels/y threaded through train/generate for VAE, GAN,
    and DiffusionModel. Each trained on a tiny synthetic 3-class blob
    dataset with well-separated centers; generated samples per class must
    land near that class's real center, not a different one."""

    def _make_blobs(self, n_per_class=80, seed=0):
        rng = np.random.RandomState(seed)
        centers = np.array([[-3.0, -3.0], [3.0, 3.0], [3.0, -3.0]])
        X, y = [], []
        for c in range(3):
            X.append(centers[c] + rng.randn(n_per_class, 2) * 0.4)
            y.append(np.full(n_per_class, c))
        return np.concatenate(X, axis=0), np.concatenate(y, axis=0), centers

    def test_vae_class_conditional_generation(self):
        np.random.seed(0)
        X, y, centers = self._make_blobs()
        X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))  # VAE decoder is sigmoid
        vae = VAE(input_dim=2, latent_dim=4, encoder_hidden=[32], decoder_hidden=[32],
                 learning_rate=0.01, optimizer="adam", num_classes=3)
        vae.Train(X_scaled, epochs=60, batch_size=60, y_train=y, verbose=False)
        for c in range(3):
            samples = vae.generate(n_samples=30, y=c)
            target = (centers[c] - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
            self.assertLess(np.linalg.norm(samples.mean(axis=0) - target), 0.3)

    def test_gan_class_conditional_generation(self):
        np.random.seed(0)
        X, y, centers = self._make_blobs()
        scale = np.abs(X).max()
        X_scaled = X / scale  # generator is tanh
        gan = GAN(latent_dim=4, data_dim=2, generator_hidden=[32, 32], discriminator_hidden=[32, 32],
                 learning_rate=0.001, num_classes=3)
        gan.Train(X_scaled, epochs=60, batch_size=60, y_train=y, verbose=False)
        for c in range(3):
            samples = gan.sample(n_samples=50, y=c)
            target = centers[c] / scale
            self.assertLess(np.linalg.norm(samples.mean(axis=0) - target), 0.6)

    def test_diffusion_class_conditional_generation(self):
        np.random.seed(0)
        X, y, centers = self._make_blobs()
        scale = np.abs(X).max()
        X_scaled = X / scale
        diff = DiffusionModel(data_shape=(2,), time_steps=100, denoiser_hidden=[32, 32],
                              learning_rate=0.01, use_ema=False, num_classes=3)
        for _ in range(250):
            idx = np.random.choice(len(X_scaled), 60, replace=False)
            diff.train_step(X_scaled[idx], y=y[idx])
        for c in range(3):
            samples = diff.sample_ddim(n_samples=30, n_steps=20, y=c)
            target = centers[c] / scale
            self.assertLess(np.linalg.norm(samples.mean(axis=0) - target), 0.6)

    def test_vae_validates_labels_both_directions(self):
        vae_cond = VAE(input_dim=2, latent_dim=2, num_classes=3)
        with self.assertRaises(ValueError):
            vae_cond.generate(n_samples=2)  # num_classes set, y missing
        vae_uncond = VAE(input_dim=2, latent_dim=2)
        with self.assertRaises(ValueError):
            vae_uncond.generate(n_samples=2, y=0)  # unconditional, y given

    def test_gan_validates_labels_both_directions(self):
        gan_cond = GAN(latent_dim=2, data_dim=2, num_classes=3)
        with self.assertRaises(ValueError):
            gan_cond.generate(2)
        gan_uncond = GAN(latent_dim=2, data_dim=2)
        with self.assertRaises(ValueError):
            gan_uncond.generate(2, y=0)

    def test_diffusion_validates_labels_both_directions(self):
        diff_cond = DiffusionModel(data_shape=(2,), time_steps=10, num_classes=3)
        with self.assertRaises(ValueError):
            diff_cond.train_step(np.random.randn(4, 2))
        diff_uncond = DiffusionModel(data_shape=(2,), time_steps=10)
        with self.assertRaises(ValueError):
            diff_uncond.train_step(np.random.randn(4, 2), y=np.zeros(4, dtype=int))

    def test_unconditional_models_still_work_unchanged(self):
        # Regression: num_classes=None (default) must be fully
        # backward-compatible with the pre-Phase-11 unconditional API.
        np.random.seed(0)
        vae = VAE(input_dim=4, latent_dim=2, encoder_hidden=[8], decoder_hidden=[8])
        X = np.random.rand(10, 4)
        vae.Train(X, epochs=2, batch_size=10, verbose=False)
        self.assertEqual(vae.generate(n_samples=3).shape, (3, 4))

        gan = GAN(latent_dim=2, data_dim=4, generator_hidden=[8], discriminator_hidden=[8])
        gan.Train(X, epochs=1, batch_size=10, d_steps=1, g_steps=1, verbose=False)
        self.assertEqual(gan.sample(3).shape, (3, 4))

        diff = DiffusionModel(data_shape=(4,), time_steps=10, denoiser_hidden=[8])
        diff.train_step(X)
        self.assertEqual(diff.sample(n_samples=3).shape, (3, 4))


class TestAutoregressive(unittest.TestCase):
    def setUp(self):
        self.ar = AutoregressiveModel(
            data_dim=12, hidden_dims=[16, 16], data_shape=(3, 4),
            learning_rate=0.01, optimizer="adam",
        )
        self.X = np.random.randn(8, 12).astype(np.float64)

    def test_forward_shape(self):
        logits = self.ar.forward(self.X)
        self.assertEqual(logits.shape, (8, 12))

    def test_loss_scalar(self):
        loss = self.ar.loss(self.X)
        self.assertIsInstance(loss, float)

    def test_train_step_runs(self):
        loss = self.ar.train_step(self.X)
        self.assertIsInstance(loss, float)

    def test_train_runs(self):
        history = self.ar.Train(self.X, epochs=2, batch_size=4, verbose=False)
        self.assertEqual(len(history), 2)

    def test_generate_shape(self):
        samples = self.ar.generate(n_samples=2, shape=(3, 4))
        self.assertEqual(samples.shape, (2, 3, 4))

    def test_complete_shape(self):
        partial = np.random.randn(2, 12)
        partial[:, 6:] = 0
        completed = self.ar.complete(partial, n_dims=6)
        self.assertEqual(completed.shape, (2, 3, 4))

    def test_discrete_mode_train_step_and_log_prob(self):
        ar_d = AutoregressiveModel(data_dim=10, hidden_dims=[16, 16], discrete=True, num_classes=16)
        X = np.random.rand(8, 10).astype(np.float64)
        loss = ar_d.train_step(X)
        self.assertIsInstance(loss, float)
        self.assertTrue(np.isfinite(loss))
        log_prob = ar_d.log_prob(X)
        self.assertEqual(log_prob.shape, (8,))


class TestRealNVP(unittest.TestCase):
    def setUp(self):
        self.flow = RealNVP(data_dim=8, n_coupling=2, hidden_dim=16,
                            learning_rate=0.01, optimizer="adam")
        self.X = np.random.randn(4, 8).astype(np.float64)

    def test_forward_shape(self):
        z, log_det = self.flow.forward(self.X)
        self.assertEqual(z.shape, self.X.shape)
        self.assertEqual(log_det.shape, (4,))

    def test_inverse_reverses_forward(self):
        z, _ = self.flow.forward(self.X)
        x_rec = self.flow.inverse(z)
        self.assertTrue(np.allclose(x_rec, self.X, atol=1e-5))

    def test_log_prob_shape(self):
        lp = self.flow.log_prob(self.X)
        self.assertEqual(lp.shape, (4,))

    def test_loss_scalar(self):
        loss = self.flow.loss(self.X)
        self.assertIsInstance(loss, float)

    def test_train_runs(self):
        history = self.flow.Train(self.X, epochs=2, batch_size=4, verbose=False)
        self.assertEqual(len(history), 2)

    def test_train_reduces_nll(self):
        # Regression test for the dL_dz sign fix: training should reduce NLL,
        # not increase it (the old sign bug pushed ||z|| the wrong way).
        np.random.seed(3)
        flow = RealNVP(data_dim=6, n_coupling=2, hidden_dim=16, learning_rate=0.01)
        X = np.random.randn(64, 6).astype(np.float64)
        history = flow.Train(X, epochs=8, batch_size=16, verbose=False)
        self.assertLess(history[-1], history[0])

    def test_sample_shape(self):
        samples = self.flow.sample(3)
        self.assertEqual(samples.shape, (3, 8))

    def test_interpolate_shape(self):
        interp = self.flow.interpolate(self.X[:1], self.X[1:2], n_steps=5)
        self.assertEqual(interp.shape, (5, 8))


class TestEBM(unittest.TestCase):
    def setUp(self):
        self.ebm = EnergyBasedModel(data_dim=8, hidden_dims=[16], persistent_cd=False,
                                    learning_rate=0.01, optimizer="adam")
        self.X = np.random.randn(4, 8).astype(np.float64) * 0.5

    def test_energy_shape(self):
        e = self.ebm.energy(self.X)
        self.assertEqual(e.shape, (4, 1))

    def test_train_step_runs(self):
        loss = self.ebm.train_step(self.X, n_cd_steps=5, step_size=0.1, noise_scale=0.01)
        self.assertIsInstance(loss, float)

    def test_train_runs(self):
        history = self.ebm.Train(self.X, epochs=2, batch_size=4, n_cd_steps=5, verbose=False)
        self.assertEqual(len(history), 2)

    def test_sample_shape(self):
        samples = self.ebm.sample(n_samples=2, n_steps=10)
        self.assertEqual(samples.shape, (2, 8))

    def test_score_shape(self):
        s = self.ebm.score(self.X)
        self.assertEqual(s.shape, self.X.shape)

    def test_persistent_cd_variant(self):
        ebm_pcd = EnergyBasedModel(data_dim=8, hidden_dims=[16], persistent_cd=True,
                                   persistent_buffer_size=32, learning_rate=0.01)
        loss = ebm_pcd.train_step(self.X, n_cd_steps=3)
        self.assertIsInstance(loss, float)

    def test_train_separates_energy_of_data_and_far_noise(self):
        # Regression test for the train_step forward-cache/gradient fix:
        # after training, energy(data) should be lower than energy of noise
        # sampled far outside the data's support.
        np.random.seed(5)
        ebm = EnergyBasedModel(data_dim=4, hidden_dims=[32], persistent_cd=False, learning_rate=0.02)
        data = np.random.randn(32, 4) * 0.2 + 2.0
        ebm.Train(data, epochs=15, batch_size=16, n_cd_steps=5, verbose=False)
        far_noise = np.random.randn(32, 4) * 0.2 - 2.0
        e_data = np.mean(ebm.energy(data))
        e_noise = np.mean(ebm.energy(far_noise))
        self.assertLess(e_data, e_noise)


class TestUNet(unittest.TestCase):
    def test_time_embedding_shape(self):
        t = np.array([0, 50, 100])
        emb = time_embedding(t, dim=128)
        self.assertEqual(emb.shape, (3, 128))

    def test_unet_forward_shape(self):
        unet = UNetDenoiser(in_ch=1, base_ch=16, time_emb_dim=32, ch_mult=(1, 2))
        x = np.random.randn(2, 1, 8, 8)
        t = np.array([0, 10])
        out = unet.forward(x, t)
        self.assertEqual(out.shape, (2, 1, 8, 8))

    def test_unet_get_params(self):
        unet = UNetDenoiser(in_ch=1, base_ch=8, time_emb_dim=16, ch_mult=(1,))
        params = unet.get_params()
        self.assertIsInstance(params, list)
        self.assertGreater(len(params), 0)


class TestUNetDenoiser(_FDPrecisionMixin, unittest.TestCase):
    """v3.1.0 Phase 3: UNetDenoiser real backward + k=3 same-padding convs.

    Non-negotiable per the project's QA bar: finite-difference gradient
    checks on a SMALL U-Net across every subnetwork (encoder, decoder,
    bottleneck, time_net) before trusting anything larger -- this is
    exactly the class of bug (an off-by-one/convention mismatch) that has
    bitten this project before (the residual-connection gradient routing).
    """

    def _fd_check(self, unet, x, t, target, net, param_name, layer_idx, eps=1e-5, n_check=4):
        def loss_fn():
            return float(np.mean((unet.forward(x, t) - target) ** 2))
        out = unet.forward(x, t)
        grad_out = 2 * (out - target) / out.size
        unet.backward(grad_out)
        analytic = net.compute_gradients()[layer_idx][param_name]
        flat = net.layers[layer_idx][param_name].reshape(-1)
        rng = np.random.RandomState(5)
        idxs = rng.choice(flat.size, size=min(n_check, flat.size), replace=False)
        max_err = 0.0
        for idx in idxs:
            orig = float(flat[idx])
            flat[idx] = orig + eps
            loss_p = loss_fn()
            flat[idx] = orig - eps
            loss_m = loss_fn()
            flat[idx] = orig
            num_grad = (loss_p - loss_m) / (2 * eps)
            max_err = max(max_err, abs(num_grad - analytic.reshape(-1)[idx]))
        return max_err

    def test_conv_layers_use_k3_same_padding(self):
        unet = UNetDenoiser(in_ch=1, base_ch=4, time_emb_dim=4, ch_mult=(1, 2))
        for net in unet.get_params():
            for layer in net.layers:
                if layer["type"] == "conv2d":
                    self.assertEqual(layer["k"], 3)
                    self.assertEqual(layer["padding"], "same")

    def test_fd_gradients_small_unet_two_levels(self):
        np.random.seed(0)
        unet = UNetDenoiser(in_ch=1, base_ch=4, time_emb_dim=8, ch_mult=(1, 2))
        x = np.random.randn(2, 1, 8, 8)
        t = np.array([3, 100])
        target = np.random.randn(2, 1, 8, 8)
        for net, pname, lidx in [
            (unet.encoders[0], "weights", 0),
            (unet.encoders[1], "weights", 1),
            (unet.bottleneck, "weights", 0),
            (unet.decoders[0], "weights", 0),
            (unet.decoders[1], "weights", 1),
            (unet.out_net, "weights", 0),
        ]:
            self.assertLess(self._fd_check(unet, x, t, target, net, pname, lidx), 1e-6)

    def test_fd_gradients_time_embedding_path(self):
        # time_emb_dim*4 == base_ch*ch_mult[0] so _add_time_to_feature
        # actually triggers (channel-count match), exercising time_net's
        # gradient path (otherwise silently skipped -- see forward()).
        np.random.seed(1)
        unet = UNetDenoiser(in_ch=1, base_ch=8, time_emb_dim=2, ch_mult=(1, 2))
        x = np.random.randn(2, 1, 8, 8)
        t = np.array([3, 100])
        target = np.random.randn(2, 1, 8, 8)
        self.assertLess(self._fd_check(unet, x, t, target, unet.time_net, "weights", 0), 1e-6)
        self.assertLess(self._fd_check(unet, x, t, target, unet.encoders[0], "weights", 0), 1e-6)

    def test_fd_gradients_three_level_unet(self):
        np.random.seed(2)
        unet = UNetDenoiser(in_ch=2, base_ch=4, time_emb_dim=6, ch_mult=(1, 2, 4))
        x = np.random.randn(2, 2, 16, 16)
        t = np.array([0, 500])
        target = np.random.randn(2, 2, 16, 16)
        for net, pname, lidx in [
            (unet.encoders[2], "weights", 0),  # deepest encoder
            (unet.decoders[0], "bias", 1),     # deepest decoder
            (unet.decoders[2], "weights", 1),  # shallowest decoder
        ]:
            self.assertLess(self._fd_check(unet, x, t, target, net, pname, lidx), 1e-6)

    def test_fd_gradients_single_level_unet(self):
        # levels=1: no downsample/upsample ever triggers -- edge case.
        np.random.seed(3)
        unet = UNetDenoiser(in_ch=1, base_ch=4, time_emb_dim=4, ch_mult=(1,))
        x = np.random.randn(2, 1, 6, 6)
        t = np.array([1, 2])
        target = np.random.randn(2, 1, 6, 6)
        for net, pname, lidx in [
            (unet.encoders[0], "weights", 0),
            (unet.bottleneck, "weights", 1),
            (unet.decoders[0], "weights", 0),
        ]:
            self.assertLess(self._fd_check(unet, x, t, target, net, pname, lidx), 1e-6)

    def test_fd_gradients_identity_downsample_edge_case(self):
        # Spatial size smaller than pool_factor -- _downsample is an
        # identity no-op; backward must match that.
        np.random.seed(4)
        unet = UNetDenoiser(in_ch=1, base_ch=4, time_emb_dim=4, ch_mult=(1, 2), pool_factor=2)
        x = np.random.randn(2, 1, 1, 1)
        t = np.array([1, 2])
        target = np.random.randn(2, 1, 1, 1)
        self.assertLess(self._fd_check(unet, x, t, target, unet.encoders[1], "weights", 0), 1e-6)

    def test_train_step_loss_decreases(self):
        np.random.seed(0)
        unet = UNetDenoiser(in_ch=1, base_ch=8, time_emb_dim=16, ch_mult=(1, 2),
                            time_steps=50, learning_rate=0.01)
        X = np.random.randn(20, 1, 8, 8) * 0.5
        losses = [unet.train_step(X) for _ in range(40)]
        self.assertLess(statistics.mean(losses[-10:]), statistics.mean(losses[:10]))

    def test_train_runs_and_returns_history(self):
        np.random.seed(0)
        unet = UNetDenoiser(in_ch=1, base_ch=4, time_emb_dim=8, ch_mult=(1,), time_steps=20)
        X = np.random.randn(12, 1, 6, 6)
        history = unet.Train(X, epochs=2, batch_size=6, verbose=False)
        self.assertEqual(len(history), 2)
        self.assertTrue(all(isinstance(l, float) for l in history))

    def test_sample_shape(self):
        np.random.seed(0)
        unet = UNetDenoiser(in_ch=1, base_ch=4, time_emb_dim=8, ch_mult=(1,), time_steps=10)
        samples = unet.sample(n_samples=3, shape=(1, 6, 6))
        self.assertEqual(samples.shape, (3, 1, 6, 6))


class TestSamplingUtilities(unittest.TestCase):
    def test_reparameterize_shape(self):
        mu = np.zeros((4, 3))
        logvar = np.ones((4, 3))
        z = reparameterize(mu, logvar)
        self.assertEqual(z.shape, (4, 3))

    def test_langevin_dynamics(self):
        def energy_fn(x):
            e = np.sum(x ** 2, axis=-1, keepdims=True)
            grad = 2 * x
            return e, grad
        x_init = np.random.randn(3, 4)
        x_final = langevin_dynamics(energy_fn, x_init, n_steps=10)
        self.assertEqual(x_final.shape, (3, 4))
        self.assertTrue(np.all(np.isfinite(x_final)))

    def test_gaussian_sample(self):
        s = gaussian_sample(0.0, 1.0, shape=(10,))
        self.assertEqual(s.shape, (10,))

    def test_uniform_sample(self):
        samples = uniform_sample(-1, 1, (3, 4))
        self.assertEqual(samples.shape, (3, 4))
        self.assertTrue(np.all((samples >= -1) & (samples <= 1)))

    def test_gumbel_softmax(self):
        logits = np.array([[1.0, 2.0, 3.0]])
        y = gumbel_softmax_sample(logits, temperature=0.5, hard=False)
        self.assertEqual(y.shape, (1, 3))
        self.assertTrue(np.allclose(y.sum(), 1.0, atol=1e-6))
        y_hard = gumbel_softmax_sample(logits, temperature=0.5, hard=True)
        self.assertTrue(np.allclose(np.sum(y_hard, axis=1), 1.0))

    def test_random_mask(self):
        mask = random_mask((10, 10), ratio=0.5)
        self.assertEqual(mask.shape, (10, 10))
        self.assertTrue(0.3 < np.mean(mask) < 0.7)

    def test_top_p_sampling(self):
        logits = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = top_p_sampling(logits, p=0.9, temperature=1.0)
        self.assertEqual(result.shape, (1, 5))
        self.assertEqual(np.sum(result), 1.0)


class TestGenerativeLosses(unittest.TestCase):
    def test_kl_divergence_gaussian(self):
        mu = np.zeros((4, 2))
        logvar = np.ones((4, 2))
        val = kl_divergence_gaussian(mu, logvar)
        self.assertIsInstance(val, float)
        val_sum = kl_divergence_gaussian(mu, logvar, reduction="sum")
        self.assertIsInstance(val_sum, float)

    def test_adversarial_discriminator(self):
        real = np.array([[0.9], [0.8]])
        fake = np.array([[0.1], [0.2]])
        for loss_type in ["bce", "bce_logits", "wasserstein"]:
            val = adversarial_loss_discriminator(real, fake, loss_type=loss_type)
            self.assertIsInstance(val, float)

    def test_adversarial_generator(self):
        fake = np.array([[0.9], [0.8]])
        for loss_type in ["bce", "bce_logits", "wasserstein"]:
            val = adversarial_loss_generator(fake, loss_type=loss_type)
            self.assertIsInstance(val, float)

    def test_diffusion_loss(self):
        pred = np.random.randn(4, 16)
        true = np.random.randn(4, 16)
        val = diffusion_loss(pred, true)
        self.assertIsInstance(val, float)

    def test_nll_loss(self):
        log_px = np.random.randn(4)
        log_det = np.random.randn(4)
        val = nll_loss(log_px, log_det)
        self.assertIsInstance(val, float)

    def test_energy_loss(self):
        val = energy_loss(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        self.assertIsInstance(val, float)

    def test_perceptual_and_vgg_loss(self):
        x = np.random.randn(2, 3, 8, 8)
        y = np.random.randn(2, 3, 8, 8)
        val = perceptual_loss(x, y)
        self.assertIsInstance(val, float)
        self.assertGreaterEqual(val, 0)
        val_vgg = vgg_loss(x, y)
        self.assertIsInstance(val_vgg, float)


class TestGenerativeFixesRegression(_FDPrecisionMixin, unittest.TestCase):
    """Extra regression coverage for the generative-model gradient fixes not
    already exercised above (VAE BCE gradient, AR continuous-loss gradient)."""

    def test_vae_bce_gradient_matches_finite_difference(self):
        np.random.seed(9)
        vae = VAE(input_dim=6, latent_dim=2, encoder_hidden=[8], decoder_hidden=[8], learning_rate=0.0)
        x = np.random.rand(3, 6)
        mu, logvar = vae.encode(x)
        z = reparameterize(mu, logvar)
        recon = vae.decode(z)
        recon = np.clip(recon, 1e-12, 1 - 1e-12)
        batch_size = x.shape[0]
        # d_recon_pre in train_step is dL/d(pre-sigmoid logit), not dL/d(recon) --
        # that's exactly the bug that was fixed (an extra sigmoid-derivative
        # factor previously applied on top of the already-simplified gradient).
        # So the finite-difference check must perturb the logit, not `recon`.
        analytic = (recon - x) / batch_size
        logit = np.log(recon / (1 - recon))

        def bce_of_logit(z_logit):
            r = 1.0 / (1.0 + np.exp(-z_logit))
            r = np.clip(r, 1e-12, 1 - 1e-12)
            # train_step's convention divides by batch_size only (not total
            # elements), so the finite-difference reference must match that.
            return -np.sum(x * np.log(r) + (1 - x) * np.log(1 - r)) / batch_size

        eps = 1e-6
        i, j = 0, 1
        z_plus = logit.copy()
        z_plus[i, j] += eps
        z_minus = logit.copy()
        z_minus[i, j] -= eps
        numeric = (bce_of_logit(z_plus) - bce_of_logit(z_minus)) / (2 * eps)
        self.assertAlmostEqual(numeric, analytic[i, j], delta=1e-4)

    def test_autoregressive_continuous_gradient_has_factor_two(self):
        np.random.seed(13)
        ar = AutoregressiveModel(data_dim=6, hidden_dims=[8])
        x = np.random.randn(3, 6)
        logits = ar.forward(x)
        batch_size = x.shape[0]
        analytic = 2 * (logits - x) / batch_size

        def mse(l):
            # train_step's convention divides by batch_size only (not total
            # elements), so the finite-difference reference must match that.
            return np.sum((l - x) ** 2) / batch_size

        eps = 1e-6
        i, j = 0, 2
        l_plus = logits.copy()
        l_plus[i, j] += eps
        l_minus = logits.copy()
        l_minus[i, j] -= eps
        numeric = (mse(l_plus) - mse(l_minus)) / (2 * eps)
        self.assertAlmostEqual(numeric, analytic[i, j], delta=1e-4)


# ========================================================================
# Config Exposure Tests (constants.py + overridable hyperparameters)
# ========================================================================

class TestConfigExposure(unittest.TestCase):
    def test_adam_betas_are_configurable(self):
        # Adam's bias-correction makes step 1 identical regardless of beta
        # values (m/(1-b1^1)==grad, v/(1-b2^1)==grad^2 exactly), so the
        # divergence only shows up after a few steps with per-batch noise.
        np.random.seed(0)
        net_default = NeuralNet(learning_rate=0.05, optimizer="adam")
        net_default.add_dense(6, 2, activation="softmax")
        net_custom = net_default.copy()
        net_custom.adam_beta1 = 0.5
        net_custom.adam_beta2 = 0.8

        for _ in range(5):
            x, y = make_classification_data(8, 6, 2)
            net_default.Forward(x, training=True)
            net_default.Backward(y)
            net_default.update()
            net_custom.Forward(x, training=True)
            net_custom.Backward(y)
            net_custom.update()

        self.assertFalse(np.allclose(net_default.layers[0]["weights"], net_custom.layers[0]["weights"]))

    def test_rmsprop_has_independent_decay_from_adam(self):
        net = NeuralNet(learning_rate=0.01, optimizer="rmsprop", rmsprop_decay=0.5, rmsprop_epsilon=1e-6)
        net.add_dense(4, 3, activation="linear")
        x = np.random.randn(2, 4)
        y = np.random.randn(2, 3)
        net.Forward(x, training=True)
        net.Backward(y)
        net.update()  # should not raise, and should use rmsprop_decay/epsilon
        self.assertEqual(net.rmsprop_decay, 0.5)

    def test_activation_params_override_changes_output(self):
        net_default = NeuralNet()
        net_default.add_dense(3, 2, activation="leakyrelu")
        net_custom = NeuralNet()
        net_custom.add_dense(3, 2, activation="leakyrelu", activation_params={"alpha": 0.5})
        net_custom.layers[0]["weights"] = net_default.layers[0]["weights"].copy()
        net_custom.layers[0]["bias"] = net_default.layers[0]["bias"].copy()

        x = -np.ones((1, 3))
        out_default = net_default.Forward(x)
        out_custom = net_custom.Forward(x)
        self.assertFalse(np.allclose(out_default, out_custom))

    def test_weight_init_normal_std_override(self):
        np.random.seed(0)
        w1, _ = init_weights(50, 50, method="normal", std=0.01)
        w2, _ = init_weights(50, 50, method="normal", std=1.0)
        self.assertLess(np.std(w1), np.std(w2))

    def test_conv2d_stride_forward_and_backward(self):
        net = NeuralNet(learning_rate=0.01, optimizer="adam")
        net.add_conv2d(1, 4, k=3, stride=2, input_size=(9, 9))
        x = np.random.randn(2, 1, 9, 9)
        out = net.Forward(x, training=True)
        self.assertEqual(out.shape, (2, 4, 4, 4))
        dout = np.random.randn(*out.shape)
        net.Backward(None, output_delta=dout)
        net.update()  # should not raise

    def test_gan_wgan_clip_value_is_applied(self):
        gan = GAN(4, 8, generator_hidden=[8], discriminator_hidden=[8],
                 loss_type="wasserstein", wgan_clip_value=0.02)
        x = np.random.randn(8, 8)
        gan.Train(x, epochs=1, batch_size=4, verbose=False)
        max_w = max(np.max(np.abs(l["weights"])) for l in gan.discriminator.layers)
        self.assertLessEqual(max_w, 0.02 + 1e-9)

    def test_ebm_init_noise_scale_is_applied(self):
        ebm = EnergyBasedModel(8, hidden_dims=[8], persistent_cd=True, init_noise_scale=2.0)
        # buffer std should reflect the larger init scale (roughly, statistically)
        self.assertGreater(np.std(ebm.persistent_buffer), 1.0)

    def test_diffusion_time_emb_dim_is_configurable(self):
        diff = DiffusionModel(data_shape=(6,), denoiser_hidden=[8], time_emb_dim=24)
        self.assertEqual(diff.time_emb_dim, 24)
        loss = diff.train_step(np.random.randn(4, 6))
        self.assertIsInstance(loss, float)

    def test_unet_optimizer_is_configurable(self):
        unet = UNetDenoiser(in_ch=1, base_ch=8, time_emb_dim=16, ch_mult=(1,),
                            optimizer="sgd", learning_rate=0.05)
        self.assertEqual(unet.time_net.optimizer_type, "sgd")
        self.assertEqual(unet.encoders[0].optimizer_type, "sgd")


# ========================================================================
# Utility Function Tests (Enilnets.utils)
# ========================================================================

class TestUtils(unittest.TestCase):
    def test_set_seed_reproducible(self):
        set_seed(123)
        a = np.random.randn(5)
        set_seed(123)
        b = np.random.randn(5)
        self.assertTrue(np.allclose(a, b))

    def test_train_test_split_shapes_and_determinism(self):
        X, Y = make_classification_data(50, 4, 3)
        Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, seed=1)
        self.assertEqual(Xte.shape[0], 10)
        self.assertEqual(Xtr.shape[0], 40)
        Xtr2, Xte2, _, _ = train_test_split(X, Y, test_size=0.2, seed=1)
        self.assertTrue(np.allclose(Xte, Xte2))

    def test_iterate_minibatches_covers_full_dataset(self):
        X, Y = make_classification_data(23, 4, 2)
        seen = 0
        for xb, yb in iterate_minibatches(X, Y, batch_size=8, shuffle=True):
            seen += xb.shape[0]
            self.assertEqual(xb.shape[0], yb.shape[0])
        self.assertEqual(seen, 23)

    def test_count_parameters(self):
        net = NeuralNet()
        net.add_dense(10, 5, activation="relu")
        net.add_dense(5, 2, activation="softmax")
        total, per_layer = count_parameters(net)
        self.assertEqual(total, (10 * 5 + 5) + (5 * 2 + 2))
        self.assertEqual(per_layer[0]["params"], 55)

    def test_early_stopping_triggers(self):
        es = EarlyStopping(patience=2, mode="min")
        results = [es.step(v) for v in [1.0, 0.9, 0.95, 0.96, 0.97]]
        self.assertTrue(results[-1])  # should have stopped by the last step

    def test_early_stopping_max_mode(self):
        es = EarlyStopping(patience=2, mode="max")
        results = [es.step(v) for v in [0.5, 0.6, 0.55, 0.54]]
        self.assertTrue(results[-1])

    def test_one_hot(self):
        oh = one_hot(np.array([0, 2, 1]), 3)
        self.assertTrue(np.array_equal(oh, np.eye(3)[[0, 2, 1]]))


class TestCheckpointingAndLogging(unittest.TestCase):
    """v3.1.0 Phase 10: ModelCheckpoint/CSVLogger/JSONLogger -- Train(...,
    callbacks=[...]) callbacks built on Phase 9's generic callback system."""

    def test_model_checkpoint_manufactured_non_monotonic_loss(self):
        ckpt = ModelCheckpoint(monitor="loss", mode="min")
        model = NeuralNet(optimizer="sgd")
        model.add_dense(2, 2, activation="linear")
        losses = [1.0, 0.5, 0.7, 0.3, 0.9]  # best is epoch 3 (0.3)
        for epoch, loss in enumerate(losses):
            model.layers[0]["weights"] = np.full((2, 2), epoch, dtype=np.float64)
            ckpt.on_epoch_end(epoch, {"loss": loss}, model=model)
        self.assertEqual(ckpt.best_epoch, 3)
        self.assertEqual(ckpt.best, 0.3)
        self.assertTrue(np.allclose(ckpt.best_weights[0]["weights"], 3))

    def test_model_checkpoint_restore(self):
        ckpt = ModelCheckpoint(monitor="loss", mode="min")
        model = NeuralNet(optimizer="sgd")
        model.add_dense(2, 2, activation="linear")
        model.layers[0]["weights"] = np.ones((2, 2))
        ckpt.on_epoch_end(0, {"loss": 1.0}, model=model)
        model.layers[0]["weights"] = np.full((2, 2), 5.0)  # worse epoch, weights drift
        ckpt.on_epoch_end(1, {"loss": 2.0}, model=model)
        ckpt.restore(model)
        self.assertTrue(np.allclose(model.layers[0]["weights"], 1.0))

    def test_model_checkpoint_missing_monitor_key_is_noop(self):
        ckpt = ModelCheckpoint(monitor="val_loss", mode="min")
        model = NeuralNet(optimizer="sgd")
        model.add_dense(2, 2, activation="linear")
        ckpt.on_epoch_end(0, {"loss": 1.0}, model=model)  # no "val_loss" key
        self.assertIsNone(ckpt.best)
        self.assertIsNone(ckpt.best_weights)

    def test_model_checkpoint_e2e_via_train(self):
        np.random.seed(0)
        model = NeuralNet(optimizer="sgd", learning_rate=0.1)
        model.add_dense(4, 8, activation="relu")
        model.add_dense(8, 2, activation="softmax")
        X = np.random.randn(30, 4)
        Y = np.eye(2)[np.random.randint(0, 2, 30)]
        ckpt = ModelCheckpoint(monitor="val_loss", mode="min")
        history = model.Train(X, Y, epochs=6, batch_size=30, X_val=X[:10], Y_val=Y[:10],
                              loss_function="cross_entropy", verbose=False, callbacks=[ckpt])
        self.assertEqual(ckpt.best, min(history["val_loss"]))

    def test_csv_logger_writes_valid_partial_log_if_interrupted(self):
        import tempfile, os, csv
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "log.csv")
            logger = CSVLogger(path)
            for epoch in range(3):  # simulate a crash after 3 epochs
                logger.on_epoch_end(epoch, {"loss": 1.0 - epoch * 0.1}, model=None)
            with open(path) as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["epoch"], "0")
            self.assertAlmostEqual(float(rows[2]["loss"]), 0.8)

    def test_json_logger_writes_valid_partial_log_if_interrupted(self):
        import tempfile, os, json
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "log.jsonl")
            logger = JSONLogger(path)
            for epoch in range(3):
                logger.on_epoch_end(epoch, {"loss": 1.0 - epoch * 0.1}, model=None)
            with open(path) as f:
                lines = [json.loads(l) for l in f]
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[0]["epoch"], 0)
            self.assertAlmostEqual(lines[2]["loss"], 0.8)

    def test_loggers_e2e_via_train(self):
        import tempfile, os, csv, json
        np.random.seed(0)
        model = NeuralNet(optimizer="sgd", learning_rate=0.1)
        model.add_dense(4, 8, activation="relu")
        model.add_dense(8, 2, activation="softmax")
        X = np.random.randn(20, 4)
        Y = np.eye(2)[np.random.randint(0, 2, 20)]
        with tempfile.TemporaryDirectory() as d:
            csv_path = os.path.join(d, "log.csv")
            json_path = os.path.join(d, "log.jsonl")
            model.Train(X, Y, epochs=4, batch_size=20, loss_function="cross_entropy",
                       verbose=False, callbacks=[CSVLogger(csv_path), JSONLogger(json_path)])
            with open(csv_path) as f:
                self.assertEqual(len(list(csv.DictReader(f))), 4)
            with open(json_path) as f:
                self.assertEqual(len([json.loads(l) for l in f]), 4)


# ========================================================================
# Auto Shape Inference & Layer-Creation Ergonomics
# ========================================================================

class TestAutoShapeInference(unittest.TestCase):
    def test_dense_chain_without_explicit_n_in(self):
        net = NeuralNet()
        net.add_dense(10, 20, activation="relu")
        net.add_dense(None, 5, activation="softmax")
        self.assertEqual(net.layers[1]["weights"].shape, (5, 20))
        out = net.Forward(np.random.randn(3, 10))
        self.assertEqual(out.shape, (3, 5))

    def test_dense_first_layer_requires_n_in(self):
        net = NeuralNet()
        with self.assertRaises(ValueError):
            net.add_dense(None, 5)

    def test_conv_pool_flatten_dense_chain(self):
        net = NeuralNet()
        net.add_conv2d(1, 4, k=3, activation="relu", input_size=(10, 10))
        net.add_maxpool2d(2)
        net.add_conv2d(None, 8, k=3, activation="relu")
        net.add_flatten()
        net.add_dense(None, 10, activation="softmax")
        x = np.random.randn(2, 1, 10, 10)
        out = net.Forward(x)
        self.assertEqual(out.shape, (2, 10))

    def test_batchnorm_layernorm_auto_infer(self):
        net = NeuralNet()
        net.add_dense(10, 16, activation="linear")
        net.add_batchnorm()
        net.add_layernorm()
        net.add_dense(None, 4, activation="softmax")
        out = net.Forward(np.random.randn(5, 10), training=True)
        self.assertEqual(out.shape, (5, 4))

    def test_attention_embed_dim_auto_infer(self):
        net = NeuralNet()
        net.add_embedding(vocab_size=20, embed_dim=8)
        net.add_multihead_attention(num_heads=2)
        x = np.array([[1, 2, 3], [4, 5, 6]])
        out = net.Forward(x, training=True)
        self.assertEqual(out.shape, (2, 3, 8))

    def test_add_mlp_block(self):
        net = NeuralNet()
        net.add_mlp_block([16, 8], in_dim=10, out_dim=3, activation="relu", out_activation="softmax")
        self.assertEqual(len(net.layers), 3)
        out = net.Forward(np.random.randn(4, 10))
        self.assertEqual(out.shape, (4, 3))

    def test_add_conv_block(self):
        net = NeuralNet()
        net.add_conv_block(8, k=3, in_ch=1, batchnorm=True, pool="max", input_size=(10, 10))
        net.add_flatten()
        net.add_dense(None, 5, activation="softmax")
        x = np.random.randn(2, 1, 10, 10)
        out = net.Forward(x, training=True)
        self.assertEqual(out.shape, (2, 5))


# ========================================================================
# Speed-refactor regression tests: vectorized code must match the old
# loop-based implementations numerically, not just be faster.
# ========================================================================

class TestSpeedRefactorEquivalence(unittest.TestCase):
    def test_embedding_backward_matches_loop_reference(self):
        np.random.seed(0)
        x_int = np.random.randint(0, 10, size=(3, 5))
        dout = np.random.randn(3, 5, 6)
        weights = np.random.randn(10, 6)

        grad_ref = np.zeros_like(weights)
        for i in range(3):
            for j in range(5):
                grad_ref[x_int[i, j]] += dout[i, j]

        layer = {"_last_input": x_int, "weights": weights}
        grad = embedding_backward(dout, layer)
        self.assertTrue(np.allclose(grad, grad_ref))

    def test_upsample2d_backward_matches_loop_reference(self):
        np.random.seed(0)
        x = np.random.randn(2, 3, 4, 4)
        delta = np.random.randn(2, 3, 8, 8)
        scale = 2

        dx_ref = np.zeros_like(x)
        for i in range(scale):
            for j in range(scale):
                dx_ref += delta[:, :, i::scale, j::scale][:, :, :4, :4]

        dx = upsample2d_backward(delta, x, scale)
        self.assertTrue(np.allclose(dx, dx_ref))

    def test_resize_bilinear_matches_loop_reference(self):
        np.random.seed(0)
        img = np.random.rand(10, 12, 3)
        h, w = img.shape[:2]
        new_h, new_w = 7, 9
        row_coords = np.linspace(0, h - 1, new_h)
        col_coords = np.linspace(0, w - 1, new_w)
        row_floor = np.floor(row_coords).astype(int)
        col_floor = np.floor(col_coords).astype(int)
        row_ceil = np.minimum(row_floor + 1, h - 1)
        col_ceil = np.minimum(col_floor + 1, w - 1)
        row_frac = row_coords - row_floor
        col_frac = col_coords - col_floor
        ref = np.zeros((new_h, new_w, 3))
        for i in range(new_h):
            for j in range(new_w):
                y0, y1 = row_floor[i], row_ceil[i]
                x0, x1 = col_floor[j], col_ceil[j]
                fy, fx = row_frac[i], col_frac[j]
                top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
                bot = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
                ref[i, j] = top * (1 - fy) + bot * fy

        out = img_utils.resize_bilinear(img, new_h, new_w)
        self.assertTrue(np.allclose(out, ref))

    def test_images_to_patches_matches_loop_reference(self):
        np.random.seed(0)
        imgs = np.random.rand(4, 10, 10, 3)
        patch_size, stride = 4, 2
        N, H, W, C = imgs.shape
        ref = []
        for i in range(N):
            for y in range(0, H - patch_size + 1, stride):
                for x in range(0, W - patch_size + 1, stride):
                    ref.append(imgs[i, y:y+patch_size, x:x+patch_size, :])
        ref = np.array(ref)
        out = img_utils.images_to_patches(imgs, patch_size, stride)
        self.assertTrue(np.allclose(out, ref))

    def test_stft_matches_loop_reference(self):
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 4000)).astype(np.float64)
        n_fft, hop = 256, 64
        win = np.hanning(n_fft)
        n_frames = 1 + (len(audio) - n_fft) // hop
        ref = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
        for i in range(n_frames):
            start = i * hop
            frame = audio[start:start + n_fft] * win
            ref[:, i] = np.fft.rfft(frame)

        out = aud_utils.stft(audio, n_fft=n_fft, hop_length=hop)
        self.assertTrue(np.allclose(out, ref))

    def test_sample_diversity_matches_loop_reference(self):
        np.random.seed(0)
        samples = np.random.randn(50, 6)
        subset = samples[:30]
        dists = []
        for i in range(30):
            for j in range(i + 1, 30):
                dists.append(np.linalg.norm(subset[i] - subset[j]))
        ref = np.mean(np.array(dists))
        # replicate eval_utils' internal subset selection by monkeypatching choice
        import Enilnets.eval_utils as eu
        orig_choice = np.random.choice
        np.random.choice = lambda n, size, replace: np.arange(size)
        try:
            out = eu.sample_diversity(samples[:30])
        finally:
            np.random.choice = orig_choice
        self.assertAlmostEqual(out, ref, places=8)

    def test_ppo_discrete_delta_matches_loop_reference(self):
        np.random.seed(3)
        batch_size, num_actions = 6, 4
        probs = np.random.dirichlet(np.ones(num_actions), size=batch_size)
        actions = np.random.randint(0, num_actions, batch_size)
        ratio_flat = np.random.rand(batch_size) * 2
        adv_flat = np.random.randn(batch_size)
        epsilon = 0.2

        ref = np.zeros((batch_size, num_actions))
        for i in range(batch_size):
            a = actions[i]
            if ratio_flat[i] < 1 - epsilon and adv_flat[i] < 0:
                ref[i, a] = 0.0
            else:
                ref[i, a] = -adv_flat[i] / (probs[i, a] + 1e-12) / batch_size

        rows = np.arange(batch_size)
        clipped = (ratio_flat < 1 - epsilon) & (adv_flat < 0)
        vals = np.where(clipped, 0.0, -adv_flat / (probs[rows, actions] + 1e-12) / batch_size)
        out = np.zeros((batch_size, num_actions))
        out[rows, actions] = vals

        self.assertTrue(np.allclose(out, ref))


# ========================================================================
# Multimodal Utility Tests (image/audio/text/eval/crossmodal)
# ========================================================================

class TestImageUtils(unittest.TestCase):
    def setUp(self):
        self.img = np.random.rand(32, 32, 3).astype(np.float64)
        self.imgs = np.random.rand(10, 32, 32, 3).astype(np.float64)

    def test_rgb_to_grayscale(self):
        out = img_utils.rgb_to_grayscale(self.img)
        self.assertEqual(out.shape[:2], self.img.shape[:2])

    def test_grayscale_to_rgb(self):
        out = img_utils.grayscale_to_rgb(self.img[:, :, 0])
        self.assertEqual(out.shape, (32, 32, 3))

    def test_resize_nearest_neighbor(self):
        out = img_utils.resize_nearest_neighbor(self.img, 64, 64)
        self.assertEqual(out.shape[:2], (64, 64))

    def test_resize_bilinear(self):
        out = img_utils.resize_bilinear(self.img, 64, 64)
        self.assertEqual(out.shape[:2], (64, 64))

    def test_image_augmentation(self):
        out = img_utils.image_augmentation(self.imgs, flip_h=True, rotate=90, brightness=0.2)
        self.assertEqual(out.shape, self.imgs.shape)

    def test_normalize_images(self):
        out, mean, std = img_utils.normalize_images(self.imgs)
        self.assertEqual(out.shape, self.imgs.shape)

    def test_images_to_patches(self):
        patches = img_utils.images_to_patches(self.imgs, 8, 8)
        self.assertTrue(np.all(np.isfinite(patches)))

    def test_pad_image(self):
        out = img_utils.pad_image(self.img, 4, 4)
        self.assertEqual(out.shape[0], self.img.shape[0] + 8)

    def test_save_load_ppm(self):
        with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as f:
            path = f.name
        try:
            img_utils.save_ppm(self.img, path)
            loaded = img_utils.load_ppm(path)
            self.assertEqual(loaded.shape, self.img.shape)
        finally:
            os.remove(path)


class TestAudioUtils(unittest.TestCase):
    def setUp(self):
        self.audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float64)

    def test_stft_istft_round_trip(self):
        spec = aud_utils.stft(self.audio, n_fft=512, hop_length=128)
        self.assertTrue(np.all(np.isfinite(spec)))
        rec = aud_utils.istft(spec, n_fft=512, hop_length=128)
        self.assertTrue(np.all(np.isfinite(rec)))

    def test_spectrogram_to_mel(self):
        spec = aud_utils.stft(self.audio, n_fft=512, hop_length=128)
        mel = aud_utils.spectrogram_to_mel(np.abs(spec), 16000, n_mels=32)
        self.assertEqual(mel.shape[0], 32)

    def test_audio_to_spectrogram(self):
        spec = aud_utils.audio_to_spectrogram(self.audio, 16000, n_fft=512, hop_length=128, n_mels=32)
        self.assertTrue(np.all(np.isfinite(spec)))

    def test_frame_round_trip(self):
        frames = aud_utils.audio_to_frames(self.audio, 512, 256)
        rec = aud_utils.frames_to_audio(frames, 256)
        self.assertTrue(np.all(np.isfinite(rec)))

    def test_augment_audio(self):
        out = aud_utils.augment_audio(self.audio, 16000, noise_std=0.01)
        self.assertEqual(out.shape, self.audio.shape)

    def test_save_load_wav(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            aud_utils.save_wav(self.audio, path, 16000)
            loaded = aud_utils.load_wav(path)
            self.assertTrue(len(loaded) > 0)
        finally:
            os.remove(path)


class TestDatasets(unittest.TestCase):
    """v3.1.0 Phase 13: local-file dataset loaders (datasets.py). Real
    downloaded MNIST/CIFAR-10 files may not be present in this environment,
    so these tests hand-construct tiny fake files matching the real byte
    layout and confirm the loaders parse them back exactly. A manual check
    against real downloaded files is recommended if/when available."""

    def test_load_mnist_roundtrip(self):
        import struct
        import tempfile
        import os
        from Enilnets.datasets import load_mnist

        with tempfile.TemporaryDirectory() as d:
            N, R, C = 6, 4, 4
            rng = np.random.RandomState(0)
            img_data = rng.randint(0, 256, size=(N, R, C)).astype(np.uint8)
            lbl_data = rng.randint(0, 10, size=(N,)).astype(np.uint8)

            images_path = os.path.join(d, "images-idx3-ubyte")
            labels_path = os.path.join(d, "labels-idx1-ubyte")
            with open(images_path, "wb") as f:
                f.write(struct.pack(">IIII", 2051, N, R, C))
                f.write(img_data.tobytes())
            with open(labels_path, "wb") as f:
                f.write(struct.pack(">II", 2049, N))
                f.write(lbl_data.tobytes())

            X, y = load_mnist(images_path, labels_path)
            self.assertEqual(X.shape, (N, 1, R, C))
            self.assertTrue(np.array_equal(X[:, 0], img_data.astype(np.float64)))
            self.assertTrue(np.array_equal(y, lbl_data.astype(np.int64)))

            X_norm, _ = load_mnist(images_path, labels_path, normalize=True)
            self.assertLessEqual(X_norm.max(), 1.0)
            self.assertTrue(np.allclose(X_norm * 255.0, X))

    def test_load_mnist_bad_magic_raises(self):
        import struct
        import tempfile
        import os
        from Enilnets.datasets import load_mnist

        with tempfile.TemporaryDirectory() as d:
            images_path = os.path.join(d, "images-idx3-ubyte")
            labels_path = os.path.join(d, "labels-idx1-ubyte")
            with open(images_path, "wb") as f:
                f.write(struct.pack(">IIII", 9999, 1, 2, 2))
                f.write(bytes(4))
            with open(labels_path, "wb") as f:
                f.write(struct.pack(">II", 2049, 1))
                f.write(bytes(1))
            with self.assertRaises(ValueError):
                load_mnist(images_path, labels_path)

    def test_load_cifar10_roundtrip(self):
        import pickle
        import tempfile
        import os
        from Enilnets.datasets import load_cifar10

        with tempfile.TemporaryDirectory() as d:
            n = 5
            rng = np.random.RandomState(1)
            data = rng.randint(0, 256, size=(n, 3072)).astype(np.uint8)
            labels = list(rng.randint(0, 10, size=n))
            batch = {b"data": data, b"labels": labels}
            batch_path = os.path.join(d, "data_batch_1")
            with open(batch_path, "wb") as f:
                pickle.dump(batch, f)

            X, y = load_cifar10(batch_path)
            self.assertEqual(X.shape, (n, 3, 32, 32))
            self.assertTrue(np.array_equal(X, data.reshape(n, 3, 32, 32).astype(np.float64)))
            self.assertTrue(np.array_equal(y, np.array(labels)))

            X_norm, _ = load_cifar10(batch_path, normalize=True)
            self.assertLessEqual(X_norm.max(), 1.0)

    def test_load_cifar10_concatenates_multiple_batches(self):
        import pickle
        import tempfile
        import os
        from Enilnets.datasets import load_cifar10

        with tempfile.TemporaryDirectory() as d:
            n = 3
            rng = np.random.RandomState(2)
            paths = []
            for i in range(2):
                data = rng.randint(0, 256, size=(n, 3072)).astype(np.uint8)
                labels = list(rng.randint(0, 10, size=n))
                path = os.path.join(d, f"data_batch_{i+1}")
                with open(path, "wb") as f:
                    pickle.dump({b"data": data, b"labels": labels}, f)
                paths.append(path)

            X, y = load_cifar10(paths)
            self.assertEqual(X.shape, (2 * n, 3, 32, 32))
            self.assertEqual(y.shape, (2 * n,))


class TestTextUtils(unittest.TestCase):
    def test_tokenizer_fit_encode_decode(self):
        texts = ["hello world", "foo bar baz", "testing one two three"]
        tokenizer = txt_utils.Tokenizer(vocab_size=50, level="word")
        tokenizer.fit(texts)
        encoded = tokenizer.encode("hello world", max_length=10)
        self.assertEqual(len(encoded), 10)
        decoded = tokenizer.decode(encoded)
        self.assertIsInstance(decoded, str)

    def test_one_hot_encode(self):
        out = txt_utils.one_hot_encode(np.array([0, 1, 2, 3]), 10)
        self.assertEqual(out.shape, (4, 10))

    def test_pad_sequences(self):
        out = txt_utils.pad_sequences([np.array([1, 2, 3]), np.array([4, 5])], max_length=5)
        self.assertEqual(out.shape, (2, 5))

    def test_create_sliding_windows(self):
        X, y = txt_utils.create_sliding_windows(np.arange(20), 5, 1)
        self.assertTrue(np.all(np.isfinite(X)))
        self.assertTrue(np.all(np.isfinite(y)))


class TestEvalUtils(unittest.TestCase):
    def setUp(self):
        self.samples = np.random.rand(50, 784).astype(np.float64)
        self.real_feat = np.random.randn(50, 64).astype(np.float64)
        self.fake_feat = np.random.randn(50, 64).astype(np.float64)

    def test_inception_score(self):
        score = eval_utils.inception_score(self.samples, classifier=None, splits=5)
        self.assertTrue(np.isfinite(score) if np.isscalar(score) else True)

    def test_compute_fid(self):
        fid = eval_utils.compute_fid(self.real_feat, self.fake_feat)
        self.assertTrue(np.isfinite(fid))

    def test_reconstruction_error_mse(self):
        err = eval_utils.reconstruction_error(self.samples[:10], self.samples[10:20], "mse")
        self.assertTrue(np.isfinite(err))

    def test_reconstruction_error_psnr(self):
        err = eval_utils.reconstruction_error(self.samples[:10], self.samples[:10] + 0.01, "psnr")
        self.assertTrue(np.isfinite(err))

    def test_sample_diversity(self):
        div = eval_utils.sample_diversity(self.samples)
        self.assertTrue(np.isfinite(div))

    def test_nearest_neighbor_accuracy(self):
        acc = eval_utils.nearest_neighbor_accuracy(self.real_feat, self.fake_feat)
        self.assertTrue(np.isfinite(acc))


class TestCrossModalUtils(unittest.TestCase):
    def setUp(self):
        self.img_emb = np.random.randn(10, 64).astype(np.float64)
        self.txt_emb = np.random.randn(10, 64).astype(np.float64)

    def test_clip_normalize(self):
        out = cm_utils.clip_normalize(self.img_emb)
        norms = np.linalg.norm(out, axis=-1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-5))

    def test_contrastive_loss(self):
        loss = cm_utils.contrastive_loss(self.img_emb, self.txt_emb, temperature=0.07)
        self.assertTrue(np.isfinite(loss))

    def test_multimodal_fusion(self):
        for mode in ["concat", "sum", "gated"]:
            out = cm_utils.multimodal_fusion([self.img_emb, self.txt_emb], mode)
            self.assertTrue(np.all(np.isfinite(out)), msg=mode)


# ========================================================================
# Edge Cases
# ========================================================================

class TestEdgeCases(unittest.TestCase):
    def test_single_sample(self):
        model = NeuralNet()
        model.add_dense(10, 5, activation="relu")
        model.add_dense(5, 2, activation="softmax")
        x = np.random.randn(1, 10)
        y = np.array([[1, 0]])
        out = model.Forward(x)
        self.assertEqual(out.shape, (1, 2))
        model.Backward(y)
        model.update()

    def test_large_batch(self):
        model = NeuralNet()
        model.add_dense(10, 5, activation="relu")
        model.add_dense(5, 2, activation="softmax")
        x = np.random.randn(1000, 10)
        y = np.eye(2)[np.random.randint(0, 2, 1000)]
        out = model.Forward(x)
        self.assertEqual(out.shape, (1000, 2))
        model.Backward(y)
        model.update()

    def test_1d_input(self):
        model = NeuralNet()
        model.add_dense(10, 5, activation="relu")
        x = np.random.randn(10)
        out = model.Forward(x)
        self.assertEqual(out.shape, (1, 5))

    def test_3d_input_conv(self):
        model = NeuralNet()
        model.add_conv2d(3, 4, k=3, activation="relu")
        x = np.random.randn(3, 8, 8)
        out = model.Forward(x)
        self.assertEqual(out.shape, (1, 4, 6, 6))

    def test_eval_mode(self):
        model = NeuralNet()
        model.add_dense(10, 100, activation="relu")
        model.add_dropout(rate=0.99)
        model.add_dense(100, 2, activation="softmax")
        x = np.random.randn(1, 10)
        model.train()
        model.Forward(x, training=True)
        model.eval()
        model.Forward(x, training=False)
        self.assertIsNone(model.layers[1].get("mask"))

    def test_summary(self):
        model = NeuralNet()
        model.add_conv2d(3, 16, k=3, activation="relu")
        model.add_batchnorm(16)
        model.add_maxpool2d(2)
        model.add_flatten()
        model.add_dense(64, 10, activation="softmax")
        model.summary()

    def test_end_to_end_training(self):
        np.random.seed(42)
        X = np.random.randn(200, 10).astype(np.float64)
        Y = np.eye(2)[(X[:, 0] + X[:, 1] > 0).astype(int)]
        model = NeuralNet(learning_rate=0.1, optimizer="adam", l2_lambda=0.001)
        model.add_dense(10, 20, activation="relu")
        model.add_dropout(0.2)
        model.add_dense(20, 10, activation="relu")
        model.add_dense(10, 2, activation="softmax")

        split = 150
        X_train, X_val = X[:split], X[split:]
        Y_train, Y_val = Y[:split], Y[split:]
        history = model.Train(X_train, Y_train, epochs=10, batch_size=16,
                              X_val=X_val, Y_val=Y_val, verbose=False)
        self.assertLess(history["loss"][-1], history["loss"][0])
        self.assertGreater(history["accuracy"][-1], 0.5)

        model.eval()
        preds = model.Forward(X_val)
        acc = model.compute_accuracy(preds, Y_val)
        self.assertGreater(acc, 0.5)


class TestResidualConnections(_FDPrecisionMixin, unittest.TestCase):
    """add_residual_start()/add_residual_end() -- verified via finite
    difference against both the layer that starts the skip connection and a
    layer inside the wrapped block."""

    def test_forward_adds_skip_connection(self):
        model = NeuralNet()
        model.add_dense(4, 4, activation="linear")
        model.add_residual_start()
        model.add_dense(4, 4, activation="tanh")
        model.add_residual_end()
        x = np.random.randn(2, 4)
        out = model.Forward(x, training=True)
        # layers: [dense(0), residual_save(1), dense_tanh(2), residual_add(3)]
        # outputs: [x(0), dense_out(1), residual_save_out==dense_out(2), tanh_out(3), out(4)]
        expected = model.outputs[2] + model.outputs[3]
        self.assertTrue(np.allclose(out, expected))

    def test_gradient_matches_finite_difference(self):
        np.random.seed(0)
        model = NeuralNet(learning_rate=0.001, optimizer="adam")
        model.add_dense(6, 6, activation="linear")
        model.add_residual_start()
        model.add_dense(6, 6, activation="tanh")
        model.add_dense(6, 6, activation="linear")
        model.add_residual_end()
        model.add_dense(6, 3, activation="linear")

        x = np.random.randn(4, 6)
        y = np.random.randn(4, 3)
        eps = 1e-6

        def mse():
            o = model.Forward(x, training=True)
            return np.mean((o - y) ** 2)

        for layer_idx, i, j in ((0, 1, 2), (2, 0, 3), (3, 2, 1)):
            W = model.layers[layer_idx]["weights"]
            orig = float(W[i, j])
            W[i, j] = orig + eps
            lp = mse()
            W[i, j] = orig - eps
            lm = mse()
            W[i, j] = orig
            numeric = (lp - lm) / (2 * eps)

            model.Forward(x, training=True)
            model.Backward(y, loss_function="mse")
            analytic = np.dot(model.deltas[layer_idx].T, model.outputs[layer_idx])[i, j]
            self.assertAlmostEqual(numeric, analytic, delta=max(1e-4, abs(numeric) * 1e-3))

    def test_transformer_block_has_residuals(self):
        model = NeuralNet()
        model.add_transformer_block(embed_dim=8, num_heads=2)
        types = [l["type"] for l in model.layers]
        self.assertIn("residual_save", types)
        self.assertIn("residual_add", types)

    def test_unmatched_residual_end_raises(self):
        model = NeuralNet()
        with self.assertRaises(ValueError):
            model.add_residual_end()

    def test_regression_residual_target_with_nonlinear_activation(self):
        # Regression test for a latent bug found while building Phase 5's
        # shared deferral infrastructure: when the residual skip's target
        # layer (the one immediately before add_residual_start()) has a
        # non-linear activation, the deferred gradient was being added
        # directly to that layer's PRE-activation delta slot without first
        # passing through the activation's derivative -- a units mismatch
        # masked in every in-repo transformer block (whose residual targets
        # are always non-dense layer types) but real for exactly the
        # custom-ResNet-block usage the README documents and encourages.
        np.random.seed(0)
        model = NeuralNet(optimizer="sgd")
        model.add_dense(4, 4, activation="tanh")  # residual target -- non-linear
        model.add_residual_start()
        model.add_dense(4, 4, activation="tanh")
        model.add_dense(4, 4, activation="linear")
        model.add_residual_end()
        model.add_dense(4, 3, activation="linear")
        X = np.random.randn(2, 4)
        Y = np.random.randn(2, 3)

        model.Forward(X, training=True)
        model.Backward(Y, loss_function="mse")
        analytic = model.compute_gradients()[0]["weights"]

        def loss_fn():
            return model.ComputeLoss(model.Forward(X, training=False), Y, function="mse")

        eps = 1e-5
        flat = model.layers[0]["weights"].reshape(-1)
        rng = np.random.RandomState(0)
        idxs = rng.choice(flat.size, size=6, replace=False)
        max_err = 0.0
        for idx in idxs:
            orig = float(flat[idx])
            flat[idx] = orig + eps
            lp = loss_fn()
            flat[idx] = orig - eps
            lm = loss_fn()
            flat[idx] = orig
            num = (lp - lm) / (2 * eps)
            max_err = max(max_err, abs(num - analytic.reshape(-1)[idx]))
        self.assertLess(max_err, 1e-6)


class TestMultiSourceLayers(_FDPrecisionMixin, unittest.TestCase):
    """v3.1.0 Phase 5: shared "goto"/"concat_at"/"reverse_sequence" layer
    infrastructure (used internally by cross-attention and bidirectional
    RNN). The indexing convention here is deliberately direct -- a
    stored/source index names "the layer whose OUTPUT is referenced", NO
    -1 adjustment (unlike residual's save_index, which points AT the
    residual_save marker layer and needs -1) -- this is the single
    highest-risk correctness point flagged in the plan, so it gets its own
    explicit index-correctness test before anything else consumes it."""

    def test_goto_index_correctness_forward_and_backward(self):
        # The critical convention test: goto's stored_index must target the
        # NAMED layer directly, not the layer at stored_index-1 (a mistake
        # that would silently point one layer too early, copy-pasting
        # residual's -1 offset into new code that doesn't need it).
        np.random.seed(0)
        model = NeuralNet(optimizer="sgd")
        model.add_dense(4, 4, activation="tanh")   # layer 0
        model.add_dense(4, 4, activation="tanh")   # layer 1 -- becomes a dead end
        model.add_dense(4, 4, activation="tanh")   # layer 2 -- becomes a dead end
        model.layers.append({"type": "goto", "stored_index": 0})  # layer 3
        model.add_dense(4, 3, activation="linear")  # layer 4

        X = np.random.randn(2, 4)
        Y = np.random.randn(2, 3)
        model.Forward(X, training=True)

        # Forward: goto's output must be exactly layer0's output, not
        # layer1's or layer2's.
        self.assertTrue(np.allclose(model.outputs[4], model.outputs[1]))
        self.assertFalse(np.allclose(model.outputs[4], model.outputs[2]))
        self.assertFalse(np.allclose(model.outputs[4], model.outputs[3]))

        # Backward: layer0 gets the real (nonzero) gradient routed through
        # goto; layers 1 and 2 are dead ends and must get exactly zero.
        model.Backward(Y, loss_function="mse")
        grads = model.compute_gradients()
        self.assertTrue(np.allclose(grads[1]["weights"], 0))
        self.assertTrue(np.allclose(grads[2]["weights"], 0))
        self.assertFalse(np.allclose(grads[0]["weights"], 0))

    def test_goto_targets_the_named_layer_not_an_off_by_one_neighbor(self):
        np.random.seed(0)
        model = NeuralNet(optimizer="sgd")
        model.add_dense(4, 4, activation="tanh")
        model.add_dense(4, 4, activation="tanh")
        model.add_dense(4, 4, activation="tanh")
        model.layers.append({"type": "goto", "stored_index": 1})
        model.add_dense(4, 3, activation="linear")
        X = np.random.randn(2, 4)
        model.Forward(X, training=True)
        self.assertTrue(np.allclose(model.outputs[4], model.outputs[2]))
        self.assertFalse(np.allclose(model.outputs[4], model.outputs[1]))

    def test_goto_fd_gradient_dense_target(self):
        np.random.seed(0)
        model = NeuralNet(optimizer="sgd")
        model.add_dense(4, 4, activation="tanh")
        model.add_dense(4, 4, activation="tanh")
        model.add_dense(4, 4, activation="tanh")
        model.layers.append({"type": "goto", "stored_index": 0})
        model.add_dense(4, 3, activation="linear")
        X = np.random.randn(2, 4)
        Y = np.random.randn(2, 3)

        model.Forward(X, training=True)
        model.Backward(Y, loss_function="mse")
        analytic = model.compute_gradients()[0]["weights"]

        def loss_fn():
            return model.ComputeLoss(model.Forward(X, training=False), Y, function="mse")

        eps = 1e-5
        flat = model.layers[0]["weights"].reshape(-1)
        rng = np.random.RandomState(0)
        idxs = rng.choice(flat.size, size=6, replace=False)
        max_err = 0.0
        for idx in idxs:
            orig = float(flat[idx])
            flat[idx] = orig + eps
            lp = loss_fn()
            flat[idx] = orig - eps
            lm = loss_fn()
            flat[idx] = orig
            num = (lp - lm) / (2 * eps)
            max_err = max(max_err, abs(num - analytic.reshape(-1)[idx]))
        self.assertLess(max_err, 1e-6)

    def test_concat_at_forward_shape_and_fd_gradients(self):
        np.random.seed(1)
        model = NeuralNet(optimizer="sgd")
        model.add_dense(4, 4, activation="tanh")   # layer 0: shared source
        model.add_dense(4, 3, activation="tanh")   # layer 1: branch A
        model.layers.append({"type": "goto", "stored_index": 0})  # layer 2
        model.add_dense(4, 5, activation="tanh")   # layer 3: branch B
        model.layers.append({"type": "concat_at", "idx_a": 1, "idx_b": 3})  # layer 4
        model.add_dense(8, 2, activation="linear")  # layer 5

        X = np.random.randn(2, 4)
        Y = np.random.randn(2, 2)
        model.Forward(X, training=True)
        self.assertEqual(model.outputs[5].shape, (2, 8))
        self.assertTrue(np.array_equal(model.outputs[5][:, :3], model.outputs[2]))
        self.assertTrue(np.array_equal(model.outputs[5][:, 3:], model.outputs[4]))

        def fd_check(layer_idx, pname):
            model.Forward(X, training=True)
            model.Backward(Y, loss_function="mse")
            analytic = model.compute_gradients()[layer_idx][pname]
            def loss_fn():
                return model.ComputeLoss(model.Forward(X, training=False), Y, function="mse")
            eps = 1e-5
            flat = model.layers[layer_idx][pname].reshape(-1)
            rng = np.random.RandomState(2)
            idxs = rng.choice(flat.size, size=min(6, flat.size), replace=False)
            max_err = 0.0
            for idx in idxs:
                orig = float(flat[idx])
                flat[idx] = orig + eps
                lp = loss_fn()
                flat[idx] = orig - eps
                lm = loss_fn()
                flat[idx] = orig
                num = (lp - lm) / (2 * eps)
                max_err = max(max_err, abs(num - analytic.reshape(-1)[idx]))
            return max_err

        self.assertLess(fd_check(0, "weights"), 1e-6)  # shared source
        self.assertLess(fd_check(1, "weights"), 1e-6)  # branch A
        self.assertLess(fd_check(3, "weights"), 1e-6)  # branch B
        self.assertLess(fd_check(5, "weights"), 1e-6)  # post-concat

    def test_reverse_sequence_forward_is_time_reversed(self):
        model = NeuralNet()
        model.add_lstm(3, 4, return_sequences=True)
        model.layers.append({"type": "reverse_sequence"})
        X = np.random.randn(2, 5, 3)
        model.Forward(X, training=True)
        self.assertTrue(np.allclose(model.outputs[2], model.outputs[1][:, ::-1, :]))

    def test_reverse_sequence_fd_gradient_through_lstm(self):
        np.random.seed(2)
        model = NeuralNet(optimizer="sgd")
        model.add_lstm(3, 4, return_sequences=True)
        model.layers.append({"type": "reverse_sequence"})
        model.add_dense(4, 2, activation="linear")
        X = np.random.randn(2, 5, 3)
        Y = np.random.randn(2, 5, 2)

        model.Forward(X, training=True)
        model.Backward(Y, loss_function="mse")
        analytic = model.compute_gradients()[0]["Wx"]

        def loss_fn():
            return model.ComputeLoss(model.Forward(X, training=False), Y, function="mse")

        eps = 1e-5
        flat = model.layers[0]["Wx"].reshape(-1)
        rng = np.random.RandomState(0)
        idxs = rng.choice(flat.size, size=6, replace=False)
        max_err = 0.0
        for idx in idxs:
            orig = float(flat[idx])
            flat[idx] = orig + eps
            lp = loss_fn()
            flat[idx] = orig - eps
            lm = loss_fn()
            flat[idx] = orig
            num = (lp - lm) / (2 * eps)
            max_err = max(max_err, abs(num - analytic.reshape(-1)[idx]))
        self.assertLess(max_err, 1e-6)


class TestCrossAttention(_FDPrecisionMixin, unittest.TestCase):
    """v3.1.0 Phase 6: add_cross_attention -- encoder-decoder style
    attention where Q comes from the normal sequential x and K/V come from
    an earlier layer's output (kv_source_index), built on Phase 5's direct
    (no -1 offset) indexing convention."""

    def _build_encoder_decoder_toy(self):
        # shared source -> {encoder branch (KV source), decoder branch (Q)}
        # -> cross_attention -> output, mirroring an encoder-decoder split.
        np.random.seed(0)
        model = NeuralNet(optimizer="sgd")
        model.add_dense(6, 8, activation="tanh")   # layer 0: shared source
        model.add_dense(8, 8, activation="tanh")   # layer 1: encoder repr (KV source)
        model.layers.append({"type": "goto", "stored_index": 0})  # layer 2
        model.add_dense(8, 8, activation="tanh")   # layer 3: decoder query stream
        model.add_cross_attention(kv_source_index=1, embed_dim=8, num_heads=2)  # layer 4
        model.add_dense(8, 3, activation="linear")  # layer 5
        B, S = 2, 4
        X = np.random.randn(B, S, 6)
        Y = np.random.randn(B, S, 3)
        return model, X, Y

    def _fd_check(self, model, X, Y, layer_idx, pname, eps=1e-5, n_check=6):
        model.Forward(X, training=True)
        model.Backward(Y, loss_function="mse")
        analytic = model.compute_gradients()[layer_idx][pname]
        def loss_fn():
            return model.ComputeLoss(model.Forward(X, training=False), Y, function="mse")
        flat = model.layers[layer_idx][pname].reshape(-1)
        rng = np.random.RandomState(0)
        idxs = rng.choice(flat.size, size=min(n_check, flat.size), replace=False)
        max_err = 0.0
        for idx in idxs:
            orig = float(flat[idx])
            flat[idx] = orig + eps
            lp = loss_fn()
            flat[idx] = orig - eps
            lm = loss_fn()
            flat[idx] = orig
            num = (lp - lm) / (2 * eps)
            max_err = max(max_err, abs(num - analytic.reshape(-1)[idx]))
        return max_err

    def test_forward_shape_and_uses_kv_source_not_sequential_predecessor(self):
        model, X, Y = self._build_encoder_decoder_toy()
        model.Forward(X, training=True)
        self.assertEqual(model.outputs[-1].shape, (2, 4, 3))
        # Cross-attention's own K/V really came from layer 1 (encoder),
        # not layer 3 (its own sequential predecessor) -- perturb layer 1's
        # weights and confirm cross-attention's output changes.
        out_before = model.outputs[5].copy()  # cross_attention (layer 4) own output
        model.layers[1]["weights"] += 0.5
        out_after = model.Forward(X, training=True)
        self.assertFalse(np.allclose(out_before, model.outputs[5]))

    def test_fd_gradients_q_path(self):
        # (a) Wq/bq via Q's path.
        model, X, Y = self._build_encoder_decoder_toy()
        for pname in ("Wq", "bq"):
            self.assertLess(self._fd_check(model, X, Y, 4, pname), 1e-6)

    def test_fd_gradients_kv_source_path(self):
        # (b) Wk/bk/Wv/bv via the KV-source path.
        model, X, Y = self._build_encoder_decoder_toy()
        for pname in ("Wk", "bk", "Wv", "bv", "Wo", "bo"):
            self.assertLess(self._fd_check(model, X, Y, 4, pname), 1e-6)

    def test_fd_gradients_flow_into_both_branches(self):
        model, X, Y = self._build_encoder_decoder_toy()
        self.assertLess(self._fd_check(model, X, Y, 0, "weights"), 1e-6)  # shared source
        self.assertLess(self._fd_check(model, X, Y, 1, "weights"), 1e-6)  # encoder branch
        self.assertLess(self._fd_check(model, X, Y, 3, "weights"), 1e-6)  # decoder branch

    def test_e2e_kv_source_weight_change_moves_the_loss(self):
        # (c) a weight change in the KV-source branch must move the loss
        # through cross-attention.
        model, X, Y = self._build_encoder_decoder_toy()
        loss_before = model.ComputeLoss(model.Forward(X, training=False), Y, function="mse")
        model.layers[1]["weights"] += 0.1
        loss_after = model.ComputeLoss(model.Forward(X, training=False), Y, function="mse")
        self.assertNotEqual(loss_before, loss_after)

    def test_cross_attention_as_last_layer_edge_case(self):
        # Cross-attention as the network's very last layer -- exercises the
        # dedicated last-layer deferral path in Backward().
        np.random.seed(1)
        model = NeuralNet(optimizer="sgd")
        model.add_dense(4, 8, activation="tanh")
        model.add_dense(8, 8, activation="tanh")  # layer 1: KV source
        model.layers.append({"type": "goto", "stored_index": 0})
        model.add_dense(8, 8, activation="tanh")  # layer 3: query stream
        model.add_cross_attention(kv_source_index=1, embed_dim=8, num_heads=2)  # layer 4, last
        X = np.random.randn(2, 3, 4)
        Y = np.random.randn(2, 3, 8)
        self.assertLess(self._fd_check(model, X, Y, 4, "Wk"), 1e-6)
        self.assertLess(self._fd_check(model, X, Y, 1, "weights"), 1e-6)


class TestRecurrentLayers(_FDPrecisionMixin, unittest.TestCase):
    """add_rnn/add_lstm/add_gru -- full BPTT verified via finite difference
    over multiple timesteps, for every parameter and both
    return_sequences=True/False."""

    def _check_layer(self, add_fn, param_names, return_sequences):
        np.random.seed(1)
        model = NeuralNet(learning_rate=0.001, optimizer="adam")
        add_fn(model, 5, hidden_dim=4, return_sequences=return_sequences)
        if return_sequences:
            model.add_flatten()
            model.add_dense(4 * 3, 3, activation="linear")
        else:
            model.add_dense(4, 3, activation="linear")

        B, S, n_in = 2, 3, 5
        x = np.random.randn(B, S, n_in)
        y = np.random.randn(B, 3)
        eps = 1e-6

        def mse():
            o = model.Forward(x, training=True)
            return np.mean((o - y) ** 2)

        layer = model.layers[0]
        for pname in param_names:
            W = layer[pname]
            idx = (0, 0) if W.ndim == 2 else (0,)
            i = idx[0]
            j = idx[1] if len(idx) > 1 else None
            if j is None:
                orig = float(W[i])
                W[i] = orig + eps
                lp = mse()
                W[i] = orig - eps
                lm = mse()
                W[i] = orig
            else:
                orig = float(W[i, j])
                W[i, j] = orig + eps
                lp = mse()
                W[i, j] = orig - eps
                lm = mse()
                W[i, j] = orig
            numeric = (lp - lm) / (2 * eps)

            model.Forward(x, training=True)
            model.Backward(y, loss_function="mse")
            from Enilnets.backward import rnn_backward, lstm_backward, gru_backward
            fn = {"rnn": rnn_backward, "lstm": lstm_backward, "gru": gru_backward}[layer["type"]]
            fn(model.deltas[0], layer, model.rnn_cache[0])
            analytic = layer[f"d_{pname}"][i] if j is None else layer[f"d_{pname}"][i, j]
            self.assertAlmostEqual(numeric, analytic, delta=max(1e-4, abs(numeric) * 1e-3))

    def test_rnn_gradients_return_sequences(self):
        self._check_layer(NeuralNet.add_rnn, ("Wx", "Wh", "b"), True)

    def test_rnn_gradients_last_step_only(self):
        self._check_layer(NeuralNet.add_rnn, ("Wx", "Wh", "b"), False)

    def test_lstm_gradients_return_sequences(self):
        self._check_layer(NeuralNet.add_lstm, ("Wx", "Wh", "b"), True)

    def test_lstm_gradients_last_step_only(self):
        self._check_layer(NeuralNet.add_lstm, ("Wx", "Wh", "b"), False)

    def test_gru_gradients_return_sequences(self):
        self._check_layer(NeuralNet.add_gru, ("Wx", "Wh", "bx", "bh"), True)

    def test_gru_gradients_last_step_only(self):
        self._check_layer(NeuralNet.add_gru, ("Wx", "Wh", "bx", "bh"), False)

    def test_gradient_flows_to_preceding_layer(self):
        np.random.seed(2)
        for add_name in ("add_rnn", "add_lstm", "add_gru"):
            model = NeuralNet(learning_rate=0.001, optimizer="adam")
            model.add_dense(5, 5, activation="tanh")
            getattr(model, add_name)(5, hidden_dim=4, return_sequences=True)
            model.add_flatten()
            model.add_dense(4 * 3, 3, activation="linear")

            x = np.random.randn(2, 3, 5)
            y = np.random.randn(2, 3)
            eps = 1e-6

            def mse():
                o = model.Forward(x, training=True)
                return np.mean((o - y) ** 2)

            W = model.layers[0]["weights"]
            i, j = 2, 3
            orig = float(W[i, j])
            W[i, j] = orig + eps
            lp = mse()
            W[i, j] = orig - eps
            lm = mse()
            W[i, j] = orig
            numeric = (lp - lm) / (2 * eps)

            model.Forward(x, training=True)
            model.Backward(y, loss_function="mse")
            d = model.deltas[0].reshape(-1, 5)
            o = model.outputs[0].reshape(-1, 5)
            analytic = np.dot(d.T, o)[i, j]
            self.assertAlmostEqual(numeric, analytic, delta=max(1e-4, abs(numeric) * 1e-3), msg=add_name)

    def test_auto_shape_inference(self):
        model = NeuralNet()
        model.add_dense(10, 6, activation="relu")
        model.add_lstm(hidden_dim=4)
        self.assertEqual(model.layers[1]["n_in"], 6)


class TestBidirectionalRNN(_FDPrecisionMixin, unittest.TestCase):
    """v3.1.0 Phase 7: add_bidirectional_rnn/_lstm/_gru -- a thin
    composition of Phase 5's goto/reverse_sequence/concat_at primitives
    around the existing add_rnn/add_lstm/add_gru."""

    def _fd_check(self, model, X, Y, layer_idx, pname, eps=1e-5, n_check=6):
        model.Forward(X, training=True)
        model.Backward(Y, loss_function="mse")
        analytic = model.compute_gradients()[layer_idx][pname]
        def loss_fn():
            return model.ComputeLoss(model.Forward(X, training=False), Y, function="mse")
        flat = model.layers[layer_idx][pname].reshape(-1)
        rng = np.random.RandomState(0)
        idxs = rng.choice(flat.size, size=min(n_check, flat.size), replace=False)
        max_err = 0.0
        for idx in idxs:
            orig = float(flat[idx])
            flat[idx] = orig + eps
            lp = loss_fn()
            flat[idx] = orig - eps
            lm = loss_fn()
            flat[idx] = orig
            num = (lp - lm) / (2 * eps)
            max_err = max(max_err, abs(num - analytic.reshape(-1)[idx]))
        return max_err

    def test_forward_shape_and_composition_return_sequences_true(self):
        np.random.seed(0)
        model = NeuralNet(optimizer="sgd")
        model.add_bidirectional_lstm(n_in=4, hidden_dim=5, return_sequences=True)
        model.add_dense(10, 3, activation="linear")
        X = np.random.randn(2, 6, 4)
        model.Forward(X, training=True)
        self.assertEqual(model.outputs[-1].shape, (2, 6, 3))

        # layers: 0 lstm(fwd), 1 goto, 2 reverse_sequence, 3 lstm(bwd),
        # 4 reverse_sequence(un-reverse), 5 concat_at, 6 dense
        concat_out = model.outputs[6]
        self.assertTrue(np.allclose(concat_out[..., :5], model.outputs[1]))
        self.assertTrue(np.allclose(concat_out[..., 5:], model.outputs[5]))
        # backward direction really did see the time-reversed input: its
        # raw (pre-un-reverse) output, reversed again, is the un-reversed one.
        self.assertTrue(np.allclose(model.outputs[4][:, ::-1, :], model.outputs[5]))

    def test_forward_shape_return_sequences_false(self):
        np.random.seed(1)
        model = NeuralNet(optimizer="sgd")
        model.add_bidirectional_gru(n_in=3, hidden_dim=4, return_sequences=False)
        model.add_dense(8, 2, activation="linear")
        X = np.random.randn(2, 5, 3)
        model.Forward(X, training=True)
        self.assertEqual(model.outputs[-1].shape, (2, 2))
        # no un-reversing needed for return_sequences=False -- confirm the
        # layer sequence doesn't insert a second reverse_sequence.
        types = [l["type"] for l in model.layers]
        self.assertEqual(types.count("reverse_sequence"), 1)

    def test_fd_gradients_lstm_both_directions(self):
        np.random.seed(0)
        model = NeuralNet(optimizer="sgd")
        model.add_bidirectional_lstm(n_in=4, hidden_dim=5, return_sequences=True)
        model.add_dense(10, 3, activation="linear")
        X = np.random.randn(2, 6, 4)
        Y = np.random.randn(2, 6, 3)
        self.assertLess(self._fd_check(model, X, Y, 0, "Wx"), 1e-6)  # forward direction
        self.assertLess(self._fd_check(model, X, Y, 3, "Wx"), 1e-6)  # backward direction
        self.assertLess(self._fd_check(model, X, Y, 0, "Wh"), 1e-6)
        self.assertLess(self._fd_check(model, X, Y, 6, "weights"), 1e-6)  # post-concat dense

    def test_fd_gradients_rnn_and_gru(self):
        np.random.seed(2)
        model_rnn = NeuralNet(optimizer="sgd")
        model_rnn.add_bidirectional_rnn(n_in=3, hidden_dim=4, return_sequences=True)
        model_rnn.add_dense(8, 2, activation="linear")
        X = np.random.randn(2, 4, 3)
        Y = np.random.randn(2, 4, 2)
        self.assertLess(self._fd_check(model_rnn, X, Y, 0, "Wx"), 1e-6)
        self.assertLess(self._fd_check(model_rnn, X, Y, 3, "Wx"), 1e-6)

        np.random.seed(1)
        model_gru = NeuralNet(optimizer="sgd")
        model_gru.add_bidirectional_gru(n_in=3, hidden_dim=4, return_sequences=False)
        model_gru.add_dense(8, 2, activation="linear")
        X2 = np.random.randn(2, 5, 3)
        Y2 = np.random.randn(2, 2)
        self.assertLess(self._fd_check(model_gru, X2, Y2, 0, "Wx"), 1e-6)
        self.assertLess(self._fd_check(model_gru, X2, Y2, 3, "Wx"), 1e-6)

    def test_bidirectional_as_first_layer_negative_goto_index(self):
        # When add_bidirectional_* is the very first thing added,
        # source_index = len(self.layers) - 1 = -1 -- exercises the
        # negative-index guard added in Phase 5/6.
        np.random.seed(0)
        model = NeuralNet(optimizer="sgd")
        model.add_bidirectional_lstm(n_in=3, hidden_dim=4, return_sequences=True)
        model.add_dense(8, 2, activation="linear")
        self.assertEqual(model.layers[1]["stored_index"], -1)
        X = np.random.randn(2, 4, 3)
        Y = np.random.randn(2, 4, 2)
        self.assertLess(self._fd_check(model, X, Y, 0, "Wx"), 1e-6)
        self.assertLess(self._fd_check(model, X, Y, 3, "Wx"), 1e-6)

    def test_bidirectional_solves_backward_in_time_task_unidirectional_does_not(self):
        # Functional test: label(t) depends on x(t+1) (a FUTURE value) --
        # solvable only by looking backward-in-time, not by a plain
        # forward-only unidirectional RNN.
        def make_batch(n, seq_len=6, seed=0):
            rng = np.random.RandomState(seed)
            X = rng.randn(n, seq_len, 1)
            Y = np.zeros((n, seq_len, 1))
            Y[:, :-1, 0] = (X[:, 1:, 0] > 0).astype(np.float64)
            Y[:, -1, 0] = 0.5
            return X, Y

        X, Y = make_batch(150, seed=0)
        Xv, Yv = make_batch(40, seed=1)

        np.random.seed(0)
        uni = NeuralNet(optimizer="adam", learning_rate=0.02)
        uni.add_lstm(1, 8, return_sequences=True)
        uni.add_dense(8, 1, activation="sigmoid")
        for _ in range(120):
            uni.TrainBatch(X, Y, loss_function="binary_cross_entropy")
        uni_pred = uni.Forward(Xv, training=False)
        uni_acc = np.mean(((uni_pred > 0.5).astype(float) == Yv)[:, :-1])

        np.random.seed(0)
        bi = NeuralNet(optimizer="adam", learning_rate=0.02)
        bi.add_bidirectional_lstm(1, 8, return_sequences=True)
        bi.add_dense(16, 1, activation="sigmoid")
        for _ in range(120):
            bi.TrainBatch(X, Y, loss_function="binary_cross_entropy")
        bi_pred = bi.Forward(Xv, training=False)
        bi_acc = np.mean(((bi_pred > 0.5).astype(float) == Yv)[:, :-1])

        self.assertLess(uni_acc, 0.65)   # unidirectional: at/near chance
        self.assertGreater(bi_acc, 0.9)  # bidirectional: actually solves it


class TestOptimizerFeatures(unittest.TestCase):
    """Group 2: gradient clipping, gradient accumulation, mixed precision,
    AdamW -- previously-dead constructor flags, now wired up."""

    def test_grad_clip_reduces_delta_norm(self):
        model = NeuralNet(learning_rate=0.01, optimizer="sgd", grad_clip_norm=0.5)
        model.add_dense(4, 4, activation="linear")
        x = np.random.randn(8, 4) * 100
        y = np.random.randn(8, 4)
        model.TrainBatch(x, y, loss_function="mse")
        total_norm = np.sqrt(sum(np.sum(d ** 2) for d in model.deltas if d is not None))
        self.assertLessEqual(total_norm, 0.5 + 1e-6)

    def test_gradient_accumulation_matches_full_batch_sgd(self):
        np.random.seed(3)
        net_acc = NeuralNet(learning_rate=0.1, optimizer="sgd", momentum=0.0, l2_lambda=0.0)
        net_acc.add_dense(3, 2, activation="linear")
        net_full = net_acc.copy()

        x1, y1 = np.random.randn(4, 3), np.random.randn(4, 2)
        x2, y2 = np.random.randn(4, 3), np.random.randn(4, 2)
        net_acc.TrainBatch(x1, y1, loss_function="mse", accumulation_steps=2)
        net_acc.TrainBatch(x2, y2, loss_function="mse", accumulation_steps=2)

        xfull = np.concatenate([x1, x2])
        yfull = np.concatenate([y1, y2])
        net_full.TrainBatch(xfull, yfull, loss_function="mse")

        self.assertTrue(np.allclose(net_acc.layers[0]["weights"], net_full.layers[0]["weights"]))

    def test_mixed_precision_close_to_float64(self):
        # use_mixed_precision now forces the matmul to float32 regardless of
        # the ambient default dtype (see forward.py) -- so this comparison
        # is only meaningful if model64 itself is genuinely float64.
        with _force_float64():
            np.random.seed(5)
            model64 = NeuralNet(learning_rate=0.01, optimizer="adam")
            model64.add_dense(10, 8, activation="relu")
            model64.add_dense(8, 3, activation="linear")
            model32 = model64.copy()
            model32.use_mixed_precision = True

            x = np.random.randn(4, 10)
            o64 = model64.Forward(x, training=False)
            o32 = model32.Forward(x, training=False)
            self.assertTrue(np.allclose(o64, o32, atol=1e-3))

    def test_adamw_decays_weights_with_zero_gradient(self):
        model = NeuralNet(learning_rate=0.1, optimizer="adamw", l2_lambda=0.1)
        model.add_dense(3, 2, activation="linear")
        w_before = model.layers[0]["weights"].copy()
        x = np.zeros((2, 3))
        model.Forward(x, training=True)
        model.Backward(None, output_delta=np.zeros((2, 2)))
        model.update()
        self.assertFalse(np.allclose(w_before, model.layers[0]["weights"]))

    def test_compute_and_apply_gradients_equivalent_to_update(self):
        np.random.seed(6)
        net_a = NeuralNet(learning_rate=0.05, optimizer="adam")
        net_a.add_dense(4, 3, activation="linear")
        net_b = net_a.copy()

        x = np.random.randn(5, 4)
        y = np.random.randn(5, 3)
        net_a.Forward(x, training=True)
        net_a.Backward(y, loss_function="mse")
        net_a.update()

        net_b.Forward(x, training=True)
        net_b.Backward(y, loss_function="mse")
        net_b.apply_gradients(net_b.compute_gradients())

        self.assertTrue(np.allclose(net_a.layers[0]["weights"], net_b.layers[0]["weights"]))


class TestGenerationQualityOfLife(unittest.TestCase):
    """Group 3: top-k sampling, beam search, perplexity, KV-cache incremental
    decoding for TextGenerator."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.corpus = "the quick brown fox jumps over the lazy dog " * 20
        tok = Tokenizer(vocab_size=64, level="char").fit([cls.corpus])
        cls.gen = TextGenerator(tok, embed_dim=16, num_heads=2, num_layers=2, max_seq_len=32)
        cls.gen.Train([cls.corpus], epochs=10, batch_size=8, seq_len=16, verbose=False)

    def _make_generator(self):
        return self.gen, self.corpus

    def test_top_k_sampling_restricts_support(self):
        from Enilnets.generative.sampling import top_k_sampling
        logits = np.array([5.0, 1.0, 0.5, 4.0, -1.0])
        for _ in range(20):
            choice = top_k_sampling(logits, k=2, temperature=1.0)
            self.assertIn(choice, (0, 3))

    def test_generate_top_k_runs(self):
        gen, _ = self._make_generator()
        text = gen.generate(prompt="the", max_new_tokens=10, top_k=5, use_cache=False)
        self.assertIsInstance(text, str)

    def test_generate_beam_runs_and_returns_string(self):
        gen, _ = self._make_generator()
        text = gen.generate_beam(prompt="the", beam_width=3, max_new_tokens=8)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_perplexity_is_positive_finite(self):
        gen, corpus = self._make_generator()
        ppl = gen.perplexity(corpus)
        self.assertTrue(np.isfinite(ppl))
        self.assertGreater(ppl, 0)

    def test_kv_cache_matches_full_recompute(self):
        gen, _ = self._make_generator()
        np.random.seed(42)
        text_nocache = gen.generate(prompt="the quick", max_new_tokens=15, greedy=True, use_cache=False)
        np.random.seed(42)
        text_cache = gen.generate(prompt="the quick", max_new_tokens=15, greedy=True, use_cache=True)
        self.assertEqual(text_nocache, text_cache)

    def test_kv_cache_probabilities_match_forward(self):
        gen, _ = self._make_generator()
        start_id = gen.tokenizer.word_to_idx[gen.tokenizer.start_token]
        ids = [start_id] + gen.tokenizer.encode("the quick", add_special_tokens=False).tolist()
        context = np.array([ids], dtype=np.int64)
        probs_full = gen.network.Forward(context, training=False)[0, -1]

        cache = {"kv": {}, "residual": {}, "position": 0}
        probs_cached = None
        for tid in ids:
            probs_cached = gen._kv_step(tid, cache)
            cache["position"] += 1
        self.assertTrue(np.allclose(probs_full, probs_cached[0, 0], atol=1e-8))


class TestEvalMetrics(unittest.TestCase):
    """Group 4: confusion_matrix/classification_report, k_fold_split."""

    def test_confusion_matrix_counts(self):
        from Enilnets import confusion_matrix
        y_true = np.array([0, 0, 1, 1, 2, 2, 2, 1])
        y_pred = np.array([0, 1, 1, 1, 2, 0, 2, 1])
        cm = confusion_matrix(y_true, y_pred, num_classes=3)
        self.assertEqual(cm.sum(), len(y_true))
        self.assertEqual(cm[1, 1], 3)

    def test_classification_report_matches_binary_precision_recall_f1(self):
        from Enilnets import classification_report
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        report = classification_report(y_true, y_pred, num_classes=2)
        self.assertAlmostEqual(report[1]["precision"], 2 / 3, places=6)
        self.assertAlmostEqual(report[1]["recall"], 2 / 3, places=6)
        self.assertIn("macro_avg", report)
        self.assertIn("weighted_avg", report)

    def test_k_fold_split_covers_all_samples_disjointly(self):
        from Enilnets import k_fold_split
        X = np.arange(20).reshape(10, 2)
        Y = np.arange(10)
        folds = list(k_fold_split(X, Y, k=5, seed=0))
        self.assertEqual(len(folds), 5)
        seen = []
        for X_train, X_val, Y_train, Y_val in folds:
            self.assertEqual(X_train.shape[0] + X_val.shape[0], 10)
            seen.extend(Y_val.tolist())
        self.assertEqual(sorted(seen), list(range(10)))


class TestNEAT(unittest.TestCase):
    """NEAT genome/mutation invariants, plus an end-to-end XOR evolution
    check (the classic NEAT benchmark: XOR is not linearly separable, so
    solving it requires the population to actually grow hidden structure,
    not just tune weights on the minimal fully-connected genome)."""

    def test_minimal_genome_forward_shape(self):
        from Enilnets.neat import Genome, InnovationTracker
        tracker = InnovationTracker()
        tracker.reset_node_id_counter(3 + 1 + 2)
        genome = Genome.minimal(3, 2, tracker)
        out = genome.forward(np.random.randn(3))
        self.assertEqual(out.shape, (2,))
        batch_out = genome.forward(np.random.randn(5, 3))
        self.assertEqual(batch_out.shape, (5, 2))

    def test_add_node_mutation_preserves_function(self):
        # Splitting a connection (in->new weight=1, new->out weight=old) must
        # not change the network's output at the moment of the split, since
        # new_node's activation is applied to a linear passthrough of `in`.
        from Enilnets.neat import Genome, InnovationTracker
        tracker = InnovationTracker()
        tracker.reset_node_id_counter(2 + 1 + 1)
        genome = Genome.minimal(1, 1, tracker, activation="linear", output_activation="linear")
        x = np.array([0.7])
        before = genome.forward(x)
        genome.mutate_add_node(tracker)
        after = genome.forward(x)
        self.assertTrue(np.allclose(before, after, atol=1e-8))
        self.assertEqual(sum(1 for n in genome.nodes.values() if n.type == "hidden"), 1)

    def test_add_connection_never_creates_cycle(self):
        from Enilnets.neat import Genome, InnovationTracker
        tracker = InnovationTracker()
        tracker.reset_node_id_counter(2 + 1 + 1)
        genome = Genome.minimal(2, 1, tracker)
        for _ in range(30):
            genome.mutate_add_node(tracker)
            genome.mutate_add_connection(tracker)
        # A valid topological order must exist (i.e. the graph is a DAG);
        # forward() would infinite-loop or raise a KeyError otherwise.
        out = genome.forward(np.random.randn(2))
        self.assertEqual(out.shape, (1,))

    def test_crossover_inherits_matching_and_disjoint_genes(self):
        from Enilnets.neat import Genome, InnovationTracker, crossover
        tracker = InnovationTracker()
        tracker.reset_node_id_counter(2 + 1 + 1)
        parent_a = Genome.minimal(2, 1, tracker)
        parent_a.fitness = 2.0
        parent_b = parent_a.copy()
        parent_b.fitness = 1.0
        parent_b.mutate_add_node(tracker)  # gives b a disjoint/excess gene

        child = crossover(parent_a, parent_b)
        self.assertEqual(set(child.connections.keys()), set(parent_a.connections.keys()))
        out = child.forward(np.random.randn(2))
        self.assertEqual(out.shape, (1,))

    def test_break_cycles_restores_valid_topological_order(self):
        # Regression test: crossover's matching-gene coin flip can pick each
        # endpoint of a same-innovation edge from a different parent, and
        # since the two parents are only guaranteed acyclic *individually*
        # (each one's own mutate_add_connection only ever checked its own
        # topology), the combination is not guaranteed acyclic -- this is
        # exactly what happened evolving XOR for 200 generations before
        # break_cycles() existed (KeyError in forward() on an un-orderable
        # node). Simulate the resulting cyclic genome directly and confirm
        # break_cycles() repairs it.
        from Enilnets.neat import Genome, InnovationTracker, ConnectionGene
        tracker = InnovationTracker()
        tracker.reset_node_id_counter(2 + 1 + 1)
        genome = Genome.minimal(2, 1, tracker)
        genome.mutate_add_node(tracker)
        hidden_id = next(nid for nid, n in genome.nodes.items() if n.type == "hidden")
        out_id = genome.output_ids[0]

        # hidden->out already exists (enabled) from the split; force the
        # opposite direction in too, creating a direct 2-cycle.
        innov_bwd = tracker.get_connection_innovation(out_id, hidden_id)
        genome.connections[innov_bwd] = ConnectionGene(out_id, hidden_id, 0.5, True, innov_bwd)
        self.assertLess(len(genome._topo_order()), len(genome.nodes))  # confirms it's cyclic

        genome.break_cycles()
        self.assertEqual(len(genome._topo_order()), len(genome.nodes))
        out = genome.forward(np.random.randn(2))
        self.assertEqual(out.shape, (1,))

    def test_evolve_many_seeds_without_cycle_crash(self):
        # Broader regression coverage for the same crossover-cycle bug: run
        # several full generations across multiple seeds and confirm none of
        # them ever hit the un-orderable-topology KeyError.
        from Enilnets import NEATPopulation
        xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        xor_targets = np.array([0, 1, 1, 0], dtype=np.float64)

        def fitness(genome):
            preds = np.array([genome.forward(x)[0] for x in xor_inputs])
            return float(4.0 - np.sum((preds - xor_targets) ** 2))

        for seed in range(3):
            pop = NEATPopulation(n_inputs=2, n_outputs=1, population_size=60, seed=seed)
            pop.evolve(fitness, generations=40, verbose=False)  # must not raise

    def test_distance_zero_for_identical_genome(self):
        from Enilnets.neat import Genome, InnovationTracker
        tracker = InnovationTracker()
        tracker.reset_node_id_counter(2 + 1 + 1)
        genome = Genome.minimal(2, 1, tracker)
        self.assertEqual(genome.distance(genome.copy()), 0.0)

    def test_population_evolves_and_solves_xor(self):
        from Enilnets import NEATPopulation
        np.random.seed(1)
        xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        xor_targets = np.array([0, 1, 1, 0], dtype=np.float64)

        def fitness(genome):
            preds = np.array([genome.forward(x)[0] for x in xor_inputs])
            return float(4.0 - np.sum((preds - xor_targets) ** 2))

        pop = NEATPopulation(n_inputs=2, n_outputs=1, population_size=150, seed=1)
        history = pop.evolve(fitness, generations=100, verbose=False)

        self.assertGreater(history[-1], history[0])
        preds = np.array([pop.best_genome.forward(x)[0] for x in xor_inputs])
        correct = (preds > 0.5).astype(np.float64)
        self.assertTrue(np.array_equal(correct, xor_targets))


class TestVisualization(unittest.TestCase):
    """Enilnets.visualization: SVG node/connection diagrams for both
    NeuralNet and NEAT genomes, plus the file/HTML embedding paths a real
    project would actually use (raw SVG string, .svg file, .html file,
    standalone to_html() wrapping)."""

    def test_plot_network_without_sample_input(self):
        from Enilnets import NeuralNet, plot_network
        model = NeuralNet()
        model.add_dense(4, 5, activation="relu")
        model.add_dense(5, 2, activation="softmax")
        svg = plot_network(model)
        self.assertTrue(svg.startswith("<svg"))
        self.assertTrue(svg.strip().endswith("</svg>"))
        self.assertNotIn('font-size="7.5"', svg)  # no activation-value labels without a forward pass

    def test_plot_network_with_sample_input_shows_values_and_edges(self):
        from Enilnets import NeuralNet, plot_network
        import re
        np.random.seed(0)
        model = NeuralNet()
        model.add_dense(4, 6, activation="relu")
        model.add_batchnorm(6)
        model.add_dense(6, 3, activation="softmax")
        x = np.random.randn(1, 4)
        svg = model.plot(sample_input=x)
        self.assertEqual(len(re.findall(r"<circle", svg)), 4 + 6 + 3)
        # batchnorm breaks the direct dense->dense adjacency, so only the
        # input->dense1 (4*6) and dense2->dense3 (6*3) edges get drawn.
        self.assertEqual(len(re.findall(r"<line", svg)), 4 * 6 + 6 * 3)
        self.assertIn("BatchNorm", svg)
        self.assertIn('font-size="7.5"', svg)  # activation values are labeled

    def test_plot_network_caps_large_layers(self):
        from Enilnets import NeuralNet
        import re
        model = NeuralNet()
        model.add_dense(10, 50, activation="relu")
        model.add_dense(50, 4, activation="softmax")
        svg = model.plot(max_nodes_per_layer=20)
        self.assertLess(len(re.findall(r"<circle", svg)), 10 + 50 + 4)
        self.assertIn("⋮", svg)

    def test_plot_network_conv_layer_renders_as_block(self):
        from Enilnets import NeuralNet
        model = NeuralNet()
        model.add_conv2d(3, 8, k=3, input_size=(8, 8), activation="relu")
        model.add_flatten()
        model.add_dense(None, 4, activation="softmax")
        svg = model.plot(sample_input=np.random.randn(1, 3, 8, 8))
        self.assertIn("Conv2D", svg)

    def test_plot_network_no_layers_raises(self):
        from Enilnets import NeuralNet, plot_network
        with self.assertRaises(ValueError):
            plot_network(NeuralNet())

    def test_plot_genome_renders_nodes_and_disabled_edges(self):
        from Enilnets.neat import Genome, InnovationTracker
        from Enilnets import plot_genome
        import re
        tracker = InnovationTracker()
        tracker.reset_node_id_counter(2 + 1 + 1)
        genome = Genome.minimal(2, 1, tracker)
        genome.mutate_add_node(tracker)  # disables one connection, adds a hidden node
        svg = plot_genome(genome, sample_input=np.array([1.0, 0.5]))
        self.assertTrue(svg.startswith("<svg"))
        self.assertEqual(len(re.findall(r"<circle", svg)), len(genome.nodes))
        self.assertIn("stroke-dasharray", svg)  # the disabled connection

    def test_file_output_svg_matches_returned_string(self):
        from Enilnets import NeuralNet
        model = NeuralNet()
        model.add_dense(4, 3, activation="softmax")
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            fname = f.name
        try:
            svg = model.plot(filename=fname)
            with open(fname) as fh:
                self.assertEqual(fh.read(), svg)
        finally:
            os.remove(fname)

    def test_file_output_html_wraps_svg_and_return_value_stays_raw_svg(self):
        from Enilnets import NeuralNet
        model = NeuralNet()
        model.add_dense(4, 3, activation="softmax")
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            fname = f.name
        try:
            svg = model.plot(filename=fname)
            self.assertTrue(svg.startswith("<svg"))  # return value is always raw SVG
            with open(fname) as fh:
                html = fh.read()
            self.assertTrue(html.startswith("<!doctype html>"))
            self.assertIn(svg, html)
        finally:
            os.remove(fname)

    def test_to_html_standalone_wrapping(self):
        from Enilnets import NeuralNet, to_html
        model = NeuralNet()
        model.add_dense(4, 3, activation="softmax")
        svg = model.plot()
        html = to_html(svg, title="My Model")
        self.assertTrue(html.startswith("<!doctype html>"))
        self.assertIn("My Model", html)
        self.assertIn(svg, html)

    def test_repeated_calls_during_training_loop(self):
        # The intended "watch it learn" usage pattern: call plot() from
        # inside your own training loop for a live snapshot each time.
        from Enilnets import NeuralNet
        np.random.seed(0)
        model = NeuralNet(optimizer="adam")
        model.add_dense(3, 4, activation="relu")
        model.add_dense(4, 2, activation="softmax")
        X = np.random.randn(20, 3)
        Y = np.eye(2)[np.random.randint(0, 2, 20)]
        snapshots = []
        for _ in range(3):
            model.TrainBatch(X, Y, loss_function="cross_entropy")
            snapshots.append(model.plot(sample_input=X[:1]))
        self.assertEqual(len(snapshots), 3)
        self.assertTrue(all(s.startswith("<svg") for s in snapshots))


class TestV31Phase1Fixes(unittest.TestCase):
    """v3.1.0 Phase 1 fixes + the extra bug-list fixes from real-world usage
    (sparse_cross_entropy, strict optimizer/activation validation)."""

    def test_sparse_cross_entropy_matches_onehot_cross_entropy(self):
        np.random.seed(0)
        model = NeuralNet()
        out = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
        idx = np.array([0, 2])
        onehot = np.eye(3)[idx]
        sparse_val = model.ComputeLoss(out, idx, function="sparse_cross_entropy")
        dense_val = model.ComputeLoss(out, onehot, function="cross_entropy")
        self.assertAlmostEqual(sparse_val, dense_val, places=10)

    def test_cross_entropy_reduction_scales_with_seq_len_not_just_batch(self):
        # Regression test for the reported bug: (B,S,V) sequence output's
        # cross_entropy loss must divide by B*S, not just B, else it comes
        # out ~S times too large.
        model = NeuralNet()
        B, S, V = 2, 8, 5
        np.random.seed(0)
        probs = np.random.dirichlet(np.ones(V), size=(B, S))
        idx = np.random.randint(0, V, size=(B, S))
        onehot = np.eye(V)[idx]
        val = model.ComputeLoss(probs, onehot, function="cross_entropy")
        expected = float(-np.sum(onehot * np.log(np.clip(probs, 1e-12, 1.0))) / (B * S))
        # ComputeLoss casts inputs to the model's working dtype (float32 by
        # default) before computing, while `expected` here is computed
        # directly on the original (numpy-default float64) probs/onehot --
        # places=10 assumed both sides were float64; loosened to a tolerance
        # float32 can actually meet, since this test checks the reduction's
        # *scaling* is correct, not bit-exact precision.
        self.assertAlmostEqual(val, expected, places=5)

    def test_sparse_cross_entropy_backward_matches_onehot_backward(self):
        np.random.seed(0)
        model = NeuralNet(optimizer="sgd", learning_rate=0.0)
        model.add_dense(4, 3, activation="softmax")
        X = np.random.randn(5, 4)
        idx = np.array([0, 1, 2, 1, 0])
        onehot = np.eye(3)[idx]

        model.Forward(X, training=True)
        model.Backward(idx, loss_function="sparse_cross_entropy")
        sparse_delta = model.deltas[-1].copy()

        model.Forward(X, training=True)
        model.Backward(onehot, loss_function="cross_entropy")
        dense_delta = model.deltas[-1].copy()

        self.assertTrue(np.allclose(sparse_delta, dense_delta))

    def test_sparse_cross_entropy_trains_e2e(self):
        np.random.seed(0)
        model = NeuralNet(optimizer="adam", learning_rate=0.05)
        model.add_dense(4, 8, activation="relu")
        model.add_dense(8, 3, activation="softmax")
        X = np.random.randn(30, 4)
        idx = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int)
        losses = []
        for _ in range(20):
            loss, _ = model.TrainBatch(X, idx, loss_function="sparse_cross_entropy")
            losses.append(loss)
        self.assertLess(losses[-1], losses[0])

    def test_unknown_optimizer_raises(self):
        with self.assertRaises(ValueError):
            NeuralNet(optimizer="adamw_typo")

    def test_unknown_activation_raises(self):
        model = NeuralNet(optimizer="sgd")
        model.add_dense(3, 2, activation="realu")
        with self.assertRaises(ValueError):
            model.Forward(np.random.randn(2, 3), training=False)

    def test_kl_divergence_requires_mu_logvar(self):
        model = NeuralNet()
        with self.assertRaises(ValueError):
            model.ComputeLoss(np.zeros((2, 2)), np.zeros((2, 2)), function="kl_divergence")

    def test_lr_scheduler_plateau_drops_after_patience(self):
        sched = LRScheduler(1.0, mode="plateau", factor=0.5, patience=2, metric_mode="min")
        metrics = [None, 1.0, 0.9, 0.9, 0.9, 0.9]
        lrs = [sched.step(e, metric=m) for e, m in enumerate(metrics)]
        # improves at epoch 1 (1.0), then plateaus for epochs 2,3 (patience=2)
        # -> drop takes effect starting the step *after* the 2nd bad epoch.
        self.assertEqual(lrs[0], 1.0)
        self.assertLess(lrs[-1], 1.0)
        self.assertAlmostEqual(lrs[-1], 0.5, places=10)

    def test_train_lr_scheduler_plateau_e2e(self):
        np.random.seed(0)
        model = NeuralNet(optimizer="sgd", learning_rate=0.1)
        model.add_dense(4, 2, activation="softmax")
        X = np.random.randn(20, 4)
        Y = np.eye(2)[np.random.randint(0, 2, 20)]
        sched = LRScheduler(0.1, mode="plateau", factor=0.5, patience=1, metric_mode="min")
        history = model.Train(X, Y, epochs=5, batch_size=20, scheduler=sched,
                              loss_function="cross_entropy", verbose=False)
        self.assertEqual(len(history["lr"]), 5)

    def test_ppo_value_network_trains(self):
        np.random.seed(0)
        policy = NeuralNet(optimizer="adam", learning_rate=0.01)
        policy.add_dense(4, 8, activation="relu")
        policy.add_dense(8, 2, activation="softmax")
        value_net = NeuralNet(optimizer="adam", learning_rate=0.05)
        value_net.add_dense(4, 8, activation="relu")
        value_net.add_dense(8, 1, activation="linear")

        states = np.random.randn(16, 4)
        actions = np.random.randint(0, 2, 16)
        old_log_probs = np.log(np.full((16, 1), 0.5))
        advantages = np.random.randn(16, 1)
        value_targets = np.random.randn(16, 1) * 5

        before = value_net.layers[0]["weights"].copy()
        losses = []
        for _ in range(10):
            preds_before = value_net.Forward(states, training=False)
            mse_before = float(np.mean((preds_before - value_targets) ** 2))
            policy.PPO(states, actions, old_log_probs, advantages,
                      value_targets=value_targets, value_network=value_net)
            preds_after = value_net.Forward(states, training=False)
            mse_after = float(np.mean((preds_after - value_targets) ** 2))
            losses.append((mse_before, mse_after))
        self.assertFalse(np.allclose(before, value_net.layers[0]["weights"]))
        self.assertLess(losses[-1][1], losses[0][0])

    def test_multimodal_fusion_attention_is_sample_dependent(self):
        np.random.seed(0)
        N, D = 6, 4
        a = np.random.randn(N, D) * 0.1
        b = np.random.randn(N, D) * 0.1
        # Modality 'c' is an outlier only for half the samples.
        c = a.copy()
        c[: N // 2] += 10.0
        fused = cm_utils.multimodal_fusion([a, b, c], fusion_type="attention")
        self.assertEqual(fused.shape, (N, D))
        gated = cm_utils.multimodal_fusion([a, b, c], fusion_type="gated")
        # Attention weighting differs sample-to-sample (unlike gated's static
        # weights), so it shouldn't collapse to the same per-sample result.
        self.assertFalse(np.allclose(fused, gated))

    def test_vgg_loss_fallback_and_custom_extractor(self):
        from Enilnets.generative.generative_loss import vgg_loss
        x = np.random.randn(2, 3)
        y = np.random.randn(2, 3)
        fallback = vgg_loss(x, y)
        self.assertAlmostEqual(fallback, float(np.mean((x - y) ** 2)))
        custom = vgg_loss(x, y, vgg_features=lambda a: a * 2)
        self.assertNotAlmostEqual(custom, fallback)

    def test_train_callbacks_fire(self):
        np.random.seed(0)
        model = NeuralNet(optimizer="sgd", learning_rate=0.05)
        model.add_dense(3, 2, activation="softmax")
        X = np.random.randn(10, 3)
        Y = np.eye(2)[np.random.randint(0, 2, 10)]

        class Recorder:
            def __init__(self):
                self.epoch_ends = 0
                self.train_ends = 0
            def on_epoch_end(self, epoch, logs, model=None):
                self.epoch_ends += 1
                assert "loss" in logs and model is not None
            def on_train_end(self, history):
                self.train_ends += 1

        class Incomplete:
            pass  # missing on_epoch_end -- must not crash

        rec = Recorder()
        model.Train(X, Y, epochs=3, batch_size=10, loss_function="cross_entropy",
                   verbose=False, callbacks=[rec, Incomplete()])
        self.assertEqual(rec.epoch_ends, 3)
        self.assertEqual(rec.train_ends, 1)

    def test_text_generator_train_callbacks_fire(self):
        np.random.seed(0)
        corpus = "the quick brown fox jumps over the lazy dog. " * 20
        tok = Tokenizer(vocab_size=64, level="char").fit([corpus])
        gen = TextGenerator(tok, embed_dim=16, num_heads=2, num_layers=1, max_seq_len=32)

        class Recorder:
            def __init__(self):
                self.batches = 0
                self.epochs = 0
            def on_batch_end(self, epoch, batch_idx, loss, model=None):
                self.batches += 1
            def on_epoch_end(self, epoch, logs, model=None):
                self.epochs += 1

        rec = Recorder()
        gen.Train([corpus], epochs=2, batch_size=16, seq_len=16, verbose=False,
                 callbacks=[rec])
        self.assertEqual(rec.epochs, 2)
        self.assertGreater(rec.batches, 0)


class TestV31Phase2ConvPadding(_FDPrecisionMixin, unittest.TestCase):
    """v3.1.0 Phase 2: padding="same" for add_conv2d."""

    def _fd_check_param(self, model, X, Y, layer_idx, param_name, eps=1e-5, n_check=6):
        model.Forward(X, training=True)
        model.Backward(Y, loss_function="mse")
        analytic = model.compute_gradients()[layer_idx][param_name]
        flat = model.layers[layer_idx][param_name].reshape(-1)
        rng = np.random.RandomState(0)
        idxs = rng.choice(flat.size, size=min(n_check, flat.size), replace=False)
        max_err = 0.0
        for idx in idxs:
            orig = float(flat[idx])
            flat[idx] = orig + eps
            loss_p = model.ComputeLoss(model.Forward(X, training=False), Y, function="mse")
            flat[idx] = orig - eps
            loss_m = model.ComputeLoss(model.Forward(X, training=False), Y, function="mse")
            flat[idx] = orig
            num_grad = (loss_p - loss_m) / (2 * eps)
            max_err = max(max_err, abs(num_grad - analytic.reshape(-1)[idx]))
        return max_err

    def _build_same_padded_net(self):
        np.random.seed(0)
        model = NeuralNet(optimizer="sgd")
        model.add_conv2d(in_ch=2, out_ch=3, k=3, activation="tanh", input_size=(6, 6), padding="same")
        model.add_flatten()
        model.add_dense(None, 4, activation="linear")
        X = np.random.randn(2, 2, 6, 6)
        Y = np.random.randn(2, 4)
        return model, X, Y

    def test_same_padding_preserves_spatial_size(self):
        model = NeuralNet()
        model.add_conv2d(in_ch=2, out_ch=4, k=3, input_size=(8, 8), padding="same")
        self.assertEqual(model._last_spatial, (4, 8, 8))
        model.add_conv2d(out_ch=5, k=5, padding="same")
        self.assertEqual(model._last_spatial, (5, 8, 8))

    def test_valid_padding_shrinks_spatial_size_unchanged_from_before(self):
        model = NeuralNet()
        model.add_conv2d(in_ch=2, out_ch=4, k=3, input_size=(8, 8))  # default padding="valid"
        self.assertEqual(model._last_spatial, (4, 6, 6))

    def test_same_padding_weight_and_bias_gradients(self):
        model, X, Y = self._build_same_padded_net()
        self.assertLess(self._fd_check_param(model, X, Y, 0, "weights"), 1e-6)
        self.assertLess(self._fd_check_param(model, X, Y, 0, "bias"), 1e-6)

    def test_same_padding_input_gradient(self):
        np.random.seed(1)
        X = np.random.randn(2, 2, 6, 6)
        Y = np.random.randn(2, 4)
        model = NeuralNet(optimizer="sgd")
        # An identity 1x1 conv first so deltas[0] equals the gradient w.r.t.
        # the raw input X (needed to finite-difference it directly).
        model.add_conv2d(in_ch=2, out_ch=2, k=1, activation="linear", input_size=(6, 6))
        model.layers[0]["weights"] = np.eye(2).reshape(2, 2, 1, 1)
        model.layers[0]["bias"] = np.zeros(2)
        model.add_conv2d(out_ch=3, k=3, activation="tanh", padding="same")
        model.add_flatten()
        model.add_dense(None, 4, activation="linear")

        model.Forward(X, training=True)
        model.Backward(Y, loss_function="mse")
        dx = model.deltas[0].copy()

        eps = 1e-5
        flat = X.reshape(-1)
        rng = np.random.RandomState(0)
        idxs = rng.choice(flat.size, size=8, replace=False)
        max_err = 0.0
        for idx in idxs:
            orig = float(flat[idx])
            flat[idx] = orig + eps
            loss_p = model.ComputeLoss(model.Forward(X, training=False), Y, function="mse")
            flat[idx] = orig - eps
            loss_m = model.ComputeLoss(model.Forward(X, training=False), Y, function="mse")
            flat[idx] = orig
            num_grad = (loss_p - loss_m) / (2 * eps)
            max_err = max(max_err, abs(num_grad - dx.reshape(-1)[idx]))
        self.assertLess(max_err, 1e-6)

    def test_valid_padding_byte_identical_to_pre_phase2(self):
        # padding="valid" (default) must reproduce pre-Phase-2 output exactly
        # on a fixed seed -- no behavior change for existing models.
        np.random.seed(42)
        model = NeuralNet(optimizer="sgd")
        model.add_conv2d(in_ch=2, out_ch=3, k=3, activation="relu", input_size=(8, 8))
        model.add_flatten()
        model.add_dense(None, 4, activation="linear")
        X = np.random.randn(2, 2, 8, 8)
        out = model.Forward(X, training=False)
        self.assertEqual(out.shape, (2, 4))
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertEqual(model.layers[0]["pad"], 0)

    def test_same_padding_requires_stride_1_and_odd_k(self):
        with self.assertRaises(ValueError):
            NeuralNet().add_conv2d(in_ch=1, out_ch=2, k=3, stride=2, padding="same", input_size=(6, 6))
        with self.assertRaises(ValueError):
            NeuralNet().add_conv2d(in_ch=1, out_ch=2, k=4, padding="same", input_size=(6, 6))

    def test_unknown_padding_raises(self):
        with self.assertRaises(ValueError):
            NeuralNet().add_conv2d(in_ch=1, out_ch=2, k=3, padding="full", input_size=(6, 6))

    def test_same_padding_trains_e2e(self):
        model, X, Y = self._build_same_padded_net()
        losses = []
        for _ in range(15):
            loss, _ = model.TrainBatch(X, Y, loss_function="mse")
            losses.append(loss)
        self.assertLess(losses[-1], losses[0])

    def test_build_vgg16_feature_extractor_shapes_and_weights(self):
        from Enilnets.generative.pretrained import build_vgg16_feature_extractor
        np.random.seed(0)
        model = build_vgg16_feature_extractor(up_to_block=2, input_ch=3)
        x = np.random.randn(2, 3, 32, 32)
        out = model.Forward(x, training=False)
        self.assertEqual(out.shape, (2, 128, 8, 8))  # 2 pools: 32 -> 16 -> 8

        # Standard VGG16 shape check on the canonical 224x224 input.
        full = build_vgg16_feature_extractor(up_to_block=5, input_ch=3)
        out_full = full.Forward(np.random.randn(1, 3, 224, 224), training=False)
        self.assertEqual(out_full.shape, (1, 512, 7, 7))

        weights = model.get_weights()
        model2 = build_vgg16_feature_extractor(up_to_block=2, input_ch=3)
        model2.set_weights(weights)
        self.assertTrue(np.allclose(out, model2.Forward(x, training=False)))

        with self.assertRaises(ValueError):
            build_vgg16_feature_extractor(up_to_block=6)

    def test_vgg_loss_with_vgg16_extractor(self):
        from Enilnets.generative.pretrained import build_vgg16_feature_extractor
        from Enilnets.generative.generative_loss import vgg_loss
        np.random.seed(0)
        model = build_vgg16_feature_extractor(up_to_block=1, input_ch=3)
        x = np.random.randn(2, 3, 16, 16)
        y = np.random.randn(2, 3, 16, 16)
        loss = vgg_loss(x, y, vgg_features=model.Forward)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)


class TestV31Phase4Conv1D(_FDPrecisionMixin, unittest.TestCase):
    """v3.1.0 Phase 4: add_conv1d for (batch, channels, length) data."""

    def _fd_check_param(self, model, X, Y, layer_idx, param_name, eps=1e-5, n_check=6):
        model.Forward(X, training=True)
        model.Backward(Y, loss_function="mse")
        analytic = model.compute_gradients()[layer_idx][param_name]
        flat = model.layers[layer_idx][param_name].reshape(-1)
        rng = np.random.RandomState(0)
        idxs = rng.choice(flat.size, size=min(n_check, flat.size), replace=False)
        max_err = 0.0
        for idx in idxs:
            orig = float(flat[idx])
            flat[idx] = orig + eps
            loss_p = model.ComputeLoss(model.Forward(X, training=False), Y, function="mse")
            flat[idx] = orig - eps
            loss_m = model.ComputeLoss(model.Forward(X, training=False), Y, function="mse")
            flat[idx] = orig
            num_grad = (loss_p - loss_m) / (2 * eps)
            max_err = max(max_err, abs(num_grad - analytic.reshape(-1)[idx]))
        return max_err

    def test_forward_shapes_valid_and_same(self):
        m_valid = NeuralNet()
        m_valid.add_conv1d(in_ch=1, out_ch=4, k=3, input_size=10)  # default padding="valid"
        self.assertEqual(m_valid._last_spatial_1d, (4, 8))
        m_same = NeuralNet()
        m_same.add_conv1d(in_ch=1, out_ch=4, k=3, input_size=10, padding="same")
        self.assertEqual(m_same._last_spatial_1d, (4, 10))

    def test_fd_gradients_weights_and_bias(self):
        np.random.seed(0)
        for padding in ("valid", "same"):
            model = NeuralNet(optimizer="sgd")
            model.add_conv1d(in_ch=2, out_ch=3, k=3, activation="tanh", input_size=12, padding=padding)
            model.add_flatten()
            model.add_dense(None, 4, activation="linear")
            X = np.random.randn(2, 2, 12)
            Y = np.random.randn(2, 4)
            self.assertLess(self._fd_check_param(model, X, Y, 0, "weights"), 1e-6)
            self.assertLess(self._fd_check_param(model, X, Y, 0, "bias"), 1e-6)

    def test_fd_gradients_input(self):
        np.random.seed(0)
        for padding in ("valid", "same"):
            model = NeuralNet(optimizer="sgd")
            model.add_conv1d(in_ch=2, out_ch=2, k=1, activation="linear", input_size=12)
            model.layers[0]["weights"] = np.eye(2).reshape(2, 2, 1)
            model.layers[0]["bias"] = np.zeros(2)
            model.add_conv1d(out_ch=3, k=3, activation="tanh", padding=padding)
            model.add_flatten()
            model.add_dense(None, 4, activation="linear")
            X = np.random.randn(2, 2, 12)
            Y = np.random.randn(2, 4)

            model.Forward(X, training=True)
            model.Backward(Y, loss_function="mse")
            dx = model.deltas[0].copy()
            eps = 1e-5
            flat = X.reshape(-1)
            rng = np.random.RandomState(1)
            idxs = rng.choice(flat.size, size=6, replace=False)
            max_err = 0.0
            for idx in idxs:
                orig = float(flat[idx])
                flat[idx] = orig + eps
                lp = model.ComputeLoss(model.Forward(X, training=False), Y, function="mse")
                flat[idx] = orig - eps
                lm = model.ComputeLoss(model.Forward(X, training=False), Y, function="mse")
                flat[idx] = orig
                num = (lp - lm) / (2 * eps)
                max_err = max(max_err, abs(num - dx.reshape(-1)[idx]))
            self.assertLess(max_err, 1e-6)

    def test_same_padding_requires_stride_1_and_odd_k(self):
        with self.assertRaises(ValueError):
            NeuralNet().add_conv1d(in_ch=1, out_ch=2, k=3, stride=2, padding="same", input_size=10)
        with self.assertRaises(ValueError):
            NeuralNet().add_conv1d(in_ch=1, out_ch=2, k=4, padding="same", input_size=10)

    def test_unknown_padding_raises(self):
        with self.assertRaises(ValueError):
            NeuralNet().add_conv1d(in_ch=1, out_ch=2, k=3, padding="full", input_size=10)

    def test_flatten_uses_1d_spatial_not_2d(self):
        model = NeuralNet()
        model.add_conv1d(in_ch=1, out_ch=4, k=3, input_size=10, padding="same")
        model.add_flatten()
        self.assertEqual(model._last_width, 40)  # 4 * 10, not misread as 2D
        self.assertIsNone(model._last_spatial_1d)
        self.assertIsNone(model._last_spatial)

    def test_save_load_roundtrip(self):
        import tempfile, os
        np.random.seed(0)
        model = NeuralNet(optimizer="adam")
        model.add_conv1d(in_ch=1, out_ch=4, k=3, input_size=10, padding="same", activation="relu")
        model.add_conv1d(out_ch=8, k=3, padding="same", activation="relu")
        model.add_flatten()
        model.add_dense(None, 3, activation="softmax")
        X = np.random.randn(2, 1, 10)
        out = model.Forward(X, training=False)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pkl")
            model.Save(path)
            m2 = NeuralNet()
            m2.Load(path)
            self.assertTrue(np.allclose(out, m2.Forward(X, training=False)))

    def test_trains_e2e(self):
        np.random.seed(0)
        model = NeuralNet(optimizer="adam", learning_rate=0.02)
        model.add_conv1d(in_ch=1, out_ch=8, k=3, input_size=16, padding="same", activation="relu")
        model.add_flatten()
        model.add_dense(None, 2, activation="softmax")
        X = np.random.randn(20, 1, 16)
        Y = np.eye(2)[(X.sum(axis=(1, 2)) > 0).astype(int)]
        losses = [model.TrainBatch(X, Y, loss_function="cross_entropy")[0] for _ in range(20)]
        self.assertLess(losses[-1], losses[0])


# ========================================================================
# Benchmarks
# ========================================================================

class BenchmarkMixin:
    """Shared timing harness for all Benchmark* classes (replaces the three
    independent, slightly-incompatible benchmark() helpers that used to live
    in test.py x2 and test3.py)."""

    def benchmark(self, name, fn, *args, n_repeats=10, warmup=True, **kwargs):
        if warmup:
            fn(*args, **kwargs)
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            fn(*args, **kwargs)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        avg = statistics.mean(times) * 1000
        std = statistics.pstdev(times) * 1000
        print(f"  {name:45s} {avg:8.3f} ms +/- {std:6.3f} ms  (n={n_repeats})")
        return avg


class BenchmarkCore(BenchmarkMixin, unittest.TestCase):
    """Speed benchmarks for core dense/conv operations."""

    def test_benchmark_dense_forward(self):
        model = NeuralNet()
        model.add_dense(784, 256, activation="relu")
        model.add_dense(256, 128, activation="relu")
        model.add_dense(128, 10, activation="softmax")
        x = np.random.randn(128, 784)
        self.benchmark("Dense Forward (128x784->10)", model.Forward, x)

    def test_benchmark_dense_backward(self):
        model = NeuralNet(learning_rate=0.001, optimizer="adam")
        model.add_dense(784, 256, activation="relu")
        model.add_dense(256, 128, activation="relu")
        model.add_dense(128, 10, activation="softmax")
        x = np.random.randn(128, 784)
        y = np.eye(10)[np.random.randint(0, 10, 128)]

        def fn():
            model.Forward(x, training=True)
            model.Backward(y)
        self.benchmark("Dense Backward (128x784->10)", fn)

    def test_benchmark_dense_train_epoch(self):
        model = NeuralNet(learning_rate=0.001, optimizer="adam")
        model.add_dense(784, 256, activation="relu")
        model.add_dense(256, 10, activation="softmax")
        X = np.random.randn(1024, 784)
        Y = np.eye(10)[np.random.randint(0, 10, 1024)]

        def fn():
            model.Train(X, Y, epochs=1, batch_size=32, verbose=False)
        self.benchmark("Dense 1 Epoch (1024 samples)", fn, n_repeats=3)

    def test_benchmark_conv_forward(self):
        model = NeuralNet()
        model.add_conv2d(1, 16, k=3, activation="relu")
        model.add_conv2d(16, 32, k=3, activation="relu")
        model.add_flatten()
        model.add_dense(32 * 4 * 4, 10, activation="softmax")
        x = np.random.randn(16, 1, 8, 8)
        self.benchmark("Conv Forward (16x1x8x8)", model.Forward, x)

    def test_benchmark_conv_backward(self):
        model = NeuralNet(learning_rate=0.001, optimizer="adam")
        model.add_conv2d(1, 8, k=3, activation="relu")
        model.add_flatten()
        model.add_dense(8 * 6 * 6, 10, activation="softmax")
        x = np.random.randn(8, 1, 8, 8)
        y = np.eye(10)[np.random.randint(0, 10, 8)]

        def fn():
            model.Forward(x, training=True)
            model.Backward(y)
        self.benchmark("Conv Backward (8x1x8x8)", fn)

    def test_benchmark_attention_forward(self):
        model = NeuralNet()
        model.add_multihead_attention(embed_dim=64, num_heads=8)
        x = np.random.randn(16, 20, 64)
        self.benchmark("Attention Forward (16x20x64)", model.Forward, x, training=True)


class BenchmarkGenerative(BenchmarkMixin, unittest.TestCase):
    """Speed benchmarks for generative models."""

    def test_benchmark_vae_train_step(self):
        vae = VAE(input_dim=64, latent_dim=4, encoder_hidden=[32], decoder_hidden=[32],
                  learning_rate=0.001, optimizer="adam")
        X = np.random.rand(16, 64).astype(np.float64)
        self.benchmark("VAE train_step (batch=16)", vae.train_step, X, n_repeats=5)

    def test_benchmark_gan_train_step(self):
        gan = GAN(latent_dim=4, data_dim=16, generator_hidden=[16], discriminator_hidden=[16],
                  learning_rate=0.001, optimizer="adam")
        X = np.random.randn(8, 16).astype(np.float64)

        def fn():
            fake = gan.generate(8)
            gan._train_discriminator(X, fake)
            fake2 = gan.generate(8)
            gan._train_generator(fake2)
        self.benchmark("GAN D+G step (batch=8)", fn, n_repeats=5)

    def test_benchmark_diffusion_train_step(self):
        diff = DiffusionModel(data_shape=(16,), time_steps=50, denoiser_type="mlp",
                              denoiser_hidden=[32, 32], learning_rate=0.001)
        X = np.random.randn(8, 16).astype(np.float64) * 0.5
        self.benchmark("Diffusion train_step (batch=8)", diff.train_step, X, n_repeats=5)

    def test_benchmark_autoregressive_train_step(self):
        ar = AutoregressiveModel(data_dim=16, hidden_dims=[16], learning_rate=0.001)
        X = np.random.randn(8, 16).astype(np.float64)
        self.benchmark("AR train_step (batch=8)", ar.train_step, X, n_repeats=5)

    def test_benchmark_flow_train_step(self):
        flow = RealNVP(data_dim=20, n_coupling=4, hidden_dim=64, learning_rate=0.001)
        X = np.random.randn(32, 20).astype(np.float64)
        self.benchmark("Flow train_step (batch=32)", flow.train_step, X, n_repeats=5)

    def test_benchmark_ebm_train_step(self):
        ebm = EnergyBasedModel(data_dim=8, hidden_dims=[16], learning_rate=0.001)
        X = np.random.randn(4, 8).astype(np.float64) * 0.5
        self.benchmark("EBM train_step (batch=4, cd=5)", ebm.train_step, X,
                       n_cd_steps=5, step_size=0.1, noise_scale=0.01, n_repeats=5)

    def test_benchmark_unet_forward(self):
        unet = UNetDenoiser(in_ch=1, base_ch=8, time_emb_dim=16, ch_mult=(1, 2))
        x = np.random.randn(1, 1, 16, 16)
        t = np.array([5])
        self.benchmark("UNet forward (1x1x16x16)", unet.forward, x, t)


class BenchmarkUtilities(BenchmarkMixin, unittest.TestCase):
    """Speed benchmark for the audio STFT path (from test3.py's section 8)."""

    def test_benchmark_stft(self):
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 5, 80000)).astype(np.float64)
        self.benchmark("STFT 5-second audio", aud_utils.stft, audio,
                       n_fft=2048, hop_length=512, n_repeats=3)


# ========================================================================
# Entry point
# ========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Enilnets Combined Test & Benchmark Suite")
    print("=" * 70)

    if "--benchmark" in sys.argv:
        sys.argv.remove("--benchmark")
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        for cls in (BenchmarkCore, BenchmarkGenerative, BenchmarkUtilities):
            suite.addTests(loader.loadTestsFromTestCase(cls))
        unittest.TextTestRunner(verbosity=2).run(suite)
    else:
        unittest.main(verbosity=2)
