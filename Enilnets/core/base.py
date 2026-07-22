"""NeuralNet: the core sequential-layer-list model class. Its methods are
implemented across several sibling modules (layers.py, forward.py,
backward.py, optimizer.py, loss.py, train.py, io.py, reinforce.py,
transformer_layers.py, visualization.py) and bound onto the class at the
bottom of this file, so a `NeuralNet` instance has the full public API
even though most of it isn't defined in this file directly."""
from typing import Any, Dict, List, Optional

from .backend import np
import copy
from . import constants
from ..nn.io import _LAYER_ARRAY_KEYS
from ..optim.optimizer import OPTIMIZERS

class NeuralNet:
    """A sequential list of layer dicts (`self.layers`), each describing
    one layer's type, weights, and hyperparameters. Build one with the
    `add_*` methods (layers.py/transformer_layers.py), then
    `Forward`/`Backward`/`update` (or the `TrainBatch`/`Train` wrappers)
    to train it."""

    def __init__(self, learning_rate: float = 0.001, optimizer: str = "adam",
                 l2_lambda: float = 0.01, momentum: float = 0.9,
                 grad_clip_norm: float = 0.0,
                 use_mixed_precision: bool = False,
                 adam_beta1: float = constants.ADAM_BETA1, adam_beta2: float = constants.ADAM_BETA2,
                 adam_epsilon: float = constants.ADAM_EPSILON,
                 rmsprop_decay: float = constants.RMSPROP_DECAY,
                 rmsprop_epsilon: float = constants.RMSPROP_EPSILON,
                 adagrad_epsilon: float = constants.ADAGRAD_EPSILON,
                 adadelta_rho: float = constants.ADADELTA_RHO,
                 adadelta_epsilon: float = constants.ADADELTA_EPSILON,
                 lion_beta1: float = constants.LION_BETA1,
                 lion_beta2: float = constants.LION_BETA2,
                 adafactor_eps1: float = constants.ADAFACTOR_EPS1,
                 adafactor_clip_threshold: float = constants.ADAFACTOR_CLIP_THRESHOLD,
                 adafactor_decay_rate: float = constants.ADAFACTOR_DECAY_RATE) -> None:
        """
        learning_rate, optimizer ("sgd"|"rmsprop"|"adagrad"|"adam"|"adamw"),
        l2_lambda, momentum (sgd only): standard optimizer hyperparameters.
        grad_clip_norm: if > 0, TrainBatch automatically calls
            clip_gradients(grad_clip_norm) after every Backward().
        use_mixed_precision: reserved for future use.
        adam_beta1/adam_beta2/adam_epsilon, rmsprop_decay/rmsprop_epsilon,
            adagrad_epsilon: per-optimizer tuning constants, defaulting to
            constants.py's standard values.
        """
        optimizer_type = optimizer.lower()
        if optimizer_type not in OPTIMIZERS:
            raise ValueError(
                f"Unknown optimizer: {optimizer!r}. Expected one of "
                + ", ".join(repr(o) for o in OPTIMIZERS) + "."
            )
        self.layers = []
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.l2_lambda = l2_lambda
        self.momentum = momentum
        self.grad_clip_norm = grad_clip_norm
        self.use_mixed_precision = use_mixed_precision
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.rmsprop_decay = rmsprop_decay
        self.rmsprop_epsilon = rmsprop_epsilon
        self.adagrad_epsilon = adagrad_epsilon
        self.adadelta_rho = adadelta_rho
        self.adadelta_epsilon = adadelta_epsilon
        self.lion_beta1 = lion_beta1
        self.lion_beta2 = lion_beta2
        self.adafactor_eps1 = adafactor_eps1
        self.adafactor_clip_threshold = adafactor_clip_threshold
        self.adafactor_decay_rate = adafactor_decay_rate
        self.outputs = []
        self.pre_activations = []
        self.batchnorm_cache = []
        self.layernorm_cache = []
        self.attention_cache = []
        self.moe_cache = []
        self.conv_cache = []
        self.rnn_cache = []
        self.deltas = []
        self.opt_state = []
        self.t = 0
        self.training = True
        self._grad_accum = []
        self._accum_steps = 0
        self._last_width = None
        self._last_spatial = None
        self._last_spatial_1d = None
        self._residual_stack = []

    def train(self) -> "NeuralNet":
        """Switch to training mode (dropout/batchnorm behave accordingly).
        Returns self for chaining."""
        self.training = True
        return self

    def eval(self) -> "NeuralNet":
        """Switch to inference mode (dropout/batchnorm behave accordingly).
        Returns self for chaining."""
        self.training = False
        return self

    def set_lr(self, lr: float) -> None:
        self.learning_rate = lr

    def get_lr(self) -> float:
        return self.learning_rate

    # Layer types whose param gradients (once Backward() has run) live
    # directly on the layer dict as d_<param> rather than being recomputed
    # from self.deltas on every compute_gradients() call -- see
    # optimizer.py's compute_gradients comments: Backward()'s main loop
    # already runs multihead_attention_backward/cross_attention_backward/
    # rnn_backward/lstm_backward/gru_backward for every layer of these
    # types EXCEPT index 0 as a side effect, caching d_Wq/d_Wh/etc.
    # directly, before clip_gradients ever runs.
    _CACHED_GRAD_PARAMS = {
        "multihead_attention": ("Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo"),
        "cross_attention": ("Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo"),
        "rnn": ("Wx", "Wh", "b"),
        "lstm": ("Wx", "Wh", "b"),
        "gru": ("Wx", "Wh", "bx", "bh"),
    }

    def clip_gradients(self, max_norm: float) -> None:
        """Clip gradients by global L2 norm across every trainable
        parameter -- matching the standard clip_grad_norm_ semantics (the
        norm thresholded is the real per-parameter gradient norm, not the
        norm of backprop deltas, which scale with batch size/sequence
        length/layer width in ways unrelated to true gradient magnitude).
        """
        if max_norm <= 0:
            return
        grads = self.compute_gradients()
        total_norm = 0.0
        for g in grads:
            if g is None:
                continue
            for arr in g.values():
                total_norm += np.sum(arr ** 2)
        total_norm = np.sqrt(total_norm)
        clip_coef = max_norm / (total_norm + 1e-12)
        if clip_coef >= 1.0:
            return
        # Scaling self.deltas makes every layer whose gradient is computed
        # FRESH from deltas on the next compute_gradients() call (dense/
        # sparse/conv, and attention/rnn/lstm/gru layers specifically at
        # index 0) come out correctly scaled.
        for i in range(len(self.deltas)):
            if self.deltas[i] is not None:
                self.deltas[i] = self.deltas[i] * clip_coef
        # Attention/rnn/lstm/gru layers at any OTHER index don't recompute
        # from self.deltas at all -- scaling deltas alone silently has no
        # effect on them, since their d_* values are already finalized.
        # Scale those cached values directly instead.
        for i, layer in enumerate(self.layers):
            if i == 0:
                continue
            params = self._CACHED_GRAD_PARAMS.get(layer["type"])
            if params is None:
                continue
            for p in params:
                key = f"d_{p}"
                if key in layer:
                    layer[key] = layer[key] * clip_coef

    def _validate_layer_idx(self, layer_idx: int) -> None:
        if not (-len(self.layers) <= layer_idx < len(self.layers)):
            raise IndexError(
                f"layer_idx={layer_idx} is out of range for a network with "
                f"{len(self.layers)} layer(s) (valid range: "
                f"{-len(self.layers)} to {len(self.layers) - 1})."
            )

    def freeze(self, layer_idx: Optional[int] = None) -> None:
        """Freeze layers (set learning_rate to 0 for specified layers or all)."""
        if layer_idx is None:
            for layer in self.layers:
                layer["_frozen"] = True
        else:
            self._validate_layer_idx(layer_idx)
            self.layers[layer_idx]["_frozen"] = True

    def unfreeze(self, layer_idx: Optional[int] = None) -> None:
        """Unfreeze layers."""
        if layer_idx is None:
            for layer in self.layers:
                layer["_frozen"] = False
        else:
            self._validate_layer_idx(layer_idx)
            self.layers[layer_idx]["_frozen"] = False

    def get_weights(self) -> List[Dict[str, Any]]:
        """Return a copy of all layer weights and biases."""
        weights = []
        for layer in self.layers:
            w = {}
            for k in _LAYER_ARRAY_KEYS:
                if k in layer:
                    w[k] = layer[k].copy()
            weights.append(w)
        return weights

    def set_weights(self, weights: List[Dict[str, Any]]) -> None:
        """Set layer weights and biases from a list of dicts."""
        for i, w in enumerate(weights):
            if i >= len(self.layers):
                break
            for k, v in w.items():
                if k in self.layers[i]:
                    self.layers[i][k] = v.copy()

    def copy(self) -> "NeuralNet":
        """Create a deep copy of the network."""
        net = NeuralNet(
            learning_rate=self.learning_rate,
            optimizer=self.optimizer_type,
            l2_lambda=self.l2_lambda,
            momentum=self.momentum,
            grad_clip_norm=self.grad_clip_norm,
            use_mixed_precision=self.use_mixed_precision,
            adam_beta1=self.adam_beta1, adam_beta2=self.adam_beta2, adam_epsilon=self.adam_epsilon,
            rmsprop_decay=self.rmsprop_decay, rmsprop_epsilon=self.rmsprop_epsilon,
            adagrad_epsilon=self.adagrad_epsilon,
            adadelta_rho=self.adadelta_rho, adadelta_epsilon=self.adadelta_epsilon,
            lion_beta1=self.lion_beta1, lion_beta2=self.lion_beta2,
            adafactor_eps1=self.adafactor_eps1,
            adafactor_clip_threshold=self.adafactor_clip_threshold,
            adafactor_decay_rate=self.adafactor_decay_rate,
        )
        net.layers = copy.deepcopy(self.layers)
        net.opt_state = copy.deepcopy(self.opt_state)
        net.t = self.t
        net.training = self.training
        net._last_width = self._last_width
        net._last_spatial = self._last_spatial
        net._last_spatial_1d = self._last_spatial_1d
        net._residual_stack = list(self._residual_stack)
        return net

    def reset_optimizer_state(self) -> None:
        """Clear optimizer state (momentum, velocity buffers)."""
        self.opt_state = []
        self.t = 0
        self._grad_accum = []
        self._accum_steps = 0

    def check_nan_inf(self) -> List[str]:
        """Check for NaN/Inf in weights, biases, and deltas."""
        issues = []
        for i, layer in enumerate(self.layers):
            for k in _LAYER_ARRAY_KEYS:
                if k in layer:
                    if not np.all(np.isfinite(layer[k])):
                        issues.append(f"Layer {i} {k} has NaN/Inf")
        for i, d in enumerate(self.deltas):
            if d is not None and not np.all(np.isfinite(d)):
                issues.append(f"Delta {i} has NaN/Inf")
        return issues

    def summary(self) -> None:
        """Print a layer-by-layer table (type, shape, parameter count) and
        the total parameter count."""
        print("Model Summary")
        print("=" * 70)
        print(f"Optimizer: {self.optimizer_type.upper()} | LR: {self.learning_rate} | L2: {self.l2_lambda}")
        print("=" * 70)
        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_type = layer["type"]
            if layer_type in ("dense", "sparse"):
                params = layer["weights"].size + layer["bias"].size
                total_params += params
                in_shape = layer["weights"].shape[1]
                out_shape = layer["weights"].shape[0]
                print(f"Layer {i}: {layer_type.upper():12s} Input: {in_shape:6d} Output: {out_shape:6d} Params: {params}")
            elif layer_type == "conv2d":
                params = layer["weights"].size + layer["bias"].size
                total_params += params
                print(f"Layer {i}: {layer_type.upper():12s} In_ch: {layer['in_ch']:3d} Out_ch: {layer['out_ch']:3d} Kernel: {layer['k']}x{layer['k']} Params: {params}")
            elif layer_type == "conv1d":
                params = layer["weights"].size + layer["bias"].size
                total_params += params
                print(f"Layer {i}: {layer_type.upper():12s} In_ch: {layer['in_ch']:3d} Out_ch: {layer['out_ch']:3d} Kernel: {layer['k']} Params: {params}")
            elif layer_type == "batchnorm":
                params = layer["gamma"].size + layer["beta"].size
                total_params += params
                print(f"Layer {i}: {layer_type.upper():12s} Features: {layer['num_features']:6d} Params: {params}")
            elif layer_type == "layernorm":
                params = layer["gamma"].size + layer["beta"].size
                total_params += params
                print(f"Layer {i}: {layer_type.upper():12s} Shape: {layer['normalized_shape']} Params: {params}")
            elif layer_type == "embedding":
                params = layer["weights"].size
                total_params += params
                print(f"Layer {i}: {layer_type.upper():12s} Vocab: {layer['vocab_size']:6d} Dim: {layer['embed_dim']:4d} Params: {params}")
            elif layer_type == "positional_encoding":
                params = layer["weights"].size if layer.get("_pos_type") == "learnable" else 0
                total_params += params
                print(f"Layer {i}: {layer_type.upper():12s} MaxLen: {layer['max_seq_len']:6d} Dim: {layer['embed_dim']:4d} Params: {params}")
            elif layer_type in ("multihead_attention", "cross_attention"):
                params = sum(layer[p].size for p in ("Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo"))
                total_params += params
                print(f"Layer {i}: {layer_type.upper():12s} Dim: {layer['embed_dim']:6d} Heads: {layer['num_heads']:4d} Params: {params}")
            elif layer_type in ("rnn", "lstm", "gru"):
                param_names = ("Wx", "Wh", "bx", "bh") if layer_type == "gru" else ("Wx", "Wh", "b")
                params = sum(layer[p].size for p in param_names)
                total_params += params
                print(f"Layer {i}: {layer_type.upper():12s} Input: {layer['n_in']:6d} Hidden: {layer['hidden_dim']:6d} Params: {params}")
            elif layer_type == "globalavgpool2d":
                print(f"Layer {i}: {layer_type.upper():12s}")
            elif layer_type == "upsample2d":
                print(f"Layer {i}: {layer_type.upper():12s} Scale: {layer['scale_factor']}")
            else:
                print(f"Layer {i}: {layer_type.upper():12s}")
        print(f"Total Parameters: {total_params}")
        print("=" * 70)

# Import and bind all submodule methods after class definition
from ..nn.vision_blocks import (add_global_maxpool2d, add_channel_pool, add_spp,
                                add_se_block, add_cbam_channel, add_cbam_block,
                                add_convnext_block, add_efficientnet_block)
from ..nn.layers import add_dense, add_sparse, add_conv2d, add_conv1d, add_flatten, add_maxpool2d, add_avgpool2d, add_batchnorm, add_dropout, add_layernorm, add_global_avgpool2d, add_upsample2d, add_embedding, add_multihead_attention, add_cross_attention, add_moe, add_mlp_block, add_conv_block, add_residual_start, add_residual_end, add_multiply_end, add_rnn, add_lstm, add_gru, reset_rnn_state, add_bidirectional_rnn, add_bidirectional_lstm, add_bidirectional_gru
from ..nn.transformer_layers import add_transformer_block, add_positional_encoding, add_vision_transformer_patch_embed
from ..nn.forward import Forward
from ..nn.backward import Backward
from ..optim.optimizer import update, compute_gradients, apply_gradients, accumulate_gradients, apply_accumulated_gradients
from ..losses.loss import ComputeLoss
from ..nn.train import TrainBatch, compute_accuracy, Train
from ..nn.io import Save, Load
from ..reinforcement.reinforce import Evolve, Reinforce, PPO, ActorCritic
from ..visualization.plotting import plot_network

NeuralNet.add_dense = add_dense
NeuralNet.add_sparse = add_sparse
NeuralNet.add_conv2d = add_conv2d
NeuralNet.add_conv1d = add_conv1d
NeuralNet.add_flatten = add_flatten
NeuralNet.add_maxpool2d = add_maxpool2d
NeuralNet.add_avgpool2d = add_avgpool2d
NeuralNet.add_batchnorm = add_batchnorm
NeuralNet.add_dropout = add_dropout
NeuralNet.add_layernorm = add_layernorm
NeuralNet.add_global_avgpool2d = add_global_avgpool2d
NeuralNet.add_upsample2d = add_upsample2d
NeuralNet.add_embedding = add_embedding
NeuralNet.add_multihead_attention = add_multihead_attention
NeuralNet.add_moe = add_moe


def moe_aux_loss(self):
    """Sum of every MoE layer's load-balancing loss from the last Forward().
    Reported separately from the task loss; its gradient is already folded
    into Backward()."""
    return float(sum(l.get("_state_aux_loss", 0.0) for l in self.layers
                     if l.get("type") == "moe"))


NeuralNet.moe_aux_loss = moe_aux_loss
NeuralNet.add_cross_attention = add_cross_attention
NeuralNet.add_mlp_block = add_mlp_block
NeuralNet.add_conv_block = add_conv_block
NeuralNet.add_residual_start = add_residual_start
NeuralNet.add_residual_end = add_residual_end
NeuralNet.add_multiply_end = add_multiply_end
NeuralNet.add_global_maxpool2d = add_global_maxpool2d
NeuralNet.add_channel_pool = add_channel_pool
NeuralNet.add_spp = add_spp
NeuralNet.add_se_block = add_se_block
NeuralNet.add_cbam_channel = add_cbam_channel
NeuralNet.add_cbam_block = add_cbam_block
NeuralNet.add_convnext_block = add_convnext_block
NeuralNet.add_efficientnet_block = add_efficientnet_block
NeuralNet.add_rnn = add_rnn
NeuralNet.add_lstm = add_lstm
NeuralNet.add_gru = add_gru
NeuralNet.reset_rnn_state = reset_rnn_state
NeuralNet.add_bidirectional_rnn = add_bidirectional_rnn
NeuralNet.add_bidirectional_lstm = add_bidirectional_lstm
NeuralNet.add_bidirectional_gru = add_bidirectional_gru
NeuralNet.add_transformer_block = add_transformer_block
NeuralNet.add_positional_encoding = add_positional_encoding
NeuralNet.add_vision_transformer_patch_embed = add_vision_transformer_patch_embed
NeuralNet.Forward = Forward
NeuralNet.predict = Forward
NeuralNet.Backward = Backward
NeuralNet.update = update
NeuralNet.compute_gradients = compute_gradients
NeuralNet.apply_gradients = apply_gradients
NeuralNet.accumulate_gradients = accumulate_gradients
NeuralNet.apply_accumulated_gradients = apply_accumulated_gradients
NeuralNet.TrainBatch = TrainBatch
NeuralNet.compute_accuracy = compute_accuracy
NeuralNet.Train = Train
NeuralNet.ComputeLoss = ComputeLoss
NeuralNet.Evolve = Evolve
NeuralNet.Reinforce = Reinforce
NeuralNet.PPO = PPO
NeuralNet.ActorCritic = ActorCritic
NeuralNet.Save = Save
NeuralNet.Load = Load
NeuralNet.plot = plot_network
