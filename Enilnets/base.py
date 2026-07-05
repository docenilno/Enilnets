import numpy as np
import copy
from . import constants

class NeuralNet:
    def __init__(self, learning_rate=0.001, optimizer="adam", l2_lambda=0.01, momentum=0.9,
                 grad_clip_norm=0.0,
                 use_mixed_precision=False,
                 adam_beta1=constants.ADAM_BETA1, adam_beta2=constants.ADAM_BETA2,
                 adam_epsilon=constants.ADAM_EPSILON,
                 rmsprop_decay=constants.RMSPROP_DECAY, rmsprop_epsilon=constants.RMSPROP_EPSILON,
                 adagrad_epsilon=constants.ADAGRAD_EPSILON):
        self.layers = []
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer.lower()
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
        self.outputs = []
        self.pre_activations = []
        self.batchnorm_cache = []
        self.layernorm_cache = []
        self.attention_cache = []
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
        self._residual_stack = []

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def set_lr(self, lr):
        self.learning_rate = lr

    def get_lr(self):
        return self.learning_rate

    def clip_gradients(self, max_norm):
        """Clip gradients by L2 norm across all layers."""
        if max_norm <= 0:
            return
        total_norm = 0.0
        for d in self.deltas:
            if d is not None:
                total_norm += np.sum(d ** 2)
        total_norm = np.sqrt(total_norm)
        clip_coef = max_norm / (total_norm + 1e-12)
        if clip_coef < 1.0:
            for i in range(len(self.deltas)):
                if self.deltas[i] is not None:
                    self.deltas[i] *= clip_coef

    def freeze(self, layer_idx=None):
        """Freeze layers (set learning_rate to 0 for specified layers or all)."""
        if layer_idx is None:
            for layer in self.layers:
                layer["_frozen"] = True
        else:
            self.layers[layer_idx]["_frozen"] = True

    def unfreeze(self, layer_idx=None):
        """Unfreeze layers."""
        if layer_idx is None:
            for layer in self.layers:
                layer["_frozen"] = False
        else:
            self.layers[layer_idx]["_frozen"] = False

    def get_weights(self):
        """Return a copy of all layer weights and biases."""
        weights = []
        for layer in self.layers:
            w = {}
            for k in ["weights", "bias", "gamma", "beta", "mask"]:
                if k in layer:
                    w[k] = layer[k].copy()
            weights.append(w)
        return weights

    def set_weights(self, weights):
        """Set layer weights and biases from a list of dicts."""
        for i, w in enumerate(weights):
            if i >= len(self.layers):
                break
            for k, v in w.items():
                if k in self.layers[i]:
                    self.layers[i][k] = v.copy()

    def copy(self):
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
        )
        net.layers = copy.deepcopy(self.layers)
        net.opt_state = copy.deepcopy(self.opt_state)
        net.t = self.t
        net.training = self.training
        net._last_width = self._last_width
        net._last_spatial = self._last_spatial
        net._residual_stack = list(self._residual_stack)
        return net

    def reset_optimizer_state(self):
        """Clear optimizer state (momentum, velocity buffers)."""
        self.opt_state = []
        self.t = 0
        self._grad_accum = []
        self._accum_steps = 0

    def check_nan_inf(self):
        """Check for NaN/Inf in weights, biases, and deltas."""
        issues = []
        for i, layer in enumerate(self.layers):
            for k in ["weights", "bias", "gamma", "beta"]:
                if k in layer:
                    if not np.all(np.isfinite(layer[k])):
                        issues.append(f"Layer {i} {k} has NaN/Inf")
        for i, d in enumerate(self.deltas):
            if d is not None and not np.all(np.isfinite(d)):
                issues.append(f"Delta {i} has NaN/Inf")
        return issues

    def summary(self):
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
            elif layer_type == "globalavgpool2d":
                print(f"Layer {i}: {layer_type.upper():12s}")
            elif layer_type == "upsample2d":
                print(f"Layer {i}: {layer_type.upper():12s} Scale: {layer['scale_factor']}")
            else:
                print(f"Layer {i}: {layer_type.upper():12s}")
        print(f"Total Parameters: {total_params}")
        print("=" * 70)

# Import and bind all submodule methods after class definition
from .layers import add_dense, add_sparse, add_conv2d, add_flatten, add_maxpool2d, add_avgpool2d, add_batchnorm, add_dropout, add_layernorm, add_global_avgpool2d, add_upsample2d, add_embedding, add_multihead_attention, add_mlp_block, add_conv_block, add_residual_start, add_residual_end, add_rnn, add_lstm, add_gru
from .transformer_layers import add_transformer_block, add_positional_encoding, add_vision_transformer_patch_embed
from .forward import Forward
from .backward import Backward
from .optimizer import update, compute_gradients, apply_gradients, accumulate_gradients, apply_accumulated_gradients
from .loss import ComputeLoss
from .train import TrainBatch, compute_accuracy, Train
from .io import Save, Load
from .reinforce import Evolve, compute_returns, Reinforce, PPO, ActorCritic
from .visualization import plot_network

NeuralNet.add_dense = add_dense
NeuralNet.add_sparse = add_sparse
NeuralNet.add_conv2d = add_conv2d
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
NeuralNet.add_mlp_block = add_mlp_block
NeuralNet.add_conv_block = add_conv_block
NeuralNet.add_residual_start = add_residual_start
NeuralNet.add_residual_end = add_residual_end
NeuralNet.add_rnn = add_rnn
NeuralNet.add_lstm = add_lstm
NeuralNet.add_gru = add_gru
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
NeuralNet.compute_returns = compute_returns
NeuralNet.Reinforce = Reinforce
NeuralNet.PPO = PPO
NeuralNet.ActorCritic = ActorCritic
NeuralNet.Save = Save
NeuralNet.Load = Load
NeuralNet.plot = plot_network
