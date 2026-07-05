import numpy as np

class NeuralNet:
    def __init__(self, learning_rate=0.001, optimizer="adam", l2_lambda=0.01, momentum=0.9):
        self.layers = []
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer.lower()
        self.l2_lambda = l2_lambda
        self.momentum = momentum
        self.outputs = []
        self.pre_activations = []
        self.batchnorm_cache = []
        self.deltas = []
        self.opt_state = []
        self.t = 0

    def summary(self):
        print("Model Summary")
        print("=" * 60)
        print(f"Optimizer: {self.optimizer_type.upper()} | LR: {self.learning_rate} | L2: {self.l2_lambda}")
        print("=" * 60)
        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_type = layer["type"]
            if layer_type in ("dense", "sparse"):
                params = layer["weights"].size + layer["bias"].size
                total_params += params
                print(f"Layer {i}: {layer_type.upper()} - Input: {layer['weights'].shape[1]}, Output: {layer['weights'].shape[0]}, Params: {params}")
            elif layer_type == "conv2d":
                params = layer["weights"].size + layer["bias"].size
                total_params += params
                print(f"Layer {i}: {layer_type.upper()} - In_ch: {layer['in_ch']}, Out_ch: {layer['out_ch']}, Kernel: {layer['k']}x{layer['k']}, Params: {params}")
            elif layer_type == "batchnorm":
                params = layer["gamma"].size + layer["beta"].size
                total_params += params
                print(f"Layer {i}: {layer_type.upper()} - Features: {layer['num_features']}, Params: {params}")
            else:
                print(f"Layer {i}: {layer_type.upper()}")
        print(f"Total Parameters: {total_params}")
        print("=" * 60)

# Import and bind all submodule methods after class definition
from .layers import add_dense, add_sparse, add_conv2d, add_flatten, add_maxpool2d, add_avgpool2d, add_batchnorm, add_dropout
from .forward import Forward
from .backward import Backward
from .optimizer import update
from .loss import ComputeLoss
from .train import TrainBatch, compute_accuracy, Train
from .io import Save, Load
from .reinforce import Evolve, compute_returns, Reinforce

NeuralNet.add_dense = add_dense
NeuralNet.add_sparse = add_sparse
NeuralNet.add_conv2d = add_conv2d
NeuralNet.add_flatten = add_flatten
NeuralNet.add_maxpool2d = add_maxpool2d
NeuralNet.add_avgpool2d = add_avgpool2d
NeuralNet.add_batchnorm = add_batchnorm
NeuralNet.add_dropout = add_dropout
NeuralNet.Forward = Forward
NeuralNet.predict = Forward
NeuralNet.Backward = Backward
NeuralNet.update = update
NeuralNet.TrainBatch = TrainBatch
NeuralNet.compute_accuracy = compute_accuracy
NeuralNet.Train = Train
NeuralNet.ComputeLoss = ComputeLoss
NeuralNet.Evolve = Evolve
NeuralNet.compute_returns = compute_returns
NeuralNet.Reinforce = Reinforce
NeuralNet.Save = Save
NeuralNet.Load = Load
