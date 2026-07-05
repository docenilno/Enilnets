"""
Shared default constants for Enilnets.

These are the single source of truth for epsilon guards against div/log-by-
zero, optimizer hyperparameters, activation clip bounds, etc. Every public
function/constructor that uses one of these exposes it as an overridable
keyword argument defaulting to the value here; the module attributes
themselves can also be changed directly (e.g. `Enilnets.constants.EPS_LOG =
1e-10`) to change the default everywhere at once, since call sites read
`constants.NAME` at call time.
"""

# Generic numerical-stability epsilons.
EPS_LOG = 1e-12    # guards log(0) in cross-entropy/BCE-style losses
EPS_DIV = 1e-8     # guards division by a norm/denominator that could be ~0

# Sigmoid / softplus overflow-safe clipping range for exp().
SIGMOID_CLIP = 500.0

# Activation function parameters.
LEAKYRELU_ALPHA = 0.01
ELU_ALPHA = 1.0

# Adam optimizer.
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8

# RMSprop.
RMSPROP_DECAY = 0.9
RMSPROP_EPSILON = 1e-8

# Adagrad.
ADAGRAD_EPSILON = 1e-8

# Weight init.
NORMAL_INIT_STD = 0.1

# Positional encoding / time embedding base frequency.
SINUSOIDAL_BASE = 10000.0

# NEAT (NeuroEvolution of Augmenting Topologies).
NEAT_COMPATIBILITY_THRESHOLD = 3.0
NEAT_C1 = 1.0   # excess-gene coefficient in the compatibility distance
NEAT_C2 = 1.0   # disjoint-gene coefficient
NEAT_C3 = 0.4   # matching-weight-difference coefficient
NEAT_WEIGHT_MUTATE_RATE = 0.8
NEAT_WEIGHT_PERTURB_RATE = 0.9   # fraction of weight mutations that perturb vs. replace
NEAT_WEIGHT_PERTURB_POWER = 0.5
NEAT_ADD_CONNECTION_RATE = 0.05
NEAT_ADD_NODE_RATE = 0.03
NEAT_CROSSOVER_RATE = 0.75
NEAT_SURVIVAL_THRESHOLD = 0.2   # fraction of each species allowed to reproduce
NEAT_STAGNATION_LIMIT = 15      # generations without improvement before a species is deprioritized
