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

# AdaDelta. The canonical formulation has no learning rate at all -- the
# step size comes from the ratio of accumulated updates to accumulated
# gradients -- so `learning_rate` acts as a plain multiplier on it and
# should normally be left at 1.0.
ADADELTA_RHO = 0.95
ADADELTA_EPSILON = 1e-6

# Lion. The paper's recommended betas differ from Adam's: a slower second
# moment (0.99) with the same first (0.9). Lion's decay is decoupled, so
# `l2_lambda` is applied straight to the weights like AdamW's.
LION_BETA1 = 0.9
LION_BETA2 = 0.99

# AdaFactor. eps1 floors the squared-gradient accumulator; clip_threshold
# caps the RMS of an update; decay_rate drives the step-dependent second-
# moment decay beta2_t = 1 - t**decay_rate.
ADAFACTOR_EPS1 = 1e-30
ADAFACTOR_CLIP_THRESHOLD = 1.0
ADAFACTOR_DECAY_RATE = -0.8

# LAMB. The layer-wise trust ratio ||w|| / ||r|| is clamped to this ceiling
# so a large weight tensor with a tiny update cannot produce an enormous
# step (the reference implementations all clamp similarly).
LAMB_MAX_TRUST_RATIO = 10.0

# RAdam. Below this value of the estimated variance-rectification term the
# adaptive denominator is not yet trustworthy, and RAdam falls back to a
# plain momentum step (Liu et al. 2020, section 4).
RADAM_RHO_THRESHOLD = 4.0

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
