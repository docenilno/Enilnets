from .core.base import NeuralNet
from .nn.train import LRScheduler
from .nn.kvcache import KVCache, cached_forward_step
from .datasets import (DataLoader, Dataset, IterableDataset, ArrayDataset,
                       MemmapDataset, StreamingDataset, Subset, ConcatDataset,
                       random_split)
from .preprocessing.pipeline import Compose
from .compression import (prune_magnitude, prune_channels,
                          PruningSchedule, sparsity, quantize_weights,
                          ActivationCalibrator)
from .optim.averaging import EMA, SWA
from .optim.lr_finder import find_learning_rate
from .generative import (
    VAE, GAN, DiffusionModel, AutoregressiveModel,
    RealNVP, EnergyBasedModel, UNetDenoiser, time_embedding, TextGenerator,
    reparameterize, langevin_dynamics, gaussian_sample,
    uniform_sample, gumbel_softmax_sample, random_mask,
    top_p_sampling, top_k_sampling,
    kl_divergence_gaussian, adversarial_loss_discriminator,
    adversarial_loss_generator, diffusion_loss, nll_loss, energy_loss,
    compute_returns,
)
from .text.text_utils import Tokenizer
from .text.subword import BPETokenizer
from .core.utils import (
    set_seed, train_test_split, iterate_minibatches, count_parameters,
    EarlyStopping, one_hot, k_fold_split,
    ModelCheckpoint, CSVLogger, JSONLogger,
)
from .metrics.eval_metrics import confusion_matrix, classification_report
from .evolving.neat import NEATPopulation, Genome as NEATGenome, crossover as neat_crossover
from .visualization import plot_network, plot_genome, to_html
from .core import constants
from .core.backend import use_gpu, gpu_available, is_gpu_enabled, use_float64, is_float64_enabled, default_dtype

__version__ = "4.11.0"
__all__ = [
    "NeuralNet", "LRScheduler", "KVCache", "cached_forward_step",
    "EMA", "SWA", "find_learning_rate",
    "DataLoader", "Dataset", "IterableDataset", "ArrayDataset",
    "MemmapDataset", "StreamingDataset", "Subset", "ConcatDataset",
    "random_split", "Compose",
    "prune_magnitude", "prune_channels", "PruningSchedule", "sparsity",
    "quantize_weights", "ActivationCalibrator",
    "use_gpu", "gpu_available", "is_gpu_enabled",
    "use_float64", "is_float64_enabled", "default_dtype",
    "VAE", "GAN", "DiffusionModel", "AutoregressiveModel",
    "RealNVP", "EnergyBasedModel", "UNetDenoiser", "time_embedding", "TextGenerator",
    "Tokenizer", "BPETokenizer",
    "reparameterize", "langevin_dynamics", "gaussian_sample",
    "uniform_sample", "gumbel_softmax_sample", "random_mask",
    "top_p_sampling", "top_k_sampling",
    "kl_divergence_gaussian", "adversarial_loss_discriminator",
    "adversarial_loss_generator", "diffusion_loss", "nll_loss", "energy_loss",
    "compute_returns",
    "set_seed", "train_test_split", "iterate_minibatches", "count_parameters",
    "EarlyStopping", "one_hot", "k_fold_split", "constants",
    "ModelCheckpoint", "CSVLogger", "JSONLogger",
    "confusion_matrix", "classification_report",
    "NEATPopulation", "NEATGenome", "neat_crossover",
    "plot_network", "plot_genome", "to_html",
]

# ---------------------------------------------------------------------------
# Backward-compatible aliases for the pre-Phase-0 flat module layout.
# Code (and pickles) written against e.g. ``Enilnets.base`` / ``Enilnets.backend``
# keeps working: each old flat module name is registered in ``sys.modules`` as
# an alias of its new subpackage home. The public re-export list above is the
# real API surface; these aliases exist purely so no existing import breaks.
# ---------------------------------------------------------------------------
import sys as _sys
from .core import base, backend, utils
from .optim import optimizer
from .losses import loss
from .metrics import eval_metrics, eval_utils
from .reinforcement import reinforce
from .evolving import neat
from .vision import image_utils
from .text import text_utils
from .audio import audio_utils
from .crossmodal import crossmodal_utils
from .nn import (
    forward, backward, layers, transformer_layers,
    train, io, activations, weight_init,
)

# Enilnets.functional -- the stateless graph-op API (roadmap item 27),
# exposed at the top level the way the roadmap's own example spells it
# (Enilnets.functional.relu(x)).
from .graph import functional
_sys.modules.setdefault(f"{__name__}.functional", functional)

_FLAT_ALIASES = {
    "base": base,
    "backend": backend,
    "constants": constants,
    "utils": utils,
    "forward": forward,
    "backward": backward,
    "layers": layers,
    "transformer_layers": transformer_layers,
    "train": train,
    "io": io,
    "activations": activations,
    "weight_init": weight_init,
    "optimizer": optimizer,
    "loss": loss,
    "eval_metrics": eval_metrics,
    "eval_utils": eval_utils,
    "reinforce": reinforce,
    "neat": neat,
    "image_utils": image_utils,
    "text_utils": text_utils,
    "audio_utils": audio_utils,
    "crossmodal_utils": crossmodal_utils,
}
for _name, _mod in _FLAT_ALIASES.items():
    _sys.modules.setdefault(f"{__name__}.{_name}", _mod)
del _sys, _name, _mod
