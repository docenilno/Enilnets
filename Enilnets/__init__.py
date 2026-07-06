from .base import NeuralNet
from .train import LRScheduler
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
from .text_utils import Tokenizer
from .utils import (
    set_seed, train_test_split, iterate_minibatches, count_parameters,
    EarlyStopping, one_hot, k_fold_split,
    ModelCheckpoint, CSVLogger, JSONLogger,
)
from .eval_metrics import confusion_matrix, classification_report
from .neat import NEATPopulation, Genome as NEATGenome, crossover as neat_crossover
from .visualization import plot_network, plot_genome, to_html
from . import constants
from .backend import use_gpu, gpu_available, is_gpu_enabled, use_float64, is_float64_enabled, default_dtype

__version__ = "4.1.0"
__all__ = [
    "NeuralNet", "LRScheduler",
    "use_gpu", "gpu_available", "is_gpu_enabled",
    "use_float64", "is_float64_enabled", "default_dtype",
    "VAE", "GAN", "DiffusionModel", "AutoregressiveModel",
    "RealNVP", "EnergyBasedModel", "UNetDenoiser", "time_embedding", "TextGenerator",
    "Tokenizer",
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
