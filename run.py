
# %%
import torch as t
from torch import nn, Tensor
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
import numpy as np
import einops
from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple
from functools import partial
from tqdm.notebook import tqdm
from fancy_einsum import einsum
import matplotlib.pyplot as plt

from utils.plotly_utils import imshow, line, hist
from utils.utils import (
    plot_features_in_2d,
    plot_features_in_Nd,
    plot_features_in_Nd_discrete,
    plot_correlated_features,
    plot_feature_geometry,
    frac_active_line_plot,
)
from Models import Model, Config, NeuronModel, NeuronComputationModel

device = t.device("mps" if t.backends.mps.is_available() else "cpu")

def get_std_importance_feature_probs(n_features=5, n_instances=8, small=True, importance_power=0.9):
    importance = (importance_power ** t.arange(n_features))
    importance = einops.rearrange(importance, "features -> () features")

    # sparsity is the same for all features in a given instance, but varies over instances
    feature_probability = (50 ** -t.linspace(0, 1, n_instances)) if small else t.tensor([1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001])
    feature_probability = einops.rearrange(feature_probability, "instances -> instances ()")

    return importance, feature_probability

def small_demonstrate_basic_superposition():
    
    cfg = Config(
        n_instances = 8,
        n_features = 5,
        n_hidden = 2,
    )
    importance, feature_probability = get_std_importance_feature_probs(n_features=cfg.n_features, n_instances=cfg.n_instances)

    line(importance.squeeze(), width=600, height=400, title="Importance of each feature (same over all instances)", labels={"y": "Feature importance", "x": "Feature"})
    line(feature_probability.squeeze(), width=600, height=400, title="Feature probability (varied over instances)", labels={"y": "Probability", "x": "Instance"})
    

    model = Model(
        cfg = cfg,
        device = device,
        importance = importance,
        feature_probability = feature_probability,
    )
    model.optimize_or_load(steps=10_000)

    plot_features_in_2d(
        model.W.detach(),
        colors = model.importance,
        title = "Superposition: 5 features represented in 2D space",
        subplot_titles = [f"1 - S = {i:.3f}" for i in feature_probability.squeeze()],
    )

    with t.inference_mode():
        batch = model.generate_batch(200)
        hidden = einops.einsum(batch, model.W, "batch_size instances features, instances hidden features -> instances hidden batch_size")

    plot_features_in_2d(hidden, title = "Hidden state representation of a random batch of data")

    plt.show()

def large_demonstrate_basic_superposition():
    n_features = 80
    n_hidden = 20

    importance, feature_probability = get_std_importance_feature_probs(n_features=n_features, n_instances=7, small=False)

    cfg = Config(
        n_instances = len(feature_probability.squeeze()),
        n_features = n_features,
        n_hidden = n_hidden,
    )

    line(importance.squeeze(), width=600, height=400, title="Importance of each feature (same over all instances)", labels={"y": "Feature importance", "x": "Feature"})
    line(feature_probability.squeeze(), width=600, height=400, title="Feature probability (varied over instances)", labels={"y": "Probability", "x": "Instance"})

    model = Model(
        cfg = cfg,
        device = device,
        importance = importance,
        feature_probability = feature_probability,
    )
    model.optimize_or_load(steps=10_000)

    plot_features_in_Nd(
        model.W,
        height = 600,
        width = 1400,
        title = "ReLU output model: n_features = 80, d_hidden = 20, I<sub>i</sub> = 0.9<sup>i</sup>",
        subplot_titles = [f"Feature prob = {i:.3f}" for i in feature_probability[:, 0]],
    )

def corr_anticorr():
    
    cfg = Config(
        n_instances = 30,
        n_features = 4,
        n_hidden = 2,
        n_correlated_pairs = 1,
        n_anticorrelated_pairs = 1,
    )

    feature_probability = 10 ** -t.linspace(0.5, 1, cfg.n_instances).to(device)

    model = Model(
        cfg = cfg,
        device = device,
        feature_probability = einops.rearrange(feature_probability, "instances -> instances ()")
    )
    
    batch = model.generate_batch(batch_size = 1)
    correlated_feature_batch, anticorrelated_feature_batch = batch[:, :, :2], batch[:, :, 2:]

    # Plot correlated features
    plot_correlated_features(correlated_feature_batch, title="Correlated Features: should always co-occur")
    plot_correlated_features(anticorrelated_feature_batch, title="Anti-correlated Features: should never co-occur")
    
    cfg = Config(
        n_instances = 5,
        n_features = 4,
        n_hidden = 2,
        n_correlated_pairs = 2,
        n_anticorrelated_pairs = 0,
    )

    # All same importance, very low feature probabilities (ranging from 5% down to 0.25%)
    importance = t.ones(cfg.n_features, dtype=t.float, device=device)
    importance = einops.rearrange(importance, "features -> () features")
    feature_probability = (400 ** -t.linspace(0.5, 1, cfg.n_instances))
    feature_probability = einops.rearrange(feature_probability, "instances -> instances ()")

    model = Model(
        cfg = cfg,
        device = device,
        importance = importance,
        feature_probability = feature_probability,
    )
    model.optimize_or_load()

    plot_features_in_2d(
        model.W,
        colors = ["blue"] * 2 + ["limegreen"] * 2, # when colors is a list of strings, it's assumed to be the colors of features
        title = "Correlated feature sets are represented in local orthogonal bases",
        subplot_titles = [f"1 - S = {i:.3f}" for i in model.feature_probability[:, 0]],
    )
    plt.show()

    cfg = Config(
        n_instances = 5,
        n_hidden = 2,
        n_features = 4,
        n_correlated_pairs = 0,
        n_anticorrelated_pairs = 2,
    )

    # All same importance, very low feature probabilities (ranging from 5% down to 0.25%)
    importance = t.ones(cfg.n_features, dtype=t.float, device=device)
    importance = einops.rearrange(importance, "features -> () features")
    feature_probability = (400 ** -t.linspace(0.5, 1, cfg.n_instances))
    feature_probability = einops.rearrange(feature_probability, "instances -> instances ()")

    model = Model(
        cfg = cfg,
        device = device,
        importance = importance,
        feature_probability = feature_probability,
    )
    model.optimize_or_load()

    plot_features_in_2d(
        model.W,
        colors = ["blue"] * 2 + ["limegreen"] * 2, # when colors is a list of strings, it's assumed to be the colors of features
        title = "Anticorrelated feature sets are represented in local orthogonal bases",
        subplot_titles = [f"1 - S = {i:.3f}" for i in model.feature_probability[:, 0]],
    )
    
    plt.show()

    cfg = Config(
        n_instances = 5,
        n_hidden = 2,
        n_features = 6,
        n_correlated_pairs = 3,
        n_anticorrelated_pairs = 0,
    )

    # All same importance, very low feature probabilities (ranging from 5% down to 0.25%)
    importance = t.ones(cfg.n_features, dtype=t.float, device=device)
    importance = einops.rearrange(importance, "features -> () features")
    feature_probability = (400 ** -t.linspace(0.5, 1, cfg.n_instances))
    feature_probability = einops.rearrange(feature_probability, "instances -> instances ()")

    model = Model(
        cfg = cfg,
        device = device,
        importance = importance,
        feature_probability = feature_probability,
    )
    
    model.optimize_or_load()
    
    plot_features_in_2d(
        model.W,
        colors = ["blue"] * 2 + ["limegreen"] * 2 +["red"]*2, # when colors is a list of strings, it's assumed to be the colors of features
        title = "Anticorrelated feature sets are represented in local orthogonal bases",
        subplot_titles = [f"1 - S = {i:.3f}" for i in model.feature_probability[:, 0]],
    )

    plt.show()

def demonstrate_NeuronModel():

    n_features = 10
    n_hidden = 5

    importance = einops.rearrange(0.75 ** t.arange(1, 1+n_features), "feats -> () feats")
    feature_probability = einops.rearrange(t.tensor([0.75, 0.35, 0.15, 0.1, 0.06, 0.02, 0.01]), "instances -> instances ()")

    cfg = Config(
        n_instances = len(feature_probability.squeeze()),
        n_features = n_features,
        n_hidden = n_hidden,
    )

    model = NeuronModel(
        cfg = cfg,
        device = device,
        importance = importance,
        feature_probability = feature_probability,
    )
    model.optimize_or_load(steps=10_000)

    plot_features_in_Nd(
        model.W,
        height = 600,
        width = 1000,
        title = "Neuron model: n_features = 10, d_hidden = 5, I<sub>i</sub> = 0.75<sup>i</sup>",
        subplot_titles = [f"1 - S = {i:.2f}" for i in feature_probability.squeeze()],
        neuron_plot = True,
    )

def demonstrate_NeuronComputationModel():
    n_features = 100
    n_hidden = 40

    importance = einops.rearrange(0.8 ** t.arange(1, 1+n_features), "feats -> () feats")
    feature_probability = einops.rearrange(t.tensor([1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]), "instances -> instances ()")

    cfg = Config(
        n_instances = len(feature_probability.squeeze()),
        n_features = n_features,
        n_hidden = n_hidden,
    )

    model = NeuronComputationModel(
        cfg = cfg,
        device = device,
        importance = importance,
        feature_probability = feature_probability,
    )
    model.optimize_or_load(steps=10_000)

    plot_features_in_Nd(
        model.W1,
        height = 800,
        width = 1600,
        title = f"Neuron computation model: n_features = {n_features}, d_hidden = {n_hidden}, I<sub>i</sub> = 0.75<sup>i</sup>",
        subplot_titles = [f"1 - S = {i:.3f}" for i in feature_probability.squeeze()],
        neuron_plot = True,
    )

    n_features = 10
    n_hidden = 10

    importance = einops.rearrange(0.8 ** t.arange(1, 1+n_features), "feats -> () feats")

    cfg = Config(
        n_instances = 5,
        n_features = n_features,
        n_hidden = n_hidden,
    )

    model = NeuronComputationModel(
        cfg = cfg,
        device = device,
        importance = importance,
        feature_probability = 0.5,
    )
    model.optimize_or_load(steps=10_000)

    plot_features_in_Nd_discrete(
        W1 = model.W1,
        W2 = model.W2,
        height = 600,
        width = 1200,
        title = f"Neuron computation model (colored discretely, by feature)",
        legend_names = [f"I<sub>{i}</sub> = {importance.squeeze()[i]:.3f}" for i in range(n_features)],
    )

if __name__ == "__main__":
    
    small_demonstrate_basic_superposition()

    large_demonstrate_basic_superposition()

    corr_anticorr()

    demonstrate_NeuronModel()

    demonstrate_NeuronComputationModel()

    





# %%
