import torch as t
from torch import nn, Tensor
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from dataclasses import dataclass
import numpy as np
import einops
from jaxtyping import Float, Int
from typing import Optional, Callable, Union, List, Tuple
from functools import partial
from tqdm import tqdm
from dataclasses import dataclass
from fancy_einsum import einsum


def linear_lr(step, steps):
    return (1 - (step / steps))

def constant_lr(*_):
    return 1.0

def cosine_decay_lr(step, steps):
    return np.cos(0.5 * np.pi * step / (steps - 1))


@dataclass
class Config:
    # We optimize n_instances models in a single training loop to let us sweep over
    # sparsity or importance curves  efficiently. You should treat `n_instances` as
    # kinda like a batch dimension, but one which is built into our training setup.
    n_instances: int
    n_features: int = 5
    n_hidden: int = 2
    n_correlated_pairs: int = 0
    n_anticorrelated_pairs: int = 0


class Model(nn.Module):
    W: Float[Tensor, "n_instances n_hidden n_features"]
    b_final: Float[Tensor, "n_instances n_features"]
    # Our linear map is x -> ReLU(W.T @ W @ x + b_final)

    def __init__(
        self,
        cfg: Config,
        feature_probability: Optional[Union[float, Tensor]] = None,
        importance: Optional[Union[float, Tensor]] = None,
        device: t.device = t.device("mps" if t.backends.mps.is_available() else "cpu"),
    ):
        super().__init__()
        self.cfg = cfg

        if feature_probability is None: feature_probability = t.ones(())
        if isinstance(feature_probability, float): feature_probability = t.tensor(feature_probability)
        self.feature_probability = feature_probability.to(device).broadcast_to((cfg.n_instances, cfg.n_features))
        
        if importance is None: importance = t.ones(())
        if isinstance(importance, float): importance = t.tensor(importance)
        self.importance = importance.to(device).broadcast_to((cfg.n_instances, cfg.n_features))

        self.W = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_hidden, cfg.n_features))))
        self.b_final = nn.Parameter(t.zeros((cfg.n_instances, cfg.n_features)))
        self.to(device)
        self.device = device
        self.steps_trained = 0

    def generate_correlated_features(self, batch_size, n_correlated_pairs) -> Float[Tensor, "batch_size instances features"]:
        '''
        Generates a batch of correlated features.
        Each output[i, j, 2k] and output[i, j, 2k + 1] are correlated, i.e. one is present iff the other is present.
        '''
        present_features = einops.repeat(t.rand((batch_size, self.cfg.n_instances, n_correlated_pairs), device=self.device) <= einops.repeat(self.feature_probability[:, :n_correlated_pairs], "instances features -> batch instances features", batch=batch_size), "batch instances features -> batch instances (features 2)")

        batch = t.zeros((batch_size, self.cfg.n_instances, 2*n_correlated_pairs), device=self.device)

        batch[present_features] = t.rand((batch_size, self.cfg.n_instances, 2*n_correlated_pairs), device=self.device)[present_features]

        return batch      


    def generate_anticorrelated_features(self, batch_size, n_anticorrelated_pairs) -> Float[Tensor, "batch_size instances features"]:
        '''
        Generates a batch of anti-correlated features.
        Each output[i, j, 2k] and output[i, j, 2k + 1] are anti-correlated, i.e. one is present iff the other is absent.
        '''
        present_features1 = t.rand((batch_size, self.cfg.n_instances, n_anticorrelated_pairs), device=self.device) <= einops.repeat(self.feature_probability[:, self.cfg.n_correlated_pairs:self.cfg.n_correlated_pairs + n_anticorrelated_pairs], "instances features -> batch instances features", batch=batch_size)
        present_features2 = ~present_features1

        batch = t.zeros((batch_size, self.cfg.n_instances, 2*n_anticorrelated_pairs), device=self.device)

        rnds = t.rand((batch_size, self.cfg.n_instances, 2*n_anticorrelated_pairs), device=self.device)

        batch[:,:, ::2][present_features1] = rnds[:, :, ::2][present_features1]
        batch[:, :, 1::2][present_features2] = rnds[:, :, 1::2][present_features2]

        return batch      


    def generate_uncorrelated_features(self, batch_size, n_uncorrelated) -> Float[Tensor, "batch_size instances features"]:
        '''
        Generates a batch of uncorrelated features.
        '''
        present_features = t.rand((batch_size, self.cfg.n_instances, n_uncorrelated), device=self.device) <= einops.repeat(self.feature_probability[:, self.cfg.n_correlated_pairs + self.cfg.n_anticorrelated_pairs:], "instances features -> batch instances features", batch=batch_size)

        batch = t.zeros((batch_size, self.cfg.n_instances, n_uncorrelated), device=self.device)

        batch[present_features] = t.rand((batch_size, self.cfg.n_instances, n_uncorrelated), device=self.device)[present_features]

        return batch        


    def generate_batch(self, batch_size):
        '''
        Generates a batch of data, with optional correslated & anticorrelated features.
        '''
        n_uncorrelated = self.cfg.n_features - 2 * self.cfg.n_correlated_pairs - 2 * self.cfg.n_anticorrelated_pairs
        data = []
        if self.cfg.n_correlated_pairs > 0:
            data.append(self.generate_correlated_features(batch_size, self.cfg.n_correlated_pairs))
        if self.cfg.n_anticorrelated_pairs > 0:
            data.append(self.generate_anticorrelated_features(batch_size, self.cfg.n_anticorrelated_pairs))
        if n_uncorrelated > 0:
            data.append(self.generate_uncorrelated_features(batch_size, n_uncorrelated))
        batch = t.cat(data, dim=-1)
        return batch

    def forward(
        self,
        features: Float[Tensor, "... instances features"]
    ) -> Float[Tensor, "... instances features"]:
        
        features = features.to(self.device)
        hidden_representation = einsum("... instances features, instances hidden_features features -> ... instances hidden_features", features, self.W)

        return F.relu(einsum("... instances hidden_features, instances hidden_features features -> ... instances features", hidden_representation, self.W) + self.b_final)        


    def calculate_loss(
        self,
        out: Float[Tensor, "batch instances features"],
        batch: Float[Tensor, "batch instances features"],
    ) -> Float[Tensor, ""]:
        '''
        Calculates the loss for a given batch, using this loss described in the Toy Models paper:

            https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating-setup-loss

        Remember, `model.importance` will always have shape (n_instances, n_features).
        '''
        non_weighted_loss = (out - batch).pow(2)
        weighted_loss = einsum("batch instances features -> ",
            non_weighted_loss * einops.repeat(self.importance, "instances features -> batch instances features", batch=out.shape[0])
        )
        return weighted_loss/(out.shape[0] * out.shape[2])
    
    def optimize_or_load(
        self,
        batch_size: int = 1024,
        steps: int = 10_000,
        log_freq: int = 100,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = constant_lr,
        save = True,
    ):
        self.steps_trained = steps
        try:
            self.load_state_dict(t.load(self.filename()))
            print("Loaded model from file.")
        except FileNotFoundError:
            print("Optimizing model...")
            self.optimize(batch_size, steps, log_freq, lr, lr_scale, save)
        
    def optimize(
        self,
        batch_size: int = 1024,
        steps: int = 10_000,
        log_freq: int = 100,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = constant_lr,
        save = True,
    ):
        '''
        Optimizes the model using the given hyperparameters.
        '''
        optimizer = t.optim.Adam(list(self.parameters()), lr=lr)

        progress_bar = tqdm(range(steps))

        for step in progress_bar:

            # Update learning rate
            step_lr = lr * lr_scale(step, steps)
            for group in optimizer.param_groups:
                group['lr'] = step_lr

            # Optimize
            optimizer.zero_grad()
            batch = self.generate_batch(batch_size)
            out = self(batch)
            loss = self.calculate_loss(out, batch)
            loss.backward()
            optimizer.step()

            # Display progress bar
            if step % log_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(loss=loss.item()/self.cfg.n_instances, lr=step_lr)
        
        self.steps_trained = steps
        
        if save:
            t.save(self.state_dict(), self.filename())
    
    def filename(self):
        return f"./saved_models/std_model_{self.cfg}.pkl"


class NeuronModel(Model):
    def __init__(
        self,
        cfg: Config,
        feature_probability: Optional[Tensor] = None,
        importance: Optional[Tensor] = None,
        device: t.device = t.device("mps" if t.backends.mps.is_available() else "cpu"),
    ):
        super().__init__(cfg, feature_probability, importance, device)

    def forward(
        self,
        features: Float[Tensor, "... instances features"]
    ) -> Float[Tensor, "... instances features"]:
        
        features = features.to(self.device)
        mid = F.relu(einsum("... instances features, instances hidden_features features -> ... instances hidden_features", features, self.W))

        return F.relu(einsum("... instances hidden_features, instances hidden_features features -> ... instances features", mid, self.W) + self.b_final)
    
    def filename(self):
        return f"./saved_models/neuron_model_{self.cfg}.pkl"


class NeuronComputationModel(Model):
    W1: Float[Tensor, "n_instances n_hidden n_features"]
    W2: Float[Tensor, "n_instances n_features n_hidden"]
    b_final: Float[Tensor, "n_instances n_features"]

    def __init__(
        self,
        cfg: Config,
        feature_probability: Optional[Tensor] = None,
        importance: Optional[Tensor] = None,
        device: t.device = t.device("mps" if t.backends.mps.is_available() else "cpu"),
    ):
        super().__init__(cfg, feature_probability, importance, device)

        del self.W
        self.W1 = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_hidden, cfg.n_features))))
        self.W2 = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_features, cfg.n_hidden))))
        self.to(device)


    def forward(
        self,
        features: Float[Tensor, "... instances features"]
    ) -> Float[Tensor, "... instances features"]:

        features = features.to(self.device)
        
        mid = F.relu(einsum("... instances features, instances hidden_features features -> ... instances hidden_features", features, self.W1))

        return F.relu(einsum("... instances hidden_features, instances features hidden_features -> ... instances features", mid, self.W2) + self.b_final)


    def generate_batch(self, batch_size) -> Tensor:

        present_features = t.rand((batch_size, self.cfg.n_instances, self.cfg.n_features), device=device) <= einops.repeat(self.feature_probability, "instances features -> batch instances features", batch=batch_size)

        batch = t.zeros((batch_size, self.cfg.n_instances, self.cfg.n_features), device=device)

        batch[present_features] = 2 * t.rand((batch_size, self.cfg.n_instances, self.cfg.n_features), device=device)[present_features] - 1

        return batch


    def calculate_loss(
        self,
        out: Float[Tensor, "batch instances features"],
        batch: Float[Tensor, "batch instances features"],
    ) -> Float[Tensor, ""]:

        non_weighted_loss = (abs(out) - abs(batch)).pow(2)
        weighted_loss = einsum("batch instances features -> ",
            non_weighted_loss * einops.repeat(self.importance, "instances features -> batch instances features", batch=out.shape[0])
        )
        return weighted_loss/(out.shape[0] * out.shape[2])
    
    def filename(self):
        return f"./saved_models/neuron_computation_model_{self.cfg}.pkl"