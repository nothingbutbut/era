"""Implementations of algorithms for continuous control."""

from typing import Callable, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from jaxrl.networks.common import MLP
from jaxrl.networks.common import default_init, split_tree
    

class Critic(nn.Module):
    hidden_dims: Sequence[int]
    depth: int = 2
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    def setup(self):
        self.output_layer = nn.Dense(1, kernel_init=default_init())
        self.output_layer(jnp.zeros((1, self.hidden_dims[-1])))

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)

        x = nn.Dense(self.hidden_dims[0], kernel_init=default_init())(inputs)
        x = nn.elu(nn.LayerNorm()(x))
        x = nn.Dense(self.hidden_dims[0], kernel_init=default_init())(x)
        x = nn.elu(nn.LayerNorm()(x))
        x = self.output_layer(x)

        return jnp.squeeze(x, -1)


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    depth: int = 2
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = Critic(self.hidden_dims, depth=self.depth, 
                         activations=self.activations)(observations, actions)
        critic2 = Critic(self.hidden_dims, depth=self.depth,
                         activations=self.activations)(observations, actions)
        return critic1, critic2
