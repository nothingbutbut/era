"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import os

from jaxrl.agents.sac.actor import update as update_actor
from jaxrl.agents.sac.critic import target_update
from jaxrl.agents.sac.critic import update as update_critic
from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey


@functools.partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, None, None, None, None, None))
def _update(
    rng: PRNGKey, actor: Model, critic: Model, target_critic: Model, batch: Batch, 
    init_discount: float, final_discount: float, tau: float, step: int, max_steps: int,
) -> Tuple[PRNGKey, Model, Model, Model, InfoDict]:
    
    step_int = step.astype(jnp.float32)
    discount = init_discount + (final_discount - init_discount) * jnp.minimum(step_int / jnp.float32(max_steps), 1.0)

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(key,
                                            actor,
                                            critic,
                                            target_critic,
                                            batch,
                                            discount)
    new_target_critic = target_update(new_critic, target_critic, tau)
    """
    rng, key = jax.random.split(rng)
    new_critic_buffer, critic_buffer_info = update_critic_buffer(
        key, critic_buffer, v_buffer, batch, discount)
    rng, key = jax.random.split(rng)
    new_v_buffer, v_buffer_info = update_v_buffer(key, v_buffer, critic_buffer_target, batch)
    new_critic_buffer_target = target_update(new_critic_buffer, critic_buffer_target, tau)
    """
    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(
        key, actor, new_critic, batch)
    return rng, new_actor, new_critic, new_target_critic, {**actor_info, **critic_info}


@functools.partial(jax.jit, static_argnames=('init_discount', 'final_discount', 'tau', 'max_steps', 'num_updates'))
def _do_multiple_updates(rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
                         batches: Batch, init_discount: float, final_discount: float, tau: float,
                         step, max_steps, num_updates: int) -> Tuple[PRNGKey, Model, Model, Model, InfoDict]:
    def one_step(i, state):
        step, rng, actor, critic, target_critic, info = state
        step = step + 1
        new_rng, new_actor, new_critic, new_target_critic, info = _update(
                rng, actor, critic, target_critic,
                jax.tree_map(lambda x: jnp.take(x, i, axis=1), batches), init_discount, final_discount, tau,
                step, max_steps,
                )
        return step, new_rng, new_actor, new_critic, new_target_critic, info
    step, rng, actor, critic, target_critic, info = one_step(0, (step, rng, actor, critic, target_critic, {}))
    return jax.lax.fori_loop(1, num_updates, one_step,
                             (step, rng, actor, critic, target_critic, info))


class SACLearner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 init_discount: float = 0.99,
                 final_discount: float = 0.99,
                 max_steps: int = 500000,
                 tau: float = 0.005,
                 init_temperature: float = 1.0,
                 num_seeds: int = 5,
                 entropy_per_dim: float = -0.5,
                 algo: str = 'sac',
                 ) -> None:
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """
        self.seeds = jnp.arange(seed, seed + num_seeds)
        action_dim = actions.shape[-1]

        self.tau = tau
        self.init_discount = init_discount
        self.final_discount = final_discount
        self.max_steps = max_steps

        def _init_models(seed):
            rng = jax.random.PRNGKey(seed)
            rng, actor_key, critic_key = jax.random.split(rng, 3)
            actor_def = policies.NormalTanhPolicy(hidden_dims, action_dim, entropy_per_dim)
            actor = Model.create(actor_def,
                                 inputs=[actor_key, observations],
                                 tx=optax.adam(learning_rate=actor_lr))

            critic_def = critic_net.DoubleCritic(hidden_dims)
            critic = Model.create(critic_def,
                                  inputs=[critic_key, observations, actions],
                                  tx=optax.adamw(learning_rate=critic_lr))
            target_critic = Model.create(
                critic_def, inputs=[critic_key, observations, actions])

            return actor, critic, target_critic, rng

        self.init_models = jax.jit(jax.vmap(_init_models))
        self.actor, self.critic, self.target_critic, self.rng = self.init_models(self.seeds)
        self.step = 1

        self.hidden_dims = hidden_dims
        self.critic_lr = critic_lr
        self.observations = observations
        self.actions = actions

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                               self.actor.params, observations,
                                               temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch, num_updates: int = 1) -> InfoDict:
        step, rng, actor, critic, target_critic, info = _do_multiple_updates(
            self.rng, self.actor, self.critic, self.target_critic, batch, self.init_discount, self.final_discount, self.tau, self.step, self.max_steps, num_updates)
        self.step = step
        self.rng = rng
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        return info

    def reset(self):
        self.step = 1
        self.actor, self.critic, self.target_critic, self.rng = self.init_models(self.seeds)