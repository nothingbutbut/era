import functools
from typing import Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from jaxrl.networks.common import Params, PRNGKey, default_init, Model

LOG_STD_MIN = -8.0
LOG_STD_MAX = 0.0

class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    entropy_per_dim: float
    log_std_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False,
                 return_params: bool=False) -> tfd.Distribution:
        
        outputs = nn.Dense(self.hidden_dims[0], kernel_init=default_init())(observations)
        outputs = nn.elu(outputs)
        outputs = nn.Dense(self.hidden_dims[0], kernel_init=default_init())(outputs)
        outputs = nn.elu(outputs)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        log_stds = nn.Dense(self.action_dim,
                            kernel_init=default_init(
                                self.log_std_scale))(outputs)

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX

        # ERA Implementation
        k = - self.action_dim * (log_std_max - self.entropy_per_dim + jnp.log(jnp.sqrt(2 * jnp.pi * jnp.e)))
        log_stds = k * nn.softmax(log_stds, axis = -1) + log_std_max
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        return means, log_stds


@functools.partial(jax.jit, static_argnames=('actor_def'))
@functools.partial(jax.vmap, in_axes=(0, None, 0, 0, None))
def _sample_actions(
        rng: PRNGKey,
        actor_def: nn.Module,
        actor_params: Params,
        observations: np.ndarray,
        temperature: float) -> Tuple[PRNGKey, jnp.ndarray]:
    
        rng, key = jax.random.split(rng)
        means, log_stds = actor_def.apply({'params': actor_params}, observations, return_params=True)
        actions, _ = sample(key, means, log_stds, temperature=temperature)
        return rng, actions


def sample_actions(
        rng: PRNGKey,
        actor_def: nn.Module,
        actor_params: Params,
        observations: np.ndarray,
        temperature: float = 1.0) -> Tuple[PRNGKey, jnp.ndarray]:
    return _sample_actions(rng, actor_def, actor_params, observations, temperature)


class TruncatedNormal:
    def __init__(self, loc, scale, low, high, epsilon=1e-6):
        self.loc = loc
        self.scale = jnp.maximum(scale, epsilon)
        self.low = low
        self.high = high
        self.epsilon = epsilon
    
    def sample(self, seed=None):
        a = (self.low - self.loc) / self.scale
        b = (self.high - self.loc) / self.scale
        
        alpha = self._standard_normal_cdf(a)
        beta = self._standard_normal_cdf(b)
        
        delta = jnp.maximum(beta - alpha, self.epsilon)
        u = jax.random.uniform(
            seed, 
            shape=self.loc.shape,
            minval=0.0,
            maxval=1.0
        )
        u = alpha + delta * jnp.clip(u, self.epsilon, 1.0 - self.epsilon)
        z = self._standard_normal_ppf(u)
        samples = self.loc + self.scale * z
        samples = jnp.clip(samples, self.low + self.epsilon, self.high - self.epsilon)
        log_probs = self.log_prob(samples)
        
        return samples, log_probs
    
    def log_prob(self, x):
        x = jnp.clip(x, self.low + self.epsilon, self.high - self.epsilon)
        z = (x - self.loc) / self.scale
        log_pdf = self._standard_normal_logpdf(z)
        a = (self.low - self.loc) / self.scale
        b = (self.high - self.loc) / self.scale
        Z = self._standard_normal_cdf(b) - self._standard_normal_cdf(a)
        Z = jnp.maximum(Z, self.epsilon)
        log_prob = log_pdf - jnp.log(self.scale) - jnp.log(Z)
        log_prob = jnp.clip(log_prob, -20.0, 20.0)
        return log_prob.sum(axis=-1)
    
    def _standard_normal_cdf(self, x):
        return 0.5 * (1.0 + jax.scipy.special.erf(x / jnp.sqrt(2.0)))
    
    def _standard_normal_ppf(self, p):
        p_safe = jnp.clip(p, self.epsilon, 1.0 - self.epsilon)
        return jnp.sqrt(2.0) * jax.scipy.special.erfinv(2.0 * p_safe - 1.0)
    
    def _standard_normal_logpdf(self, x):
        return -0.5 * x**2 - 0.5 * jnp.log(2.0 * jnp.pi)


def sample(
        rng: PRNGKey,
        means: jnp.ndarray,
        log_stds: jnp.ndarray,
        temperature: float = 1.0
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    trunc_normal = TruncatedNormal(
        loc=jnp.tanh(means).clip(-1.0 + 1e-6, 1.0 - 1e-6),
        scale=jnp.exp(log_stds) * temperature,
        low=-1.0 + 1e-6,
        high=1.0 - 1e-6,
    )
    actions, log_prob = trunc_normal.sample(seed=rng)
    return actions, log_prob


def log_prob(
        means: jnp.ndarray,
        log_stds: jnp.ndarray,
        actions: jnp.ndarray) -> jnp.ndarray:
    trunc_normal = TruncatedNormal(
        loc=jnp.tanh(means).clip(-1.0 + 1e-6, 1.0 - 1e-6),
        scale=jnp.exp(log_stds),
        low=-1.0 + 1e-6,
        high=1.0 - 1e-6,
    )
    return trunc_normal.log_prob(actions).clip(-20.0, 20.0).sum(axis=-1)