from typing import Tuple

import jax.numpy as jnp

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey, tree_norm
from jaxrl.networks.policies import sample, log_prob


def update(key: PRNGKey, actor: Model, critic: Model, batch: Batch) -> Tuple[Model, InfoDict]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        means, log_stds = actor.apply({'params': actor_params}, batch.observations, return_params=True)
        actions, log_probs = sample(key, means, log_stds)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = -q.mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'actor_pnorm': tree_norm(actor_params),
            'actor_action': jnp.mean(jnp.abs(actions))
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    info['actor_gnorm'] = info.pop('grad_norm')

    return new_actor, info