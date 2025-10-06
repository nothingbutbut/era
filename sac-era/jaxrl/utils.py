import os
import numpy as np
import wandb
import logging
import tensorflow_probability.substrates.numpy as tfp
        
def make_env(benchmark, env_name, seed, num_envs):
    if benchmark == 'dmc':
        from jaxrl.envs.dmc_gym import make_env_dmc
        return make_env_dmc(env_name, seed=seed, num_envs=num_envs)
    if benchmark == 'gym':
        from jaxrl.envs.openai_gym import make_env_gym
        return make_env_gym(env_name, seed=seed, num_envs=num_envs)
    if benchmark == 'hb':
        from jaxrl.envs.hb import make_env_gym_hb
        return make_env_gym_hb(env_name, seed=seed, num_envs=num_envs)
    raise NotImplementedError

        