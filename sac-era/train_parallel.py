import os
import random

import argparse
#device = 0
argparser = argparse.ArgumentParser()
argparser.add_argument('--device', type=int, default=0)
# debug mode
argparser.add_argument('--debug', action='store_true')
args, unknown = argparser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
import math

FLAGS = flags.FLAGS

flags.DEFINE_string('exp', '', 'Experiment description (not actually used).')
flags.DEFINE_string('benchmark', 'dmc', 'dmc/mw/myo/gym/adroit/dexhand')
flags.DEFINE_string('env_name', 'dog-run', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 10000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 25000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6)+1, 'Number of training steps.')
flags.DEFINE_integer('replay_buffer_size', int(1e6)+1, 'Number of training steps.')
flags.DEFINE_integer('start_training', 5000,
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_integer('updates_per_step', 2, 'Number of updates per step.')
flags.DEFINE_integer('num_seeds', 5, 'Number of parallel seeds to run.')
flags.DEFINE_integer('device', 0, 'GPU device to use.')
flags.DEFINE_boolean('debug', False, 'Debug mode.')

# ERA Related
flags.DEFINE_float('entropy_per_dim', 0.25, 'Target entropy per dimension.')

# wandb
flags.DEFINE_string('project', 'SAC-ERA', 'Wandb project name.')

config_flags.DEFINE_config_file(
    'config',
    'configs/sac_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

from jaxrl.agents import SACLearner
from jaxrl.datasets import ParallelReplayBuffer
from jaxrl.utils import make_env
import copy
import pickle

import wandb
os.environ["WANDB_MODE"] = "offline"

import jax
assert jax.device_count() == 1 # We only support single GPU training for now.
print("JAX using device: ", args.device)

def log_multiple_seeds_to_wandb(step, infos):
    dict_to_log = {}
    for info_key in infos:
        for seed, value in enumerate(infos[info_key]):
            dict_to_log[f'seed{seed}/{info_key}'] = value
    wandb.log(dict_to_log, step=step)

max_returns : float = -math.inf

def evaluate_if_time_to(i, agent, eval_env, eval_returns, info, seeds):
    global max_returns
    if i % FLAGS.eval_interval == 0:
        eval_stats = eval_env.evaluate(agent, FLAGS.eval_episodes)
        for j, seed in enumerate(seeds):
            eval_returns[j].append(
                (i, eval_stats['return'][j]))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{seed}.txt'),
                       eval_returns[j],
                       fmt=['%d', '%.1f'])
        log_multiple_seeds_to_wandb(i, eval_stats)


def main(_):
    FLAGS.save_dir = os.path.join(FLAGS.save_dir,
                                 FLAGS.benchmark + '-' + FLAGS.env_name + '-' + FLAGS.entropy_per_dim.__str__() + '-seed' + FLAGS.seed.__str__())
    wandb.init(
        project=FLAGS.project,
        name=FLAGS.benchmark + '-' + FLAGS.env_name + '-' + FLAGS.entropy_per_dim.__str__() + '-seed' + FLAGS.seed.__str__(),
        )
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    if args.debug:
        FLAGS.max_steps = 1000
        FLAGS.eval_interval = 800
        FLAGS.log_interval = 800
        FLAGS.start_training = 50
        FLAGS.num_seeds = 2

    env = make_env(FLAGS.benchmark, FLAGS.env_name, FLAGS.seed, num_envs=FLAGS.num_seeds)
    eval_env = make_env(FLAGS.benchmark, FLAGS.env_name, FLAGS.seed + 42, num_envs=FLAGS.num_seeds)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    # Kwards setup
    all_kwargs = FLAGS.flag_values_dict()
    all_kwargs.update(all_kwargs.pop('config'))
    kwargs = dict(FLAGS.config)

    agent = SACLearner(FLAGS.seed,
                  env.observation_space.sample()[0, np.newaxis],
                  env.action_space.sample()[0, np.newaxis], num_seeds=FLAGS.num_seeds,
                  entropy_per_dim=FLAGS.entropy_per_dim,
                  **kwargs)

    replay_buffer = ParallelReplayBuffer(env.observation_space, env.action_space.shape[-1],
                                         FLAGS.replay_buffer_size,
                                         num_seeds=FLAGS.num_seeds)
    observations, truns, rewards, infos = env.reset(), False, 0.0, {}
    start_step, update_count, eval_returns = 1, 0, [[] for _ in range(FLAGS.num_seeds)]

    for i in tqdm.tqdm(range(start_step, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            actions = env.action_space.sample()
        else:
            actions = agent.sample_actions(observations)

        step_info = env.step(actions)
        if len(step_info) == 5:
            next_observations, rewards, terms, truns, _ = step_info
        elif len(step_info) == 4:
            next_observations, rewards, terms, truns = step_info

        masks = env.generate_masks(terms, truns)

        replay_buffer.insert(observations, actions, rewards, masks, truns,
                             next_observations, do_decay=False)
        observations = next_observations

        observations, terms, truns, reward_mask = env.reset_where_done(observations, terms, truns)

        if i >= FLAGS.start_training:
            batches = replay_buffer.sample_parallel_multibatch(FLAGS.batch_size, FLAGS.updates_per_step)
            infos = agent.update(batches, num_updates=FLAGS.updates_per_step)
            update_count += FLAGS.updates_per_step
            if i % FLAGS.log_interval == 0:
                log_multiple_seeds_to_wandb(i, infos)

        evaluate_if_time_to(i, agent, eval_env, eval_returns, infos, list(range(FLAGS.seed, FLAGS.seed+FLAGS.num_seeds)))


if __name__ == '__main__':
    app.run(main)
