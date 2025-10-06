import os
os.environ['MUJOCO_GL'] = 'egl'
import gymnasium as gym
import numpy as np
from typing import Dict, Optional, OrderedDict
from dm_control import suite
from dm_env import specs
from gym import spaces


'''
# Support following api:
- env.reset() -> np.ndarray (obs)
- env.action_space:
    - gym.spaces.Box
- env.observation_space:
    - gym.spaces.Box
- env.step(action: np.ndarray) -> next_observations, rewards, terms, truns, info
- env.generate_masks(terms, truns) -> masks
- env.reset_where_done(observations, terms, truns) -> observations, terms, truns, None
'''

# dm_control environment wrapper
def _dmc_spec2gym_space(spec):
    if isinstance(spec, OrderedDict):
        total_dim = int(sum(np.prod(s.shape) for s in spec.values()))
        return spaces.Box(low=-float('inf'), high=float('inf'), shape=(total_dim,), dtype=np.float64)
    elif isinstance(spec, specs.BoundedArray):
        return spaces.Box(low=spec.minimum, high=spec.maximum, shape=spec.shape, dtype=spec.dtype)
    elif isinstance(spec, specs.Array):
        return spaces.Box(low=-float('inf'), high=float('inf'), shape=spec.shape, dtype=spec.dtype)
    else:
        raise NotImplementedError
    
def _obs2state(obs):
    return np.concatenate([ v.flatten() for k,v in obs.items()])

class dmc_wrapper:
    def __init__(self, domain_name: str, task_name: str, task_kwargs: Optional[Dict] = {}, environment_kwargs=None):
        self.env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs=task_kwargs, environment_kwargs=environment_kwargs)

        # get action space and observation space
        self.action_space = _dmc_spec2gym_space(self.env.action_spec())
        self.observation_space = _dmc_spec2gym_space(self.env.observation_spec())

        self._max_episode_steps = 1000
        self._step = 0

    def timestep2ret(self, timestep):
        next_state = _obs2state(timestep.observation)
        reward = timestep.reward
        done = timestep.last()
        term = (self._step == self._max_episode_steps)
        info = {}
        return next_state, reward, done, term, info
    
    def reset(self):
        self._step = 0
        timestep = self.env.reset()
        return _obs2state(timestep.observation)
    
    def step(self, action):
        self._step += 1
        timestep = self.env.step(action)
        return self.timestep2ret(timestep)
    
    def render(self):
        return self.env.physics.render(height=256, width=256, camera_id=0)

class RescaleAction:
    def __init__ (self, env: dmc_wrapper):
        self.env = env
        self.low = self.env.action_space.low
        self.high = self.env.action_space.high
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=self.env.action_space.shape, dtype=np.float64)
    
    def step(self, action):
        action = self.low + (action + 1.0) * 0.5 * (self.high - self.low)
        return self.env.step(action)

    # other methods
    def __getattr__(self, name):
        return getattr(self.env, name)

def _make_env_dmc(env_name: str, seed: int):
    domain_name, task_name = env_name.split('-')
    return RescaleAction(dmc_wrapper(domain_name, task_name, task_kwargs={"random": seed}))

class make_env_dmc:
    def __init__(self, env_name: str, seed: int, num_envs: int, max_t=1000):
        env_fns = [lambda i=i: _make_env_dmc(env_name, seed + i) for i in range(num_envs)]
        self.envs = [env_fn() for env_fn in env_fns]
        self.max_t = max_t
        self.num_seeds = len(self.envs)
        self.action_space = spaces.Box(low=self.envs[0].action_space.low[None].repeat(len(self.envs), axis=0),
                                       high=self.envs[0].action_space.high[None].repeat(len(self.envs), axis=0),
                                       shape=(len(self.envs), self.envs[0].action_space.shape[0]),
                                       dtype=self.envs[0].action_space.dtype)
        self.observation_space = spaces.Box(low=self.envs[0].observation_space.low[None].repeat(len(self.envs), axis=0),
                                            high=self.envs[0].observation_space.high[None].repeat(len(self.envs), axis=0),
                                            shape=(len(self.envs), self.envs[0].observation_space.shape[0]),
                                            dtype=self.envs[0].observation_space.dtype)
        
    def _reset_idx(self, idx):
        return self.envs[idx].reset()
    
    def reset_where_done(self, observations, terms, truns):
        resets = np.zeros((terms.shape))
        for j, (term, trun) in enumerate(zip(terms, truns)):
            if (term == True) or (trun == True):
                observations[j], terms[j], truns[j] = self._reset_idx(j), False, False
                resets[j] = 1
        return observations, terms, truns, resets

    def generate_masks(self, terms, truns):
        masks = []
        for term, trun in zip(terms, truns):
            if not term or trun:
                mask = 1.0
            else:
                mask = 0.0
            masks.append(mask)
        masks = np.array(masks)
        return masks

    def reset(self):
        obs = []
        for env in self.envs:
            obs.append(env.reset())
        return np.stack(obs)
    
    def render(self):
        """
        Render all the environments
        output shape: (num_seeds, H, W, 3)
        """
        frames = []
        for env in self.envs:
            frames.append(env.render())
        return np.stack(frames)

    def step(self, actions):
        obs, rews, terms, truns = [], [], [], []
        for env, action in zip(self.envs, actions):
            ob, reward, done, trun, info = env.step(action)
            obs.append(ob)
            rews.append(reward)
            terms.append(done)
            truns.append(trun)
        return np.stack(obs), np.stack(rews), np.stack(terms), np.stack(truns), None

    def evaluate(self, agent, num_episodes=5, temperature=0.0):
        returns_mean = np.zeros(self.num_seeds)
        returns = np.zeros(self.num_seeds)
        episode_count = np.zeros(self.num_seeds)
        state = self.reset()

        while episode_count.min() < num_episodes:
            actions = agent.sample_actions(state, temperature=temperature)
            new_state, reward, term, trun, _ = self.step(actions)

            returns += reward
            state = new_state
            state, term, trun, reset_mask = self.reset_where_done(state, term, trun)
            episode_count += reset_mask
            returns_mean += reset_mask * returns
            returns *= (1 - reset_mask)

        result = {"return": returns_mean / episode_count}
        return result