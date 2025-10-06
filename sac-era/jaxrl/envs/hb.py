import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RecordEpisodeStatistics as EpisodeMonitor
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ["PYOPENGL_PLATFORM"] = "egl"

class HBGymnasiumVersionWrapper(gym.Wrapper):
    """
    humanoid bench originally requires gymnasium==0.29.1
    however, we are currently using gymnasium==1.0.0a2,
    hence requiring some minor fix to the rendering function
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.task = env.unwrapped.task

    def render(self):
        return self.task._env.mujoco_renderer.render(self.task._env.render_mode)


class ActionRepeatWrapper:
    def __init__(self, env: gym.Env, action_repeat: int = 2):
        self.action_repeat = action_repeat
        self.env = env
    
    def step(self, action: np.ndarray):
        total_reward = 0.0
        for _ in range(self.action_repeat):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            if term or trunc:
                break
        return obs, total_reward, term, trunc, info
    
    def __getattr__(self, name):
        return getattr(self.env, name)


class NoTerminationWrapper:
    def __init__(self, env: gym.Env):
        self.env = env
    
    def step(self, action: np.ndarray):
        obs, reward, term, trunc, info = self.env.step(action)
        return obs, reward, False, trunc, info
    
    def __getattr__(self, name):
        return getattr(self.env, name)


def make_humanoid_env(env_name: str, seed: int, monitor_episode: bool = True) -> gym.Env:
    import humanoid_bench
    additional_kwargs = {"render_mode": "rgb_array"}
    if env_name == "h1hand-package-v0":
        additional_kwargs = {"policy_path": None}
    env = gym.make(env_name, **additional_kwargs)
    env = HBGymnasiumVersionWrapper(env)
    # We use action repeat of 2 + No Termination
    env = NoTerminationWrapper(env)
    env = ActionRepeatWrapper(env, action_repeat=2)
    if monitor_episode:
        env = EpisodeMonitor(env)
    env.reset(seed=seed)
    return env


class make_env_gym_hb(gym.Env):
    def __init__(self, env_name='h1humanoid-v0', seed=0, num_envs=2):
        np.random.seed(seed)
        seeds = np.random.randint(0, 1e6, (num_envs))
        self.envs = [gym.wrappers.RescaleAction(
                         make_humanoid_env(env_name, int(s), monitor_episode=False),
                         -1.0, 1.0
                     ) for s in seeds]
        self.num_envs = len(self.envs)
        self.timesteps = np.zeros(self.num_envs)
        self.action_space = spaces.Box(
            low=self.envs[0].action_space.low[None].repeat(len(self.envs), axis=0),
            high=self.envs[0].action_space.high[None].repeat(len(self.envs), axis=0),
            shape=(len(self.envs), self.envs[0].action_space.shape[0]),
            dtype=self.envs[0].action_space.dtype,
        )
        self.observation_space = spaces.Box(
            low=self.envs[0].observation_space.low[None].repeat(len(self.envs), axis=0),
            high=self.envs[0].observation_space.high[None].repeat(len(self.envs), axis=0),
            shape=(len(self.envs), self.envs[0].observation_space.shape[0]),
            dtype=self.envs[0].observation_space.dtype,
        )
        self.action_dim = self.envs[0].action_space.shape[0]

    def _reset_idx(self, idx):
        seed_ = np.random.randint(0, 1e6)
        obs, _ = self.envs[idx].reset(seed=seed_)
        return obs

    def reset_where_done(self, observations, terms, truns):
        resets = np.zeros((terms.shape))
        for j, (term, trun) in enumerate(zip(terms, truns)):
            if term or trun:
                observations[j], terms[j], truns[j] = self._reset_idx(j), False, False
                resets[j] = 1
                self.timesteps[j] = 0
        return observations, terms, truns, resets

    def reset(self):
        obs = []
        for env in self.envs:
            seed_ = np.random.randint(0, 1e6)
            ob, _ = env.reset(seed=seed_)
            obs.append(ob)
        return np.stack(obs)

    def generate_masks(self, terms, truns):
        masks = []
        for term, trun in zip(terms, truns):
            if not term or trun:
                mask = 1.0
            else:
                mask = 0.0
            masks.append(mask)
        return np.array(masks)

    def step(self, actions):
        obs, rews, terms, truns = [], [], [], []
        for env, action in zip(self.envs, actions):
            ob, reward, term, trun, _ = env.step(action)
            obs.append(ob)
            rews.append(reward)
            terms.append(term)
            truns.append(trun)
        self.timesteps += 1
        return np.stack(obs), np.stack(rews), np.stack(terms), np.stack(truns)

    def random_step(self):
        actions = np.random.uniform(-1, 1, (self.num_envs, self.action_dim))
        obs, rews, terms, truns = self.step(actions)
        return obs, rews, terms, truns, actions
    
    def render(self, mode='human')-> np.ndarray:
        """
        Render all the environments
        output shape: (num_envs, H, W, 3)
        """
        frames = []
        for env in self.envs:
            frames.append(env.render())
        return np.stack(frames)

    def evaluate(self, agent, num_episodes=5, temperature=0.0):
        returns_mean = np.zeros(self.num_envs)
        returns = np.zeros(self.num_envs)
        episode_count = np.zeros(self.num_envs)
        state = self.reset()
        while episode_count.min() < num_episodes:
            actions = agent.sample_actions(state, temperature=temperature)
            new_state, reward, term, trun = self.step(actions)
            returns += reward
            state = new_state
            state, term, trun, reset_mask = self.reset_where_done(state, term, trun)
            episode_count += reset_mask
            returns_mean += reset_mask * returns
            returns *= (1 - reset_mask)
        result = {"return": returns_mean / episode_count}
        return result