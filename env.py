from sys import platform

import gym

gym.logger.set_level(40)

from mlagents.envs import UnityEnvironment
import numpy as np


class GymEnv:

    def __init__(self, env_name, seed=None) -> None:
        super().__init__()
        self.name = env_name
        self.gym_env = gym.make(env_name)
        self.gym_env.seed(seed)
        self.num_states = self.gym_env.observation_space.shape[0]
        action_space = self.gym_env.action_space
        if type(action_space).__name__ == 'Box':
            self.num_actions = action_space.shape[0]
        else:
            self.num_actions = self.gym_env.action_space.n
        self.num_agents = 1

    def seed(self, seed):
        self.gym_env.seed(seed)

    def reset(self, **kwargs):
        return np.asarray([self.gym_env.reset()])

    def step(self, actions):
        next_state, reward, done, _ = self.gym_env.step(actions[0])

        return np.asarray([next_state]), np.asarray([reward]), np.asarray([done]), None


def unity_filename(env_name):
    if platform == "linux" or platform == "linux2":
        env_filename = 'envs/%s_Linux_NoVis/%s.x86_64' % (env_name, env_name)
    elif platform == "darwin":
        env_filename = 'envs/%s.app' % env_name
    elif platform == "win32":
        env_filename = 'envs/%s_Windows_x86_64/%s.exe' % (env_name, env_name)

    return env_filename


class UnityEnv:

    def __init__(self, env_name, **kwargs) -> None:
        super().__init__()
        filename = unity_filename(env_name)
        self.unity_env = UnityEnvironment(file_name=filename, **kwargs)
        brain_name = self.unity_env.brain_names[0]
        self.name = brain_name.replace("Brain", "")
        brain = self.unity_env.brains[brain_name]

        env_info = self.unity_env.reset(train_mode=True)[brain_name]

        self.brain_name = brain_name
        self.num_agents = len(env_info.agents)
        self.num_actions = list(brain.vector_action_space_size)[0]
        self.states = env_info.vector_observations
        self.num_states = self.states.shape[1]

    def reset(self, train_mode=False):
        env_info = self.unity_env.reset(train_mode=train_mode)[self.brain_name]
        return env_info.vector_observations

    def step(self, actions):
        env_info = self.unity_env.step(actions)
        env_info = env_info[self.brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done

        return np.asarray(next_states), np.asarray(rewards), np.asarray(dones), env_info
