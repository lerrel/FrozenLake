import gym
from gym import ObservationWrapper
import numpy as np
from IPython import embed

class ContinuousWrapper(ObservationWrapper):

    def observation(self, observation):
        ## For the first time observation is called
        if type(self.observation_space) == gym.spaces.Discrete:
            self.disc_n = self.env.observation_space.n
            self.observation_space = gym.spaces.Box(np.zeros(self.disc_n), np.ones(self.disc_n), dtype=np.float32)
            self.is_discrete = True

        if self.is_discrete is not None:
            obs = np.zeros(self.disc_n)
            obs[observation] = 1.0
            return obs
        else:
            return observation
