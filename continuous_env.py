import gym
from gym import ObservationWrapper
import numpy as np
from IPython import embed

class ChoiceDist(object):

    def __init__(self, choices, p=None):
        self.choices = choices
        self.p = p
        if self.p == None:
            self.p = 1./len(choices)*np.ones(len(choices))

    def sample(self):
        val = np.random.choice(self.choices, p=self.p)
        if type(val) in [int, np.int, np.int0, np.int8, np.int16, np.int32, np.int64]:
            pass
        else:
            val = val.sample()
        return val


class FixedResetDist(object):

    def __init__(self, val=0):
        self.fixed_val = val

    def sample(self):
        return self.fixed_val

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

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        if self.reset_dist is None:
            self.set_reset_dist(FixedResetDist(0))
        state = self.reset_dist.sample()
        if self.env.observation_space.contains(state) == True:
            self.env.env.s = state
        elif state is not None:
            raise ValueError
        else:
            pass
        return self.observation(self.env.env.s)

    def set_reset_dist(self, reset_dist):
        self.reset_dist = reset_dist
