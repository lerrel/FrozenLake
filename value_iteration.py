# Taken from https://github.com/wagonhelm/Value-Iteration/blob/master/FrozenIce.ipynb
import numpy as np
import gym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import cloudpickle as pkl
from IPython import embed

parser = argparse.ArgumentParser()
parser.add_argument("--env", default='FrozenLake-v0', type=str, help="Name of environment")
args = parser.parse_args()
env_name = args.env
traj_name = '{}.pkl'.format(env_name)
print('Running Value Iteration on {}'.format(env_name))
env = gym.make(env_name)

#env = gym.make('FrozenLake8x8-v0')
#env = gym.make('FrozenLake-v0')
#env = gym.make('FrozenLakeDet-v0')
#env = gym.make('FrozenLake8x8Det-v0')

render_viz = True
gamma = 0.999
test_episodes = 1

class mdp_policy(object):
    def __init__(self, V, P, nA, gamma):
        self.V = V
        self.P = P
        self.nA = nA
        self.gamma = gamma

    def act(self, state):
        action = np.argmax([sum([p*(r + self.gamma*self.V[s_]) for p, s_, r, _ in self.P[state][a]]) for a in range(self.nA)])
        return action

n_states = env.observation_space.n
n_actions = env.action_space.n

values = np.zeros(n_states)


print('Performing Value Iteration')
while True:
    delta = 0
    for states in reversed(range(n_states)):
        v = values[states]
        values[states] = np.max([sum([p*(r + gamma*values[s_]) for p, s_, r, _ in env.env.P[states][a]]) for a in range(env.env.nA)])
        delta = max(delta,abs(v-values[states]))
    # print("Delta: {}".format(delta))
    if delta < 1e-30:
        break

Policy = mdp_policy(values, env.env.P, env.env.nA, gamma)
print('Testing Value Iteration')
history = []
bestAverage = []
state = env.reset()z
if render_viz==True:
    env.render()

trajs = []
for i in range(test_episodes):
    traj =[]
    while True:
        action = Policy.act(state)
        new_state, reward, done, info = env.step(action)
        traj.append([state, action, new_state, reward, done, info])
        state = new_state
        if render_viz==True:
            print(action)
            env.render()
        if done:
            history.append(reward)
            env.reset()
            break
    trajs.append(traj)
pkl.dump( trajs, open( 'trajs/{}'.format(traj_name), "wb" ))