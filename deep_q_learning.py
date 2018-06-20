'''
DQN for FrozenLake.
Performance benchmarks:
    FrozenLake-v0 - 0.74
    FrozenLakeDet-v0 - 0.97
    FrozenLake8x8-v0 - 0.66
    FrozenLake8x8Det-v0 - 0.97
'''

import os
import gym
import numpy as np
from baselines import deepq
from continuous_env import ContinuousWrapper, FixedResetDist, ChoiceDist
from IPython import embed
import matplotlib.pyplot as plt
import argparse
import cloudpickle as pkl

TRAJ_DIR = '/home/lerrel/mbrl/FrozenLake/trajs'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default='FrozenLake-v0', type=str, help="Name of environment")
    parser.add_argument("--plan", default=0, type=int, help="sample resets from the planner")
    parser.add_argument("--n_steps", default=20000, type=int, help="number of steps for running DQN")
    parser.add_argument("--alpha", default=0.5, type=float, help="reset from planned steps prob")
    parser.add_argument("--exp_name", default=None, type=str, help="Name of the folder to be stored under")
    parser.add_argument("--seed", default=0, type=int, help="seed for the experiment")
    
    args = parser.parse_args()
    env_name = args.env
    isplan = args.plan
    ## Need to figure out how to actually set the seed
    seed = args.seed
    n_steps = args.n_steps
    alpha = args.alpha
    assert alpha>=0 and alpha<=1, 'alpha needs to be a probability, i.e in [0,1]'
    if isplan == False:
        alpha = 1.0
    exp_name = args.exp_name
    if exp_name is None:
        exp_name = os.path.join('exps', env_name, 'plan' if isplan else 'noplan', str(alpha), str(seed))
    if not os.path.exists(exp_name):
        os.makedirs(exp_name)
    model_name = os.path.join(exp_name, 'model.pkl')
    fig_name = os.path.join(exp_name, 'test_curve.png')
    data_name = os.path.join(exp_name, 'data.pkl')
    plan_name = os.path.join(TRAJ_DIR, env_name) + '.pkl'
    
    test_env = gym.make(env_name)
    test_env = ContinuousWrapper(test_env)
    test_env.reset();

    if isplan == True:
        plan_trajs = pkl.load(open(plan_name, 'rb'))
        all_trajs = np.concatenate(plan_trajs)
        all_states = all_trajs[:, 0]
        InitResetDist = FixedResetDist(0)
        StateResetDist = ChoiceDist(all_states)
        ResetDist = ChoiceDist([InitResetDist, StateResetDist], p = [alpha, 1-alpha])
    else:
        ResetDist = FixedResetDist(0)
    env = gym.make(env_name)
    env = ContinuousWrapper(env)
    env.set_reset_dist(ResetDist)
    env.reset()
    
    # Enabling layer_norm here is import for parameter space noise!
    model = deepq.models.mlp([])
    act, train_rews, test_rews = deepq.learn(
        env,
        test_env,
        q_func=model,
        lr=1e-3,
        test_freq=200,
        n_tests=10,
        max_timesteps=n_steps,
        buffer_size=50000,
        exploration_fraction=0.3,
        exploration_final_eps=0.0,
        checkpoint_freq=None,
        print_freq=10,
    )
    data = {
        'test_rewards':test_rews,
        'train_rewards':train_rews,
        'args':args
    }
    test_curve = [np.mean(tr) for tr in data['test_rewards']]
    plt.plot(test_curve, color=(0.5,0.1,0.1), linewidth=2.0)
    plt.ylim([0.0,1.0])
    plt.savefig(fig_name)
    pkl.dump(data, open(data_name, 'wb'))
    act.save(model_name)

if __name__ == '__main__':
    main()
