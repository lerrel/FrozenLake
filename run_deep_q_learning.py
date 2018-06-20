"""
Running DQN experiments for parallel execs and std graphs.
Credit: https://gist.github.com/davidgardenier/c4008e70eea93e548533fd9f93a5a330
"""

import argparse
import subprocess
from IPython import embed
import os

template = 'python -m deep_q_learning --env {} --plan {} --alpha {} --n_steps {} --exp_name {} --seed {}'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default='FrozenLake-v0', type=str, help="Name of environment")
    parser.add_argument("--plan", default=0, type=int, help="sample resets from the planner")
    parser.add_argument("--n_exps", default=5, type=int, help="number of experiments to get mean and std")
    parser.add_argument("--n_steps", default=20000, type=int, help="number of steps for running DQN")
    parser.add_argument("--alpha", default=0.5, type=float, help="reset from planned steps prob")
    
    args = parser.parse_args()
    env_name = args.env
    isplan = args.plan
    n_exps = args.n_exps
    n_steps = args.n_steps
    alpha = args.alpha
    assert alpha>=0 and alpha<=1, 'alpha needs to be a probability, i.e in [0,1]'
    if isplan == False:
        alpha = 1.0

    processes = []

    for seed in range(n_exps):
        exp_name = os.path.join('exps', env_name, 'plan' if isplan else 'noplan', str(alpha), str(seed))
        if not os.path.exists(exp_name):
            os.makedirs(exp_name)
        command = template.format(*[env_name, isplan, alpha, n_steps, exp_name, seed])
        print('Running seed #{} command: {}'.format(seed,command))
        process = subprocess.Popen(command, shell=True)
        processes.append(process)
    
    output = [p.wait() for p in processes]
    print('Completed All experiments')

if __name__ == '__main__':
    main()