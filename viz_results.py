'''
Visualize the test curves with std.
'''
import matplotlib.pyplot as plt
import os
import argparse
import cloudpickle as pkl
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, help="Path to experiments")
    args = parser.parse_args()
    exp_name = args.exp_name
    fig_name = os.path.join(exp_name, 'test_summary.png')

    all_contents = [os.path.join(exp_name, t) for t in os.listdir(exp_name)]
    all_dirs = []
    for c in all_contents:
        if os.path.isdir(c):
            all_dirs.append(c)

    test_exp_summary = []
    for ad in all_dirs:
        seed_data = pkl.load(open(os.path.join(ad, 'data.pkl'), 'rb'))
        test_exp_summary.append([np.mean(tr) for tr in seed_data['test_rewards']])
    test_exp_mean = np.mean(test_exp_summary,0)
    test_exp_std = np.std(test_exp_summary,0)
    x = [i for i in range(len(test_exp_mean))]
    plt.plot(x, test_exp_mean, color=(0.5,0.1,0.1), linewidth=2.0)
    plt.fill_between(x, test_exp_mean-test_exp_std, test_exp_mean+test_exp_std,color=(0.5,0.1,0.1), alpha=0.5)
    plt.ylim([0.0,1.0])
    plt.savefig(fig_name)

if __name__ == '__main__':
    main()
