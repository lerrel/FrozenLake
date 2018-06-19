'''
DQN for FrozenLake.
Performance benchmarks:
    FrozenLake-v0 - 0.74
    FrozenLakeDet-v0 - 0.97
    FrozenLake8x8-v0 - 0.66
    FrozenLake8x8Det-v0 - 0.97
'''
import gym

from baselines import deepq
from continuous_env import ContinuousWrapper, FixedResetDist
from IPython import embed


def main():
    env_name = 'FrozenLake8x8Det-v0'
    model_name = '{}_model.pkl'.format(env_name)
    env = gym.make(env_name)
    env = ContinuousWrapper(env)
    #env.set_reset_dist(FixedResetDist(62))
    env.reset();
    #embed()
    # Enabling layer_norm here is import for parameter space noise!
    model = deepq.models.mlp([])
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.0,
        print_freq=10,
    )
    print("Saving model to {}".format(model_name))
    act.save("models/{}".format(model_name))

if __name__ == '__main__':
    main()
