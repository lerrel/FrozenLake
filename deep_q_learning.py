'''
Frozen version of DQN for FrozenLake to work
'''
import gym

from baselines import deepq
from continuous_env import ContinuousWrapper
from IPython import embed


def main():
    env_name = 'FrozenLake-v0'
    model_name = '{}_model.pkl'.format(env_name)
    env = gym.make(env_name)
    env = ContinuousWrapper(env)
    env.reset();
    #embed()
    # Enabling layer_norm here is import for parameter space noise!
    model = deepq.models.mlp([64])
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        print_freq=10,
    )
    print("Saving model to {}".format(model_name))
    act.save("models/{}".format(model_name))

if __name__ == '__main__':
    main()
