import argparse

import numpy as np
from stable_baselines3.common.env_checker import check_env

from reward_shaping.training.utils import make_env, make_agent


def generic_env_test(env_name, task, reward_name, potential=False):
    env, env_params = make_env(env_name, task, reward_name, use_potential=potential, eval=True, logdir=None, seed=0)
    # check
    check_env(env)
    # evaluation
    _ = env.reset()
    env.render()
    tot_reward = 0.0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(reward)
        tot_reward += reward
        env.render()
    print(f"[{reward_name}] tot reward: {tot_reward:.3f}")
    env.close()
    return True


def generic_training(env, task, reward):
    # create training environment
    train_env, env_params = make_env(env, task, reward, seed=0)
    # create agent
    model = make_agent(env, train_env, "sac", logdir=None)
    # train
    model.learn(total_timesteps=500)
    train_env.close()