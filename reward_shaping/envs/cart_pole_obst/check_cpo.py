import pathlib

import matplotlib.pyplot as plt
import yaml
from stable_baselines3.common.env_checker import check_env

from reward_shaping.training.utils import make_env, make_reward_wrap


def main(reward):
    env_name = "cart_pole_obst"
    task = "fixed_height"
    env, env_params = make_env(env_name, task, logdir=None, seed=0, prob_sampling_feasible=1.0)
    env = make_reward_wrap(env_name, env, env_params, reward)

    # evaluation
    obs = env.reset()
    env.render()
    rewards = []
    tot_reward = 0.0
    for i in range(10000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(reward)
        tot_reward += reward
        env.render()
        if done:
            rewards.append(tot_reward)
            obs = env.reset()
            print(f"reward: {tot_reward:.3f}")
            tot_reward = 0.0
            input()
    try:
        check_env(env)
        result = True
    except Exception as err:
        result = False
        print(err)
    print(f"Check env: {result}")

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", type=str, default="sparse")
    args = parser.parse_args()

    main(args.reward)
