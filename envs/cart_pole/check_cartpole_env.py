import pathlib

import gym
import yaml
from stable_baselines3.common.env_checker import check_env

import callbacks
from stable_baselines3.common.callbacks import CheckpointCallback
import argparse as parser

# define problem
from envs.cart_pole.cp_continuousobstacle_env import CartPoleContObsEnv
from envs.cart_pole.rewards.baselines import SparseNoFalldownReward


def main(reward):
    task = "no_random"
    env_config = pathlib.Path(f"tasks/{task}.yml")
    with open(env_config, 'r') as file:
        env_params = yaml.load(file, yaml.FullLoader)
    env = CartPoleContObsEnv(**env_params)
    if reward == "indicator":
        from envs.cart_pole.rewards.indicator_based import IndicatorWithContinuousTargetReward
        env = IndicatorWithContinuousTargetReward(env)
    elif reward == "indicator_sparse":
        from envs.cart_pole.rewards.indicator_based import IndicatorWithSparseTargetReward
        env = IndicatorWithSparseTargetReward(env)
    elif reward == "indicator_progress":
        from envs.cart_pole.rewards.indicator_based import IndicatorWithProgressTargetReward
        env = IndicatorWithProgressTargetReward(env)
    elif reward == "weighted":
        from envs.cart_pole.rewards.baselines import WeightedReward
        env = WeightedReward(env)
    elif reward == "sparse":
        from envs.cart_pole.rewards.baselines import SparseReward
        env = SparseReward(env)
    elif reward == "sparse_nofall":
        from envs.cart_pole.rewards.baselines import SparseReward
        env = SparseNoFalldownReward(env)
    elif reward == "sparse_stl":
        from envs.cart_pole.rewards.baselines import SparseSTLReward
        env = SparseSTLReward(env)


    # evaluation
    obs = env.reset()
    env.render()
    rewards = []
    tot_reward = 0.0
    for i in range(200):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        tot_reward += reward
        env.render()
        if done:
            rewards.append(tot_reward)
            obs = env.reset()
            rob = env.compute_episode_robustness(env.last_complete_episode)
            print(f"reward: {tot_reward:.3f}, robustness: {rob:.3f}")
            tot_reward = 0.0
    try:
        check_env(env)
        result = True
    except Exception as err:
        result = False
        print(err)
    print(f"Check env: {result}")

    env.close()


if __name__ == "__main__":
    rewards = ['indicator', 'indicator_sparse', 'indicator_progress', 'weighted', 'sparse', 'sparse_nofall', 'sparse_stl']
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", choices=rewards, default="indicator")
    args = parser.parse_args()

    main(args.reward)
