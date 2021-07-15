import pathlib

import yaml
from stable_baselines3.common.env_checker import check_env

# define problem
from envs.cart_pole_obst.cp_continuousobstacle_env import CartPoleContObsEnv
from envs.cart_pole_obst.rewards import get_reward
import numpy as np


def main(reward):
    task = "fixed_height"
    env_config = pathlib.Path(f"tasks/{task}.yml")
    with open(env_config, 'r') as file:
        env_params = yaml.load(file, yaml.FullLoader)
    env = CartPoleContObsEnv(**env_params, eval=True, seed=0)
    env = get_reward(reward)(env)
    """
    if reward == "indicator":
        from envs.cart_pole_obst.rewards import IndicatorWithContinuousTargetReward
        env = IndicatorWithContinuousTargetReward(env)
    elif reward == "indicator_sparse":
        from envs.cart_pole_obst.rewards import IndicatorWithSparseTargetReward
        env = IndicatorWithSparseTargetReward(env)
    elif reward == "indicator_progress":
        from envs.cart_pole_obst.rewards import IndicatorWithProgressTargetReward
        env = IndicatorWithProgressTargetReward(env)
    """

    # evaluation
    obs = env.reset()
    print(obs[0])
    env.render()
    rewards = []
    tot_reward = 0.0
    for i in range(10000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        tot_reward += reward
        env.render()
        #env.render_hierarchy()
        if done:
            rewards.append(tot_reward)
            obs = env.reset()
            rob = env.compute_episode_robustness(env.last_complete_episode)
            print(f"reward: {tot_reward:.3f}, robustness: {rob:.3f}")
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
    rewards = ['indicator', 'indicator_sparse', 'indicator_progress', 'continuous', 'sparse', 'stl', 'cont_gh',
               'cont_gh_pot']
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", choices=rewards, default="indicator")
    args = parser.parse_args()

    main(args.reward)
