import pathlib

import yaml
from stable_baselines3.common.env_checker import check_env

from envs.cart_pole.rewards import get_reward
from hierarchy.graph_hierarchical_reward import HierarchicalGraphRewardWrapper
from utils import make_env


def main(reward):
    env = "cart_pole"
    task = "target"
    env, _ = make_env(env, task, logdir=None, eval=True)
    env = get_reward(reward)(env)
    if isinstance(env, HierarchicalGraphRewardWrapper):
        env.hierarchy.render()

    # evaluation
    obs = env.reset()
    env.render()
    rewards = []
    tot_reward = 0.0
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        tot_reward += reward
        env.render()
        if done:
            rewards.append(tot_reward)
            obs = env.reset()
            rob = env.compute_episode_robustness(env.last_complete_episode, env.last_bool_spec)
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", type=str, default="sparse")
    args = parser.parse_args()

    main(args.reward)
