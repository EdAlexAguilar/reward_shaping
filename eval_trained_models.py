import argparse
import json
import pathlib
import warnings
from typing import Dict, Tuple, List
import re

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

from reward_shaping.training.utils import make_env


def parse_env_task(filepath: str):
    env, task = None, None
    for env_name in ["cart_pole_obst", "bipedal_walker", "lunar_lander", "racecar"]:
        if env_name in filepath:
            env = env_name
            break
    for task_name in ["fixed_height", "forward", "hardcore", "land", "drive"]:
        if task_name in filepath:
            task = task_name
            break
    if not env or not task:
        raise ValueError(f"not able to parse env/task in {filepath}")
    return env, task


def parse_reward(filepath: str):
    for reward in ["default", "tltl", "bhnr", "morl_uni", "morl_dec", "hprs"]:
        if reward in filepath:
            return reward
    raise ValueError(f"reward not found in {filepath}")


def main(args):
    checkpoint = args.checkpoint
    render = not args.no_render
    # create eval env
    env_name, task_name = parse_env_task(str(checkpoint))
    reward_name = parse_reward(str(checkpoint))
    # run evaluation
    print(f"[evaluation] env: {env_name}, task: {task_name}, reward: {reward_name}")
    env, env_params = make_env(env_name, task_name, 'eval', eval=True, logdir=None, seed=0)
    model = SAC.load(str(checkpoint))
    rewards, eplens = evaluate_policy(model, env,
                                      n_eval_episodes=args.n_episodes,
                                      deterministic=True,
                                      render=render,
                                      return_episode_rewards=True)
    env.close()
    print(
        f"[results] nr episodes: {len(rewards)}, mean reward: {np.mean(rewards):.5f}, mean lengths: {np.mean(eplens):.5f}")


if __name__ == "__main__":
    import time

    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="model checkpoint to evaluate")
    parser.add_argument("--n_episodes", type=int, default=1, help="nr evaluation episodes")
    parser.add_argument("-no_render", action="store_true", help="disable rendering")
    args = parser.parse_args()
    main(args)
    tf = time.time()
    print(f"[done] elapsed time: {tf - t0:.2f} seconds")
