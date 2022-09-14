import argparse
import pathlib

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

from reward_shaping.training.utils import make_env
from utils.utils import parse_env_task, parse_reward


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
