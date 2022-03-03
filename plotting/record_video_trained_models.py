import argparse
import json
import pathlib
import warnings
from typing import Dict, Tuple, List
import re

import numpy as np
import yaml
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from stable_baselines3 import SAC

from plotting.utils import get_files, parse_reward, parse_env_task
from reward_shaping.training.custom_evaluation import evaluate_policy_with_monitors
from reward_shaping.training.utils import make_env

file_regex = "*model*steps.zip"


def plot_file_info(args):
    for regex in args.regex:
        # filter files
        filter = lambda f: "skip" not in str(f) and args.min_steps <= int(
            re.findall(r'\d+', f.stem)[-1]) <= args.max_steps
        files = [f for f in get_files(args.logdir, regex, fileregex=file_regex) if filter(f)]
        envs_task_rew_files = group_checkpoints_per_env_task_reward(files)
        # print statistics
        print(f"regex: {regex}, nr files: {len(files)}")
        for env in envs_task_rew_files:
            for task in envs_task_rew_files[env]:
                for rew in envs_task_rew_files[env][task]:
                    n_files = len(envs_task_rew_files[env][task][rew])
                    print(f"\tenv: {env}, task: {task}, reward {rew}: {n_files} models")


def group_checkpoints_per_env_task_reward(checkpoints: List[pathlib.Path]):
    envs_task_rew_files = {}
    for f in checkpoints:
        # parse env
        try:
            env, task = parse_env_task(str(f))
            rew = parse_reward(str(f))
        except Exception:
            warnings.warn(f"not able to parse logdir {f}")
            continue
        # fill the nested dictionary
        if env in envs_task_rew_files:
            if task in envs_task_rew_files[env]:
                if rew in envs_task_rew_files[env][task]:
                    envs_task_rew_files[env][task][rew].append(f)
                else:
                    envs_task_rew_files[env][task][rew] = [f]
            else:
                envs_task_rew_files[env][task] = {}
                envs_task_rew_files[env][task][rew] = [f]
        else:
            envs_task_rew_files[env] = {}
            envs_task_rew_files[env][task] = {}
            envs_task_rew_files[env][task][rew] = [f]
    return envs_task_rew_files


def record_rollout(model, env, outfile, deterministic=True, render=False) -> List[np.ndarray]:
    recorder = VideoRecorder(env, base_path=outfile)
    obs = env.reset()
    done = False
    steps = 0
    rtg = 0
    while not done:
        steps += 1
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)
        rtg += reward
        recorder.capture_frame()
        if render:
            env.render()
    return steps, rtg


def main(args):
    # print debug info
    if args.info:
        plot_file_info(args)
        return
    # def out dir for storing videos
    outdir = args.outdir / f"{int(time.time())}"
    outdir.mkdir(parents=True, exist_ok=True)

    # def function to filter files
    def my_filter(f: pathlib.Path, min_steps, max_steps) -> bool:
        if "skip" in str(f):
            return False
        return min_steps <= int(re.findall(r'\d+', f.stem)[-1]) <= max_steps

    # collect checkpoints
    for regex in args.regex:
        min_steps, max_steps = args.min_steps, args.max_steps
        files = [f for f in get_files(args.logdir, regex, fileregex=file_regex) if my_filter(f, min_steps, max_steps)]
        print(f"regex: {regex}, nr files: {len(files)}")
        timeid = int(time.time())
        envs_task_rew_files = group_checkpoints_per_env_task_reward(files)
        #
        for env_name in envs_task_rew_files:
            for task_name in envs_task_rew_files[env_name]:
                for reward_name in envs_task_rew_files[env_name][task_name]:
                    print(f"[simulation] env: {env_name}, task: {task_name}, reward: {reward_name}")
                    env, env_params = make_env(env_name, task_name, 'eval', eval=True, logdir=None, seed=0)
                    for i, cpfile in enumerate(envs_task_rew_files[env_name][task_name][reward_name]):
                        model = SAC.load(str(cpfile))
                        ep_lengths, rewards = [], []
                        for ep in range(args.n_episodes):
                            outfile = str(outdir / f'{env_name}_{task_name}_{reward_name}_{timeid}_cp{i}_ep{ep + 1}')
                            steps, reward = record_rollout(model, env, outfile=outfile, deterministic=True,
                                                           render=args.render)
                            ep_lengths.append(steps)
                            rewards.append(reward)
                        print(
                            f"\tcheckpoint {i + 1}: nr episodes: {len(ep_lengths)}, "
                            f"mean ep lengths: {np.mean(ep_lengths):.3f} +- {np.std(ep_lengths):.3f}, "
                            f"mean rewards: {np.mean(rewards):.3f} +- {np.std(rewards):.3f}")
                        with open(str(outdir / f'{env_name}_{task_name}_{reward_name}_{timeid}_cp{i}.txt'), 'w+') as f:
                            json.dump({"checkpoints": [str(f) for f in envs_task_rew_files[env_name][task_name][reward_name]],
                                       "n_episodes": args.n_episodes,
                                       "rewards": rewards,
                                       "ep_lengths": ep_lengths}, f)
                    env.close()


if __name__ == "__main__":
    import time

    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=pathlib.Path, required=True)
    parser.add_argument("--regex", type=str, default=["**"], nargs="+",
                        help="for each regex, group data for `{logdir}/{regex}/model*steps.zip`")
    parser.add_argument("--min_steps", type=int, default=0, help="filter checkpoint with steps > min_steps")
    parser.add_argument("--max_steps", type=int, default=1e10, help="filter checkpoint with steps < max_steps")
    parser.add_argument("--n_episodes", type=int, default=1, help="nr evaluation episodes")
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("videos"), help="where save videos")
    parser.add_argument("-save", action="store_true")
    parser.add_argument("-info", action="store_true")
    parser.add_argument("-render", action="store_true")
    args = parser.parse_args()
    main(args)
    tf = time.time()
    print(f"[done] elapsed time: {tf - t0:.2f} seconds")
