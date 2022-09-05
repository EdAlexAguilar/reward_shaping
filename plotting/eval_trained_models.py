import argparse
import json
import pathlib
import warnings
from typing import Dict, Tuple, List
import re

import numpy as np
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


def main(args):
    # print debug info
    if args.info:
        plot_file_info(args)
        return
    # collect checkpoints
    for regex in args.regex:
        filter = lambda f: "skip" not in str(f) and args.min_steps <= int(
            re.findall(r'\d+', f.stem)[-1]) <= args.max_steps
        files = [f for f in get_files(args.logdir, regex, fileregex=file_regex) if filter(f)]
        print(f"regex: {regex}, nr files: {len(files)}")
        envs_task_rew_files = group_checkpoints_per_env_task_reward(files)
        #
        results: Dict[
            Tuple[str, str, str], Tuple[int, int]] = {}  # results dict: (env x task x reward x rew x cls) -> succr
        for env_name in envs_task_rew_files:
            if not env_name in results:
                results[env_name] = {}
            for task_name in envs_task_rew_files[env_name]:
                if not task_name in results[env_name]:
                    results[env_name][task_name] = {}
                for reward_name in envs_task_rew_files[env_name][task_name]:
                    if not reward_name in results[env_name][task_name]:
                        results[env_name][task_name][reward_name] = {}
                    print(f"\n\n[simulation] env: {env_name}, task: {task_name}, reward: {reward_name}")
                    env, env_params = make_env(env_name, task_name, 'eval', eval=True, logdir=None, seed=0)
                    list_of_metrics = [f"{req}_counter" for req in env.req_labels]
                    # init results
                    results[env_name][task_name][reward_name]["rewards"] = []
                    results[env_name][task_name][reward_name]["ep_lengths"] = []
                    for metric in list_of_metrics:
                        results[env_name][task_name][reward_name][metric] = []
                    for i, cpfile in enumerate(envs_task_rew_files[env_name][task_name][reward_name]):
                        model = SAC.load(str(cpfile))
                        rewards, eplens, metrics = evaluate_policy_with_monitors(model, env,
                                                                                 n_eval_episodes=args.n_episodes,
                                                                                 deterministic=True,
                                                                                 render=args.render,
                                                                                 return_episode_rewards=True,
                                                                                 list_of_metrics=list_of_metrics)
                        # concatenate evaluations (json does not recognize np datatype, convert to python int and float)
                        results[env_name][task_name][reward_name]["rewards"] += [float(r) for r in rewards]
                        results[env_name][task_name][reward_name]["ep_lengths"] += [int(l) for l in eplens]
                        for metric in list_of_metrics:
                            results[env_name][task_name][reward_name][metric] += [int(l) for l in metrics[metric]]
                        print(f"\tcheckpoint {i + 1}: nr episodes: {len(rewards)}, " \
                              f"mean reward: {np.mean(rewards):.5f}, mean lengths: {np.mean(eplens):.5f}")
                    env.close()
        # save
        if args.save:
            with open(f"offline_evaluation_episodes{args.n_episodes}_{time.time()}.json", "w+") as f:
                json.dump(results, f)
        # print
        for env_name in results:
            for task_name in results[env_name]:
                for reward_name in results[env_name][task_name]:
                    print(f"[results] env: {env_name}, task: {task_name}, reward: {reward_name}")
                    rewards = results[env_name][task_name][reward_name]["rewards"]
                    safety_sr = np.mean([int(r > 1) for r in rewards])
                    safety_target_sr = np.mean([int(r > 1.5) for r in rewards])
                    safety_target_comfort_sr = np.mean([float(r > 1.5) * 1 / 0.25 * float(r - 1.5) for r in rewards])
                    for metric, succrate in zip(["safety", "safety+target", "safety+target+comfort"],
                                                [safety_sr, safety_target_sr, safety_target_comfort_sr]):
                        print(f"\t{metric}: {succrate}")
                    # plot individual requirements
                    print()
                    print("[comfort reqs]")
                    for metric, vals in results[env_name][task_name][reward_name].items():
                        if not metric.startswith("c"):
                            continue
                        res = np.array(vals) / np.array(results[env_name][task_name][reward_name]["ep_lengths"])
                        mu, std = np.mean(res), np.std(res)
                        print(f"\t{metric}: {mu} +- {std}")


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
    parser.add_argument("-save", action="store_true")
    parser.add_argument("-info", action="store_true")
    parser.add_argument("-render", action="store_true")
    args = parser.parse_args()
    main(args)
    tf = time.time()
    print(f"[done] elapsed time: {tf - t0:.2f} seconds")
