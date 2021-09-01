import argparse
import json
import pathlib
import time
from typing import List, Dict

import numpy as np
import yaml
import json

from gym.wrappers import Monitor
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from stable_baselines3 import SAC

from reward_shaping.core.helper_fns import monitor_episode
from reward_shaping.training.utils import make_env
from shutil import copyfile


def make_log_dirs(args):
    basedir = f"logs/evaluations/{args.env}"
    logdir_template = "{}/{}_{}_{}"
    logdir = pathlib.Path(logdir_template.format(basedir, args.task, args.eval_reward, int(time.time())))
    checkpointdir = logdir / "checkpoint"
    episodedir = logdir / "episodes"
    logdir.mkdir(parents=True, exist_ok=True)
    checkpointdir.mkdir(parents=True, exist_ok=True)
    episodedir.mkdir(parents=True, exist_ok=True)
    # store input params
    with open(logdir / f"args.yml", "w") as file:
        yaml.dump(args, file)
    # copy checkpoint
    copyfile(args.checkpoint, checkpointdir / args.checkpoint.parts[-1])
    return logdir


def evaluate_individual_requirements(env, monitored_episode):
    # todo: question: do we want to return the boolean eval of individual requirements, the episodic rob, or the whole rob signal?
    stl_conf = env.env._stl_conf
    variables, types = stl_conf.monitoring_variables, stl_conf.monitoring_types
    results = {}
    for req, spec in stl_conf.requirements_dict.items():
        robustness = monitor_episode(spec, variables, types, monitored_episode)
        results[req] = robustness
    return results


def evaluate_individual_nodes(env):
    nodes = env.env._reward_fn._graph.nodes
    result = {}
    for node in nodes:
        reward = nodes[node]['reward']
        sat = nodes[node]['sat']
        score = nodes[node]['score']
        result[node] = (reward, sat, score)
    return result


def evaluate_model(env_name, model, env, reward_name, n_episodes=None, n_steps=None, logdir=None,
                   render=True, record=False):
    assert not n_episodes or not n_steps
    assert not n_steps or not n_episodes
    tot_reward = 0.0
    episodes, steps = 0, 0
    episode = []
    all_results = []
    all_node_status = []
    recorder = VideoRecorder(env, base_path=str(logdir/f'episodes/episode_{episodes + 1}'))
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        episode.append({'observation': obs, 'action': action})
        obs, reward, done, info = env.step(action)
        tot_reward += reward
        steps += 1
        if 'gb' in reward_name:
            nodes_status = evaluate_individual_nodes(env)
            all_node_status.append(nodes_status)
        if render:
            env.render()
        if record:
            recorder.capture_frame()
        if done:
            episode.append({'observation': obs, 'action': None})
            # result
            print(f"[Info] Tot reward: {tot_reward}")
            monitored_episode, results = None, None
            if 'stl' in reward_name and env_name == "cart_pole_obst":
                monitored_episode = env.get_monitored_episode()
                results = evaluate_individual_requirements(env, monitored_episode)
                all_results.append(results)
                for req, rob in results.items():
                    print(f"[STL] {req}: {rob}")
                print()
            # store
            if logdir:
                filename = logdir / "episodes" / f"episode_{episodes + 1}"
                np.savez(file=filename, episode=episode, monitored_episode=monitored_episode,
                         results=results, nodes_status=all_node_status)
                nodefilename = str(filename) + "_nodes.txt"
                with open(nodefilename, 'w') as outfile:
                    json.dump(all_node_status, outfile)
            # reset
            episodes += 1
            recorder.close()
            recorder = VideoRecorder(env, base_path=str(logdir / f'episodes/episode_{episodes + 1}'))
            obs = env.reset()
            episode = []
            all_node_status = []
            tot_reward = 0.0
            if n_episodes and episodes >= n_episodes:
                print(f"[Info] episodes: {episodes}, step: {steps}, max number of episodes reached")
                break
        if n_steps and steps >= n_steps:
            print(f"[Info] episodes: {episodes}, step: {steps}, max number of steps reached")
            break
    if logdir:
        filename = logdir / "episodes" / f"results.txt"
        with open(filename, 'w') as outfile:
            json.dump(all_results, outfile)


def main(args):
    env_name, task, reward = args.env, args.task, args.eval_reward
    cp_filepath = args.checkpoint
    eval_episodes, no_render, save, record = args.eval_episodes, args.no_render, args.save, args.record
    #
    logdir = make_log_dirs(args) if save or record else None
    # make env
    env, env_params = make_env(env_name, task, reward)
    # resume agent
    model = SAC.load(cp_filepath)
    evaluate_model(env_name, model, env, reward, n_episodes=eval_episodes, logdir=logdir, render=not no_render,
                   record=record)


def get_default_task(env):
    if env == "cart_pole_obst":
        return 'fixed_height'
    if env == "bipedal_walker":
        return 'forward'
    if env == "lunar_lander":
        return 'land'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, choices=['cart_pole_obst', 'bipedal_walker', 'lunar_lander'])
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--eval_reward", type=str, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--eval_episodes", type=int, required=False, default=10)
    parser.add_argument("-no_render", action='store_true')
    parser.add_argument("-save", action='store_true')
    parser.add_argument("-record", action='store_true')
    # parse args
    args = parser.parse_args()
    args.task = args.task if args.task is not None else get_default_task(args.env)
    main(args)
