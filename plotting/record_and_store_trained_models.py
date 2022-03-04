import argparse
import json
import pathlib
import numpy as np
import time
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from stable_baselines3 import SAC

from plotting.utils import parse_env_task
from reward_shaping.training.utils import make_env


def record_rollout(model, env, outfile, deterministic=True, render=False):
    recorder = VideoRecorder(env, base_path=outfile)
    obs = env.reset()
    done = False
    steps = 0
    rtg = 0
    observations, infos = [], []
    rewards = []
    while not done:
        steps += 1
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)
        observations.append(obs)
        infos.append(info)
        rewards.append(reward)
        rtg += reward
        recorder.capture_frame()
        if render:
            env.render()
    np.savez(outfile, observations=observations, infos=infos, rewards=rewards)
    return steps, rtg


def plot_file_info(args):
    for cp in args.checkpoints:
        print(f"[info] file: {str(cp)}, exists: {cp.exists()}")


def main(args):
    # print debug info
    if args.info:
        plot_file_info(args)
        return
    # def out dir for storing videos
    outdir = args.outdir / f"{int(time.time())}"
    outdir.mkdir(parents=True, exist_ok=True)

    # collect checkpoints
    for i, cpfile in enumerate(args.checkpoints):
        env_name, task_name = parse_env_task(str(cpfile))
        env, env_params = make_env(env_name, task_name, 'eval', eval=True, logdir=None, seed=1)
        model = SAC.load(str(cpfile))
        for ep in range(args.n_episodes):
            outfile = str(outdir / f'{env_name}_{task_name}_cp{i}_ep{ep + 1}')
            steps, reward = record_rollout(model, env, outfile=outfile, deterministic=True, render=args.render)
            print(f"\tcheckpoint {i + 1}: episode: {ep + 1}: steps: {steps}, reward: {reward}")
        with open(str(outdir / f'{env_name}_{task_name}.txt'), 'w+') as f:
            json.dump({"checkpoint": str(cpfile),
                       "n_episodes": args.n_episodes}, f)
        env.close()


if __name__ == "__main__":
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=pathlib.Path, nargs="+", required=True)
    parser.add_argument("--n_episodes", type=int, default=1, help="nr evaluation episodes")
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("recording"), help="where save output")
    parser.add_argument("-info", action="store_true")
    parser.add_argument("-render", action="store_true")
    args = parser.parse_args()
    main(args)
    tf = time.time()
    print(f"[done] elapsed time: {tf - t0:.2f} seconds")
