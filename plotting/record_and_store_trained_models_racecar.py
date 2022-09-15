import argparse
import json
import pathlib

import PIL.Image
import imageio
import numpy as np
import time
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from stable_baselines3 import SAC

from plotting.utils import parse_env_task, parse_reward
from reward_shaping.training.utils import make_env


def record_rollout(model, env, outfile, deterministic=True, render=False):
    obs = env.reset()
    done = False
    steps = 0
    rtg = 0
    observations, infos = [], []
    rewards = []
    frames = []
    while not done:
        steps += 1
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)
        observations.append(obs)
        infos.append(info)
        rewards.append(reward)
        rtg += reward
        frame = env.render(mode="rgb_array")
        frames.append(frame)
    np.savez(outfile, observations=observations, infos=infos, rewards=rewards, allow_pickle=True)
    frame_dir = pathlib.Path(outfile)
    frame_dir.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(frame_dir / 'movie.gif', mode='I', fps=10) as writer:
        for i, frame in enumerate(frames):
            PIL.Image.fromarray(frame).save(frame_dir / f"frame_{i}.png")
            writer.append_data(frame)

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
        reward_name = parse_reward(str(cpfile))
        env, env_params = make_env(env_name, task_name, 'eval', eval=True, logdir=None, seed=1)
        model = SAC.load(str(cpfile))
        for ep in range(args.n_episodes):
            print(f"\tcheckpoint {i + 1}: {cpfile}, episode: {ep + 1}")
            outfile = str(outdir / f'{env_name}_{task_name}_{reward_name}_cp{i}_ep{ep + 1}')
            steps, reward = record_rollout(model, env, outfile=outfile, deterministic=True, render=args.render)
            print(f"\tcheckpoint {i + 1}: episode: {ep + 1}: steps: {steps}, reward: {reward}")
        with open(str(outdir / f'{env_name}_{task_name}_{i}.txt'), 'w+') as f:
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
