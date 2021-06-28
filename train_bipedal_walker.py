import time
import numpy as np
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from argparse import Namespace
import callbacks
from utils import make_log_dirs, make_env


def main(args):
    args.task = "bipedal_walker"
    logdir, checkpointdir = make_log_dirs(args)
    env, env_params = make_env(args.task, args.terminate_on_collision, logdir)

    model = PPO("MlpPolicy", env, tensorboard_log=logdir, verbose=1, n_steps=2048,
                batch_size=64, gae_lambda=0.95, gamma=0.99, n_epochs=10, ent_coef=0.001, learning_rate=0.00025,
                clip_range=0.2)

    eval_env = gym.wrappers.Monitor(env, logdir / "videos")
    video_cb = callbacks.VideoRecorderCallback(eval_env, render_freq=100000, n_eval_episodes=1)
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=checkpointdir,
                                             name_prefix='model')
    model.learn(total_timesteps=args.steps, callback=[video_cb, checkpoint_callback])

    rewards = []
    for i in range(5):
        obs = env.reset()
        done = False
        sum_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            sum_reward += reward
        rewards.append(sum_reward)

    env.close()


if __name__ == "__main__":
    import argparse as parser

    parser = parser.ArgumentParser()
    parser.add_argument("--reward", type=str, default="default",
                        choices=['indicator', 'indicator_reverse', 'weighted', 'default'])
    parser.add_argument("--algo", type=str, required=True, choices=['ppo'])
    parser.add_argument("--steps", type=int, default=2e6)
    parser.add_argument("-terminate_on_collision", action="store_true")
    parser.add_argument("-shift_reward", action="store_true")
    parser.add_argument("-clip_reward", action="store_true")
    parser.add_argument("-unit_scaling", action="store_true")
    args = parser.parse_args()
    main(args)
