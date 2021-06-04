import pathlib
import time

import gym

from stable_baselines3 import PPO
import numpy as np

import callbacks
from envs.cp_continuousobstacle_env import CartPoleContObsEnv
from envs.reward_envs import HierarchicalRewardWrapper
from stable_baselines3.common.callbacks import CheckpointCallback
import argparse as parser

# define problem
parser = parser.ArgumentParser()
parser.add_argument("--task", type=str, required=True, choices=['cart_pole'])
parser.add_argument("--reward", type=str, required=True, choices=['indicator'])
parser.add_argument("--algo", type=str, required=True, choices=['ppo'])
parser.add_argument("--clip_reward", type=str, default=False)
args = parser.parse_args()

task = args.task
reward = args.reward
rl_algo = args.algo
clip_reward = args.clip_reward

# logging
logdir = pathlib.Path(f"logs/{task}_{reward}_clip{clip_reward}_{rl_algo}_{int(time.time())}")
checkpointdir = logdir / "checkpoint"
logdir.mkdir(parents=True, exist_ok=True)
checkpointdir.mkdir(parents=True, exist_ok=True)


# define simulation parameters
sim_params = {
    'x_threshold': 2.5,  # episode ends when the distance from origin goes above this threshold
    'theta_threshold_deg': 24,  # episode ends when the pole angle (w.r.t. vertical axis) goes above this threshold
    'max_steps': 400,  # episode ends after 400 steps
}
env = CartPoleContObsEnv(x_threshold=sim_params['x_threshold'],
                         theta_threshold_deg=sim_params['theta_threshold_deg'],
                         max_steps=sim_params['max_steps'])

# hierarchical definition of rewards: safety, target, comfort rules
if reward == "indicator":
    keep_balance = lambda state: (np.deg2rad(24) - np.abs(state[2])) / np.deg2rad(sim_params['theta_threshold_deg'])
    # todo: limit speed or actuators
    reach_origin = lambda state: (0.0 - np.abs(state[0])) / sim_params['x_threshold']
    hierarchy = {
        'safety': [keep_balance],
        'target': [reach_origin],
        'comfort': []
    }
    env = HierarchicalRewardWrapper(env, hierarchy, clip_negative_rewards=clip_reward)
else:
    raise NotImplementedError()

# define rl agent
if rl_algo == "ppo":
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
else:
    raise NotImplementedError()

# trainig
n_steps = 1e6
eval_every = 1e5
checkpoint_every = 1e4

eval_env = gym.wrappers.Monitor(env, logdir / "videos")
video_cb = callbacks.VideoRecorderCallback(eval_env, render_freq=eval_every, n_eval_episodes=1)
checkpoint_callback = CheckpointCallback(save_freq=checkpoint_every, save_path=checkpointdir,
                                         name_prefix='model')
model.learn(total_timesteps=n_steps, callback=[video_cb, checkpoint_callback])

# evaluation
obs = env.reset()
env.render()
rewards = []
tot_reward = 0.0
for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    tot_reward += reward
    env.render()
    if done:
        rewards.append(tot_reward)
        obs = env.reset()
        tot_reward = 0.0

env.close()
