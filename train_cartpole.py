import pathlib
import time

import gym
import yaml
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
parser.add_argument("--algo", type=str, required=True, choices=['ppo', 'ppo_sde'])
parser.add_argument("--clip_reward", type=bool, default=False)
parser.add_argument("--shift_reward", type=bool, default=False)
args = parser.parse_args()

task = args.task
reward = args.reward
rl_algo = args.algo
clip_reward = args.clip_reward
shift_reward = args.shift_reward

# logging
logdir = pathlib.Path(f"logs/{task}_{reward}_clip{clip_reward}_shift{shift_reward}_{rl_algo}_{int(time.time())}")
checkpointdir = logdir / "checkpoint"
logdir.mkdir(parents=True, exist_ok=True)
checkpointdir.mkdir(parents=True, exist_ok=True)

# define simulation parameters
sim_params = {
    'x_threshold': 2.5,  # episode ends when the distance from origin goes above this threshold
    'theta_threshold_deg': 24,  # episode ends when the pole angle (w.r.t. vertical axis) goes above this threshold
    'max_steps': 400,  # episode ends after this nr of steps
    'cart_min_offset': 1.0,  # initial position sampled with this min distance (offset w.r.t. vertical axis)
    'cart_max_offset': 2.0,  # initial position sampled with this max distance (offset w.r.t. vertical axis)
    'obstacle_min_w': 0.25,  # obstacle sizes sampled within these ranges for width, height
    'obstacle_max_w': 0.25,
    'obstacle_min_h': 0.1,
    'obstacle_max_h': 0.1,
    'obstacle_min_dist': 0.1,  # obstacle initial position is within a given distance from the cart center
    'obstacle_max_dist': 0.3,  # this distance is between the closest point of the obstacle and the cart center
}
with open(logdir / 'env_params.yaml', 'w') as file:
    yaml.dump(sim_params, file)

env = CartPoleContObsEnv(x_threshold=sim_params['x_threshold'],
                         theta_threshold_deg=sim_params['theta_threshold_deg'],
                         max_steps=sim_params['max_steps'],
                         cart_min_initial_offset=sim_params['cart_min_offset'],
                         cart_max_initial_offset=sim_params['cart_max_offset'],
                         obstacle_min_w=sim_params['obstacle_min_w'],
                         obstacle_max_w=sim_params['obstacle_max_w'],
                         obstacle_min_h=sim_params['obstacle_min_h'],
                         obstacle_max_h=sim_params['obstacle_max_h'],
                         obstacle_min_dist=sim_params['obstacle_min_dist'],
                         obstacle_max_dist=sim_params['obstacle_max_dist'])

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
elif rl_algo == "ppo_sde":
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir, use_sde=True)
else:
    raise NotImplementedError()

# trainig
n_steps = 10e6
eval_every = 5e5
checkpoint_every = 5e5

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
