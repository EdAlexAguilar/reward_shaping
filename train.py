import pathlib
import time

import gym

from stable_baselines3 import PPO
import numpy as np

import callbacks
from envs.cp_continuousobstacle_env import CartPoleContObsEnv
from envs.reward_envs import HierarchicalRewardWrapper

# define problem
task = "cart_pole"
reward = "indicator"
rl_algo = "ppo"

# logging
logdir = pathlib.Path(f"logs/{task}_{reward}_{rl_algo}_{int(time.time())}")
logdir.mkdir(parents=True, exist_ok=True)

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
    reach_origin = lambda state: (0.0 - np.abs(state[0])) / sim_params['x_threshold']
    hierarchy = {
        'safety': [keep_balance],
        'target': [reach_origin],
        'comfort': []
    }
    env = HierarchicalRewardWrapper(env, hierarchy)
else:
    raise NotImplementedError()

# define rl agent
if rl_algo == "ppo":
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
else:
    raise NotImplementedError()

# trainig
n_steps = 100_000

eval_env = gym.wrappers.Monitor(env, logdir / "videos")
video_cb = callbacks.VideoRecorderCallback(eval_env, render_freq=int(n_steps//10), n_eval_episodes=1)

model.learn(total_timesteps=n_steps, callback=video_cb)

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
