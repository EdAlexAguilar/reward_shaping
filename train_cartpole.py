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
parser.add_argument("--reward", type=str, required=True, choices=['indicator'])
parser.add_argument("--algo", type=str, required=True, choices=['ppo', 'ppo_sde'])
parser.add_argument("--steps", type=int, default=1e6)
parser.add_argument("-terminate_on_collision", action="store_true")
parser.add_argument("-shift_reward", action="store_true")
parser.add_argument("-clip_reward", action="store_true")
args = parser.parse_args()


task = "cart_pole"
reward = args.reward
rl_algo = args.algo
steps = args.steps
clip_reward = args.clip_reward
shift_reward = args.shift_reward
terminate_on_collision = args.terminate_on_collision

# logging
logdir = pathlib.Path(f"logs/{task}_{reward}_clip{clip_reward}_shift{shift_reward}_terminate{terminate_on_collision}_{int(time.time())}")
checkpointdir = logdir / "checkpoint"
logdir.mkdir(parents=True, exist_ok=True)
checkpointdir.mkdir(parents=True, exist_ok=True)

# store input params
with open(logdir / f"args.yml", "w") as file:
    yaml.dump(args, file)

# load env params
env_config = pathlib.Path("envs") / f"{task}.yml"
with open(env_config, 'r') as file:
    env_params = yaml.load(file, yaml.FullLoader)
# eventually overwrite some default param
env_params['terminate_on_collision'] = terminate_on_collision
with open(logdir / f"{  task}.yml", "w") as file:
    yaml.dump(env_params, file)
# make env
env = CartPoleContObsEnv(**env_params)

# hierarchical definition of rewards: safety, target, comfort rules
if reward == "indicator":
    # todo: additionally, add comfort requirement to constraint actuators or velocity limits
    keep_balance = lambda state: (np.deg2rad(env_params['theta_threshold_deg']) - np.abs(state[2])) / np.deg2rad(env_params['theta_threshold_deg'])
    reach_origin = lambda state: (0.0 - np.abs(state[0])) / env_params['x_threshold']
    hierarchy = {
        'safety': [keep_balance],
        'target': [reach_origin],
        'comfort': []
    }
    env = HierarchicalRewardWrapper(env, hierarchy, clip_negative_rewards=clip_reward, shift_rewards=shift_reward)
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
n_steps = steps
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
