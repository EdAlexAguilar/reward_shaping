import pathlib
import time

import numpy as np
import yaml

from envs.cart_pole.cp_continuousobstacle_env import CartPoleContObsEnv
from envs.reward_envs import HierarchicalRewardWrapper


def make_log_dirs(args):
    logdir = pathlib.Path(
        f"logs/{args.task}_{args.reward}_clip{args.clip_reward}_shift{args.shift_reward}_terminate{args.terminate_on_collision}_unitScale{args.unit_scaling}_{int(time.time())}")
    checkpointdir = logdir / "checkpoint"
    logdir.mkdir(parents=True, exist_ok=True)
    checkpointdir.mkdir(parents=True, exist_ok=True)
    # store input params
    with open(logdir / f"args.yml", "w") as file:
        yaml.dump(args, file)
    return logdir, checkpointdir


def make_base_env(task, env_params={}):
    if task == "cart_pole":
        env = CartPoleContObsEnv(**env_params)
    else:
        raise NotImplementedError(f"not implemented env for {task}")
    return env


def make_env(task, terminate_on_collision, logdir=None):
    env_config = pathlib.Path(f"envs/{task}") / f"{task}.yml"
    with open(env_config, 'r') as file:
        env_params = yaml.load(file, yaml.FullLoader)
    # eventually overwrite some default param
    env_params['terminate_on_collision'] = terminate_on_collision
    # copy params in logdit (optional)
    if logdir:
        with open(logdir / f"{task}.yml", "w") as file:
            yaml.dump(env_params, file)
    # make env
    env = make_base_env(task, env_params)
    return env, env_params


def make_agent(env, rl_algo, logdir):
    if rl_algo == "ppo":
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
    elif rl_algo == "ppo_sde":
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir, use_sde=True)
    else:
        raise NotImplementedError()
    return model


def make_reward_wrap(task, env, reward, env_params, reward_params):
    if task == "cart_pole":
        # SAFETY: no collision
        # rho := -1 if collision else 0
        no_collision_rho = lambda x, theta: -1 * env.obstacle.intersect(x, theta)
        collision_rho_min, collision_rho_max = -1, 0  # either -1 if collision, or 0 if no collision
        no_collision = lambda state: (
        (no_collision_rho(state[0], state[2]) - collision_rho_min) / (collision_rho_max - collision_rho_min),
        no_collision_rho(state[0], state[2]) >= 0)
        # TARGET: balance the pole
        # phi := theta >= -24deg and theta <= 24deg
        # rho := min(theta+24, 24-theta)
        keep_balance_rho = lambda theta: min(theta - np.deg2rad(env_params['theta_deg_target_min']),
                                             np.deg2rad(env_params['theta_deg_target_max']) - theta)
        # theta_rho_min, theta_rho_max are the min and max value of robustness at extremes (where rob is min/maximized)
        theta_rho_min = min(keep_balance_rho(np.deg2rad(-env_params['theta_threshold_deg'])),
                            keep_balance_rho(np.deg2rad(env_params['theta_threshold_deg'])))
        theta_target_center = env_params['theta_deg_target_min'] + \
                              (env_params['theta_deg_target_max'] - env_params['theta_deg_target_min']) / 2
        theta_rho_max = keep_balance_rho(np.deg2rad(theta_target_center))
        keep_balance = lambda state: ((keep_balance_rho(state[2]) - theta_rho_min) / (theta_rho_max - theta_rho_min),
                                      keep_balance_rho(state[2]) >= 0)
        # COMFORT: reach the origin
        # phi_1 := x == 0 <-> x>=0 and x<=0
        # rho_1 := min(x, -x)
        reach_origin_rho = lambda x: min(x - env_params['x_target_min'], env_params['x_target_max'] - x)
        # x_rho_min, x_rho_max are the min and max value of robustness at extremes
        x_rho_min = min(reach_origin_rho(-env_params['x_threshold']), reach_origin_rho(env_params['x_threshold']))
        x_target_center = env_params['x_target_min'] + (env_params['x_target_max'] - env_params['x_target_min']) / 2
        x_rho_max = reach_origin_rho(x_target_center)
        reach_origin = lambda state: ((reach_origin_rho(state[0]) - x_rho_min) / (x_rho_max - x_rho_min),
                                      reach_origin_rho(state[0]) >= 0)
        # phi_2 := -1 * overcome_obstacle(x)
        # rho_2 := -1 if not overcome else 0
        overcome_obs_rho = lambda x: -1 * (not env.overcome_obstacle(x))
        overcome_rho_min, overcome_rho_max = -1, 0  # either -1 if not overcome, or 0
        overcome_obs = lambda state: (
        (overcome_obs_rho(state[0]) - overcome_rho_min) / (overcome_rho_max - overcome_rho_min),
        overcome_obs_rho(state[0]) >= 0)
        if reward == "indicator":
            # define hierachy
            hierarchy = {
                'safety': [no_collision, keep_balance],
                'target': [overcome_obs],
                'comfort': [reach_origin]
            }
            env = HierarchicalRewardWrapper(env, hierarchy, clip_negative_rewards=reward_params['clip_reward'],
                                            shift_rewards=reward_params['shift_reward'],
                                            unit_scaling=reward_params['unit_scaling'])
        elif reward == "indicator_reverse":
            # Always indicator reward but first objective is to reach the origin, then to keep balance
            hierarchy = {
                'safety': [reach_origin],
                'target': [keep_balance],
                'comfort': [no_collision]
            }
            env = HierarchicalRewardWrapper(env, hierarchy, clip_negative_rewards=reward_params['clip_reward'],
                                            shift_rewards=reward_params['shift_reward'],
                                            unit_scaling=reward_params['unit_scaling'])
        else:
            raise NotImplementedError()
    return env
