import pathlib
import time

import numpy as np
import yaml

from envs.cp_continuousobstacle_env import CartPoleContObsEnv
from envs.reward_envs import HierarchicalRewardWrapper


def make_log_dirs(args):
    logdir = pathlib.Path(
        f"logs/{args.task}_{args.reward}_clip{args.clip_reward}_shift{args.shift_reward}_terminate{args.terminate_on_collision}_{int(time.time())}")
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


def make_env(args, logdir=None):
    env_config = pathlib.Path("envs") / f"{args.task}.yml"
    with open(env_config, 'r') as file:
        env_params = yaml.load(file, yaml.FullLoader)
    # eventually overwrite some default param
    env_params['terminate_on_collision'] = args.terminate_on_collision
    # copy params in logdit (optional)
    if logdir:
        with open(logdir / f"{args.task}.yml", "w") as file:
            yaml.dump(env_params, file)
    # make env
    env = make_base_env(args.task, env_params)
    return env, env_params


def distance_to_mid_target_range(x, xtargetmin, xtargetmax, xmin=None, xmax=None):
    """
    Check if x is in range [xmin, xmax]
    If it is in range: return closeness to the mid point, value in [0., 1.] (greater is better)
    If it is not in range: return distance to the mid point, value in [-inf, 0.0) (greater is better)

    Note: if `xmin`, `xmax` provided: the distance when outrange is normalized in this range
    Note: the sign indicates if it is within the range (>0) or out the range (<0)
    """
    geq_min = +1 if x - xtargetmin >= 0 else -1
    leq_max = +1 if xtargetmax - x >= 0 else -1
    distance_to_mid = abs(x - (xtargetmin + (xtargetmax - xtargetmin) / 2.0))
    if geq_min > 0 and leq_max > 0:
        closeness_mid = (1.0 - np.min([distance_to_mid / (xtargetmax - xtargetmin + 0.001), 1.0]))
        result = geq_min * leq_max * closeness_mid
    else:
        if xmin is None or xmax is None:
            result = geq_min * leq_max * distance_to_mid
        else:
            # normalize in +-1
            norm_distance = distance_to_mid / (xmax - xmin)
            result = geq_min * leq_max * norm_distance
    return result


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
        if reward == "indicator":
            # todo: additionally, add comfort requirement to constraint actuators or velocity limits
            keep_balance = lambda state: distance_to_mid_target_range(state[2],
                                                                      np.deg2rad(env_params['theta_deg_target_min']),
                                                                      np.deg2rad(env_params['theta_deg_target_max']),
                                                                      -np.deg2rad(env_params['theta_threshold_deg']),
                                                                      +np.deg2rad(env_params['theta_threshold_deg'])
                                                                      )
            reach_origin = lambda state: distance_to_mid_target_range(state[0], env_params['x_target_min'],
                                                                      env_params['x_target_max'],
                                                                      -env_params['x_threshold'],
                                                                      +env_params['x_threshold']
                                                                      )
            hierarchy = {
                'safety': [keep_balance],
                'target': [reach_origin],
                'comfort': []
            }
            env = HierarchicalRewardWrapper(env, hierarchy, clip_negative_rewards=reward_params['clip_reward'],
                                            shift_rewards=reward_params['shift_reward'])
        elif reward=="indicator_reverse":
            # Always indicator reward but first objective is to reach the origin, then to keep balance
            keep_balance = lambda state: distance_to_mid_target_range(state[2],
                                                                      np.deg2rad(env_params['theta_deg_target_min']),
                                                                      np.deg2rad(env_params['theta_deg_target_max']),
                                                                      -np.deg2rad(env_params['theta_threshold_deg']),
                                                                      +np.deg2rad(env_params['theta_threshold_deg'])
                                                                      )
            reach_origin = lambda state: distance_to_mid_target_range(state[0], env_params['x_target_min'],
                                                                      env_params['x_target_max'],
                                                                      -env_params['x_threshold'],
                                                                      +env_params['x_threshold']
                                                                      )
            hierarchy = {
                'safety': [reach_origin],
                'target': [keep_balance],
                'comfort': []
            }
            env = HierarchicalRewardWrapper(env, hierarchy, clip_negative_rewards=reward_params['clip_reward'],
                                            shift_rewards=reward_params['shift_reward'])
        else:
            raise NotImplementedError()
    return env
