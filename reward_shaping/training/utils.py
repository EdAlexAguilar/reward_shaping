import os
import pathlib

import matplotlib.pyplot as plt
import yaml
from gym.wrappers import FlattenObservation
from stable_baselines3.common.env_checker import check_env

from reward_shaping.core.wrappers import RewardWrapper
from reward_shaping.envs.racecar.wrappers.wrappers import FixSpeedControl
from reward_shaping.monitor.task import RLTask


def make_env(env_name, task, reward, eval=False, logdir=None, seed=0):
    # make base env
    extra_params = load_eval_params(env_name, task) if eval else {}
    extra_params['seed'] = seed
    env_params = load_env_params(env_name, task, **extra_params)
    # copy params in logdir (optional)
    if logdir:
        with open(logdir / f"{task}.yml", "w") as file:
            yaml.dump(env_params, file)
    env = make_base_env(env_name, env_params)
    # set reward
    env = make_reward_wrap(env_name, env, env_params, reward)
    if env_name == "f1tenth":
        from reward_shaping.envs.f1tenth.core.wrappers.wrappers import FixResetWrapper
        env = FixResetWrapper(env, mode="grid")
    elif env_name == "racecar":
        from reward_shaping.envs.racecar.wrappers import FixResetWrapper
        env = FixResetWrapper(env, mode="grid" if eval else "random")
    else:
        env = FlattenObservation(env)
    check_env(env)
    return env, env_params


def load_env_params(env, task, **kwargs):
    try:
        config = pathlib.Path(f"{os.path.dirname(__file__)}/../envs/{env}/config") / f"{task}.yml"
        with open(config, 'r') as file:
            params = yaml.load(file, yaml.FullLoader)
    except FileNotFoundError as error:
        params = {'task': task}
    # update params
    for key, value in kwargs.items():
        params[key] = value
    return params


def load_eval_params(env, task):
    """ this can be use to pass additional parameters to constraint the evaluation episodes."""
    return {'eval': True}


def make_base_env(env, env_params={}):
    if env == "cart_pole_obst":
        from reward_shaping.envs import CartPoleContObsEnv
        from reward_shaping.envs.cart_pole_obst.specs import get_all_specs
        env = CartPoleContObsEnv(**env_params)
        specs = [(k, op, build_pred(env_params)) for k, (op, build_pred) in get_all_specs().items()]
        env = RLTask(env=env, requirements=specs)
    elif env == "bipedal_walker":
        from reward_shaping.envs import BipedalWalker
        from reward_shaping.envs.bipedal_walker.specs import get_all_specs
        env = BipedalWalker(**env_params)
        specs = [(k, op, build_pred(env_params)) for k, (op, build_pred) in get_all_specs().items()]
        env = RLTask(env=env, requirements=specs)
    elif env == "lunar_lander":
        from reward_shaping.envs import LunarLanderContinuous
        from reward_shaping.envs.lunar_lander.specs import get_all_specs
        env = LunarLanderContinuous(**env_params)
        specs = [(k, op, build_pred(env_params)) for k, (op, build_pred) in get_all_specs().items()]
        env = RLTask(env=env, requirements=specs)
    elif env == "f1tenth":
        from reward_shaping.envs.f1tenth.core.single_agent_env import SingleAgentRaceEnv
        from reward_shaping.envs.f1tenth.core.wrappers.wrappers import FlattenAction, FrameSkip
        from gym.wrappers import RescaleAction
        from reward_shaping.envs.f1tenth.specs import get_all_specs
        env = SingleAgentRaceEnv(map_name="InformatikLectureHall", **env_params)
        specs = [(k, op, build_pred(env_params)) for k, (op, build_pred) in get_all_specs().items()]
        env = FrameSkip(env, skip=env_params['observations_conf']['frame_skip'])
        env = RLTask(env=env, requirements=specs)
        env = FlattenAction(env)
        env = RescaleAction(env, a=-1, b=+1)
    elif env == "racecar":
        from reward_shaping.envs.racecar.single_agent_env import CustomSingleAgentRaceEnv
        from reward_shaping.envs.racecar.vectorized_single_agent_env import ChangingTrackSingleAgentRaceEnv
        from reward_shaping.envs.racecar.specs import get_all_specs
        from reward_shaping.envs.racecar.wrappers import FlattenAction
        env = ChangingTrackSingleAgentRaceEnv(**env_params)
        specs = [(k, op, build_pred(env_params)) for k, (op, build_pred) in get_all_specs().items()]
        env = RLTask(env=env, requirements=specs)
        env = FixSpeedControl(env, fixed_speed=0.0)        # 0 in the normalized scale is half of max speed
        env = FlattenAction(env)
    else:
        raise NotImplementedError(f"not implemented env for {env}")
    return env


def make_agent(env_name, env, reward, rl_algo, logdir=None):
    policy = "MultiInputPolicy" if env_name in ["f1tenth", "racecar"] else "MlpPolicy"
    # load model parameters
    algo = rl_algo.split("_", 1)[0]
    algo_config = pathlib.Path(f"{os.path.dirname(__file__)}/../envs/{env_name}/hparams") / f"{algo}.yml"
    if algo_config.exists():
        with open(algo_config, 'r') as file:
            algo_params = yaml.load(file, yaml.FullLoader)
    else:
        algo_params = {}
    if 'tl' in reward:
        # propagate the terminal reward over all the states in the episode
        algo_params['gamma'] = 1.0
    # create model
    if algo == "ppo":
        from stable_baselines3 import PPO
        model = PPO(policy, env, verbose=1, tensorboard_log=logdir, **algo_params)
    elif algo == "sac":
        from stable_baselines3 import SAC
        model = SAC(policy, env, verbose=1, tensorboard_log=logdir, **algo_params)
    elif algo == "ddpg":
        from stable_baselines3 import DDPG
        model = DDPG(policy, env, verbose=1, tensorboard_log=logdir, **algo_params)
    elif algo == "ars":
        from sb3_contrib import ARS
        model = ARS(policy, env, verbose=1, tensorboard_log=logdir, **algo_params)
    elif algo == "td3":
        from stable_baselines3 import TD3
        model = TD3(policy, env, verbose=1, tensorboard_log=logdir, **algo_params)
    else:
        raise NotImplementedError()
    # copy params in logdir (optional)
    if logdir:
        with open(logdir / f"{rl_algo}.yml", "w") as file:
            yaml.dump(algo_params, file)
    return model


def get_reward_conf(env_name, env_params, reward):
    if env_name == "cart_pole_obst":
        from reward_shaping.envs.cart_pole_obst import get_reward
        reward_conf = get_reward(reward)(env_params=env_params)
    elif env_name == "bipedal_walker":
        from reward_shaping.envs.bipedal_walker import get_reward
        reward_conf = get_reward(reward)(env_params=env_params)
    elif env_name == "lunar_lander":
        from reward_shaping.envs.lunar_lander import get_reward
        reward_conf = get_reward(reward)(env_params=env_params)
    elif env_name == "f1tenth":
        from reward_shaping.envs.f1tenth import get_reward
        reward_conf = get_reward(reward)(env_params=env_params)
    elif env_name == "racecar":
        from reward_shaping.envs.racecar import get_reward
        reward_conf = get_reward(reward)(env_params=env_params)
    else:
        raise NotImplementedError(f'{reward} not implemented for {env_name}')
    return reward_conf


def make_reward_wrap(env_name, env, env_params, reward, logdir=None):
    reward_conf = get_reward_conf(env_name, env_params, reward)
    if 'tltl' in reward:
        from reward_shaping.core.wrappers import TLRewardWrapper
        env = TLRewardWrapper(env, tl_conf=reward_conf, window_len=None, eval_at_end=True)
    elif 'bhnr' in reward:
        from reward_shaping.core.wrappers import TLRewardWrapper
        window = int(env_params["max_steps"] // 20)
        env = TLRewardWrapper(env, tl_conf=reward_conf, window_len=window, eval_at_end=False)
    elif 'eval' in reward:
        from reward_shaping.core.wrappers import EvaluationRewardWrapper
        env = EvaluationRewardWrapper(env, conf=reward_conf)
    else:
        reward_fn = reward_conf
        env = RewardWrapper(env, reward_fn=reward_fn)
    if env_name == "f1tenth":
        from reward_shaping.envs.f1tenth.core.wrappers.wrappers import FilterObservationWrapper, NormalizeObservations
        env = FilterObservationWrapper(env, ["lidar_occupancy", "speed_cmd", "steering_cmd"])
    if env_name == "racecar":
        from reward_shaping.envs.racecar.wrappers import FilterObservationWrapper
        env = FilterObservationWrapper(env, ['lidar_occupancy', 'steering', 'speed', 'dist_to_wall'])
    return env
