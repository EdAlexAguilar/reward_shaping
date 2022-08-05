import os
import pathlib

import yaml
from gym.wrappers import FlattenObservation
from stable_baselines3.common.env_checker import check_env

from reward_shaping.core.wrappers import RewardWrapper
from reward_shaping.envs.wrappers import FlattenAction, FrameSkip, DeltaSpeedWrapper
from reward_shaping.envs.wrappers import ActionHistoryWrapper, ObservationHistoryWrapper
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
    env = make_base_env(env_name, task, env_params)
    # set reward
    env = make_reward_wrap(env_name, env, env_params, reward)
    env = make_observation_wrap(env_name, env, env_params)
    env = FlattenObservation(env)
    env = FlattenAction(env)
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
    params = {'eval': True}
    if env == "bipedal_walker":
        params["max_steps"] = 1000
    if env == "lunar_lander":
        params["terminate_if_notawake"] = False
    return params


def make_base_env(env, task, env_params={}):
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
    elif "racecar" in env:
        # base env for either racecar or racecar2
        if env == "racecar":
            from reward_shaping.envs.racecar.single_agent_racecar_env import RacecarEnv
            from reward_shaping.envs.racecar.specs import get_all_specs
            env = RacecarEnv(**env_params)
        else:
            from reward_shaping.envs.racecar2.multi_agent_racecar_env import MultiAgentRacecarEnv
            from reward_shaping.envs.racecar2.specs import get_all_specs
            env = MultiAgentRacecarEnv(**env_params)

        # skip frame to match hardware frequency
        env = FrameSkip(env, skip=env_params["frame_skip"])  # skip 10 frames means control at 10 Hz
        # include past actions in observations
        if env_params["observation_config"]["use_history_wrapper"] == True:
            env = ActionHistoryWrapper(env, n_last_actions=env_params["observation_config"]["n_last_actions"])
        # change action space to control increase/decrease speed
        if env_params["action_config"]["delta_speed"] == True:
            assert all([p in env_params for p in ["frame_skip", "action_config"]]), "missing parameters racecar"
            env = DeltaSpeedWrapper(env, **env_params)

        specs = [(k, op, build_pred(env_params)) for k, (op, build_pred) in get_all_specs().items()]
        env = RLTask(env=env, requirements=specs)
    else:
        raise NotImplementedError(f"not implemented env for {env}")
    return env


def make_agent(env_name, env, reward, rl_algo, logdir=None):
    policy = "MlpPolicy"
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
    elif env_name == "racecar":
        from reward_shaping.envs.racecar import get_reward
        reward_conf = get_reward(reward)(env_params=env_params)
    elif env_name == "racecar2":
        from reward_shaping.envs.racecar2 import get_reward
        reward_conf = get_reward(reward)(env_params=env_params)
    else:
        raise NotImplementedError(f'{reward} not implemented for {env_name}')
    return reward_conf


def make_reward_wrap(env_name, env, env_params, reward, logdir=None):
    reward_conf = get_reward_conf(env_name, env_params, reward)
    if 'tltl' in reward:
        from reward_shaping.core.wrappers import TLRewardWrapper
        env = TLRewardWrapper(env, tl_conf=reward_conf, window_len=None, eval_at_end=True, semantics="stl")
    elif 'bhnr' in reward:
        from reward_shaping.core.wrappers import TLRewardWrapper
        window = int(env_params["max_steps"] // 20)
        env = TLRewardWrapper(env, tl_conf=reward_conf, window_len=window, eval_at_end=False, semantics="filtering")
    elif 'eval' in reward:
        from reward_shaping.core.wrappers import EvaluationRewardWrapper
        env = EvaluationRewardWrapper(env, conf=reward_conf)
    else:
        reward_fn = reward_conf
        env = RewardWrapper(env, reward_fn=reward_fn)
    return env


def make_observation_wrap(env_name, env, env_params={}):
    """ goal: filter and normalize observations """
    if env_name == "bipedal_walker":
        # in bipedal walker, the agent do not observe its position 'x'
        from reward_shaping.envs.wrappers import FilterObservationWrapper
        fields = [k for k in env.observation_space.spaces.keys() if k != "x"]
        env = FilterObservationWrapper(env, fields)
    if "racecar" in env_name:
        from reward_shaping.envs.wrappers import FilterObservationWrapper, NormalizeObservationWithMinMax, FrameSkip
        env = NormalizeObservationWithMinMax(env, {"lidar_64": (0.0, 15.0),  # norm lidar rays from 0, 15 meters
                                                   "velocity_x": (0.0, 3.5),  # norm velocity from 0, 3.5 m/s
                                                   "last_actions": (-1.0, 1.0)  # norm actions in +-1
                                                   })
        for obs in env_params["observation_config"]["obs_names"]:
            env = ObservationHistoryWrapper(env, obs_name=obs,
                                            n_last_observations=env_params["observation_config"]["n_last_observations"])
        fields = ["last_lidar_64", "last_velocity_x", "last_actions"]
        env = FilterObservationWrapper(env, fields)

    return env
