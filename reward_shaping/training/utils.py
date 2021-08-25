import pathlib

import matplotlib.pyplot as plt
import yaml
import os

from gym.wrappers import FlattenObservation

from reward_shaping.core.helper_fns import PotentialReward
from reward_shaping.core.wrappers import RewardWrapper


def make_env(env_name, task, reward, use_potential=False, eval=False, logdir=None, seed=0):
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
    env = make_reward_wrap(env_name, env, env_params, reward, use_potential=use_potential)
    env = FlattenObservation(env)
    return env, env_params


def load_env_params(env, task, **kwargs):
    try:
        config = pathlib.Path(f"{os.path.dirname(__file__)}/../envs/{env}/tasks") / f"{task}.yml"
        with open(config, 'r') as file:
            params = yaml.load(file, yaml.FullLoader)
    except FileNotFoundError as error:
        params = {'task': task}
    # update params
    for key, value in kwargs.items():
        params[key] = value
    return params


def load_eval_params(env, task):
    if env == "cart_pole_obst" and task == "random_height":
        params = {"eval": True, "prob_sampling_feasible": 0.5}
    else:
        params = {"eval": True}
    return params


def make_base_env(env, env_params={}):
    if env == "cart_pole_obst":
        from reward_shaping.envs import CartPoleContObsEnv
        env = CartPoleContObsEnv(**env_params)
    elif env == "bipedal_walker":
        from reward_shaping.envs import BipedalWalker
        env = BipedalWalker(**env_params)
    else:
        raise NotImplementedError(f"not implemented env for {env}")
    return env


def make_agent(env_name, env, rl_algo, logdir=None):
    # load model parameters
    algo = rl_algo.split("_", 1)[0]
    algo_config = pathlib.Path(f"{os.path.dirname(__file__)}/../envs/{env_name}/hparams") / f"{algo}.yml"
    if algo_config.exists():
        with open(algo_config, 'r') as file:
            algo_params = yaml.load(file, yaml.FullLoader)
    else:
        algo_params = {}
    # create model
    if algo == "ppo":
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir, **algo_params)
    elif algo == "sac":
        from stable_baselines3 import SAC
        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=logdir, **algo_params)
    else:
        raise NotImplementedError()
    # copy params in logdir (optional)
    if logdir:
        with open(logdir / f"{rl_algo}.yml", "w") as file:
            yaml.dump(algo_params, file)
    return model


def get_reward_conf(env_name, env_params, reward):
    if env_name == "cart_pole":
        # env = get_reward(reward)()
        raise DeprecationWarning("this env is not updated")
    elif env_name == "cart_pole_obst":
        from reward_shaping.envs.cart_pole_obst import get_reward
        reward_conf = get_reward(reward)(env_params=env_params)
    elif env_name == "bipedal_walker":
        from reward_shaping.envs.bipedal_walker import get_reward
        reward_conf = get_reward(reward)(env_params=env_params)
    else:
        raise NotImplementedError(f'{reward} not implemented for {env_name}')
    return reward_conf


def make_reward_wrap(env_name, env, env_params, reward, use_potential=False, logdir=None):
    reward_conf = get_reward_conf(env_name, env_params, reward)
    if 'stl' in reward:
        assert not use_potential, 'potential function not support for stl reward'
        from reward_shaping.core.wrappers import STLRewardWrapper
        env = STLRewardWrapper(env, stl_conf=reward_conf)
    else:
        if 'gb' in reward:
            from reward_shaping.core.configs import BuildGraphReward
            reward_fn = BuildGraphReward.from_conf(graph_config=reward_conf)
            reward_fn.render()
            if logdir is not None:
                plt.savefig(logdir / "graph_reward.pdf")
            else:
                plt.show()
        else:
            reward_fn = reward_conf
        if use_potential:
            reward_fn = PotentialReward(reward_fn)
        env = RewardWrapper(env, reward_fn=reward_fn)
    return env
