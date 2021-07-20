import pathlib
import time
import yaml
import os

def make_log_dirs(args):
    logdir_template = "logs/{}/{}_{}_{}"
    logdir = pathlib.Path(logdir_template.format(args.env, args.task, args.reward, int(time.time())))
    checkpointdir = logdir / "checkpoint"
    logdir.mkdir(parents=True, exist_ok=True)
    checkpointdir.mkdir(parents=True, exist_ok=True)
    # store input params
    with open(logdir / f"args.yml", "w") as file:
        yaml.dump(args, file)
    return logdir, checkpointdir


def make_base_env(env, env_params={}):
    if env == "cart_pole":
        from envs.cart_pole.cp_continuous_env import CartPoleContEnv
        env = CartPoleContEnv(**env_params)
    elif env == "cart_pole_obst":
        from envs.cart_pole_obst.cp_continuousobstacle_env import CartPoleContObsEnv
        env = CartPoleContObsEnv(**env_params)
    elif env == "bipedal_walker":
        from envs.bipedal_walker.bipedal_walker import BipedalWalker
        env = BipedalWalker(**env_params)
    else:
        raise NotImplementedError(f"not implemented env for {env}")
    return env


def make_env(env, task, logdir=None, **kwargs):
    env_config = pathlib.Path(f"{os.path.dirname(__file__)}/envs/{env}/tasks") / f"{task}.yml"
    if env_config.exists():
        with open(env_config, 'r') as file:
            env_params = yaml.load(file, yaml.FullLoader)
    else:
        env_params = {}
    # copy params in logdir (optional)
    if logdir:
        with open(logdir / f"{task}.yml", "w") as file:
            yaml.dump(env_params, file)
    # update params
    for key, value in kwargs.items():
        env_params[key] = value
    # make env
    env = make_base_env(env, env_params)
    return env, env_params


def make_agent(env_name, env, rl_algo, logdir=None):
    # load model parameters
    algo = rl_algo.split("_", 1)[0]
    algo_config = pathlib.Path(f"envs/{env_name}/hparams") / f"{algo}.yml"
    if algo_config.exists():
        with open(algo_config, 'r') as file:
            algo_params = yaml.load(file, yaml.FullLoader)
    else:
        algo_params = {}
    # create model
    if algo == "ppo":
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir, **algo_params)
    elif algo == "ppo_sde":
        from stable_baselines3 import PPO
        algo_params['use_sde'] = True
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


def make_reward_wrap(env_name, env, reward):
    if env_name == "cart_pole":
        from envs.cart_pole.rewards import get_reward
        env = get_reward(reward)(env)
    elif env_name == "cart_pole_obst":
        from envs.cart_pole_obst.rewards import get_reward
        env = get_reward(reward)(env)
    else:
        raise NotImplementedError()
    return env
