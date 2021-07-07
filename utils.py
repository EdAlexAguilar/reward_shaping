import pathlib
import time
import yaml

from envs.cart_pole.cp_continuousobstacle_env import CartPoleContObsEnv


def make_log_dirs(args):
    logdir_template = "logs/{}/{}_terminate{}_clip{}_unitScale{}_{}"
    logdir = pathlib.Path(logdir_template.format(args.task, args.reward, args.terminate_on_collision,
                                                 args.clip_reward, args.unit_scaling, int(time.time())))
    checkpointdir = logdir / "checkpoint"
    logdir.mkdir(parents=True, exist_ok=True)
    checkpointdir.mkdir(parents=True, exist_ok=True)
    # store input params
    with open(logdir / f"args.yml", "w") as file:
        yaml.dump(args, file)
    return logdir, checkpointdir


def make_base_env(task, env_params={}):
    if task in ["cart_pole"]:
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
    elif rl_algo == "sac":
        from stable_baselines3 import SAC
        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
    else:
        raise NotImplementedError()
    return model


def make_reward_wrap(task, env, reward, reward_params):
    if task == "cart_pole":
        if reward == "indicator":
            from envs.cart_pole.rewards.indicator_based import IndicatorWithContinuousTargetReward
            env = IndicatorWithContinuousTargetReward(env, **reward_params)
        elif reward == "indicator_sparse":
            from envs.cart_pole.rewards.indicator_based import IndicatorWithSparseTargetReward
            env = IndicatorWithSparseTargetReward(env, **reward_params)
        elif reward == "indicator_progress":
            from envs.cart_pole.rewards.indicator_based import IndicatorWithProgressTargetReward
            env = IndicatorWithProgressTargetReward(env, **reward_params)
        elif reward == "weighted":
            from envs.cart_pole.rewards.baselines import WeightedReward
            env = WeightedReward(env)
        elif reward == "sparse":
            from envs.cart_pole.rewards.baselines import SparseReward
            env = SparseReward(env)
    else:
        raise NotImplementedError()
    return env
