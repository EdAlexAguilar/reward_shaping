import pathlib
import time
from argparse import Namespace

import yaml
from gym.wrappers import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from .callbacks import VideoRecorderCallback
from .utils import make_env, make_agent


def make_log_dirs(args):
    logdir_template = "logs/{}/{}_{}_Seed{}_{}"
    logdir = pathlib.Path(logdir_template.format(args.env, args.task, args.reward, args.seed, int(time.time())))
    checkpointdir = logdir / "checkpoint"
    logdir.mkdir(parents=True, exist_ok=True)
    checkpointdir.mkdir(parents=True, exist_ok=True)
    # store input params
    with open(logdir / f"args.yml", "w") as file:
        yaml.dump(args, file)
    return logdir, checkpointdir


def get_callbacks(env, logdir, checkpointdir, train_params):
    video_env = Monitor(env, logdir / "videos")
    video_cb = VideoRecorderCallback(video_env, render_freq=train_params['video_every'],
                                     n_eval_episodes=train_params['n_recorded_episodes'])
    eval_cb = EvalCallback(env, eval_freq=train_params['eval_every'],
                           n_eval_episodes=train_params['n_eval_episodes'],
                           deterministic=True, render=False)
    checkpoint_cb = CheckpointCallback(save_freq=train_params['checkpoint_every'], save_path=checkpointdir,
                                       name_prefix='model')
    return [video_cb, eval_cb, checkpoint_cb]


def evaluate(env, agent, steps):
    obs = env.reset()
    env.render()
    rewards = []
    tot_reward = 0.0
    for i in range(steps):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        tot_reward += reward
        env.render()
        if done:
            rewards.append(tot_reward)
            obs = env.reset()
            tot_reward = 0.0
    print(f"[Rollout {steps} steps] Result: episodes: {len(rewards)}, mean reward: {sum(rewards) / len(rewards)}")


def train(env, task, reward, train_params, algo="sac", seed=0):
    # logs
    args = Namespace(env=env, task=task, reward=reward, algo=algo, seed=seed)
    logdir, checkpointdir = make_log_dirs(args)
    # prepare envs
    train_env, trainenv_params = make_env(env, task, reward, eval=False, logdir=logdir, seed=seed)
    eval_env, evalenv_params = make_env(env, task, reward="stl", eval=True, seed=seed)
    # create agent
    model = make_agent(env, train_env, algo, logdir)
    # train
    callbacks = get_callbacks(eval_env, logdir, checkpointdir, train_params)
    model.learn(total_timesteps=train_params['steps'], callback=callbacks)
    # evaluation
    evaluate(eval_env, model, steps=500)
    # close envs
    train_env.close()
    eval_env.close()
