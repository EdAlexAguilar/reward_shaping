import gym
from gym.wrappers import Monitor

from callbacks import RobustMonitoringCallback, VideoRecorderCallback
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps, EvalCallback
import argparse as parser

from hierarchy.graph_hierarchical_reward import HierarchicalGraphRewardWrapper
from utils import make_agent, make_reward_wrap, make_log_dirs, make_env


def train(args):
    # task selection
    # tasks 'balance', 'target': continuous cartpole env, with resp. goal of balancing and reach target while balancing
    # task 'fixed_height', 'random_height': continuous cartpole with obstacle placed at fixed of random height
    if args.task in ['balance', 'target']:
        args.env = "cart_pole"
    else:
        args.env = "cart_pole_obst"
    logdir, checkpointdir = make_log_dirs(args)

    # create training environment
    train_env, env_params = make_env(args.env, args.task, logdir, seed=args.seed)
    train_env = make_reward_wrap(args.env, train_env, args.reward)
    if isinstance(train_env, HierarchicalGraphRewardWrapper):
        import matplotlib.pyplot as plt
        train_env.hierarchy.render()
        plt.savefig("hierarchy.pdf")

    # create eval environments
    if args.task == 'random_height':
        # if conditional environment, create 2 distinct eval envs
        eval_env, _ = make_env(args.env, args.task, eval=True, prob_sampling_feasible=0.5, seed=args.seed)
        eval_envs = [eval_env]
    else:
        # if normal environment without conditions, then eval env is the train env
        eval_env, _ = make_env(args.env, args.task, logdir=None, eval=True, seed=args.seed)
        eval_envs = [eval_env]

    # create agent
    model = make_agent(args.env, train_env, args.algo, logdir)

    # prepare for training
    train_params = {'steps': args.steps, 'video_every': int(args.steps / 10), 'n_recorded_episodes': 5,
                    'eval_every': min(10000, int(args.steps / 10)), 'n_eval_episodes': 5,
                    'checkpoint_every': int(args.steps / 10)}
    # callbacks
    callbacks_list = []
    for eval_env in eval_envs:
        video_env = Monitor(make_reward_wrap(args.env, eval_env, args.reward), logdir / "videos")
        video_cb = VideoRecorderCallback(video_env, render_freq=train_params['video_every'],
                                         n_eval_episodes=train_params['n_recorded_episodes'])
        callbacks_list.append(video_cb)
        stl_env = make_reward_wrap(args.env, eval_env, 'bool_stl')
        eval_cb = EvalCallback(stl_env, eval_freq=train_params['eval_every'],
                               n_eval_episodes=train_params['n_eval_episodes'],
                               deterministic=True, render=False)
        callbacks_list.append(eval_cb)
    checkpoint_callback = CheckpointCallback(save_freq=train_params['checkpoint_every'], save_path=checkpointdir,
                                             name_prefix='model')
    callbacks_list.append(checkpoint_callback)

    # train
    model.learn(total_timesteps=train_params['steps'],
                callback=callbacks_list)
    # evaluation
    for env in eval_envs:
        obs = env.reset()
        env.render()
        rewards = []
        tot_reward = 0.0
        for i in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            tot_reward += reward
            env.render()
            if done:
                rewards.append(tot_reward)
                obs = env.reset()
                tot_reward = 0.0
        env.close()


if __name__ == "__main__":
    tasks = ['balance', 'target', 'fixed_height', 'random_height']
    parser = parser.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=tasks)
    parser.add_argument("--reward", type=str, required=True)
    parser.add_argument("--algo", type=str, required=True, choices=['ppo', 'ppo_sde', 'sac'])
    parser.add_argument("--steps", type=int, default=1e6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args)
