import gym

from callbacks import RobustMonitoringCallback, VideoRecorderCallback
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
import argparse as parser

from utils import make_agent, make_reward_wrap, make_log_dirs, make_env


def main(args):
    # create log
    if args.task in ['balance', 'target']:
        args.env = "cart_pole"
    else:
        args.env = "cart_pole_obst"
    args.terminate_on_collision = True
    logdir, checkpointdir = make_log_dirs(args)
    # create environment
    env, env_params = make_env(args.env, args.task, logdir)
    env = make_reward_wrap(args.env, env, args.reward)
    # create agent
    model = make_agent(args.env, env, args.algo, logdir)

    # prepare for training
    train_params = {'steps': args.steps, 'eval_every': int(args.steps / 10), 'rob_eval_every': 1000,
                    'checkpoint_every': int(args.steps / 10)}
    eval_env, _ = make_env(args.env, args.task, eval=True, logdir=None)
    eval_env = gym.wrappers.Monitor(env, logdir / "videos")
    video_cb = VideoRecorderCallback(eval_env, render_freq=train_params['eval_every'], n_eval_episodes=2)
    checkpoint_callback = CheckpointCallback(save_freq=train_params['checkpoint_every'], save_path=checkpointdir,
                                             name_prefix='model')
    monitoring_callback = EveryNTimesteps(n_steps=train_params['rob_eval_every'], callback=RobustMonitoringCallback())
    # train
    model.learn(total_timesteps=train_params['steps'], callback=[video_cb, checkpoint_callback, monitoring_callback])
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


if __name__ == "__main__":
    rewards = ['sparse', 'continuous', 'stl', 'cont_gh', 'cont_gh_pot', 'sdisc_gh', 'sdisc_gh_pot']
    tasks = ['balance', 'target', 'fixed_height', 'random_height']
    parser = parser.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=tasks)
    parser.add_argument("--reward", type=str, required=True, choices=rewards)
    parser.add_argument("--algo", type=str, required=True, choices=['ppo', 'ppo_sde', 'sac'])
    parser.add_argument("--steps", type=int, default=1e6)
    parser.add_argument("-clip_reward", action="store_true")
    parser.add_argument("-unit_scaling", action="store_true")
    args = parser.parse_args()
    main(args)
