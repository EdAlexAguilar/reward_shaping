import gym

from callbacks import RobustMonitoringCallback, VideoRecorderCallback
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
import argparse as parser

from utils import make_agent, make_reward_wrap, make_log_dirs, make_env


def main(args):
    # create log
    args.env = "cart_pole"
    args.terminate_on_collision = True
    logdir, checkpointdir = make_log_dirs(args)
    # create environment
    env, env_params = make_env(args.env, args.task, logdir)
    reward_params = {'clip_to_positive': args.clip_reward, 'unit_scaling': args.unit_scaling}
    env = make_reward_wrap(args.env, env, args.reward, reward_params)
    # create agent
    model = make_agent(args.env, env, args.algo, logdir)

    # prepare for training
    train_params = {'steps': args.steps, 'eval_every': int(args.steps / 10), 'rob_eval_every': 1000,
                    'checkpoint_every': int(args.steps / 10)}
    eval_env = gym.wrappers.Monitor(env, logdir / "videos")
    video_cb = VideoRecorderCallback(eval_env, render_freq=train_params['eval_every'], n_eval_episodes=1)
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
    rewards = ['indicator', 'indicator_sparse', 'indicator_progress', 'weighted', 'sparse']
    tasks = ['random', 'no_random']
    parser = parser.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=tasks)
    parser.add_argument("--reward", type=str, required=True, choices=rewards)
    parser.add_argument("--algo", type=str, required=True, choices=['ppo', 'ppo_sde', 'sac'])
    parser.add_argument("--steps", type=int, default=1e6)
    parser.add_argument("-clip_reward", action="store_true")
    parser.add_argument("-unit_scaling", action="store_true")
    args = parser.parse_args()
    main(args)
