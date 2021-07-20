import gym

from callbacks import RobustMonitoringCallback, VideoRecorderCallback
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
import argparse as parser

from utils import make_agent, make_reward_wrap, make_log_dirs, make_env


def main(args):
    # task selection
    # tasks 'balance', 'target': continuous cartpole env, with resp. goal of balancing and reach target while balancing
    # task 'fixed_height', 'random_height': continuous cartpole with obstacle placed at fixed of random height
    if args.task in ['balance', 'target']:
        args.env = "cart_pole"
    else:
        args.env = "cart_pole_obst"

    args.terminate_on_collision = True
    logdir, checkpointdir = make_log_dirs(args)
    # create environments
    train_env, env_params = make_env(args.env, args.task, logdir)
    train_env = make_reward_wrap(args.env, train_env, args.reward)
    eval_feas_env, _ = make_env(args.env, args.task, prob_sampling_feasible=1.0)
    eval_nfeas_env, _ = make_env(args.env, args.task, prob_sampling_feasible=0.0)
    eval_feas_env = gym.wrappers.Monitor(eval_feas_env, logdir / "videos")
    eval_nfeas_env = gym.wrappers.Monitor(eval_nfeas_env, logdir / "videos")
    # create agent
    model = make_agent(args.env, train_env, args.algo, logdir)

    # prepare for training
    train_params = {'steps': args.steps, 'eval_every': int(args.steps / 10), 'rob_eval_every': 1000,
                    'checkpoint_every': int(args.steps / 10), 'n_eval_episodes': 2}
    # callbacks
    video_feas_cb = VideoRecorderCallback(eval_feas_env, render_freq=train_params['eval_every'],
                                          n_eval_episodes=train_params['n_eval_episodes'])
    video_nfeas_cb = VideoRecorderCallback(eval_nfeas_env, render_freq=train_params['eval_every'],
                                           n_eval_episodes=train_params['n_eval_episodes'])
    checkpoint_callback = CheckpointCallback(save_freq=train_params['checkpoint_every'], save_path=checkpointdir,
                                             name_prefix='model')
    monitoring_callback = EveryNTimesteps(n_steps=train_params['rob_eval_every'], callback=RobustMonitoringCallback())
    # train
    model.learn(total_timesteps=train_params['steps'],
                callback=[video_feas_cb, video_nfeas_cb, checkpoint_callback, monitoring_callback])
    # evaluation
    for env in [eval_feas_env, eval_nfeas_env]:
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
    args = parser.parse_args()
    main(args)
