import gym

import callbacks
from stable_baselines3.common.callbacks import CheckpointCallback
import argparse as parser

# define problem
from utils import make_agent, make_reward_wrap, make_log_dirs, make_env


def main(args):
    # config logs
    args.task = "cart_pole"
    logdir, checkpointdir = make_log_dirs(args)
    # prepare env with custom reward
    env, env_params = make_env(args.task, args.terminate_on_collision, logdir)
    reward_params = {'clip_reward': args.clip_reward, 'shift_reward': args.shift_reward,
                     'unit_scaling': args.unit_scaling}
    env = make_reward_wrap(args.task, env, args.reward, env_params, reward_params)
    # make rl agent
    model = make_agent(env, args.algo, logdir)
    # training
    training_params = {'steps': args.steps, 'eval_every': 1e5, 'checkpoint_every': 5e5}
    eval_env = gym.wrappers.Monitor(env, logdir / "videos")
    video_cb = callbacks.VideoRecorderCallback(eval_env, render_freq=training_params['eval_every'], n_eval_episodes=1)
    checkpoint_callback = CheckpointCallback(save_freq=training_params['checkpoint_every'], save_path=checkpointdir,
                                             name_prefix='model')
    model.learn(total_timesteps=training_params['steps'], callback=[video_cb, checkpoint_callback])
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
    parser = parser.ArgumentParser()
    parser.add_argument("--reward", type=str, required=True, choices=['indicator', 'indicator_reverse'])
    parser.add_argument("--algo", type=str, required=True, choices=['ppo', 'ppo_sde'])
    parser.add_argument("--steps", type=int, default=1e6)
    parser.add_argument("-terminate_on_collision", action="store_true")
    parser.add_argument("-shift_reward", action="store_true")
    parser.add_argument("-clip_reward", action="store_true")
    parser.add_argument("-unit_scaling", action="store_true")
    args = parser.parse_args()
    main(args)
