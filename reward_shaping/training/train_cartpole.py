from gym.wrappers import Monitor, FlattenObservation

from callbacks import VideoRecorderCallback
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import argparse as parser

from utils import make_agent, make_reward_wrap, make_log_dirs, make_env


def rollout(env, model, steps=500):
    obs = env.reset()
    env.render()
    rewards = []
    tot_reward = 0.0
    for i in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        tot_reward += reward
        env.render()
        if done:
            rewards.append(tot_reward)
            obs = env.reset()
            tot_reward = 0.0
    return rewards


def prepare_callbacks(env, env_name, env_params, logdir, checkpointdir, train_params):
    video_env = Monitor(env, logdir / "videos")
    video_cb = VideoRecorderCallback(video_env, render_freq=train_params['video_every'],
                                     n_eval_episodes=train_params['n_recorded_episodes'])
    stl_env = make_reward_wrap(env_name, env, env_params=env_params, reward='bool_stl')
    eval_cb = EvalCallback(env, eval_freq=train_params['eval_every'],
                           n_eval_episodes=train_params['n_eval_episodes'],
                           deterministic=True, render=False)
    checkpoint_cb = CheckpointCallback(save_freq=train_params['checkpoint_every'], save_path=checkpointdir,
                                       name_prefix='model')
    return [video_cb, eval_cb, checkpoint_cb]


def train(args):
    # task selection
    # tasks 'balance', 'target': continuous cartpole env, with resp. goal of balancing and reach target while balancing
    # task 'fixed_height', 'random_height': continuous cartpole with obstacle placed at fixed of random height
    if args.task in ['balance', 'target']:
        args.env = "cart_pole"
        raise DeprecationWarning("simple cartpole still need to be merged with the last changes ")
    else:
        args.env = "cart_pole_obst"
    logdir, checkpointdir = make_log_dirs(args)

    # create training environment
    train_env, env_params = make_env(args.env, args.task, logdir, seed=args.seed)
    train_env = make_reward_wrap(args.env, train_env, env_params, args.reward)
    train_env = FlattenObservation(train_env)

    # create eval environments
    if args.task == 'random_height':
        # if conditional environment, create 2 distinct eval envs
        eval_env, eval_env_params = make_env(args.env, args.task, eval=True, prob_sampling_feasible=0.5, seed=args.seed)
    else:
        # if normal environment without conditions, then eval env is the train env
        eval_env, eval_env_params = make_env(args.env, args.task, eval=True, seed=args.seed)
    eval_env = make_reward_wrap(args.env, eval_env, eval_env_params, 'bool_stl')
    eval_env = FlattenObservation(eval_env)

    # create agent
    model = make_agent(args.env, train_env, args.algo, logdir)

    # prepare for training
    train_params = {'steps': args.steps, 'video_every': int(args.steps / 10), 'n_recorded_episodes': 5,
                    'eval_every': min(10000, int(args.steps / 10)), 'n_eval_episodes': 5,
                    'checkpoint_every': int(args.steps / 10)}
    # callbacks
    callbacks_list = prepare_callbacks(eval_env, args.env, eval_env_params, logdir, checkpointdir, train_params)

    # train
    model.learn(total_timesteps=train_params['steps'], callback=callbacks_list)
    # evaluation
    steps = 500
    rewards = rollout(eval_env, model, steps=steps)
    print(
        f"\n\n[Final Eval] Result: steps: {steps}, episodes: {len(rewards)}, mean reward: {sum(rewards) / len(rewards)}")
    # close envs
    train_env.close()
    eval_env.close()


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
