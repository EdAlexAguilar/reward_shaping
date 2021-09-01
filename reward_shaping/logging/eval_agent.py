import argparse
import pathlib

from stable_baselines3 import SAC

from reward_shaping.training.utils import make_env


def evaluate_model(model, env, n_episodes=None, n_steps=None, render=True):
    assert not n_episodes or not n_steps
    assert not n_steps or not n_episodes
    obs = env.reset()
    tot_reward = 0.0
    episodes, steps = 0, 0
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        tot_reward += reward
        steps += 1
        if render:
            env.render()
        if done:
            print(f"tot reward: {tot_reward}")
            obs = env.reset()
            episodes += 1
            tot_reward = 0.0
            if n_episodes and episodes >= n_episodes:
                break
            if n_steps and steps >= n_steps:
                break


def main(args):
    env_name, task, reward = args.env, args.task, args.eval_reward
    cp_filepath = args.checkpoint
    eval_episodes, no_render = args.eval_episodes, args.no_render
    # make env
    env, env_params = make_env(env_name, task, reward)
    # resume agent
    model = SAC.load(cp_filepath)
    #
    evaluate_model(model, env, n_episodes=eval_episodes, render=not no_render)

def get_default_task(env):
    if env == "cart_pole_obst":
        return 'fixed_height'
    if env=="bipedal_walker":
        return 'forward'
    if env=="lunar_lander":
        return 'land'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, choices=['cart_pole_obst', 'bipedal_walker', 'lunar_lander'])
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--eval_reward", type=str, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--eval_episodes", type=int, required=False, default=10)
    parser.add_argument("-no_render", action='store_true')
    # parse args
    args = parser.parse_args()
    args.task = args.task if args.task is not None else get_default_task(args.env)
    main(args)
