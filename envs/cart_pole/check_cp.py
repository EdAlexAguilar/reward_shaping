import pathlib

import yaml
from stable_baselines3.common.env_checker import check_env

# define problem
from envs.cart_pole.cp_continuous_env import CartPoleContEnv
from envs.cart_pole.rewards import get_reward
from envs.cart_pole_obst.rewards.baselines import SparseNoFalldownReward


def main(reward):
    task = "target"
    env_config = pathlib.Path(f"tasks/{task}.yml")
    with open(env_config, 'r') as file:
        env_params = yaml.load(file, yaml.FullLoader)
    env = CartPoleContEnv(**env_params, eval=True)
    env = get_reward(reward)(env)
    """
    if reward == "indicator":
        from envs.cart_pole_obst.rewards import IndicatorWithContinuousTargetReward
        env = IndicatorWithContinuousTargetReward(env)
    elif reward == "indicator_sparse":
        from envs.cart_pole_obst.rewards import IndicatorWithSparseTargetReward
        env = IndicatorWithSparseTargetReward(env)
    elif reward == "indicator_progress":
        from envs.cart_pole_obst.rewards import IndicatorWithProgressTargetReward
        env = IndicatorWithProgressTargetReward(env)
    """

    # evaluation
    obs = env.reset()
    env.render()
    rewards = []
    tot_reward = 0.0
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        tot_reward += reward
        env.render()
        if done:
            rewards.append(tot_reward)
            obs = env.reset()
            rob = env.compute_episode_robustness(env.last_complete_episode)
            print(f"reward: {tot_reward:.3f}, robustness: {rob:.3f}")
            tot_reward = 0.0
            input()
    try:
        check_env(env)
        result = True
    except Exception as err:
        result = False
        print(err)
    print(f"Check env: {result}")

    env.close()


if __name__ == "__main__":
    rewards = ['indicator', 'indicator_sparse', 'indicator_progress', 'continuous', 'sparse', 'sparse_nofall',
               'stl']
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", choices=rewards, default="indicator")
    args = parser.parse_args()

    main(args.reward)
