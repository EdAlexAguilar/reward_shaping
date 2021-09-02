import argparse

import numpy as np
from stable_baselines3.common.env_checker import check_env

from reward_shaping.core.wrappers import RewardWrapper
from reward_shaping.envs.cart_pole_obst.cp_continuousobstacle_env import Obstacle
from reward_shaping.training.utils import make_env, make_agent, load_env_params, make_base_env, get_reward_conf


def generic_env_test(env_name, task, reward_name, potential=False):
    seed = np.random.randint(0, 1000000)
    env, env_params = make_env(env_name, task, reward_name, use_potential=potential, eval=True, logdir=None, seed=seed)
    # check
    check_env(env)
    # evaluation
    for _ in range(1):
        _ = env.reset()
        env.render()
        tot_reward = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(reward)
            tot_reward += reward
            env.render()
        print(f"[{reward_name}] tot reward: {tot_reward:.3f}")
    env.close()
    return True


def generic_training(env, task, reward):
    # create training environment
    seed = np.random.randint(0, 1000000)
    train_env, env_params = make_env(env, task, reward, seed=seed)
    # create agent
    model = make_agent(env, train_env, reward, "sac", logdir=None)
    # train
    model.learn(total_timesteps=500)
    train_env.close()


def plot_cpole_reward(reward):
    assert 'stl' not in reward
    # make env
    env_name, task = 'cart_pole_obst', 'fixed_height'
    env_params = load_env_params(env_name, task)
    env = make_base_env(env_name, env_params)
    reward_conf = get_reward_conf(env_name, env_params, reward)
    if 'gb' in reward:
        from reward_shaping.core.configs import BuildGraphReward
        reward_fn = BuildGraphReward.from_conf(graph_config=reward_conf)
    else:
        reward_fn = reward_conf
    env = RewardWrapper(env, reward_fn=reward_fn)
    # crate state space
    xs = np.linspace(-env_params['x_limit'] - 0.5, env_params['x_limit'] + 0.5)
    thetas = np.linspace(-np.deg2rad(env_params['theta_limit']) - 0.1, np.deg2rad(env_params['theta_limit']) + 0.1)
    axle_y = 1.0 + 0.25 / 4
    polelen=1.0
    dist_to_ground = 0.95
    obstacle = Obstacle(axle_y, polelen, 0.5, axle_y+dist_to_ground, 0.2, 0.1)
    x_vel, theta_vel, battery = 0.0, 0.0, 1.0     # ignore them, not used in reward

    reward_landscape = np.zeros((len(xs), len(thetas)))
    for i in range(len(xs)):
        for j in range(len(thetas)):
            x, theta = xs[i], thetas[j]
            collision = obstacle.intersect(x, theta)
            outside = x > env_params['x_limit']
            falldown = theta > np.deg2rad(env_params['theta_limit'])
            state = {
                "x":x, "x_vel": x_vel, "theta": theta, "theta_vel": theta_vel, "battery": battery,
                "obstacle_left": obstacle.left_x, "obstacle_right": obstacle.right_x,
                "obstacle_bottom": obstacle.bottom_y, "obstacle_top": obstacle.top_y,
                "collision": 1.0 if collision else 0.0
            }
            next_state = state
            info = {'time': 0,
                    'x_limit': 2.5, 'theta_limit': np.deg2rad(env_params['theta_limit']),
                    'x_target': env_params['x_target'], 'x_target_tol': env_params['x_target_tol'],
                    'theta_target': env_params['theta_target'], 'theta_target_tol': env_params['theta_target_tol'],
                    'pole_length': polelen, 'axle_y': axle_y,
                    'is_feasible': True, 'feasible_height': dist_to_ground,
                    'collision': collision, 'outside': outside, 'falldown': falldown,
                    'default_reward': 0.0}
            reward_landscape[i, j] = reward_fn(state=state, action=None, next_state=next_state, info=info)
    import matplotlib.pyplot as plt
    plt.title(reward)
    plt.imshow(reward_landscape)
    plt.xlabel("Theta")
    plt.ylabel("X")
    plt.colorbar()
    plt.xticks()


def plot_cpole_progreward(reward, constant_progress=0.001):
    assert reward in ['gb_pcr_bi', 'gb_bpr_ci']
    # make env
    env_name, task = 'cart_pole_obst', 'fixed_height'
    env_params = load_env_params(env_name, task)
    env = make_base_env(env_name, env_params)
    reward_conf = get_reward_conf(env_name, env_params, reward)
    if 'gb' in reward:
        from reward_shaping.core.configs import BuildGraphReward
        reward_fn = BuildGraphReward.from_conf(graph_config=reward_conf)
    else:
        reward_fn = reward_conf
    env = RewardWrapper(env, reward_fn=reward_fn)
    # crate state space
    xs = np.linspace(-env_params['x_limit'] - 0.5, env_params['x_limit'] + 0.5)
    thetas = np.linspace(-np.deg2rad(env_params['theta_limit']) - 0.1, np.deg2rad(env_params['theta_limit']) + 0.1)
    axle_y = 1.0 + 0.25 / 4
    polelen=1.0
    dist_to_ground = 0.95
    obstacle = Obstacle(axle_y, polelen, 0.5, axle_y+dist_to_ground, 0.2, 0.1)
    x_vel, theta_vel, battery = 0.0, 0.0, 1.0     # ignore them, not used in reward

    reward_landscape = np.zeros((len(xs), len(thetas)))
    for i in range(len(xs)):
        for j in range(len(thetas)):
            x, theta = xs[i], thetas[j]
            collision = obstacle.intersect(x, theta)
            outside = x > env_params['x_limit']
            falldown = theta > np.deg2rad(env_params['theta_limit'])
            state = {
                "x":x, "x_vel": x_vel, "theta": theta, "theta_vel": theta_vel, "battery": battery,
                "obstacle_left": obstacle.left_x, "obstacle_right": obstacle.right_x,
                "obstacle_bottom": obstacle.bottom_y, "obstacle_top": obstacle.top_y,
                "collision": 1.0 if collision else 0.0
            }
            next_state = state.copy()
            next_state['x'] += np.sign(env_params['x_target'] - state['x']) * constant_progress
            info = {'time': 0, 'tau': 0.02,
                    'x_limit': 2.5, 'theta_limit': np.deg2rad(env_params['theta_limit']),
                    'x_target': env_params['x_target'], 'x_target_tol': env_params['x_target_tol'],
                    'theta_target': env_params['theta_target'], 'theta_target_tol': env_params['theta_target_tol'],
                    'pole_length': polelen, 'axle_y': axle_y,
                    'is_feasible': True, 'feasible_height': dist_to_ground,
                    'collision': collision, 'outside': outside, 'falldown': falldown,
                    'default_reward': 0.0}
            reward_landscape[i, j] = reward_fn(state=state, action=None, next_state=next_state, info=info)
    print(f"Reward {reward}: min: {np.min(reward_landscape)}, max: {np.max(reward_landscape)}")
    import matplotlib.pyplot as plt
    plt.title(f"{reward} - Prog={constant_progress}")
    plt.imshow(reward_landscape)
    plt.xlabel("Theta")
    plt.ylabel("X")
    plt.colorbar()
    plt.xticks()
