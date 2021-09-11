import numpy as np

from reward_shaping.core.configs import BuildGraphReward
from reward_shaping.envs.cart_pole_obst.cp_continuousobstacle_env import Obstacle
from reward_shaping.training.utils import load_env_params, get_reward_conf

xs = np.linspace(-3.0, 3.0)
thetas = np.linspace(-np.pi / 2 - 0.1, np.pi / 2 + 0.1)
dist_obs = np.linspace(0.0, 2.5)

env_name, task = 'cart_pole_obst', 'fixed_height'
env_params = load_env_params(env_name, task)
env_params.update({'polelen': 1.0, 'axle_y': 1.0 + 0.25 / 4, 'dist_to_ground': 0.95})
obstacle = Obstacle(env_params['axle_y'], env_params['polelen'],
                    0.5, env_params['axle_y'] + env_params['dist_to_ground'], 0.2, 0.1)
info = {'time': 0, 'tau': 0.02,
        'x_limit': env_params['x_limit'], 'theta_limit': np.deg2rad(env_params['theta_limit']),
        'x_target': env_params['x_target'], 'x_target_tol': env_params['x_target_tol'],
        'theta_target': env_params['theta_target'], 'theta_target_tol': env_params['theta_target_tol'],
        'pole_length': env_params['polelen'], 'axle_y': env_params['axle_y'],
        }

state = {
    "x": 0.0, "x_vel": 0.0, "theta": 0.0, "theta_vel": 0.0, "battery": 0.0,
    "obstacle_left": obstacle.left_x, "obstacle_right": obstacle.right_x,
    "obstacle_bottom": obstacle.bottom_y, "obstacle_top": obstacle.top_y,
}
next_state = {
    "x": 0.0, "x_vel": 0.0, "theta": 0.0, "theta_vel": 0.0, "battery": 0.0,
    "obstacle_left": obstacle.left_x, "obstacle_right": obstacle.right_x,
    "obstacle_bottom": obstacle.bottom_y, "obstacle_top": obstacle.top_y,
}


def get_gb_reward(reward):
    # make env
    reward_conf = get_reward_conf(env_name, env_params, reward)
    reward_fn = BuildGraphReward.from_conf(graph_config=reward_conf)
    return reward_fn


def plot_safety_nodes(reward_fn):
    # S coll: loop over dist to obstacle
    node = "S_coll"
    nodes = reward_fn._graph.nodes
    discretization = 0.01
    bins = np.arange(0.0, 2.5, discretization)
    obst_dist_reward_bins = np.zeros(len(bins))
    obst_dist_sat_bins = np.zeros(len(bins))
    obst_dist_k_bins = np.zeros(len(bins))
    n_bins = len(obst_dist_k_bins)
    for i in range(len(xs)):
        for j in range(len(thetas)):
            x, theta = xs[i], thetas[j]
            info['collision'] = obstacle.intersect(x, theta)
            state['x'] = x
            state['theta'] = theta
            next_state['x'] = x
            next_state['theta'] = theta
            next_state['collision'] = 1.0 if info['collision'] else 0.0
            _ = reward_fn(state, None, next_state, info)  # force graph evaluation
            # update structs
            obst_dist = obstacle.get_pole_dist(x, theta)
            id = np.clip(obst_dist // discretization, 0, n_bins - 1)
            obst_dist_reward_bins[int(id)] += nodes[node]['reward']
            obst_dist_sat_bins[int(id)] += nodes[node]['sat']
            obst_dist_k_bins[int(id)] += 1
    obst_dist_reward_bins = obst_dist_reward_bins / obst_dist_k_bins
    obst_dist_sat_bins = obst_dist_sat_bins / obst_dist_k_bins
    import matplotlib.pyplot as plt
    plt.plot(bins, obst_dist_reward_bins, label="reward")
    plt.plot(bins, obst_dist_sat_bins, label="sat")
    plt.xlabel("Distance to Obstacle (|pole - obst|)")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def plot_score_sat_curves(reward):
    assert reward in ['gb_pcr_bi', 'gb_bpr_ci']
    reward_fn = get_gb_reward(reward)
    plot_safety_nodes(reward_fn)


if __name__ == "__main__":
    reward = "gb_bpr_ci"
    plot_score_sat_curves(reward)
