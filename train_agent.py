import cp_continuousobstacle_env
import ppo_mp as ppo
import numpy as np
import multiprocessing as mp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import tensorflow as tf  # to avoid TF import verbosity

# Set TF to work on CPU
tf.config.set_visible_devices([], 'GPU')
# Number of Cores available
CORES = mp.cpu_count()


# Environment Constants
# ------------------
# Reward Specifications Constants
REWARD_THETA_BOUND = 24 # in degrees
REWARD_X_BOUND = 0.1
REWARD_TARGET_XDOT = -1
# Episodic Constants
MAX_STEPS = 400
X_THRESHOLD = 2.2
THETA_THRESHOLD = 90

X_TOLERANCE = 0.07


class EnvironmentIndicatorReward(cp_continuousobstacle_env.CartPoleContObsEnv):
    def __init__(self, indicator_tolerance, x_threshold=X_THRESHOLD, theta_threshold_deg=THETA_THRESHOLD, max_steps=MAX_STEPS):
        super().__init__(x_threshold, theta_threshold_deg, max_steps)
        self.indicator_tolerance = indicator_tolerance # np.array with angle tolerance and x tolerance
        self.safe_angle_tolerance = indicator_tolerance[0]*np.pi/360
        self.safe_x_tolerance = indicator_tolerance[1]

    def reward(self):
        x, x_dot, theta, theta_dot, battery, obs_l, obs_r, obs_h = self.state
        safe_distance_closest = np.sqrt(obs_h * (2 - obs_h)) + X_TOLERANCE
        obstacle_side = 1 if np.sign(obs_l) == -1 else 0
        if x >= (obs_l - obstacle_side * safe_distance_closest) and x <= (obs_r + (1 - obstacle_side) * safe_distance_closest):
            r1 = ppo.linear_decay(theta, 0, REWARD_THETA_BOUND * 3)
            id1 = 1
        else:
            r1 = ppo.linear_decay(theta, 0, REWARD_THETA_BOUND)
            id1 = 1 if (abs(theta) < self.safe_angle_tolerance) else 0
        if abs(x)<REWARD_X_BOUND:
            r2 = ppo.linear_decay(x, 0, REWARD_X_BOUND)
            id2 = 1
        else:
            r2 = ppo.linear_decay(np.sign(x)*x_dot, REWARD_TARGET_XDOT, 1) # max(0, -np.sign(x)*x_dot)
            id2 = 1 if (np.sign(x)*x_dot < 0) else 0
        return (1/3)*(r1 + id1*r2 + id1*id2*battery)

    def set_obstacle_width_height(self, width, height):
        self.obstacle_max_width = width
        self.obstacle_max_height = height

if __name__ == "__main__":
    names = ['indicator_narrow_1']
    params = {names[0]: np.array([REWARD_THETA_BOUND, REWARD_X_BOUND])}
    envs= {names[0]: EnvironmentIndicatorReward(params[names[0]])}

    NUM_LEVELS = 1 # Number of training levels - in increasing difficulty
    # obstacle_widths = np.linspace(0.01, 0.08, num=NUM_LEVELS)
    # obstacle_heights = np.linspace(0.005, 0.1, num=NUM_LEVELS)
    obstacle_widths = [0.08]
    obstacle_heights = [0.1]
    for env_name in names:
        seed = 20210400  # April 2021
        for level in range(NUM_LEVELS):
            env = envs[env_name]
            env.set_obstacle_width_height(obstacle_widths[level], obstacle_heights[level])
            processes = []
            for ii in range(CORES):
                keyw = {'name': f'{env_name}_L{str(level)}_{ii}','seed': seed+ii, 'load_path': None}
                if level!=0:
                    keyw['load_path'] = f'{env_name}_L{str(level-1)}_{ii}'
                p = mp.Process(target=ppo.train_PPO, args=(env, ), kwargs=keyw)
                p.start()
                processes.append(p)
            for proc in processes:
                proc.join()