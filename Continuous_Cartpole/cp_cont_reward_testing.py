import cp_continuous_env, cp_ppo_videomaker, cp_ppo_utils
import numpy as np


# Reward Specifications Constants
REWARD_THETA_BOUND = 24 # The cutoff for reward in degrees
REWARD_X_BOUND = 0.75 # The cutoff for reward in position deviation
# Episodic Constants
MAX_STEPS = 500
X_THRESHOLD = 2.5 # Value where episode ends
THETA_THRESHOLD = 30 # Value where episode ends
SOLVED_REWARD_BOUND = int(0.8*MAX_STEPS)

"""
Currently, reward functions are part of an environment.
So the way to define new functions is by defining them as methods within a new class.
Below are two examples
"""

def linear_decay(z, z0, scale, max_val=1, min_val=0):
    """
    Linear decay centered around z0
    """
    return max(min_val, max_val - abs((z-z0)/scale))

# Example 1
class EnvironmentWeightedReward(cp_continuous_env.CartPoleContEnv):
    def __init__(self, weights, x_threshold=X_THRESHOLD, theta_threshold_deg=THETA_THRESHOLD, max_steps=MAX_STEPS):
        super().__init__(x_threshold, theta_threshold_deg, max_steps)
        self.weights = weights # 3-dim normalized nparray

    def reward(self):
        x, x_dot, theta, theta_dot, battery = self.state
        r1 = linear_decay(theta, 0, REWARD_THETA_BOUND)
        r2 = linear_decay(x, 0, REWARD_X_BOUND)
        return self.weights[0]*r1 + self.weights[1]*r2 + self.weights[2]*battery

# Example 2
class EnvironmentIndicatorReward(cp_continuous_env.CartPoleContEnv):
    def __init__(self, indicator_tolerance, x_threshold=X_THRESHOLD, theta_threshold_deg=THETA_THRESHOLD, max_steps=MAX_STEPS):
        super().__init__(x_threshold, theta_threshold_deg, max_steps)
        self.indicator_tolerance = indicator_tolerance # np.array with angle tolerance and x tolernce
        self.safe_angle_tolerance = indicator_tolerance[0]*np.pi/360
        self.safe_x_tolerance = indicator_tolerance[1]

    def reward(self):
        x, x_dot, theta, theta_dot, battery = self.state
        r1 = linear_decay(theta, 0, REWARD_THETA_BOUND)
        r2 = linear_decay(x, 0, REWARD_X_BOUND)
        id1 = 1 if (abs(theta) < self.safe_angle_tolerance) else 0
        id2 = 1 if (abs(x) < self.safe_x_tolerance) else 0
        return (1/3)*(r1 + id1*r2 + id1*id2*battery)

if __name__ == "__main__":
    names = ['weighted_066','weighted_05','indicator']
    params = {}
    params[names[0]] = np.array([0.48, 0.313, 0.207]) # Each weight is 66% of last one
    params[names[1]] = np.array([0.575, 0.285, 0.14]) # Each weight is 50% of last one
    params[names[2]] = np.array([REWARD_THETA_BOUND, REWARD_X_BOUND]) # Indicator approach
    envs= {}
    envs[names[0]] = EnvironmentWeightedReward(params[names[0]])
    envs[names[1]] = EnvironmentWeightedReward(params[names[1]])
    envs[names[2]] = EnvironmentIndicatorReward(params[names[2]])

    for name in names:
        environment = envs[name]
        rewards = cp_ppo_utils.train_PPO(environment, name=name, solved_bound=SOLVED_REWARD_BOUND)
        np.savetxt(f'{name}_rewards.csv', rewards, delimiter=",")
