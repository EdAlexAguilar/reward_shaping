import gym
import numpy as np

from envs.cart_pole_obst.rewards.subtask_rewards import CollisionReward, ReachTargetReward, SparseReachTargetReward, \
    BalanceReward, NormalizedReward, FalldownReward


class ContinuousReward(gym.RewardWrapper):
    """
    reward(s,a) := - dist_target + dist_obst
    """

    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        # note: the need of this overwriting fo rew/ret is purely for rendering purposes
        # in this way, the env.render method with render the correct reward
        self.rew = self.reward_in_state(self.state)
        self.ret += self.rew
        self.env.rew = self.rew
        self.env.ret = self.ret
        return self.rew

    def reward_in_state(self, state):
        x, theta = state[0], state[2]
        pole_x = x + self.env.pole_length * np.sin(theta)
        pole_y = self.env.axle_y + self.env.pole_length * np.cos(theta)
        obst_x = self.env.obstacle.left_x + (self.env.obstacle.right_x - self.env.obstacle.left_x) / 2.0
        obst_y = self.env.obstacle.bottom_y + (self.env.obstacle.top_y - self.env.obstacle.bottom_y) / 2.0
        dist_target = abs(state[0] - self.env.x_target)
        dist_obst = np.sqrt((obst_x - pole_x)**2 + (obst_y - pole_y)**2)
        return - dist_target + dist_obst


class SparseReward(gym.RewardWrapper):
    """
    reward(s,a) := penalty, if collision or falldown
    reward(s,a) := bonus, if target is reached
    """

    def __init__(self, env):
        super().__init__(env)
        self.rew = 0.0
        self.ret = 0.0

    def reward(self, reward):
        # note: the need of this overwriting fo rew/ret is purely for rendering purposes
        # in this way, the env.render method with render the correct reward
        self.rew = self.reward_in_state(self.state)
        self.ret += self.rew
        self.env.rew = self.rew
        self.env.ret = self.ret
        return self.rew

    def reward_in_state(self, state):
        x, theta = state[0], state[2]
        reward_target = +10.0 if abs(x - self.env.x_target) <= self.env.x_target_tol else 0.0
        reward_falldown = -10 if abs(theta) > self.env.theta_threshold_radians else 0.0
        reward_collision = -10 if self.env.obstacle.intersect(x, theta) else 0.0
        # this is a workaround to visualize the reward while rendering
        self.env.safety_tot = reward_falldown + reward_collision
        self.env.target_tot = reward_target
        self.env.comfort_tot = 0.0
        return reward_collision + reward_falldown + reward_target

    def reset(self):
        obs = super(SparseReward, self).reset()
        self.rew = 0.0
        self.ret = 0.0
        return obs


class STLReward(gym.RewardWrapper):
    """
    reward(s,a) := rho(phi,episode), if terminal
    reward(s,a) := 0, otherwise

    note: not sure this formulation is markovian, because the reward assignment requires to observe the full episode
    """

    def __init__(self, env):
        super().__init__(env)
        self.rew = 0.0
        self.ret = 0.0

    def reward(self, reward):
        # note: the need of this overwriting fo rew/ret is purely for rendering purposes
        # in this way, the env.render method with render the correct reward
        self.rew = self.reward_in_state(self.state)
        self.ret += self.rew
        self.env.rew = self.rew
        self.env.ret = self.ret
        return self.rew

    def reward_in_state(self, state):
        if self.env.done:
            # if done, then last_complete_episode will contain the whole episode we need to compute robustness
            # moreover, last_cont_spec will contain the stl spec of the episode according to the obstacle position
            return self.env.compute_episode_robustness(self.env.last_complete_episode,
                                                       self.env.last_cont_spec)
        else:
            return 0.0

    def reset(self):
        obs = super(STLReward, self).reset()
        self.rew = 0.0
        self.ret = 0.0
        return obs
