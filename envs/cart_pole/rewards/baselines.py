import gym
import numpy as np

from envs.cart_pole.rewards.subtask_rewards import CollisionReward, ReachTargetReward, SparseReachTargetReward, \
    BalanceReward, NormalizedReward


class WeightedReward(gym.RewardWrapper):
    """
    reward(s,a) := penalty, if collision
    reward(s,a) := 0.5 * (target_reward, balance_reward), otherwise
    """

    def __init__(self, env):
        super().__init__(env)
        self.collision = CollisionReward(env=env, collision_penalty=-10.0, no_collision_bonus=0.0)
        # normalized target reward
        fun = ReachTargetReward(x_target=env.x_target, x_target_tol=env.x_target_tol)
        min_r_state = np.array([env.x_threshold, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        max_r_state = np.array([env.x_target, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        min_r, max_r = min(fun(-min_r_state), fun(min_r_state)), fun(max_r_state)
        self.reach_origin = NormalizedReward(fun, min_reward=min_r, max_reward=max_r)
        # normalized balance reward
        fun = BalanceReward(theta_target=env.theta_target, theta_target_tol=env.theta_target_tol)
        min_r_state = np.array([0.0, 0.0, env.theta_threshold_radians, 0.0, 0.0, 0.0, 0.0, 0.0])
        max_r_state = np.array([0.0, 0.0, env.theta_target, 0.0, 0.0, 0.0, 0.0, 0.0])
        min_r, max_r = min(fun(-min_r_state), fun(min_r_state)), fun(max_r_state)
        self.keep_balance = NormalizedReward(fun, min_reward=min_r, max_reward=max_r)

    def reward(self, reward):
        # note: the need of this overwriting fo rew/ret is purely for rendering purposes
        # in this way, the env.render method with render the correct reward
        self.rew = self.reward_in_state(self.state)
        self.ret += self.rew
        self.env.rew = self.rew
        self.env.ret = self.ret
        return self.rew

    def reward_in_state(self, state):
        reward_collision = self.collision(state)
        reward_target = self.reach_origin(state)
        reward_balance = self.keep_balance(state)
        if reward_collision < 0.0:
            # this is a workaround to visualize the reward while rendering
            self.env.safety_tot = reward_collision
            self.env.target_tot = 0.0
            self.env.comfort_tot = 0.0
            return reward_collision
        else:
            # this is a workaround to visualize the reward while rendering
            self.env.safety_tot = reward_collision
            self.env.target_tot = 0.5 * reward_target
            self.env.comfort_tot = 0.5 * reward_balance
            return 0.5 * (reward_target + reward_balance)


class SparseReward(gym.RewardWrapper):
    """
    reward(s,a) := penalty, if collision
    reward(s,a) := bonus, if target is reached
    """
    def __init__(self, env):
        super().__init__(env)
        self.collision = CollisionReward(env=env, collision_penalty=-10.0, no_collision_bonus=0.0)
        self.reach_origin = SparseReachTargetReward(x_target=env.x_target, x_target_tol=env.x_target_tol,
                                                    target_reward=10.0)
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
        reward_target = self.reach_origin(state)
        reward_collision = self.collision(state)
        # this is a workaround to visualize the reward while rendering
        self.env.safety_tot = reward_collision
        self.env.target_tot = reward_target
        self.env.comfort_tot = 0.0
        return reward_target + reward_collision

    def reset(self):
        obs = super(SparseReward, self).reset()
        self.rew = 0.0
        self.ret = 0.0
        return obs
