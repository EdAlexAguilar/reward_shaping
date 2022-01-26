from reward_shaping.core.reward import RewardFunction
from reward_shaping.envs.cart_pole_obst.rewards import CPOSparseTargetReward, CPOProgressTargetReward
import numpy as np


class CPOHierarchicalShapingOnSparseTargetReward(CPOSparseTargetReward):

    def _hrs_potential(self, state, info):
        assert all([s in state for s in ["x", "theta", "collision"]])
        assert all([i in info for i in ["x_target", "theta_limit", "theta_target_tol", "pole_length", "axle_y"]])
        x, theta, collision = state['x'], state['theta'], state["collision"]
        #
        falldown = (abs(theta) <= info["theta_limit"])
        outside = (abs(x) <= info["x_limit"])
        collision = (collision <= 0)
        safety_reward = int(falldown) + int(outside) + int(collision)
        safety_weight = int(falldown) * int(outside) * int(collision)
        #
        pole_x, pole_y = x + info['pole_length'] * np.sin(theta), \
                         info['axle_y'] + info['pole_length'] * np.cos(theta)
        goal_x, goal_y = info['x_target'], info['axle_y'] + info['pole_length']
        dist_goal = np.linalg.norm([goal_x - pole_x, goal_y - pole_y])
        target_reward = 1 - np.clip(dist_goal, 0, 2.5) / 2.5
        target_weight = target_reward
        #
        dist_to_balance = 1 / info["theta_target_tol"] * np.clip(abs(theta), 0, info["theta_target_tol"])
        comfort_reward = 1 - dist_to_balance
        return safety_reward + safety_weight * target_reward + safety_weight * target_weight * comfort_reward

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        # note: for episodic undiscounted (gamma=1) config, terminal state must have 0 potential
        reward = super(CPOHierarchicalShapingOnSparseTargetReward, self).__call__(state, action, next_state, info)
        if not info["done"]:
            potential = self._hrs_potential(next_state, info) - self._hrs_potential(state, info)
        else:
            potential = - self._hrs_potential(state, info)
        return reward + potential


class CPOHierarchicalShapingOnSafeProgressReward(CPOProgressTargetReward):

    def _hrs_potential(self, state, info):
        assert all([s in state for s in ["x", "theta", "collision"]])
        assert all([i in info for i in ["x_target", "theta_limit", "theta_target_tol", "pole_length", "axle_y"]])
        x, theta, collision = state['x'], state['theta'], state["collision"]
        #
        falldown = (theta <= info["theta_limit"])
        outside = (x <= info["x_limit"])
        collision = (collision <= 0)
        safety_reward = int(falldown) + int(outside) + int(collision)
        safety_weight = int(falldown) * int(outside) * int(collision)
        #
        pole_x, pole_y = x + info['pole_length'] * np.sin(theta), \
                         info['axle_y'] + info['pole_length'] * np.cos(theta)
        goal_x, goal_y = info['x_target'], info['axle_y'] + info['pole_length']
        dist_goal = np.linalg.norm([goal_x - pole_x, goal_y - pole_y])
        target_reward = 1 - np.clip(dist_goal, 0, 2.5) / 2.5
        target_weight = target_reward
        #
        dist_to_balance = 1 / info["theta_target_tol"] * np.clip(abs(theta), 0, info["theta_target_tol"])
        comfort_reward = 1 - dist_to_balance
        return safety_reward + safety_weight * target_reward + safety_weight * target_weight * comfort_reward

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        # note: for episodic undiscounted (gamma=1) config, terminal state must have 0 potential
        reward = super(CPOHierarchicalShapingOnSafeProgressReward, self).__call__(state, action, next_state, info)
        if not info["done"]:
            potential = self._hrs_potential(next_state, info) - self._hrs_potential(state, info)
        else:
            potential = - self._hrs_potential(state, info)
        return reward + potential


class CPOHierarchicalPotentialShaping(RewardFunction):
    def _clip_and_norm(self, v, min, max):
        return (np.clip(v, min, max) - min) / (max-min)

    def _check_goal_condition(self, state, info):
        return abs(state['x'] - info['x_target']) <= info['x_target_tol'] and \
               abs(state['theta']) <= info["theta_target_tol"]

    def _safety_potential(self, state, info):
        """
        idea: no encourage safety robustness, but use boolean signal.
        then, the difference of potential is 0 as long as keeping safety, negative (-1) when violating it.
        """
        x, theta, collision = state['x'], state['theta'], state["collision"]
        falldown = (theta <= info["theta_limit"])
        outside = (x <= info["x_limit"])
        collision = (collision <= 0)
        return int(falldown) + int(outside) + int(collision)

    def _target_potential(self, state, info):
        """
        idea: since the task is to conquer the origin, the potential of a state depends on two factors:
            - the distance to the target (if not reached yet), and the persistence on the target (once reached)
        """
        # evaluate dist to goal
        x, theta, collision = state['x'], state['theta'], state["collision"]
        pole_x, pole_y = x + info['pole_length'] * np.sin(theta), \
                         info['axle_y'] + info['pole_length'] * np.cos(theta)
        goal_x, goal_y = info['x_target'], info['axle_y'] + info['pole_length']
        dist_goal = np.linalg.norm([goal_x - pole_x, goal_y - pole_y])
        target_reward = 1 - np.clip(dist_goal, 0, 2.5) / 2.5
        # hierarchical weights
        safety_w = self._safety_potential(state, info)
        return safety_w * target_reward

    def _comfort_potential(self, state, info):
        x, theta, collision = state['x'], state['theta'], state["collision"]
        comfort_reward = 1 - self._clip_and_norm(abs(theta), info["theta_target_tol"], info["theta_limit"])
        # hierarchical weights
        safety_w, target_w = self._safety_potential(state, info), self._target_potential(state, info)
        return safety_w * target_w * comfort_reward

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert all([s in state for s in ["x", "theta", "collision"]])
        assert all([i in info for i in ["x_target", "theta_limit", "theta_target_tol", "pole_length", "axle_y"]])
        # base reward
        base_reward = 1.0 if self._check_goal_condition(next_state, info) else 0.0
        if info["done"]:
            return base_reward
        # hierarchical shaping function
        shaping_safety = self._safety_potential(next_state, info) - self._safety_potential(state, info)
        shaping_target = self._target_potential(next_state, info) - self._target_potential(state, info)
        shaping_comfort = self._comfort_potential(next_state, info) - self._comfort_potential(state, info)
        return base_reward + shaping_safety + shaping_target + shaping_comfort
