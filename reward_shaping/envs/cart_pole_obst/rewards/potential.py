from reward_shaping.envs.cart_pole_obst.rewards import CPOSparseTargetReward, CPOProgressTargetReward
import numpy as np


class CPOHierarchicalShapingOnSparseTargetReward(CPOSparseTargetReward):

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
        return safety_reward + target_reward + safety_weight * target_weight * comfort_reward

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        # note: for episodic undiscounted (gamma=1) tasks, terminal state must have 0 potential
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
        # note: for episodic undiscounted (gamma=1) tasks, terminal state must have 0 potential
        reward = super(CPOHierarchicalShapingOnSafeProgressReward, self).__call__(state, action, next_state, info)
        if not info["done"]:
            potential = self._hrs_potential(next_state, info) - self._hrs_potential(state, info)
        else:
            potential = - self._hrs_potential(state, info)
        return reward + potential
