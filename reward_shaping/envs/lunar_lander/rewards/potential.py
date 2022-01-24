import numpy as np

from reward_shaping.envs.lunar_lander.rewards.baselines import LLSparseTargetReward


class LLHierarchicalShapingOnSparseTargetReward(LLSparseTargetReward):
    def _hrs_potential(self, state, info):
        assert all([s in state for s in ["collision", "x", "y"]])
        assert all([i in info for i in ["x_limit", "x_target", "y_target"]])
        x, y, collision = state["x"], state["y"], state["collision"]
        #
        collision_bool = int(collision <= 0)
        outside_bool = int(abs(x) <= info["x_limit"])
        safety_reward = collision_bool + outside_bool
        safety_weight = collision_bool * outside_bool
        #
        dist_goal = np.linalg.norm([x - info["x_target"], y - info["y_target"]])
        target_reward = 1 - np.clip(dist_goal, 0, 1.5) / 1.5
        target_weight = target_reward
        #
        angle, angle_speed = state["angle"], state["angle_speed"]
        comf_angle = 1 - (1 / info["angle_limit"] * np.clip(abs(angle), 0, info["angle_limit"]))
        comf_angle_speed = 1 - (1 / info["angle_speed_limit"] * np.clip(abs(angle_speed), 0, info["angle_speed_limit"]))
        return safety_reward + target_reward + safety_weight * target_weight * (comf_angle + comf_angle_speed)

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        # note: for episodic undiscounted (gamma=1) config, terminal state must have 0 potential
        reward = super(LLHierarchicalShapingOnSparseTargetReward, self).__call__(state, action, next_state, info)
        if not info["done"]:
            potential = self._hrs_potential(next_state, info) - self._hrs_potential(state, info)
        else:
            potential = - self._hrs_potential(state, info)
        return reward + potential
