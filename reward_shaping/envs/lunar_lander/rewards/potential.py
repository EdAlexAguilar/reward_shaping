import numpy as np

from reward_shaping.core.reward import RewardFunction


class LLHierarchicalShapingOnSparseTargetReward(RewardFunction):
    def _check_goal_condition(self, state, info):
        dist_x = info["halfwidth_landing_area"] - abs(state["x"])
        dist_y = 0.0001 - abs(state["y"])
        return min(dist_x(state, info), dist_y(state, info)) >= 0

    def _clip_and_norm(self, v, min, max):
        return (np.clip(v, min, max) - min) / (max-min)

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
        comf_angle = 1 - self._clip_and_norm(abs(angle), info["angle_limit"], 1.0)
        comf_angle_speed = 1 - self._clip_and_norm(abs(angle_speed), info["angle_speed_limit"], 1.0)
        return safety_reward + target_reward + safety_weight * target_weight * (comf_angle + comf_angle_speed)

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        # note: for episodic undiscounted (gamma=1) config, terminal state must have 0 potential
        base_reward = 1.0 if self._check_goal_condition(next_state, info) else 0.0
        if info["done"]:
            return base_reward
        # shaping
        shaping = self._hrs_potential(next_state, info) - self._hrs_potential(state, info)
        return base_reward + shaping
