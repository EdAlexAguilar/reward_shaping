import numpy as np

from reward_shaping.core.reward import RewardFunction


class BWHierarchicalPotentialShaping(RewardFunction):
    def _safety_potential(self, state, info):
        return int(state["collision"] <= 0)

    def _target_potential(self, state, info):
        return np.clip(state["x"], 0.0, 1)  # already normalize, safety check to avoid unexpected values

    def _comfort_potential(self, state, info):
        vx, vy = state["horizontal_speed"], state["vertical_speed"]
        phi, phi_dot = state["hull_angle"], state["hull_angle_speed"]
        # keep minimal speed
        comfort_vx = (1 / (0.5 - info["speed_x_target"]) * np.clip(vx, info['speed_x_target'], 0.5))
        # keep comfortable angle
        comf_angle = 1 - (1 / info["angle_hull_limit"] * np.clip(abs(phi), 0, info["angle_hull_limit"]))
        # keep comfortable oscillations
        comf_vy = 1 - (1 / info["speed_y_limit"] * np.clip(abs(vy), 0, info["speed_y_limit"]))
        # keep comfortable angular velocity
        comf_angle_vel = 1 - (1 / info["angle_vel_limit"] * np.clip(abs(phi_dot), 0, info["angle_vel_limit"]))
        # hierarchical weights
        safety_w, target_w = self._safety_potential(state, info), self._target_potential(state, info)
        return safety_w * target_w * (comfort_vx + comf_vy + comf_angle + comf_angle_vel)

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        # base reward
        base_reward = 0.0
        if next_state["x"] >= info["norm_target_x"]:
            base_reward = 1.0
        # shaping
        if info["done"]:
            return base_reward
        shaping_safety = self._safety_potential(next_state, info) - self._safety_potential(state, info)
        shaping_target = self._target_potential(next_state, info) - self._target_potential(state, info)
        shaping_comfort = self._comfort_potential(next_state, info) - self._comfort_potential(state, info)
        return base_reward + shaping_safety + shaping_target + shaping_comfort
