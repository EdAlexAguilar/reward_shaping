import numpy as np

from reward_shaping.envs.bipedal_walker.rewards import BWSparseTargetReward
from reward_shaping.envs.bipedal_walker.rewards.baselines import BWSparseLivenessTargetReward


class BWHierarchicalShapingOnSparseTargetReward(BWSparseTargetReward):
    def _hrs_potential(self, state, info):
        assert all(
            [s in state for s in ["horizontal_speed", "hull_angle", "hull_angle_speed", "vertical_speed", "collision"]])
        assert all([i in info for i in ["speed_x_target", "angle_hull_limit", "speed_y_limit", "angle_vel_limit"]])
        #
        safety_reward = safety_weight = int(state["collision"] <= 0)
        #
        target_reward = np.clip(state['horizontal_speed'] - info['speed_x_target'], 0, 0.5) / 0.5   # norm in 0,1
        target_weight = target_reward
        #
        phi, vy, phi_dot = state["hull_angle"], state["vertical_speed"], state["hull_angle_speed"]
        comf_angle = 1 - (1 / info["angle_hull_limit"] * np.clip(abs(phi), 0, info["angle_hull_limit"]))
        comf_vy = 1 - (1 / info["speed_y_limit"] * np.clip(abs(vy), 0, info["speed_y_limit"]))
        comf_angle_vel = 1 - (1 / info["angle_vel_limit"] * np.clip(abs(phi_dot), 0, info["angle_vel_limit"]))
        return safety_reward + target_reward + safety_weight * target_weight * (comf_angle + comf_vy + comf_angle_vel)

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        # note: for episodic undiscounted (gamma=1) tasks, terminal state must have 0 potential
        reward = super(BWHierarchicalShapingOnSparseTargetReward, self).__call__(state, action, next_state, info)
        if not info["done"]:
            potential = self._hrs_potential(next_state, info) - self._hrs_potential(state, info)
        else:
            potential = - self._hrs_potential(state, info)
        return reward + potential


class BWHierarchicalShapingOnSparseTargetRewardLiveness(BWSparseLivenessTargetReward):
    def _hrs_potential(self, state, info):
        assert all(
            [s in state for s in ["horizontal_speed", "hull_angle", "hull_angle_speed", "vertical_speed", "collision"]])
        assert all([i in info for i in ["speed_x_target", "angle_hull_limit", "speed_y_limit", "angle_vel_limit"]])
        #
        safety_reward = safety_weight = int(state["collision"] <= 0)
        #
        target_reward = 1 - np.clip((info['target_x'] - info['position_x']) / info["target_x"], 0.0, 1.0)   # norm in 0,1
        target_weight = target_reward
        #
        phi, vy, phi_dot = state["hull_angle"], state["vertical_speed"], state["hull_angle_speed"]
        comf_angle = 1 - (1 / info["angle_hull_limit"] * np.clip(abs(phi), 0, info["angle_hull_limit"]))
        comf_vy = 1 - (1 / info["speed_y_limit"] * np.clip(abs(vy), 0, info["speed_y_limit"]))
        comf_angle_vel = 1 - (1 / info["angle_vel_limit"] * np.clip(abs(phi_dot), 0, info["angle_vel_limit"]))
        return safety_reward + target_reward + safety_weight * target_weight * (comf_angle + comf_vy + comf_angle_vel)

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        # note: for episodic undiscounted (gamma=1) tasks, terminal state must have 0 potential
        reward = super(BWHierarchicalShapingOnSparseTargetRewardLiveness, self).__call__(state, action, next_state, info)
        if not info["done"]:
            potential = self._hrs_potential(next_state, info) - self._hrs_potential(state, info)
        else:
            potential = - self._hrs_potential(state, info)
        return reward + potential