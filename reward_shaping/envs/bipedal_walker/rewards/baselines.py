from reward_shaping.core.reward import WeightedReward
import numpy as np

from reward_shaping.core.utils import get_normalized_reward
import reward_shaping.envs.bipedal_walker.rewards.subtask_rewards as fns


class WeightedBaselineReward(WeightedReward):
    """
    reward(s,a) := w_s * sum([score in safeties]) + w_t * sum([score in targets]) + w_c * sum([score in comforts])
    """

    def __init__(self, env_params, safety_weight=1.0, target_weight=0.5, comfort_weight=0.25):
        # parameters
        super().__init__()
        self._safety_weight = safety_weight
        self._target_weight = target_weight
        self._comfort_weight = comfort_weight
        # prepare env info for normalize the functions
        info = {
            "angle_hull_limit": env_params['angle_hull_limit'],
            "speed_y_limit": env_params['speed_y_limit'],
            "angle_vel_limit": env_params['angle_vel_limit'],
            "speed_x_target": env_params['speed_x_target']
        }
        # safety rules
        falldown_fn = fns.BinaryFalldownReward(falldown_penalty=0.0, no_falldown_bonus=1.0)
        # target rules (no need indicators)
        target_fn, _ = get_normalized_reward(fns.SpeedTargetReward(),
                                             min_r_state=[0] * 2 + [info['speed_x_target']] + [0] * 21,
                                             max_r_state=[0] * 2 + [1] + [0] * 21,
                                             info=info)
        # comfort rules
        angle_comfort_fn, _ = get_normalized_reward(fns.ContinuousHullAngleReward(),
                                                    min_r_state=[info['angle_hull_limit']] + [0.0] * 23,
                                                    max_r_state=[0.0] + [0.0] * 23,
                                                    info=info)
        vert_speed_comfort_fn, _ = get_normalized_reward(fns.ContinuousVerticalSpeedReward(),
                                                         min_r_state=[0.0] * 3 + [info['speed_y_limit']] + [0.0] * 20,
                                                         max_r_state=[0.0] * 24,
                                                         info=info)
        angle_vel_comfort_fn, _ = get_normalized_reward(fns.ContinuousHullAngleVelocityReward(),
                                                        min_r_state=[0.0] + [info['angle_vel_limit']] + [0.0] * 22,
                                                        max_r_state=[0.0] * 24,
                                                        info=info)

        self._safety_rules = [falldown_fn]
        self._target_rules = [target_fn]
        self._comfort_rules = [angle_comfort_fn, vert_speed_comfort_fn, angle_vel_comfort_fn]
