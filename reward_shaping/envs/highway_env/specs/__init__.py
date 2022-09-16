from reward_shaping.monitor.formula import Operator
import numpy as np

_registry = {}


def get_spec(name):
    return _registry[name]


def get_all_specs():
    return _registry


def register_spec(name, operator, build_predicate):
    if name not in _registry.keys():
        _registry[name] = (operator, build_predicate)


def _build_safety_RSS(_):
    return lambda state, info: -1 if state['violated_safe_distance'] == 1 else +1


def _build_hard_speed_limit(_):
    return lambda state, info: -1 if state['violated_hard_speed_limit'] == 1 else +1


def _build_slower_than_left(_):
    return lambda state, info: 1 - ((np.clip(state['max_velocity_difference_to_left'],
                                             0, info['HARD_SPEED_LIMIT'])) / info['HARD_SPEED_LIMIT'])


def _build_soft_speed_limit(_):
    return lambda state, info: (np.clip(info['SOFT_SPEED_LIMIT'] - state['observation'][0][3],
                                        0, info['HARD_SPEED_LIMIT'])) / info['HARD_SPEED_LIMIT']


def _build_speed_lower_bound(_):
    return lambda state, info: (np.clip(state['observation'][0][3] - info['SPEED_LOWER_BOUND'],
                                        0, info['HARD_SPEED_LIMIT'])) / info['HARD_SPEED_LIMIT']


def _build_reach_target(_):
    return lambda state, info: 1 if (state['distance_to_target'] <= info['TARGET_DISTANCE_TOL']) else -1


register_spec("s1_safedist", Operator.ENSURE, _build_safety_RSS)
register_spec("s2_hardlim", Operator.ENSURE, _build_hard_speed_limit)
register_spec("t_origin", Operator.ACHIEVE, _build_reach_target)
register_spec("c1_slwleft", Operator.ENCOURAGE, _build_slower_than_left)
register_spec("c2_softlim", Operator.ENCOURAGE, _build_soft_speed_limit)
register_spec("c3_lowspeed", Operator.ENCOURAGE, _build_speed_lower_bound)
