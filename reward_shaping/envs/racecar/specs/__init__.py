import numpy as np

from reward_shaping.monitor.formula import Operator

_registry = {}


def get_spec(name):
    return _registry[name]


def get_all_specs():
    return _registry


def register_spec(name, operator, build_predicate):
    if name not in _registry.keys():
        _registry[name] = (operator, build_predicate)


def _build_no_collision(_):
    """
    ensure collision <= 0
    """
    return lambda state, info: -1.0 if state['collision'] > 0 else +1


def _build_complete_lap(env_params):
    """
    achieve progress >= target_progress

    note: info['progress'] contains the normalized lap progress w.r.t. the grid position
    """
    return lambda state, info: state["progress"] - env_params["reward_params"]["target_progress"]


def _build_keep_center(env_params):
    """
    encourage dist2obst >= target_dist2obst

    info['dist2obst'] is the normalized distance to static obstacles in the map (e.g., walls).
    the normalization in 0..1 is done w.r.t.
        min value (ie., 0 along the walls), and
        max value (in the largest section of the track)

    NOTE: it is measured at the agent pose, so it could be >0 even when colliding with the wall

    The threshold defines then a corridor centered on the centerline,
    For example, requiring info['dist2obst'] >= 0.5 means a corridor large 0.5*max_track_width
    """
    return lambda state, info: state["dist2obst"] - env_params["reward_params"]["target_dist2obst"]


def _build_small_steering(env_params):
    """
    encourage abs(last steering) <= max_steering

    note:   assume actions is a (k,2)-dim array containing the last k actions,
            where the 1st action is steering and 2nd action is speed.
    """
    return lambda state, info: env_params["reward_params"]["comfort_max_steering"] - abs(state["last_actions"][-1][0])


def _build_min_velocity(env_params):
    """
    encourage velocity_x >= min_velocity
    """
    return lambda state, info: state["velocity_x"] - env_params["reward_params"]["min_velx"]


def _build_max_velocity(env_params):
    """
    encourage velocity_x <= max_velocity
    """
    return lambda state, info: env_params["reward_params"]["max_velx"] - state["velocity_x"]


def _build_smooth_controls(env_params):
    """
    encourage (action[-1] - action[-2])**2 <= comfortable_value
    """
    return lambda state, info: env_params["reward_params"]["comfort_max_norm"] - np.linalg.norm(state["last_actions"][-1] - state["last_actions"][-2])


register_spec('s1_coll', Operator.ENSURE, _build_no_collision)
register_spec("t_lap", Operator.ACHIEVE, _build_complete_lap)
register_spec("c1_center", Operator.ENCOURAGE, _build_keep_center)
register_spec("c3_minvel", Operator.ENCOURAGE, _build_min_velocity)
register_spec("c4_maxvel", Operator.ENCOURAGE, _build_max_velocity)
register_spec("c2_smallsteer", Operator.ENCOURAGE, _build_small_steering)
register_spec("c5_smooth", Operator.ENCOURAGE, _build_smooth_controls)
