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
    return lambda state, info: info['wall_collision']


def _build_complete_lap(env_params):
    """
    info['progress'] contains the normalized lap progress w.r.t. the grid position
    """
    return lambda state, info: info["progress"] >= env_params["target_progress"]


def _build_keep_center(env_params):
    """
    info['obstacle'] is the normalized distance to static obstacles in the map (e.g., walls).
    the normalization in 0..1 is done w.r.t.
        min value (ie., 0 along the walls), and
        max value (in the largest section of the track)

    the threshold defines then a corridor centered on the centerline,
    for example, requiring info['obstacle'] >= 0.5 means a corridor large 0.5*max_track_width
    """
    return lambda state, info: info["obstacle"] >= env_params["target_dist2obst"]


register_spec('s1_coll', Operator.ENSURE, _build_no_collision)
register_spec("t_lap", Operator.ACHIEVE, _build_complete_lap)
register_spec("c1_center", Operator.ENCOURAGE, _build_keep_center)
