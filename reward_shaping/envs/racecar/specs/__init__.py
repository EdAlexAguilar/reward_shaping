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
    return lambda state, info: -1 if info['wall_collision'] else +1


def _build_no_reverse(_):
    return lambda state, info: -1 if info['wrong_way'] else +1


def _build_complete_lap(_):
    """ Known bug: when crossing the starting line (ie, completing one lap), the progress step from 0.99 to 1.99 and then restart from 1.01.
    This depends on the centerline waypoints"""
    return lambda state, info: 1.0 if info["lap"] > 0 else -1.0


def _build_speed_limit(_):
    # note: we are evaluating using privileged information of ground-truth velocity
    return lambda state, info: info["norm_speed_limit"] - state["velocity"][0]


def _build_comfortable_steering(_):
    return lambda state, info: info["norm_comf_steering"] - abs(state["steering"])


def _build_keep_right(_):
    """ req: abs(dist_to_wall - target_dist) <= tolerance_margin"""
    return lambda state, info: info["tolerance_margin"] - abs(state["dist_to_wall"] - info["comf_dist_to_wall"])


register_spec('s_coll', Operator.ENSURE, _build_no_collision)
register_spec('s_reverse', Operator.ENSURE, _build_no_reverse)
register_spec("t_lap", Operator.ACHIEVE, _build_complete_lap)
register_spec("c1_speed", Operator.ENCOURAGE, _build_speed_limit)
register_spec("c2_steering", Operator.ENCOURAGE, _build_comfortable_steering)
register_spec("c3_keep_right", Operator.ENCOURAGE, _build_keep_right)
