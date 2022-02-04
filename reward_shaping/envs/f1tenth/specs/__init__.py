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
    return lambda state, info: -1 if state['collision'] == 1 else +1


def _build_no_reverse(_):
    return lambda state, info: -1 if state['reverse'] == 1 else +1


def _build_complete_lap(_):
    """ Known bug: when crossing the starting line (ie, completing one lap), the progress step from 0.99 to 1.99 and then restart from 1.01.
    This depends on the centerline waypoints"""
    return lambda state, info: state["progress"] - 1.0


def _build_speed_limit(_):
    upper = lambda state, info: info["comfortable_speed_max"] - state["velocity"]
    lower = lambda state, info: state["velocity"] - info["comfortable_speed_min"]
    return lambda state, info: min(upper(state, info), lower(state, info))


def _build_comfortable_steering(_):
    return lambda state, info: info["comfortable_steering"] - abs(state["steering"])


def _build_keep_right(_):
    lanes = {"right": 0, "left": 1}
    return lambda state, info: 1.0 if state["lane"] == lanes["right"] else -1


register_spec('s_coll', Operator.ENSURE, _build_no_collision)
register_spec('s_reverse', Operator.ENSURE, _build_no_reverse)
register_spec("t_lap", Operator.ACHIEVE, _build_complete_lap)
register_spec("c1_speed", Operator.ENCOURAGE, _build_speed_limit)
register_spec("c2_steering", Operator.ENCOURAGE, _build_comfortable_steering)
register_spec("c3_keep_right_lane", Operator.ENCOURAGE, _build_keep_right)
