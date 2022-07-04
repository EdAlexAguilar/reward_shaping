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
    return lambda state, info: +1
    # TODO implement it properly
    return lambda state, info: -1 if state['collision'] == 1 else +1


def _build_complete_lap(env_params):
    return lambda state, info: +1
    # TODO implement it properly
    assert "halfwidth_landing_area" in env_params and "landing_height" in env_params
    dist_x = lambda state, info: env_params["halfwidth_landing_area"] - abs(state["x"])
    dist_y = lambda state, info: env_params["landing_height"] - abs(state["y"])
    return lambda state, info: min(dist_x(state, info), dist_y(state, info))


def _build_keep_center(env_params):
    return lambda state, info: +1
    # TODO implement it properly
    assert 'angle_limit' in env_params
    return lambda state, info: env_params['angle_limit'] - abs(state['angle'])


register_spec('s1_coll', Operator.ENSURE, _build_no_collision)
register_spec("t_lap", Operator.CONQUER, _build_complete_lap)
register_spec("c1_center", Operator.ENCOURAGE, _build_keep_center)
