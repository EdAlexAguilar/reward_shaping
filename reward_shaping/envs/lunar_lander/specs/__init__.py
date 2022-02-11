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


def _build_no_outside(_):
    return lambda state, info: 1.0 - abs(state['x'])  # x coord is already normalized in -1, +1


def _build_reach_target(env_params):
    assert "halfwidth_landing_area" in env_params and "landing_height" in env_params
    dist_x = lambda state, info: env_params["halfwidth_landing_area"] - abs(state["x"])
    dist_y = lambda state, info: env_params["landing_height"] - abs(state["y"])
    return lambda state, info: min(dist_x(state, info), dist_y(state, info))


def _build_comfortable_angle(env_params):
    assert 'angle_limit' in env_params
    return lambda state, info: env_params['angle_limit'] - abs(state['angle'])


def _build_comfortable_angle_vel(env_params):
    assert 'angle_speed_limit' in env_params
    return lambda state, info: env_params['angle_speed_limit'] - abs(state['angle_speed'])


register_spec('s1_coll', Operator.ENSURE, _build_no_collision)
register_spec('s2_exit', Operator.ENSURE, _build_no_outside)
register_spec("t_origin", Operator.CONQUER, _build_reach_target)
register_spec("c1_ang", Operator.ENCOURAGE, _build_comfortable_angle)
register_spec("c2_angvel", Operator.ENCOURAGE, _build_comfortable_angle_vel)
