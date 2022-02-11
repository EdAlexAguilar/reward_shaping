from reward_shaping.monitor.formula import Operator

_registry = {}


def get_spec(name):
    return _registry[name]


def get_all_specs():
    return _registry


def register_spec(name, operator, build_predicate):
    if name not in _registry.keys():
        _registry[name] = (operator, build_predicate)


def _build_no_collision(env_params):
    assert 'dist_hull_limit' in env_params
    return lambda state, info: -1.0 if state['collision'] > 0 else +1

def _build_achieve_goal(env_params):
    return lambda state, info: info["position_x"] - info["target_x"]


def _build_comfortable_angle(env_params):
    assert 'angle_hull_limit' in env_params
    return lambda state, info: env_params['angle_hull_limit'] - abs(state['hull_angle'])


def _build_comfortable_vx(env_params):
    assert 'speed_x_target' in env_params
    return lambda state, info: state['horizontal_speed'] - env_params['speed_x_target']


def _build_comfortable_vy(env_params):
    assert 'speed_y_limit' in env_params
    return lambda state, info: env_params['speed_y_limit'] - abs(state['vertical_speed'])


def _build_comfortable_angle_vel(env_params):
    assert 'angle_vel_limit' in env_params
    return lambda state, info: env_params['angle_vel_limit'] - abs(state['hull_angle_speed'])


register_spec("s1_coll", Operator.ENSURE, _build_no_collision)
register_spec("t_goal", Operator.ACHIEVE, _build_achieve_goal)
register_spec("c1_ang", Operator.ENCOURAGE, _build_comfortable_angle)
register_spec("c2_vx", Operator.ENCOURAGE, _build_comfortable_vx)
register_spec("c3_vy", Operator.ENCOURAGE, _build_comfortable_vy)
register_spec("c4_angvel", Operator.ENCOURAGE, _build_comfortable_angle_vel)
