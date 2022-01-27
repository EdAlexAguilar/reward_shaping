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


def _build_no_falldown(env_params):
    assert "theta_limit" in env_params
    return lambda state, info: np.deg2rad(env_params["theta_limit"]) - abs(state["theta"])


def _build_no_outside(env_params):
    assert "x_limit" in env_params
    return lambda state, info: env_params["x_limit"] - abs(state["x"])


def _build_no_collision(_):
    return lambda state, info: -1 if state['collision'] == 1 else +1


def _build_reach_target(env_params):
    # normalization of robustnesss does not play a big role here because we care only about the sign (sat/unsat)
    # to eventually increment/reset the target counter
    dist_x = lambda state, info: info['x_target_tol'] - abs(state["x"] - env_params['x_target'])
    dist_theta = lambda state, info: info['theta_target_tol'] - abs(state['theta'])
    return lambda state, info: min(dist_x(state, info), dist_theta(state, info))


def _build_balance(env_params):
    assert "theta_target_tol" in env_params and "theta_target" in env_params
    return lambda state, info: env_params["theta_target_tol"] - abs(state["theta"] - env_params["theta_target"])


register_spec("s1_fall", Operator.ENSURE, _build_no_falldown)
register_spec("s2_exit", Operator.ENSURE, _build_no_outside)
register_spec("s3_coll", Operator.ENSURE, _build_no_collision)
register_spec("t_origin", Operator.CONQUER, _build_reach_target)
register_spec("c_balance", Operator.ENCOURAGE, _build_balance)