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


def _build_complete_lap(env_params):
    return lambda state, info: info["progress"]


register_spec('s1_coll', Operator.ENSURE, _build_no_collision)
register_spec("t_lap", Operator.CONQUER, _build_complete_lap)