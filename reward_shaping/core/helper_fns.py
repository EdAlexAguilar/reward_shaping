from typing import List, Dict, Any

import numpy as np
import rtamt

from reward_shaping.core.reward import RewardFunction
from reward_shaping.lti_filtering.specification import MTLDiscreteTimeSpecification


class DefaultReward(RewardFunction):
    """
    this is a dummy reward for using the default reward of an environment,
    it assumes the default reward computed by the original implementation is passed in the info
    """

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'default_reward' in info
        return info['default_reward']


def monitor_stl_episode(stl_spec: str, vars: List[str], types: List[str], episode: Dict[str, Any]):
    spec = rtamt.STLSpecification()
    for v, t in zip(vars, types):
        spec.declare_var(v, f'{t}')
    spec.spec = stl_spec
    try:
        spec.parse()
    except rtamt.STLParseException:
        return
    # preprocess format, evaluate, post process
    robustness_trace = spec.evaluate(episode)
    return robustness_trace


def monitor_mtl_filtering_episode(mtl_spec: str, vars: List[str], types: List[str], episode: Dict[str, Any]):
    spec = MTLDiscreteTimeSpecification(mtl_spec)
    for v, t in zip(vars, types):
        spec.declare_var(v, f'{t}')
    spec.spec = mtl_spec
    try:
        spec.parse()
    except rtamt.STLParseException:
        return
    # preprocess format, evaluate, post process
    robustness_trace = spec.evaluate(episode)
    return robustness_trace